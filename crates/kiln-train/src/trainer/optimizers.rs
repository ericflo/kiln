use super::*;

/// Gradient output from one tape-authoritative training step.
///
/// The active tape scope is mandatory. Gradients are keyed by each configured
/// LoRA [`Parameter::tensor_id`] and remain kt-native through observation,
/// accumulation, exact-contract validation, and optimizer dispatch.
pub enum GradSource {
    /// kt-native gradients (the SOLE grad producer post-#1082). Keyed by
    /// `Parameter::tensor_id()`; values are `kiln_tensor::Tensor`. The
    /// candle `Candle(GradStore)` variant is GONE — every candle
    /// `loss.backward()` producer was deleted in the candle drop.
    Kt(kiln_autograd::GradStore),
}

impl GradSource {
    /// Number of parameters that received a gradient.
    pub fn num_grad_ids(&self) -> usize {
        match self {
            GradSource::Kt(kt) => kt.len(),
        }
    }

    /// Borrow the underlying kt `GradStore`.
    pub fn kt(&self) -> &kiln_autograd::GradStore {
        match self {
            GradSource::Kt(kt) => kt,
        }
    }

    /// Owned kt grad for `param`, or `None` if the store has no grad for
    /// it. Used by diagnostic / convergence-gate sites. (#1082)
    pub fn grad_for(&self, param: &Parameter) -> Option<KtTensor> {
        match self {
            GradSource::Kt(kt) => kt.get(param.tensor_id()).cloned(),
        }
    }
}

/// (#1082) Optimizer-step dispatcher over [`GradSource`] — kt-native only.
pub fn optimizer_step_dispatch(
    backend: &dyn BackendRuntime,
    params: &mut TrainableLoraParams,
    grads: &GradSource,
    lr: f64,
    optimizer: Optimizer,
    opt_state: Option<&mut OptimizerState>,
) -> Result<()> {
    match grads {
        GradSource::Kt(kt) => {
            optimizer_step_from_kt_grad_store(backend, params, kt, lr, optimizer, opt_state)
        }
    }
}

pub(super) fn ensure_training_optimizer_fallback_allowed(
    backend: &dyn BackendRuntime,
    device: kiln_tensor::Device,
    optimizer_name: &'static str,
) -> Result<()> {
    let policy = BackendCapabilityQueries::backend_capabilities(backend)
        .fallback
        .training_optimizer;
    if policy.allows_fallback() {
        return Ok(());
    }
    anyhow::bail!(
        "{optimizer_name} optimizer fallback policy {:?} for {} training hot path on {}; \
         native optimizer dispatch is required and no runtime fallback override is supported",
        policy,
        BackendIdentity::runtime_name(backend),
        device.short_name()
    )
}

/// (#1082) LoRA grad-norm observer dispatcher — kt-native only.
pub fn observe_lora_grad_norms_dispatch(
    accumulator: &mut crate::train_receipt::LoraGradNormAccumulator,
    params: &TrainableLoraParams,
    grads: &GradSource,
) -> Result<()> {
    match grads {
        GradSource::Kt(kt) => observe_lora_grad_norms_from_kt_grad_store(accumulator, params, kt),
    }
}

/// Dispatch the configured optimizer against an exact kt gradient set.
/// Stateful optimizers increment their step only after the gradient contract
/// has accepted every configured leaf.
/// (#1082) Apply one kt-native SGD update (param = param - lr*grad) to a
/// single LoRA `Parameter`, preferring the on-device registry path when
/// param + grad are both resident (the backend trait takes kt tensors).
///
/// On-device path: register the kt grad → `OptimizerBackend` writes the param
/// buffer in place → evict the grad. The `Parameter`'s master is
/// left stale; `sync_to_master` pulls the registry back before save.
///
/// CPU fallback: compute `param - lr*grad` kt-natively and install it via
/// `replace_backward_storage` + `replace_forward_storage` (preserving
/// `tensor_id`).
pub(super) fn apply_sgd_update_kt(
    backend: &dyn BackendRuntime,
    param: &mut Parameter,
    grad: &KtTensor,
    lr: f64,
    resident_activation: bool,
) -> Result<()> {
    let primary = param.forward_storage().primary_tensor().clone();
    if resident_activation && ResidencyBackend::runtime_has_resident_activation(backend, &primary) {
        ResidencyBackend::runtime_register_resident_activation(backend, grad)?;
        let dispatched =
            match OptimizerBackend::runtime_dispatch_sgd_step(backend, &primary, grad, lr as f32) {
                Ok(b) => b,
                Err(e) => {
                    ResidencyBackend::runtime_evict_resident_activation(backend, grad);
                    return Err(e);
                }
            };
        if dispatched {
            ResidencyBackend::runtime_evict_resident_activation(backend, grad);
            return Ok(());
        }
        ResidencyBackend::runtime_evict_resident_activation(backend, grad);
    }
    ensure_training_optimizer_fallback_allowed(backend, primary.device(), "SGD")?;
    // CPU/host fallback: master = master - lr*grad, kt-native (F32
    // accumulate then back to param dtype, mirroring the old candle math).
    let dtype = primary.dtype();
    let master_f32 = primary
        .to_dtype(KtDType::F32)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: master to f32: {e}"))?;
    let grad_f32 = grad
        .to_dtype(KtDType::F32)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: grad to f32: {e}"))?;
    let scaled = kiln_tensor::ops::mul_scalar(&grad_f32, lr as f32)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: grad*lr: {e}"))?;
    let updated_f32 = kiln_tensor::ops::sub(&master_f32, &scaled)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: master-update: {e}"))?;
    let updated = updated_f32
        .to_dtype(dtype)
        .map_err(|e| anyhow::anyhow!("apply_sgd_update_kt: back to {dtype:?}: {e}"))?;
    param
        .replace_plain_trainable_tensor(updated)
        .map_err(|error| anyhow::anyhow!("apply_sgd_update_kt: preserve identity: {error}"))?;
    if resident_activation {
        ResidencyBackend::runtime_update_resident_activation(
            backend,
            param.forward_storage().primary_tensor(),
        )?;
    }
    Ok(())
}

/// (#1082) Apply one AdamW step to a single LoRA `Parameter`.
///
/// On-device path (resident): when the param **and** its `m`/`v` device
/// moment tensors are all resident, dispatch the CUDA AdamW kernel which
/// updates **param, m, and v in place** in one launch. This is the
/// production path (BF16 CUDA, LoRA params resident). The `m`/`v` passed
/// are the REAL per-param device moments from `OptimizerState.moments`
/// (NOT the param aliased onto itself — that was the C1 corruption bug).
/// The forward storage shares the master tensor (LoRA A/B are plain dense
/// BF16, forward primary == master), so the in-place param update is
/// immediately visible to the next forward; no refresh needed.
///
/// Host fallback (non-resident): drive the CPU reference
/// `kiln_optim::AdamW` (`OptimStep::step`), which owns its own host-side
/// moments keyed by `Parameter::tensor_id()` and installs the new master
/// via `replace_backward_storage` (preserving `tensor_id`). The forward
/// storage is refreshed from the new master.
///
/// `lr`/`beta1`/`beta2`/`eps`/`weight_decay` are threaded directly from
/// the optimizer config (no more `ADAMW_ACTIVE_HP` thread-local shim —
/// that hack existed only because the moments were host-side and the
/// device path had no real hp source). `step` is the global 1-indexed
/// step counter (shared by all params for standard AdamW bias correction).
///
/// `grad` must match the param's AMP `backward_compute_dtype` (BF16 in
/// production). The exact gradient contract checks this before any optimizer
/// state or parameter is mutated.
#[allow(clippy::too_many_arguments)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OptimizerStateAuthority {
    Device,
    Host,
}

pub(super) fn apply_adamw_update_kt(
    backend: &dyn BackendRuntime,
    param: &mut Parameter,
    adamw: &mut KtAdamW,
    moments: Option<&KtAdamWMoments>,
    grad: &KtTensor,
    lr: f64,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
    resident_activation: bool,
) -> Result<OptimizerStateAuthority> {
    let primary = param.forward_storage().primary_tensor().clone();
    // On-device registry path: param + grad + the REAL per-param m/v must
    // all be resident, then the CUDA kernel updates param/m/v in place.
    if let Some(moments) = moments
        && resident_activation
        && ResidencyBackend::runtime_has_resident_activation(backend, &primary)
        && ResidencyBackend::runtime_has_resident_activation(backend, &moments.m)
        && ResidencyBackend::runtime_has_resident_activation(backend, &moments.v)
    {
        ResidencyBackend::runtime_register_resident_activation(backend, grad)?;
        let dispatched = match OptimizerBackend::runtime_dispatch_adamw_step(
            backend,
            &primary,
            grad,
            &moments.m,
            &moments.v,
            lr as f32,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        ) {
            Ok(b) => b,
            Err(e) => {
                ResidencyBackend::runtime_evict_resident_activation(backend, grad);
                return Err(e);
            }
        };
        if dispatched {
            // The kernel updated param/m/v in place. Forward primary IS
            // the master for LoRA params, so the update is already live;
            // re-assert residency of the param buffer for the next fwd.
            ResidencyBackend::runtime_evict_resident_activation(backend, grad);
            ResidencyBackend::runtime_update_resident_activation(backend, &primary)?;
            return Ok(OptimizerStateAuthority::Device);
        }
        ResidencyBackend::runtime_evict_resident_activation(backend, grad);
    }

    ensure_training_optimizer_fallback_allowed(backend, primary.device(), "AdamW")?;
    // Host fallback: drive the CPU reference `kiln_optim::AdamW`. The exact
    // gradient contract already established the policy dtype, shape, and
    // device, so this boundary must not coerce an invalid gradient.
    adamw
        .step(param, grad)
        .map_err(|e| anyhow::anyhow!("apply_adamw_update_kt: kiln_optim AdamW step: {e}"))?;
    // `AdamW::step` swaps the master via `replace_backward_storage`
    // (preserving tensor_id). Refresh the forward storage from the new
    // master so the next forward reads the updated weights.
    if let Some(new_master) = param.backward_storage().cloned() {
        param
            .replace_plain_trainable_tensor(new_master)
            .map_err(|error| anyhow::anyhow!("AdamW preserve parameter identity: {error}"))?;
    }
    if resident_activation {
        ResidencyBackend::runtime_update_resident_activation(
            backend,
            param.forward_storage().primary_tensor(),
        )?;
    }
    Ok(OptimizerStateAuthority::Host)
}

/// (#1082) Apply one Muon step to a single LoRA `Parameter`.
///
/// On-device path (resident): when the param **and** its per-param
/// momentum device tensor are both resident, dispatch the fused Muon
/// kernel (`runtime_dispatch_muon_step`) which updates **param and
/// momentum in place** in one launch — heavy-ball momentum, then (for
/// rank-2 matrices) Newton-Schulz orthogonalization of the (Nesterov)
/// look-ahead with the RMS-matching scale, then the decoupled-weight-
/// decay descent step. The forward storage shares the master tensor
/// (LoRA A/B are plain dense BF16, forward primary == master), so the
/// in-place update is immediately visible to the next forward.
///
/// Host fallback (non-resident): drive the CPU reference
/// `kiln_optim::Muon` (`OptimStep::step`), which owns its own host-side
/// momentum keyed by `Parameter::tensor_id()` and installs the new
/// master via `replace_backward_storage` (preserving `tensor_id`). The
/// forward storage is refreshed from the new master.
///
/// `lr` and the Muon hyperparameters are threaded directly from the
/// optimizer config each step (honouring the LR schedule).
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_muon_update_kt(
    backend: &dyn BackendRuntime,
    param: &mut Parameter,
    muon: &mut KtMuon,
    momentum_state: Option<&KtMuonMomentum>,
    grad: &KtTensor,
    lr: f64,
    momentum: f32,
    nesterov: bool,
    ns_iters: u32,
    weight_decay: f32,
    resident_activation: bool,
) -> Result<OptimizerStateAuthority> {
    let primary = param.forward_storage().primary_tensor().clone();
    // On-device registry path: param + grad + the per-param momentum
    // must all be resident, then the kernel updates param/momentum in
    // place.
    if let Some(momentum_state) = momentum_state
        && resident_activation
        && ResidencyBackend::runtime_has_resident_activation(backend, &primary)
        && ResidencyBackend::runtime_has_resident_activation(backend, &momentum_state.m)
    {
        ResidencyBackend::runtime_register_resident_activation(backend, grad)?;
        let dispatched = match OptimizerBackend::runtime_dispatch_muon_step(
            backend,
            &primary,
            grad,
            &momentum_state.m,
            lr as f32,
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
        ) {
            Ok(b) => b,
            Err(e) => {
                ResidencyBackend::runtime_evict_resident_activation(backend, grad);
                return Err(e);
            }
        };
        if dispatched {
            // The kernel updated param/momentum in place. Forward
            // primary IS the master for LoRA params, so the update
            // is already live; re-assert residency for the next fwd.
            ResidencyBackend::runtime_evict_resident_activation(backend, grad);
            ResidencyBackend::runtime_update_resident_activation(backend, &primary)?;
            return Ok(OptimizerStateAuthority::Device);
        }
        ResidencyBackend::runtime_evict_resident_activation(backend, grad);
    }

    ensure_training_optimizer_fallback_allowed(backend, primary.device(), "Muon")?;
    // Host fallback: drive the CPU reference `kiln_optim::Muon`. Thread
    // the scheduled lr + config hyperparameters in so the host path
    // honours the LR schedule and keeps the optimizer config the single
    // source of truth. The host Muon owns its own host-side momentum +
    // step counter keyed by `tensor_id`.
    muon.lr = lr as f32;
    muon.momentum = momentum;
    muon.nesterov = nesterov;
    muon.ns_iters = ns_iters;
    muon.weight_decay = weight_decay;
    // The exact gradient contract has already checked dtype/shape/device;
    // silently casting here would turn a producer defect into a plausible step.
    muon.step(param, grad)
        .map_err(|e| anyhow::anyhow!("apply_muon_update_kt: kiln_optim Muon step: {e}"))?;
    // `Muon::step` swaps the master via `replace_backward_storage`
    // (preserving tensor_id). Refresh the forward storage from the new
    // master so the next forward reads the updated weights.
    if let Some(new_master) = param.backward_storage().cloned() {
        param
            .replace_plain_trainable_tensor(new_master)
            .map_err(|error| anyhow::anyhow!("Muon preserve parameter identity: {error}"))?;
    }
    if resident_activation {
        ResidencyBackend::runtime_update_resident_activation(
            backend,
            param.forward_storage().primary_tensor(),
        )?;
    }
    Ok(OptimizerStateAuthority::Host)
}

/// (#1082) Accumulate kt gradients from a kt-native [`kiln_autograd::GradStore`]
/// (keyed by `Parameter::tensor_id()`) into `dst` (a kt `GradMap`).
/// The source must contain exactly one shape/dtype/device-valid gradient for
/// every configured LoRA leaf. Entries are created on the first source and
/// summed thereafter; gradients stay on-device. The optimizer boundary scans
/// the final accumulated values once before any mutation.
pub(crate) fn accumulate_grads(
    dst: &mut GradMap,
    src: &kiln_autograd::GradStore,
    params: &TrainableLoraParams,
) -> Result<()> {
    validate_exact_lora_grad_store_metadata(params, src, "accumulate_grads source")?;
    for (id, grad) in src.iter() {
        if let Some(existing) = dst.get(id) {
            let summed = kiln_tensor::ops::add(existing, grad)
                .map_err(|e| anyhow::anyhow!("accumulate_grads: kt add: {e}"))?;
            dst.insert(*id, summed);
        } else {
            dst.insert(*id, grad.clone());
        }
    }
    // Adding exact sources preserves the destination id set and tensor
    // metadata. Keep this post-merge guard cheap; the final optimizer boundary
    // performs the one required finite-value scan of the accumulated result.
    validate_exact_lora_grad_map_metadata(params, dst, "accumulate_grads result")?;
    Ok(())
}

/// (#1082) [`accumulate_grads`] dispatcher over [`GradSource`] for the
/// GRPO token-level aggregation boundary — kt-native only now. Routes the
/// kt `GradStore` straight into the kt `GradMap` keyed by
/// `Parameter::tensor_id()`.
pub(super) fn accumulate_grads_dispatch(
    dst: &mut GradMap,
    src: &GradSource,
    params: &TrainableLoraParams,
) -> Result<()> {
    match src {
        GradSource::Kt(kt) => accumulate_grads(dst, kt, params),
    }
}

/// (#1082) SGD update from an accumulated kt gradient map (keyed by
/// `Parameter::tensor_id()`).
pub(super) fn sgd_step_from_map(
    backend: &dyn BackendRuntime,
    params: &mut TrainableLoraParams,
    grads: &GradMap,
    lr: f64,
) -> Result<()> {
    let resident_activation = ResidencyBackend::runtime_supports_resident_activation(backend);
    for param in params.all_params_mut() {
        let id = param.tensor_id();
        let grad = grads.get(&id).ok_or_else(|| {
            anyhow::anyhow!("sgd_step_from_map: exact gradient contract lost tensor_id={id}")
        })?;
        apply_sgd_update_kt(backend, param, grad, lr, resident_activation)?;
    }
    Ok(())
}

/// (#1082) Configured-optimizer dispatch from an accumulated kt gradient
/// map (keyed by `Parameter::tensor_id()`). Drives `kiln_optim::AdamW`
/// (or kt SGD) per param.
pub(crate) fn optimizer_step_from_map(
    backend: &dyn BackendRuntime,
    params: &mut TrainableLoraParams,
    grads: &GradMap,
    lr: f64,
    optimizer: Optimizer,
    opt_state: Option<&mut OptimizerState>,
) -> Result<()> {
    validate_exact_lora_grad_map(params, grads, "optimizer_step_from_map")?;
    match optimizer {
        Optimizer::Sgd => sgd_step_from_map(backend, params, grads, lr),
        Optimizer::AdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
        } => {
            let state = opt_state.ok_or_else(|| {
                anyhow::anyhow!("optimizer_step_from_map: AdamW requires OptimizerState")
            })?;
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            // Global 1-indexed step counter (shared by all params), bumped
            // once per optimizer step for AdamW bias correction. Disjoint
            // borrows of `adamw` (mut, host fallback) vs `moments` (shared,
            // device m/v) via the match binding.
            match state {
                OptimizerState::AdamW {
                    adamw,
                    moments,
                    host_authoritative,
                    step,
                } => {
                    *step = step.saturating_add(1);
                    let step = *step;
                    for param in params.all_params_mut() {
                        let id = param.tensor_id();
                        let grad = grads.get(&id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step_from_map: exact gradient contract lost tensor_id={id}"
                            )
                        })?;
                        let m = moments.get(&id);
                        let authority = apply_adamw_update_kt(
                            backend,
                            param,
                            adamw,
                            m,
                            grad,
                            lr,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            step,
                            resident_activation,
                        )?;
                        match authority {
                            OptimizerStateAuthority::Device => {
                                host_authoritative.remove(&id);
                            }
                            OptimizerStateAuthority::Host => {
                                host_authoritative.insert(id);
                            }
                        }
                    }
                    Ok(())
                }
                _ => anyhow::bail!(
                    "optimizer_step_from_map: AdamW optimizer requires AdamW OptimizerState"
                ),
            }
        }
        Optimizer::Muon {
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
        } => {
            let state = opt_state.ok_or_else(|| {
                anyhow::anyhow!("optimizer_step_from_map: Muon requires OptimizerState")
            })?;
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            match state {
                OptimizerState::Muon {
                    muon,
                    momenta,
                    host_authoritative,
                    step,
                } => {
                    *step = step.saturating_add(1);
                    for param in params.all_params_mut() {
                        let id = param.tensor_id();
                        let grad = grads.get(&id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step_from_map: exact gradient contract lost tensor_id={id}"
                            )
                        })?;
                        let mom = momenta.get(&id);
                        let authority = apply_muon_update_kt(
                            backend,
                            param,
                            muon,
                            mom,
                            grad,
                            lr,
                            momentum,
                            nesterov,
                            ns_iters,
                            weight_decay,
                            resident_activation,
                        )?;
                        match authority {
                            OptimizerStateAuthority::Device => {
                                host_authoritative.remove(&id);
                            }
                            OptimizerStateAuthority::Host => {
                                host_authoritative.insert(id);
                            }
                        }
                    }
                    Ok(())
                }
                _ => anyhow::bail!(
                    "optimizer_step_from_map: Muon optimizer requires Muon OptimizerState"
                ),
            }
        }
    }
}

/// (#1082) kt-native-grad consumer — the SOLE optimizer consumer post
/// candle-drop. Reads gradients from a kt-native
/// [`kiln_autograd::GradStore`] (keyed by `Parameter::tensor_id()`,
/// values `kiln_tensor::Tensor`), produced by
/// [`standard_forward_backward_tape_authoritative_kt`] /
/// [`grpo_step_forward_backward_tape_authoritative_kt`] / the
/// checkpointed kt producer.
///
/// For each LoRA `Parameter` it looks the grad up by `tensor_id()` and
/// steps the param kt-natively: SGD via [`apply_sgd_update_kt`], AdamW via
/// `kiln_optim::AdamW` (`OptimStep::step`) inside [`apply_adamw_update_kt`].
/// NO candle grad copy, NO candle `Var` master — the LoRA `Parameter`'s kt
/// master is updated in place (preserving `tensor_id`).
pub(crate) fn optimizer_step_from_kt_grad_store(
    backend: &dyn BackendRuntime,
    params: &mut TrainableLoraParams,
    grads: &kiln_autograd::GradStore,
    lr: f64,
    optimizer: Optimizer,
    opt_state: Option<&mut OptimizerState>,
) -> Result<()> {
    validate_exact_lora_grad_store(params, grads, "optimizer_step_from_kt_grad_store")?;
    match optimizer {
        Optimizer::Sgd => {
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            for param in params.all_params_mut() {
                let id = param.tensor_id();
                let kt_grad = grads.get(id).ok_or_else(|| {
                    anyhow::anyhow!(
                        "optimizer_step_from_kt_grad_store: exact gradient contract lost tensor_id={id}"
                    )
                })?;
                apply_sgd_update_kt(backend, param, kt_grad, lr, resident_activation)?;
            }
            Ok(())
        }
        Optimizer::AdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
        } => {
            let state = opt_state.ok_or_else(|| {
                anyhow::anyhow!("optimizer_step_from_kt_grad_store: AdamW requires OptimizerState")
            })?;
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            // Global 1-indexed step counter (shared by all params), bumped
            // once per optimizer step for AdamW bias correction. Disjoint
            // borrows of `adamw` (mut, host fallback) vs `moments` (shared,
            // device m/v) via the match binding.
            match state {
                OptimizerState::AdamW {
                    adamw,
                    moments,
                    host_authoritative,
                    step,
                } => {
                    *step = step.saturating_add(1);
                    let step = *step;
                    for param in params.all_params_mut() {
                        let id = param.tensor_id();
                        let kt_grad = grads.get(id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step_from_kt_grad_store: exact gradient contract lost tensor_id={id}"
                            )
                        })?;
                        let m = moments.get(&id);
                        let authority = apply_adamw_update_kt(
                            backend,
                            param,
                            adamw,
                            m,
                            kt_grad,
                            lr,
                            beta1,
                            beta2,
                            eps,
                            weight_decay,
                            step,
                            resident_activation,
                        )?;
                        match authority {
                            OptimizerStateAuthority::Device => {
                                host_authoritative.remove(&id);
                            }
                            OptimizerStateAuthority::Host => {
                                host_authoritative.insert(id);
                            }
                        }
                    }
                    Ok(())
                }
                _ => anyhow::bail!(
                    "optimizer_step_from_kt_grad_store: AdamW optimizer requires AdamW OptimizerState"
                ),
            }
        }
        Optimizer::Muon {
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
        } => {
            let state = opt_state.ok_or_else(|| {
                anyhow::anyhow!("optimizer_step_from_kt_grad_store: Muon requires OptimizerState")
            })?;
            let resident_activation =
                ResidencyBackend::runtime_supports_resident_activation(backend);
            match state {
                OptimizerState::Muon {
                    muon,
                    momenta,
                    host_authoritative,
                    step,
                } => {
                    *step = step.saturating_add(1);
                    for param in params.all_params_mut() {
                        let id = param.tensor_id();
                        let kt_grad = grads.get(id).ok_or_else(|| {
                            anyhow::anyhow!(
                                "optimizer_step_from_kt_grad_store: exact gradient contract lost tensor_id={id}"
                            )
                        })?;
                        let mom = momenta.get(&id);
                        let authority = apply_muon_update_kt(
                            backend,
                            param,
                            muon,
                            mom,
                            kt_grad,
                            lr,
                            momentum,
                            nesterov,
                            ns_iters,
                            weight_decay,
                            resident_activation,
                        )?;
                        match authority {
                            OptimizerStateAuthority::Device => {
                                host_authoritative.remove(&id);
                            }
                            OptimizerStateAuthority::Host => {
                                host_authoritative.insert(id);
                            }
                        }
                    }
                    Ok(())
                }
                _ => anyhow::bail!(
                    "optimizer_step_from_kt_grad_store: Muon optimizer requires Muon OptimizerState"
                ),
            }
        }
    }
}

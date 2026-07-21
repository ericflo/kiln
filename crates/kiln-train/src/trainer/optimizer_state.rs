use super::*;

// (#1082) C1 fix: the kt-native restoration of the pre-flip candle
// `AdamWMoments{m,v:Var}`. The on-device CUDA AdamW kernel updates
// param/m/v in place, so it needs real per-param device moment tensors —
// not the param aliased onto itself. `KtAdamWMoments` holds those two
// device tensors; `OptimizerState.moments` maps `Parameter::tensor_id()`
// → `KtAdamWMoments`. The `KtAdamW` instance is the host (non-resident)
// fallback only.

/// (#1082) AdamW per-parameter first/second-moment device tensors.
///
/// `m` and `v` are zero-init kt tensors of the same shape+dtype as the
/// LoRA param master, allocated on the param's device. The on-device
/// CUDA AdamW kernel (`dispatch_adamw_step`) reads+writes both in place
/// each step (decoupled WD on the param, biased moments on m/v). Restores
/// the pre-flip candle `AdamWMoments{m: Var, v: Var}` in kt form.
pub struct KtAdamWMoments {
    pub m: KtTensor,
    pub v: KtTensor,
}

/// (#1082) Muon per-parameter momentum device tensor.
///
/// `m` is a zero-init kt tensor of the same shape+dtype as the LoRA
/// param master, allocated on the param's device. The on-device Muon
/// kernel (`runtime_dispatch_muon_step`) reads+writes it in place each
/// step (heavy-ball momentum), then orthogonalizes the look-ahead and
/// updates the param. Unlike AdamW there is no second moment — Muon's
/// state is a single momentum buffer per parameter.
pub struct KtMuonMomentum {
    pub m: KtTensor,
}

/// (#1082) kt-native optimizer state. One variant per stateful
/// optimizer; SGD is stateless and passes `None`.
///
/// - [`OptimizerState::AdamW`]: per-param device `m`/`v` (the real Adam
///   state on the resident/device path) + the CPU reference
///   `kiln_optim::AdamW` host fallback + a global 1-indexed `step`
///   counter (standard AdamW bias correction).
/// - [`OptimizerState::Muon`]: per-param device momentum `m` (the
///   heavy-ball state the on-device Newton-Schulz kernel updates) + the
///   CPU reference `kiln_optim::Muon` host fallback + a global `step`
///   counter (used only as a stochastic-rounding decorrelator on the
///   host path; Muon needs no bias correction).
///
/// The wrapper keeps the trainer's `opt_state: Option<&mut OptimizerState>`
/// signatures unchanged across all dispatch sites.
pub enum OptimizerState {
    AdamW {
        adamw: KtAdamW,
        moments: HashMap<KtTensorId, KtAdamWMoments>,
        host_authoritative: HashSet<KtTensorId>,
        step: u32,
    },
    Muon {
        muon: KtMuon,
        momenta: HashMap<KtTensorId, KtMuonMomentum>,
        host_authoritative: HashSet<KtTensorId>,
        step: u32,
    },
}

impl OptimizerState {
    /// Register every per-param device state tensor as a resident
    /// activation so the on-device kernel's `has_resident_activation`
    /// gate passes (otherwise it returns `false` → host fallback). For
    /// AdamW that is `m`+`v`; for Muon the single momentum `m`.
    ///
    /// No-op on backends without resident-activation support (the host
    /// `kiln_optim` references handle those).
    pub fn register_with_backend(&self, backend: &dyn BackendRuntime) -> Result<()> {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return Ok(());
        }
        match self {
            OptimizerState::AdamW { moments, .. } => {
                for m in moments.values() {
                    ResidencyBackend::runtime_register_resident_activation(backend, &m.m)?;
                    ResidencyBackend::runtime_register_resident_activation(backend, &m.v)?;
                }
            }
            OptimizerState::Muon { momenta, .. } => {
                for mom in momenta.values() {
                    ResidencyBackend::runtime_register_resident_activation(backend, &mom.m)?;
                }
            }
        }
        Ok(())
    }

    /// Inverse of [`Self::register_with_backend`]: release every state
    /// tensor from the resident registry at training completion.
    pub fn evict_from_backend(&self, backend: &dyn BackendRuntime) {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return;
        }
        match self {
            OptimizerState::AdamW { moments, .. } => {
                for m in moments.values() {
                    ResidencyBackend::runtime_evict_resident_activation(backend, &m.m);
                    ResidencyBackend::runtime_evict_resident_activation(backend, &m.v);
                }
            }
            OptimizerState::Muon { momenta, .. } => {
                for mom in momenta.values() {
                    ResidencyBackend::runtime_evict_resident_activation(backend, &mom.m);
                }
            }
        }
    }

    /// The global 1-indexed optimizer step counter (shared by both
    /// stateful variants).
    pub fn step_count(&self) -> u32 {
        match self {
            OptimizerState::AdamW { step, .. } => *step,
            OptimizerState::Muon { step, .. } => *step,
        }
    }

    pub(super) fn checkpoint_rounding_policy(&self) -> StochasticRoundingPolicy {
        match self {
            OptimizerState::AdamW { adamw, .. } => adamw.rounding_policy(),
            OptimizerState::Muon { muon, .. } => muon.rounding_policy(),
        }
    }

    pub(super) fn checkpoint_state_dtype(&self) -> Result<KtDType> {
        match self {
            OptimizerState::AdamW { moments, .. } => moments
                .values()
                .next()
                .map(|state| state.m.dtype())
                .context("AdamW checkpoint state has no parameter moments"),
            OptimizerState::Muon { momenta, .. } => momenta
                .values()
                .next()
                .map(|state| state.m.dtype())
                .context("Muon checkpoint state has no parameter momentum"),
        }
    }

    /// AdamW per-param moment map, if this is AdamW state (diagnostic /
    /// test accessor).
    pub fn adamw_moments(&self) -> Option<&HashMap<KtTensorId, KtAdamWMoments>> {
        match self {
            OptimizerState::AdamW { moments, .. } => Some(moments),
            OptimizerState::Muon { .. } => None,
        }
    }

    /// Muon per-param momentum map, if this is Muon state (diagnostic /
    /// test accessor).
    pub fn muon_momenta(&self) -> Option<&HashMap<KtTensorId, KtMuonMomentum>> {
        match self {
            OptimizerState::Muon { momenta, .. } => Some(momenta),
            OptimizerState::AdamW { .. } => None,
        }
    }

    /// Pull resident device optimizer buffers into their kt tensor owners.
    /// Host fallback state already lives in `adamw`/`muon` and needs no sync.
    pub fn sync_to_master(&mut self, backend: &dyn BackendRuntime) -> Result<usize> {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return Ok(0);
        }
        fn sync_one(backend: &dyn BackendRuntime, tensor: &mut KtTensor) -> Result<bool> {
            if !ResidencyBackend::runtime_has_resident_activation(backend, tensor) {
                return Ok(false);
            }
            let id = tensor.id();
            let dims = tensor.dims().to_vec();
            let dtype = tensor.dtype();
            if let Some(resolved) = ResidencyBackend::runtime_resolve_resident_activation(
                backend, tensor, &dims, dtype,
            )? {
                let resolved = if resolved.device() == tensor.device() {
                    resolved
                } else {
                    resolved
                        .to_device(tensor.device())
                        .context("realign resolved optimizer state to its owner device")?
                };
                *tensor = checkpoint_tensor_with_id(resolved, id, "optimizer resident sync")?;
                Ok(true)
            } else {
                Ok(false)
            }
        }

        let mut synced = 0;
        match self {
            OptimizerState::AdamW { moments, .. } => {
                for state in moments.values_mut() {
                    synced += usize::from(sync_one(backend, &mut state.m)?);
                    synced += usize::from(sync_one(backend, &mut state.v)?);
                }
            }
            OptimizerState::Muon { momenta, .. } => {
                for state in momenta.values_mut() {
                    synced += usize::from(sync_one(backend, &mut state.m)?);
                }
            }
        }
        Ok(synced)
    }

    /// Capture optimizer tensors by stable parameter name into CPU storage.
    /// Device buffers and CPU fallback state share one F32 safetensors
    /// representation; per-param counters are U32 scalar tensors.
    pub(crate) fn capture_checkpoint_state(
        &mut self,
        params: &TrainableLoraParams,
        backend: &dyn BackendRuntime,
    ) -> Result<CheckpointTensorSnapshot> {
        self.sync_to_master(backend)?;
        let mut owned: Vec<(String, KtTensor)> = Vec::new();
        match self {
            OptimizerState::AdamW {
                adamw,
                moments,
                host_authoritative,
                step,
            } => {
                for (key, param) in params.checkpoint_params() {
                    let id = param.tensor_id();
                    let shape = param.forward_storage().primary_tensor().dims().to_vec();
                    let (m, v, param_step) = if host_authoritative.contains(&id) {
                        let host = adamw.moments(id).with_context(|| {
                            format!("checkpoint AdamW authoritative host moments missing for {key}")
                        })?;
                        anyhow::ensure!(
                            host.m.len() == param.forward_storage().primary_tensor().elem_count()
                                && host.v.len()
                                    == param.forward_storage().primary_tensor().elem_count(),
                            "checkpoint AdamW host moment shape drift for {key}"
                        );
                        (
                            KtTensor::from_vec_on(
                                kiln_tensor::Device::Cpu,
                                host.m.clone(),
                                shape.clone(),
                            )?,
                            KtTensor::from_vec_on(kiln_tensor::Device::Cpu, host.v.clone(), shape)?,
                            u32::try_from(host.step).with_context(|| {
                                format!("checkpoint AdamW per-param step overflow for {key}")
                            })?,
                        )
                    } else {
                        let state = moments.get(&id).with_context(|| {
                            format!("checkpoint AdamW device moments missing for {key}")
                        })?;
                        (
                            checkpoint_tensor_to_cpu_f32(&state.m, &format!("{key}.adamw.m"))?,
                            checkpoint_tensor_to_cpu_f32(&state.v, &format!("{key}.adamw.v"))?,
                            *step,
                        )
                    };
                    checkpoint_ensure_finite_f32(&m, &format!("{key}.adamw.m"))?;
                    checkpoint_ensure_finite_f32(&v, &format!("{key}.adamw.v"))?;
                    owned.push((format!("{key}.adamw.m"), m));
                    owned.push((format!("{key}.adamw.v"), v));
                    owned.push((
                        format!("{key}.adamw.step"),
                        KtTensor::from_vec_on(kiln_tensor::Device::Cpu, vec![param_step], vec![1])?,
                    ));
                }
            }
            OptimizerState::Muon {
                muon,
                momenta,
                host_authoritative,
                step,
            } => {
                for (key, param) in params.checkpoint_params() {
                    let id = param.tensor_id();
                    let shape = param.forward_storage().primary_tensor().dims().to_vec();
                    let (momentum, param_step) = if host_authoritative.contains(&id) {
                        let host = muon.momentum_for(id).with_context(|| {
                            format!("checkpoint Muon authoritative host momentum missing for {key}")
                        })?;
                        anyhow::ensure!(
                            host.m.len() == param.forward_storage().primary_tensor().elem_count(),
                            "checkpoint Muon host momentum shape drift for {key}"
                        );
                        (
                            KtTensor::from_vec_on(kiln_tensor::Device::Cpu, host.m.clone(), shape)?,
                            u32::try_from(host.step).with_context(|| {
                                format!("checkpoint Muon per-param step overflow for {key}")
                            })?,
                        )
                    } else {
                        let state = momenta.get(&id).with_context(|| {
                            format!("checkpoint Muon device momentum missing for {key}")
                        })?;
                        (
                            checkpoint_tensor_to_cpu_f32(
                                &state.m,
                                &format!("{key}.muon.momentum"),
                            )?,
                            *step,
                        )
                    };
                    checkpoint_ensure_finite_f32(&momentum, &format!("{key}.muon.momentum"))?;
                    owned.push((format!("{key}.muon.momentum"), momentum));
                    owned.push((
                        format!("{key}.muon.step"),
                        KtTensor::from_vec_on(kiln_tensor::Device::Cpu, vec![param_step], vec![1])?,
                    ));
                }
            }
        }
        CheckpointTensorSnapshot::new(owned, "optimizer")
    }

    /// Save optimizer state directly. Production loop checkpointing uses the
    /// split capture/publish path below so filesystem latency never extends
    /// the serving GPU write section; this wrapper remains useful to codecs
    /// and focused tests.
    pub fn save_checkpoint_state(
        &mut self,
        params: &TrainableLoraParams,
        backend: &dyn BackendRuntime,
        path: &Path,
    ) -> Result<()> {
        self.capture_checkpoint_state(params, backend)?.save(path)
    }

    /// Restore optimizer tensors into both the device-owned buffers and the
    /// CPU fallback optimizer. Populating both prevents a post-resume routing
    /// change from silently resetting momentum.
    pub fn load_checkpoint_state(
        &mut self,
        params: &TrainableLoraParams,
        path: &Path,
        expected_step: u32,
    ) -> Result<()> {
        let mut loaded = kiln_tensor::safetensors::load_cpu(path)
            .map_err(|error| anyhow::anyhow!("load checkpoint optimizer state: {error}"))?;
        let suffixes: &[&str] = match self {
            OptimizerState::AdamW { .. } => &["adamw.m", "adamw.v", "adamw.step"],
            OptimizerState::Muon { .. } => &["muon.momentum", "muon.step"],
        };
        let expected: BTreeSet<_> = params
            .checkpoint_param_keys()
            .into_iter()
            .flat_map(|key| suffixes.iter().map(move |suffix| format!("{key}.{suffix}")))
            .collect();
        let actual: BTreeSet<_> = loaded.keys().cloned().collect();
        anyhow::ensure!(
            actual == expected,
            "checkpoint optimizer tensor set mismatch: expected {expected:?}, found {actual:?}"
        );

        match self {
            OptimizerState::AdamW {
                adamw,
                moments,
                host_authoritative,
                step,
            } => {
                host_authoritative.clear();
                for (key, param) in params.checkpoint_params() {
                    let id = param.tensor_id();
                    let m_key = format!("{key}.adamw.m");
                    let v_key = format!("{key}.adamw.v");
                    let step_key = format!("{key}.adamw.step");
                    let m = loaded.remove(&m_key).expect("validated AdamW m must exist");
                    let v = loaded.remove(&v_key).expect("validated AdamW v must exist");
                    checkpoint_validate_f32_state_shape(&m, param, &m_key)?;
                    checkpoint_validate_f32_state_shape(&v, param, &v_key)?;
                    checkpoint_ensure_finite_f32(&m, &m_key)?;
                    checkpoint_ensure_finite_f32(&v, &v_key)?;
                    let param_step = checkpoint_read_step(
                        &loaded
                            .remove(&step_key)
                            .expect("validated AdamW step must exist"),
                        &step_key,
                    )?;
                    anyhow::ensure!(
                        param_step <= expected_step,
                        "checkpoint AdamW step {param_step} for {key} exceeds global step {expected_step}"
                    );
                    let state = moments.get_mut(&id).with_context(|| {
                        format!("checkpoint AdamW destination moments missing for {key}")
                    })?;
                    state.m = checkpoint_restore_state_tensor(&m, &state.m, &m_key)?;
                    state.v = checkpoint_restore_state_tensor(&v, &state.v, &v_key)?;
                    adamw.restore_moments(
                        id,
                        KtHostAdamWMoments {
                            m: m.to_vec::<f32>()?,
                            v: v.to_vec::<f32>()?,
                            step: u64::from(param_step),
                            location: KtMomentLocation::Device,
                        },
                    )?;
                }
                *step = expected_step;
            }
            OptimizerState::Muon {
                muon,
                momenta,
                host_authoritative,
                step,
            } => {
                host_authoritative.clear();
                for (key, param) in params.checkpoint_params() {
                    let id = param.tensor_id();
                    let momentum_key = format!("{key}.muon.momentum");
                    let step_key = format!("{key}.muon.step");
                    let momentum = loaded
                        .remove(&momentum_key)
                        .expect("validated Muon momentum must exist");
                    checkpoint_validate_f32_state_shape(&momentum, param, &momentum_key)?;
                    checkpoint_ensure_finite_f32(&momentum, &momentum_key)?;
                    let param_step = checkpoint_read_step(
                        &loaded
                            .remove(&step_key)
                            .expect("validated Muon step must exist"),
                        &step_key,
                    )?;
                    anyhow::ensure!(
                        param_step <= expected_step,
                        "checkpoint Muon step {param_step} for {key} exceeds global step {expected_step}"
                    );
                    let state = momenta.get_mut(&id).with_context(|| {
                        format!("checkpoint Muon destination momentum missing for {key}")
                    })?;
                    state.m = checkpoint_restore_state_tensor(&momentum, &state.m, &momentum_key)?;
                    muon.restore_momentum(
                        id,
                        KtHostMuonState {
                            m: momentum.to_vec::<f32>()?,
                            step: u64::from(param_step),
                        },
                    )?;
                }
                *step = expected_step;
            }
        }
        Ok(())
    }
}

pub(super) fn checkpoint_tensor_to_cpu_f32(tensor: &KtTensor, label: &str) -> Result<KtTensor> {
    tensor
        .to_dtype(KtDType::F32)
        .and_then(|tensor| tensor.to_device(kiln_tensor::Device::Cpu))
        .and_then(|tensor| tensor.contiguous())
        .map_err(|error| anyhow::anyhow!("checkpoint optimizer tensor {label}: {error}"))
}

#[derive(Debug)]
pub(crate) struct CheckpointTensorSnapshot {
    pub(super) kind: &'static str,
    pub(super) tensors: Vec<(String, KtTensor)>,
}

impl CheckpointTensorSnapshot {
    pub(super) fn new(tensors: Vec<(String, KtTensor)>, kind: &'static str) -> Result<Self> {
        let unique_names: BTreeSet<_> = tensors.iter().map(|(name, _)| name.as_str()).collect();
        anyhow::ensure!(
            unique_names.len() == tensors.len(),
            "checkpoint {kind} snapshot contains duplicate tensor names"
        );
        Ok(Self { kind, tensors })
    }

    pub(crate) fn save(&self, path: &Path) -> Result<()> {
        let tensors: HashMap<&str, &KtTensor> = self
            .tensors
            .iter()
            .map(|(key, tensor)| (key.as_str(), tensor))
            .collect();
        kiln_tensor::safetensors::save_cpu(&tensors, path)
            .map_err(|error| anyhow::anyhow!("save checkpoint {} state: {error}", self.kind))
    }
}

pub(super) fn checkpoint_ensure_finite_f32(tensor: &KtTensor, label: &str) -> Result<()> {
    anyhow::ensure!(
        tensor.dtype() == KtDType::F32,
        "checkpoint optimizer tensor {label} must be F32, found {}",
        tensor.dtype()
    );
    anyhow::ensure!(
        tensor
            .to_vec::<f32>()?
            .iter()
            .all(|value| value.is_finite()),
        "checkpoint optimizer tensor {label} contains non-finite values"
    );
    Ok(())
}

pub(super) fn checkpoint_ensure_finite_tensor(tensor: &KtTensor, label: &str) -> Result<()> {
    let values = tensor
        .to_dtype(KtDType::F32)
        .and_then(|tensor| tensor.to_device(kiln_tensor::Device::Cpu))
        .and_then(|tensor| tensor.contiguous())
        .and_then(|tensor| tensor.to_vec::<f32>())
        .map_err(|error| anyhow::anyhow!("read checkpoint tensor {label}: {error}"))?;
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "checkpoint tensor {label} contains non-finite values"
    );
    Ok(())
}

pub(super) fn checkpoint_validate_f32_state_shape(
    tensor: &KtTensor,
    param: &Parameter,
    label: &str,
) -> Result<()> {
    checkpoint_ensure_finite_f32(tensor, label)?;
    let expected = param.forward_storage().primary_tensor().dims();
    anyhow::ensure!(
        tensor.dims() == expected,
        "checkpoint optimizer tensor {label} shape mismatch: expected {expected:?}, found {:?}",
        tensor.dims()
    );
    Ok(())
}

pub(super) fn checkpoint_restore_state_tensor(
    source_f32: &KtTensor,
    destination: &KtTensor,
    label: &str,
) -> Result<KtTensor> {
    let restored = source_f32
        .to_dtype(destination.dtype())
        .and_then(|tensor| tensor.to_device(destination.device()))
        .map_err(|error| anyhow::anyhow!("restore checkpoint optimizer tensor {label}: {error}"))?;
    checkpoint_tensor_with_id(restored, destination.id(), label)
}

pub(super) fn checkpoint_tensor_with_id(
    tensor: KtTensor,
    id: KtTensorId,
    label: &str,
) -> Result<KtTensor> {
    KtTensor::from_parts(tensor.storage().clone(), tensor.layout().clone(), id).map_err(|error| {
        anyhow::anyhow!("preserve checkpoint tensor identity for {label}: {error}")
    })
}

pub(super) fn checkpoint_read_step(tensor: &KtTensor, label: &str) -> Result<u32> {
    anyhow::ensure!(
        tensor.dtype() == KtDType::U32 && tensor.dims() == [1],
        "checkpoint optimizer step tensor {label} must be U32[1], found {}{:?}",
        tensor.dtype(),
        tensor.dims()
    );
    Ok(tensor.to_vec::<u32>()?[0])
}

/// (#1082) Build `Option<OptimizerState>` from the configured optimizer:
/// `None` for SGD (stateless), `Some(KtAdamW-backed state)` for AdamW.
/// Consolidates the three identical production blocks that previously
/// `match`ed `config.optimizer` + pre-allocated candle moment `Var`s.
pub(crate) fn make_opt_state(
    params: &TrainableLoraParams,
    optimizer: Optimizer,
    lr: f64,
    device: &Device,
) -> Result<Option<OptimizerState>> {
    match optimizer {
        Optimizer::Sgd => Ok(None),
        Optimizer::AdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
        } => Ok(Some(params.allocate_adamw_state(
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            device,
        )?)),
        Optimizer::Muon {
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
        } => Ok(Some(params.allocate_muon_state(
            lr,
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
            device,
        )?)),
    }
}

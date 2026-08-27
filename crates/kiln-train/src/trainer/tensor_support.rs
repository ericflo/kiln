use super::*;

// ---------------------------------------------------------------------------
// (#1082) Small candle helpers that consolidate the most frequently-repeated
// inline-qualified `*` patterns in this file. Each helper takes
// and returns candle types (autograd-tracked); the win is purely textual:
// one helper-side `*` reference replaces N caller-side ones.
//
// These are NOT a `use candle_*` import — they are free functions /
// extension traits in the trainer module, so the audit invariant
// "trainer.rs has zero `use candle_*` imports at module top" still holds.
// Every candle prefix here is inline-qualified inside the helper body.
//
// Reduction accounting (post-bb1210dc, baseline 598 lines containing
// `*`):
//   * `TensorCastExt::to_f32_dtype` consolidates the most common pattern
//     in the file — `.to_dtype(DType::F32)?` chained onto a
//     tensor. Each rewritten call site loses one `*`
//     reference (the `DType::F32` qualifier). Net win: roughly `N - 5`
//     lines for N migrated sites.
//   * `zeros_f32_on` consolidates the
//     `Tensor::zeros(shape, DType::F32, device)`
//     constructor. Each rewritten call site loses two refs
//     (`Tensor::zeros` and `DType::F32`).
//   * `cpu_device` consolidates `Device::Cpu`.
// ---------------------------------------------------------------------------

/// Extension trait that consolidates the
/// `.to_dtype(DType::F32)?` cast — the single most common
/// inline-qualified pattern in this file. Method form keeps call sites
/// chainable.
pub(super) trait TensorCastExt {
    fn to_f32_dtype(&self) -> Result<Tensor>;
}

impl TensorCastExt for Tensor {
    #[inline]
    fn to_f32_dtype(&self) -> Result<Tensor> {
        // Note: this body explicitly calls `to_dtype` to avoid infinite
        // recursion via the extension-trait method we are defining here.
        Ok(Tensor::to_dtype(self, DType::F32)?)
    }
}

/// Allocate a zero-filled F32 tensor on `device`. Consolidates the
/// `Tensor::zeros(shape, DType::F32, device)`
/// constructor.
#[inline]
pub(super) fn zeros_f32_on<S: Into<Shape>>(shape: S, device: &Device) -> Result<Tensor> {
    Ok(Tensor::zeros(shape, DType::F32, device)?)
}

/// Return a candle CPU device. Consolidates `Device::Cpu`
/// (~70 sites pre-consolidation, mostly `let device = Device::Cpu;`
/// in `#[cfg(test)]` blocks).
#[inline]
pub(super) fn cpu_device() -> Device {
    Device::Cpu
}

/// The reduction-axis marker for "the last dimension" — passed to
/// `Tensor::sum_keepdim` / `max_keepdim` / `mean_keepdim` / `log_sum_exp`
/// in this file. Consolidates `D::Minus1` (~21 sites
/// pre-consolidation).
pub(super) const LAST_DIM: D = D::Minus1;

// `tensor_new` and `tensor_from_vec` (which require candle `NdArray`
// and `WithDType` generic bounds) have moved to `crate::cd_types` so
// this file holds zero direct candle paths for the generic constructor
// helpers. (#1082)

// (#1082) `var_from_tensor` (candle `Var::from_tensor`) removed: the
// trainable LoRA params are `kiln_param::Parameter` now, built via
// `lora_parameter_from_kt` below. The kt tape is the sole grad producer
// (no candle autograd `Var` tracking).

// ---------------------------------------------------------------------------
// (#1082) Type aliases for the most-repeated candle generic-parameter
// patterns in this file. These are NOT `use candle_*` imports — they are
// `type` aliases local to this module, so the audit invariant "trainer.rs
// has zero `use candle_*` imports at module top" still holds. The aliases
// keep all candle types fully spelled out at the alias definition site;
// every callsite that previously embedded two `*` references
// (e.g. `HashMap<TensorId, Tensor>`) collapses to
// one alias name (`GradMap`), netting out one candle reference per site.
// ---------------------------------------------------------------------------

/// (#1082) Map from a LoRA `Parameter`'s kt `TensorId` to its accumulated
/// kt gradient `Tensor`. Was `HashMap<candle TensorId, candle Tensor>`;
/// now fully kt-native (keys = `Parameter::tensor_id()`, values =
/// `kiln_tensor::Tensor`). Used by GRPO token-level cross-completion grad
/// accumulation.
pub(super) type GradMap = std::collections::HashMap<KtTensorId, KtTensor>;

/// Concatenate a slice of `&Tensor` refs along `dim`. Consolidates the
/// Tensor::cat call site (~7 sites in the segment-level gradient +
/// activation stitching paths).
#[inline]
pub(super) fn cat_tensors(refs: &[&Tensor], dim: usize) -> Result<Tensor> {
    Ok(Tensor::cat(refs, dim)?)
}

/// Allocate a zero-filled tensor with caller-supplied dtype + device.
/// Consolidates the Tensor::zeros constructor (~8 sites in segment / tile /
/// boundary-state init paths where the dtype is not statically F32).
// keep: reserved — the dtype-parameterized siblings of the live
// `zeros_f32_on`; last used by the pre-#1082 candle paths. The in-tree
// replacement record at tests/mod.rs:3007 names them as the production
// helpers the test-only kt helpers (`kt_zeros_f32_on` / `kt_ones_f32_on`)
// superseded at the kt-field sites.
#[allow(dead_code)]
#[inline]
pub(super) fn zeros_dtype_on<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    Ok(Tensor::zeros(shape, dtype, device)?)
}

/// Allocate a ones-filled tensor with caller-supplied dtype + device.
/// Consolidates the Tensor::ones constructor (~5 sites in q_norm/k_norm
/// init + gradient-test fixtures).
// keep: reserved — see `zeros_dtype_on` above (same #1082 replacement
// record at tests/mod.rs:3007).
#[allow(dead_code)]
#[inline]
pub(super) fn ones_dtype_on<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    Ok(Tensor::ones(shape, dtype, device)?)
}

/// (#1082) Allocate a zero-filled LoRA `Parameter` (the LoRA-B init —
/// B=zeros so the initial LoRA contribution is zero). Replaces the candle
/// `Var::zeros` constructor. The AdamW moment allocation that also used
/// `Var::zeros` is gone (`kiln_optim::AdamW` owns its own moments keyed by
/// `Parameter::tensor_id()`).
pub(super) fn lora_param_zeros(
    shape: (usize, usize),
    dtype: DType,
    device: &Device,
) -> Result<Parameter> {
    let n = shape.0 * shape.1;
    let data = vec![0.0f32; n];
    let master = build_lora_master_kt(&data, &[shape.0, shape.1], dtype, device)
        .context("lora_param_zeros: build kt LoRA-B master")?;
    Ok(lora_parameter_from_kt(master))
}

#[inline]
pub(super) fn training_precision_policy_for_device(device: &Device) -> TrainingPrecisionPolicy {
    backend::training_precision_policy_for_device_kt(*device)
}

#[inline]
pub(crate) fn training_precision_policy_for_backend(
    backend: &dyn BackendRuntime,
) -> TrainingPrecisionPolicy {
    TrainingLossBackend::runtime_training_precision_policy(backend)
}

/// Validate the exact optimizer kind, derived LoRA dtype, and immutable write
/// policy before any resident model, trainable parameter, or optimizer-state
/// allocation. The per-step fallback guard remains necessary for dynamic
/// residency/dispatch failures.
pub(crate) fn ensure_training_optimizer_supported(
    workload: &str,
    backend: &dyn BackendRuntime,
    optimizer: Optimizer,
    base_weight_dtype: kiln_tensor::DType,
    lora_rank: usize,
) -> Result<TrainingOptimizerRequest> {
    let capabilities = BackendCapabilityQueries::backend_capabilities(backend);
    capabilities
        .training
        .resolve_optimizer_request(
            optimizer.kind(),
            base_weight_dtype,
            TrainingOptimizerRounding::RoundToNearest,
            lora_rank,
        )
        .map_err(|error| {
            anyhow::anyhow!(
                "{workload} optimizer is unsupported by backend `{}`: {error}",
                BackendIdentity::runtime_name(backend)
            )
        })
}

/// Cheap public-entry validation for optimizer hyperparameters and the exact
/// backend/dtype/rank tuple. Call this before source inspection or governor
/// initialization; execution paths repeat the capability check before their
/// first resident allocation so capability drift still fails closed.
pub(crate) fn ensure_training_optimizer_device_supported(
    workload: &str,
    weights: &GpuWeights,
    runtime_device: Device,
    optimizer: Optimizer,
    lora_rank: usize,
) -> Result<()> {
    optimizer
        .validate_hyperparameters()
        .with_context(|| format!("{workload}: invalid optimizer configuration"))?;
    TrainingOptimizerSupport::for_device(runtime_device)
        .resolve_optimizer_request(
            training_precision_policy_for_device(&runtime_device),
            optimizer.kind(),
            weights.embed_tokens.dtype(),
            TrainingOptimizerRounding::RoundToNearest,
            lora_rank,
        )
        .with_context(|| {
            format!(
                "{workload} optimizer is unsupported for configured runtime device {runtime_device}"
            )
        })?;
    Ok(())
}

pub(crate) fn ensure_training_optimizer_entry_supported(
    workload: &str,
    weights: &GpuWeights,
    runtime: &crate::TrainingRuntimeContext,
    optimizer: Optimizer,
    lora_rank: usize,
) -> Result<Device> {
    let runtime_device = training_device_for_weights(weights, runtime)
        .with_context(|| format!("{workload}: resolve runtime device"))?;
    ensure_training_optimizer_device_supported(
        workload,
        weights,
        runtime_device,
        optimizer,
        lora_rank,
    )?;
    Ok(runtime_device)
}

pub(super) fn training_activation_bytes_per_elem_for_policy(
    weights: &GpuWeights,
    policy: TrainingPrecisionPolicy,
    has_linear_attention: bool,
) -> usize {
    const GDN_TAPE_EFFECTIVE_BYTES_PER_ELEM: usize = 10;

    if policy.uses_f32_activations_for_mixed_base_weights() {
        // Backends with mixed BF16 base weights and F32 training activations
        // keep hidden activations in F32 for the tape path.
        return 4;
    }
    let base = match weights.embed_tokens.dtype() {
        kiln_tensor::DType::BF16 | kiln_tensor::DType::F16 => 2,
        kiln_tensor::DType::F32 => 4,
        _ => 4,
    };
    if has_linear_attention
        || weights
            .layers
            .iter()
            .any(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)))
    {
        // GDN replay records q/k/v, gate, recurrent, qk-norm, and gated-norm
        // tensors in addition to the hidden stream. Use an intentionally
        // inflated effective width so very long contexts prefer one-layer replay
        // scopes on tight VRAM. Long-context SFT spools checkpoint boundaries
        // off-device, so the extra segment count does not pin every boundary on
        // the GPU.
        base.max(GDN_TAPE_EFFECTIVE_BYTES_PER_ELEM)
    } else {
        base
    }
}

pub(crate) fn training_activation_bytes_per_elem_for_backend(
    weights: &GpuWeights,
    backend: &dyn BackendRuntime,
) -> usize {
    training_activation_bytes_per_elem_for_policy(
        weights,
        training_precision_policy_for_backend(backend),
        false,
    )
}

#[cfg(test)]
pub(crate) fn training_activation_bytes_per_elem(weights: &GpuWeights, device: &Device) -> usize {
    training_activation_bytes_per_elem_for_policy(
        weights,
        training_precision_policy_for_device(device),
        false,
    )
}

pub(super) fn model_config_has_linear_attention(model_config: &ModelConfig) -> bool {
    model_config.num_full_attention_layers < model_config.num_layers
}

#[inline]
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub(super) fn final_rmsnorm_backward_route_for_backend(
    backend: &dyn BackendRuntime,
) -> FinalRmsNormBackwardRoute {
    TrainingLossBackend::runtime_final_rmsnorm_backward_route(backend)
}

#[inline]
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub(super) fn grpo_kl_auxiliary_route_for_backend(
    backend: &dyn BackendRuntime,
) -> GrpoKlAuxiliaryRoute {
    TrainingLossBackend::runtime_grpo_kl_auxiliary_route(backend)
}
/// AMP policy for a trainable LoRA parameter, derived from the backend-selected
/// LoRA storage dtype. Vulkan and the portable reference use F32 LoRA tensors;
/// stamping those parameters with the historical hard-coded BF16 policy made
/// the first host optimizer step silently narrow their master to BF16.
#[inline]
pub(super) fn lora_amp_policy(dtype: KtDType) -> KtAmpPolicy {
    match dtype {
        KtDType::F32 => KtAmpPolicy::fp32_reference(),
        KtDType::BF16 => KtAmpPolicy::qwen3p5_4b_default(),
        KtDType::F16 => KtAmpPolicy {
            forward_compute_dtype: KtDType::F16,
            backward_compute_dtype: KtDType::F16,
            master_dtype: KtDType::F16,
            accumulation_dtype: KtDType::F32,
        },
        _ => unreachable!("LoRA parameters require F32, BF16, or F16 storage, got {dtype}"),
    }
}

/// (#1082) Build a trainable LoRA `Parameter` from a kt master tensor.
/// The forward storage IS the master (LoRA A/B are dense BF16, no
/// quantization), so `forward_storage().primary_tensor()` and
/// `backward_storage()` share the same kt tensor. The `Parameter`'s
/// stable kt `tensor_id` becomes the tape grad key + the optimizer
/// moment key.
#[inline]
pub(super) fn lora_parameter_from_kt(master: KtTensor) -> Parameter {
    let policy = lora_amp_policy(master.dtype());
    Parameter::trainable(KtForwardStorage::Plain(master.clone()), master, policy)
}

/// Sample a Kaiming-uniform LoRA-A initialization.
///
/// When `rng` is `Some`, the values are drawn from the supplied RNG so the
/// init is byte-deterministic across runs; this is the path used when the
/// caller passes `seed: Some(_)`. When `rng` is `None`, we fall back to
/// `Var::rand_f64`, which uses the device-global RNG (seeded earlier with
/// `device.set_seed` on backends that support it).
// (#1082) Now returns a kt-native LoRA `Parameter` (Kaiming-uniform A,
// BF16 master) instead of a candle `Var`. The A values are drawn on the
// host (deterministic when `rng` is `Some`) and uploaded to a kt CUDA
// tensor via the bridge so the param's primary kt tensor lives on
// `device` — exactly where the kt tape forward + the resident-activation
// registry expect it. `dtype` is BF16 in production.
pub(super) fn kaiming_uniform_a(
    rng: Option<&mut StdRng>,
    bound: f64,
    shape: (usize, usize),
    dtype: DType,
    device: &Device,
) -> Result<Parameter> {
    let bound_f32 = bound as f32;
    let n = shape.0 * shape.1;
    let data: Vec<f32> = match rng {
        Some(rng) => (0..n)
            .map(|_| rng.random_range(-bound_f32..bound_f32))
            .collect(),
        None => {
            // Deterministic-init contract: callers that pass `seed:
            // Some(_)` always hand us an `rng`. The `None` path (device
            // RNG) is the non-reproducible fallback; draw from a
            // thread RNG so we stay candle-free (candle `Var::rand_f64`
            // is gone with the autograd `Var`).
            let mut trng = StdRng::seed_from_u64(rand::random());
            (0..n)
                .map(|_| trng.random_range(-bound_f32..bound_f32))
                .collect()
        }
    };
    // Build the A master directly as a kt CUDA tensor on `device`.
    let master = build_lora_master_kt(&data, &[shape.0, shape.1], dtype, device)
        .context("kaiming_uniform_a: build kt LoRA-A master")?;
    Ok(lora_parameter_from_kt(master))
}

/// (#1082) Upload f32 host values to a kt tensor on `device`, cast to
/// `dtype` (BF16 in production). Lands directly on the requested device
/// via the candle-free `Tensor::from_vec_on` host->device upload — the
/// host->kt CUDA upload helper the old candle bridge was waiting for now
/// exists in kiln-tensor, so this is fully kt-native (no candle hop).
pub(super) fn build_lora_master_kt(
    data: &[f32],
    shape: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<KtTensor> {
    // Land the f32 host data on `device` (CPU direct, CUDA via H2D copy),
    // then cast to the requested dtype (BF16 in production).
    Tensor::from_vec_on(*device, data.to_vec(), shape.to_vec())?
        .to_dtype(dtype)
        .map_err(|e| anyhow::anyhow!("build_lora_master_kt: to_dtype: {e}"))
}

/// Convert our ChatMessage to the core tokenizer's ChatMessage.
pub(super) fn to_core_messages(msgs: &[ChatMessage]) -> Vec<kiln_core::tokenizer::ChatMessage> {
    msgs.to_vec()
}

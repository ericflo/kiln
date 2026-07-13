//! Backend runtime abstraction for Kiln's platform-specific kernels.
//!
//! Most of the forward pass is expressed as `candle_core::Tensor` ops that
//! run on any candle device. A few ops — FlashAttention-2 forward /
//! paged-decode and the Gated DeltaNet fused recurrent + forward-substitution
//! kernels — have no candle equivalent and are implemented per-platform as
//! CUDA or (later) Metal kernels. This trait is the seam that lets the
//! forward pass dispatch those ops without threading `#[cfg(feature = "cuda")]`
//! gates through every call site.
//!
//! **`Option<candle_core::Tensor>` return**: `Ok(None)` means "this backend declines this
//! call — fall back to the portable candle path". Matches the existing
//! `try_flash_attn_paged_decode` precondition-miss contract and extends it
//! to all kernel ops.
//!
//! **`supports_*` hints**: let the caller skip preamble work (e.g., a
//! `contiguous()` copy before the trait call) when the backend will decline
//! anyway. Intended to be constant-return for each concrete backend.

use anyhow::Result;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Process-global flag set when Vulkan is the active backend.
///
/// candle-core has no `candle_core::Device::Vulkan`, so call sites in `forward.rs` and
/// `trainer.rs` see `candle_core::Device::Cpu` even when the real compute lives on a
/// `vk::Device`. They use this flag to choose Vulkan-aware behavior
/// (e.g., always dropping the per-projection candle CPU originals after
/// upload, since on Vulkan they would double the system-RAM footprint
/// of every weight) without having to thread a `BackendRuntime` handle
/// through every helper.
static VULKAN_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Mark that the Vulkan backend has been selected for this process.
///
/// Idempotent. Safe to call from device-selection paths and from
/// `for_device`'s Vulkan arm so the flag is set even when tests skip
/// the server-level device selection.
pub fn mark_vulkan_active() {
    VULKAN_ACTIVE.store(true, Ordering::Relaxed);
}

/// Returns true once `mark_vulkan_active()` has been called in this process.
/// Whether THIS process trains on the Vulkan hybrid substrate: vulkan
/// feature compiled in AND a Vulkan device present (same runtime probe
/// `for_device_kt` keys on). Used by the trainer to place training
/// state (LoRA params, optimizer moments) on the activation device.
#[cfg(feature = "vulkan")]
pub fn vulkan_training_substrate_active() -> bool {
    vulkan::vulkan_is_available()
}

pub fn vulkan_active() -> bool {
    VULKAN_ACTIVE.load(Ordering::Relaxed)
}

/// Test-only helper: lets unit tests assert the behavior of
/// `vulkan_active()`-gated code without polluting other tests' view of the
/// flag. Reset to the prior value via the returned guard.
#[cfg(test)]
pub fn test_only_set_vulkan_active(value: bool) -> VulkanActiveGuard {
    let prev = VULKAN_ACTIVE.swap(value, Ordering::Relaxed);
    VulkanActiveGuard { prev }
}

#[cfg(test)]
pub struct VulkanActiveGuard {
    prev: bool,
}

#[cfg(test)]
impl Drop for VulkanActiveGuard {
    fn drop(&mut self) {
        VULKAN_ACTIVE.store(self.prev, Ordering::Relaxed);
    }
}

pub mod cpu;

pub mod capability;

pub mod residency;

#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) mod cuda_rocm_common;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "metal")]
pub(crate) mod metal_attention;

#[cfg(feature = "metal")]
pub(crate) mod metal_config;

#[cfg(feature = "metal")]
pub(crate) mod metal_conv1d;

#[cfg(feature = "metal")]
pub(crate) mod metal_core;

#[cfg(feature = "metal")]
pub(crate) mod metal_dense;

#[cfg(feature = "metal")]
pub(crate) mod metal_gdn;

#[cfg(feature = "metal")]
pub(crate) mod metal_icb;

#[cfg(feature = "metal")]
pub(crate) mod metal_lm_head;

#[cfg(feature = "metal")]
pub(crate) mod metal_msl;

#[cfg(feature = "metal")]
pub(crate) mod metal_norm;

#[cfg(feature = "metal")]
pub(crate) mod metal_paged;

#[cfg(feature = "metal")]
pub(crate) mod metal_pipeline;

#[cfg(feature = "metal")]
pub(crate) mod metal_precompile;

#[cfg(feature = "metal")]
pub(crate) mod metal_residency;

#[cfg(feature = "metal")]
pub(crate) mod metal_runtime;

#[cfg(feature = "metal")]
pub(crate) mod metal_training;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_config;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_conv1d;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_decode_state;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_dense;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_device;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_gdn;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_tensor_bridge;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_attention;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_linear;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_training;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_residency;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_resources;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_weights;

#[cfg(feature = "rocm")]
pub mod rocm;

// (#1082) backend::vulkan_linear_op + vulkan_lora_op removed: those
// `candle_core::CustomOp1` / `CustomOp3` wrappers existed only to wire the
// Vulkan matmul / LoRA-delta dispatch into candle's `.backward()`. With the
// kt autograd tape (`kiln_autograd`) as the sole grad producer, the candle
// autograd islands are dead — `VulkanBackend::{linear_prefill_apply,
// lora_delta_resident}` now decline so the kt-recorded forward path owns the
// projection / LoRA matmuls and the tape produces their gradients.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SftFlceLossRoute {
    /// Materialize logits and use the portable loss path.
    FullLogits,
    /// Use the kt-tape FLCE loss root shared by CUDA and ROCm tensors.
    KtTapeFlce,
    /// Use Vulkan's active-row fused SFT FLCE shaders.
    VulkanActiveRows,
}

impl SftFlceLossRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            SftFlceLossRoute::FullLogits => "full_logits",
            SftFlceLossRoute::KtTapeFlce => "kt_tape_flce",
            SftFlceLossRoute::VulkanActiveRows => "vulkan_active_rows",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrpoLossRoute {
    /// Use the shared kt composite GRPO loss root and backend/device fast paths it owns.
    KtComposite,
    /// Use Vulkan's active-row fused GRPO loss shaders.
    VulkanActiveRows,
}

impl GrpoLossRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            GrpoLossRoute::KtComposite => "kt_composite",
            GrpoLossRoute::VulkanActiveRows => "vulkan_active_rows",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrpoKlAuxiliaryRoute {
    /// Use host-side composite threshold/coeff computation.
    HostComposite,
    /// Use CUDA/ROCm device-side auxiliary reductions when their envelopes match.
    CudaRocmDeviceFastPath,
}

impl GrpoKlAuxiliaryRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            GrpoKlAuxiliaryRoute::HostComposite => "host_composite",
            GrpoKlAuxiliaryRoute::CudaRocmDeviceFastPath => "cuda_rocm_device_fast_path",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpdLossRoute {
    /// OPD is not advertised for the portable backend capability surface.
    Unsupported,
    /// Use the shared kt-tape Phase-B OPD loss root.
    KtTapePhaseB,
    /// Use Vulkan's active-hidden fused OPD shaders.
    VulkanActiveHidden,
}

impl OpdLossRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            OpdLossRoute::Unsupported => "unsupported",
            OpdLossRoute::KtTapePhaseB => "kt_tape_phase_b",
            OpdLossRoute::VulkanActiveHidden => "vulkan_active_hidden",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpdPhaseBBackwardRoute {
    /// OPD Phase-B backward is not advertised for this backend.
    Unsupported,
    /// Use the device-agnostic kt composite backward.
    KtComposite,
    /// Use the CUDA/ROCm fused unit-gradient Phase-B backward leaf.
    CudaRocmFusedUnitGrad,
    /// Use Vulkan's active-hidden fused loss/gradient shader pair.
    VulkanActiveHidden,
}

impl OpdPhaseBBackwardRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            OpdPhaseBBackwardRoute::Unsupported => "unsupported",
            OpdPhaseBBackwardRoute::KtComposite => "kt_composite",
            OpdPhaseBBackwardRoute::CudaRocmFusedUnitGrad => "cuda_rocm_fused_unit_grad",
            OpdPhaseBBackwardRoute::VulkanActiveHidden => "vulkan_active_hidden",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalRmsNormBackwardRoute {
    /// Use the device-agnostic kt composite final RMSNorm backward.
    KtComposite,
    /// Allow the CUDA/ROCm fused final-RMSNorm tail leaf when its shape/dtype
    /// envelope matches, falling back to the kt composite math otherwise.
    CudaRocmFusedTail,
}

impl FinalRmsNormBackwardRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            FinalRmsNormBackwardRoute::KtComposite => "kt_composite",
            FinalRmsNormBackwardRoute::CudaRocmFusedTail => "cuda_rocm_fused_tail",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingTapeRoute {
    /// kt tape-authoritative forward/backward is not advertised.
    Unsupported,
    /// Use kt tape-authoritative forward/backward, with dtype constraints
    /// supplied by [`TrainingPrecisionPolicy`].
    KtTapeAuthoritative,
}

impl TrainingTapeRoute {
    pub const fn as_str(self) -> &'static str {
        match self {
            TrainingTapeRoute::Unsupported => "unsupported",
            TrainingTapeRoute::KtTapeAuthoritative => "kt_tape_authoritative",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingCapabilities {
    pub projection_training: &'static str,
    pub flce_loss: &'static str,
    pub tape_forward_backward_route: TrainingTapeRoute,
    pub sft_flce_loss_route: SftFlceLossRoute,
    pub grpo_loss_route: GrpoLossRoute,
    pub grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute,
    pub opd_loss_route: OpdLossRoute,
    pub opd_phase_b_backward_route: OpdPhaseBBackwardRoute,
    pub final_rmsnorm_backward_route: FinalRmsNormBackwardRoute,
    pub rmsnorm_training: &'static str,
    pub resident_activation: &'static str,
    pub lora_delta_training: &'static str,
    pub sgd_step: &'static str,
    pub adamw_step: &'static str,
    pub native_training: &'static str,
}

impl TrainingCapabilities {
    pub const fn portable() -> Self {
        Self {
            projection_training: "portable candle autograd",
            flce_loss: "portable candle/FLCE dispatch when configured",
            tape_forward_backward_route: TrainingTapeRoute::Unsupported,
            sft_flce_loss_route: SftFlceLossRoute::FullLogits,
            grpo_loss_route: GrpoLossRoute::KtComposite,
            grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute::HostComposite,
            opd_loss_route: OpdLossRoute::Unsupported,
            opd_phase_b_backward_route: OpdPhaseBBackwardRoute::Unsupported,
            final_rmsnorm_backward_route: FinalRmsNormBackwardRoute::KtComposite,
            rmsnorm_training: "portable candle autograd",
            resident_activation: "not implemented",
            lora_delta_training: "portable candle autograd",
            sgd_step: "portable candle Var::set",
            adamw_step: "portable candle Var::set",
            native_training: "not implemented",
        }
    }
}

const TRAINING_DTYPE_F32: &[kiln_tensor::DType] = &[kiln_tensor::DType::F32];
const TRAINING_DTYPE_F32_BF16: &[kiln_tensor::DType] =
    &[kiln_tensor::DType::F32, kiln_tensor::DType::BF16];
const TRAINING_DTYPE_FLOAT_NATIVE: &[kiln_tensor::DType] = &[
    kiln_tensor::DType::F32,
    kiln_tensor::DType::BF16,
    kiln_tensor::DType::F16,
];
const TRAINING_DTYPE_BF16_FOCUSED: &[kiln_tensor::DType] = &[kiln_tensor::DType::BF16];

/// Backend precision contract for shared kt-tape training.
///
/// This is separate from [`TrainingCapabilities`]: capabilities describe which
/// hooks are native, while this policy describes the dtype envelope those hooks
/// expect. Phase 6 uses this to keep SFT/GRPO/OPD orchestration backend-neutral
/// while preserving Vulkan's mixed F32/BF16 model and Metal's BF16-focused path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrainingPrecisionPolicy {
    pub name: &'static str,
    pub activation_dtypes: &'static [kiln_tensor::DType],
    pub base_weight_dtypes: &'static [kiln_tensor::DType],
    pub lora_parameter_dtypes: &'static [kiln_tensor::DType],
    pub loss_accumulation_dtype: kiln_tensor::DType,
    pub optimizer_parameter_dtypes: &'static [kiln_tensor::DType],
    pub mixed_rms_norm_weight_dtype: Option<kiln_tensor::DType>,
    pub streaming_prefill_tile_tokens: usize,
    pub detached_full_attn_tile_tokens: usize,
    pub detached_full_attn_boundary_tile_tokens: usize,
    pub detached_full_attn_tape_replay_tile_tokens: usize,
    pub tape_streaming_tile_tokens: usize,
    pub paged_prefill_medium_tile_tokens: Option<usize>,
    pub paged_prefill_medium_tile_max_tokens: Option<usize>,
    pub exact_gdn_backward_tile_tokens: Option<usize>,
    pub mixed_precision: bool,
    pub notes: &'static str,
}

impl TrainingPrecisionPolicy {
    pub const fn portable() -> Self {
        Self {
            name: "cpu_f32_reference",
            activation_dtypes: TRAINING_DTYPE_F32,
            base_weight_dtypes: TRAINING_DTYPE_F32,
            lora_parameter_dtypes: TRAINING_DTYPE_F32,
            loss_accumulation_dtype: kiln_tensor::DType::F32,
            optimizer_parameter_dtypes: TRAINING_DTYPE_F32,
            mixed_rms_norm_weight_dtype: None,
            streaming_prefill_tile_tokens: 8192,
            detached_full_attn_tile_tokens: 8192,
            detached_full_attn_boundary_tile_tokens: 8192,
            detached_full_attn_tape_replay_tile_tokens: 8192,
            tape_streaming_tile_tokens: 8192,
            paged_prefill_medium_tile_tokens: None,
            paged_prefill_medium_tile_max_tokens: None,
            exact_gdn_backward_tile_tokens: None,
            mixed_precision: false,
            notes: "CPU reference training uses F32 tensors and portable optimizer math.",
        }
    }

    pub const fn cuda() -> Self {
        Self {
            name: "cuda_native_float",
            activation_dtypes: TRAINING_DTYPE_FLOAT_NATIVE,
            base_weight_dtypes: TRAINING_DTYPE_FLOAT_NATIVE,
            lora_parameter_dtypes: TRAINING_DTYPE_F32_BF16,
            loss_accumulation_dtype: kiln_tensor::DType::F32,
            optimizer_parameter_dtypes: TRAINING_DTYPE_F32_BF16,
            mixed_rms_norm_weight_dtype: None,
            streaming_prefill_tile_tokens: 1024,
            detached_full_attn_tile_tokens: 8192,
            detached_full_attn_boundary_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_FLASH_DEFAULT_TILE,
            detached_full_attn_tape_replay_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_FLASH_DEFAULT_TILE,
            tape_streaming_tile_tokens: 1024,
            paged_prefill_medium_tile_tokens: None,
            paged_prefill_medium_tile_max_tokens: None,
            exact_gdn_backward_tile_tokens: Some(1024),
            mixed_precision: true,
            notes: "CUDA keeps kt tape authoritative and routes BF16/F16/F32 leaves through CUDA-native kernels where available.",
        }
    }

    pub const fn rocm() -> Self {
        Self {
            name: "rocm_native_float",
            activation_dtypes: TRAINING_DTYPE_FLOAT_NATIVE,
            base_weight_dtypes: TRAINING_DTYPE_FLOAT_NATIVE,
            lora_parameter_dtypes: TRAINING_DTYPE_F32_BF16,
            loss_accumulation_dtype: kiln_tensor::DType::F32,
            optimizer_parameter_dtypes: TRAINING_DTYPE_F32_BF16,
            mixed_rms_norm_weight_dtype: None,
            streaming_prefill_tile_tokens: 1024,
            detached_full_attn_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE,
            detached_full_attn_boundary_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE,
            detached_full_attn_tape_replay_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE,
            tape_streaming_tile_tokens: 1024,
            paged_prefill_medium_tile_tokens: Some(1024),
            paged_prefill_medium_tile_max_tokens: Some(20_000),
            exact_gdn_backward_tile_tokens: None,
            mixed_precision: true,
            notes: "ROCm mirrors CUDA's kt-tape dtype envelope while dispatching through HIP/hipBLASLt-native leaves where available; materializing SDPA paths dynamically shrink exact full-attention tiles to fit live memory.",
        }
    }

    pub const fn metal() -> Self {
        Self {
            name: "metal_bf16_uma",
            activation_dtypes: TRAINING_DTYPE_BF16_FOCUSED,
            base_weight_dtypes: TRAINING_DTYPE_BF16_FOCUSED,
            lora_parameter_dtypes: TRAINING_DTYPE_F32_BF16,
            loss_accumulation_dtype: kiln_tensor::DType::F32,
            optimizer_parameter_dtypes: TRAINING_DTYPE_F32_BF16,
            mixed_rms_norm_weight_dtype: None,
            streaming_prefill_tile_tokens: 2048,
            detached_full_attn_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE,
            detached_full_attn_boundary_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE,
            detached_full_attn_tape_replay_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE,
            tape_streaming_tile_tokens: 2048,
            paged_prefill_medium_tile_tokens: None,
            paged_prefill_medium_tile_max_tokens: None,
            exact_gdn_backward_tile_tokens: None,
            mixed_precision: true,
            notes: "Metal training is BF16-focused on UMA buffers, with F32 loss accumulation and F32/BF16 AdamW residency.",
        }
    }

    pub const fn vulkan() -> Self {
        Self {
            name: "vulkan_mixed_f32_bf16",
            activation_dtypes: TRAINING_DTYPE_F32,
            base_weight_dtypes: TRAINING_DTYPE_F32_BF16,
            lora_parameter_dtypes: TRAINING_DTYPE_F32,
            loss_accumulation_dtype: kiln_tensor::DType::F32,
            optimizer_parameter_dtypes: TRAINING_DTYPE_F32_BF16,
            mixed_rms_norm_weight_dtype: Some(kiln_tensor::DType::BF16),
            streaming_prefill_tile_tokens: 2048,
            detached_full_attn_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE,
            detached_full_attn_boundary_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE,
            detached_full_attn_tape_replay_tile_tokens:
                crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE,
            tape_streaming_tile_tokens: 2048,
            paged_prefill_medium_tile_tokens: None,
            paged_prefill_medium_tile_max_tokens: None,
            exact_gdn_backward_tile_tokens: None,
            mixed_precision: true,
            notes: "Vulkan keeps training activations and LoRA parameters F32 while allowing BF16 base weights through explicit VkTensor buffer bridges.",
        }
    }

    #[cfg(test)]
    pub const fn for_device_family(device: kiln_tensor::Device) -> Self {
        match device {
            kiln_tensor::Device::Cuda(_) => Self::cuda(),
            kiln_tensor::Device::Rocm(_) => Self::rocm(),
            kiln_tensor::Device::Metal(_) => Self::metal(),
            kiln_tensor::Device::Vulkan(_) => Self::vulkan(),
            kiln_tensor::Device::Cpu => Self::portable(),
            _ => Self::portable(),
        }
    }

    pub fn lora_parameter_dtype_for_base_weight(
        &self,
        base_weight_dtype: kiln_tensor::DType,
    ) -> kiln_tensor::DType {
        if self.lora_parameter_dtypes.len() == 1 {
            self.lora_parameter_dtypes[0]
        } else {
            base_weight_dtype
        }
    }

    pub fn uses_f32_activations_for_mixed_base_weights(&self) -> bool {
        self.activation_dtypes.len() == 1
            && self.activation_dtypes[0] == kiln_tensor::DType::F32
            && self.lora_parameter_dtypes.len() == 1
            && self.lora_parameter_dtypes[0] == kiln_tensor::DType::F32
            && self
                .base_weight_dtypes
                .iter()
                .any(|dtype| *dtype == kiln_tensor::DType::BF16)
    }

    pub fn supports_rms_norm_weight_dtype_for_activation(
        &self,
        activation_dtype: kiln_tensor::DType,
        weight_dtype: kiln_tensor::DType,
    ) -> bool {
        if activation_dtype == weight_dtype {
            return self
                .activation_dtypes
                .iter()
                .any(|dtype| *dtype == activation_dtype);
        }
        self.uses_f32_activations_for_mixed_base_weights()
            && activation_dtype == kiln_tensor::DType::F32
            && self.mixed_rms_norm_weight_dtype == Some(weight_dtype)
    }

    pub fn supports_mixed_base_weight_dtype_for_activation(
        &self,
        activation_dtype: kiln_tensor::DType,
        weight_dtype: kiln_tensor::DType,
    ) -> bool {
        activation_dtype != weight_dtype
            && self.uses_f32_activations_for_mixed_base_weights()
            && activation_dtype == kiln_tensor::DType::F32
            && self
                .base_weight_dtypes
                .iter()
                .any(|dtype| *dtype == weight_dtype)
    }

    pub fn activation_dtype_for_embedding_output(
        &self,
        embedding_dtype: kiln_tensor::DType,
    ) -> kiln_tensor::DType {
        if self.uses_f32_activations_for_mixed_base_weights()
            && embedding_dtype == kiln_tensor::DType::BF16
        {
            kiln_tensor::DType::F32
        } else {
            embedding_dtype
        }
    }

    pub fn exact_gdn_backward_tile_tokens_or(&self, fallback: usize) -> usize {
        self.exact_gdn_backward_tile_tokens.unwrap_or(fallback)
    }

    pub fn streaming_prefill_tile_tokens_for_seq_len(&self, seq_len: usize) -> usize {
        match (
            self.paged_prefill_medium_tile_tokens,
            self.paged_prefill_medium_tile_max_tokens,
        ) {
            (Some(tile), Some(max_tokens)) if seq_len <= max_tokens => tile,
            _ => self.streaming_prefill_tile_tokens,
        }
    }
}

/// Policy for backend fallbacks that leave the intended native path.
///
/// Phase 2 uses this to make decode/training behavior explicit: correctness
/// paths can still use portable CPU references, while hot paths can require a
/// backend-native implementation or fail with a clear error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackPolicy {
    /// CPU reference or low-risk correctness path; portable fallback is allowed.
    CorrectnessAllowed,
    /// Fallback is allowed, but callers should record or log that it happened.
    WarnAndCount,
    /// The call is on a hot path; falling back should surface an error.
    ErrorInHotPath,
    /// A native backend implementation is required for this operation.
    NativeRequired,
}

impl FallbackPolicy {
    pub const fn allows_fallback(self) -> bool {
        matches!(
            self,
            FallbackPolicy::CorrectnessAllowed | FallbackPolicy::WarnAndCount
        )
    }
}

fn startup_backend_kind(name: &str, device: kiln_tensor::Device) -> kiln_tensor::Backend {
    match name {
        "cpu" => kiln_tensor::Backend::Cpu,
        "cuda" => kiln_tensor::Backend::Cuda,
        "metal" => kiln_tensor::Backend::Metal,
        "vulkan" => kiln_tensor::Backend::Vulkan,
        "rocm" => kiln_tensor::Backend::Rocm,
        _ => device.backend(),
    }
}

fn precompile_startup_kernels_for_backend(name: &str, device: kiln_tensor::Device) -> Result<()> {
    match startup_backend_kind(name, device) {
        #[cfg(feature = "metal")]
        kiln_tensor::Backend::Metal => {
            let start = std::time::Instant::now();
            match metal::precompile_custom_kernels(&device) {
                Ok(()) => tracing::info!(
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "Metal custom kernels precompiled during background prewarm"
                ),
                Err(err) => tracing::warn!(
                    error = %err,
                    "Metal custom kernel precompile failed; falling back to lazy compilation"
                ),
            }
            Ok(())
        }
        #[cfg(feature = "vulkan")]
        kiln_tensor::Backend::Vulkan => {
            let start = std::time::Instant::now();
            match vulkan::precompile_custom_kernels() {
                Ok(()) => tracing::info!(
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "Vulkan custom kernels precompiled during background prewarm"
                ),
                Err(err) => tracing::warn!(
                    error = %err,
                    "Vulkan custom kernel precompile failed; falling back to lazy compilation"
                ),
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub trait BackendRuntime:
    BackendIdentity
    + StartupBackend
    + ExternalYieldBackend
    + AttentionBackend
    + GdnBackend
    + ConvBackend
    + LinearBackend
    + ResidencyBackend
    + SamplingBackend
    + OptimizerBackend
    + PagedKvBackend
    + ReplayBackend
    + TrainingLossBackend
    + Send
    + Sync
    + std::fmt::Debug
{
    /// Human-readable name (`"cuda"`, `"metal"`, `"cpu"`). Surfaced in
    /// `/health` and logs.
    fn name(&self) -> &'static str {
        BackendIdentity::runtime_name(self)
    }

    /// The `kiln_tensor::Device` this backend drives. All tensors passed to
    /// trait methods must live on this device.
    ///
    /// Returned by value: `kiln_tensor::Device` is `Copy` and small (one
    /// discriminant + a `usize` index). Phase 7 of #1082 migrated the
    /// return type off `&candle_core::Device` so backend selection no
    /// longer threads candle types through every trait method. Backends
    /// that still need a candle `candle_core::Device` internally (e.g. for the kernel
    /// trait methods that take `candle_core::Tensor` parameters) keep a
    /// candle device cached alongside the kt one and bridge as needed.
    fn device(&self) -> kiln_tensor::Device {
        BackendIdentity::runtime_device(self)
    }

    /// `dyn Any` downcast target. Used by the Vulkan-resident decode
    /// fast-path in `transformer_block_paged_with_rope_tables` to recover
    /// the concrete `VulkanBackend` for direct access to its
    /// resident-decode primitives. The default impl returns a
    /// no-op-`Any`-shaped reference; the concrete backend overrides
    /// this to return `self`.
    fn as_any(&self) -> &dyn std::any::Any {
        BackendIdentity::runtime_as_any(self)
    }
}

/// Backend identity facet for the Phase 1 `BackendRuntime` split.
#[allow(clippy::too_many_arguments)]
pub trait BackendIdentity: Send + Sync + std::fmt::Debug {
    fn runtime_name(&self) -> &'static str;

    fn runtime_device(&self) -> kiln_tensor::Device;

    fn runtime_as_any(&self) -> &dyn std::any::Any;
}

/// Focused startup/prewarm facet for backend-owned startup work.
pub trait StartupBackend: BackendIdentity + Send + Sync + std::fmt::Debug {
    fn runtime_precompile_startup_kernels(&self) -> Result<()> {
        precompile_startup_kernels_for_backend(self.runtime_name(), self.runtime_device())
    }
}

/// Synchronization boundary required before model execution yields control to
/// an external scheduler such as the serving actor.
///
/// Implementations must wait for every backend-owned queue or stream that may
/// still be reading or mutating model state. A successful return makes it safe
/// for the scheduler to publish partial progress, run another request, or
/// discard resources owned by the yielded operation. This is deliberately a
/// backend facet rather than a `kiln_tensor::Device` dispatch: some runtimes,
/// most notably Vulkan, submit through both a tensor companion device and a
/// backend-private logical device, and only the backend knows the complete set
/// of queues that must be drained.
///
/// The caller must hold its model/backend execution serialization guard from
/// the final submission through this call and the subsequent progress commit.
/// The boundary covers work submitted before the call; it is not itself a
/// scheduling mutex for concurrent submissions.
pub trait ExternalYieldBackend: BackendIdentity + Send + Sync + std::fmt::Debug {
    /// On error, completion is unknown. Callers must stop or quarantine the
    /// runtime rather than publish progress or recycle mutable device state.
    fn runtime_synchronize_external_yield(&self) -> Result<()>;
}

/// Focused `AttentionBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait AttentionBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_flash_attn_prefill(&self) -> bool {
        false
    }

    fn runtime_supports_flash_attn_prefill_head_major(&self) -> bool {
        false
    }

    fn runtime_supports_flash_attn_paged_decode(&self) -> bool {
        false
    }

    fn runtime_flash_attn_paged_decode_contiguous(
        &self,
        _q: &kiln_tensor::Tensor,
        _k_pool: &kiln_tensor::Tensor,
        _v_pool: &kiln_tensor::Tensor,
        _start_slot: usize,
        _total_seqlen_k: usize,
        _softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_flash_attn_paged_decode_contiguous_batch(
        &self,
        _q: &kiln_tensor::Tensor,
        _k_pool: &kiln_tensor::Tensor,
        _v_pool: &kiln_tensor::Tensor,
        _start_slots: &kiln_tensor::Tensor,
        _total_seqlen_k: usize,
        _softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_supports_strict_paged_decode_contiguous_batch(&self) -> bool {
        true
    }

    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
        &self,
        _q: &kiln_tensor::Tensor,
        _k_pool: &kiln_tensor::Tensor,
        _v_pool: &kiln_tensor::Tensor,
        _block_table: &kiln_tensor::Tensor,
        _seqused_k: &kiln_tensor::Tensor,
        _max_seqlen_k: usize,
        _page_block_size: usize,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_flash_attn_prefill(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_flash_attn_prefill_head_major(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_flash_attn_paged_decode(
        &self,
        _q: &kiln_tensor::Tensor,
        _k_pool: &kiln_tensor::Tensor,
        _v_pool: &kiln_tensor::Tensor,
        _block_table: &kiln_tensor::Tensor,
        _total_seqlen_k: usize,
        _page_block_size: usize,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }
}

/// Focused `PagedKvBackend` facet for paged KV cache materialization helpers.
#[allow(clippy::too_many_arguments)]
pub trait PagedKvBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_paged_kv_head_major_read(&self) -> bool {
        false
    }

    fn runtime_supports_paged_kv_head_major_read_append_token_major(&self) -> bool {
        false
    }

    fn runtime_paged_kv_head_major_read(
        &self,
        _k_pool: &kiln_tensor::Tensor,
        _v_pool: &kiln_tensor::Tensor,
        _start_slot: usize,
        _seq_len: usize,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        Ok(None)
    }

    fn runtime_paged_kv_head_major_read_append_token_major(
        &self,
        _k_pool: &kiln_tensor::Tensor,
        _v_pool: &kiln_tensor::Tensor,
        _start_slot: usize,
        _prefix_len: usize,
        _k_tail: &kiln_tensor::Tensor,
        _v_tail: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        Ok(None)
    }
}

/// Focused `GdnBackend` facet for Gated DeltaNet kernels.
#[allow(clippy::too_many_arguments)]
pub trait GdnBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_gdn_forward_substitution(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_recurrent_step(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_chunk_prep(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_chunk_scan(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_full_chunk_forward(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_full_chunk_forward_head_last(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_recurrent_prefill_head_last(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_recurrent_prefill_native_head_last(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_recurrent_qk_norm_prefill_native_head_last(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_decode_qk_norm_gates_recurrent(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_gates(&self) -> bool {
        false
    }

    fn runtime_supports_gdn_gated_rms_norm(&self) -> bool {
        false
    }

    fn runtime_gdn_forward_substitution(
        &self,
        _a_strict: &kiln_tensor::Tensor,
        _v_prime: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_solve_tri_transpose(
        &self,
        _a_strict: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
        _dw: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_recurrent_step(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
        _g: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_chunkwise_forward(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
        _g: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
        _chunk_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_chunk_prep(
        &self,
        _g: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _kkt: &kiln_tensor::Tensor,
        _qkt: &kiln_tensor::Tensor,
        _ks_entry: &kiln_tensor::Tensor,
        _q_s: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        Ok(None)
    }

    fn runtime_gdn_chunk_scan(
        &self,
        _a_strict: &kiln_tensor::Tensor,
        _b_mask: &kiln_tensor::Tensor,
        _v_prime: &kiln_tensor::Tensor,
        _q_s_scaled: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
        _decay_last_col: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        Ok(None)
    }

    fn runtime_gdn_full_chunk_forward(
        &self,
        _g: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _kkt: &kiln_tensor::Tensor,
        _qkt: &kiln_tensor::Tensor,
        _ks_entry: &kiln_tensor::Tensor,
        _q_s: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
        _k_t: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_full_chunk_forward_head_last_into(
        &self,
        _g: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _kkt: &kiln_tensor::Tensor,
        _qkt: &kiln_tensor::Tensor,
        _ks_entry: &kiln_tensor::Tensor,
        _q_s: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
        _k_t: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
        _out: &kiln_tensor::Tensor,
        _t_start: usize,
        _seq_len: usize,
    ) -> Result<bool> {
        Ok(false)
    }

    fn runtime_gdn_recurrent_prefill_head_last(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
        _g: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_recurrent_prefill_native_head_last(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
        _g: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_recurrent_qk_norm_prefill_native_head_last(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
        _g: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
        _q_scale: f64,
        _qk_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_decode_gates_recurrent(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _a_log: &kiln_tensor::Tensor,
        _dt_bias: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
        _z: &kiln_tensor::Tensor,
        _weight: &kiln_tensor::Tensor,
        _eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_decode_qk_norm_gates_recurrent(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _a_log: &kiln_tensor::Tensor,
        _dt_bias: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
        _q_scale: f64,
        _qk_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_decode_qk_norm_gates_recurrent_rmsnorm(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _a_log: &kiln_tensor::Tensor,
        _dt_bias: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
        _z: &kiln_tensor::Tensor,
        _weight: &kiln_tensor::Tensor,
        _q_scale: f64,
        _qk_eps: f64,
        _rms_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_decode_gates_recurrent_rmsnorm(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _a_log: &kiln_tensor::Tensor,
        _dt_bias: &kiln_tensor::Tensor,
        _state: &mut kiln_tensor::Tensor,
        _z: &kiln_tensor::Tensor,
        _weight: &kiln_tensor::Tensor,
        _eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_gdn_in_proj_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _in_proj_qkv_t: &kiln_tensor::Tensor,
        _in_proj_z_t: &kiln_tensor::Tensor,
        _in_proj_a_t: &kiln_tensor::Tensor,
        _in_proj_b_t: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        Ok(None)
    }

    fn runtime_gdn_ab_in_proj_prefill(
        &self,
        _x: &kiln_tensor::Tensor,
        _in_proj_ab_t: &kiln_tensor::Tensor,
        _nv: usize,
        _seq_len: usize,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        Ok(None)
    }

    fn runtime_gdn_gates(
        &self,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _a_log: &kiln_tensor::Tensor,
        _dt_bias: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        Ok(None)
    }

    fn runtime_gdn_gated_rms_norm(
        &self,
        _x: &kiln_tensor::Tensor,
        _z: &kiln_tensor::Tensor,
        _weight: &kiln_tensor::Tensor,
        _eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }
}

/// Focused `ConvBackend` facet for causal convolution kernels.
#[allow(clippy::too_many_arguments)]
pub trait ConvBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_causal_conv1d_update(&self) -> bool {
        false
    }

    fn runtime_supports_causal_conv1d_prefill(&self) -> bool {
        false
    }

    fn runtime_causal_conv1d_update(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight: &kiln_tensor::Tensor,
        _conv_state: &mut kiln_tensor::Tensor,
        _kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_causal_conv1d_prefill(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight: &kiln_tensor::Tensor,
        _conv_state: &mut kiln_tensor::Tensor,
        _kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackendMatmulLayout {
    Plain,
    LhsTransposed,
    RhsTransposed,
    BothTransposed,
}

pub(crate) fn requested_matmul_layout(
    req: &capability::MatmulRequest,
    lhs: &kiln_tensor::Tensor,
    rhs: &kiln_tensor::Tensor,
) -> Option<BackendMatmulLayout> {
    if req.to_blas_request(1).is_err()
        || req.epilogue != capability::MatmulEpilogue::Identity
        || req.out_layout != capability::MatmulOperandLayout::RowMajor
        || req.lhs_shape.as_slice() != lhs.dims()
        || req.rhs_shape.as_slice() != rhs.dims()
        || req.lhs_dtype != lhs.dtype()
        || req.rhs_dtype != rhs.dtype()
    {
        return None;
    }

    match (req.lhs_layout, req.rhs_layout) {
        (capability::MatmulOperandLayout::RowMajor, capability::MatmulOperandLayout::RowMajor) => {
            Some(BackendMatmulLayout::Plain)
        }
        (capability::MatmulOperandLayout::ColMajor, capability::MatmulOperandLayout::RowMajor) => {
            Some(BackendMatmulLayout::LhsTransposed)
        }
        (capability::MatmulOperandLayout::RowMajor, capability::MatmulOperandLayout::ColMajor) => {
            Some(BackendMatmulLayout::RhsTransposed)
        }
        (capability::MatmulOperandLayout::ColMajor, capability::MatmulOperandLayout::ColMajor) => {
            Some(BackendMatmulLayout::BothTransposed)
        }
    }
}

pub(crate) fn matmul_request_support_rank(req: &capability::MatmulRequest) -> Option<usize> {
    let row_major_request = req.lhs_layout == capability::MatmulOperandLayout::RowMajor
        && req.rhs_layout == capability::MatmulOperandLayout::RowMajor
        && req.out_layout == capability::MatmulOperandLayout::RowMajor;
    if req.to_blas_request(1).is_err()
        || req.homogeneous_dtype().is_none()
        || req.accumulation != capability::MatmulAccumulation::F32
        || req.out_layout != capability::MatmulOperandLayout::RowMajor
        || match req.epilogue {
            capability::MatmulEpilogue::Identity => false,
            capability::MatmulEpilogue::Bias => !row_major_request,
            _ => true,
        }
    {
        return None;
    }
    req.rank()
}

pub(crate) fn matmul_support_from_native(native: bool) -> capability::Support {
    if native {
        capability::Support::NativeWithConstraints
    } else {
        capability::Support::HostFallbackAllowed
    }
}

/// Focused `LinearBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait LinearBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_matmul_request(
        &self,
        _req: &capability::MatmulRequest,
    ) -> capability::Support {
        capability::Support::Unsupported
    }

    fn runtime_matmul(
        &self,
        _req: &capability::MatmulRequest,
        _lhs: &kiln_tensor::Tensor,
        _rhs: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_linear_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_linear_prefill_apply(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_linear_prefill_apply_offset(
        &self,
        _x: &kiln_tensor::Tensor,
        _full_weight_t: &kiln_tensor::Tensor,
        _chunk_start: usize,
        _chunk_len: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_lora_delta_resident(
        &self,
        _x: &kiln_tensor::Tensor,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_lora_decode_add(
        &self,
        _base: &kiln_tensor::Tensor,
        _x: &kiln_tensor::Tensor,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_prewarm_decode_weights(&self, _weights: &crate::forward::GpuWeights) -> Result<()> {
        Ok(())
    }

    fn runtime_full_attn_qkv_combined_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _qkv_weight_t: Option<&kiln_tensor::Tensor>,
        _qkv_w8: Option<&crate::rocm_w8_proj::RocmW8Proj>,
        _q_dim: usize,
        _k_dim: usize,
        _v_dim: usize,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        Ok(None)
    }

    fn runtime_full_attn_qkv_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _q_weight_t: &kiln_tensor::Tensor,
        _k_weight_t: &kiln_tensor::Tensor,
        _v_weight_t: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        Ok(None)
    }

    fn runtime_mlp_gate_up_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _gate_weight_t: &kiln_tensor::Tensor,
        _up_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn runtime_mlp_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _gate_weight_t: &kiln_tensor::Tensor,
        _up_weight_t: &kiln_tensor::Tensor,
        _down_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }
}

/// Focused `SamplingBackend` facet for fused LM-head token selection.
#[allow(clippy::too_many_arguments)]
pub trait SamplingBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_linear_decode_argmax(&self) -> bool {
        false
    }

    fn runtime_linear_decode_argmax(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<u32>> {
        Ok(None)
    }

    fn runtime_supports_linear_decode_argmax_batch(&self) -> bool {
        false
    }

    fn runtime_linear_decode_argmax_batch(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    fn runtime_supports_linear_decode_sample(&self, _top_k: u32) -> bool {
        false
    }

    fn runtime_linear_decode_sample(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight_t: &kiln_tensor::Tensor,
        _history_indices: &[u32],
        _history_counts: &[u32],
        _repetition_penalty: f32,
        _presence_penalty: f32,
        _frequency_penalty: f32,
        _temperature: f32,
        _top_k: u32,
        _top_p: f32,
        _min_p: f32,
        _seed: u64,
    ) -> Result<Option<u32>> {
        Ok(None)
    }

    fn runtime_supports_linear_decode_sample_batch(
        &self,
        _top_k: &[u32],
        _temperatures: &[f32],
    ) -> bool {
        false
    }

    fn runtime_linear_decode_sample_batch(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight_t: &kiln_tensor::Tensor,
        _history_rows: &[u32],
        _history_indices: &[u32],
        _history_counts: &[u32],
        _repetition_penalties: &[f32],
        _presence_penalties: &[f32],
        _frequency_penalties: &[f32],
        _temperatures: &[f32],
        _top_k: &[u32],
        _top_p: &[f32],
        _min_p: &[f32],
        _seeds: &[u64],
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }
}

/// Focused `ResidencyBackend` facade over authoritative resident registries.
#[allow(clippy::too_many_arguments)]
pub trait ResidencyBackend:
    BackendIdentity + residency::ResidentRegistry + Send + Sync + std::fmt::Debug
{
    fn runtime_supports_resident_activation(&self) -> bool {
        false
    }

    fn runtime_register_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        residency::ResidentRegistry::register_resource(
            self,
            tensor,
            residency::ResidentResourceFamily::Activation,
        )
        .map(|_| ())
    }

    fn runtime_evict_resident_activation(&self, tensor: &kiln_tensor::Tensor) {
        residency::ResidentRegistry::evict_resource(
            self,
            tensor,
            residency::ResidentResourceFamily::Activation,
        );
    }

    fn runtime_update_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        residency::ResidentRegistry::update_resource(
            self,
            tensor,
            residency::ResidentResourceFamily::Activation,
        )
        .map(|_| ())
    }

    fn runtime_has_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> bool {
        residency::ResidentRegistry::has_resident_resource(
            self,
            tensor,
            residency::ResidentResourceFamily::Activation,
        )
    }

    fn runtime_resident_activation_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
    ) -> Option<residency::ResidentResource> {
        residency::ResidentRegistry::resident_resource(
            self,
            tensor,
            residency::ResidentResourceFamily::Activation,
        )
    }

    fn runtime_resolve_resident_activation(
        &self,
        tensor: &kiln_tensor::Tensor,
        shape: &[usize],
        dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        residency::ResidentRegistry::resolve_resource(
            self,
            tensor,
            residency::ResidentResourceFamily::Activation,
            shape,
            dtype,
        )
    }

    fn runtime_enter_gdn_recurrent_resident_state_scope(&self) -> bool {
        false
    }

    fn runtime_exit_gdn_recurrent_resident_state_scope(&self) {}

    fn runtime_materialize_gdn_recurrent_resident_state(
        &self,
        _state: &mut kiln_tensor::Tensor,
    ) -> Result<()> {
        Ok(())
    }

    fn runtime_evict_gdn_recurrent_resident_state(&self, _state: &kiln_tensor::Tensor) {}

    fn runtime_has_gdn_recurrent_resident_state(&self, _state: &kiln_tensor::Tensor) -> bool {
        false
    }

    fn runtime_assemble_gdn_recurrent_resident_batch_rows(
        &self,
        _rows: &[&kiln_tensor::Tensor],
        _batch: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        Ok(false)
    }

    fn runtime_scatter_gdn_recurrent_resident_batch_rows(
        &self,
        _batch: &kiln_tensor::Tensor,
        _destinations: &mut [&mut kiln_tensor::Tensor],
    ) -> Result<bool> {
        Ok(false)
    }

    fn runtime_assemble_linear_attn_gdn_state_batch_kt(
        &self,
        _row_keys: &[kiln_tensor::TensorId],
        _batch_key: kiln_tensor::TensorId,
    ) -> Result<bool> {
        Ok(false)
    }

    fn runtime_scatter_linear_attn_gdn_state_batch_kt(
        &self,
        _batch_key: kiln_tensor::TensorId,
        _row_keys: &[kiln_tensor::TensorId],
    ) -> Result<bool> {
        Ok(false)
    }

    fn runtime_seed_linear_attn_gdn_state_kt(
        &self,
        _recurrent: &kiln_tensor::Tensor,
        _conv: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        Ok(false)
    }

    fn runtime_has_linear_attn_gdn_state_kt(&self, _key: kiln_tensor::TensorId) -> bool {
        false
    }
}

/// Focused `OptimizerBackend` facet for on-device optimizer updates.
#[allow(clippy::too_many_arguments)]
pub trait OptimizerBackend: Send + Sync + std::fmt::Debug {
    fn runtime_dispatch_sgd_step(
        &self,
        _param: &kiln_tensor::Tensor,
        _grad: &kiln_tensor::Tensor,
        _lr: f32,
    ) -> Result<bool> {
        Ok(false)
    }

    fn runtime_dispatch_adamw_step(
        &self,
        _param: &kiln_tensor::Tensor,
        _grad: &kiln_tensor::Tensor,
        _first_moment: &kiln_tensor::Tensor,
        _second_moment: &kiln_tensor::Tensor,
        _lr: f32,
        _beta1: f32,
        _beta2: f32,
        _eps: f32,
        _weight_decay: f32,
        _step: u32,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Fused on-device Muon step (momentum-orthogonalized SGD).
    ///
    /// Updates `param` and the per-param heavy-ball `momentum` buffer in
    /// place: `momentum = momentum_coef * momentum + grad`; the
    /// (Nesterov) look-ahead is orthogonalized via Newton-Schulz for
    /// rank-2 weights (the LoRA A/B matrices) with the RMS-matching
    /// `sqrt(max(rows, cols))` scale, then `param = param * (1 - lr *
    /// weight_decay) - lr * update`. Non-matrix params and ranks beyond
    /// the kernel's shared-memory bound fall back to plain momentum SGD
    /// inside the kernel. Returns `Ok(true)` when handled on-device,
    /// `Ok(false)` to defer to the host `kiln_optim::Muon` reference.
    #[allow(clippy::too_many_arguments)]
    fn runtime_dispatch_muon_step(
        &self,
        _param: &kiln_tensor::Tensor,
        _grad: &kiln_tensor::Tensor,
        _momentum: &kiln_tensor::Tensor,
        _lr: f32,
        _momentum_coef: f32,
        _nesterov: bool,
        _ns_iters: u32,
        _weight_decay: f32,
    ) -> Result<bool> {
        Ok(false)
    }
}

/// Focused `TrainingLossBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait TrainingLossBackend: Send + Sync + std::fmt::Debug {
    fn runtime_training_capabilities(&self) -> TrainingCapabilities {
        TrainingCapabilities::portable()
    }

    fn runtime_training_precision_policy(&self) -> TrainingPrecisionPolicy {
        TrainingPrecisionPolicy::portable()
    }

    fn runtime_sft_flce_loss_route(&self) -> SftFlceLossRoute {
        self.runtime_training_capabilities().sft_flce_loss_route
    }

    fn runtime_tape_forward_backward_route(&self) -> TrainingTapeRoute {
        self.runtime_training_capabilities()
            .tape_forward_backward_route
    }

    fn runtime_grpo_loss_route(&self) -> GrpoLossRoute {
        self.runtime_training_capabilities().grpo_loss_route
    }

    fn runtime_grpo_kl_auxiliary_route(&self) -> GrpoKlAuxiliaryRoute {
        self.runtime_training_capabilities().grpo_kl_auxiliary_route
    }

    fn runtime_opd_loss_route(&self) -> OpdLossRoute {
        self.runtime_training_capabilities().opd_loss_route
    }

    fn runtime_opd_phase_b_backward_route(&self) -> OpdPhaseBBackwardRoute {
        self.runtime_training_capabilities()
            .opd_phase_b_backward_route
    }

    fn runtime_final_rmsnorm_backward_route(&self) -> FinalRmsNormBackwardRoute {
        self.runtime_training_capabilities()
            .final_rmsnorm_backward_route
    }
}

/// Focused `ReplayBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait ReplayBackend:
    BackendIdentity + AttentionBackend + Send + Sync + std::fmt::Debug
{
    fn runtime_decode_resident_pool_ready(
        &self,
        _max_hidden: usize,
        _max_intermediate: usize,
        _max_batch: usize,
    ) -> bool {
        false
    }

    fn runtime_supports_resident_decode(&self) -> bool {
        false
    }

    fn runtime_supports_replay_request(
        &self,
        req: &capability::ReplayRequest,
    ) -> capability::Support {
        if !req.replay_safe || !req.has_valid_bounds() {
            return capability::Support::Unsupported;
        }

        capability::Support::from_supports_predicate(match req.kind {
            capability::ReplayRequestKind::ResidentDecode => {
                self.runtime_supports_resident_decode()
            }
            capability::ReplayRequestKind::PagedDecodeGraphOutputs => {
                AttentionBackend::runtime_supports_flash_attn_paged_decode(self)
            }
        })
    }

    fn runtime_replay_key_for_request(
        &self,
        req: &capability::ReplayRequest,
    ) -> kiln_graph::ReplayKey {
        req.replay_key(BackendIdentity::runtime_device(self).backend())
    }

    fn runtime_replay_authority(&self) -> capability::ReplayAuthority {
        capability::ReplayAuthority::for_backend(
            BackendIdentity::runtime_name(self),
            BackendIdentity::runtime_device(self),
        )
    }

    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        _graph_outputs: Option<(&kiln_tensor::Tensor, &kiln_tensor::Tensor)>,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        AttentionBackend::runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
            self,
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            max_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
    }
}

// (#1082 candle removal) The candle-typed `for_device` shim was deleted along
// with the candle-parity opt-in feature that gated it — the last candle
// activator in the workspace. Production dispatch goes through the kt-native
// `for_device_kt` below; every backend caller (CUDA / Metal / Vulkan) is
// candle-free.

/// Pick the right backend for a given kt device (#1082 — the production
/// dispatcher; candle-free on every build).
///
/// On Metal devices, `--features metal` uses Kiln's native Metal kernels
/// (`MetalBackend::new(kt_device)`, no candle round-trip). The former MLX bridge
/// was removed because it only accelerated attention while paying host-copy
/// overheads and bypassing Kiln's Qwen3.5 GDN decode kernels.
///
/// `kt::Device::Vulkan(_)` drives the Vulkan backend, which manages its own
/// `vk::Device` internally; CUDA devices dispatch the CUDA fused-op path.
///
/// `kt::Device::Cuda(_)` and `kt::Device::Metal(_)` arms only compile when the
/// matching cargo feature is enabled; without it the `match` falls through to
/// the CPU / Vulkan-runtime-detect arm. The dispatch is candle-free — each arm
/// constructs its backend straight from the kt `Device` (#1082 candle removal).
///
/// This is always-on (no cuda feature gate): multi-backend builds (Metal,
/// Vulkan, CPU) of kiln-server call this entry without `--features cuda`.
pub fn for_device_kt(device: &kiln_tensor::Device) -> Arc<dyn BackendRuntime> {
    // (#1082 DoD-100 step 4) kt-native dispatcher. On a pure-CUDA build this
    // references NO candle types: the Cuda arm constructs `CudaBackend` from
    // the kt device directly, and the `_` arm builds a kt-typed `CpuBackend`.
    // The metal/vulkan arms are cfg-gated (out of the cuda build) and still
    // bridge to candle for those backends' candle-typed constructors.
    match device {
        #[cfg(feature = "cuda")]
        kiln_tensor::Device::Cuda(_) => Arc::new(cuda::CudaBackend::new(*device)),
        #[cfg(feature = "rocm")]
        kiln_tensor::Device::Rocm(_) => Arc::new(rocm::RocmBackend::new(*device)),
        #[cfg(feature = "metal")]
        kiln_tensor::Device::Metal(_) => {
            // #1082: MetalBackend is kt-native — pass the kt device directly.
            Arc::new(metal::MetalBackend::new(*device))
        }
        _ => {
            // Vulkan is detected at runtime (kt has a Vulkan variant but the
            // VulkanBackend manages its own vk::Device, so it takes a kt
            // Cpu sentinel). CPU/unmapped fall through to the kt CpuBackend.
            #[cfg(feature = "vulkan")]
            {
                if vulkan::vulkan_is_available() {
                    mark_vulkan_active();
                    return Arc::new(vulkan::VulkanBackend::new(kiln_tensor::Device::Cpu));
                }
            }
            Arc::new(cpu::CpuBackend::new(kiln_tensor::Device::Cpu))
        }
    }
}

/// Construct exactly the backend named by an explicit runtime binding.
///
/// Unlike [`for_device_kt`], CPU is never treated as Vulkan's historical host
/// tensor sentinel. This is the selector for fallible, explicitly initialized
/// inference runtimes; compatibility owners may continue using runtime
/// autodetection through `for_device_kt`.
pub fn for_explicit_device_kt(device: kiln_tensor::Device) -> Result<Arc<dyn BackendRuntime>> {
    match device {
        kiln_tensor::Device::Cpu => Ok(Arc::new(cpu::CpuBackend::new(device))),
        kiln_tensor::Device::Cuda(_) => {
            #[cfg(feature = "cuda")]
            {
                return Ok(Arc::new(cuda::CudaBackend::new(device)));
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("explicit CUDA runtime requested from a build without the cuda feature");
        }
        kiln_tensor::Device::Rocm(_) => {
            #[cfg(feature = "rocm")]
            {
                return Ok(Arc::new(rocm::RocmBackend::new(device)));
            }
            #[cfg(not(feature = "rocm"))]
            anyhow::bail!("explicit ROCm runtime requested from a build without the rocm feature");
        }
        kiln_tensor::Device::Metal(_) => {
            #[cfg(feature = "metal")]
            {
                return Ok(Arc::new(metal::MetalBackend::new(device)));
            }
            #[cfg(not(feature = "metal"))]
            anyhow::bail!(
                "explicit Metal runtime requested from a build without the metal feature"
            );
        }
        kiln_tensor::Device::Vulkan(_) => {
            #[cfg(feature = "vulkan")]
            {
                anyhow::ensure!(
                    vulkan::vulkan_is_available(),
                    "explicit Vulkan runtime requested but no Vulkan device is available"
                );
                mark_vulkan_active();
                return Ok(Arc::new(vulkan::VulkanBackend::new(
                    kiln_tensor::Device::Cpu,
                )));
            }
            #[cfg(not(feature = "vulkan"))]
            anyhow::bail!(
                "explicit Vulkan runtime requested from a build without the vulkan feature"
            );
        }
        other => anyhow::bail!("explicit runtime requested for unsupported device: {other:?}"),
    }
}

/// Training precision policy selected through the concrete backend facet.
///
/// Vulkan's hybrid path may carry CPU-host tensors, but those tensors acquire
/// Vulkan policy only after an owning runtime explicitly constructs and marks
/// the Vulkan backend. Mere hardware availability must not reinterpret a CPU
/// training job as Vulkan.
pub fn training_precision_policy_for_device_kt(
    device: kiln_tensor::Device,
) -> TrainingPrecisionPolicy {
    match device {
        #[cfg(feature = "cuda")]
        kiln_tensor::Device::Cuda(_) => {
            let backend = cuda::CudaBackend::new(device);
            TrainingLossBackend::runtime_training_precision_policy(&backend)
        }
        #[cfg(feature = "rocm")]
        kiln_tensor::Device::Rocm(_) => {
            let backend = rocm::RocmBackend::new(device);
            TrainingLossBackend::runtime_training_precision_policy(&backend)
        }
        #[cfg(feature = "metal")]
        kiln_tensor::Device::Metal(_) => {
            let backend = metal::MetalBackend::new(device);
            TrainingLossBackend::runtime_training_precision_policy(&backend)
        }
        #[cfg(feature = "vulkan")]
        kiln_tensor::Device::Vulkan(_) => TrainingPrecisionPolicy::vulkan(),
        _ => {
            // CPU-host Vulkan tensors inherit policy only from an explicitly
            // selected Vulkan runtime, never from an availability probe.
            #[cfg(feature = "vulkan")]
            if matches!(device, kiln_tensor::Device::Cpu) && vulkan_active() {
                return TrainingPrecisionPolicy::vulkan();
            }
            let backend = cpu::CpuBackend::new(device);
            TrainingLossBackend::runtime_training_precision_policy(&backend)
        }
    }
}

/// Training tape route selected through the concrete backend facet.
///
/// Like [`training_precision_policy_for_device_kt`], this avoids the
/// `for_device_kt` CPU-as-Vulkan runtime-detect behavior so CPU tensors keep the
/// portable unsupported route while accelerator tensors use their advertised
/// `TrainingLossBackend` contract.
pub fn training_tape_route_for_device_kt(device: kiln_tensor::Device) -> TrainingTapeRoute {
    match device {
        #[cfg(feature = "cuda")]
        kiln_tensor::Device::Cuda(_) => {
            let backend = cuda::CudaBackend::new(device);
            TrainingLossBackend::runtime_tape_forward_backward_route(&backend)
        }
        #[cfg(feature = "rocm")]
        kiln_tensor::Device::Rocm(_) => {
            let backend = rocm::RocmBackend::new(device);
            TrainingLossBackend::runtime_tape_forward_backward_route(&backend)
        }
        #[cfg(feature = "metal")]
        kiln_tensor::Device::Metal(_) => {
            let backend = metal::MetalBackend::new(device);
            TrainingLossBackend::runtime_tape_forward_backward_route(&backend)
        }
        #[cfg(feature = "vulkan")]
        kiln_tensor::Device::Vulkan(_) => {
            vulkan::VulkanBackend::training_capabilities_static().tape_forward_backward_route
        }
        _ => {
            // CPU-host tensors use the Vulkan tape route only after the
            // owning runtime explicitly selected and marked Vulkan.
            #[cfg(feature = "vulkan")]
            if matches!(device, kiln_tensor::Device::Cpu) && vulkan_active() {
                return vulkan::VulkanBackend::training_capabilities_static()
                    .tape_forward_backward_route;
            }
            let backend = cpu::CpuBackend::new(device);
            TrainingLossBackend::runtime_tape_forward_backward_route(&backend)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_cpu_binding_never_auto_promotes_to_vulkan() {
        let backend = for_explicit_device_kt(kiln_tensor::Device::Cpu).unwrap();
        assert_eq!(BackendIdentity::runtime_name(backend.as_ref()), "cpu");
        assert_eq!(
            BackendIdentity::runtime_device(backend.as_ref()),
            kiln_tensor::Device::Cpu
        );
    }

    #[derive(Debug)]
    struct ResidentActivationProbeBackend {
        name: &'static str,
        resident: bool,
    }

    impl BackendIdentity for ResidentActivationProbeBackend {
        fn runtime_name(&self) -> &'static str {
            self.name
        }

        fn runtime_device(&self) -> kiln_tensor::Device {
            kiln_tensor::Device::Cpu
        }

        fn runtime_as_any(&self) -> &dyn std::any::Any {
            &()
        }
    }

    impl StartupBackend for ResidentActivationProbeBackend {}

    impl ExternalYieldBackend for ResidentActivationProbeBackend {
        fn runtime_synchronize_external_yield(&self) -> Result<()> {
            Ok(())
        }
    }

    impl AttentionBackend for ResidentActivationProbeBackend {}

    impl GdnBackend for ResidentActivationProbeBackend {}

    impl ConvBackend for ResidentActivationProbeBackend {}

    impl LinearBackend for ResidentActivationProbeBackend {
        fn runtime_supports_matmul_request(
            &self,
            req: &capability::MatmulRequest,
        ) -> capability::Support {
            let vulkan_resident_mixed_rank2 = self.name == "vulkan"
                && req.rank() == Some(2)
                && req.to_blas_request(1).is_ok()
                && matches!(req.logical_mnk(), Some((m, n, k)) if m > 0 && n > 0 && k > 0)
                && req.lhs_layout == capability::MatmulOperandLayout::RowMajor
                && req.rhs_layout == capability::MatmulOperandLayout::RowMajor
                && req.out_layout == capability::MatmulOperandLayout::RowMajor
                && req.epilogue == capability::MatmulEpilogue::Identity
                && req.lhs_dtype == kiln_tensor::DType::F32
                && req.rhs_dtype == kiln_tensor::DType::BF16
                && req.out_dtype == kiln_tensor::DType::F32;
            if vulkan_resident_mixed_rank2 {
                return capability::Support::NativeWithConstraints;
            }
            let Some(rank) = matmul_request_support_rank(req) else {
                return capability::Support::Unsupported;
            };
            let native = match self.name {
                "cpu" => true,
                "cuda" | "rocm" => match req.epilogue {
                    capability::MatmulEpilogue::Identity => true,
                    capability::MatmulEpilogue::Bias => rank == 2,
                    _ => false,
                },
                "metal" => {
                    req.lhs_dtype == kiln_tensor::DType::BF16
                        && matches!(req.epilogue, capability::MatmulEpilogue::Identity)
                }
                "vulkan" => {
                    matches!(req.epilogue, capability::MatmulEpilogue::Identity)
                        && (req.lhs_dtype == kiln_tensor::DType::F32
                            || req.lhs_dtype == kiln_tensor::DType::BF16 && rank > 2)
                }
                _ => false,
            };
            matmul_support_from_native(native)
        }
    }

    impl residency::ResidentRegistry for ResidentActivationProbeBackend {
        fn register_resource(
            &self,
            tensor: &kiln_tensor::Tensor,
            family: residency::ResidentResourceFamily,
        ) -> Result<Option<residency::ResidentResource>> {
            Ok(self.resident_resource(tensor, family))
        }

        fn update_resource(
            &self,
            tensor: &kiln_tensor::Tensor,
            family: residency::ResidentResourceFamily,
        ) -> Result<Option<residency::ResidentResource>> {
            Ok(self
                .resident_resource(tensor, family)
                .map(|resource| resource.with_state(residency::ResidentResourceState::DirtyDevice)))
        }

        fn evict_resource(
            &self,
            _tensor: &kiln_tensor::Tensor,
            _family: residency::ResidentResourceFamily,
        ) {
        }

        fn resident_resource(
            &self,
            tensor: &kiln_tensor::Tensor,
            family: residency::ResidentResourceFamily,
        ) -> Option<residency::ResidentResource> {
            if family != residency::ResidentResourceFamily::Activation || !self.resident {
                return None;
            }
            Some(residency::ResidentResource::from_tensor_for_backend(
                tensor,
                residency::resident_backend_for_runtime(self.runtime_name(), tensor.device()),
                residency::ResidentResourceFamily::Activation,
                residency::resident_ownership_for_backend(self.runtime_name()),
            ))
        }
    }

    impl ResidencyBackend for ResidentActivationProbeBackend {}

    impl SamplingBackend for ResidentActivationProbeBackend {}

    impl OptimizerBackend for ResidentActivationProbeBackend {}

    impl PagedKvBackend for ResidentActivationProbeBackend {}

    impl ReplayBackend for ResidentActivationProbeBackend {}

    impl TrainingLossBackend for ResidentActivationProbeBackend {}

    impl BackendRuntime for ResidentActivationProbeBackend {}

    #[test]
    fn cpu_external_yield_boundary_is_synchronous_noop() {
        let backend = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);
        ExternalYieldBackend::runtime_synchronize_external_yield(&backend)
            .expect("CPU external-yield synchronization should succeed");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_external_yield_boundary_reaches_the_real_context() {
        if kiln_tensor::primary_cuda_context(0).is_err() {
            return;
        }
        let backend = cuda::CudaBackend::new(kiln_tensor::Device::Cuda(0));
        ExternalYieldBackend::runtime_synchronize_external_yield(&backend)
            .expect("CUDA external-yield synchronization should drain the device context");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_external_yield_boundary_reaches_the_real_device() {
        if kiln_tensor::primary_rocm_context(0).is_err() {
            return;
        }
        let backend = rocm::RocmBackend::new(kiln_tensor::Device::Rocm(0));
        ExternalYieldBackend::runtime_synchronize_external_yield(&backend)
            .expect("ROCm external-yield synchronization should drain the device");
    }

    #[test]
    fn portable_training_capabilities_are_conservative() {
        let caps = TrainingCapabilities::portable();
        assert_eq!(caps.resident_activation, "not implemented");
        assert_eq!(caps.native_training, "not implemented");
        assert!(caps.projection_training.contains("candle"));
        assert_eq!(
            caps.tape_forward_backward_route,
            TrainingTapeRoute::Unsupported
        );
        assert_eq!(caps.sft_flce_loss_route, SftFlceLossRoute::FullLogits);
        assert_eq!(caps.grpo_loss_route, GrpoLossRoute::KtComposite);
        assert_eq!(
            caps.grpo_kl_auxiliary_route,
            GrpoKlAuxiliaryRoute::HostComposite
        );
        assert_eq!(caps.opd_loss_route, OpdLossRoute::Unsupported);
        assert_eq!(
            caps.opd_phase_b_backward_route,
            OpdPhaseBBackwardRoute::Unsupported
        );
        assert_eq!(
            caps.final_rmsnorm_backward_route,
            FinalRmsNormBackwardRoute::KtComposite
        );

        let policy = TrainingPrecisionPolicy::portable();
        assert_eq!(policy.name, "cpu_f32_reference");
        assert_eq!(policy.activation_dtypes, &[kiln_tensor::DType::F32]);
        assert_eq!(policy.base_weight_dtypes, &[kiln_tensor::DType::F32]);
        assert!(!policy.mixed_precision);
    }

    #[test]
    fn training_precision_policies_capture_backend_differences() {
        let cuda = TrainingPrecisionPolicy::cuda();
        assert!(cuda.activation_dtypes.contains(&kiln_tensor::DType::BF16));
        assert!(cuda.activation_dtypes.contains(&kiln_tensor::DType::F16));
        assert_eq!(cuda.exact_gdn_backward_tile_tokens, Some(1024));
        assert_eq!(cuda.streaming_prefill_tile_tokens, 1024);
        assert_eq!(cuda.detached_full_attn_tile_tokens, 8192);
        assert_eq!(
            cuda.detached_full_attn_boundary_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_FLASH_DEFAULT_TILE
        );
        assert_eq!(
            cuda.detached_full_attn_tape_replay_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_FLASH_DEFAULT_TILE
        );
        assert_eq!(cuda.tape_streaming_tile_tokens, 1024);
        assert!(cuda.mixed_precision);

        let rocm = TrainingPrecisionPolicy::rocm();
        assert_eq!(rocm.streaming_prefill_tile_tokens, 1024);
        assert_eq!(
            rocm.detached_full_attn_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
        );
        assert_eq!(
            rocm.detached_full_attn_boundary_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
        );
        assert_eq!(
            rocm.detached_full_attn_tape_replay_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
        );
        assert_eq!(rocm.tape_streaming_tile_tokens, 1024);
        assert_eq!(rocm.paged_prefill_medium_tile_tokens, Some(1024));
        assert_eq!(rocm.paged_prefill_medium_tile_max_tokens, Some(20_000));
        assert_eq!(rocm.streaming_prefill_tile_tokens_for_seq_len(20_000), 1024);

        let metal = TrainingPrecisionPolicy::metal();
        assert_eq!(metal.activation_dtypes, &[kiln_tensor::DType::BF16]);
        assert_eq!(metal.loss_accumulation_dtype, kiln_tensor::DType::F32);
        assert_eq!(metal.streaming_prefill_tile_tokens, 2048);
        assert_eq!(
            metal.detached_full_attn_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
        );
        assert_eq!(
            metal.detached_full_attn_boundary_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
        );
        assert_eq!(
            metal.detached_full_attn_tape_replay_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
        );
        assert_eq!(metal.tape_streaming_tile_tokens, 2048);
        assert!(metal.notes.contains("UMA"));

        let vulkan = TrainingPrecisionPolicy::vulkan();
        assert_eq!(vulkan.activation_dtypes, &[kiln_tensor::DType::F32]);
        assert_eq!(vulkan.lora_parameter_dtypes, &[kiln_tensor::DType::F32]);
        assert_eq!(
            vulkan.mixed_rms_norm_weight_dtype,
            Some(kiln_tensor::DType::BF16)
        );
        assert_eq!(vulkan.exact_gdn_backward_tile_tokens, None);
        assert_eq!(vulkan.streaming_prefill_tile_tokens, 2048);
        assert_eq!(
            vulkan.detached_full_attn_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
        );
        assert_eq!(
            vulkan.detached_full_attn_boundary_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
        );
        assert_eq!(
            vulkan.detached_full_attn_tape_replay_tile_tokens,
            crate::forward::DETACHED_FULL_ATTN_MATERIALIZED_DEFAULT_TILE
        );
        assert_eq!(vulkan.tape_streaming_tile_tokens, 2048);
        assert!(
            vulkan
                .base_weight_dtypes
                .contains(&kiln_tensor::DType::BF16)
        );
        assert!(vulkan.mixed_precision);
    }

    #[test]
    fn training_precision_policy_maps_device_family() {
        for (device, expected) in [
            (kiln_tensor::Device::Cpu, "cpu_f32_reference"),
            (kiln_tensor::Device::Cuda(0), "cuda_native_float"),
            (kiln_tensor::Device::Rocm(0), "rocm_native_float"),
            (kiln_tensor::Device::Metal(0), "metal_bf16_uma"),
            (kiln_tensor::Device::Vulkan(0), "vulkan_mixed_f32_bf16"),
        ] {
            assert_eq!(
                TrainingPrecisionPolicy::for_device_family(device).name,
                expected
            );
        }
    }

    #[test]
    fn training_precision_policy_selects_lora_parameter_dtype() {
        let cuda = TrainingPrecisionPolicy::cuda();
        assert_eq!(
            cuda.lora_parameter_dtype_for_base_weight(kiln_tensor::DType::BF16),
            kiln_tensor::DType::BF16
        );
        assert_eq!(
            cuda.lora_parameter_dtype_for_base_weight(kiln_tensor::DType::F16),
            kiln_tensor::DType::F16
        );

        let vulkan = TrainingPrecisionPolicy::vulkan();
        assert_eq!(
            vulkan.lora_parameter_dtype_for_base_weight(kiln_tensor::DType::BF16),
            kiln_tensor::DType::F32
        );
        assert!(vulkan.uses_f32_activations_for_mixed_base_weights());
        assert!(vulkan.supports_rms_norm_weight_dtype_for_activation(
            kiln_tensor::DType::F32,
            kiln_tensor::DType::BF16
        ));
        assert!(vulkan.supports_mixed_base_weight_dtype_for_activation(
            kiln_tensor::DType::F32,
            kiln_tensor::DType::BF16
        ));
        assert_eq!(
            vulkan.activation_dtype_for_embedding_output(kiln_tensor::DType::BF16),
            kiln_tensor::DType::F32
        );
        assert!(
            !TrainingPrecisionPolicy::cuda().supports_rms_norm_weight_dtype_for_activation(
                kiln_tensor::DType::F32,
                kiln_tensor::DType::BF16
            )
        );
        assert!(
            !TrainingPrecisionPolicy::cuda().supports_mixed_base_weight_dtype_for_activation(
                kiln_tensor::DType::F32,
                kiln_tensor::DType::BF16
            )
        );
        assert_eq!(
            TrainingPrecisionPolicy::cuda()
                .activation_dtype_for_embedding_output(kiln_tensor::DType::BF16),
            kiln_tensor::DType::BF16
        );
        assert!(
            TrainingPrecisionPolicy::metal().supports_rms_norm_weight_dtype_for_activation(
                kiln_tensor::DType::BF16,
                kiln_tensor::DType::BF16
            )
        );
        assert!(
            !TrainingPrecisionPolicy::metal().supports_mixed_base_weight_dtype_for_activation(
                kiln_tensor::DType::BF16,
                kiln_tensor::DType::BF16
            )
        );
        assert_eq!(
            TrainingPrecisionPolicy::metal()
                .activation_dtype_for_embedding_output(kiln_tensor::DType::BF16),
            kiln_tensor::DType::BF16
        );
        assert!(!TrainingPrecisionPolicy::portable().uses_f32_activations_for_mixed_base_weights());
        assert!(!TrainingPrecisionPolicy::metal().uses_f32_activations_for_mixed_base_weights());
    }

    #[test]
    fn sft_flce_loss_route_strings_are_report_stable() {
        assert_eq!(SftFlceLossRoute::FullLogits.as_str(), "full_logits");
        assert_eq!(SftFlceLossRoute::KtTapeFlce.as_str(), "kt_tape_flce");
        assert_eq!(
            SftFlceLossRoute::VulkanActiveRows.as_str(),
            "vulkan_active_rows"
        );
        assert_eq!(GrpoLossRoute::KtComposite.as_str(), "kt_composite");
        assert_eq!(
            GrpoLossRoute::VulkanActiveRows.as_str(),
            "vulkan_active_rows"
        );
        assert_eq!(
            GrpoKlAuxiliaryRoute::HostComposite.as_str(),
            "host_composite"
        );
        assert_eq!(
            GrpoKlAuxiliaryRoute::CudaRocmDeviceFastPath.as_str(),
            "cuda_rocm_device_fast_path"
        );
        assert_eq!(OpdLossRoute::Unsupported.as_str(), "unsupported");
        assert_eq!(OpdLossRoute::KtTapePhaseB.as_str(), "kt_tape_phase_b");
        assert_eq!(
            OpdLossRoute::VulkanActiveHidden.as_str(),
            "vulkan_active_hidden"
        );
        assert_eq!(OpdPhaseBBackwardRoute::Unsupported.as_str(), "unsupported");
        assert_eq!(OpdPhaseBBackwardRoute::KtComposite.as_str(), "kt_composite");
        assert_eq!(
            OpdPhaseBBackwardRoute::CudaRocmFusedUnitGrad.as_str(),
            "cuda_rocm_fused_unit_grad"
        );
        assert_eq!(
            OpdPhaseBBackwardRoute::VulkanActiveHidden.as_str(),
            "vulkan_active_hidden"
        );
        assert_eq!(
            FinalRmsNormBackwardRoute::KtComposite.as_str(),
            "kt_composite"
        );
        assert_eq!(
            FinalRmsNormBackwardRoute::CudaRocmFusedTail.as_str(),
            "cuda_rocm_fused_tail"
        );
        assert_eq!(TrainingTapeRoute::Unsupported.as_str(), "unsupported");
        assert_eq!(
            TrainingTapeRoute::KtTapeAuthoritative.as_str(),
            "kt_tape_authoritative"
        );
    }

    #[test]
    fn capability_snapshot_maps_cpu_support_predicates() {
        let cpu = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);
        let caps = capability::BackendCapabilitySnapshot::from_backend(&cpu);

        assert_eq!(
            capability::BackendCapabilityQueries::capability_snapshot(&cpu),
            caps
        );
        assert_eq!(caps.backend, "cpu");
        assert_eq!(caps.device, kiln_tensor::Device::Cpu);
        assert_eq!(caps.training, TrainingCapabilities::portable());
        assert_eq!(caps.resident_decode, capability::Support::Declined);
        assert_eq!(caps.resident_activation, capability::Support::Declined);
        assert_eq!(caps.flash_attn_prefill, capability::Support::Declined);
        assert_eq!(caps.flash_attn_paged_decode, capability::Support::Declined);
        assert_eq!(caps.paged_kv_head_major_read, capability::Support::Declined);
        assert_eq!(caps.gdn_recurrent_step, capability::Support::Declined);
        assert_eq!(caps.causal_conv1d_update, capability::Support::Declined);
        assert_eq!(caps.linear_decode_argmax, capability::Support::Declined);

        let attention_req = capability::AttentionRequest::flash_prefill(
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            1,
            16,
            128,
            false,
        );
        assert_eq!(
            attention_req.shape_key(),
            vec![
                vec![1, 16, 128],
                vec![1, 16, 128],
                vec![1, 16, 128],
                vec![1, 16, 128],
            ]
        );
        assert_eq!(attention_req.layout, capability::AttentionLayout::Sdpa);
        assert_eq!(
            capability::BackendCapabilityQueries::supports_attention_request(&cpu, &attention_req),
            capability::Support::Declined
        );

        let linear_req = capability::LinearRequest::decode_argmax(
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::I64,
            1,
            false,
        )
        .with_shapes(vec![1, 4096], vec![32000, 4096], vec![1]);
        assert_eq!(
            linear_req.shape_key(),
            vec![vec![1, 4096], vec![32000, 4096], vec![1]]
        );
        assert_eq!(linear_req.layout, capability::LinearLayouts::ROW_MAJOR);
        assert_eq!(
            capability::BackendCapabilityQueries::supports_linear_request(&cpu, &linear_req),
            capability::Support::Declined
        );

        let replay_req = capability::ReplayRequest::resident_decode(8, 16, 2)
            .with_dtype(kiln_tensor::DType::BF16);
        assert_eq!(replay_req.shape_key(), vec![8, 16, 2]);
        assert_eq!(replay_req.layout, capability::ReplayLayout::StableResident);
        assert_eq!(
            capability::BackendCapabilityQueries::supports_replay_request(&cpu, &replay_req),
            capability::Support::Declined
        );
        assert_eq!(
            replay_req.replay_key(kiln_tensor::Backend::Cpu),
            kiln_graph::ReplayKey::new(
                kiln_tensor::Backend::Cpu,
                "resident_decode",
                vec![8, 16, 2],
                Some(kiln_tensor::DType::BF16),
                2,
                true,
            )
        );

        let unsafe_replay_req = replay_req.clone().with_replay_safe(false);
        assert_eq!(
            capability::BackendCapabilityQueries::supports_replay_request(&cpu, &unsafe_replay_req,),
            capability::Support::Unsupported
        );
    }

    #[test]
    fn backend_capabilities_aggregate_maps_cpu_contract() {
        let cpu = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);
        let caps = capability::BackendCapabilityQueries::backend_capabilities(&cpu);

        assert_eq!(caps.backend, "cpu");
        assert_eq!(caps.device, kiln_tensor::Device::Cpu);
        assert_eq!(caps.storage.backend, kiln_tensor::Backend::Cpu);
        assert_eq!(
            caps.storage.resident_activation,
            capability::Support::Declined
        );
        assert_eq!(
            caps.matmul.rank2_f32,
            capability::Support::NativeWithConstraints
        );
        assert_eq!(
            caps.matmul.batched_bf16,
            capability::Support::NativeWithConstraints
        );
        assert_eq!(caps.attention.flash_prefill, capability::Support::Declined);
        assert_eq!(caps.gdn.recurrent_step, capability::Support::Declined);
        assert_eq!(caps.decode.linear_argmax, capability::Support::Declined);
        assert_eq!(caps.decode_batcher.max_batch, 8);
        assert_eq!(caps.decode_batcher.wait_micros, 0);
        assert!(!caps.decode_batcher.allow_mixed_seq_lens);
        assert_eq!(caps.training.precision, TrainingPrecisionPolicy::portable());
        assert_eq!(
            caps.graph_replay.resident_decode,
            capability::Support::Declined
        );
        assert_eq!(
            caps.fallback.generic_device_op,
            FallbackPolicy::CorrectnessAllowed
        );
        assert_eq!(
            caps.fallback.decode_hot_path,
            FallbackPolicy::CorrectnessAllowed
        );
        assert_eq!(caps.fallback.decode_hot_path_debug_env, None);
        assert_eq!(
            caps.fallback.training_optimizer,
            FallbackPolicy::CorrectnessAllowed
        );
        assert_eq!(caps.fallback.training_optimizer_debug_env, None);

        let vulkan_probe = ResidentActivationProbeBackend {
            name: "vulkan",
            resident: false,
        };
        let vulkan_caps = capability::BackendCapabilityQueries::backend_capabilities(&vulkan_probe);
        assert_eq!(vulkan_caps.device, kiln_tensor::Device::Cpu);
        assert_eq!(vulkan_caps.storage.backend, kiln_tensor::Backend::Vulkan);
        assert_eq!(vulkan_caps.decode_batcher.max_batch, 64);
        assert_eq!(vulkan_caps.decode_batcher.wait_micros, 5_000);
        assert!(vulkan_caps.decode_batcher.allow_mixed_seq_lens);
        assert_eq!(
            vulkan_caps.fallback.decode_hot_path,
            FallbackPolicy::NativeRequired
        );
        assert_eq!(
            vulkan_caps.fallback.decode_hot_path_debug_env,
            Some("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK")
        );
        assert_eq!(
            vulkan_caps.fallback.training_optimizer,
            FallbackPolicy::NativeRequired
        );
        assert_eq!(
            vulkan_caps.fallback.training_optimizer_debug_env,
            Some("KILN_VULKAN_TRAINING_OPTIMIZER_FALLBACK")
        );
    }

    #[test]
    fn matmul_request_capability_is_conservative() {
        let cpu = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);
        let plain = capability::MatmulRequest::plain(
            vec![2, 3],
            vec![3, 4],
            kiln_tensor::DType::F32,
            false,
        );
        assert_eq!(plain.logical_mnk(), Some((2, 4, 3)));
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&cpu, &plain),
            capability::Support::NativeWithConstraints
        );

        let incompatible = capability::MatmulRequest::plain(
            vec![2, 3],
            vec![5, 4],
            kiln_tensor::DType::F32,
            false,
        );
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&cpu, &incompatible),
            capability::Support::Unsupported
        );
        let fused_relu = plain
            .clone()
            .with_epilogue(capability::MatmulEpilogue::Relu);
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&cpu, &fused_relu),
            capability::Support::Unsupported
        );

        let cuda = ResidentActivationProbeBackend {
            name: "cuda",
            resident: false,
        };
        let cuda_transposed = capability::MatmulRequest::plain(
            vec![2, 3],
            vec![4, 3],
            kiln_tensor::DType::BF16,
            false,
        )
        .with_layouts(
            capability::MatmulOperandLayout::RowMajor,
            capability::MatmulOperandLayout::ColMajor,
            capability::MatmulOperandLayout::RowMajor,
        );
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&cuda, &cuda_transposed),
            capability::Support::NativeWithConstraints
        );
        let transposed_bias = cuda_transposed
            .clone()
            .with_epilogue(capability::MatmulEpilogue::Bias);
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&cuda, &transposed_bias),
            capability::Support::Unsupported
        );
        let mixed = plain.clone().with_dtypes(
            kiln_tensor::DType::F32,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::F32,
        );
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&cuda, &mixed),
            capability::Support::Unsupported
        );

        let bias = capability::MatmulRequest::plain(
            vec![2, 3],
            vec![3, 4],
            kiln_tensor::DType::BF16,
            false,
        )
        .with_epilogue(capability::MatmulEpilogue::Bias);
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&cuda, &bias),
            capability::Support::NativeWithConstraints
        );

        let metal = ResidentActivationProbeBackend {
            name: "metal",
            resident: false,
        };
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&metal, &plain),
            capability::Support::HostFallbackAllowed
        );
        let metal_bf16 = capability::MatmulRequest::plain(
            vec![2, 3],
            vec![3, 4],
            kiln_tensor::DType::BF16,
            false,
        );
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&metal, &metal_bf16),
            capability::Support::NativeWithConstraints
        );

        let vulkan = ResidentActivationProbeBackend {
            name: "vulkan",
            resident: false,
        };
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&vulkan, &metal_bf16),
            capability::Support::HostFallbackAllowed
        );
        let vulkan_batched_bf16 = capability::MatmulRequest::plain(
            vec![2, 4, 8, 16],
            vec![2, 4, 16, 32],
            kiln_tensor::DType::BF16,
            true,
        );
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(
                &vulkan,
                &vulkan_batched_bf16
            ),
            capability::Support::NativeWithConstraints
        );

        let vulkan_mixed_rank2 = plain.clone().with_dtypes(
            kiln_tensor::DType::F32,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::F32,
        );
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(
                &vulkan,
                &vulkan_mixed_rank2
            ),
            capability::Support::NativeWithConstraints
        );
        let vulkan_mixed_batched = capability::MatmulRequest::plain(
            vec![2, 3, 4],
            vec![2, 4, 5],
            kiln_tensor::DType::F32,
            false,
        )
        .with_dtypes(
            kiln_tensor::DType::F32,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::F32,
        );
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(
                &vulkan,
                &vulkan_mixed_batched
            ),
            capability::Support::Unsupported
        );
    }

    #[test]
    fn matmul_request_projects_to_blas_shape_contract() {
        let bias = capability::MatmulRequest::plain(
            vec![2, 3],
            vec![3, 4],
            kiln_tensor::DType::BF16,
            true,
        )
        .with_epilogue(capability::MatmulEpilogue::Bias);

        let blas = bias.to_blas_request(3).expect("bias request projects");
        assert_eq!(blas.m, 2);
        assert_eq!(blas.n, 4);
        assert_eq!(blas.k, 3);
        assert_eq!(blas.dtype, kiln_tensor::DType::BF16);
        assert_eq!(blas.dtype_name(), "bf16");
        assert_eq!(blas.lhs_dtype_name(), "bf16");
        assert_eq!(blas.rhs_dtype_name(), "bf16");
        assert_eq!(blas.out_dtype_name(), "bf16");
        assert!(!blas.is_mixed_dtype());
        assert_eq!(blas.lhs_layout.blas_name(), "row");
        assert_eq!(blas.rhs_layout.blas_name(), "row");
        assert_eq!(blas.out_layout.blas_name(), "row");
        assert_eq!(blas.accumulation, capability::MatmulAccumulation::F32);
        assert_eq!(blas.epilogue.blas_name(), "bias");
        assert_eq!(blas.batch, capability::MatmulBatchPolicy::Single);
        assert!(blas.replay_safe);
        assert_eq!(blas.concurrent_streams, 3);

        let batched = capability::MatmulRequest::plain(
            vec![2, 4, 8, 16],
            vec![2, 4, 16, 32],
            kiln_tensor::DType::F32,
            true,
        );
        let blas_batched = batched
            .to_blas_request(1)
            .expect("batched request projects");
        assert_eq!(
            (blas_batched.m, blas_batched.n, blas_batched.k),
            (8, 32, 16)
        );
        assert_eq!(
            blas_batched.batch,
            capability::MatmulBatchPolicy::Batched { batches: 8 }
        );

        let transposed_rhs = capability::MatmulRequest::plain(
            vec![2, 3],
            vec![4, 3],
            kiln_tensor::DType::BF16,
            true,
        )
        .with_layouts(
            capability::MatmulOperandLayout::RowMajor,
            capability::MatmulOperandLayout::ColMajor,
            capability::MatmulOperandLayout::RowMajor,
        );
        assert_eq!(transposed_rhs.logical_mnk(), Some((2, 4, 3)));
        let transposed_blas = transposed_rhs
            .to_blas_request(2)
            .expect("transposed rhs request projects losslessly");
        assert_eq!(
            (transposed_blas.m, transposed_blas.n, transposed_blas.k),
            (2, 4, 3)
        );
        assert_eq!(
            transposed_blas.rhs_layout,
            capability::MatmulOperandLayout::ColMajor
        );
        assert_eq!(transposed_blas.concurrent_streams, 2);

        let transposed_lhs = capability::MatmulRequest::plain(
            vec![3, 2],
            vec![3, 4],
            kiln_tensor::DType::BF16,
            false,
        )
        .with_layouts(
            capability::MatmulOperandLayout::ColMajor,
            capability::MatmulOperandLayout::RowMajor,
            capability::MatmulOperandLayout::RowMajor,
        );
        assert_eq!(transposed_lhs.logical_mnk(), Some((2, 4, 3)));
        assert_eq!(
            transposed_lhs
                .to_blas_request(1)
                .expect("transposed lhs request projects")
                .lhs_layout,
            capability::MatmulOperandLayout::ColMajor
        );

        let incompatible = capability::MatmulRequest::plain(
            vec![2, 3],
            vec![5, 4],
            kiln_tensor::DType::F32,
            false,
        );
        assert_eq!(
            incompatible.to_blas_request(1),
            Err(capability::MatmulRequestProjectionError::IncompatibleShape)
        );

        let mixed = capability::MatmulRequest {
            rhs_dtype: kiln_tensor::DType::F32,
            ..bias
        };
        assert!(mixed.has_mixed_dtypes());
        assert_eq!(mixed.homogeneous_dtype(), None);
        let mixed_blas = mixed
            .to_blas_request(1)
            .expect("mixed dtype request should project without dropping dtype metadata");
        assert!(mixed_blas.is_mixed_dtype());
        assert_eq!(mixed_blas.lhs_dtype, kiln_tensor::DType::BF16);
        assert_eq!(mixed_blas.rhs_dtype, kiln_tensor::DType::F32);
        assert_eq!(mixed_blas.out_dtype, kiln_tensor::DType::BF16);
        assert_eq!(
            mixed.to_blas_request(0),
            Err(capability::MatmulRequestProjectionError::InvalidConcurrentStreams)
        );
    }

    #[test]
    fn linear_backend_runtime_matmul_routes_cpu_request() -> Result<()> {
        let cpu = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);
        let lhs = kiln_tensor::Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])?;
        let rhs =
            kiln_tensor::Tensor::from_slice(&[7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2])?;
        let req = capability::MatmulRequest::plain(
            lhs.dims().to_vec(),
            rhs.dims().to_vec(),
            kiln_tensor::DType::F32,
            false,
        );
        let out = LinearBackend::runtime_matmul(&cpu, &req, &lhs, &rhs)?
            .expect("cpu backend should route plain matmul request");
        assert_eq!(out.dims(), &[2, 2]);
        assert_eq!(out.to_vec::<f32>()?, vec![58.0, 64.0, 139.0, 154.0]);

        let rhs_t =
            kiln_tensor::Tensor::from_slice(&[7.0_f32, 9.0, 11.0, 8.0, 10.0, 12.0], vec![2, 3])?;
        let transposed_req = capability::MatmulRequest::plain(
            lhs.dims().to_vec(),
            rhs_t.dims().to_vec(),
            kiln_tensor::DType::F32,
            false,
        )
        .with_layouts(
            capability::MatmulOperandLayout::RowMajor,
            capability::MatmulOperandLayout::ColMajor,
            capability::MatmulOperandLayout::RowMajor,
        );
        let transposed_out = LinearBackend::runtime_matmul(&cpu, &transposed_req, &lhs, &rhs_t)?
            .expect("cpu backend should route rhs-transposed matmul request");
        assert_eq!(
            transposed_out.to_vec::<f32>()?,
            vec![58.0, 64.0, 139.0, 154.0]
        );

        let batched_lhs = kiln_tensor::Tensor::from_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        )?;
        let batched_rhs = kiln_tensor::Tensor::from_slice(
            &[
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 3, 2],
        )?;
        let batched_req = capability::MatmulRequest::plain(
            batched_lhs.dims().to_vec(),
            batched_rhs.dims().to_vec(),
            kiln_tensor::DType::F32,
            false,
        );
        let batched_out =
            LinearBackend::runtime_matmul(&cpu, &batched_req, &batched_lhs, &batched_rhs)?
                .expect("batched CPU backend should route rank-3 matmul request");
        assert_eq!(batched_out.dims(), &[2, 2, 2]);
        assert_eq!(
            batched_out.flatten_all()?.to_vec1::<f32>()?,
            vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0]
        );

        let rhs_bf16 = rhs.to_dtype(kiln_tensor::DType::BF16)?;
        let mixed_req = req.clone().with_dtypes(
            kiln_tensor::DType::F32,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::F32,
        );
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(&cpu, &mixed_req),
            capability::Support::NativeWithConstraints,
            "CPU backend should advertise the mixed-dtype request it can route through the oracle"
        );
        let mixed_out = LinearBackend::runtime_matmul(&cpu, &mixed_req, &lhs, &rhs_bf16)?
            .expect("mixed dtype CPU request should route through the F32 oracle");
        assert_eq!(
            mixed_out.flatten_all()?.to_vec1::<f32>()?,
            vec![58.0, 64.0, 139.0, 154.0]
        );

        let mixed_decline = req.clone().with_dtypes(
            kiln_tensor::DType::F32,
            kiln_tensor::DType::F32,
            kiln_tensor::DType::BF16,
        );
        let bf16_out = LinearBackend::runtime_matmul(&cpu, &mixed_decline, &lhs, &rhs)?
            .expect("CPU runtime_matmul should honor requested output dtype");
        assert_eq!(
            bf16_out.dtype(),
            kiln_tensor::DType::BF16,
            "CPU runtime_matmul should cast to requested output dtype"
        );

        Ok(())
    }

    #[test]
    fn portable_backend_declines_resident_decode_by_default() {
        // The default trait implementation must return false so that
        // every non-Vulkan backend continues to route through the
        // unchanged `model_forward_paged_last_token*` path — the
        // contract pinned by gate (c) of docs/vk_resident_decode_plan.md.
        let cpu = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);
        assert!(
            !ReplayBackend::runtime_supports_resident_decode(&cpu),
            "CPU backend must decline resident decode so non-Vulkan call sites \
             continue to use the existing per-call kt-tensor path"
        );
    }

    #[test]
    fn resident_activation_resource_describes_reported_resident_tensor() -> Result<()> {
        let tensor = kiln_tensor::Tensor::from_slice(&[1.0_f32, 2.0], vec![2])?;
        let backend = ResidentActivationProbeBackend {
            name: "vulkan",
            resident: false,
        };

        assert!(
            ResidencyBackend::runtime_resident_activation_resource(&backend, &tensor).is_none()
        );

        let backend = ResidentActivationProbeBackend {
            name: "vulkan",
            resident: true,
        };
        let resource =
            ResidencyBackend::runtime_resident_activation_resource(&backend, &tensor).unwrap();

        assert_eq!(resource.tensor_id, tensor.id());
        assert_eq!(resource.backend, kiln_tensor::Backend::Vulkan);
        assert_eq!(resource.device, kiln_tensor::Device::Cpu);
        assert_eq!(
            resource.family,
            residency::ResidentResourceFamily::Activation
        );
        assert_eq!(
            resource.ownership,
            residency::ResidentOwnership::RegistryOwned
        );
        assert_eq!(
            resource.state,
            residency::ResidentResourceState::RegisteredClean
        );
        assert_eq!(resource.shape, vec![2]);
        assert_eq!(resource.layout.strides, vec![1]);
        assert_eq!(resource.layout.start_offset, 0);
        assert!(resource.layout.contiguous);
        assert_eq!(resource.byte_len, 8);
        assert_eq!(resource.addressable_byte_len, 8);
        assert_eq!(
            resource.to_replay_resource_ref().backend,
            kiln_tensor::Backend::Vulkan
        );
        Ok(())
    }

    #[test]
    fn resident_registry_adapter_routes_activation_family() -> Result<()> {
        let tensor = kiln_tensor::Tensor::from_slice(&[1.0_f32, 2.0], vec![2])?;
        let backend = ResidentActivationProbeBackend {
            name: "vulkan",
            resident: true,
        };

        let resource = residency::ResidentRegistry::resident_resource(
            &backend,
            &tensor,
            residency::ResidentResourceFamily::Activation,
        )
        .unwrap();
        assert_eq!(resource.tensor_id, tensor.id());
        assert_eq!(resource.backend, kiln_tensor::Backend::Vulkan);
        assert_eq!(
            resource.family,
            residency::ResidentResourceFamily::Activation
        );

        assert!(residency::ResidentRegistry::has_resident_resource(
            &backend,
            &tensor,
            residency::ResidentResourceFamily::Activation,
        ));
        assert!(
            residency::ResidentRegistry::register_resource(
                &backend,
                &tensor,
                residency::ResidentResourceFamily::Activation,
            )?
            .is_some()
        );
        assert!(
            residency::ResidentRegistry::update_resource(
                &backend,
                &tensor,
                residency::ResidentResourceFamily::Activation,
            )?
            .is_some()
        );

        assert!(
            residency::ResidentRegistry::resident_resource(
                &backend,
                &tensor,
                residency::ResidentResourceFamily::OptimizerParam,
            )
            .is_none()
        );
        assert!(
            residency::ResidentRegistry::register_resource(
                &backend,
                &tensor,
                residency::ResidentResourceFamily::PagedKv,
            )?
            .is_none()
        );
        residency::ResidentRegistry::evict_resource(
            &backend,
            &tensor,
            residency::ResidentResourceFamily::Activation,
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backend_declines_resident_decode() {
        // CUDA keeps activations resident through the kt CUDA tensor lifecycle
        // — the resident-decode plan does not apply. (#1082: kt-native, no
        // candle; gate on the kt context probe and build the kt device.)
        if kiln_tensor::primary_cuda_context(0).is_err() {
            return; // No CUDA at test time — skip.
        }
        let cuda = cuda::CudaBackend::new(kiln_tensor::Device::Cuda(0));
        assert!(
            !ReplayBackend::runtime_supports_resident_decode(&cuda),
            "CUDA backend must decline resident decode; gate (c) requires CUDA path unchanged"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_backend_device_accessor_returns_kt() {
        // #1082 Phase 7: BackendRuntime::device() returns kt::Device by
        // value. The CUDA backend must preserve the cuda(i) device identity.
        // (#1082: kt-native, no candle round-trip.)
        if kiln_tensor::primary_cuda_context(0).is_err() {
            return;
        }
        let cuda = cuda::CudaBackend::new(kiln_tensor::Device::Cuda(0));
        assert_eq!(cuda.device(), kiln_tensor::Device::Cuda(0));
    }

    #[test]
    fn cpu_backend_device_accessor_returns_kt() {
        // Mirrors the CUDA assertion above for the always-available CPU
        // path. Confirms the trait surface stayed kt-typed even on the
        // portable fallback. (#1082)
        let cpu = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);
        assert_eq!(cpu.device(), kiln_tensor::Device::Cpu);
    }

    fn assert_focused_facets_forward_advertised_capabilities<B>(backend: &B, label: &str)
    where
        B: BackendRuntime
            + BackendIdentity
            + AttentionBackend
            + PagedKvBackend
            + GdnBackend
            + ConvBackend
            + LinearBackend
            + SamplingBackend
            + ResidencyBackend
            + OptimizerBackend
            + TrainingLossBackend
            + ReplayBackend
            + ?Sized,
    {
        macro_rules! assert_forwards {
            ($focused:expr, $runtime:expr, $method:literal) => {
                assert_eq!(
                    $focused, $runtime,
                    "{} focused facet did not forward {}",
                    label, $method
                );
            };
        }

        assert_forwards!(
            BackendIdentity::runtime_name(backend),
            BackendRuntime::name(backend),
            "name"
        );
        assert_forwards!(
            BackendIdentity::runtime_device(backend),
            BackendRuntime::device(backend),
            "device"
        );
        assert_forwards!(
            BackendIdentity::runtime_as_any(backend).type_id(),
            BackendRuntime::as_any(backend).type_id(),
            "as_any"
        );

        let replay_req = capability::ReplayRequest::paged_decode_graph_outputs(8, 16, 2)
            .with_dtype(kiln_tensor::DType::BF16);
        assert_forwards!(
            ReplayBackend::runtime_supports_replay_request(backend, &replay_req),
            capability::BackendCapabilityQueries::supports_replay_request(backend, &replay_req),
            "supports_replay_request"
        );
        assert_forwards!(
            ReplayBackend::runtime_replay_key_for_request(backend, &replay_req),
            capability::BackendCapabilityQueries::replay_key_for_request(backend, &replay_req),
            "replay_key_for_request"
        );
        assert_forwards!(
            ReplayBackend::runtime_replay_authority(backend),
            capability::ReplayAuthority::for_backend(
                BackendRuntime::name(backend),
                BackendRuntime::device(backend),
            ),
            "replay_authority"
        );
    }

    #[test]
    fn focused_facets_forward_cpu_advertised_capabilities() -> Result<()> {
        let cpu = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);
        assert_focused_facets_forward_advertised_capabilities(&cpu, "cpu");

        let t = kiln_tensor::Tensor::zeros_cpu(vec![1], kiln_tensor::DType::F32);
        assert!(ResidencyBackend::runtime_resident_activation_resource(&cpu, &t).is_none());

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn focused_facets_forward_cuda_advertised_capabilities() {
        let cuda = cuda::CudaBackend::new(kiln_tensor::Device::Cuda(0));
        assert_focused_facets_forward_advertised_capabilities(&cuda, "cuda");
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn focused_facets_forward_rocm_advertised_capabilities() {
        let rocm = rocm::RocmBackend::new(kiln_tensor::Device::Rocm(0));
        assert_focused_facets_forward_advertised_capabilities(&rocm, "rocm");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn focused_facets_forward_metal_advertised_capabilities() {
        let metal = metal::MetalBackend::new(kiln_tensor::Device::Metal(0));
        assert_focused_facets_forward_advertised_capabilities(&metal, "metal");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn focused_facets_forward_vulkan_advertised_capabilities() {
        let vulkan = vulkan::VulkanBackend::new(kiln_tensor::Device::Cpu);
        assert_focused_facets_forward_advertised_capabilities(&vulkan, "vulkan");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_backend_publishes_decode_resident_pool() {
        // Gate (b) of the vk-resident-decode plan: when the Vulkan
        // backend is up, `decode_resident_pool(...)` returns Some on
        // hardware that fits 3-4 slots within 1% of the device-local
        // heap. The RTX 6000 Ada the test host uses has ~48 GiB of
        // VRAM; the 10 MiB Qwen3.5-4B ring fits trivially.
        if !vulkan::vulkan_is_available() {
            return;
        }
        let backend = vulkan::VulkanBackend::new(kiln_tensor::Device::Cpu);
        let pool = backend.decode_resident_pool(2560, 9216, 64);
        assert!(
            pool.is_some(),
            "decode_resident_pool must succeed on a discrete GPU with ample VRAM"
        );
        let pool = pool.unwrap();
        assert!(
            pool.num_slots() >= 3,
            "pool must fit at least the 3-slot minimum, got {}",
            pool.num_slots()
        );
        // OnceLock contract: second call returns the same Arc.
        let again = backend.decode_resident_pool(2560, 9216, 64).unwrap();
        assert!(Arc::ptr_eq(pool, again));
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_backend_supports_resident_decode_when_device_up() {
        // When the host has a working Vulkan device, the Vulkan backend
        // must return true so call sites in `model_forward_paged_last_token*`
        // route through the resident path.
        if !vulkan::vulkan_is_available() {
            return;
        }
        let backend = vulkan::VulkanBackend::new(kiln_tensor::Device::Cpu);
        assert!(
            ReplayBackend::runtime_supports_resident_decode(&backend),
            "Vulkan backend must support resident decode by default when the \
             logical device is up"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_training_capabilities_do_not_overclaim_native_training() {
        let caps = cuda::CudaBackend::training_capabilities_static();
        assert!(caps.projection_training.contains("offset chunk hook"));
        assert!(
            caps.lora_delta_training
                .contains("declines tape-tracked tensors")
        );
        assert_eq!(
            caps.resident_activation,
            "kt TensorId lifecycle registry; kt CUDA tensors are canonical"
        );
        assert!(caps.sgd_step.contains("CUDA in-place optimizer kernel"));
        assert!(caps.adamw_step.contains("CUDA in-place optimizer kernel"));
        assert_eq!(caps.native_training, "not implemented");
        assert_eq!(
            caps.tape_forward_backward_route,
            TrainingTapeRoute::KtTapeAuthoritative
        );
        assert_eq!(caps.sft_flce_loss_route, SftFlceLossRoute::KtTapeFlce);
        assert_eq!(caps.grpo_loss_route, GrpoLossRoute::KtComposite);
        assert_eq!(
            caps.grpo_kl_auxiliary_route,
            GrpoKlAuxiliaryRoute::CudaRocmDeviceFastPath
        );
        assert_eq!(caps.opd_loss_route, OpdLossRoute::KtTapePhaseB);
        assert_eq!(
            caps.opd_phase_b_backward_route,
            OpdPhaseBBackwardRoute::CudaRocmFusedUnitGrad
        );
        assert_eq!(
            caps.final_rmsnorm_backward_route,
            FinalRmsNormBackwardRoute::CudaRocmFusedTail
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_training_capabilities_route_sft_flce_through_kt_tape() {
        let caps = rocm::RocmBackend::training_capabilities_static();
        assert_eq!(
            caps.tape_forward_backward_route,
            TrainingTapeRoute::KtTapeAuthoritative
        );
        assert_eq!(caps.sft_flce_loss_route, SftFlceLossRoute::KtTapeFlce);
        assert_eq!(caps.grpo_loss_route, GrpoLossRoute::KtComposite);
        assert_eq!(
            caps.grpo_kl_auxiliary_route,
            GrpoKlAuxiliaryRoute::CudaRocmDeviceFastPath
        );
        assert_eq!(caps.opd_loss_route, OpdLossRoute::KtTapePhaseB);
        assert_eq!(
            caps.opd_phase_b_backward_route,
            OpdPhaseBBackwardRoute::CudaRocmFusedUnitGrad
        );
        assert_eq!(
            caps.final_rmsnorm_backward_route,
            FinalRmsNormBackwardRoute::CudaRocmFusedTail
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_training_capabilities_advertise_adamw_but_decline_sgd() {
        let caps = metal::MetalBackend::training_capabilities_static();
        assert!(caps.sgd_step.contains("declined"));
        assert!(caps.adamw_step.contains("Metal in-place AdamW"));
        assert!(caps.resident_activation.contains("Metal TensorId"));
        assert_eq!(
            caps.tape_forward_backward_route,
            TrainingTapeRoute::KtTapeAuthoritative
        );
        assert_eq!(caps.sft_flce_loss_route, SftFlceLossRoute::FullLogits);
        assert_eq!(caps.grpo_loss_route, GrpoLossRoute::KtComposite);
        assert_eq!(
            caps.grpo_kl_auxiliary_route,
            GrpoKlAuxiliaryRoute::HostComposite
        );
        assert_eq!(caps.opd_loss_route, OpdLossRoute::KtTapePhaseB);
        assert_eq!(
            caps.opd_phase_b_backward_route,
            OpdPhaseBBackwardRoute::KtComposite
        );
        assert_eq!(
            caps.final_rmsnorm_backward_route,
            FinalRmsNormBackwardRoute::KtComposite
        );

        let backend = metal::MetalBackend::new(kiln_tensor::Device::Metal(0));
        assert_eq!(
            TrainingLossBackend::runtime_training_capabilities(&backend),
            caps
        );
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_training_capabilities_route_sft_flce_through_active_rows() {
        let caps = vulkan::VulkanBackend::training_capabilities_static();
        assert_eq!(
            caps.tape_forward_backward_route,
            TrainingTapeRoute::KtTapeAuthoritative
        );
        assert_eq!(caps.sft_flce_loss_route, SftFlceLossRoute::VulkanActiveRows);
        assert_eq!(caps.grpo_loss_route, GrpoLossRoute::VulkanActiveRows);
        assert_eq!(
            caps.grpo_kl_auxiliary_route,
            GrpoKlAuxiliaryRoute::HostComposite
        );
        assert_eq!(caps.opd_loss_route, OpdLossRoute::VulkanActiveHidden);
        assert_eq!(
            caps.opd_phase_b_backward_route,
            OpdPhaseBBackwardRoute::VulkanActiveHidden
        );
        assert_eq!(
            caps.final_rmsnorm_backward_route,
            FinalRmsNormBackwardRoute::KtComposite
        );
    }
}

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
pub(crate) mod vulkan_tensor_bridge;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_attention;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_training;

#[cfg(feature = "vulkan")]
pub(crate) mod vulkan_residency;

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
pub struct TrainingCapabilities {
    pub projection_training: &'static str,
    pub flce_loss: &'static str,
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
            mixed_precision: true,
            notes: "ROCm mirrors CUDA's kt-tape dtype envelope while dispatching through HIP/hipBLASLt-native leaves where available.",
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
            mixed_precision: true,
            notes: "Vulkan keeps training activations and LoRA parameters F32 while allowing BF16 base weights through explicit VkTensor buffer bridges.",
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

pub trait BackendRuntime: Send + Sync + std::fmt::Debug {
    /// Human-readable name (`"cuda"`, `"metal"`, `"cpu"`). Surfaced in
    /// `/health` and logs.
    fn name(&self) -> &'static str;

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
    fn device(&self) -> kiln_tensor::Device;

    /// `dyn Any` downcast target. Used by the Vulkan-resident decode
    /// fast-path in `transformer_block_paged_with_rope_tables` to recover
    /// the concrete `VulkanBackend` for direct access to its
    /// resident-decode primitives. The default impl returns a
    /// no-op-`Any`-shaped reference; the concrete backend overrides
    /// this to return `self`.
    fn as_any(&self) -> &dyn std::any::Any {
        // Default: return a Unit Any so `downcast_ref` against any
        // concrete type returns None. Concrete backends override.
        &()
    }

    /// Operator-facing summary of which training paths are backend-native,
    /// candle-on-device, or intentionally declined. This is telemetry only:
    /// dispatch methods remain the source of truth for actual behavior.
    fn training_capabilities(&self) -> TrainingCapabilities {
        TrainingCapabilities::portable()
    }

    fn training_precision_policy(&self) -> TrainingPrecisionPolicy {
        TrainingPrecisionPolicy::portable()
    }

    /// First-use feasibility check for the Vulkan-resident decode pool:
    /// returns true when a 3-4 slot ring sized to `max(hidden, intermediate)
    /// × max_batch × 4` bytes can be allocated within 1 % of the device-local
    /// heap. CPU / CUDA / Metal return false by default; the Vulkan backend
    /// constructs (and caches) the pool here.
    ///
    /// Gate (b) of docs/vk_resident_decode_plan.md. The cached `None`
    /// outcome means subsequent decode steps see the same `false` answer
    /// without re-probing the device.
    #[allow(unused_variables)]
    fn decode_resident_pool_ready(
        &self,
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> bool {
        false
    }

    /// Whether the backend can run a full decode step "device-resident":
    /// pay one host→device upload (input token / hidden) and one device→host
    /// readback (sampled token / last-row logits) per decode step, instead
    /// of `N kernel calls × (extract + upload + readback)` per layer.
    ///
    /// CUDA and Metal already keep activations resident through candle and
    /// MPS respectively, so for those backends the resident-decode plan is
    /// a no-op and they keep returning false here. The Vulkan backend
    /// returns true when feature-gated `KILN_VULKAN_RESIDENT_DECODE`
    /// is on (default on when the kernel ring fits in the device's memory
    /// budget) AND the device has actually been brought up.
    ///
    /// Callers in `model_forward_paged_last_token*` use this predicate to
    /// route into the resident decode path. Returning false routes to
    /// today's candle_core::Tensor-shaped path unchanged.
    fn supports_resident_decode(&self) -> bool {
        false
    }

    fn supports_flash_attn_prefill(&self) -> bool {
        false
    }

    fn supports_flash_attn_prefill_head_major(&self) -> bool {
        false
    }

    fn supports_flash_attn_paged_decode(&self) -> bool {
        false
    }

    /// FlashAttention-style decode for the common single-sequence case where
    /// the live KV slots are already one contiguous run in the paged cache.
    ///
    /// `q`: `[1, num_heads, 1, head_dim]`; `k_pool`/`v_pool`:
    /// `[total_slots, num_kv_heads, head_dim]`. Returns `[1, 1,
    /// num_heads * head_dim]`.
    fn flash_attn_paged_decode_contiguous(
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

    /// Batched variant of [`Self::flash_attn_paged_decode_contiguous`] for a
    /// group of decode rows whose live KV windows are each one contiguous run
    /// in the paged cache and share a common sequence length.
    ///
    /// `q`: `[batch, num_heads, 1, head_dim]`; `start_slots`: `[batch]` u32.
    /// Returns `[batch, 1, num_heads * head_dim]`.
    fn flash_attn_paged_decode_contiguous_batch(
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

    /// Whether the backend implements the strict (uniform-`start_pos`)
    /// `flash_attn_paged_decode_contiguous_batch` kernel.
    ///
    /// Callers can use this to short-circuit before allocating the
    /// `start_slots` device tensor (the `candle_core::Tensor::from_slice` build in
    /// `gqa_attention_paged_decode_contiguous_batch::try_strict`) on
    /// backends that will decline the trait call anyway. Important
    /// under CUDA graph capture: the `candle_core::Tensor::from_slice` would emit a
    /// captured `cudaMemcpyHtoDAsync` that targets storage which is
    /// freed at end of capture, leaving a recycled-VA write hazard on
    /// every replay even though no kernel reads from it (suspect 6 in
    /// `bench-results/cuda-graph-bs2-secondary-audit.md`, #1082).
    ///
    /// Default `true` preserves the historical
    /// "try-strict-then-fall-through-on-`None`" behavior so backends
    /// that already opt-in via `flash_attn_paged_decode_contiguous_batch`
    /// keep working unchanged. CUDA overrides to `false` because the
    /// strict kernel has no CUDA impl today and the captured HtoD
    /// scratch is a clean-up wart.
    fn supports_strict_paged_decode_contiguous_batch(&self) -> bool {
        true
    }

    /// Varlen variant of [`Self::flash_attn_paged_decode_contiguous_batch`] for
    /// a group of decode rows with divergent K/V lengths under continuous
    /// batching. Uses block-table addressing so K/V need not be contiguous in
    /// the paged cache.
    ///
    /// `q`: `[batch, 1, num_heads, head_dim]` bf16; `block_table`:
    /// `[batch, max_blocks_per_seq]` u32; `seqused_k`: `[batch]` i32 holding
    /// per-row attention length. Returns `[batch, 1, num_heads, head_dim]`.
    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
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

    /// CUDA-graph-aware variant of
    /// [`Self::flash_attn_paged_decode_contiguous_batch_dyn_seqlen`] that
    /// accepts caller-owned `(out, softmax_lse)` device tensors so the
    /// captured graph reads/writes its paged-decode scratch from stable
    /// runner-owned storage instead of from transient `candle_core::Tensor::zeros`
    /// allocations inside the kernel wrapper.
    ///
    /// `graph_outputs = Some((out, softmax_lse))` skips the per-call
    /// `candle_core::Tensor::zeros` inside the captured region; the runner pre-allocates
    /// these tensors and re-uses them across replays (see
    /// `BatchedPagedDecodeGraphInputs::{attn_out, softmax_lse}`).
    /// `graph_outputs = None` matches the non-graph behavior of
    /// [`Self::flash_attn_paged_decode_contiguous_batch_dyn_seqlen`] and is
    /// the default for backends that don't support stable graph buffers.
    ///
    /// The default impl routes through
    /// [`Self::flash_attn_paged_decode_contiguous_batch_dyn_seqlen`] (i.e.
    /// ignores `graph_outputs`), so non-CUDA backends don't need to override.
    /// CUDA overrides this and threads `graph_outputs` into
    /// `kiln_flash_attn::flash_attn_paged_decode_dyn_seqlen`. Part of #1082
    /// — see `bench-results/cuda-graph-bs2-secondary-audit.md` suspects 3+4.
    #[allow(clippy::too_many_arguments)]
    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
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
        self.flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
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

    fn supports_paged_kv_head_major_read(&self) -> bool {
        false
    }

    fn supports_paged_kv_head_major_read_append_token_major(&self) -> bool {
        false
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        false
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        false
    }

    fn enter_gdn_recurrent_resident_state_scope(&self) -> bool {
        false
    }

    fn exit_gdn_recurrent_resident_state_scope(&self) {}

    fn materialize_gdn_recurrent_resident_state(&self, _state: &mut kiln_tensor::Tensor) -> Result<()> {
        Ok(())
    }

    fn evict_gdn_recurrent_resident_state(&self, _state: &kiln_tensor::Tensor) {}

    fn has_gdn_recurrent_resident_state(&self, _state: &kiln_tensor::Tensor) -> bool {
        false
    }

    /// True when the backend's resident activation registry is
    /// non-trivially implemented — i.e. `register_resident_activation`
    /// actually uploads the tensor and `has_resident_activation` will
    /// return true after registration. False for the default no-op
    /// implementations. Callers that want to opt OUT of the lifecycle
    /// hook calls entirely (to avoid the per-call overhead of
    /// `extract_tensor_bytes` + buffer alloc on Vulkan) can gate on
    /// this. The default impls are cheap enough that it's safe to
    /// always invoke them, so most callers should not bother.
    fn supports_resident_activation(&self) -> bool {
        false
    }

    /// Register a non-weight tensor (e.g. a checkpoint-segment activation
    /// boundary) as registry-resident on the device. The default
    /// implementation is a no-op — backends that don't have a resident
    /// activation registry can safely ignore the call.
    ///
    /// Phase 3.1 of the residency plan. Generalises the GDN-specific
    /// `materialize_gdn_recurrent_resident_state` hook above. Once
    /// Phase 3.2 lands, `checkpointed_forward_backward` calls this for
    /// each segment-output tensor so the recompute pass can read the
    /// boundary back from device memory instead of the candle CPU mirror.
    fn register_resident_activation(&self, _tensor: &kiln_tensor::Tensor) -> Result<()> {
        Ok(())
    }

    /// Evict a previously-registered activation from the residency
    /// registry. Caller invokes this when the autograd pass no longer
    /// needs the tensor (e.g. after a segment's backward completes).
    /// No-op default.
    fn evict_resident_activation(&self, _tensor: &kiln_tensor::Tensor) {}

    /// Re-upload the tensor's current bytes into its registry buffer
    /// (if registered). Caller invokes this when the kt master
    /// storage has been mutated outside of the registry — e.g. after
    /// the optimizer step writes a new value to a registered
    /// LoRA parameter. Without this, `lora_delta_resident` and friends
    /// would keep reading the original init bytes from the buffer.
    ///
    /// No-op default; backends without a registry have nothing to
    /// keep in sync.
    fn update_resident_activation(&self, _tensor: &kiln_tensor::Tensor) -> Result<()> {
        Ok(())
    }

    /// True when the given tensor has been registered as
    /// resident-on-device. Used by routing code to decide between the
    /// resident fast path and the legacy CPU-roundtrip path. False by
    /// default so callers without registry support continue to use the
    /// legacy path.
    fn has_resident_activation(&self, _tensor: &kiln_tensor::Tensor) -> bool {
        false
    }

    /// Metadata-only descriptor for an activation that the backend already
    /// reports as resident. This does not allocate or register anything; it
    /// lets Phase 3 callers reason about residency state through one shared
    /// contract while concrete registries continue to own the actual bytes.
    fn resident_activation_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
    ) -> Option<residency::ResidentResource> {
        if self.has_resident_activation(tensor) {
            Some(residency::ResidentResource::from_tensor_for_backend(
                tensor,
                residency::resident_backend_for_runtime(self.name(), tensor.device()),
                residency::ResidentResourceFamily::Activation,
                residency::resident_ownership_for_backend(self.name()),
            ))
        } else {
            None
        }
    }

    /// Read a previously-registered activation back from device into
    /// a fresh CPU `kiln_tensor::Tensor` with the given shape and dtype. Returns
    /// `Ok(None)` when the activation isn't resident — caller should
    /// then use whatever CPU-side storage they retained originally.
    ///
    /// Phase 3.2 of the residency plan: pairs with
    /// `register_resident_activation` to let `checkpointed_forward_backward`
    /// drop the CPU mirror after registering, then re-materialise
    /// only when the recompute pass actually needs the boundary.
    /// Today's no-op default returns `Ok(None)` so callers without
    /// registry support fall through to the legacy code path.
    fn resolve_resident_activation(
        &self,
        _tensor: &kiln_tensor::Tensor,
        _shape: &[usize],
        _dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// Phase 4.2 hook: in-place SGD update `param -= lr * grad`
    /// against device-resident parameter and gradient buffers.
    /// Returns true when the dispatch succeeded; false when the
    /// backend can't service the request and the caller should fall
    /// back to the candle CPU path (`var.set(var - lr * grad)`).
    ///
    /// Callers must register both `param` and `grad` as resident
    /// activations first (Phase 3.1 hooks). The default implementation
    /// is a no-op returning false; the Vulkan backend's impl will land
    /// alongside Phase 4.1's resident `TrainableLoraParams`.
    fn dispatch_sgd_step(&self, _param: &kiln_tensor::Tensor, _grad: &kiln_tensor::Tensor, _lr: f32) -> Result<bool> {
        Ok(false)
    }

    /// AdamW slot per the residency plan §4.2 ("AdamW slot for later
    /// — leave the kernel name and signature in place; do not
    /// implement the moving averages until requested").
    ///
    /// Inputs: param + grad + first-moment buffer + second-moment
    /// buffer. All four must be registry-resident with matching
    /// shape and dtype. Hyperparams: lr, beta1, beta2, eps,
    /// weight_decay, step (1-indexed). Returns true on dispatch
    /// success, false on decline.
    ///
    /// Default no-op so trait callers pick up the eventual Vulkan
    /// impl without code changes. Trainer doesn't call this yet.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_adamw_step(
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

    /// Phase 4.1 step 2 hook: compute the LoRA delta
    /// `(x @ A.T @ B.T) * scale` against registry-resident A and B.
    /// Returns `Ok(Some(delta))` with the delta in `x.dtype()` when
    /// the backend can service the request; `Ok(None)` when it
    /// can't (either backend doesn't support it, or A/B aren't
    /// resident, or shapes don't fit kernel constraints) and the
    /// caller should fall back to the candle CPU
    /// `compute_lora_delta` path.
    ///
    /// Reading A and B from the registry means the LoRA forward
    /// path no longer reads the kt master's CPU storage
    /// for data — only for shape metadata. Phase 4.2's
    /// `dispatch_sgd_step` can then write to the same registry
    /// buffers in place without a sync-back to host storage.
    fn lora_delta_resident(
        &self,
        _x: &kiln_tensor::Tensor,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn assemble_gdn_recurrent_resident_batch_rows(
        &self,
        _rows: &[&kiln_tensor::Tensor],
        _batch: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        Ok(false)
    }

    fn scatter_gdn_recurrent_resident_batch_rows(
        &self,
        _batch: &kiln_tensor::Tensor,
        _destinations: &mut [&mut kiln_tensor::Tensor],
    ) -> Result<bool> {
        Ok(false)
    }

    fn assemble_linear_attn_gdn_state_batch_kt(
        &self,
        _row_keys: &[kiln_tensor::TensorId],
        _batch_key: kiln_tensor::TensorId,
    ) -> Result<bool> {
        Ok(false)
    }

    fn scatter_linear_attn_gdn_state_batch_kt(
        &self,
        _batch_key: kiln_tensor::TensorId,
        _row_keys: &[kiln_tensor::TensorId],
    ) -> Result<bool> {
        Ok(false)
    }

    fn seed_linear_attn_gdn_state_kt(
        &self,
        _recurrent: &kiln_tensor::Tensor,
        _conv: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        Ok(false)
    }

    fn has_linear_attn_gdn_state_kt(&self, _key: kiln_tensor::TensorId) -> bool {
        false
    }

    fn supports_gdn_chunk_prep(&self) -> bool {
        false
    }

    fn supports_gdn_chunk_scan(&self) -> bool {
        false
    }

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        false
    }

    fn supports_gdn_full_chunk_forward_head_last(&self) -> bool {
        false
    }

    fn supports_gdn_recurrent_prefill_head_last(&self) -> bool {
        false
    }

    fn supports_gdn_recurrent_prefill_native_head_last(&self) -> bool {
        false
    }

    fn supports_gdn_recurrent_qk_norm_prefill_native_head_last(&self) -> bool {
        false
    }

    fn supports_gdn_decode_gates_recurrent_unexpanded_qk(&self) -> bool {
        false
    }

    fn supports_gdn_decode_qk_norm_gates_recurrent(&self) -> bool {
        false
    }

    /// FlashAttention-2 forward for prefill (no KV cache, seq_len > 1).
    ///
    /// `q`, `k`, `v`: `[batch, seq_len, num_heads, head_dim]` bf16 contiguous.
    /// Caller must GQA-expand K/V to match Q's head count. Returns
    /// `[batch, seq_len, num_heads, head_dim]` bf16.
    fn flash_attn_prefill(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// FlashAttention-2 forward for prefill with Q/K/V already in SDPA layout.
    ///
    /// `q`: `[batch, num_heads, seq_len, head_dim]` bf16 contiguous. `k` and
    /// `v`: `[batch, num_kv_heads, seq_len, head_dim]` bf16 contiguous.
    /// Backends may decline when they lack native GQA support. Returns
    /// `[batch, num_heads, seq_len, head_dim]` bf16.
    fn flash_attn_prefill_head_major(
        &self,
        _q: &kiln_tensor::Tensor,
        _k: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// FlashAttention-2 paged decode (single query token against paged K/V pool).
    ///
    /// `q`: `[batch, 1, num_heads, head_dim]` bf16. `k_pool`/`v_pool`:
    /// `[total_slots, num_kv_heads, head_dim]` bf16. `block_table`:
    /// `[batch, max_blocks_per_seq]` u32. Returns `[batch, 1, num_heads, head_dim]`.
    ///
    /// Returning `Ok(None)` is valid for backends that can't satisfy the
    /// call's preconditions (e.g. non-contiguous blocks, unsupported page
    /// size); callers fall back to `paged_cache.read + naive softmax`.
    #[allow(clippy::too_many_arguments)]
    fn flash_attn_paged_decode(
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

    /// Materialize a contiguous head-major K/V view from a contiguous paged
    /// cache slot run.
    ///
    /// `k_pool`/`v_pool`: `[total_slots, num_kv_heads, head_dim]`.
    /// Returns `[1, num_kv_heads, seq_len, head_dim]` tensors suitable for
    /// head-major SDPA.
    fn paged_kv_head_major_read(
        &self,
        _k_pool: &kiln_tensor::Tensor,
        _v_pool: &kiln_tensor::Tensor,
        _start_slot: usize,
        _seq_len: usize,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        Ok(None)
    }

    /// Materialize a contiguous head-major K/V view from a contiguous paged
    /// cache slot run, then append a contiguous token-major tail directly into
    /// the same output buffer.
    ///
    /// `k_pool`/`v_pool`: `[total_slots, num_kv_heads, head_dim]`.
    /// `k_tail`/`v_tail`: `[1, tail_len, num_kv_heads, head_dim]`.
    /// Returns `[1, num_kv_heads, prefix_len + tail_len, head_dim]` tensors.
    fn paged_kv_head_major_read_append_token_major(
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

    /// Gated DeltaNet chunkwise forward-substitution (prefill path).
    /// Computes `W = (I + A_strict)^{-1} (beta * V_prime)`.
    ///
    /// `a_strict`: `[B, H, C, C]` bf16 (strictly lower-triangular).
    /// `v_prime`: `[B, H, C, dv]` bf16. `beta`: `[B, H, C]` bf16.
    /// Returns `W: [B, H, C, dv]` bf16.
    ///
    /// Backend kernels may advertise narrower envelopes; callers enforce the
    /// shared `C <= 128` cap and implementations can return `None` for shapes
    /// they do not handle.
    fn gdn_forward_substitution(
        &self,
        _a_strict: &kiln_tensor::Tensor,
        _v_prime: &kiln_tensor::Tensor,
        _beta: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// Gated DeltaNet single-token recurrent step (decode fast path).
    ///
    /// `q`, `k`: `[B, H, dk]` bf16. `v`: `[B, H, dv]` bf16.
    /// `beta`, `g`: `[B, H]` bf16. `state`: `[B, H, dk, dv]` bf16,
    /// mutated in place. Returns `out: [B, H, dv]` bf16.
    fn gdn_recurrent_step(
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

    /// Gated DeltaNet PREFILL chunkwise forward (GPU-parallel, forward-only).
    ///
    /// `q`,`k`: `[B, nv, T, dk]`. `v`: `[B, nv, T, dv]`. `beta`,`g`: `[B, nv, T]`.
    /// `state`: `[B, nv, dk, dv]`, replaced in place with the post-scan state.
    /// Returns `out: [B, nv, T, dv]` — the SAME contract + layout as the
    /// portable `gdn_chunkwise_recurrence` CPU reference, but it runs the
    /// per-chunk matmuls (KKT, forward-sub, state update) on the GPU in parallel
    /// instead of as raw kt matmuls (which execute on CPU-host tensors on
    /// Vulkan). This is the proper Vulkan prefill path; the CPU chunkwise is the
    /// fallback. (#1082) Returns `Ok(None)` to decline (caller uses the CPU path).
    fn gdn_chunkwise_forward(
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

    /// Fused GDN chunk-prep kernel (prefill outer recurrence).
    ///
    /// Collapses the 7+ candle op launches (cumsum, decay matrix, exp, masked
    /// scales, v_prime, q_s_scaled, decay_last_col, p_last) inside the
    /// chunkwise recurrence's inner loop into a single CUDA launch per
    /// (chunk × batch × head). Matmuls (KKT, QKT, ks_entry, q_s) stay on
    /// cuBLAS — this kernel consumes their outputs.
    ///
    /// `g`: `[B, H, C]` bf16. `v`: `[B, H, C, dv]` bf16.
    /// `kkt`, `qkt`: `[B, H, C, C]` bf16. `ks_entry`, `q_s`: `[B, H, C, dv]` bf16.
    ///
    /// Returns `(a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last)`:
    ///   - `a_strict`:       `[B, H, C, C]` bf16 — `kkt * decay * strict_lower`
    ///   - `b_mask`:         `[B, H, C, C]` bf16 — `qkt * decay * causal_lower`
    ///   - `v_prime`:        `[B, H, C, dv]` bf16 — `v - ks_entry * p`
    ///   - `q_s_scaled`:     `[B, H, C, dv]` bf16 — `q_s * p`
    ///   - `decay_last_col`: `[B, H, C]` bf16 — `exp(big_g[C-1] - big_g[i])`
    ///   - `p_last`:         `[B, H]` bf16 — `exp(big_g[C-1])`
    ///
    /// Returning `Ok(None)` is valid for backends that can't satisfy the
    /// envelope; callers fall back to the candle-op path.
    fn gdn_chunk_prep(
        &self,
        _g: &kiln_tensor::Tensor,
        _v: &kiln_tensor::Tensor,
        _kkt: &kiln_tensor::Tensor,
        _qkt: &kiln_tensor::Tensor,
        _ks_entry: &kiln_tensor::Tensor,
        _q_s: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        Ok(None)
    }

    fn gdn_chunk_scan(
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

    fn gdn_full_chunk_forward(
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

    #[allow(clippy::too_many_arguments)]
    fn gdn_full_chunk_forward_head_last_into(
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

    fn gdn_recurrent_prefill_head_last(
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

    fn gdn_recurrent_prefill_native_head_last(
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

    #[allow(clippy::too_many_arguments)]
    fn gdn_recurrent_qk_norm_prefill_native_head_last(
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

    /// Fused native-MTP GDN decode gates + recurrent update.
    ///
    /// Narrow CUDA/Metal decode path for `seq_len == 1` bf16 tensors. Returns
    /// `[B, 1, value_heads, dv]` before gated RMSNorm, mutating `state` in
    /// place. `Ok(None)` means the backend declines and the caller should use
    /// the split gates/recurrent/gated_norm path.
    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_gates_recurrent(
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

    /// Fused native-MTP GDN decode Q/K L2-normalization + gates + recurrent
    /// update.
    ///
    /// Narrow CUDA decode path for `seq_len == 1` bf16 tensors. It accepts raw
    /// unexpanded Q/K heads, applies the same bf16 qk_norm epilogue as the split
    /// path, returns `[B, 1, value_heads, dv]` before gated RMSNorm, and mutates
    /// `state` in place.
    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_qk_norm_gates_recurrent(
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

    /// Fused native-MTP GDN decode Q/K L2-normalization + gates + recurrent
    /// update + gated RMSNorm.
    ///
    /// Narrow CUDA decode path for `seq_len == 1` bf16/F32 tensors. It returns
    /// `[B, 1, value_heads, dv]` after gated RMSNorm and mutates `state` in
    /// place.
    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_qk_norm_gates_recurrent_rmsnorm(
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

    /// Fused native-MTP GDN decode gates + recurrent update + gated RMSNorm.
    ///
    /// Narrow decode path for `seq_len == 1`. Returns `[B, 1, value_heads, dv]`
    /// after gated RMSNorm, mutating `state` in place.
    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_gates_recurrent_rmsnorm(
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

    /// Fused GDN input projections.
    ///
    /// Collapses the four `broadcast_matmul` calls in Step 1 (`qkv`, `z`,
    /// `a`, `b`) into one backend launch when the backend supports the shape.
    /// Returns `(mixed_qkv, z, a, b)` with shapes matching the portable matmul
    /// path.
    #[allow(clippy::too_many_arguments)]
    fn gdn_in_proj_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _in_proj_qkv_t: &kiln_tensor::Tensor,
        _in_proj_z_t: &kiln_tensor::Tensor,
        _in_proj_a_t: &kiln_tensor::Tensor,
        _in_proj_b_t: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        Ok(None)
    }

    /// Transposed linear projection.
    ///
    /// `x` is `[batch, seq_len, hidden]`, `weight_t` is `[hidden, out_dim]`,
    /// and the output shape is `[batch, seq_len, out_dim]`. Backends should
    /// return `Ok(None)` for unsupported shapes, dtypes, LoRA paths, or debug
    /// modes.
    fn linear_decode(&self, _x: &kiln_tensor::Tensor, _weight_t: &kiln_tensor::Tensor) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// Tape-recorded transposed linear projection for prefill / training.
    ///
    /// Same shapes as `linear_decode` but the result must be wired into the
    /// kt autograd tape (`kiln_autograd`) so `Tape::backward()` produces a
    /// real gradient. Backends route through the kt-tape-recording matmul.
    /// Backends without a tape-recording path return `Ok(None)` so the
    /// caller falls back to the portable kt matmul (which the tape records).
    fn linear_prefill_apply(&self, _x: &kiln_tensor::Tensor, _weight_t: &kiln_tensor::Tensor) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// Same as `linear_prefill_apply` but operates on a column slice of a
    /// larger weight tensor: dispatches the matmul against
    /// `full_weight_t[:, chunk_start .. chunk_start + chunk_len]`. Backends
    /// that can keep `full_weight_t` resident as a single buffer and
    /// dispatch per-chunk via offset addressing avoid the per-chunk
    /// re-upload that the naive `linear_prefill_apply(_, narrowed)` path
    /// would pay for every unique narrowed `TensorId`.
    ///
    /// Used by the FLCE chunked head loop. The result need not be
    /// autograd-tracked — FLCE owns its own analytic backward; the result
    /// is consumed inside the FLCE analytic-backward path.
    fn linear_prefill_apply_offset(
        &self,
        _x: &kiln_tensor::Tensor,
        _full_weight_t: &kiln_tensor::Tensor,
        _chunk_start: usize,
        _chunk_len: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn supports_linear_decode_argmax(&self) -> bool {
        false
    }

    /// Single-token transposed linear projection with argmax reduction.
    ///
    /// Used by greedy LM-head decode when logits do not need to be materialized
    /// on the host. `x` is `[1, 1, hidden]`, `weight_t` is `[hidden, out_dim]`.
    fn linear_decode_argmax(&self, _x: &kiln_tensor::Tensor, _weight_t: &kiln_tensor::Tensor) -> Result<Option<u32>> {
        Ok(None)
    }

    fn supports_linear_decode_argmax_batch(&self) -> bool {
        false
    }

    /// Whether the backend has a fused on-device stochastic-sampling
    /// pipeline (lm_head + token penalties + top-k + softmax + min-p +
    /// top-p + categorical sample). When `true`, the model runner
    /// routes non-greedy decode through [`Self::linear_decode_sample`]
    /// and reads back ONLY the 4-byte sampled token — no full-vocab
    /// readback. Backends without this fast path (CUDA / Metal /
    /// dummy) keep using candle's on-device sampling via the regular
    /// `linear_decode` → `sample_with_full_params` flow.
    fn supports_linear_decode_sample(&self, _top_k: u32) -> bool {
        false
    }

    /// Fused stochastic decode: takes the same `(x, weight_t)` inputs
    /// as `linear_decode_argmax` plus the full sampling-state vector
    /// (token history + every Qwen3.5 sampler knob), runs the whole
    /// pipeline on-device, and returns just the sampled token id.
    /// Returns `Ok(None)` when the backend declines the request (e.g.
    /// `top_k > kernel-supported max`) so the caller can fall back to
    /// the legacy host sampler.
    #[allow(clippy::too_many_arguments)]
    fn linear_decode_sample(
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

    fn supports_linear_decode_sample_batch(&self, _top_k: &[u32], _temperatures: &[f32]) -> bool {
        false
    }

    /// Batched fused stochastic decode. `x` is `[batch, 1, hidden]`,
    /// `weight_t` is `[hidden, out_dim]`, flattened history arrays contain
    /// unique token counts per row, and every per-row sampler vector has
    /// length `batch`.
    #[allow(clippy::too_many_arguments)]
    fn linear_decode_sample_batch(
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

    /// Batched single-token transposed linear projection with argmax reduction.
    ///
    /// Used by greedy native-batch LM-head decode when logits do not need to be
    /// materialized on the host. `x` is `[batch, 1, hidden]`, `weight_t` is
    /// `[hidden, out_dim]`, and the result contains one token id per batch row.
    fn linear_decode_argmax_batch(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Forward-only LoRA delta/add for decode.
    ///
    /// `base` is the already-computed base projection output, `x` is the
    /// projection input, and `a`/`b` are PEFT LoRA matrices. Backends must
    /// return `Ok(None)` for tape-tracked tensors; training needs the
    /// kt-tape-recorded differentiable path.
    fn lora_decode_add(
        &self,
        _base: &kiln_tensor::Tensor,
        _x: &kiln_tensor::Tensor,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// Warm backend-resident decode weights after model load.
    ///
    /// CPU/CUDA/Metal either keep model tensors resident through Candle or
    /// have their own upload path. Vulkan's current Candle-CPU integration
    /// maintains a side cache of immutable projection buffers, so it can move
    /// the first-token upload cost out of the measured decode path.
    fn prewarm_decode_weights(&self, _weights: &crate::forward::GpuWeights) -> Result<()> {
        Ok(())
    }

    /// Drop the candle CPU storage of pre-transposed weight caches
    /// (`*_proj_t`, `embed_tokens_t`) that have already been uploaded
    /// to the backend's persistent device cache during
    /// `prewarm_decode_weights`.
    ///
    /// On Vulkan/UMA APUs this is the biggest remaining residency
    /// win: the transposed-cache copies are ~6-7 GB across 32 layers
    /// of Qwen3.5-4B, and after upload they're functionally dead
    /// weight on the candle CPU side — the kernels read from the
    /// device-resident `VulkanBuffer` keyed by the cache. Replacing
    /// each tensor with a 1-element BF16 stub and re-keying the
    /// backend's TensorId→buffer cache to the stub's new TensorId
    /// preserves the kernel-lookup path while reclaiming the bytes.
    ///
    /// Default no-op; only Vulkan implements it today.
    /// Returns the number of tensors actually stubbed (for telemetry).
    fn drop_uploaded_bf16_weights(
        &self,
        _weights: &mut crate::forward::GpuWeights,
        _device: &kiln_tensor::Device,
    ) -> Result<usize> {
        Ok(0)
    }

    /// Fused single-token full-attention Q/K/V projections.
    ///
    /// `x` is `[1, 1, hidden]`; weights are pre-transposed as
    /// `[hidden, out_dim]`; returned tensors are `[1, 1, q_dim]`,
    /// `[1, 1, k_dim]`, and `[1, 1, v_dim]`.
    fn full_attn_qkv_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _q_weight_t: &kiln_tensor::Tensor,
        _k_weight_t: &kiln_tensor::Tensor,
        _v_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        Ok(None)
    }

    /// Fused single-token MLP gate/up projection.
    ///
    /// `x` is `[1, 1, hidden]`; both weights are `[hidden, intermediate]`.
    /// Returns `[1, 1, intermediate]` containing `silu(x @ gate_t) * (x @ up_t)`.
    fn mlp_gate_up_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _gate_weight_t: &kiln_tensor::Tensor,
        _up_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// Fused single-token MLP that keeps the SwiGLU hidden activation on backend device.
    ///
    /// `x` is `[1, 1, hidden]`; `gate_weight_t` and `up_weight_t` are
    /// `[hidden, intermediate]`; `down_weight_t` is `[intermediate, out_dim]`.
    fn mlp_decode(
        &self,
        _x: &kiln_tensor::Tensor,
        _gate_weight_t: &kiln_tensor::Tensor,
        _up_weight_t: &kiln_tensor::Tensor,
        _down_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    fn supports_gdn_gates(&self) -> bool {
        false
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        false
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        false
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
        false
    }

    /// Fused single-step causal depthwise conv1d + state update + silu.
    ///
    /// Replaces the candle `to_f32 -> cat(state, x) -> sum(window * weight) ->
    /// narrow/contiguous -> silu` chain inside `kiln/gdn/conv` with one CUDA
    /// launch per (batch, channel).
    ///
    /// `x`: `[B, C, 1]` bf16 contiguous. `weight`: `[C, 1, K]` bf16 contiguous
    /// (or `[C, K]` equivalently — width stride = 1). `conv_state`:
    /// `[B, C, K-1]` F32, mutated in place to drop oldest col and append
    /// newest `x`. `kernel_size`: must be 4 for the current CUDA
    /// specialisation.
    ///
    /// Returns `Ok(Some(out))` with `out: [B, C, 1]` F32 (silu-fused), or
    /// `Ok(None)` when the backend declines (wrong dtype, wrong K, envelope
    /// violation, disabled via env kill switch). When `Some`, the caller must
    /// NOT apply `silu` again — it is fused into the kernel epilogue.
    fn causal_conv1d_update(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight: &kiln_tensor::Tensor,
        _conv_state: &mut kiln_tensor::Tensor,
        _kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// Fused prefill causal depthwise conv1d + state update + silu.
    ///
    /// `x`: `[B, C, T]` bf16 contiguous with `T > 1`. `weight`: `[C, 1, K]`
    /// bf16 contiguous (or `[C, K]`). `conv_state`: `[B, C, K-1]` F32,
    /// mutated in place after all outputs have consumed the entry state.
    ///
    /// Returns `Ok(Some(out))` with `out: [B, C, T]` F32 (silu-fused), or
    /// `Ok(None)` when the backend declines. When `Some`, the caller must not
    /// apply `silu` again.
    fn causal_conv1d_prefill(
        &self,
        _x: &kiln_tensor::Tensor,
        _weight: &kiln_tensor::Tensor,
        _conv_state: &mut kiln_tensor::Tensor,
        _kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }

    /// Fused GDN gate computation.
    ///
    /// Collapses the Step-6 `sigmoid(b)` + `-exp(A_log) * softplus(a + dt_bias)`
    /// chain into one CUDA launch. Inputs are bf16 tensors of shape
    /// `[B, T, nv]` for `a`, `b` and `[nv]` for `a_log`, `dt_bias`.
    /// Returns `(beta, g)`, both bf16 `[B, T, nv]`, or `Ok(None)` when
    /// the backend declines (wrong dtype, envelope violation, disabled).
    fn gdn_gates(
        &self,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _a_log: &kiln_tensor::Tensor,
        _dt_bias: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        Ok(None)
    }

    /// Fused GDN gated RMSNorm.
    ///
    /// Computes `rms_norm(x, weight) * silu(z)` for Gated DeltaNet outputs.
    /// `x` and `z` are `[B, T, H, D]`, and `weight` is `[D]`.
    /// Returns a tensor with the same shape as `x`. Backends may return the
    /// model dtype directly; the call site already casts to the requested
    /// dtype after reshaping, matching the portable fallback.
    fn gdn_gated_rms_norm(
        &self,
        _x: &kiln_tensor::Tensor,
        _z: &kiln_tensor::Tensor,
        _weight: &kiln_tensor::Tensor,
        _eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        Ok(None)
    }
}

/// Backend identity facet for the Phase 1 `BackendRuntime` split.
///
/// The methods are prefixed with `runtime_` so this scaffold can coexist
/// with the broad compatibility trait while call sites migrate family by family.
#[allow(clippy::too_many_arguments)]
pub trait BackendIdentity: Send + Sync + std::fmt::Debug {
    fn runtime_name(&self) -> &'static str;

    fn runtime_device(&self) -> kiln_tensor::Device;

    fn runtime_as_any(&self) -> &dyn std::any::Any;
}

/// Focused `AttentionBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait AttentionBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_flash_attn_prefill(&self) -> bool;

    fn runtime_supports_flash_attn_prefill_head_major(&self) -> bool;

    fn runtime_supports_flash_attn_paged_decode(&self) -> bool;

    fn runtime_flash_attn_paged_decode_contiguous(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        total_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_flash_attn_paged_decode_contiguous_batch(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slots: &kiln_tensor::Tensor,
        total_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_supports_strict_paged_decode_contiguous_batch(&self) -> bool;

    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_flash_attn_prefill(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_flash_attn_prefill_head_major(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_flash_attn_paged_decode(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        total_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>>;
}

/// Focused `PagedKvBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait PagedKvBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_paged_kv_head_major_read(&self) -> bool;

    fn runtime_supports_paged_kv_head_major_read_append_token_major(&self) -> bool;

    fn runtime_paged_kv_head_major_read(
        &self,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        seq_len: usize,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>>;

    fn runtime_paged_kv_head_major_read_append_token_major(
        &self,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        prefix_len: usize,
        k_tail: &kiln_tensor::Tensor,
        v_tail: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>>;
}

/// Focused `GdnBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait GdnBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_gdn_forward_substitution(&self) -> bool;

    fn runtime_supports_gdn_recurrent_step(&self) -> bool;

    fn runtime_supports_gdn_chunk_prep(&self) -> bool;

    fn runtime_supports_gdn_chunk_scan(&self) -> bool;

    fn runtime_supports_gdn_full_chunk_forward(&self) -> bool;

    fn runtime_supports_gdn_full_chunk_forward_head_last(&self) -> bool;

    fn runtime_supports_gdn_recurrent_prefill_head_last(&self) -> bool;

    fn runtime_supports_gdn_recurrent_prefill_native_head_last(&self) -> bool;

    fn runtime_supports_gdn_recurrent_qk_norm_prefill_native_head_last(&self) -> bool;

    fn runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk(&self) -> bool;

    fn runtime_supports_gdn_decode_qk_norm_gates_recurrent(&self) -> bool;

    fn runtime_supports_gdn_gates(&self) -> bool;

    fn runtime_supports_gdn_gated_rms_norm(&self) -> bool;

    fn runtime_gdn_forward_substitution(
        &self,
        a_strict: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_recurrent_step(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_chunkwise_forward(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        chunk_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_chunk_prep(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    >;

    fn runtime_gdn_chunk_scan(
        &self,
        a_strict: &kiln_tensor::Tensor,
        b_mask: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        q_s_scaled: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        decay_last_col: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>>;

    fn runtime_gdn_full_chunk_forward(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        k_t: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_full_chunk_forward_head_last_into(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        k_t: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        out: &kiln_tensor::Tensor,
        t_start: usize,
        seq_len: usize,
    ) -> Result<bool>;

    fn runtime_gdn_recurrent_prefill_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_recurrent_prefill_native_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_recurrent_qk_norm_prefill_native_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        q_scale: f64,
        qk_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_decode_gates_recurrent(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_decode_qk_norm_gates_recurrent(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        q_scale: f64,
        qk_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_decode_qk_norm_gates_recurrent_rmsnorm(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        q_scale: f64,
        qk_eps: f64,
        rms_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_decode_gates_recurrent_rmsnorm(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_gdn_in_proj_decode(
        &self,
        x: &kiln_tensor::Tensor,
        in_proj_qkv_t: &kiln_tensor::Tensor,
        in_proj_z_t: &kiln_tensor::Tensor,
        in_proj_a_t: &kiln_tensor::Tensor,
        in_proj_b_t: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    >;

    fn runtime_gdn_gates(
        &self,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>>;

    fn runtime_gdn_gated_rms_norm(
        &self,
        x: &kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>>;
}

/// Focused `ConvBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait ConvBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_causal_conv1d_update(&self) -> bool;

    fn runtime_supports_causal_conv1d_prefill(&self) -> bool;

    fn runtime_causal_conv1d_update(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_causal_conv1d_prefill(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>>;
}

/// Focused `LinearBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait LinearBackend: Send + Sync + std::fmt::Debug {
    fn runtime_linear_decode(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_linear_prefill_apply(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_linear_prefill_apply_offset(
        &self,
        x: &kiln_tensor::Tensor,
        full_weight_t: &kiln_tensor::Tensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_lora_delta_resident(
        &self,
        x: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_lora_decode_add(
        &self,
        base: &kiln_tensor::Tensor,
        x: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_prewarm_decode_weights(&self, weights: &crate::forward::GpuWeights) -> Result<()>;

    fn runtime_drop_uploaded_bf16_weights(
        &self,
        weights: &mut crate::forward::GpuWeights,
        device: &kiln_tensor::Device,
    ) -> Result<usize>;

    fn runtime_full_attn_qkv_decode(
        &self,
        x: &kiln_tensor::Tensor,
        q_weight_t: &kiln_tensor::Tensor,
        k_weight_t: &kiln_tensor::Tensor,
        v_weight_t: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    >;

    fn runtime_mlp_gate_up_decode(
        &self,
        x: &kiln_tensor::Tensor,
        gate_weight_t: &kiln_tensor::Tensor,
        up_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_mlp_decode(
        &self,
        x: &kiln_tensor::Tensor,
        gate_weight_t: &kiln_tensor::Tensor,
        up_weight_t: &kiln_tensor::Tensor,
        down_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>>;
}

/// Focused `SamplingBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait SamplingBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_linear_decode_argmax(&self) -> bool;

    fn runtime_linear_decode_argmax(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<u32>>;

    fn runtime_supports_linear_decode_argmax_batch(&self) -> bool;

    fn runtime_linear_decode_argmax_batch(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<Vec<u32>>>;

    fn runtime_supports_linear_decode_sample(&self, top_k: u32) -> bool;

    fn runtime_linear_decode_sample(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        seed: u64,
    ) -> Result<Option<u32>>;

    fn runtime_supports_linear_decode_sample_batch(
        &self,
        top_k: &[u32],
        temperatures: &[f32],
    ) -> bool;

    fn runtime_linear_decode_sample_batch(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
        history_rows: &[u32],
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalties: &[f32],
        presence_penalties: &[f32],
        frequency_penalties: &[f32],
        temperatures: &[f32],
        top_k: &[u32],
        top_p: &[f32],
        min_p: &[f32],
        seeds: &[u64],
    ) -> Result<Option<Vec<u32>>>;
}

/// Focused `ResidencyBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait ResidencyBackend: Send + Sync + std::fmt::Debug {
    fn runtime_supports_resident_activation(&self) -> bool;

    fn runtime_register_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()>;

    fn runtime_evict_resident_activation(&self, tensor: &kiln_tensor::Tensor);

    fn runtime_update_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()>;

    fn runtime_has_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> bool;

    fn runtime_resident_activation_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
    ) -> Option<residency::ResidentResource>;

    fn runtime_resolve_resident_activation(
        &self,
        tensor: &kiln_tensor::Tensor,
        shape: &[usize],
        dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>>;

    fn runtime_enter_gdn_recurrent_resident_state_scope(&self) -> bool;

    fn runtime_exit_gdn_recurrent_resident_state_scope(&self);

    fn runtime_materialize_gdn_recurrent_resident_state(
        &self,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<()>;

    fn runtime_evict_gdn_recurrent_resident_state(&self, state: &kiln_tensor::Tensor);

    fn runtime_has_gdn_recurrent_resident_state(&self, state: &kiln_tensor::Tensor) -> bool;

    fn runtime_assemble_gdn_recurrent_resident_batch_rows(
        &self,
        rows: &[&kiln_tensor::Tensor],
        batch: &kiln_tensor::Tensor,
    ) -> Result<bool>;

    fn runtime_scatter_gdn_recurrent_resident_batch_rows(
        &self,
        batch: &kiln_tensor::Tensor,
        destinations: &mut [&mut kiln_tensor::Tensor],
    ) -> Result<bool>;

    fn runtime_assemble_linear_attn_gdn_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
    ) -> Result<bool>;

    fn runtime_scatter_linear_attn_gdn_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
    ) -> Result<bool>;

    fn runtime_seed_linear_attn_gdn_state_kt(
        &self,
        recurrent: &kiln_tensor::Tensor,
        conv: &kiln_tensor::Tensor,
    ) -> Result<bool>;

    fn runtime_has_linear_attn_gdn_state_kt(&self, key: kiln_tensor::TensorId) -> bool;
}

/// Focused `OptimizerBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait OptimizerBackend: Send + Sync + std::fmt::Debug {
    fn runtime_dispatch_sgd_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        lr: f32,
    ) -> Result<bool>;

    fn runtime_dispatch_adamw_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        first_moment: &kiln_tensor::Tensor,
        second_moment: &kiln_tensor::Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<bool>;
}

/// Focused `TrainingLossBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait TrainingLossBackend: Send + Sync + std::fmt::Debug {
    fn runtime_training_capabilities(&self) -> TrainingCapabilities;

    fn runtime_training_precision_policy(&self) -> TrainingPrecisionPolicy;
}

/// Focused `ReplayBackend` facet delegated by the current `BackendRuntime` facade.
#[allow(clippy::too_many_arguments)]
pub trait ReplayBackend: Send + Sync + std::fmt::Debug {
    fn runtime_decode_resident_pool_ready(
        &self,
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> bool;

    fn runtime_supports_resident_decode(&self) -> bool;

    fn runtime_supports_replay_request(
        &self,
        req: &capability::ReplayRequest,
    ) -> capability::Support;

    fn runtime_replay_key_for_request(
        &self,
        req: &capability::ReplayRequest,
    ) -> kiln_graph::ReplayKey;

    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        graph_outputs: Option<(&kiln_tensor::Tensor, &kiln_tensor::Tensor)>,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>>;
}

// Blanket forwarding impls keep the focused traits behavior-identical to the
// compatibility facade while later PRs move call sites to one facet at a time.
#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> BackendIdentity for T {
    fn runtime_name(&self) -> &'static str {
        BackendRuntime::name(self)
    }

    fn runtime_device(&self) -> kiln_tensor::Device {
        BackendRuntime::device(self)
    }

    fn runtime_as_any(&self) -> &dyn std::any::Any {
        BackendRuntime::as_any(self)
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> AttentionBackend for T {
    fn runtime_supports_flash_attn_prefill(&self) -> bool {
        BackendRuntime::supports_flash_attn_prefill(self)
    }

    fn runtime_supports_flash_attn_prefill_head_major(&self) -> bool {
        BackendRuntime::supports_flash_attn_prefill_head_major(self)
    }

    fn runtime_supports_flash_attn_paged_decode(&self) -> bool {
        BackendRuntime::supports_flash_attn_paged_decode(self)
    }

    fn runtime_flash_attn_paged_decode_contiguous(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        total_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::flash_attn_paged_decode_contiguous(
            self,
            q,
            k_pool,
            v_pool,
            start_slot,
            total_seqlen_k,
            softmax_scale,
        )
    }

    fn runtime_flash_attn_paged_decode_contiguous_batch(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slots: &kiln_tensor::Tensor,
        total_seqlen_k: usize,
        softmax_scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::flash_attn_paged_decode_contiguous_batch(
            self,
            q,
            k_pool,
            v_pool,
            start_slots,
            total_seqlen_k,
            softmax_scale,
        )
    }

    fn runtime_supports_strict_paged_decode_contiguous_batch(&self) -> bool {
        BackendRuntime::supports_strict_paged_decode_contiguous_batch(self)
    }

    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
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

    fn runtime_flash_attn_prefill(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::flash_attn_prefill(self, q, k, v, softmax_scale, causal)
    }

    fn runtime_flash_attn_prefill_head_major(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::flash_attn_prefill_head_major(self, q, k, v, softmax_scale, causal)
    }

    fn runtime_flash_attn_paged_decode(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        total_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::flash_attn_paged_decode(
            self,
            q,
            k_pool,
            v_pool,
            block_table,
            total_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> PagedKvBackend for T {
    fn runtime_supports_paged_kv_head_major_read(&self) -> bool {
        BackendRuntime::supports_paged_kv_head_major_read(self)
    }

    fn runtime_supports_paged_kv_head_major_read_append_token_major(&self) -> bool {
        BackendRuntime::supports_paged_kv_head_major_read_append_token_major(self)
    }

    fn runtime_paged_kv_head_major_read(
        &self,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        seq_len: usize,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        BackendRuntime::paged_kv_head_major_read(self, k_pool, v_pool, start_slot, seq_len)
    }

    fn runtime_paged_kv_head_major_read_append_token_major(
        &self,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        start_slot: usize,
        prefix_len: usize,
        k_tail: &kiln_tensor::Tensor,
        v_tail: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        BackendRuntime::paged_kv_head_major_read_append_token_major(
            self, k_pool, v_pool, start_slot, prefix_len, k_tail, v_tail,
        )
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> GdnBackend for T {
    fn runtime_supports_gdn_forward_substitution(&self) -> bool {
        BackendRuntime::supports_gdn_forward_substitution(self)
    }

    fn runtime_supports_gdn_recurrent_step(&self) -> bool {
        BackendRuntime::supports_gdn_recurrent_step(self)
    }

    fn runtime_supports_gdn_chunk_prep(&self) -> bool {
        BackendRuntime::supports_gdn_chunk_prep(self)
    }

    fn runtime_supports_gdn_chunk_scan(&self) -> bool {
        BackendRuntime::supports_gdn_chunk_scan(self)
    }

    fn runtime_supports_gdn_full_chunk_forward(&self) -> bool {
        BackendRuntime::supports_gdn_full_chunk_forward(self)
    }

    fn runtime_supports_gdn_full_chunk_forward_head_last(&self) -> bool {
        BackendRuntime::supports_gdn_full_chunk_forward_head_last(self)
    }

    fn runtime_supports_gdn_recurrent_prefill_head_last(&self) -> bool {
        BackendRuntime::supports_gdn_recurrent_prefill_head_last(self)
    }

    fn runtime_supports_gdn_recurrent_prefill_native_head_last(&self) -> bool {
        BackendRuntime::supports_gdn_recurrent_prefill_native_head_last(self)
    }

    fn runtime_supports_gdn_recurrent_qk_norm_prefill_native_head_last(&self) -> bool {
        BackendRuntime::supports_gdn_recurrent_qk_norm_prefill_native_head_last(self)
    }

    fn runtime_supports_gdn_decode_gates_recurrent_unexpanded_qk(&self) -> bool {
        BackendRuntime::supports_gdn_decode_gates_recurrent_unexpanded_qk(self)
    }

    fn runtime_supports_gdn_decode_qk_norm_gates_recurrent(&self) -> bool {
        BackendRuntime::supports_gdn_decode_qk_norm_gates_recurrent(self)
    }

    fn runtime_supports_gdn_gates(&self) -> bool {
        BackendRuntime::supports_gdn_gates(self)
    }

    fn runtime_supports_gdn_gated_rms_norm(&self) -> bool {
        BackendRuntime::supports_gdn_gated_rms_norm(self)
    }

    fn runtime_gdn_forward_substitution(
        &self,
        a_strict: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_forward_substitution(self, a_strict, v_prime, beta)
    }

    fn runtime_gdn_recurrent_step(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_recurrent_step(self, q, k, v, beta, g, state)
    }

    fn runtime_gdn_chunkwise_forward(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        chunk_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_chunkwise_forward(self, q, k, v, beta, g, state, chunk_size)
    }

    fn runtime_gdn_chunk_prep(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
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
        BackendRuntime::gdn_chunk_prep(self, g, v, kkt, qkt, ks_entry, q_s)
    }

    fn runtime_gdn_chunk_scan(
        &self,
        a_strict: &kiln_tensor::Tensor,
        b_mask: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        q_s_scaled: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        decay_last_col: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        BackendRuntime::gdn_chunk_scan(
            self,
            a_strict,
            b_mask,
            v_prime,
            q_s_scaled,
            beta,
            decay_last_col,
        )
    }

    fn runtime_gdn_full_chunk_forward(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        k_t: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_full_chunk_forward(
            self, g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
        )
    }

    fn runtime_gdn_full_chunk_forward_head_last_into(
        &self,
        g: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        kkt: &kiln_tensor::Tensor,
        qkt: &kiln_tensor::Tensor,
        ks_entry: &kiln_tensor::Tensor,
        q_s: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        k_t: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        out: &kiln_tensor::Tensor,
        t_start: usize,
        seq_len: usize,
    ) -> Result<bool> {
        BackendRuntime::gdn_full_chunk_forward_head_last_into(
            self, g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
        )
    }

    fn runtime_gdn_recurrent_prefill_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_recurrent_prefill_head_last(self, q, k, v, beta, g, state)
    }

    fn runtime_gdn_recurrent_prefill_native_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_recurrent_prefill_native_head_last(self, q, k, v, beta, g, state)
    }

    fn runtime_gdn_recurrent_qk_norm_prefill_native_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        q_scale: f64,
        qk_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_recurrent_qk_norm_prefill_native_head_last(
            self, q, k, v, beta, g, state, q_scale, qk_eps,
        )
    }

    fn runtime_gdn_decode_gates_recurrent(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_decode_gates_recurrent(
            self, q, k, v, a, b, a_log, dt_bias, state, z, weight, eps,
        )
    }

    fn runtime_gdn_decode_qk_norm_gates_recurrent(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        q_scale: f64,
        qk_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_decode_qk_norm_gates_recurrent(
            self, q, k, v, a, b, a_log, dt_bias, state, q_scale, qk_eps,
        )
    }

    fn runtime_gdn_decode_qk_norm_gates_recurrent_rmsnorm(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        q_scale: f64,
        qk_eps: f64,
        rms_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_decode_qk_norm_gates_recurrent_rmsnorm(
            self, q, k, v, a, b, a_log, dt_bias, state, z, weight, q_scale, qk_eps, rms_eps,
        )
    }

    fn runtime_gdn_decode_gates_recurrent_rmsnorm(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
        state: &mut kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_decode_gates_recurrent_rmsnorm(
            self, q, k, v, a, b, a_log, dt_bias, state, z, weight, eps,
        )
    }

    fn runtime_gdn_in_proj_decode(
        &self,
        x: &kiln_tensor::Tensor,
        in_proj_qkv_t: &kiln_tensor::Tensor,
        in_proj_z_t: &kiln_tensor::Tensor,
        in_proj_a_t: &kiln_tensor::Tensor,
        in_proj_b_t: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        BackendRuntime::gdn_in_proj_decode(
            self,
            x,
            in_proj_qkv_t,
            in_proj_z_t,
            in_proj_a_t,
            in_proj_b_t,
        )
    }

    fn runtime_gdn_gates(
        &self,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        BackendRuntime::gdn_gates(self, a, b, a_log, dt_bias)
    }

    fn runtime_gdn_gated_rms_norm(
        &self,
        x: &kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::gdn_gated_rms_norm(self, x, z, weight, eps)
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> ConvBackend for T {
    fn runtime_supports_causal_conv1d_update(&self) -> bool {
        BackendRuntime::supports_causal_conv1d_update(self)
    }

    fn runtime_supports_causal_conv1d_prefill(&self) -> bool {
        BackendRuntime::supports_causal_conv1d_prefill(self)
    }

    fn runtime_causal_conv1d_update(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::causal_conv1d_update(self, x, weight, conv_state, kernel_size)
    }

    fn runtime_causal_conv1d_prefill(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::causal_conv1d_prefill(self, x, weight, conv_state, kernel_size)
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> LinearBackend for T {
    fn runtime_linear_decode(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::linear_decode(self, x, weight_t)
    }

    fn runtime_linear_prefill_apply(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::linear_prefill_apply(self, x, weight_t)
    }

    fn runtime_linear_prefill_apply_offset(
        &self,
        x: &kiln_tensor::Tensor,
        full_weight_t: &kiln_tensor::Tensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::linear_prefill_apply_offset(self, x, full_weight_t, chunk_start, chunk_len)
    }

    fn runtime_lora_delta_resident(
        &self,
        x: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::lora_delta_resident(self, x, a, b, scale)
    }

    fn runtime_lora_decode_add(
        &self,
        base: &kiln_tensor::Tensor,
        x: &kiln_tensor::Tensor,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::lora_decode_add(self, base, x, a, b, scale)
    }

    fn runtime_prewarm_decode_weights(&self, weights: &crate::forward::GpuWeights) -> Result<()> {
        BackendRuntime::prewarm_decode_weights(self, weights)
    }

    fn runtime_drop_uploaded_bf16_weights(
        &self,
        weights: &mut crate::forward::GpuWeights,
        device: &kiln_tensor::Device,
    ) -> Result<usize> {
        BackendRuntime::drop_uploaded_bf16_weights(self, weights, device)
    }

    fn runtime_full_attn_qkv_decode(
        &self,
        x: &kiln_tensor::Tensor,
        q_weight_t: &kiln_tensor::Tensor,
        k_weight_t: &kiln_tensor::Tensor,
        v_weight_t: &kiln_tensor::Tensor,
    ) -> Result<
        Option<(
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
            kiln_tensor::Tensor,
        )>,
    > {
        BackendRuntime::full_attn_qkv_decode(self, x, q_weight_t, k_weight_t, v_weight_t)
    }

    fn runtime_mlp_gate_up_decode(
        &self,
        x: &kiln_tensor::Tensor,
        gate_weight_t: &kiln_tensor::Tensor,
        up_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::mlp_gate_up_decode(self, x, gate_weight_t, up_weight_t)
    }

    fn runtime_mlp_decode(
        &self,
        x: &kiln_tensor::Tensor,
        gate_weight_t: &kiln_tensor::Tensor,
        up_weight_t: &kiln_tensor::Tensor,
        down_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::mlp_decode(self, x, gate_weight_t, up_weight_t, down_weight_t)
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> SamplingBackend for T {
    fn runtime_supports_linear_decode_argmax(&self) -> bool {
        BackendRuntime::supports_linear_decode_argmax(self)
    }

    fn runtime_linear_decode_argmax(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<u32>> {
        BackendRuntime::linear_decode_argmax(self, x, weight_t)
    }

    fn runtime_supports_linear_decode_argmax_batch(&self) -> bool {
        BackendRuntime::supports_linear_decode_argmax_batch(self)
    }

    fn runtime_linear_decode_argmax_batch(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<Vec<u32>>> {
        BackendRuntime::linear_decode_argmax_batch(self, x, weight_t)
    }

    fn runtime_supports_linear_decode_sample(&self, top_k: u32) -> bool {
        BackendRuntime::supports_linear_decode_sample(self, top_k)
    }

    fn runtime_linear_decode_sample(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalty: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        min_p: f32,
        seed: u64,
    ) -> Result<Option<u32>> {
        BackendRuntime::linear_decode_sample(
            self,
            x,
            weight_t,
            history_indices,
            history_counts,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            temperature,
            top_k,
            top_p,
            min_p,
            seed,
        )
    }

    fn runtime_supports_linear_decode_sample_batch(
        &self,
        top_k: &[u32],
        temperatures: &[f32],
    ) -> bool {
        BackendRuntime::supports_linear_decode_sample_batch(self, top_k, temperatures)
    }

    fn runtime_linear_decode_sample_batch(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
        history_rows: &[u32],
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalties: &[f32],
        presence_penalties: &[f32],
        frequency_penalties: &[f32],
        temperatures: &[f32],
        top_k: &[u32],
        top_p: &[f32],
        min_p: &[f32],
        seeds: &[u64],
    ) -> Result<Option<Vec<u32>>> {
        BackendRuntime::linear_decode_sample_batch(
            self,
            x,
            weight_t,
            history_rows,
            history_indices,
            history_counts,
            repetition_penalties,
            presence_penalties,
            frequency_penalties,
            temperatures,
            top_k,
            top_p,
            min_p,
            seeds,
        )
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> ResidencyBackend for T {
    fn runtime_supports_resident_activation(&self) -> bool {
        BackendRuntime::supports_resident_activation(self)
    }

    fn runtime_register_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        BackendRuntime::register_resident_activation(self, tensor)
    }

    fn runtime_evict_resident_activation(&self, tensor: &kiln_tensor::Tensor) {
        BackendRuntime::evict_resident_activation(self, tensor)
    }

    fn runtime_update_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        BackendRuntime::update_resident_activation(self, tensor)
    }

    fn runtime_has_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> bool {
        BackendRuntime::has_resident_activation(self, tensor)
    }

    fn runtime_resident_activation_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
    ) -> Option<residency::ResidentResource> {
        BackendRuntime::resident_activation_resource(self, tensor)
    }

    fn runtime_resolve_resident_activation(
        &self,
        tensor: &kiln_tensor::Tensor,
        shape: &[usize],
        dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::resolve_resident_activation(self, tensor, shape, dtype)
    }

    fn runtime_enter_gdn_recurrent_resident_state_scope(&self) -> bool {
        BackendRuntime::enter_gdn_recurrent_resident_state_scope(self)
    }

    fn runtime_exit_gdn_recurrent_resident_state_scope(&self) {
        BackendRuntime::exit_gdn_recurrent_resident_state_scope(self)
    }

    fn runtime_materialize_gdn_recurrent_resident_state(
        &self,
        state: &mut kiln_tensor::Tensor,
    ) -> Result<()> {
        BackendRuntime::materialize_gdn_recurrent_resident_state(self, state)
    }

    fn runtime_evict_gdn_recurrent_resident_state(&self, state: &kiln_tensor::Tensor) {
        BackendRuntime::evict_gdn_recurrent_resident_state(self, state)
    }

    fn runtime_has_gdn_recurrent_resident_state(&self, state: &kiln_tensor::Tensor) -> bool {
        BackendRuntime::has_gdn_recurrent_resident_state(self, state)
    }

    fn runtime_assemble_gdn_recurrent_resident_batch_rows(
        &self,
        rows: &[&kiln_tensor::Tensor],
        batch: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        BackendRuntime::assemble_gdn_recurrent_resident_batch_rows(self, rows, batch)
    }

    fn runtime_scatter_gdn_recurrent_resident_batch_rows(
        &self,
        batch: &kiln_tensor::Tensor,
        destinations: &mut [&mut kiln_tensor::Tensor],
    ) -> Result<bool> {
        BackendRuntime::scatter_gdn_recurrent_resident_batch_rows(self, batch, destinations)
    }

    fn runtime_assemble_linear_attn_gdn_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
    ) -> Result<bool> {
        BackendRuntime::assemble_linear_attn_gdn_state_batch_kt(self, row_keys, batch_key)
    }

    fn runtime_scatter_linear_attn_gdn_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
    ) -> Result<bool> {
        BackendRuntime::scatter_linear_attn_gdn_state_batch_kt(self, batch_key, row_keys)
    }

    fn runtime_seed_linear_attn_gdn_state_kt(
        &self,
        recurrent: &kiln_tensor::Tensor,
        conv: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        BackendRuntime::seed_linear_attn_gdn_state_kt(self, recurrent, conv)
    }

    fn runtime_has_linear_attn_gdn_state_kt(&self, key: kiln_tensor::TensorId) -> bool {
        BackendRuntime::has_linear_attn_gdn_state_kt(self, key)
    }
}

impl<T: BackendRuntime + ?Sized> residency::ResidentRegistry for T {
    fn register_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: residency::ResidentResourceFamily,
    ) -> Result<Option<residency::ResidentResource>> {
        match family {
            residency::ResidentResourceFamily::Activation => {
                BackendRuntime::register_resident_activation(self, tensor)?;
                Ok(BackendRuntime::resident_activation_resource(self, tensor))
            }
            _ => Ok(None),
        }
    }

    fn update_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: residency::ResidentResourceFamily,
    ) -> Result<Option<residency::ResidentResource>> {
        match family {
            residency::ResidentResourceFamily::Activation => {
                BackendRuntime::update_resident_activation(self, tensor)?;
                Ok(BackendRuntime::resident_activation_resource(self, tensor))
            }
            _ => Ok(None),
        }
    }

    fn evict_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: residency::ResidentResourceFamily,
    ) {
        if family == residency::ResidentResourceFamily::Activation {
            BackendRuntime::evict_resident_activation(self, tensor);
        }
    }

    fn resident_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: residency::ResidentResourceFamily,
    ) -> Option<residency::ResidentResource> {
        match family {
            residency::ResidentResourceFamily::Activation => {
                BackendRuntime::resident_activation_resource(self, tensor)
            }
            _ => None,
        }
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> OptimizerBackend for T {
    fn runtime_dispatch_sgd_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        lr: f32,
    ) -> Result<bool> {
        BackendRuntime::dispatch_sgd_step(self, param, grad, lr)
    }

    fn runtime_dispatch_adamw_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        first_moment: &kiln_tensor::Tensor,
        second_moment: &kiln_tensor::Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<bool> {
        BackendRuntime::dispatch_adamw_step(
            self,
            param,
            grad,
            first_moment,
            second_moment,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        )
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> TrainingLossBackend for T {
    fn runtime_training_capabilities(&self) -> TrainingCapabilities {
        BackendRuntime::training_capabilities(self)
    }

    fn runtime_training_precision_policy(&self) -> TrainingPrecisionPolicy {
        BackendRuntime::training_precision_policy(self)
    }
}

#[allow(clippy::too_many_arguments)]
impl<T: BackendRuntime + ?Sized> ReplayBackend for T {
    fn runtime_decode_resident_pool_ready(
        &self,
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> bool {
        BackendRuntime::decode_resident_pool_ready(self, max_hidden, max_intermediate, max_batch)
    }

    fn runtime_supports_resident_decode(&self) -> bool {
        BackendRuntime::supports_resident_decode(self)
    }

    fn runtime_supports_replay_request(
        &self,
        req: &capability::ReplayRequest,
    ) -> capability::Support {
        capability::BackendCapabilityQueries::supports_replay_request(self, req)
    }

    fn runtime_replay_key_for_request(
        &self,
        req: &capability::ReplayRequest,
    ) -> kiln_graph::ReplayKey {
        capability::BackendCapabilityQueries::replay_key_for_request(self, req)
    }

    fn runtime_flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
        &self,
        q: &kiln_tensor::Tensor,
        k_pool: &kiln_tensor::Tensor,
        v_pool: &kiln_tensor::Tensor,
        block_table: &kiln_tensor::Tensor,
        seqused_k: &kiln_tensor::Tensor,
        graph_outputs: Option<(&kiln_tensor::Tensor, &kiln_tensor::Tensor)>,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        BackendRuntime::flash_attn_paged_decode_contiguous_batch_dyn_seqlen_with_graph_outputs(
            self,
            q,
            k_pool,
            v_pool,
            block_table,
            seqused_k,
            graph_outputs,
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


#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct ResidentActivationProbeBackend {
        name: &'static str,
        resident: bool,
    }

    impl BackendRuntime for ResidentActivationProbeBackend {
        fn name(&self) -> &'static str {
            self.name
        }

        fn device(&self) -> kiln_tensor::Device {
            kiln_tensor::Device::Cpu
        }

        fn has_resident_activation(&self, _tensor: &kiln_tensor::Tensor) -> bool {
            self.resident
        }
    }

    #[test]
    fn portable_training_capabilities_are_conservative() {
        let caps = TrainingCapabilities::portable();
        assert_eq!(caps.resident_activation, "not implemented");
        assert_eq!(caps.native_training, "not implemented");
        assert!(caps.projection_training.contains("candle"));

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
        assert!(cuda.mixed_precision);

        let metal = TrainingPrecisionPolicy::metal();
        assert_eq!(metal.activation_dtypes, &[kiln_tensor::DType::BF16]);
        assert_eq!(metal.loss_accumulation_dtype, kiln_tensor::DType::F32);
        assert!(metal.notes.contains("UMA"));

        let vulkan = TrainingPrecisionPolicy::vulkan();
        assert_eq!(vulkan.activation_dtypes, &[kiln_tensor::DType::F32]);
        assert_eq!(vulkan.lora_parameter_dtypes, &[kiln_tensor::DType::F32]);
        assert!(vulkan.base_weight_dtypes.contains(&kiln_tensor::DType::BF16));
        assert!(vulkan.mixed_precision);
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
        assert_eq!(
            caps.flash_attn_paged_decode,
            capability::Support::Declined
        );
        assert_eq!(
            caps.paged_kv_head_major_read,
            capability::Support::Declined
        );
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
            capability::BackendCapabilityQueries::supports_attention_request(
                &cpu,
                &attention_req
            ),
            capability::Support::Declined
        );

        let linear_req = capability::LinearRequest::decode_argmax(
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::BF16,
            kiln_tensor::DType::I64,
            1,
            false,
        );
        assert_eq!(
            capability::BackendCapabilityQueries::supports_linear_request(&cpu, &linear_req),
            capability::Support::Declined
        );

        let replay_req = capability::ReplayRequest::resident_decode(8, 16, 2)
            .with_dtype(kiln_tensor::DType::BF16);
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
            capability::BackendCapabilityQueries::supports_replay_request(
                &cpu,
                &unsafe_replay_req,
            ),
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
        assert_eq!(
            caps.attention.flash_prefill,
            capability::Support::Declined
        );
        assert_eq!(caps.gdn.recurrent_step, capability::Support::Declined);
        assert_eq!(caps.decode.linear_argmax, capability::Support::Declined);
        assert_eq!(
            caps.training.precision,
            TrainingPrecisionPolicy::portable()
        );
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
        assert_eq!(
            caps.fallback.training_optimizer,
            FallbackPolicy::CorrectnessAllowed
        );

        let vulkan_probe = ResidentActivationProbeBackend {
            name: "vulkan",
            resident: false,
        };
        let vulkan_caps =
            capability::BackendCapabilityQueries::backend_capabilities(&vulkan_probe);
        assert_eq!(vulkan_caps.device, kiln_tensor::Device::Cpu);
        assert_eq!(vulkan_caps.storage.backend, kiln_tensor::Backend::Vulkan);
        assert_eq!(
            vulkan_caps.fallback.decode_hot_path,
            FallbackPolicy::NativeRequired
        );
        assert_eq!(
            vulkan_caps.fallback.training_optimizer,
            FallbackPolicy::NativeRequired
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
            capability::BackendCapabilityQueries::supports_matmul_request(
                &cpu,
                &incompatible
            ),
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
            capability::BackendCapabilityQueries::supports_matmul_request(
                &metal,
                &metal_bf16
            ),
            capability::Support::NativeWithConstraints
        );

        let vulkan = ResidentActivationProbeBackend {
            name: "vulkan",
            resident: false,
        };
        assert_eq!(
            capability::BackendCapabilityQueries::supports_matmul_request(
                &vulkan,
                &metal_bf16
            ),
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
        assert_eq!(blas.lhs_layout.blas_name(), "row");
        assert_eq!(blas.rhs_layout.blas_name(), "row");
        assert_eq!(blas.out_layout.blas_name(), "row");
        assert_eq!(blas.epilogue.blas_name(), "bias");
        assert_eq!(blas.batch, capability::MatmulBatchPolicy::Single);
        assert_eq!(blas.concurrent_streams, 3);

        let batched = capability::MatmulRequest::plain(
            vec![2, 4, 8, 16],
            vec![2, 4, 16, 32],
            kiln_tensor::DType::F32,
            true,
        );
        let blas_batched = batched.to_blas_request(1).expect("batched request projects");
        assert_eq!((blas_batched.m, blas_batched.n, blas_batched.k), (8, 32, 16));
        assert_eq!(
            blas_batched.batch,
            capability::MatmulBatchPolicy::Batched { batches: 8 }
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
        assert_eq!(
            mixed.to_blas_request(1),
            Err(capability::MatmulRequestProjectionError::MixedDTypes)
        );
        assert_eq!(
            mixed.to_blas_request(0),
            Err(capability::MatmulRequestProjectionError::InvalidConcurrentStreams)
        );
    }

    #[test]
    fn portable_backend_declines_resident_decode_by_default() {
        // The default trait implementation must return false so that
        // every non-Vulkan backend continues to route through the
        // unchanged `model_forward_paged_last_token*` path — the
        // contract pinned by gate (c) of docs/vk_resident_decode_plan.md.
        let cpu = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);
        assert!(
            !cpu.supports_resident_decode(),
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

        assert!(backend.resident_activation_resource(&tensor).is_none());

        let backend = ResidentActivationProbeBackend {
            name: "vulkan",
            resident: true,
        };
        let resource = backend.resident_activation_resource(&tensor).unwrap();

        assert_eq!(resource.tensor_id, tensor.id());
        assert_eq!(resource.backend, kiln_tensor::Backend::Vulkan);
        assert_eq!(resource.device, kiln_tensor::Device::Cpu);
        assert_eq!(resource.family, residency::ResidentResourceFamily::Activation);
        assert_eq!(resource.ownership, residency::ResidentOwnership::RegistryOwned);
        assert_eq!(resource.state, residency::ResidentResourceState::RegisteredClean);
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
        assert_eq!(resource.family, residency::ResidentResourceFamily::Activation);

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

        assert!(residency::ResidentRegistry::resident_resource(
            &backend,
            &tensor,
            residency::ResidentResourceFamily::OptimizerParam,
        )
        .is_none());
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
            !cuda.supports_resident_decode(),
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

    #[test]
    fn focused_facets_forward_cpu_advertised_capabilities() -> Result<()> {
        let cpu = cpu::CpuBackend::new(kiln_tensor::Device::Cpu);

        assert_eq!(BackendIdentity::runtime_name(&cpu), BackendRuntime::name(&cpu));
        assert_eq!(
            BackendIdentity::runtime_device(&cpu),
            BackendRuntime::device(&cpu)
        );

        assert_eq!(
            AttentionBackend::runtime_supports_flash_attn_prefill(&cpu),
            BackendRuntime::supports_flash_attn_prefill(&cpu)
        );
        assert_eq!(
            AttentionBackend::runtime_supports_flash_attn_prefill_head_major(&cpu),
            BackendRuntime::supports_flash_attn_prefill_head_major(&cpu)
        );
        assert_eq!(
            AttentionBackend::runtime_supports_flash_attn_paged_decode(&cpu),
            BackendRuntime::supports_flash_attn_paged_decode(&cpu)
        );
        assert_eq!(
            AttentionBackend::runtime_supports_strict_paged_decode_contiguous_batch(&cpu),
            BackendRuntime::supports_strict_paged_decode_contiguous_batch(&cpu)
        );

        assert_eq!(
            PagedKvBackend::runtime_supports_paged_kv_head_major_read(&cpu),
            BackendRuntime::supports_paged_kv_head_major_read(&cpu)
        );
        assert_eq!(
            PagedKvBackend::runtime_supports_paged_kv_head_major_read_append_token_major(&cpu),
            BackendRuntime::supports_paged_kv_head_major_read_append_token_major(&cpu)
        );

        assert_eq!(
            GdnBackend::runtime_supports_gdn_forward_substitution(&cpu),
            BackendRuntime::supports_gdn_forward_substitution(&cpu)
        );
        assert_eq!(
            GdnBackend::runtime_supports_gdn_recurrent_step(&cpu),
            BackendRuntime::supports_gdn_recurrent_step(&cpu)
        );
        assert_eq!(
            GdnBackend::runtime_supports_gdn_chunk_prep(&cpu),
            BackendRuntime::supports_gdn_chunk_prep(&cpu)
        );
        assert_eq!(
            GdnBackend::runtime_supports_gdn_chunk_scan(&cpu),
            BackendRuntime::supports_gdn_chunk_scan(&cpu)
        );
        assert_eq!(
            GdnBackend::runtime_supports_gdn_full_chunk_forward(&cpu),
            BackendRuntime::supports_gdn_full_chunk_forward(&cpu)
        );
        assert_eq!(
            GdnBackend::runtime_supports_gdn_gates(&cpu),
            BackendRuntime::supports_gdn_gates(&cpu)
        );
        assert_eq!(
            GdnBackend::runtime_supports_gdn_gated_rms_norm(&cpu),
            BackendRuntime::supports_gdn_gated_rms_norm(&cpu)
        );

        assert_eq!(
            ConvBackend::runtime_supports_causal_conv1d_update(&cpu),
            BackendRuntime::supports_causal_conv1d_update(&cpu)
        );
        assert_eq!(
            ConvBackend::runtime_supports_causal_conv1d_prefill(&cpu),
            BackendRuntime::supports_causal_conv1d_prefill(&cpu)
        );

        assert_eq!(
            SamplingBackend::runtime_supports_linear_decode_argmax(&cpu),
            BackendRuntime::supports_linear_decode_argmax(&cpu)
        );
        assert_eq!(
            SamplingBackend::runtime_supports_linear_decode_argmax_batch(&cpu),
            BackendRuntime::supports_linear_decode_argmax_batch(&cpu)
        );
        assert_eq!(
            SamplingBackend::runtime_supports_linear_decode_sample(&cpu, 8),
            BackendRuntime::supports_linear_decode_sample(&cpu, 8)
        );
        assert_eq!(
            SamplingBackend::runtime_supports_linear_decode_sample_batch(&cpu, &[8], &[1.0]),
            BackendRuntime::supports_linear_decode_sample_batch(&cpu, &[8], &[1.0])
        );

        assert_eq!(
            ResidencyBackend::runtime_supports_resident_activation(&cpu),
            BackendRuntime::supports_resident_activation(&cpu)
        );
        let t = kiln_tensor::Tensor::zeros_cpu(vec![1], kiln_tensor::DType::F32);
        assert_eq!(
            ResidencyBackend::runtime_resident_activation_resource(&cpu, &t),
            BackendRuntime::resident_activation_resource(&cpu, &t)
        );
        assert_eq!(
            ResidencyBackend::runtime_enter_gdn_recurrent_resident_state_scope(&cpu),
            BackendRuntime::enter_gdn_recurrent_resident_state_scope(&cpu)
        );

        assert_eq!(
            ReplayBackend::runtime_decode_resident_pool_ready(&cpu, 8, 16, 2),
            BackendRuntime::decode_resident_pool_ready(&cpu, 8, 16, 2)
        );
        assert_eq!(
            ReplayBackend::runtime_supports_resident_decode(&cpu),
            BackendRuntime::supports_resident_decode(&cpu)
        );
        let replay_req = capability::ReplayRequest::paged_decode_graph_outputs(8, 16, 2)
            .with_dtype(kiln_tensor::DType::BF16);
        assert_eq!(
            ReplayBackend::runtime_supports_replay_request(&cpu, &replay_req),
            capability::BackendCapabilityQueries::supports_replay_request(&cpu, &replay_req)
        );
        assert_eq!(
            ReplayBackend::runtime_replay_key_for_request(&cpu, &replay_req),
            capability::BackendCapabilityQueries::replay_key_for_request(&cpu, &replay_req)
        );

        assert_eq!(
            TrainingLossBackend::runtime_training_capabilities(&cpu),
            BackendRuntime::training_capabilities(&cpu)
        );
        assert_eq!(
            TrainingLossBackend::runtime_training_precision_policy(&cpu),
            BackendRuntime::training_precision_policy(&cpu)
        );

        assert_eq!(
            OptimizerBackend::runtime_dispatch_sgd_step(&cpu, &t, &t, 0.1)?,
            BackendRuntime::dispatch_sgd_step(&cpu, &t, &t, 0.1)?
        );
        assert_eq!(
            OptimizerBackend::runtime_dispatch_adamw_step(
                &cpu, &t, &t, &t, &t, 0.1, 0.9, 0.999, 1e-8, 0.0, 1
            )?,
            BackendRuntime::dispatch_adamw_step(&cpu, &t, &t, &t, &t, 0.1, 0.9, 0.999, 1e-8, 0.0, 1)?
        );

        Ok(())
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
            backend.supports_resident_decode(),
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
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_training_capabilities_advertise_adamw_but_decline_sgd() {
        let caps = metal::MetalBackend::training_capabilities_static();
        assert!(caps.sgd_step.contains("declined"));
        assert!(caps.adamw_step.contains("Metal in-place AdamW"));
        assert!(caps.resident_activation.contains("Metal TensorId"));

        let backend = metal::MetalBackend::new(kiln_tensor::Device::Metal(0));
        assert_eq!(backend.training_capabilities(), caps);
    }
}

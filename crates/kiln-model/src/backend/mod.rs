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
//! **`Option<Tensor>` return**: `Ok(None)` means "this backend declines this
//! call — fall back to the portable candle path". Matches the existing
//! `try_flash_attn_paged_decode` precondition-miss contract and extends it
//! to all kernel ops.
//!
//! **`supports_*` hints**: let the caller skip preamble work (e.g., a
//! `contiguous()` copy before the trait call) when the backend will decline
//! anyway. Intended to be constant-return for each concrete backend.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Process-global flag set when Vulkan is the active backend.
///
/// candle-core has no `Device::Vulkan`, so call sites in `forward.rs` and
/// `trainer.rs` see `Device::Cpu` even when the real compute lives on a
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

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "vulkan")]
pub mod vulkan_linear_op;
#[cfg(feature = "vulkan")]
pub mod vulkan_lora_op;

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

pub trait BackendRuntime: Send + Sync + std::fmt::Debug {
    /// Human-readable name (`"cuda"`, `"metal"`, `"cpu"`). Surfaced in
    /// `/health` and logs.
    fn name(&self) -> &'static str;

    /// The candle `Device` this backend drives. All tensors passed to trait
    /// methods must live on this device.
    fn device(&self) -> &Device;

    /// Operator-facing summary of which training paths are backend-native,
    /// candle-on-device, or intentionally declined. This is telemetry only:
    /// dispatch methods remain the source of truth for actual behavior.
    fn training_capabilities(&self) -> TrainingCapabilities {
        TrainingCapabilities::portable()
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
        _q: &Tensor,
        _k_pool: &Tensor,
        _v_pool: &Tensor,
        _start_slot: usize,
        _total_seqlen_k: usize,
        _softmax_scale: f32,
    ) -> Result<Option<Tensor>> {
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
        _q: &Tensor,
        _k_pool: &Tensor,
        _v_pool: &Tensor,
        _start_slots: &Tensor,
        _total_seqlen_k: usize,
        _softmax_scale: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
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
        _q: &Tensor,
        _k_pool: &Tensor,
        _v_pool: &Tensor,
        _block_table: &Tensor,
        _seqused_k: &Tensor,
        _max_seqlen_k: usize,
        _page_block_size: usize,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<Tensor>> {
        Ok(None)
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

    fn materialize_gdn_recurrent_resident_state(&self, _state: &mut Tensor) -> Result<()> {
        Ok(())
    }

    fn evict_gdn_recurrent_resident_state(&self, _state: &Tensor) {}

    fn has_gdn_recurrent_resident_state(&self, _state: &Tensor) -> bool {
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
    fn register_resident_activation(&self, _tensor: &Tensor) -> Result<()> {
        Ok(())
    }

    /// Evict a previously-registered activation from the residency
    /// registry. Caller invokes this when the autograd pass no longer
    /// needs the tensor (e.g. after a segment's backward completes).
    /// No-op default.
    fn evict_resident_activation(&self, _tensor: &Tensor) {}

    /// Re-upload the tensor's current bytes into its registry buffer
    /// (if registered). Caller invokes this when the candle CPU
    /// storage has been mutated outside of the registry — e.g. after
    /// the candle-CPU SGD step writes a new value to a registered
    /// LoRA Var. Without this, `lora_delta_resident` and friends
    /// would keep reading the original init bytes from the buffer.
    ///
    /// No-op default; backends without a registry have nothing to
    /// keep in sync.
    fn update_resident_activation(&self, _tensor: &Tensor) -> Result<()> {
        Ok(())
    }

    /// True when the given tensor has been registered as
    /// resident-on-device. Used by routing code to decide between the
    /// resident fast path and the legacy CPU-roundtrip path. False by
    /// default so callers without registry support continue to use the
    /// legacy path.
    fn has_resident_activation(&self, _tensor: &Tensor) -> bool {
        false
    }

    /// Read a previously-registered activation back from device into
    /// a fresh CPU `Tensor` with the given shape and dtype. Returns
    /// `Ok(None)` when the activation isn't resident — caller should
    /// then use whatever CPU-side storage they retained originally.
    ///
    /// Phase 3.2 of the residency plan: pairs with
    /// `register_resident_activation` to let `checkpointed_forward_backward`
    /// drop the candle CPU mirror after registering, then re-materialise
    /// only when the recompute pass actually needs the boundary.
    /// Today's no-op default returns `Ok(None)` so callers without
    /// registry support fall through to the legacy code path.
    fn resolve_resident_activation(
        &self,
        _tensor: &Tensor,
        _shape: &[usize],
        _dtype: DType,
    ) -> Result<Option<Tensor>> {
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
    fn dispatch_sgd_step(&self, _param: &Tensor, _grad: &Tensor, _lr: f32) -> Result<bool> {
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
        _param: &Tensor,
        _grad: &Tensor,
        _first_moment: &Tensor,
        _second_moment: &Tensor,
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
    /// path no longer reads `var.as_tensor()`'s candle CPU storage
    /// for data — only for shape metadata. Phase 4.2's
    /// `dispatch_sgd_step` can then write to the same registry
    /// buffers in place without a sync-back to candle storage.
    fn lora_delta_resident(
        &self,
        _x: &Tensor,
        _a: &Tensor,
        _b: &Tensor,
        _scale: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn assemble_gdn_recurrent_resident_batch_rows(
        &self,
        _rows: &[&Tensor],
        _batch: &Tensor,
    ) -> Result<bool> {
        Ok(false)
    }

    fn scatter_gdn_recurrent_resident_batch_rows(
        &self,
        _batch: &Tensor,
        _destinations: &mut [&mut Tensor],
    ) -> Result<bool> {
        Ok(false)
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
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<Tensor>> {
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
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<Tensor>> {
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
        _q: &Tensor,
        _k_pool: &Tensor,
        _v_pool: &Tensor,
        _block_table: &Tensor,
        _total_seqlen_k: usize,
        _page_block_size: usize,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<Tensor>> {
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
        _k_pool: &Tensor,
        _v_pool: &Tensor,
        _start_slot: usize,
        _seq_len: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
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
        _k_pool: &Tensor,
        _v_pool: &Tensor,
        _start_slot: usize,
        _prefix_len: usize,
        _k_tail: &Tensor,
        _v_tail: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
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
        _a_strict: &Tensor,
        _v_prime: &Tensor,
        _beta: &Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Gated DeltaNet single-token recurrent step (decode fast path).
    ///
    /// `q`, `k`: `[B, H, dk]` bf16. `v`: `[B, H, dv]` bf16.
    /// `beta`, `g`: `[B, H]` bf16. `state`: `[B, H, dk, dv]` bf16,
    /// mutated in place. Returns `out: [B, H, dv]` bf16.
    fn gdn_recurrent_step(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _beta: &Tensor,
        _g: &Tensor,
        _state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
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
        _g: &Tensor,
        _v: &Tensor,
        _kkt: &Tensor,
        _qkt: &Tensor,
        _ks_entry: &Tensor,
        _q_s: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)>> {
        Ok(None)
    }

    fn gdn_chunk_scan(
        &self,
        _a_strict: &Tensor,
        _b_mask: &Tensor,
        _v_prime: &Tensor,
        _q_s_scaled: &Tensor,
        _beta: &Tensor,
        _decay_last_col: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        Ok(None)
    }

    fn gdn_full_chunk_forward(
        &self,
        _g: &Tensor,
        _v: &Tensor,
        _kkt: &Tensor,
        _qkt: &Tensor,
        _ks_entry: &Tensor,
        _q_s: &Tensor,
        _beta: &Tensor,
        _k_t: &Tensor,
        _state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_full_chunk_forward_head_last_into(
        &self,
        _g: &Tensor,
        _v: &Tensor,
        _kkt: &Tensor,
        _qkt: &Tensor,
        _ks_entry: &Tensor,
        _q_s: &Tensor,
        _beta: &Tensor,
        _k_t: &Tensor,
        _state: &mut Tensor,
        _out: &Tensor,
        _t_start: usize,
        _seq_len: usize,
    ) -> Result<bool> {
        Ok(false)
    }

    fn gdn_recurrent_prefill_head_last(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _beta: &Tensor,
        _g: &Tensor,
        _state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn gdn_recurrent_prefill_native_head_last(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _beta: &Tensor,
        _g: &Tensor,
        _state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn gdn_recurrent_qk_norm_prefill_native_head_last(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _beta: &Tensor,
        _g: &Tensor,
        _state: &mut Tensor,
        _q_scale: f64,
        _qk_eps: f64,
    ) -> Result<Option<Tensor>> {
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
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _a: &Tensor,
        _b: &Tensor,
        _a_log: &Tensor,
        _dt_bias: &Tensor,
        _state: &mut Tensor,
        _z: &Tensor,
        _weight: &Tensor,
        _eps: f64,
    ) -> Result<Option<Tensor>> {
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
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _a: &Tensor,
        _b: &Tensor,
        _a_log: &Tensor,
        _dt_bias: &Tensor,
        _state: &mut Tensor,
        _q_scale: f64,
        _qk_eps: f64,
    ) -> Result<Option<Tensor>> {
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
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _a: &Tensor,
        _b: &Tensor,
        _a_log: &Tensor,
        _dt_bias: &Tensor,
        _state: &mut Tensor,
        _z: &Tensor,
        _weight: &Tensor,
        _q_scale: f64,
        _qk_eps: f64,
        _rms_eps: f64,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Fused native-MTP GDN decode gates + recurrent update + gated RMSNorm.
    ///
    /// Narrow decode path for `seq_len == 1`. Returns `[B, 1, value_heads, dv]`
    /// after gated RMSNorm, mutating `state` in place.
    #[allow(clippy::too_many_arguments)]
    fn gdn_decode_gates_recurrent_rmsnorm(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _a: &Tensor,
        _b: &Tensor,
        _a_log: &Tensor,
        _dt_bias: &Tensor,
        _state: &mut Tensor,
        _z: &Tensor,
        _weight: &Tensor,
        _eps: f64,
    ) -> Result<Option<Tensor>> {
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
        _x: &Tensor,
        _in_proj_qkv_t: &Tensor,
        _in_proj_z_t: &Tensor,
        _in_proj_a_t: &Tensor,
        _in_proj_b_t: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor)>> {
        Ok(None)
    }

    /// Transposed linear projection.
    ///
    /// `x` is `[batch, seq_len, hidden]`, `weight_t` is `[hidden, out_dim]`,
    /// and the output shape is `[batch, seq_len, out_dim]`. Backends should
    /// return `Ok(None)` for unsupported shapes, dtypes, LoRA paths, or debug
    /// modes.
    fn linear_decode(&self, _x: &Tensor, _weight_t: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Autograd-safe transposed linear projection for prefill / training.
    ///
    /// Same shapes as `linear_decode` but the result must be wired into the
    /// candle autograd graph so `.backward()` produces a real gradient.
    /// Implementations typically wrap the dispatch in a `CustomOp1` with a
    /// proper `bwd` impl. Backends without an autograd-safe path return
    /// `Ok(None)` so the caller falls back to the candle CPU matmul.
    fn linear_prefill_apply(&self, _x: &Tensor, _weight_t: &Tensor) -> Result<Option<Tensor>> {
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
    /// is consumed inside the FLCE CustomOp1's `cpu_fwd`.
    fn linear_prefill_apply_offset(
        &self,
        _x: &Tensor,
        _full_weight_t: &Tensor,
        _chunk_start: usize,
        _chunk_len: usize,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn supports_linear_decode_argmax(&self) -> bool {
        false
    }

    /// Single-token transposed linear projection with argmax reduction.
    ///
    /// Used by greedy LM-head decode when logits do not need to be materialized
    /// on the host. `x` is `[1, 1, hidden]`, `weight_t` is `[hidden, out_dim]`.
    fn linear_decode_argmax(&self, _x: &Tensor, _weight_t: &Tensor) -> Result<Option<u32>> {
        Ok(None)
    }

    fn supports_linear_decode_argmax_batch(&self) -> bool {
        false
    }

    /// Batched single-token transposed linear projection with argmax reduction.
    ///
    /// Used by greedy native-batch LM-head decode when logits do not need to be
    /// materialized on the host. `x` is `[batch, 1, hidden]`, `weight_t` is
    /// `[hidden, out_dim]`, and the result contains one token id per batch row.
    fn linear_decode_argmax_batch(
        &self,
        _x: &Tensor,
        _weight_t: &Tensor,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Forward-only LoRA delta/add for decode.
    ///
    /// `base` is the already-computed base projection output, `x` is the
    /// projection input, and `a`/`b` are PEFT LoRA matrices. Backends must
    /// return `Ok(None)` for tracked tensors; training needs the differentiable
    /// Candle path.
    fn lora_decode_add(
        &self,
        _base: &Tensor,
        _x: &Tensor,
        _a: &Tensor,
        _b: &Tensor,
        _scale: f32,
    ) -> Result<Option<Tensor>> {
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
        _device: &Device,
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
        _x: &Tensor,
        _q_weight_t: &Tensor,
        _k_weight_t: &Tensor,
        _v_weight_t: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        Ok(None)
    }

    /// Fused single-token MLP gate/up projection.
    ///
    /// `x` is `[1, 1, hidden]`; both weights are `[hidden, intermediate]`.
    /// Returns `[1, 1, intermediate]` containing `silu(x @ gate_t) * (x @ up_t)`.
    fn mlp_gate_up_decode(
        &self,
        _x: &Tensor,
        _gate_weight_t: &Tensor,
        _up_weight_t: &Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Fused single-token MLP that keeps the SwiGLU hidden activation on backend device.
    ///
    /// `x` is `[1, 1, hidden]`; `gate_weight_t` and `up_weight_t` are
    /// `[hidden, intermediate]`; `down_weight_t` is `[intermediate, out_dim]`.
    fn mlp_decode(
        &self,
        _x: &Tensor,
        _gate_weight_t: &Tensor,
        _up_weight_t: &Tensor,
        _down_weight_t: &Tensor,
    ) -> Result<Option<Tensor>> {
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
        _x: &Tensor,
        _weight: &Tensor,
        _conv_state: &mut Tensor,
        _kernel_size: usize,
    ) -> Result<Option<Tensor>> {
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
        _x: &Tensor,
        _weight: &Tensor,
        _conv_state: &mut Tensor,
        _kernel_size: usize,
    ) -> Result<Option<Tensor>> {
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
        _a: &Tensor,
        _b: &Tensor,
        _a_log: &Tensor,
        _dt_bias: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
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
        _x: &Tensor,
        _z: &Tensor,
        _weight: &Tensor,
        _eps: f64,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

/// Pick the right backend for a given candle device.
///
/// On Metal devices, `--features metal` uses Kiln's native candle-metal
/// backend and Metal kernels. The former MLX bridge was removed because it
/// only accelerated attention while paying Candle<->MLX host-copy overheads
/// and bypassing Kiln's Qwen3.5 GDN decode kernels.
///
/// Vulkan devices are detected at runtime — candle-core has no native Vulkan
/// device, so we always pass a CPU device to `VulkanBackend` and let it
/// manage its own `vk::Device` internally.
pub fn for_device(device: &Device) -> Arc<dyn BackendRuntime> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => Arc::new(cuda::CudaBackend::new(device.clone())),
        #[cfg(feature = "metal")]
        Device::Metal(_) => Arc::new(metal::MetalBackend::new(device.clone())),
        _ => {
            // Vulkan: candle-core has no Device::Vulkan, so we detect at runtime
            #[cfg(feature = "vulkan")]
            {
                if vulkan::vulkan_is_available() {
                    mark_vulkan_active();
                    return Arc::new(vulkan::VulkanBackend::new(device.clone()));
                }
            }
            Arc::new(cpu::CpuBackend::new(device.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_training_capabilities_are_conservative() {
        let caps = TrainingCapabilities::portable();
        assert_eq!(caps.resident_activation, "not implemented");
        assert_eq!(caps.native_training, "not implemented");
        assert!(caps.projection_training.contains("candle"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_training_capabilities_do_not_overclaim_native_training() {
        let caps = cuda::CudaBackend::training_capabilities_static();
        assert!(caps.projection_training.contains("offset chunk hook"));
        assert!(
            caps.lora_delta_training
                .contains("declines tracked tensors")
        );
        assert_eq!(
            caps.resident_activation,
            "TensorId lifecycle registry; candle CUDA tensors are canonical"
        );
        assert!(caps.sgd_step.contains("CUDA in-place optimizer kernel"));
        assert!(caps.adamw_step.contains("CUDA in-place optimizer kernel"));
        assert_eq!(caps.native_training, "not implemented");
    }
}

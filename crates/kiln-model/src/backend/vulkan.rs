//! Vulkan backend: FlashAttention-2 and Gated DeltaNet fused kernels via Vulkan.
//!
//! kt (`kiln_tensor`) has no native Vulkan device, so this backend manages
//! its own `vk::Device`. Normal inference still exposes a kt
//! `kiln_tensor::Device::Cpu` surface and may fall back to portable kt ops
//! when a Vulkan backend method declines a call. Vulkan-native SFT/GRPO
//! training use the separate `VkTensor` stack to keep weights, activations,
//! loss, backward, and optimizer updates resident on Vulkan buffers.
//!
//! `Ok(None)` responses route the caller to the portable kt path.

use anyhow::{Context, Result};

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use super::vulkan_config::VulkanRuntimeConfig;
use super::vulkan_residency::{
    ResidentActivationEntry, contains_recurrent_state_resident_buffer,
    enter_recurrent_state_resident_scope, exit_recurrent_state_resident_scope,
    get_recurrent_state_resident_buffer, insert_recurrent_state_resident_buffer,
    recurrent_state_resident_buffers_for, recurrent_state_resident_scope_active,
    remove_recurrent_state_resident_buffer, replace_recurrent_state_resident_buffer,
    take_recurrent_state_resident_buffer, with_resident_registry,
};
use super::vulkan_tensor_bridge::{
    kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape,
    kt_tensor_to_packed_bf16_bytes_with_shape,
};
use super::{
    AttentionBackend, BackendIdentity, BackendRuntime, ConvBackend, ExternalYieldBackend,
    GdnBackend, LinearBackend, OptimizerBackend, PagedKvBackend, ReplayBackend, ResidencyBackend,
    SamplingBackend, StartupBackend, TrainingCapabilities, TrainingLossBackend,
    TrainingPrecisionPolicy, matmul_request_support_rank, matmul_support_from_native,
    vulkan_attention, vulkan_conv1d, vulkan_dense, vulkan_device, vulkan_gdn, vulkan_linear,
    vulkan_training, vulkan_weights,
};
use crate::forward::GpuWeights;

pub use super::vulkan_device::{
    precompile_custom_kernels, vulkan_device_name, vulkan_is_available,
};
pub use super::vulkan_training::{dispatch_adamw_step_buffers, dispatch_sgd_step_buffers};

/// Vulkan backend for Kiln.
///
/// Manages its own Vulkan device and dispatches compute shaders for
/// FlashAttention-2, Gated DeltaNet, and supporting operations.
#[derive(Debug)]
pub struct VulkanBackend {
    /// `kiln_tensor::Device` form advertised by `BackendRuntime::device()`.
    /// `kt::Device::Vulkan(0)` when the Vulkan logical device is up;
    /// `kt::Device::Cpu` otherwise, matching the CPU-fallback advertised
    /// by `name()` when `vulkan_device` is `None`. Cached at construction
    /// so the hot trait accessor does not bridge per call. (#1082)
    device_kt: kiln_tensor::Device,
    pub(super) resident_activation_registry: super::vulkan_residency::ResidentActivationRegistry,
    /// Cached at construction: reading env vars per decode step × 24 GDN layers
    /// shows up in decode NVTX captures. Env vars don't change at runtime.
    pub(super) gdn_enabled: bool,
    pub(super) gdn_prefill_in_proj_enabled: bool,
    pub(super) gdn_gates_enabled: bool,
    pub(super) gdn_gated_rms_norm_enabled: bool,
    pub(super) gdn_full_chunk_forward_enabled: bool,
    pub(super) fused_conv1d_update_enabled: bool,
    pub(super) fused_conv1d_prefill_enabled: bool,
    pub(super) conv1d_prefill_single_submit_enabled: bool,
    pub(super) gdn_forward_sub_enabled: bool,
    pub(super) gdn_decode_fused_enabled: bool,
    pub(super) gdn_recurrent_unexpanded_qk_enabled: bool,
    pub(super) gdn_recurrent_qk_norm_unexpanded_enabled: bool,
    pub(super) linear_decode_enabled: bool,
    pub(super) linear_argmax_batch_enabled: bool,
    pub(super) full_attn_qkv_enabled: bool,
    pub(super) paged_attn_decode_batch_enabled: bool,
    pub(super) mlp_decode_enabled: bool,
    pub(super) mlp_gate_up_enabled: bool,
    pub(super) mlp_bf16_gate_up_f32_down_enabled: bool,
    pub(super) bf16_packed_linear_weights_enabled: bool,
    pub(super) bf16_packed_gdn_in_proj_weights_enabled: bool,
    pub(super) bf16_packed_full_attn_qkv_weights_enabled: bool,
    pub(super) bf16_packed_mlp_decode_weights_enabled: bool,
    pub(super) weight_prewarm_enabled: bool,
    pub(super) recurrent_state_residency_enabled: bool,
    /// Cached `supports_resident_decode()` evaluation. The trait method
    /// is called per-call on the hot path; reading env vars and checking
    /// the device handle every time would be wasteful. Set at
    /// construction from `KILN_VULKAN_RESIDENT_DECODE` (default on when
    /// the device is up) and never changes.
    resident_decode_enabled: bool,
    /// Lazily constructed fixed ring of 3-4 reusable intermediate
    /// `VulkanBuffer`s sized to `max(hidden, intermediate) × max_batch × 4`
    /// bytes. The first resident-decode call ever made on this backend
    /// publishes the ring; subsequent calls reuse the same slots.
    ///
    /// `OnceLock<Option<...>>` so a backend that fails the pool
    /// feasibility check (Strix Halo near the 16 GiB UMA limit) caches
    /// the `None` and routes every subsequent call to the per-call
    /// kt `kiln_tensor::Tensor` path without re-checking.
    pub(super) decode_resident_pool: OnceLock<Option<Arc<kiln_vulkan_kernel::DecodeResidentPool>>>,
    /// Lazily constructed Vulkan-resident paged KV cache. Mirrors the
    /// legacy `PagedKvCache` layout in device-local f32 buffers so the
    /// resident decode dispatchers can read/write K/V without crossing
    /// the host boundary. The first resident decode call that needs the
    /// cache constructs it for the active model geometry.
    pub(super) vk_paged_kv_cache: OnceLock<Option<Arc<kiln_vulkan_kernel::VkPagedKvCache>>>,
    /// Set of full-attention layer indices whose K/V state has already
    /// been seeded into the Vulkan-resident paged cache from the legacy
    /// candle pool. Each full-attention layer is seeded once at the
    /// first call to the resident block helper for that layer; subsequent
    /// decode steps only do per-token slot writes.
    pub(super) seeded_full_attn_layers: Mutex<HashSet<usize>>,
    /// Batched resident decode rows whose prompt K/V blocks have been seeded.
    /// Keyed by `(full_attention_layer_idx, decode_row_id)`.
    pub(super) seeded_resident_decode_rows: Mutex<HashSet<(usize, u64)>>,
    /// kt-native mirrors for the single-submit resident decode path.
    pub(super) linear_attn_recurrent_state_kt:
        Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    pub(super) linear_attn_conv_state_kt:
        Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    pub(super) seeded_linear_attn_layers_kt: Mutex<HashSet<kiln_tensor::TensorId>>,
    /// Last `start_pos` we saw on the Vulkan-resident decode path.
    /// Within a single request the resident decode runs once per
    /// token with monotonically incrementing `start_pos`; a jump
    /// (the first call after server start, or any new request whose
    /// first decode step doesn't land at `last + 1`) marks a session
    /// boundary, and `note_resident_session()` clears the per-layer
    /// seeded sets so the next call re-seeds the resident
    /// `VkPagedKvCache` from this request's prefill. Cheap because
    /// the re-seed is now slot-range-aware (see
    /// `vk_decode_resident::seed_vk_kv_cache_layer_blocks_from_kt`).
    pub(super) last_resident_start_pos: Mutex<Option<usize>>,
    /// Scratch activation buffers reused across resident decode calls,
    /// keyed by a stable role string. Each entry persists for the
    /// backend's lifetime (single-sequence decode reuses the same
    /// buffers across layers and across tokens). Avoids the
    /// `create_device_local` + `Drop` pair that ran on every call
    /// (≈ 200 µs × 12 buffers × N layers per token).
    pub(super) resident_scratch:
        Mutex<HashMap<&'static str, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    /// (#1082) kt-native weight caches keyed on the **kt** `TensorId`. The
    /// decode hot path hands weights through as kt tensors whose `TensorId`
    /// is stable for the model's lifetime (one Parameter, one id — issue
    /// anti-pattern #11). The earlier candle-keyed caches were a trap on
    /// Vulkan: the decode methods bridged each weight via `kt_logits_to_candle`
    /// *per call*, minting a fresh candle `TensorId` every token, so the cache
    /// MISSED every token → re-extract + re-upload the full weight set (~1
    /// GB/token incl. the 778 MB lm_head) into NEW buffers that accumulated
    /// unbounded. That single bug caused both the 25x decode slowdown (16 →
    /// 0.6 tok/s) and the OOM. Keying on the stable kt id uploads each weight
    /// exactly once and extracts bytes straight from kt storage — no candle
    /// copy.
    ///
    /// This field must drop before `vulkan_device`: `VulkanBuffer` owns raw
    /// memory that must be freed before the logical Vulkan device is destroyed.
    pub(super) weight_cache_kt:
        Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    pub(super) bf16_packed_weight_cache_kt:
        Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    /// Vulkan device (owned, not from candle-core).
    ///
    /// `Arc` rather than `Box` so a `CustomOp1` impl that wants to dispatch
    /// a Vulkan kernel from inside `cpu_fwd` can capture a refcounted
    /// handle to the device — the candle CustomOp trait requires the op
    /// state to be `'static + Send + Sync`, which a borrow off `&self`
    /// can never satisfy.
    pub(super) vulkan_device: Option<Arc<kiln_vulkan_kernel::VulkanDevice>>,
}

impl VulkanBackend {
    pub fn training_capabilities_static() -> TrainingCapabilities {
        vulkan_training::training_capabilities_static()
    }

    pub fn new(device: kiln_tensor::Device) -> Self {
        let config = VulkanRuntimeConfig::from_env();

        let vulkan_device = vulkan_device::new_backend_device();

        // Advertise `kt::Device::Vulkan(0)` when the logical device is up,
        // matching what `for_device_kt` callers would have constructed.
        // When the device failed to come up we still need a sensible kt
        // identity for the BackendRuntime accessor; the CPU fallback path
        // returns `kt::Device::Cpu` so trait callers consistently see "no
        // Vulkan" without a separate predicate. (#1082)
        let device_kt = if vulkan_device.is_some() {
            kiln_tensor::Device::Vulkan(0)
        } else {
            device
        };

        Self {
            device_kt,
            resident_activation_registry: super::vulkan_residency::new_resident_activation_registry(
            ),
            gdn_enabled: config.gdn_enabled,
            gdn_prefill_in_proj_enabled: config.gdn_prefill_in_proj_enabled,
            gdn_gates_enabled: config.gdn_gates_enabled,
            gdn_gated_rms_norm_enabled: config.gdn_gated_rms_norm_enabled,
            gdn_full_chunk_forward_enabled: config.gdn_full_chunk_forward_enabled,
            fused_conv1d_update_enabled: config.fused_conv1d_update_enabled,
            fused_conv1d_prefill_enabled: config.fused_conv1d_prefill_enabled,
            conv1d_prefill_single_submit_enabled: config.conv1d_prefill_single_submit_enabled,
            gdn_forward_sub_enabled: config.gdn_forward_sub_enabled,
            gdn_decode_fused_enabled: config.gdn_decode_fused_enabled,
            gdn_recurrent_unexpanded_qk_enabled: config.gdn_recurrent_unexpanded_qk_enabled,
            gdn_recurrent_qk_norm_unexpanded_enabled: config
                .gdn_recurrent_qk_norm_unexpanded_enabled,
            linear_decode_enabled: config.linear_decode_enabled,
            linear_argmax_batch_enabled: config.linear_argmax_batch_enabled,
            full_attn_qkv_enabled: config.full_attn_qkv_enabled,
            paged_attn_decode_batch_enabled: config.paged_attn_decode_batch_enabled,
            mlp_decode_enabled: config.mlp_decode_enabled,
            mlp_gate_up_enabled: config.mlp_gate_up_enabled,
            mlp_bf16_gate_up_f32_down_enabled: config.mlp_bf16_gate_up_f32_down_enabled,
            bf16_packed_linear_weights_enabled: config.bf16_packed_linear_weights_enabled,
            bf16_packed_gdn_in_proj_weights_enabled: config.bf16_packed_gdn_in_proj_weights_enabled,
            bf16_packed_full_attn_qkv_weights_enabled: config
                .bf16_packed_full_attn_qkv_weights_enabled,
            bf16_packed_mlp_decode_weights_enabled: config.bf16_packed_mlp_decode_weights_enabled,
            weight_prewarm_enabled: config.weight_prewarm_enabled,
            recurrent_state_residency_enabled: config.recurrent_state_residency_enabled,
            resident_decode_enabled: config.resident_decode_enabled,
            decode_resident_pool: OnceLock::new(),
            vk_paged_kv_cache: OnceLock::new(),
            seeded_full_attn_layers: Mutex::new(HashSet::new()),
            seeded_resident_decode_rows: Mutex::new(HashSet::new()),
            linear_attn_recurrent_state_kt: Mutex::new(HashMap::new()),
            linear_attn_conv_state_kt: Mutex::new(HashMap::new()),
            seeded_linear_attn_layers_kt: Mutex::new(HashSet::new()),
            last_resident_start_pos: Mutex::new(None),
            resident_scratch: Mutex::new(HashMap::new()),
            weight_cache_kt: Mutex::new(HashMap::new()),
            bf16_packed_weight_cache_kt: Mutex::new(HashMap::new()),
            vulkan_device,
        }
    }

    pub(super) fn has_vulkan(&self) -> bool {
        self.vulkan_device.is_some()
    }

    /// Direct accessor for the owned `VulkanDevice`. Returns `None`
    /// when device initialization failed (CPU fallback path); callers
    /// that need device-resident work must short-circuit on `None`.
    pub fn vulkan_device(&self) -> Option<&Arc<kiln_vulkan_kernel::VulkanDevice>> {
        self.vulkan_device.as_ref()
    }

    /// kt-native f32 weight buffer cache: keys the buffer cache on the kt
    /// `TensorId` and uploads on cache miss.
    pub fn cached_f32_weight_buffer_kt(
        &self,
        weight: &kiln_tensor::Tensor,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        vulkan_weights::cached_f32_weight_buffer_kt(self, weight)
    }

    /// kt-native bf16-packed weight buffer cache keyed on the kt `TensorId`.
    pub fn cached_bf16_packed_weight_buffer_kt(
        &self,
        weight: &kiln_tensor::Tensor,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        vulkan_weights::cached_bf16_packed_weight_buffer_kt(self, weight)
    }

    /// kt-native: whether to use the bf16-packed linear-weight decode path.
    pub(super) fn use_bf16_packed_linear_weight_kt(&self, weight: &kiln_tensor::Tensor) -> bool {
        vulkan_weights::use_bf16_packed_linear_weight_kt(self, weight)
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        // The `VulkanBuffer`s in these caches own raw device memory that must
        // be freed before `vulkan_device` is destroyed. Clear them explicitly
        // so the drop order is deterministic. (#1082: candle-keyed twins gone;
        // only the kt-keyed caches remain.)
        if let Ok(mut cache) = self.weight_cache_kt.lock() {
            cache.clear();
        }
        if let Ok(mut cache) = self.bf16_packed_weight_cache_kt.lock() {
            cache.clear();
        }
    }
}

impl BackendIdentity for VulkanBackend {
    fn runtime_name(&self) -> &'static str {
        if self.has_vulkan() { "vulkan" } else { "cpu" }
    }

    fn runtime_device(&self) -> kiln_tensor::Device {
        self.device_kt
    }

    fn runtime_as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl StartupBackend for VulkanBackend {}

impl ExternalYieldBackend for VulkanBackend {
    fn runtime_synchronize_external_yield(&self) -> Result<()> {
        let Some(device) = self.vulkan_device.as_ref() else {
            return Ok(());
        };
        device.synchronize_queue("external model yield")
    }
}

#[allow(clippy::too_many_arguments)]
impl ConvBackend for VulkanBackend {
    fn runtime_supports_causal_conv1d_update(&self) -> bool {
        vulkan_conv1d::supports_causal_conv1d_update(self)
    }

    fn runtime_supports_causal_conv1d_prefill(&self) -> bool {
        vulkan_conv1d::supports_causal_conv1d_prefill(self)
    }

    fn runtime_causal_conv1d_update(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state_kt: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_conv1d::causal_conv1d_update(self, x, weight, conv_state_kt, kernel_size)
    }

    fn runtime_causal_conv1d_prefill(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state_kt: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_conv1d::causal_conv1d_prefill(self, x, weight, conv_state_kt, kernel_size)
    }
}

#[allow(clippy::too_many_arguments)]
impl SamplingBackend for VulkanBackend {
    fn runtime_supports_linear_decode_argmax(&self) -> bool {
        vulkan_linear::supports_linear_decode_argmax(self)
    }

    fn runtime_linear_decode_argmax(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<u32>> {
        vulkan_linear::linear_decode_argmax(self, x, weight_t)
    }

    fn runtime_supports_linear_decode_argmax_batch(&self) -> bool {
        vulkan_linear::supports_linear_decode_argmax_batch(self)
    }

    fn runtime_supports_linear_decode_sample(&self, top_k: u32) -> bool {
        vulkan_linear::supports_linear_decode_sample(self, top_k)
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
        vulkan_linear::linear_decode_sample(
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
        vulkan_linear::supports_linear_decode_sample_batch(self, top_k, temperatures)
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
        vulkan_linear::linear_decode_sample_batch(
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

    fn runtime_linear_decode_argmax_batch(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<Vec<u32>>> {
        vulkan_linear::linear_decode_argmax_batch(self, x, weight_t)
    }
}

#[allow(clippy::too_many_arguments)]
impl OptimizerBackend for VulkanBackend {
    fn runtime_dispatch_sgd_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        lr: f32,
    ) -> Result<bool> {
        vulkan_training::dispatch_sgd_step(self, param, grad, lr)
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
        vulkan_training::dispatch_adamw_step(
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

    fn runtime_dispatch_muon_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        momentum: &kiln_tensor::Tensor,
        lr: f32,
        momentum_coef: f32,
        nesterov: bool,
        ns_iters: u32,
        weight_decay: f32,
    ) -> Result<bool> {
        vulkan_training::dispatch_muon_step(
            self,
            param,
            grad,
            momentum,
            lr,
            momentum_coef,
            nesterov,
            ns_iters,
            weight_decay,
        )
    }
}

impl PagedKvBackend for VulkanBackend {}

#[allow(clippy::too_many_arguments)]
impl AttentionBackend for VulkanBackend {
    fn runtime_supports_flash_attn_prefill(&self) -> bool {
        // The flash_attn.comp placeholder is replaced by the
        // sdpa_prefill_f32.comp kernel landed in commit dc4664ed.
        // Default-enabled now that the kernel is parity-tested at
        // multiple shapes (including Qwen3.5-4B head_dim=128) and
        // bounded in dispatch size (workgroup_count = T × H × B
        // is well under any reasonable Vulkan limit for production
        // shapes). Set `KILN_VULKAN_SDPA=0` to opt out.
        if !self.has_vulkan() {
            return false;
        }
        kiln_core::env_flag::env_flag("KILN_VULKAN_SDPA", true)
    }

    fn runtime_supports_flash_attn_prefill_head_major(&self) -> bool {
        // Not implemented — return false so callers keep their preamble.
        false
    }

    fn runtime_supports_flash_attn_paged_decode(&self) -> bool {
        self.has_vulkan() && self.paged_attn_decode_batch_enabled
    }

    fn runtime_flash_attn_prefill(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_attention::flash_attn_prefill(self, q, k, v, softmax_scale, causal)
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
        vulkan_attention::flash_attn_paged_decode(
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
        vulkan_attention::flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
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

// #1082 DoD-101/102: BackendRuntime decode methods flipped to kt; metal/vulkan impls need matching flip when their builds are restored.
#[allow(clippy::too_many_arguments)]
impl GdnBackend for VulkanBackend {
    fn runtime_supports_gdn_forward_substitution(&self) -> bool {
        vulkan_gdn::supports_gdn_forward_substitution(self)
    }

    fn runtime_supports_gdn_recurrent_step(&self) -> bool {
        vulkan_gdn::supports_gdn_recurrent_step(self)
    }

    fn runtime_supports_gdn_recurrent_prefill_native_head_last(&self) -> bool {
        vulkan_gdn::supports_gdn_recurrent_prefill_native_head_last(self)
    }

    fn runtime_supports_gdn_recurrent_qk_norm_prefill_native_head_last(&self) -> bool {
        vulkan_gdn::supports_gdn_recurrent_qk_norm_prefill_native_head_last(self)
    }

    fn runtime_supports_gdn_chunk_prep(&self) -> bool {
        vulkan_gdn::supports_gdn_chunk_prep(self)
    }

    fn runtime_supports_gdn_chunk_scan(&self) -> bool {
        vulkan_gdn::supports_gdn_chunk_scan(self)
    }

    fn runtime_supports_gdn_full_chunk_forward(&self) -> bool {
        vulkan_gdn::supports_gdn_full_chunk_forward(self)
    }

    fn runtime_supports_gdn_gates(&self) -> bool {
        vulkan_gdn::supports_gdn_gates(self)
    }

    fn runtime_supports_gdn_gated_rms_norm(&self) -> bool {
        vulkan_gdn::supports_gdn_gated_rms_norm(self)
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
        vulkan_gdn::gdn_in_proj_decode(
            self,
            x,
            in_proj_qkv_t,
            in_proj_z_t,
            in_proj_a_t,
            in_proj_b_t,
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
        state_kt: &mut kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_gdn::gdn_decode_gates_recurrent_rmsnorm(
            self, q, k, v, a, b, a_log, dt_bias, state_kt, z, weight, eps,
        )
    }

    fn runtime_gdn_forward_substitution(
        &self,
        a_strict: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_gdn::gdn_forward_substitution(self, a_strict, v_prime, beta)
    }

    fn runtime_gdn_solve_tri_transpose(
        &self,
        a_strict: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        dw: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_gdn::gdn_solve_tri_transpose(self, a_strict, beta, dw)
    }

    fn runtime_gdn_recurrent_prefill_native_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state_kt: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_gdn::gdn_recurrent_prefill_native_head_last(self, q, k, v, beta, g, state_kt)
    }

    fn runtime_gdn_recurrent_qk_norm_prefill_native_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state_kt: &mut kiln_tensor::Tensor,
        q_scale: f64,
        qk_eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_gdn::gdn_recurrent_qk_norm_prefill_native_head_last(
            self, q, k, v, beta, g, state_kt, q_scale, qk_eps,
        )
    }

    fn runtime_gdn_recurrent_step(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state_kt: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_gdn::gdn_recurrent_step(self, q, k, v, beta, g, state_kt)
    }

    fn runtime_gdn_chunkwise_forward(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state_kt: &mut kiln_tensor::Tensor,
        chunk_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_gdn::gdn_chunkwise_forward(self, q, k, v, beta, g, state_kt, chunk_size)
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
        vulkan_gdn::gdn_chunk_prep(self, g, v, kkt, qkt, ks_entry, q_s)
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
        vulkan_gdn::gdn_chunk_scan(
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
        state_kt: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_gdn::gdn_full_chunk_forward(self, g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state_kt)
    }

    fn runtime_gdn_gates(
        &self,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        vulkan_gdn::gdn_gates(self, a, b, a_log, dt_bias)
    }

    fn runtime_gdn_gated_rms_norm(
        &self,
        x: &kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_gdn::gdn_gated_rms_norm(self, x, z, weight, eps)
    }
}

#[allow(clippy::too_many_arguments)]
impl LinearBackend for VulkanBackend {
    fn runtime_supports_matmul_request(
        &self,
        req: &super::capability::MatmulRequest,
    ) -> super::capability::Support {
        let Some(rank) = matmul_request_support_rank(req) else {
            return super::capability::Support::Unsupported;
        };
        matmul_support_from_native(
            matches!(req.epilogue, super::capability::MatmulEpilogue::Identity)
                && (req.lhs_dtype == kiln_tensor::DType::F32
                    || req.lhs_dtype == kiln_tensor::DType::BF16 && rank > 2),
        )
    }

    fn runtime_matmul(
        &self,
        req: &super::capability::MatmulRequest,
        lhs: &kiln_tensor::Tensor,
        rhs: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_linear::matmul(self, req, lhs, rhs)
    }

    fn runtime_lora_delta_resident(
        &self,
        _x: &kiln_tensor::Tensor,
        _a: &kiln_tensor::Tensor,
        _b: &kiln_tensor::Tensor,
        _scale: f32,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // (#1082) Decline. This hook previously dispatched the on-device
        // LoRA delta through `VulkanLoraOp` (a `candle_core::CustomOp3`)
        // purely so candle's `loss.backward()` could recover grad_A /
        // grad_B. With the kt autograd tape (`kiln_autograd`) as the sole
        // grad producer, that candle autograd island is gone — the forward
        // LoRA delta is recorded onto the tape by the portable kt
        // `compute_lora_delta` path in forward.rs, and `Tape::backward()`
        // produces the gradients. Returning `Ok(None)` routes the caller to
        // that kt-recorded path.
        Ok(None)
    }

    fn runtime_linear_decode(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_linear::linear_decode(self, x, weight_t)
    }

    fn runtime_linear_prefill_apply(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_linear::linear_prefill_apply(self, x, weight_t)
    }

    fn runtime_linear_prefill_apply_offset(
        &self,
        x: &kiln_tensor::Tensor,
        full_weight_t: &kiln_tensor::Tensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_linear::linear_prefill_apply_offset(self, x, full_weight_t, chunk_start, chunk_len)
    }

    fn runtime_prewarm_decode_weights(&self, weights: &GpuWeights) -> Result<()> {
        vulkan_weights::prewarm_decode_weights(self, weights)
    }

    fn runtime_drop_uploaded_bf16_weights(
        &self,
        weights: &mut crate::forward::GpuWeights,
        device: &kiln_tensor::Device,
    ) -> Result<usize> {
        vulkan_weights::drop_uploaded_bf16_weights(self, weights, device)
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
        vulkan_dense::full_attn_qkv_decode(self, x, q_weight_t, k_weight_t, v_weight_t)
    }

    fn runtime_mlp_gate_up_decode(
        &self,
        x: &kiln_tensor::Tensor,
        gate_weight_t: &kiln_tensor::Tensor,
        up_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_dense::mlp_gate_up_decode(self, x, gate_weight_t, up_weight_t)
    }

    fn runtime_mlp_decode(
        &self,
        x: &kiln_tensor::Tensor,
        gate_weight_t: &kiln_tensor::Tensor,
        up_weight_t: &kiln_tensor::Tensor,
        down_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_dense::mlp_decode(self, x, gate_weight_t, up_weight_t, down_weight_t)
    }
}

fn vulkan_resident_activation_resource(
    tensor: &kiln_tensor::Tensor,
    state: super::residency::ResidentResourceState,
    resident_byte_len: usize,
) -> super::residency::ResidentResource {
    let resident_dtype = if tensor.dtype() == kiln_tensor::DType::BF16 {
        kiln_tensor::DType::BF16
    } else {
        kiln_tensor::DType::F32
    };
    super::residency::ResidentResource::from_tensor_for_backend(
        tensor,
        super::residency::resident_backend_for_runtime("vulkan", tensor.device()),
        super::residency::ResidentResourceFamily::Activation,
        super::residency::ResidentOwnership::RegistryOwned,
    )
    .with_state(state)
    .with_replay_stability(super::residency::ReplayStability::StableAcrossReplay)
    .with_resident_allocation(resident_dtype, resident_byte_len, resident_byte_len)
}

impl super::residency::ResidentRegistry for VulkanBackend {
    fn register_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) -> Result<Option<super::residency::ResidentResource>> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return Ok(None);
        }
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(None);
        };
        // (#1082) kt-native: the residency registry is keyed on the kt
        // `TensorId` directly; byte extraction reads straight from kt storage.
        let id = tensor.id();
        let existing = with_resident_registry(&self.resident_activation_registry, |cache| {
            cache.get_mut(&id).map(|entry| {
                entry.resource = entry
                    .resource
                    .clone()
                    .with_state(super::residency::ResidentResourceState::RegisteredClean);
                entry.resource.clone()
            })
        });
        if existing.is_some() {
            return Ok(existing);
        }
        // Encoding choice per dtype:
        //   - BF16 -> packed BF16 (2 bytes/elem), byte-compatible with
        //     every Vulkan kernel that uses `load_weight(idx)` to
        //     decode `data_w[idx >> 1]` as two BF16 lanes per u32.
        //     Required for the LoRA `lora_delta_resident` path and
        //     any future BF16-input training kernel.
        //   - All other dtypes -> F32 bytes (4 bytes/elem). This is
        //     what the existing boundary-state resolve path
        //     expects (`create_tensor_from_data` decodes F32 then
        //     casts).
        //
        // `resolve_resource` knows about both encodings and reconstructs
        // Tensors appropriately.
        let bytes = if tensor.dtype() == kiln_tensor::DType::BF16 {
            kt_tensor_to_packed_bf16_bytes_with_shape(tensor)?.0
        } else {
            kt_tensor_to_f32_bytes_with_shape(tensor)?.0
        };
        // Some Vulkan drivers reject zero-size buffer allocations; we
        // also have no use for a zero-byte registry entry. Bail
        // silently -- has_resident_activation will return false and
        // the caller falls through to its CPU path.
        if bytes.is_empty() {
            return Ok(None);
        }
        let device = vk_device.device();
        let device_local_mt = vk_device.device_local_mem_type();
        let host_visible_mt = vk_device.host_visible_mem_type();
        let queue = vk_device.queue();
        let queue_family = vk_device.queue_family_index();
        let buffer = kiln_vulkan_kernel::VulkanBuffer::create_device_local(
            device,
            device_local_mt,
            bytes.len() as u64,
        )
        .context("register_resident_activation: alloc buffer")?;
        kiln_vulkan_kernel::VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            queue_family,
            &buffer,
            &bytes,
        )
        .context("register_resident_activation: upload bytes")?;
        let buffer = Arc::new(buffer);
        // One-shot trace so the operator can confirm the activation
        // residency lifecycle is engaging during training without
        // per-call log spam. The first registration is the most
        // informative -- usually the embedding boundary at the
        // start of checkpointed_forward_backward.
        static FIRST_REGISTERED_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        FIRST_REGISTERED_LOGGED.get_or_init(|| {
            tracing::info!(
                tensor_dims = ?tensor.dims(),
                tensor_dtype = ?tensor.dtype(),
                bytes = bytes.len(),
                "VulkanBackend::register_resident_activation first call"
            );
        });
        let resource = vulkan_resident_activation_resource(
            tensor,
            super::residency::ResidentResourceState::RegisteredClean,
            bytes.len(),
        );
        with_resident_registry(&self.resident_activation_registry, |cache| {
            cache.insert(id, ResidentActivationEntry::new(buffer, resource.clone()));
        });
        Ok(Some(resource))
    }

    fn update_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) -> Result<Option<super::residency::ResidentResource>> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return Ok(None);
        }
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(None);
        };
        // (#1082) kt-native: registry keyed on the kt `TensorId` directly.
        let id = tensor.id();
        let buffer = with_resident_registry(&self.resident_activation_registry, |cache| {
            cache.get(&id).map(|entry| Arc::clone(&entry.buffer))
        });
        let Some(buffer) = buffer else {
            return Ok(None);
        };
        // Same encoding choice as register_resource.
        let bytes = if tensor.dtype() == kiln_tensor::DType::BF16 {
            kt_tensor_to_packed_bf16_bytes_with_shape(tensor)?.0
        } else {
            kt_tensor_to_f32_bytes_with_shape(tensor)?.0
        };
        if bytes.is_empty() {
            return Ok(None);
        }
        anyhow::ensure!(
            bytes.len() as u64 == buffer.size(),
            "update_resident_activation: tensor bytes ({}) != buffer size ({})",
            bytes.len(),
            buffer.size(),
        );
        kiln_vulkan_kernel::VulkanBuffer::upload_data(
            vk_device.device(),
            vk_device.host_visible_mem_type(),
            vk_device.queue(),
            vk_device.queue_family_index(),
            &buffer,
            &bytes,
        )
        .context("update_resident_activation: re-upload bytes")?;
        let resource = vulkan_resident_activation_resource(
            tensor,
            super::residency::ResidentResourceState::DirtyDevice,
            bytes.len(),
        );
        with_resident_registry(&self.resident_activation_registry, |cache| {
            if let Some(entry) = cache.get_mut(&id) {
                entry.resource = resource.clone();
            }
        });
        Ok(Some(resource))
    }

    fn evict_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) {
        if family != super::residency::ResidentResourceFamily::Activation {
            return;
        }
        // (#1082) kt-native: registry keyed on the kt `TensorId` directly.
        let id = tensor.id();
        with_resident_registry(&self.resident_activation_registry, |cache| {
            cache.remove(&id);
        });
    }

    fn resident_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
    ) -> Option<super::residency::ResidentResource> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return None;
        }
        // (#1082) kt-native: registry keyed on the kt `TensorId` directly.
        let id = tensor.id();
        with_resident_registry(&self.resident_activation_registry, |cache| {
            cache.get(&id).map(|entry| entry.resource.clone())
        })
    }

    fn resolve_resource(
        &self,
        tensor: &kiln_tensor::Tensor,
        family: super::residency::ResidentResourceFamily,
        shape: &[usize],
        dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        if family != super::residency::ResidentResourceFamily::Activation {
            return Ok(None);
        }
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(None);
        };
        // (#1082) kt-native: registry keyed on the kt `TensorId`; the result
        // is reconstructed directly as a kt tensor of `dtype`.
        let id = tensor.id();
        let buffer = with_resident_registry(&self.resident_activation_registry, |cache| {
            cache.get(&id).map(|entry| Arc::clone(&entry.buffer))
        });
        let Some(buffer) = buffer else {
            return Ok(None);
        };
        let bytes = kiln_vulkan_kernel::VulkanBuffer::read_back(
            vk_device.device(),
            vk_device.host_visible_mem_type(),
            vk_device.queue(),
            vk_device.queue_family_index(),
            &buffer,
        )
        .context("resolve_resident_activation: read_back")?;
        // Inverse of the encoding choice in register_resource.
        // BF16 registry entries hold packed bf16 (2 bytes/elem);
        // other dtypes hold F32 bytes. Reconstruct BF16 by bit-expanding each
        // 16-bit lane into f32 (`bits << 16`) and then casting back to BF16.
        let resolved = if dtype == kiln_tensor::DType::BF16 {
            anyhow::ensure!(
                bytes.len() % 2 == 0,
                "resolve_resident_activation BF16: buffer byte count {} is not a multiple of 2",
                bytes.len()
            );
            let elem_count: usize = shape.iter().product();
            let stored = bytes.len() / 2;
            anyhow::ensure!(
                stored >= elem_count,
                "resolve_resident_activation BF16: buffer holds {} bf16 elements, \
                 expected at least {} for shape {:?}",
                stored,
                elem_count,
                shape,
            );
            let mut f32_data = Vec::with_capacity(elem_count);
            for i in 0..elem_count {
                let lo = bytes[i * 2] as u32;
                let hi = bytes[i * 2 + 1] as u32;
                let bf16_bits = (hi << 8) | lo;
                f32_data.push(f32::from_bits(bf16_bits << 16));
            }
            kiln_tensor::Tensor::from_vec(f32_data, shape.to_vec())
                .map_err(|e| anyhow::anyhow!("resolve_resident_activation BF16: from_vec: {e}"))?
                .to_dtype(kiln_tensor::DType::BF16)
                .map_err(|e| anyhow::anyhow!("resolve_resident_activation BF16: to_dtype: {e}"))?
        } else {
            kt_tensor_from_f32_bytes(&bytes, shape, dtype)
                .context("resolve_resident_activation: create_tensor_from_data")?
        };
        Ok(Some(resolved))
    }
}

#[allow(clippy::too_many_arguments)]
impl ResidencyBackend for VulkanBackend {
    fn runtime_enter_gdn_recurrent_resident_state_scope(&self) -> bool {
        if !self.recurrent_state_residency_enabled || !self.has_vulkan() || !self.gdn_enabled {
            return false;
        }
        enter_recurrent_state_resident_scope();
        true
    }

    fn runtime_exit_gdn_recurrent_resident_state_scope(&self) {
        if self.recurrent_state_residency_enabled {
            exit_recurrent_state_resident_scope();
        }
    }

    fn runtime_materialize_gdn_recurrent_resident_state(
        &self,
        state_kt: &mut kiln_tensor::Tensor,
    ) -> Result<()> {
        if !self.recurrent_state_residency_enabled {
            return Ok(());
        }
        // (#1082) kt-native: the recurrent-state resident cache is keyed on
        // the kt `TensorId` directly (stable across the state's lifetime), and the
        // materialized state is written back into the kt arg in place.
        let state_id = state_kt.id();
        let resident_state = take_recurrent_state_resident_buffer(state_id);
        let Some(resident_state) = resident_state else {
            return Ok(());
        };
        let state_dims = state_kt.dims().to_vec();
        let state_dtype = state_kt.dtype();

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let data = kiln_vulkan_kernel::VulkanBuffer::read_back(
            vk_device.device(),
            vk_device.host_visible_mem_type(),
            vk_device.queue(),
            vk_device.queue_family_index(),
            &resident_state,
        )
        .context("failed to materialize resident GDN recurrent state")?;
        *state_kt = kt_tensor_from_f32_bytes(&data, &state_dims, state_dtype)?;
        Ok(())
    }

    fn runtime_evict_gdn_recurrent_resident_state(&self, state: &kiln_tensor::Tensor) {
        if !self.recurrent_state_residency_enabled {
            return;
        }
        // (#1082) kt-native: key the cache on the kt `TensorId` directly.
        let state_id = state.id();
        remove_recurrent_state_resident_buffer(state_id);
    }

    fn runtime_has_gdn_recurrent_resident_state(&self, state: &kiln_tensor::Tensor) -> bool {
        if !self.recurrent_state_residency_enabled {
            return false;
        }
        // (#1082) kt-native: key the cache on the kt `TensorId` directly.
        let state_id = state.id();
        contains_recurrent_state_resident_buffer(state_id)
    }

    fn runtime_supports_resident_activation(&self) -> bool {
        // Vulkan implements resident activation through its concrete
        // ResidentRegistry. The registry can still decline at call time when
        // the process has no logical Vulkan device.
        true
    }

    fn runtime_register_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        super::residency::ResidentRegistry::register_resource(
            self,
            tensor,
            super::residency::ResidentResourceFamily::Activation,
        )
        .map(|_| ())
    }

    fn runtime_evict_resident_activation(&self, tensor: &kiln_tensor::Tensor) {
        super::residency::ResidentRegistry::evict_resource(
            self,
            tensor,
            super::residency::ResidentResourceFamily::Activation,
        );
    }

    fn runtime_update_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        super::residency::ResidentRegistry::update_resource(
            self,
            tensor,
            super::residency::ResidentResourceFamily::Activation,
        )
        .map(|_| ())
    }

    fn runtime_has_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> bool {
        super::residency::ResidentRegistry::has_resident_resource(
            self,
            tensor,
            super::residency::ResidentResourceFamily::Activation,
        )
    }

    fn runtime_resolve_resident_activation(
        &self,
        tensor: &kiln_tensor::Tensor,
        shape: &[usize],
        dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        super::residency::ResidentRegistry::resolve_resource(
            self,
            tensor,
            super::residency::ResidentResourceFamily::Activation,
            shape,
            dtype,
        )
    }

    fn runtime_assemble_gdn_recurrent_resident_batch_rows(
        &self,
        rows: &[&kiln_tensor::Tensor],
        batch: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        // kt guard read directly off the kt args before the bridge.
        if !self.recurrent_state_residency_enabled
            || !recurrent_state_resident_scope_active()
            || !self.has_vulkan()
            || rows.is_empty()
        {
            return Ok(false);
        }
        // (#1082) kt-native: `rows`/`batch` are already kt; the
        // recurrent-state resident cache is keyed on the kt `TensorId` directly.
        let Ok((batch_rows, heads, dk, dv)) = batch.dims4() else {
            return Ok(false);
        };
        if rows.len() != batch_rows {
            return Ok(false);
        }
        for row in rows {
            let Ok((row_batch, row_heads, row_dk, row_dv)) = row.dims4() else {
                return Ok(false);
            };
            if (row_batch, row_heads, row_dk, row_dv) != (1, heads, dk, dv)
                || row.dtype() != batch.dtype()
                || !matches!(row.device(), kiln_tensor::Device::Cpu)
            {
                return Ok(false);
            }
        }

        let row_buffers = recurrent_state_resident_buffers_for(rows.iter().map(|row| row.id()));
        let Some(row_buffers) = row_buffers else {
            return Ok(false);
        };
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let batch_buffer = kiln_vulkan_kernel::kernels::copy_gdn_recurrent_state_rows_to_batch(
            vk_device,
            &row_buffers,
        )
        .context("failed to assemble resident GDN recurrent batch rows")?;
        insert_recurrent_state_resident_buffer(batch.id(), batch_buffer);
        Ok(true)
    }

    fn runtime_scatter_gdn_recurrent_resident_batch_rows(
        &self,
        batch: &kiln_tensor::Tensor,
        destinations: &mut [&mut kiln_tensor::Tensor],
    ) -> Result<bool> {
        // kt guard read directly off the kt args before the bridge.
        if !self.recurrent_state_residency_enabled
            || !recurrent_state_resident_scope_active()
            || !self.has_vulkan()
            || destinations.is_empty()
        {
            return Ok(false);
        }
        // (#1082) kt-native: `batch`/`destinations` are already kt; the
        // residency cache is keyed on the kt `TensorId` directly.
        let Ok((batch_rows, heads, dk, dv)) = batch.dims4() else {
            return Ok(false);
        };
        if destinations.len() != batch_rows {
            return Ok(false);
        }
        let batch_buffer = get_recurrent_state_resident_buffer(batch.id());
        let Some(batch_buffer) = batch_buffer else {
            return Ok(false);
        };
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let row_buffers = kiln_vulkan_kernel::kernels::split_gdn_recurrent_state_batch_rows(
            vk_device,
            &batch_buffer,
            batch_rows,
        )
        .context("failed to scatter resident GDN recurrent batch rows")?;
        if row_buffers.len() != destinations.len() {
            return Ok(false);
        }

        let mut staged_rows = Vec::with_capacity(destinations.len());
        for (row_idx, (dst, row_buffer)) in
            destinations.iter().zip(row_buffers.into_iter()).enumerate()
        {
            // kt id of the current destination keys the cache eviction.
            let old_id = dst.id();
            let placeholder = batch.narrow(0, row_idx, 1)?.contiguous()?;
            if placeholder.dtype() != batch.dtype()
                || placeholder.dims() != [1, heads, dk, dv]
                || !matches!(placeholder.device(), kiln_tensor::Device::Cpu)
            {
                return Ok(false);
            }
            // kt id of the newly-written destination keys the insert.
            let new_id = placeholder.id();
            staged_rows.push((old_id, new_id, placeholder, row_buffer));
        }

        for (dst, (old_id, new_id, placeholder, row_buffer)) in
            destinations.iter_mut().zip(staged_rows.into_iter())
        {
            **dst = placeholder;
            replace_recurrent_state_resident_buffer(old_id, new_id, row_buffer);
        }
        remove_recurrent_state_resident_buffer(batch.id());

        Ok(true)
    }

    fn runtime_assemble_linear_attn_gdn_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
    ) -> Result<bool> {
        VulkanBackend::assemble_linear_attn_gdn_state_batch_kt(self, row_keys, batch_key)
    }

    fn runtime_scatter_linear_attn_gdn_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
    ) -> Result<bool> {
        VulkanBackend::scatter_linear_attn_gdn_state_batch_kt(self, batch_key, row_keys)
    }

    fn runtime_seed_linear_attn_gdn_state_kt(
        &self,
        recurrent: &kiln_tensor::Tensor,
        conv: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        VulkanBackend::seed_linear_attn_gdn_state_kt(self, recurrent, conv)
    }

    fn runtime_has_linear_attn_gdn_state_kt(&self, key: kiln_tensor::TensorId) -> bool {
        VulkanBackend::has_linear_attn_gdn_state_kt(self, key)
    }
}

impl TrainingLossBackend for VulkanBackend {
    fn runtime_training_capabilities(&self) -> TrainingCapabilities {
        Self::training_capabilities_static()
    }

    fn runtime_training_precision_policy(&self) -> TrainingPrecisionPolicy {
        vulkan_training::training_precision_policy()
    }
}

impl BackendRuntime for VulkanBackend {}

impl ReplayBackend for VulkanBackend {
    fn runtime_decode_resident_pool_ready(
        &self,
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> bool {
        if !self.has_vulkan() || !self.resident_decode_enabled {
            return false;
        }
        self.decode_resident_pool(max_hidden, max_intermediate, max_batch)
            .is_some()
    }

    fn runtime_supports_resident_decode(&self) -> bool {
        // The Vulkan-resident decode path (docs/vk_resident_decode_plan.md)
        // applies whenever the logical device is up. The runtime pool
        // feasibility check (the "fall back if the device can't fit even
        // the minimum pool" rule in gate (b)) is enforced later, the
        // first time a resident decode actually requests a buffer.
        self.has_vulkan() && self.resident_decode_enabled
    }
}

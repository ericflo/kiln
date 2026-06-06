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
    contains_recurrent_state_resident_buffer, enter_recurrent_state_resident_scope,
    exit_recurrent_state_resident_scope, get_recurrent_state_resident_buffer,
    insert_recurrent_state_resident_buffer, recurrent_state_resident_buffers_for,
    recurrent_state_resident_scope_active, remove_recurrent_state_resident_buffer,
    replace_recurrent_state_resident_buffer, take_recurrent_state_resident_buffer,
    with_resident_registry,
};
use super::vulkan_tensor_bridge::{
    kt_tensor_from_f32_bytes, kt_tensor_to_f32_bytes_with_shape,
    kt_tensor_to_packed_bf16_bytes_with_shape, upload_gdn_chunkwise_inputs_from_cpu_bytes_vk,
    vk_f32_tensors_to_cpu_tensors_batched_vk,
};
use super::{
    vulkan_attention, vulkan_device, vulkan_linear, vulkan_training, vulkan_weights, BackendRuntime,
    TrainingCapabilities, TrainingPrecisionPolicy,
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
    /// Cached at construction: reading env vars per decode step × 24 GDN layers
    /// shows up in decode NVTX captures. Env vars don't change at runtime.
    gdn_enabled: bool,
    gdn_prefill_in_proj_enabled: bool,
    gdn_gates_enabled: bool,
    gdn_gated_rms_norm_enabled: bool,
    gdn_full_chunk_forward_enabled: bool,
    fused_conv1d_update_enabled: bool,
    fused_conv1d_prefill_enabled: bool,
    conv1d_prefill_single_submit_enabled: bool,
    gdn_forward_sub_enabled: bool,
    gdn_decode_fused_enabled: bool,
    gdn_recurrent_unexpanded_qk_enabled: bool,
    gdn_recurrent_qk_norm_unexpanded_enabled: bool,
    pub(super) linear_decode_enabled: bool,
    pub(super) linear_argmax_batch_enabled: bool,
    full_attn_qkv_enabled: bool,
    pub(super) paged_attn_decode_batch_enabled: bool,
    mlp_decode_enabled: bool,
    mlp_gate_up_enabled: bool,
    mlp_bf16_gate_up_f32_down_enabled: bool,
    pub(super) bf16_packed_linear_weights_enabled: bool,
    pub(super) bf16_packed_gdn_in_proj_weights_enabled: bool,
    pub(super) bf16_packed_full_attn_qkv_weights_enabled: bool,
    pub(super) bf16_packed_mlp_decode_weights_enabled: bool,
    pub(super) weight_prewarm_enabled: bool,
    recurrent_state_residency_enabled: bool,
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
    pub(super) decode_resident_pool:
        OnceLock<Option<Arc<kiln_vulkan_kernel::DecodeResidentPool>>>,
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

fn fused_gdn_resident_state_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_RESIDENT_STATE").is_err()
    })
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

// #1082 DoD-101/102: BackendRuntime decode methods flipped to kt; metal/vulkan impls need matching flip when their builds are restored.
impl BackendRuntime for VulkanBackend {
    fn name(&self) -> &'static str {
        if self.has_vulkan() {
            "vulkan"
        } else {
            "cpu"
        }
    }

    fn device(&self) -> kiln_tensor::Device {
        self.device_kt
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn training_capabilities(&self) -> TrainingCapabilities {
        Self::training_capabilities_static()
    }

    fn training_precision_policy(&self) -> TrainingPrecisionPolicy {
        vulkan_training::training_precision_policy()
    }

    fn decode_resident_pool_ready(
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

    fn supports_resident_decode(&self) -> bool {
        // The Vulkan-resident decode path (docs/vk_resident_decode_plan.md)
        // applies whenever the logical device is up. The runtime pool
        // feasibility check (the "fall back if the device can't fit even
        // the minimum pool" rule in gate (b)) is enforced later, the
        // first time a resident decode actually requests a buffer.
        self.has_vulkan() && self.resident_decode_enabled
    }

    fn supports_flash_attn_prefill(&self) -> bool {
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

    fn supports_flash_attn_prefill_head_major(&self) -> bool {
        // Not implemented — return false so callers keep their preamble.
        false
    }

    fn supports_flash_attn_paged_decode(&self) -> bool {
        self.has_vulkan() && self.paged_attn_decode_batch_enabled
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        // solve_tri is experimental: shared-memory layout not yet validated
        // against CPU parity, and may exceed maxComputeSharedMemorySize on many
        // GPUs. Opt-in only via KILN_ENABLE_VULKAN_GDN_FORWARD_SUB.
        self.has_vulkan() && self.gdn_forward_sub_enabled
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_recurrent_prefill_native_head_last(&self) -> bool {
        self.has_vulkan() && self.gdn_recurrent_unexpanded_qk_enabled
    }

    fn supports_gdn_recurrent_qk_norm_prefill_native_head_last(&self) -> bool {
        self.has_vulkan() && self.gdn_recurrent_qk_norm_unexpanded_enabled
    }

    fn enter_gdn_recurrent_resident_state_scope(&self) -> bool {
        if !self.recurrent_state_residency_enabled || !self.has_vulkan() || !self.gdn_enabled {
            return false;
        }
        enter_recurrent_state_resident_scope();
        true
    }

    fn exit_gdn_recurrent_resident_state_scope(&self) {
        if self.recurrent_state_residency_enabled {
            exit_recurrent_state_resident_scope();
        }
    }

    fn materialize_gdn_recurrent_resident_state(
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

    fn evict_gdn_recurrent_resident_state(&self, state: &kiln_tensor::Tensor) {
        if !self.recurrent_state_residency_enabled {
            return;
        }
        // (#1082) kt-native: key the cache on the kt `TensorId` directly.
        let state_id = state.id();
        remove_recurrent_state_resident_buffer(state_id);
    }

    fn has_gdn_recurrent_resident_state(&self, state: &kiln_tensor::Tensor) -> bool {
        if !self.recurrent_state_residency_enabled {
            return false;
        }
        // (#1082) kt-native: key the cache on the kt `TensorId` directly.
        let state_id = state.id();
        contains_recurrent_state_resident_buffer(state_id)
    }

    fn supports_resident_activation(&self) -> bool {
        // Vulkan implements all three Phase 3.1 hooks against
        // RESIDENT_ACTIVATION_REGISTRY. Returns true even when the
        // process has no Vulkan device — `register_resident_activation`
        // will short-circuit to Ok(()) in that case, but the
        // capability semantics are still "this backend's registry is
        // wired non-trivially when conditions allow."
        true
    }

    /// Phase 3.1 hook: register a non-weight tensor as resident on the
    /// device. Uploads `tensor`'s bytes to a fresh `VulkanBuffer` and
    /// records the buffer under the tensor's `kiln_tensor::TensorId`. The caller
    /// owns lifecycle — Phase 3.2 will pair every register with a
    /// matching evict at the appropriate autograd boundary. Until then
    /// any caller using this hook must clean up explicitly to avoid
    /// leaking VRAM.
    fn register_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(());
        };
        // (#1082) kt-native: the residency registry is keyed on the kt
        // `TensorId` directly; byte extraction reads straight from kt storage.
        let id = tensor.id();
        let already_registered = with_resident_registry(|cache| cache.contains_key(&id));
        if already_registered {
            return Ok(());
        }
        // Encoding choice per dtype:
        //   - BF16 → packed BF16 (2 bytes/elem), byte-compatible with
        //     every Vulkan kernel that uses `load_weight(idx)` to
        //     decode `data_w[idx >> 1]` as two BF16 lanes per u32.
        //     Required for the LoRA `lora_delta_resident` path and
        //     any future BF16-input training kernel.
        //   - All other dtypes → F32 bytes (4 bytes/elem). This is
        //     what the existing boundary-state resolve path
        //     expects (`create_tensor_from_data` decodes F32 then
        //     casts).
        //
        // `resolve_resident_activation` knows about both encodings
        // and reconstructs Tensors appropriately.
        let bytes = if tensor.dtype() == kiln_tensor::DType::BF16 {
            kt_tensor_to_packed_bf16_bytes_with_shape(tensor)?.0
        } else {
            kt_tensor_to_f32_bytes_with_shape(tensor)?.0
        };
        // Some Vulkan drivers reject zero-size buffer allocations; we
        // also have no use for a zero-byte registry entry. Bail
        // silently — has_resident_activation will return false and
        // the caller falls through to its CPU path.
        if bytes.is_empty() {
            return Ok(());
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
        // informative — usually the embedding boundary at the
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
        with_resident_registry(|cache| {
            cache.insert(id, buffer);
        });
        Ok(())
    }

    fn evict_resident_activation(&self, tensor: &kiln_tensor::Tensor) {
        // (#1082) kt-native: registry keyed on the kt `TensorId` directly.
        let id = tensor.id();
        with_resident_registry(|cache| {
            cache.remove(&id);
        });
    }

    fn update_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> Result<()> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(());
        };
        // (#1082) kt-native: registry keyed on the kt `TensorId` directly.
        let id = tensor.id();
        let buffer = with_resident_registry(|cache| cache.get(&id).cloned());
        let Some(buffer) = buffer else {
            // Not registered — caller probably skipped the registration
            // path. No-op.
            return Ok(());
        };
        // Same encoding choice as register_resident_activation.
        let bytes = if tensor.dtype() == kiln_tensor::DType::BF16 {
            kt_tensor_to_packed_bf16_bytes_with_shape(tensor)?.0
        } else {
            kt_tensor_to_f32_bytes_with_shape(tensor)?.0
        };
        if bytes.is_empty() {
            return Ok(());
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
        Ok(())
    }

    fn has_resident_activation(&self, tensor: &kiln_tensor::Tensor) -> bool {
        // (#1082) kt-native: registry keyed on the kt `TensorId` directly.
        let id = tensor.id();
        with_resident_registry(|cache| cache.contains_key(&id))
    }

    fn resolve_resident_activation(
        &self,
        tensor: &kiln_tensor::Tensor,
        shape: &[usize],
        dtype: kiln_tensor::DType,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(None);
        };
        // (#1082) kt-native: registry keyed on the kt `TensorId`; the result
        // is reconstructed directly as a kt tensor of `dtype`.
        let id = tensor.id();
        let buffer = with_resident_registry(|cache| cache.get(&id).cloned());
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
        // Inverse of the encoding choice in register_resident_activation.
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

    fn dispatch_sgd_step(
        &self,
        param: &kiln_tensor::Tensor,
        grad: &kiln_tensor::Tensor,
        lr: f32,
    ) -> Result<bool> {
        vulkan_training::dispatch_sgd_step(self, param, grad, lr)
    }

    fn dispatch_adamw_step(
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

    fn lora_delta_resident(
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

    fn assemble_gdn_recurrent_resident_batch_rows(
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

    fn scatter_gdn_recurrent_resident_batch_rows(
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

        for (row_idx, (dst, row_buffer)) in destinations
            .iter_mut()
            .zip(row_buffers.into_iter())
            .enumerate()
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
            // Write the placeholder back into the kt destination.
            **dst = placeholder;
            // kt id of the newly-written destination keys the insert.
            let new_id = dst.id();
            replace_recurrent_state_resident_buffer(old_id, new_id, row_buffer);
        }
        remove_recurrent_state_resident_buffer(batch.id());

        Ok(true)
    }

    fn assemble_linear_attn_gdn_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
    ) -> Result<bool> {
        VulkanBackend::assemble_linear_attn_gdn_state_batch_kt(self, row_keys, batch_key)
    }

    fn scatter_linear_attn_gdn_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
    ) -> Result<bool> {
        VulkanBackend::scatter_linear_attn_gdn_state_batch_kt(self, batch_key, row_keys)
    }

    fn seed_linear_attn_gdn_state_kt(
        &self,
        recurrent: &kiln_tensor::Tensor,
        conv: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        VulkanBackend::seed_linear_attn_gdn_state_kt(self, recurrent, conv)
    }

    fn has_linear_attn_gdn_state_kt(&self, key: kiln_tensor::TensorId) -> bool {
        VulkanBackend::has_linear_attn_gdn_state_kt(self, key)
    }

    fn supports_gdn_chunk_prep(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_chunk_scan(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        self.has_vulkan() && self.gdn_full_chunk_forward_enabled
    }

    fn supports_gdn_gates(&self) -> bool {
        self.has_vulkan() && self.gdn_gates_enabled
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        self.has_vulkan() && self.gdn_gated_rms_norm_enabled
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        // Single-token update still regresses Strix Halo decode latency.
        self.has_vulkan() && self.fused_conv1d_update_enabled
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
        self.has_vulkan() && self.fused_conv1d_prefill_enabled
    }

    fn flash_attn_prefill(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_attention::flash_attn_prefill(self, q, k, v, softmax_scale, causal)
    }

    fn flash_attn_paged_decode(
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

    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
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

    fn gdn_in_proj_decode(
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
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.gdn_enabled || x.dtype() != kiln_tensor::DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu)
            || !matches!(in_proj_qkv_t.device(), kiln_tensor::Device::Cpu)
            || !matches!(in_proj_z_t.device(), kiln_tensor::Device::Cpu)
            || !matches!(in_proj_a_t.device(), kiln_tensor::Device::Cpu)
            || !matches!(in_proj_b_t.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }
        // (#1082) Fully kt-native: shapes off kt, weight buffers keyed on the
        // stable kt id (upload once), x bytes + outputs straight from/to kt.
        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        if seq_len != 1 && !self.gdn_prefill_in_proj_enabled {
            return Ok(None);
        }

        let Ok((qkv_hidden, qkv_dim)) = in_proj_qkv_t.dims2() else {
            return Ok(None);
        };
        let Ok((z_hidden, z_dim)) = in_proj_z_t.dims2() else {
            return Ok(None);
        };
        let Ok((a_hidden, a_dim)) = in_proj_a_t.dims2() else {
            return Ok(None);
        };
        let Ok((b_hidden, b_dim)) = in_proj_b_t.dims2() else {
            return Ok(None);
        };
        if qkv_hidden != hidden || z_hidden != hidden || a_hidden != hidden || b_hidden != hidden {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let row_count = batch * seq_len;
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let use_bf16 = self.bf16_packed_gdn_in_proj_weights_enabled
            && in_proj_qkv_t.dtype() == kiln_tensor::DType::BF16
            && in_proj_z_t.dtype() == kiln_tensor::DType::BF16
            && in_proj_a_t.dtype() == kiln_tensor::DType::BF16
            && in_proj_b_t.dtype() == kiln_tensor::DType::BF16;
        let (qkv_b, z_b, a_b, b_b) = if use_bf16 {
            let qkv_buf = self.cached_bf16_packed_weight_buffer_kt(in_proj_qkv_t)?;
            let z_buf = self.cached_bf16_packed_weight_buffer_kt(in_proj_z_t)?;
            let a_buf = self.cached_bf16_packed_weight_buffer_kt(in_proj_a_t)?;
            let b_buf = self.cached_bf16_packed_weight_buffer_kt(in_proj_b_t)?;
            kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached_bf16_weights_bytes(
                vk_device, &x_data, row_count, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim,
                z_dim, a_dim, b_dim,
            )
            .context("gdn_in_proj_decode kernel failed")?
        } else {
            let qkv_buf = self.cached_f32_weight_buffer_kt(in_proj_qkv_t)?;
            let z_buf = self.cached_f32_weight_buffer_kt(in_proj_z_t)?;
            let a_buf = self.cached_f32_weight_buffer_kt(in_proj_a_t)?;
            let b_buf = self.cached_f32_weight_buffer_kt(in_proj_b_t)?;
            kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached_bytes(
                vk_device, &x_data, row_count, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim,
                z_dim, a_dim, b_dim,
            )
            .context("gdn_in_proj_decode kernel failed")?
        };
        Ok(Some((
            kt_tensor_from_f32_bytes(&qkv_b, &[batch, seq_len, qkv_dim], kiln_tensor::DType::F32)?,
            kt_tensor_from_f32_bytes(&z_b, &[batch, seq_len, z_dim], kiln_tensor::DType::F32)?,
            kt_tensor_from_f32_bytes(&a_b, &[batch, seq_len, a_dim], kiln_tensor::DType::F32)?,
            kt_tensor_from_f32_bytes(&b_b, &[batch, seq_len, b_dim], kiln_tensor::DType::F32)?,
        )))
    }

    fn gdn_decode_gates_recurrent_rmsnorm(
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
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.gdn_enabled || q.dtype() != kiln_tensor::DType::F32 {
            return Ok(None);
        }
        if !matches!(q.device(), kiln_tensor::Device::Cpu)
            || !matches!(k.device(), kiln_tensor::Device::Cpu)
            || !matches!(v.device(), kiln_tensor::Device::Cpu)
            || !matches!(a.device(), kiln_tensor::Device::Cpu)
            || !matches!(b.device(), kiln_tensor::Device::Cpu)
            || !matches!(a_log.device(), kiln_tensor::Device::Cpu)
            || !matches!(dt_bias.device(), kiln_tensor::Device::Cpu)
            || !matches!(state_kt.device(), kiln_tensor::Device::Cpu)
            || !matches!(z.device(), kiln_tensor::Device::Cpu)
            || !matches!(weight.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }
        // (#1082) kt-native: all args are already kt. `state_kt` is mutated in
        // place at each return that may have updated the recurrent state.
        let Ok((batch, seq_len, nv, dk)) = q.dims4() else {
            return Ok(None);
        };
        let Ok((k_batch, k_seq, k_nv, k_dk)) = k.dims4() else {
            return Ok(None);
        };
        let Ok((v_batch, v_seq, v_nv, dv)) = v.dims4() else {
            return Ok(None);
        };
        let Ok((z_batch, z_seq, z_nv, z_dv)) = z.dims4() else {
            return Ok(None);
        };
        let Ok((state_batch, state_nv, state_dk, state_dv)) = state_kt.dims4() else {
            return Ok(None);
        };
        if batch == 1 && !self.gdn_decode_fused_enabled {
            return Ok(None);
        }
        if seq_len != 1
            || k_batch != batch
            || k_seq != 1
            || v_batch != batch
            || v_seq != 1
            || z_batch != batch
            || z_seq != 1
            || k_nv != nv
            || v_nv != nv
            || z_nv != nv
            || k_dk != dk
            || state_batch != batch
            || state_nv != nv
            || state_dk != dk
            || state_dv != dv
            || z_dv != dv
            || dv > 256
        {
            return Ok(None);
        }
        if a.dims() != [batch, 1, nv]
            || b.dims() != [batch, 1, nv]
            || a_log.dims() != [nv]
            || dt_bias.dims() != [nv]
            || weight.dims() != [dv]
        {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
        if batch > 1
            && fused_gdn_resident_state_enabled()
            && recurrent_state_resident_scope_active()
        {
            let state_id = state_kt.id();
            let resident_state = get_recurrent_state_resident_buffer(state_id);
            let (batch_d, _, nv, dk) = q.dims4()?;
            let dv = v.dims4()?.3;
            let q_dtype = q.dtype();
            let q_b = kt_tensor_to_f32_bytes_with_shape(q)?.0;
            let k_b = kt_tensor_to_f32_bytes_with_shape(k)?.0;
            let v_b = kt_tensor_to_f32_bytes_with_shape(v)?.0;
            let a_b = kt_tensor_to_f32_bytes_with_shape(a)?.0;
            let b_b = kt_tensor_to_f32_bytes_with_shape(b)?.0;
            let a_log_b = kt_tensor_to_f32_bytes_with_shape(a_log)?.0;
            let dt_bias_b = kt_tensor_to_f32_bytes_with_shape(dt_bias)?.0;
            let z_b = kt_tensor_to_f32_bytes_with_shape(z)?.0;
            let weight_b = kt_tensor_to_f32_bytes_with_shape(weight)?.0;
            let state_b = if resident_state.is_none() {
                Some(kt_tensor_to_f32_bytes_with_shape(state_kt)?.0)
            } else {
                None
            };
            let (out_data, resident_state) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state_bytes(
                    vk_device,
                    &q_b, &k_b, &v_b, &a_b, &b_b, &a_log_b, &dt_bias_b,
                    state_b.as_deref(),
                    &z_b, &weight_b,
                    batch_d, nv, dk, dv,
                    eps as f32,
                    resident_state,
                )
                .context("gdn_decode_gates_recurrent_rmsnorm resident-state kernel failed")?;
            let out = kt_tensor_from_f32_bytes(&out_data, &[batch_d, 1, nv, dv], q_dtype)?;
            insert_recurrent_state_resident_buffer(state_id, resident_state);
            return Ok(Some(out));
        }
        let (batch, _, nv, dk) = q.dims4()?;
        let dv = v.dims4()?.3;
        let q_dtype = q.dtype();
        let state_dtype = state_kt.dtype();
        let state_dims = state_kt.dims().to_vec();
        let input_tensors: [&kiln_tensor::Tensor; 10] =
            [q, k, v, a, b, a_log, dt_bias, &*state_kt, z, weight];
        let mut input_data: Vec<Vec<u8>> = Vec::with_capacity(input_tensors.len());
        for tensor in &input_tensors {
            input_data.push(kt_tensor_to_f32_bytes_with_shape(tensor)?.0);
        }
        let (out_data, new_state_data) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm_bytes(
                vk_device,
                &input_data,
                batch,
                nv,
                dk,
                dv,
                eps as f32,
                skip_state_readback,
            )
            .context("gdn_decode_gates_recurrent_rmsnorm kernel failed")?;
        let out = kt_tensor_from_f32_bytes(&out_data, &[batch, 1, nv, dv], q_dtype)?;
        if !skip_state_readback {
            if let Some(sd) = new_state_data {
                *state_kt = kt_tensor_from_f32_bytes(&sd, &state_dims, state_dtype)?;
            }
        }
        Ok(Some(out))
    }

    fn linear_decode(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_linear::linear_decode(self, x, weight_t)
    }

    fn linear_prefill_apply(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_linear::linear_prefill_apply(self, x, weight_t)
    }

    fn linear_prefill_apply_offset(
        &self,
        x: &kiln_tensor::Tensor,
        full_weight_t: &kiln_tensor::Tensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        vulkan_linear::linear_prefill_apply_offset(self, x, full_weight_t, chunk_start, chunk_len)
    }

    fn supports_linear_decode_argmax(&self) -> bool {
        vulkan_linear::supports_linear_decode_argmax(self)
    }

    fn linear_decode_argmax(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<u32>> {
        vulkan_linear::linear_decode_argmax(self, x, weight_t)
    }

    fn supports_linear_decode_argmax_batch(&self) -> bool {
        vulkan_linear::supports_linear_decode_argmax_batch(self)
    }

    fn supports_linear_decode_sample(&self, top_k: u32) -> bool {
        vulkan_linear::supports_linear_decode_sample(self, top_k)
    }

    fn linear_decode_sample(
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

    fn supports_linear_decode_sample_batch(&self, top_k: &[u32], temperatures: &[f32]) -> bool {
        vulkan_linear::supports_linear_decode_sample_batch(self, top_k, temperatures)
    }

    fn linear_decode_sample_batch(
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

    fn linear_decode_argmax_batch(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<Vec<u32>>> {
        vulkan_linear::linear_decode_argmax_batch(self, x, weight_t)
    }

    fn prewarm_decode_weights(&self, weights: &GpuWeights) -> Result<()> {
        vulkan_weights::prewarm_decode_weights(self, weights)
    }

    fn drop_uploaded_bf16_weights(
        &self,
        weights: &mut crate::forward::GpuWeights,
        device: &kiln_tensor::Device,
    ) -> Result<usize> {
        vulkan_weights::drop_uploaded_bf16_weights(self, weights, device)
    }

    fn full_attn_qkv_decode(
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
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.full_attn_qkv_enabled || x.dtype() != kiln_tensor::DType::F32
        {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu)
            || !matches!(q_weight_t.device(), kiln_tensor::Device::Cpu)
            || !matches!(k_weight_t.device(), kiln_tensor::Device::Cpu)
            || !matches!(v_weight_t.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }
        // (#1082) Fully kt-native: shapes off kt, QKV weight buffers keyed on
        // the stable kt id (upload once), x bytes + outputs straight from/to kt.
        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        // Multi-token (prefill-ish) shapes still go through the unfused
        // path: this kernel family is the single-token decode projection.
        // Batched single-token decode IS supported via the `_batched` dispatch.
        if seq_len != 1 || batch == 0 {
            return Ok(None);
        }
        let Ok((q_hidden, q_dim)) = q_weight_t.dims2() else {
            return Ok(None);
        };
        let Ok((k_hidden, k_dim)) = k_weight_t.dims2() else {
            return Ok(None);
        };
        let Ok((v_hidden, v_dim)) = v_weight_t.dims2() else {
            return Ok(None);
        };
        if q_hidden != hidden || k_hidden != hidden || v_hidden != hidden {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let bf16 = self.bf16_packed_full_attn_qkv_weights_enabled
            && q_weight_t.dtype() == kiln_tensor::DType::BF16
            && k_weight_t.dtype() == kiln_tensor::DType::BF16
            && v_weight_t.dtype() == kiln_tensor::DType::BF16;
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let (q_b, k_b, v_b) = if batch == 1 {
            if bf16 {
                let q_buf = self.cached_bf16_packed_weight_buffer_kt(q_weight_t)?;
                let k_buf = self.cached_bf16_packed_weight_buffer_kt(k_weight_t)?;
                let v_buf = self.cached_bf16_packed_weight_buffer_kt(v_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_bf16_weights_bytes(
                    vk_device, &x_data, &q_buf, &k_buf, &v_buf, hidden, q_dim, k_dim, v_dim,
                )
            } else {
                let q_buf = self.cached_f32_weight_buffer_kt(q_weight_t)?;
                let k_buf = self.cached_f32_weight_buffer_kt(k_weight_t)?;
                let v_buf = self.cached_f32_weight_buffer_kt(v_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_bytes(
                    vk_device, &x_data, &q_buf, &k_buf, &v_buf, hidden, q_dim, k_dim, v_dim,
                )
            }
            .context("full_attn_qkv_decode kernel failed")?
        } else if bf16 {
            let q_buf = self.cached_bf16_packed_weight_buffer_kt(q_weight_t)?;
            let k_buf = self.cached_bf16_packed_weight_buffer_kt(k_weight_t)?;
            let v_buf = self.cached_bf16_packed_weight_buffer_kt(v_weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_batched_bf16_weights_bytes(
                vk_device, &x_data, &q_buf, &k_buf, &v_buf, batch, hidden, q_dim, k_dim, v_dim,
            )
            .context("full_attn_qkv_decode_batched_bf16w kernel failed")?
        } else {
            let q_buf = self.cached_f32_weight_buffer_kt(q_weight_t)?;
            let k_buf = self.cached_f32_weight_buffer_kt(k_weight_t)?;
            let v_buf = self.cached_f32_weight_buffer_kt(v_weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_batched_bytes(
                vk_device, &x_data, &q_buf, &k_buf, &v_buf, batch, hidden, q_dim, k_dim, v_dim,
            )
            .context("full_attn_qkv_decode_batched kernel failed")?
        };
        Ok(Some((
            kt_tensor_from_f32_bytes(&q_b, &[batch, 1, q_dim], kiln_tensor::DType::F32)?,
            kt_tensor_from_f32_bytes(&k_b, &[batch, 1, k_dim], kiln_tensor::DType::F32)?,
            kt_tensor_from_f32_bytes(&v_b, &[batch, 1, v_dim], kiln_tensor::DType::F32)?,
        )))
    }

    fn mlp_gate_up_decode(
        &self,
        x: &kiln_tensor::Tensor,
        gate_weight_t: &kiln_tensor::Tensor,
        up_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.mlp_gate_up_enabled || x.dtype() != kiln_tensor::DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu)
            || !matches!(gate_weight_t.device(), kiln_tensor::Device::Cpu)
            || !matches!(up_weight_t.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }
        // (#1082) kt-native: shapes off the kt tensors, weight buffers keyed
        // on the stable kt id (upload once), x bytes straight from kt storage.
        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        let Ok((gate_hidden, intermediate)) = gate_weight_t.dims2() else {
            return Ok(None);
        };
        let Ok((up_hidden, up_intermediate)) = up_weight_t.dims2() else {
            return Ok(None);
        };
        if gate_hidden != hidden || up_hidden != hidden || up_intermediate != intermediate {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let gate_buf = self.cached_f32_weight_buffer_kt(gate_weight_t)?;
        let up_buf = self.cached_f32_weight_buffer_kt(up_weight_t)?;
        let row_count = batch * seq_len;
        let dispatch_x = if seq_len == 1 {
            x.clone()
        } else {
            x.reshape((row_count, 1usize, hidden))?
        };
        let x_data = kt_tensor_to_f32_bytes_with_shape(&dispatch_x)?.0;
        let out_data = kiln_vulkan_kernel::kernels::dispatch_mlp_gate_up_decode_cached_bytes(
            vk_device,
            &x_data,
            row_count,
            hidden,
            intermediate,
            &gate_buf,
            &up_buf,
        )
        .context("mlp_gate_up_decode kernel failed")?;
        let out = kt_tensor_from_f32_bytes(
            &out_data,
            &[row_count, 1, intermediate],
            kiln_tensor::DType::F32,
        )?;
        let out = if seq_len == 1 {
            out
        } else {
            out.reshape((batch, seq_len, intermediate))?
        };
        Ok(Some(out))
    }

    fn mlp_decode(
        &self,
        x: &kiln_tensor::Tensor,
        gate_weight_t: &kiln_tensor::Tensor,
        up_weight_t: &kiln_tensor::Tensor,
        down_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.mlp_decode_enabled || x.dtype() != kiln_tensor::DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu)
            || !matches!(gate_weight_t.device(), kiln_tensor::Device::Cpu)
            || !matches!(up_weight_t.device(), kiln_tensor::Device::Cpu)
            || !matches!(down_weight_t.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }
        // (#1082) Fully kt-native: shapes off the kt tensors, weight buffers
        // keyed on the stable kt id (upload once), x bytes straight from kt.
        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        let Ok((gate_hidden, intermediate)) = gate_weight_t.dims2() else {
            return Ok(None);
        };
        let Ok((up_hidden, up_intermediate)) = up_weight_t.dims2() else {
            return Ok(None);
        };
        let Ok((down_intermediate, out_dim)) = down_weight_t.dims2() else {
            return Ok(None);
        };
        if gate_hidden != hidden
            || up_hidden != hidden
            || up_intermediate != intermediate
            || down_intermediate != intermediate
        {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let row_count = batch * seq_len;
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let use_bf16_mlp_weights = self.bf16_packed_mlp_decode_weights_enabled
            && gate_weight_t.dtype() == kiln_tensor::DType::BF16
            && up_weight_t.dtype() == kiln_tensor::DType::BF16
            && down_weight_t.dtype() == kiln_tensor::DType::BF16;
        let out_data =
            if row_count >= 8 && self.mlp_bf16_gate_up_f32_down_enabled && use_bf16_mlp_weights {
                let gate_buf = self.cached_bf16_packed_weight_buffer_kt(gate_weight_t)?;
                let up_buf = self.cached_bf16_packed_weight_buffer_kt(up_weight_t)?;
                let down_buf = self.cached_f32_weight_buffer_kt(down_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_gate_up_f32_down_bytes(
                    vk_device,
                    &x_data,
                    row_count,
                    &gate_buf,
                    &up_buf,
                    &down_buf,
                    hidden,
                    intermediate,
                    out_dim,
                )
                .context("mlp_decode kernel failed")?
            } else if use_bf16_mlp_weights {
                let gate_buf = self.cached_bf16_packed_weight_buffer_kt(gate_weight_t)?;
                let up_buf = self.cached_bf16_packed_weight_buffer_kt(up_weight_t)?;
                let down_buf = self.cached_bf16_packed_weight_buffer_kt(down_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_weights_bytes(
                    vk_device,
                    &x_data,
                    row_count,
                    &gate_buf,
                    &up_buf,
                    &down_buf,
                    hidden,
                    intermediate,
                    out_dim,
                )
                .context("mlp_decode kernel failed")?
            } else {
                let gate_buf = self.cached_f32_weight_buffer_kt(gate_weight_t)?;
                let up_buf = self.cached_f32_weight_buffer_kt(up_weight_t)?;
                let down_buf = self.cached_f32_weight_buffer_kt(down_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bytes(
                    vk_device,
                    &x_data,
                    row_count,
                    &gate_buf,
                    &up_buf,
                    &down_buf,
                    hidden,
                    intermediate,
                    out_dim,
                )
                .context("mlp_decode kernel failed")?
            };
        Ok(Some(kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, seq_len, out_dim],
            kiln_tensor::DType::F32,
        )?))
    }

    fn gdn_forward_substitution(
        &self,
        a_strict: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if a_strict.dtype() != kiln_tensor::DType::BF16 {
            return Ok(None);
        }
        // (#1082) kt-native: byte extraction reads straight from kt storage.
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let v_dims = v_prime.dims();
        let (batch, heads, chunk, dv) = (v_dims[0], v_dims[1], v_dims[2], v_dims[3]);
        let a_strict_bytes = kt_tensor_to_f32_bytes_with_shape(a_strict)?.0;
        let v_prime_bytes = kt_tensor_to_f32_bytes_with_shape(v_prime)?.0;
        let beta_bytes = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
        let out_data = kiln_vulkan_kernel::kernels::dispatch_gdn_forward_substitution_bytes(
            vk_device,
            &a_strict_bytes,
            &v_prime_bytes,
            &beta_bytes,
            batch,
            heads,
            chunk,
            dv,
        )
        .context("gdn_forward_substitution kernel failed")?;
        let out = kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, heads, chunk, dv],
            kiln_tensor::DType::F32,
        )?;
        Ok(Some(out))
    }

    fn gdn_recurrent_prefill_native_head_last(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state_kt: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan()
            || !self.gdn_recurrent_unexpanded_qk_enabled
            || !matches!(
                q.dtype(),
                kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
            )
        {
            return Ok(None);
        }
        if !matches!(q.device(), kiln_tensor::Device::Cpu)
            || !matches!(k.device(), kiln_tensor::Device::Cpu)
            || !matches!(v.device(), kiln_tensor::Device::Cpu)
            || !matches!(beta.device(), kiln_tensor::Device::Cpu)
            || !matches!(g.device(), kiln_tensor::Device::Cpu)
            || !matches!(state_kt.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }
        // (#1082) kt-native: all args are already kt; `state_kt` is mutated in
        // place. The recurrent-state resident cache keys on the kt `TensorId`.
        let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
            return Ok(None);
        };
        let Ok((k_batch, k_seq_len, k_heads, k_dk)) = k.dims4() else {
            return Ok(None);
        };
        let Ok((v_batch, v_seq_len, heads, dv)) = v.dims4() else {
            return Ok(None);
        };
        let Ok((beta_batch, beta_seq_len, beta_heads)) = beta.dims3() else {
            return Ok(None);
        };
        let Ok((g_batch, g_seq_len, g_heads)) = g.dims3() else {
            return Ok(None);
        };
        let Ok((state_batch, state_heads, state_dk, state_dv)) = state_kt.dims4() else {
            return Ok(None);
        };
        if seq_len != 1
            || k_batch != batch
            || k_seq_len != seq_len
            || k_heads != q_heads
            || k_dk != dk
            || v_batch != batch
            || v_seq_len != seq_len
            || beta_batch != batch
            || beta_seq_len != seq_len
            || beta_heads != heads
            || g_batch != batch
            || g_seq_len != seq_len
            || g_heads != heads
            || state_batch != batch
            || state_heads != heads
            || state_dk != dk
            || state_dv != dv
            || q_heads == 0
            || heads % q_heads != 0
        {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        if self.recurrent_state_residency_enabled
            && recurrent_state_resident_scope_active()
            && state_kt.dtype() == q.dtype()
        {
            let state_id = state_kt.id();
            let resident_state = get_recurrent_state_resident_buffer(state_id);
            let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
            let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
            let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
            let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
            let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
            let state_data_owned = if resident_state.is_none() {
                Some(kt_tensor_to_f32_bytes_with_shape(state_kt)?.0)
            } else {
                None
            };
            let (batch, seq_len, q_heads, dk) = q.dims4()?;
            let (_, _, heads, dv) = v.dims4()?;
            let q_dtype = q.dtype();
            let (out_data, resident_state) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_native_head_last_resident_state_bytes(
                    vk_device,
                    &q_data, &k_data, &v_data, &beta_data, &g_data,
                    state_data_owned.as_deref(),
                    batch, seq_len, q_heads, heads, dk, dv,
                    resident_state,
                )
                .context("gdn_recurrent_step native-head resident-state Vulkan kernel failed")?;
            // `out_data` is the un-unsqueezed [batch, heads, dv] layout.
            // Reconstruct the kt tensor and re-unsqueeze to match prior public shape.
            let out_no_seq = kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], q_dtype)?;
            let out = out_no_seq.unsqueeze(1)?;
            insert_recurrent_state_resident_buffer(state_id, resident_state);
            return Ok(Some(out));
        }
        let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
        let (batch, _seq, q_heads, dk) = q.dims4()?;
        let (_, _, heads, dv) = v.dims4()?;
        let q_dtype = q.dtype();
        let state_dtype = state_kt.dtype();
        let state_dims = state_kt.dims().to_vec();
        let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
        let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
        let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
        let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
        let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
        let state_data = kt_tensor_to_f32_bytes_with_shape(state_kt)?.0;
        let (out_data, new_state_data) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_native_head_last_with_options_bytes(
                vk_device,
                &q_data,
                &k_data,
                &v_data,
                &beta_data,
                &g_data,
                &state_data,
                batch,
                q_heads,
                heads,
                dk,
                dv,
                skip_state_readback,
            )
            .context("gdn_recurrent_step native-head Vulkan kernel failed")?;
        let out =
            kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], q_dtype)?.unsqueeze(1)?;
        if let Some(sd) = new_state_data {
            *state_kt = kt_tensor_from_f32_bytes(&sd, &state_dims, state_dtype)?;
        }
        Ok(Some(out))
    }

    fn gdn_recurrent_qk_norm_prefill_native_head_last(
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
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan()
            || !self.gdn_recurrent_qk_norm_unexpanded_enabled
            || !matches!(
                q.dtype(),
                kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
            )
        {
            return Ok(None);
        }
        if !matches!(q.device(), kiln_tensor::Device::Cpu)
            || !matches!(k.device(), kiln_tensor::Device::Cpu)
            || !matches!(v.device(), kiln_tensor::Device::Cpu)
            || !matches!(beta.device(), kiln_tensor::Device::Cpu)
            || !matches!(g.device(), kiln_tensor::Device::Cpu)
            || !matches!(state_kt.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }
        // (#1082) kt-native: all args are already kt; `state_kt` is mutated in
        // place at the return below.
        let Ok((_, _, _, dk)) = q.dims4() else {
            return Ok(None);
        };
        let expected_scale = 1.0 / (dk as f64).sqrt();
        if (q_scale - expected_scale).abs() > 1e-6 || (qk_eps - 1e-6).abs() > 1e-12 {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
        let (batch, _seq, q_heads, dk) = q.dims4()?;
        let (_, _, heads, dv) = v.dims4()?;
        let state_dtype = state_kt.dtype();
        let state_dims = state_kt.dims().to_vec();
        let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
        let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
        let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
        let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
        let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
        let state_data = kt_tensor_to_f32_bytes_with_shape(state_kt)?.0;
        let (out_data, new_state_data) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_qk_norm_step_native_head_last_with_options_bytes(
                vk_device,
                &q_data,
                &k_data,
                &v_data,
                &beta_data,
                &g_data,
                &state_data,
                batch,
                q_heads,
                heads,
                dk,
                dv,
                skip_state_readback,
            )
            .context("gdn_recurrent_qk_norm native-head Vulkan kernel failed")?;
        let out =
            kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], state_dtype)?.unsqueeze(1)?;
        if let Some(sd) = new_state_data {
            *state_kt = kt_tensor_from_f32_bytes(&sd, &state_dims, state_dtype)?;
        }
        Ok(Some(out))
    }

    fn gdn_recurrent_step(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state_kt: &mut kiln_tensor::Tensor,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if !matches!(
            q.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        ) {
            return Ok(None);
        }
        // (#1082) kt-native: all args are already kt; `state_kt` is mutated in
        // place. The recurrent-state resident cache keys on the kt `TensorId`.
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        if self.recurrent_state_residency_enabled && recurrent_state_resident_scope_active() {
            let state_id = state_kt.id();
            let resident_state = get_recurrent_state_resident_buffer(state_id);

            let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
            let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
            let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
            let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
            let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
            let state_data_owned = if resident_state.is_none() {
                Some(kt_tensor_to_f32_bytes_with_shape(state_kt)?.0)
            } else {
                None
            };
            let q_dims = q.dims();
            let (batch, heads, dk) = (q_dims[0], q_dims[1], q_dims[2]);
            let dv = v.dims()[2];
            let q_dtype = q.dtype();
            let (out_data, resident_state) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_resident_state_bytes(
                    vk_device,
                    &q_data,
                    &k_data,
                    &v_data,
                    &beta_data,
                    &g_data,
                    state_data_owned.as_deref(),
                    batch,
                    heads,
                    dk,
                    dv,
                    resident_state,
                )
                .context("gdn_recurrent_step resident-state kernel failed")?;
            let out = kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], q_dtype)?;

            insert_recurrent_state_resident_buffer(state_id, resident_state);
            return Ok(Some(out));
        }

        let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
        let q_dims = q.dims();
        let (batch, heads, dk) = (q_dims[0], q_dims[1], q_dims[2]);
        let dv = v.dims()[2];
        let q_dtype = q.dtype();
        let state_dtype = state_kt.dtype();
        let state_dims = state_kt.dims().to_vec();
        let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
        let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
        let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
        let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
        let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
        let state_data = kt_tensor_to_f32_bytes_with_shape(state_kt)?.0;
        let (out_data, new_state_data) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_with_options_bytes(
                vk_device,
                &q_data,
                &k_data,
                &v_data,
                &beta_data,
                &g_data,
                &state_data,
                batch,
                heads,
                dk,
                dv,
                skip_state_readback,
            )
            .context("gdn_recurrent_step kernel failed")?;
        let out = kt_tensor_from_f32_bytes(&out_data, &[batch, heads, dv], q_dtype)?;
        if let Some(sd) = new_state_data {
            *state_kt = kt_tensor_from_f32_bytes(&sd, &state_dims, state_dtype)?;
        }
        Ok(Some(out))
    }

    fn gdn_chunkwise_forward(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        g: &kiln_tensor::Tensor,
        state_kt: &mut kiln_tensor::Tensor,
        chunk_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // Proper Vulkan GDN prefill: run the chunkwise scan on the GPU in
        // parallel (`vk_gdn_chunkwise_forward_no_grad`) instead of the CPU
        // chunkwise (raw kt matmuls on CPU-host tensors). F32 only on Vulkan
        // (activations are F32). kt-native: extract f32 straight from kt
        // storage, no candle bridge. (#1082)
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if q.dtype() != kiln_tensor::DType::F32 || state_kt.dtype() != kiln_tensor::DType::F32 {
            return Ok(None);
        }
        if std::env::var("KILN_DISABLE_VULKAN_GDN_CHUNKWISE_FORWARD").is_ok() {
            return Ok(None);
        }
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(None);
        };

        let state_shape = state_kt.shape().to_vec();
        let (q_vk, k_vk, v_vk, beta_vk, g_vk, mut state_vk) =
            if let Some([q_vk, k_vk, v_vk, beta_vk, g_vk, state_vk]) =
                upload_gdn_chunkwise_inputs_from_cpu_bytes_vk(
                    vk_device, q, k, v, beta, g, state_kt,
                )?
            {
                (q_vk, k_vk, v_vk, beta_vk, g_vk, state_vk)
            } else {
                let load =
                    |t: &kiln_tensor::Tensor| -> Result<kiln_vulkan_kernel::vk_tensor::VkTensor> {
                        let shape = t.shape().to_vec();
                        let data = t
                            .flatten_all()
                            .map_err(|e| anyhow::anyhow!("gdn_chunkwise_forward: flatten: {e}"))?
                            .to_vec1::<f32>()
                            .map_err(|e| {
                                anyhow::anyhow!("gdn_chunkwise_forward: to_vec1 f32: {e}")
                            })?;
                        kiln_vulkan_kernel::vk_tensor::VkTensor::from_f32_slice(
                            &data,
                            shape,
                            vk_device.clone(),
                        )
                    };
                (
                    load(q)?,
                    load(k)?,
                    load(v)?,
                    load(beta)?,
                    load(g)?,
                    load(state_kt)?,
                )
            };

        let out_vk = if std::env::var("KILN_DISABLE_VULKAN_GDN_CHUNKWISE_SINGLE_SUBMIT").is_ok() {
            if kiln_core::env_flag::env_flag("KILN_VULKAN_GDN_CHUNKWISE_FALLBACK", false) {
                tracing::warn!("single-submit Vulkan GDN chunkwise prefill disabled; falling back");
                kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise_forward_no_grad(
                    &q_vk,
                    &k_vk,
                    &v_vk,
                    &beta_vk,
                    &g_vk,
                    &mut state_vk,
                    chunk_size,
                )
                .context("vk_gdn_chunkwise_forward_no_grad fallback")?
            } else {
                anyhow::bail!(
                    "single-submit Vulkan GDN chunkwise prefill disabled; fallback disabled"
                );
            }
        } else {
            match kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise_forward_no_grad_single_submit(
                    &q_vk,
                    &k_vk,
                    &v_vk,
                    &beta_vk,
                    &g_vk,
                    &mut state_vk,
                    chunk_size,
                ) {
                    Ok(out) => out,
                    Err(err) => {
                        if kiln_core::env_flag::env_flag(
                            "KILN_VULKAN_GDN_CHUNKWISE_FALLBACK",
                            false,
                        ) {
                            tracing::warn!(
                                error = %err,
                                "single-submit Vulkan GDN chunkwise prefill failed; falling back"
                            );
                            kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise_forward_no_grad(
                                &q_vk,
                                &k_vk,
                                &v_vk,
                                &beta_vk,
                                &g_vk,
                                &mut state_vk,
                                chunk_size,
                            )
                            .context("vk_gdn_chunkwise_forward_no_grad fallback")?
                        } else {
                            return Err(err).context(
                                "single-submit Vulkan GDN chunkwise prefill failed; fallback disabled",
                            );
                        }
                    }
                }
        };

        // Read back output + the updated state together, then rebuild CPU-host
        // kt tensors without decoding through an intermediate Vec<f32>.
        let [out_kt, new_state]: [kiln_tensor::Tensor; 2] =
            vk_f32_tensors_to_cpu_tensors_batched_vk(&[
                (&out_vk, "gdn_chunkwise_forward output"),
                (&state_vk, "gdn_chunkwise_forward state"),
            ])?
            .try_into()
            .map_err(|readbacks: Vec<_>| {
                anyhow::anyhow!(
                    "gdn_chunkwise_forward: read back {} tensors, expected 2",
                    readbacks.len()
                )
            })?;
        anyhow::ensure!(
            new_state.shape() == state_shape.as_slice(),
            "gdn_chunkwise_forward: state shape mismatch after readback: got {:?}, expected {:?}",
            new_state.shape(),
            state_shape
        );
        *state_kt = new_state;
        Ok(Some(out_kt))
    }

    fn gdn_chunk_prep(
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
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if g.dtype() != kiln_tensor::DType::BF16 {
            return Ok(None);
        }
        // (#1082) kt-native: byte extraction + reconstruction run on kt args.
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
        let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
        let kkt_data = kt_tensor_to_f32_bytes_with_shape(kkt)?.0;
        let qkt_data = kt_tensor_to_f32_bytes_with_shape(qkt)?.0;
        let ks_entry_data = kt_tensor_to_f32_bytes_with_shape(ks_entry)?.0;
        let q_s_data = kt_tensor_to_f32_bytes_with_shape(q_s)?.0;
        let g_dims = g.dims();
        let (batch, heads, chunk) = (g_dims[0], g_dims[1], g_dims[2]);
        let dv = v.dims()[3];
        let (a_strict_b, b_mask_b, v_prime_b, q_s_scaled_b, decay_last_col_b, p_last_b) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_prep_bytes(
                vk_device,
                &g_data,
                &v_data,
                &kkt_data,
                &qkt_data,
                &ks_entry_data,
                &q_s_data,
                batch,
                heads,
                chunk,
                dv,
            )
            .context("gdn_chunk_prep kernel failed")?;
        let cc_shape = [batch, heads, chunk, chunk];
        let cv_shape = [batch, heads, chunk, dv];
        let decay_shape = [batch, heads, chunk];
        let p_last_shape = [batch, heads];
        Ok(Some((
            kt_tensor_from_f32_bytes(&a_strict_b, &cc_shape, kiln_tensor::DType::BF16)?,
            kt_tensor_from_f32_bytes(&b_mask_b, &cc_shape, kiln_tensor::DType::BF16)?,
            kt_tensor_from_f32_bytes(&v_prime_b, &cv_shape, kiln_tensor::DType::BF16)?,
            kt_tensor_from_f32_bytes(&q_s_scaled_b, &cv_shape, kiln_tensor::DType::BF16)?,
            kt_tensor_from_f32_bytes(&decay_last_col_b, &decay_shape, kiln_tensor::DType::BF16)?,
            kt_tensor_from_f32_bytes(&p_last_b, &p_last_shape, kiln_tensor::DType::BF16)?,
        )))
    }

    fn gdn_chunk_scan(
        &self,
        a_strict: &kiln_tensor::Tensor,
        b_mask: &kiln_tensor::Tensor,
        v_prime: &kiln_tensor::Tensor,
        q_s_scaled: &kiln_tensor::Tensor,
        beta: &kiln_tensor::Tensor,
        decay_last_col: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if a_strict.dtype() != kiln_tensor::DType::BF16 {
            return Ok(None);
        }
        // (#1082) kt-native: byte extraction + reconstruction run on kt args.
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let a_strict_data = kt_tensor_to_f32_bytes_with_shape(a_strict)?.0;
        let b_mask_data = kt_tensor_to_f32_bytes_with_shape(b_mask)?.0;
        let v_prime_data = kt_tensor_to_f32_bytes_with_shape(v_prime)?.0;
        let q_s_scaled_data = kt_tensor_to_f32_bytes_with_shape(q_s_scaled)?.0;
        let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
        let decay_last_col_data = kt_tensor_to_f32_bytes_with_shape(decay_last_col)?.0;
        let v_prime_dims = v_prime.dims();
        let (batch, heads, chunk, dv) = (
            v_prime_dims[0],
            v_prime_dims[1],
            v_prime_dims[2],
            v_prime_dims[3],
        );
        let (out_data, p_out_data) = kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_scan_bytes(
            vk_device,
            &a_strict_data,
            &b_mask_data,
            &v_prime_data,
            &q_s_scaled_data,
            &beta_data,
            &decay_last_col_data,
            batch,
            heads,
            chunk,
            dv,
        )
        .context("gdn_chunk_scan kernel failed")?;
        let out_tensor = kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, heads, chunk, dv],
            kiln_tensor::DType::BF16,
        )?;
        let p_out_tensor = kt_tensor_from_f32_bytes(
            &p_out_data,
            &[batch, heads, chunk, dv],
            kiln_tensor::DType::BF16,
        )?;
        Ok(Some((out_tensor, p_out_tensor)))
    }

    fn gdn_full_chunk_forward(
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
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if g.dtype() != kiln_tensor::DType::BF16 {
            return Ok(None);
        }
        // (#1082) kt-native: all args are already kt; `state_kt` is mutated in
        // place at the return below.
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let g_data = kt_tensor_to_f32_bytes_with_shape(g)?.0;
        let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
        let kkt_data = kt_tensor_to_f32_bytes_with_shape(kkt)?.0;
        let qkt_data = kt_tensor_to_f32_bytes_with_shape(qkt)?.0;
        let ks_entry_data = kt_tensor_to_f32_bytes_with_shape(ks_entry)?.0;
        let q_s_data = kt_tensor_to_f32_bytes_with_shape(q_s)?.0;
        let beta_data = kt_tensor_to_f32_bytes_with_shape(beta)?.0;
        let k_t_data = kt_tensor_to_f32_bytes_with_shape(k_t)?.0;
        let state_data = kt_tensor_to_f32_bytes_with_shape(state_kt)?.0;
        let g_dims = g.dims();
        let (batch, heads, chunk) = (g_dims[0], g_dims[1], g_dims[2]);
        let dv = v.dims()[3];
        let dk = k_t.dims()[2];
        let state_dims = state_kt.dims().to_vec();
        let (out_data, new_state_data) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_full_chunk_forward_bytes(
                vk_device,
                &g_data,
                &v_data,
                &kkt_data,
                &qkt_data,
                &ks_entry_data,
                &q_s_data,
                &beta_data,
                &k_t_data,
                &state_data,
                batch,
                heads,
                chunk,
                dk,
                dv,
            )
            .context("gdn_full_chunk_forward kernel failed")?;
        let out = kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, heads, chunk, dv],
            kiln_tensor::DType::BF16,
        )?;
        *state_kt =
            kt_tensor_from_f32_bytes(&new_state_data, &state_dims, kiln_tensor::DType::BF16)?;
        Ok(Some(out))
    }

    fn gdn_gates(
        &self,
        a: &kiln_tensor::Tensor,
        b: &kiln_tensor::Tensor,
        a_log: &kiln_tensor::Tensor,
        dt_bias: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.gdn_gates_enabled {
            return Ok(None);
        }
        if !matches!(
            a.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        ) {
            return Ok(None);
        }
        // (#1082) kt-native: weight buffers keyed on the stable kt id; byte
        // extraction + reconstruction run on the kt args.
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let nv = a_log.elem_count();
        if dt_bias.elem_count() != nv {
            return Ok(None);
        }
        let a_log_buf = self.cached_f32_weight_buffer_kt(a_log)?;
        let dt_bias_buf = self.cached_f32_weight_buffer_kt(dt_bias)?;

        // Output shape matches input shape [B, T, nv]
        let out_shape = a.dims().to_vec();
        let a_data = kt_tensor_to_f32_bytes_with_shape(a)?.0;
        let b_data = kt_tensor_to_f32_bytes_with_shape(b)?.0;
        let output_dtype = a.dtype();
        let (beta_b, g_b) = kiln_vulkan_kernel::kernels::dispatch_gdn_gates_cached_bytes(
            vk_device,
            &a_data,
            &b_data,
            &a_log_buf,
            &dt_bias_buf,
            nv,
            &out_shape,
        )
        .context("gdn_gates kernel failed")?;
        let beta = kt_tensor_from_f32_bytes(&beta_b, &out_shape, output_dtype)?;
        let g = kt_tensor_from_f32_bytes(&g_b, &out_shape, output_dtype)?;
        Ok(Some((beta, g)))
    }

    fn gdn_gated_rms_norm(
        &self,
        x: &kiln_tensor::Tensor,
        z: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        eps: f64,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.gdn_gated_rms_norm_enabled {
            return Ok(None);
        }
        if !matches!(
            x.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        ) {
            return Ok(None);
        }
        // (#1082) kt-native: weight buffer keyed on the stable kt id; byte
        // extraction + reconstruction run on the kt args.
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let hidden = weight.elem_count();
        if hidden == 0 || x.elem_count() % hidden != 0 {
            return Ok(None);
        }
        let weight_buf = self.cached_f32_weight_buffer_kt(weight)?;

        // Output shape matches x shape
        let out_shape = x.dims().to_vec();
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let z_data = kt_tensor_to_f32_bytes_with_shape(z)?.0;
        let output_dtype = x.dtype();
        let out_data = kiln_vulkan_kernel::kernels::dispatch_gdn_gated_rms_norm_cached_bytes(
            vk_device,
            &x_data,
            &z_data,
            &weight_buf,
            hidden,
            eps as f32,
            &out_shape,
        )
        .context("gdn_gated_rms_norm kernel failed")?;
        let out = kt_tensor_from_f32_bytes(&out_data, &out_shape, output_dtype)?;
        Ok(Some(out))
    }

    fn causal_conv1d_update(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state_kt: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.fused_conv1d_update_enabled {
            return Ok(None);
        }
        if !matches!(
            x.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        ) {
            return Ok(None);
        }
        // (#1082) kt-native: all args are already kt; `conv_state_kt` is
        // mutated in place at the return below.
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let weight_data = kt_tensor_to_f32_bytes_with_shape(weight)?.0;
        let state_data = kt_tensor_to_f32_bytes_with_shape(conv_state_kt)?.0;
        let dims = x.dims();
        anyhow::ensure!(
            dims.len() == 3,
            "causal_conv1d_update: x must be 3-D, got {:?}",
            dims
        );
        let (batch, channels, seq_len) = (dims[0], dims[1], dims[2]);
        let conv_state_shape = conv_state_kt.dims().to_vec();
        let (out_data, state_data_out) =
            kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_update_bytes(
                vk_device,
                &x_data,
                &weight_data,
                &state_data,
                batch,
                channels,
                seq_len,
                kernel_size,
            )
            .context("causal_conv1d_update kernel failed")?;
        let out_shape: Vec<usize> = dims.to_vec();
        let out = kt_tensor_from_f32_bytes(&out_data, &out_shape, kiln_tensor::DType::F32)?;
        *conv_state_kt =
            kt_tensor_from_f32_bytes(&state_data_out, &conv_state_shape, kiln_tensor::DType::F32)?;
        Ok(Some(out))
    }

    fn causal_conv1d_prefill(
        &self,
        x: &kiln_tensor::Tensor,
        weight: &kiln_tensor::Tensor,
        conv_state_kt: &mut kiln_tensor::Tensor,
        kernel_size: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.fused_conv1d_prefill_enabled {
            return Ok(None);
        }
        if !matches!(
            x.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        ) {
            return Ok(None);
        }
        // (#1082) kt-native: all args are already kt; `conv_state_kt` is
        // mutated in place at the return below.
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (out, new_state) = if self.conv1d_prefill_single_submit_enabled {
            let weight_buf = self.cached_f32_weight_buffer_kt(weight)?;
            let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
            let state_data = kt_tensor_to_f32_bytes_with_shape(conv_state_kt)?.0;
            let x_dims = x.dims();
            let (batch, channels, seq_len) = (x_dims[0], x_dims[1], x_dims[2]);
            let conv_state_dims = conv_state_kt.dims().to_vec();
            let (out_data, new_state_data) =
                kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill_cached_weight_bytes(
                    vk_device,
                    &x_data,
                    &weight_buf,
                    &state_data,
                    batch,
                    channels,
                    seq_len,
                    kernel_size,
                )
                .context("causal_conv1d_prefill cached-weight single-submit kernel failed")?;
            let out = kt_tensor_from_f32_bytes(&out_data, x_dims, kiln_tensor::DType::F32)?;
            let new_state = kt_tensor_from_f32_bytes(
                &new_state_data,
                &conv_state_dims,
                kiln_tensor::DType::F32,
            )?;
            (out, new_state)
        } else {
            {
                let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
                let weight_data = kt_tensor_to_f32_bytes_with_shape(weight)?.0;
                let state_data = kt_tensor_to_f32_bytes_with_shape(conv_state_kt)?.0;
                let x_dims = x.dims();
                let (batch, channels, seq_len) = (x_dims[0], x_dims[1], x_dims[2]);
                let conv_state_dims = conv_state_kt.dims().to_vec();
                let (out_data, new_state_data) =
                    kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill_bytes(
                        vk_device,
                        &x_data,
                        &weight_data,
                        &state_data,
                        batch,
                        channels,
                        seq_len,
                        kernel_size,
                    )
                    .context("causal_conv1d_prefill kernel failed")?;
                let out = kt_tensor_from_f32_bytes(&out_data, x_dims, kiln_tensor::DType::F32)?;
                let new_state = kt_tensor_from_f32_bytes(
                    &new_state_data,
                    &conv_state_dims,
                    kiln_tensor::DType::F32,
                )?;
                (out, new_state)
            }
        };
        *conv_state_kt = new_state;
        Ok(Some(out))
    }
}

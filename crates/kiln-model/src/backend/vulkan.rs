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
    kt_tensor_to_packed_bf16_bytes_with_shape, resident_sdpa_prefill_b1,
    upload_gdn_chunkwise_inputs_from_cpu_bytes_vk, vk_f32_tensors_to_cpu_tensors_batched_vk,
};
use super::{vulkan_training, BackendRuntime, TrainingCapabilities, TrainingPrecisionPolicy};
use crate::forward::{GpuAttentionWeights, GpuWeights};

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
    linear_decode_enabled: bool,
    linear_argmax_batch_enabled: bool,
    full_attn_qkv_enabled: bool,
    paged_attn_decode_batch_enabled: bool,
    mlp_decode_enabled: bool,
    mlp_gate_up_enabled: bool,
    mlp_bf16_gate_up_f32_down_enabled: bool,
    bf16_packed_linear_weights_enabled: bool,
    bf16_packed_gdn_in_proj_weights_enabled: bool,
    bf16_packed_full_attn_qkv_weights_enabled: bool,
    bf16_packed_mlp_decode_weights_enabled: bool,
    weight_prewarm_enabled: bool,
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
    decode_resident_pool:
        OnceLock<Option<Arc<kiln_vulkan_kernel::DecodeResidentPool>>>,
    /// Lazily constructed Vulkan-resident paged KV cache. Mirrors the
    /// legacy `PagedKvCache` layout in device-local f32 buffers so the
    /// resident decode dispatchers can read/write K/V without crossing
    /// the host boundary. The first resident decode call that needs the
    /// cache constructs it for the active model geometry.
    vk_paged_kv_cache:
        OnceLock<Option<Arc<kiln_vulkan_kernel::VkPagedKvCache>>>,
    /// Set of full-attention layer indices whose K/V state has already
    /// been seeded into the Vulkan-resident paged cache from the legacy
    /// candle pool. Each full-attention layer is seeded once at the
    /// first call to the resident block helper for that layer; subsequent
    /// decode steps only do per-token slot writes.
    seeded_full_attn_layers: Mutex<HashSet<usize>>,
    /// Batched resident decode rows whose prompt K/V blocks have been seeded.
    /// Keyed by `(full_attention_layer_idx, decode_row_id)`.
    seeded_resident_decode_rows: Mutex<HashSet<(usize, u64)>>,
    /// kt-native mirrors for the single-submit resident decode path.
    linear_attn_recurrent_state_kt:
        Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    linear_attn_conv_state_kt:
        Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    seeded_linear_attn_layers_kt: Mutex<HashSet<kiln_tensor::TensorId>>,
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
    last_resident_start_pos: Mutex<Option<usize>>,
    /// Scratch activation buffers reused across resident decode calls,
    /// keyed by a stable role string. Each entry persists for the
    /// backend's lifetime (single-sequence decode reuses the same
    /// buffers across layers and across tokens). Avoids the
    /// `create_device_local` + `Drop` pair that ran on every call
    /// (≈ 200 µs × 12 buffers × N layers per token).
    resident_scratch: Mutex<HashMap<&'static str, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
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
    weight_cache_kt: Mutex<HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    bf16_packed_weight_cache_kt:
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

/// When set, the multi-batch paged attention decode path walks the
/// block_table inside the Vulkan shader instead of compacting K/V on the
/// host. Default: enabled. Disable via
/// `KILN_DISABLE_VULKAN_PAGED_DECODE_GPU_GATHER=1` to force a visible native
/// helper error for parity comparisons.
fn paged_decode_gpu_gather_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_PAGED_DECODE_GPU_GATHER").is_err()
    })
}

fn generic_paged_decode_splitk_chunks(batch: usize, max_blocks_per_seq: usize) -> usize {
    kiln_vulkan_kernel::kernels::paged_attn_decode_splitk_chunks(batch, max_blocks_per_seq)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_vulkan_paged_decode_bytes(
    vk_device: &kiln_vulkan_kernel::VulkanDevice,
    q_data: &[u8],
    k_pool_data: &[u8],
    v_pool_data: &[u8],
    batch: usize,
    num_heads: usize,
    head_dim: usize,
    total_slots: usize,
    num_kv_heads: usize,
    block_data: &[u32],
    seq_lens: &[u32],
    max_blocks_per_seq: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<Vec<u8>> {
    let num_chunks = generic_paged_decode_splitk_chunks(batch, max_blocks_per_seq);
    if num_chunks > 1 {
        kiln_vulkan_kernel::kernels::dispatch_paged_attn_decode_batch_paged_splitk_f32_bytes(
            vk_device,
            q_data,
            k_pool_data,
            v_pool_data,
            batch,
            num_heads,
            head_dim,
            total_slots,
            num_kv_heads,
            block_data,
            seq_lens,
            max_blocks_per_seq,
            page_block_size,
            softmax_scale,
            num_chunks,
        )
        .context("Vulkan split-K paged decode kernel failed")
    } else {
        kiln_vulkan_kernel::kernels::dispatch_paged_attn_decode_batch_paged_f32_bytes(
            vk_device,
            q_data,
            k_pool_data,
            v_pool_data,
            batch,
            num_heads,
            head_dim,
            total_slots,
            num_kv_heads,
            block_data,
            seq_lens,
            max_blocks_per_seq,
            page_block_size,
            softmax_scale,
        )
        .context("Vulkan paged decode kernel failed")
    }
}

/// Read `KILN_VULKAN_LINEAR` env var. When enabled, the autograd-safe
/// `linear_prefill_apply` path wraps the existing Vulkan linear kernel in
/// a `CustomOp1` so training projections produce a tracked tensor whose
/// backward computes a real gradient instead of dropping it at the leaf
/// returned by the inference-shaped `linear_decode`.
///
/// Default: **enabled**. The previous opt-in default reflected the
/// post-host-crash uncertainty: lm_head forward at the original
/// `/tmp/sft-data.jsonl` repro shape would queue ~4.36M workgroups
/// in one submit on a 40-CU APU and hang the box. Mitigations now in
/// place make the dispatch safe by construction:
///   - `VulkanLinearOp` chunks oversized BF16 matmuls along the
///     output dim (fwd) or batch dim (bwd) so each per-chunk submit
///     stays under the 20 GFLOP per-submit ceiling (commit ca4f53ef);
///   - FLCE provider auto-engages at `active_count ≥ 16` so the SFT
///     loss path goes through chunked FLCE rather than the unfused
///     lm_head dispatch (commit 6182f74);
///   - `linear_prefill_apply_offset` sub-chunks any FLCE chunk that
///     would itself exceed the ceiling.
/// (#1082) Per-dispatch FLOP ceiling for the Vulkan-routed matmul.
///
/// Migrated inline from the deleted `backend::vulkan_linear_op` module
/// (its `candle_core::CustomOp1` training wrapper was removed when the kt
/// autograd tape became the sole grad producer). The forward-only FLCE
/// offset path in `linear_prefill_apply_offset` still needs the ceiling to
/// sub-chunk oversized dispatches: the host hard-hung twice on Strix Halo
/// when a single oversized submit (~4.36M workgroups) was queued, so the
/// ceiling caps per-submit FLOP. Tunable via `KILN_VULKAN_LINEAR_MAX_GFLOP`
/// (parsed once; `0` disables the guard).
const DEFAULT_MAX_FLOP_PER_DISPATCH: u64 = 20_000_000_000;

/// FLOP estimate for `[batch, hidden] @ [hidden, out_dim]` (one mul + one
/// add per inner term).
fn matmul_flop(batch: usize, hidden: usize, out_dim: usize) -> u64 {
    (batch as u64)
        .saturating_mul(hidden as u64)
        .saturating_mul(out_dim as u64)
        .saturating_mul(2)
}

fn max_flop_per_dispatch() -> u64 {
    static CEILING: OnceLock<u64> = OnceLock::new();
    *CEILING.get_or_init(|| {
        std::env::var("KILN_VULKAN_LINEAR_MAX_GFLOP")
            .ok()
            .as_deref()
            .map(str::trim)
            .and_then(|s| s.parse::<f64>().ok())
            .map(|gflop| {
                if gflop <= 0.0 {
                    u64::MAX
                } else {
                    (gflop * 1.0e9_f64).round() as u64
                }
            })
            .unwrap_or(DEFAULT_MAX_FLOP_PER_DISPATCH)
    })
}

/// True when the requested matmul shape would exceed the per-dispatch FLOP
/// ceiling; the caller sub-chunks via [`max_chunk_dim_for_flop`].
fn dispatch_exceeds_safety_ceiling(batch: usize, hidden: usize, out_dim: usize) -> bool {
    matmul_flop(batch, hidden, out_dim) > max_flop_per_dispatch()
}

/// Largest `chunk_dim` such that `2 × other_dim_product × chunk_dim ≤
/// max_flop_per_dispatch()`. Always ≥ 1; returns `usize::MAX` when the
/// guard is disabled.
fn max_chunk_dim_for_flop(other_dim_product: usize) -> usize {
    let max_flop = max_flop_per_dispatch();
    if max_flop == u64::MAX {
        return usize::MAX;
    }
    let denom = (other_dim_product as u64).saturating_mul(2).max(1);
    let chunk = (max_flop / denom) as usize;
    chunk.max(1)
}

impl VulkanBackend {
    pub fn training_capabilities_static() -> TrainingCapabilities {
        vulkan_training::training_capabilities_static()
    }

    pub fn new(device: kiln_tensor::Device) -> Self {
        let config = VulkanRuntimeConfig::from_env();

        let vulkan_device = match kiln_vulkan_kernel::VulkanDevice::new() {
            Ok(dev) => {
                let prewarm_start = std::time::Instant::now();
                match kiln_vulkan_kernel::kernels::prewarm_builtin_pipelines(&dev) {
                    Ok(()) => tracing::info!(
                        elapsed_ms = prewarm_start.elapsed().as_millis() as u64,
                        "Vulkan compute pipelines prewarmed"
                    ),
                    Err(e) => tracing::warn!(
                        error = %e,
                        "Vulkan pipeline prewarm failed; falling back to lazy pipeline creation"
                    ),
                }
                tracing::info!(
                    vendor = dev.vendor_string(),
                    device = dev.device_name(),
                    "Vulkan device initialized"
                );
                Some(Arc::new(dev))
            }
            Err(e) => {
                tracing::warn!(error = %e, "Vulkan device initialization failed, falling back to CPU");
                None
            }
        };

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

    /// Lazily construct (and cache) the resident-decode buffer ring.
    ///
    /// Returns `Some(&pool)` when the ring fits within 1% of the
    /// device-local heap and every slot allocation succeeds.
    /// Returns `None` (after a one-time `tracing::warn!`) when the
    /// device can't fit the minimum 3 slots — e.g. Strix Halo near
    /// its 16 GiB UMA limit. The `None` outcome is cached so the
    /// per-call kt `kiln_tensor::Tensor` fallback does not re-probe on every decode
    /// step.
    pub fn decode_resident_pool(
        &self,
        max_hidden: usize,
        max_intermediate: usize,
        max_batch: usize,
    ) -> Option<&Arc<kiln_vulkan_kernel::DecodeResidentPool>> {
        let dev = self.vulkan_device.as_ref()?;
        self.decode_resident_pool
            .get_or_init(|| {
                match kiln_vulkan_kernel::DecodeResidentPool::try_new(
                    dev,
                    max_hidden,
                    max_intermediate,
                    max_batch,
                ) {
                    Ok(Some(pool)) => Some(Arc::new(pool)),
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Vulkan-resident decode pool construction errored; \
                             falling back to per-call kt Tensor path"
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Lazily construct (and cache) the Vulkan-resident paged KV cache
    /// for the given geometry.
    ///
    /// `num_full_attn_layers`, `num_blocks`, `block_size`, `num_kv_heads`,
    /// `head_dim` mirror the legacy `PagedKvCache::new` geometry — the
    /// resident cache is a device-local sibling laid out element-for-
    /// element compatible with the existing paged-attn shaders.
    ///
    /// Returns `Some(&cache)` when the device allocation succeeds. Returns
    /// `None` (with a one-time `tracing::warn!`) when the device can't fit
    /// the geometry; callers fall back to the legacy CPU-backed pool.
    /// The `None` outcome is cached on the backend so subsequent calls
    /// don't re-probe.
    pub fn vk_paged_kv_cache(
        &self,
        num_full_attn_layers: usize,
        num_blocks: usize,
        block_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Option<&Arc<kiln_vulkan_kernel::VkPagedKvCache>> {
        let dev = self.vulkan_device.as_ref()?;
        self.vk_paged_kv_cache
            .get_or_init(|| {
                match kiln_vulkan_kernel::VkPagedKvCache::try_new(
                    dev,
                    num_full_attn_layers,
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    head_dim,
                ) {
                    Ok(Some(cache)) => Some(Arc::new(cache)),
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "Vulkan-resident paged KV cache construction errored; \
                             falling back to legacy CPU-backed pool"
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Test (without mutating) whether the given full-attention layer
    /// has already been seeded into the resident KV cache for this
    /// session. Returns true after the first successful call to
    /// `mark_full_attn_layer_seeded` for the same `layer_idx`.
    pub fn full_attn_layer_seeded(&self, layer_idx: usize) -> bool {
        match self.seeded_full_attn_layers.lock() {
            Ok(g) => g.contains(&layer_idx),
            Err(_) => false,
        }
    }

    /// Mark the given full-attention layer as having been seeded into
    /// the resident KV cache for this session.
    pub fn mark_full_attn_layer_seeded(&self, layer_idx: usize) {
        if let Ok(mut g) = self.seeded_full_attn_layers.lock() {
            g.insert(layer_idx);
        }
    }

    /// Reset the seeded-layer set. Tests / multi-session callers call
    /// this when the kt paged cache may have been reset between
    /// resident decode calls; otherwise the resident path keeps reusing
    /// stale K/V state.
    pub fn reset_full_attn_seeded(&self) {
        if let Ok(mut g) = self.seeded_full_attn_layers.lock() {
            g.clear();
        }
        if let Ok(mut g) = self.seeded_resident_decode_rows.lock() {
            g.clear();
        }
    }

    pub fn resident_decode_row_seeded(&self, layer_idx: usize, row_id: u64) -> bool {
        match self.seeded_resident_decode_rows.lock() {
            Ok(g) => g.contains(&(layer_idx, row_id)),
            Err(_) => false,
        }
    }

    pub fn mark_resident_decode_row_seeded(&self, layer_idx: usize, row_id: u64) {
        if let Ok(mut g) = self.seeded_resident_decode_rows.lock() {
            g.insert((layer_idx, row_id));
        }
    }

    pub fn reset_resident_decode_row_seeded(&self) {
        if let Ok(mut g) = self.seeded_resident_decode_rows.lock() {
            g.clear();
        }
    }

    /// Note this resident decode call's `start_pos`. Within one
    /// request the resident path advances `start_pos` by 1 per token;
    /// a discontinuity (first call after server boot, or a new
    /// request whose first decode step doesn't follow the previous
    /// request's last step) signals a fresh session — at that point
    /// we clear the per-layer seeded flags so the next per-layer call
    /// re-seeds the resident `VkPagedKvCache` from this request's
    /// prefill. Returns `true` when a new session was detected.
    ///
    /// Without this, a second `/v1/chat/completions` request reuses
    /// the persistent `VkPagedKvCache` slot data the previous request
    /// wrote (because `seeded_full_attn_layers`, keyed only by layer
    /// index, is stuck at `true` from request 1) — the model then
    /// reasons about the prior request's prompt.
    pub fn note_resident_session(&self, start_pos: usize) -> bool {
        let mut last = match self.last_resident_start_pos.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let is_new_session = match *last {
            // Same `start_pos` = another layer's call within the same
            // decode token. Same `start_pos + 1` = the next decode
            // step in the same request. Anything else = boundary.
            Some(prev) => start_pos != prev && start_pos != prev.wrapping_add(1),
            None => true,
        };
        // Only advance on a strictly-incrementing step so multi-layer
        // calls within the same token don't trigger a spurious reset.
        if match *last {
            Some(prev) => start_pos == prev.wrapping_add(1) || is_new_session,
            None => true,
        } {
            *last = Some(start_pos);
        }
        drop(last);
        if is_new_session {
            self.reset_full_attn_seeded();
            self.reset_linear_attn_seeded();
        }
        is_new_session
    }

    pub fn reset_linear_attn_seeded(&self) {
        if let Ok(mut g) = self.seeded_linear_attn_layers_kt.lock() {
            g.clear();
        }
    }

    pub fn linear_attn_recurrent_state_buffer_kt(
        &self,
        key: kiln_tensor::TensorId,
        bytes: u64,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let dev = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let mut g = self
            .linear_attn_recurrent_state_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("kt recurrent state mutex poisoned"))?;
        if let Some(buf) = g.get(&key) {
            if buf.size() >= bytes {
                return Ok(Arc::clone(buf));
            }
        }
        let buf = kiln_vulkan_kernel::VulkanBuffer::create_device_local(
            dev.device(),
            dev.device_local_mem_type(),
            bytes,
        )
        .context("alloc kt linear-attn recurrent state buffer")?;
        let arc = Arc::new(buf);
        g.insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    pub fn linear_attn_conv_state_buffer_kt(
        &self,
        key: kiln_tensor::TensorId,
        bytes: u64,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let dev = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let mut g = self
            .linear_attn_conv_state_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("kt conv state mutex poisoned"))?;
        if let Some(buf) = g.get(&key) {
            if buf.size() >= bytes {
                return Ok(Arc::clone(buf));
            }
        }
        let buf = kiln_vulkan_kernel::VulkanBuffer::create_device_local(
            dev.device(),
            dev.device_local_mem_type(),
            bytes,
        )
        .context("alloc kt linear-attn conv state buffer")?;
        let arc = Arc::new(buf);
        g.insert(key, Arc::clone(&arc));
        Ok(arc)
    }

    pub fn linear_attn_layer_seeded_kt(&self, key: kiln_tensor::TensorId) -> bool {
        match self.seeded_linear_attn_layers_kt.lock() {
            Ok(g) => g.contains(&key),
            Err(_) => false,
        }
    }

    pub fn mark_linear_attn_layer_seeded_kt(&self, key: kiln_tensor::TensorId) {
        if let Ok(mut g) = self.seeded_linear_attn_layers_kt.lock() {
            g.insert(key);
        }
    }

    fn assemble_linear_attn_state_batch_kt(
        &self,
        state_map: &Mutex<
            HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>,
        >,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
        label: &'static str,
    ) -> Result<bool> {
        if row_keys.is_empty() {
            return Ok(false);
        }
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(false);
        };
        let row_buffers = {
            let g = state_map
                .lock()
                .map_err(|_| anyhow::anyhow!("kt {label} state mutex poisoned"))?;
            let mut out = Vec::with_capacity(row_keys.len());
            for key in row_keys {
                let Some(buf) = g.get(key) else {
                    return Ok(false);
                };
                out.push(Arc::clone(buf));
            }
            out
        };
        let batch_buffer =
            kiln_vulkan_kernel::kernels::copy_device_buffer_rows_to_batch(vk_device, &row_buffers)
                .with_context(|| format!("assemble kt {label} state batch rows"))?;
        state_map
            .lock()
            .map_err(|_| anyhow::anyhow!("kt {label} state mutex poisoned"))?
            .insert(batch_key, batch_buffer);
        Ok(true)
    }

    fn scatter_linear_attn_state_batch_kt(
        &self,
        state_map: &Mutex<
            HashMap<kiln_tensor::TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>,
        >,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
        label: &'static str,
    ) -> Result<bool> {
        if row_keys.is_empty() {
            return Ok(false);
        }
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(false);
        };
        let batch_buffer = {
            let g = state_map
                .lock()
                .map_err(|_| anyhow::anyhow!("kt {label} state mutex poisoned"))?;
            let Some(buf) = g.get(&batch_key) else {
                return Ok(false);
            };
            Arc::clone(buf)
        };
        let row_buffers = kiln_vulkan_kernel::kernels::split_device_buffer_batch_rows(
            vk_device,
            &batch_buffer,
            row_keys.len(),
        )
        .with_context(|| format!("scatter kt {label} state batch rows"))?;
        {
            let mut g = state_map
                .lock()
                .map_err(|_| anyhow::anyhow!("kt {label} state mutex poisoned"))?;
            for (key, buf) in row_keys.iter().copied().zip(row_buffers.into_iter()) {
                g.insert(key, buf);
            }
        }
        Ok(true)
    }

    fn assemble_linear_attn_recurrent_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
    ) -> Result<bool> {
        self.assemble_linear_attn_state_batch_kt(
            &self.linear_attn_recurrent_state_kt,
            row_keys,
            batch_key,
            "recurrent",
        )
    }

    fn assemble_linear_attn_conv_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
    ) -> Result<bool> {
        self.assemble_linear_attn_state_batch_kt(
            &self.linear_attn_conv_state_kt,
            row_keys,
            batch_key,
            "conv",
        )
    }

    fn scatter_linear_attn_recurrent_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
    ) -> Result<bool> {
        self.scatter_linear_attn_state_batch_kt(
            &self.linear_attn_recurrent_state_kt,
            batch_key,
            row_keys,
            "recurrent",
        )
    }

    fn scatter_linear_attn_conv_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
    ) -> Result<bool> {
        self.scatter_linear_attn_state_batch_kt(
            &self.linear_attn_conv_state_kt,
            batch_key,
            row_keys,
            "conv",
        )
    }

    pub fn assemble_linear_attn_gdn_state_batch_kt(
        &self,
        row_keys: &[kiln_tensor::TensorId],
        batch_key: kiln_tensor::TensorId,
    ) -> Result<bool> {
        let recurrent_ok =
            self.assemble_linear_attn_recurrent_state_batch_kt(row_keys, batch_key)?;
        if !recurrent_ok {
            return Ok(false);
        }
        let conv_ok = self.assemble_linear_attn_conv_state_batch_kt(row_keys, batch_key)?;
        if !conv_ok {
            return Ok(false);
        }
        self.mark_linear_attn_layer_seeded_kt(batch_key);
        Ok(true)
    }

    pub fn scatter_linear_attn_gdn_state_batch_kt(
        &self,
        batch_key: kiln_tensor::TensorId,
        row_keys: &[kiln_tensor::TensorId],
    ) -> Result<bool> {
        let recurrent_ok =
            self.scatter_linear_attn_recurrent_state_batch_kt(batch_key, row_keys)?;
        if !recurrent_ok {
            return Ok(false);
        }
        let conv_ok = self.scatter_linear_attn_conv_state_batch_kt(batch_key, row_keys)?;
        if !conv_ok {
            return Ok(false);
        }
        if let Ok(mut seeded) = self.seeded_linear_attn_layers_kt.lock() {
            for key in row_keys {
                seeded.insert(*key);
            }
        }
        Ok(true)
    }

    pub fn seed_linear_attn_gdn_state_kt(
        &self,
        recurrent_t: &kiln_tensor::Tensor,
        conv_t: &kiln_tensor::Tensor,
    ) -> Result<bool> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(false);
        };
        let key = recurrent_t.id();
        let recurrent_bytes = (recurrent_t.elem_count() * std::mem::size_of::<f32>()) as u64;
        let conv_bytes = (conv_t.elem_count() * std::mem::size_of::<f32>()) as u64;
        let recurrent_buf = self.linear_attn_recurrent_state_buffer_kt(key, recurrent_bytes)?;
        let conv_buf = self.linear_attn_conv_state_buffer_kt(key, conv_bytes)?;
        crate::vk_decode_resident::seed_recurrent_state_kt(
            vk_device,
            &recurrent_buf,
            recurrent_t,
        )?;
        crate::vk_decode_resident::seed_conv_state_kt(vk_device, &conv_buf, conv_t)?;
        self.mark_linear_attn_layer_seeded_kt(key);
        Ok(true)
    }

    pub fn has_linear_attn_gdn_state_kt(&self, key: kiln_tensor::TensorId) -> bool {
        if !self.linear_attn_layer_seeded_kt(key) {
            return false;
        }
        let recurrent_present = self
            .linear_attn_recurrent_state_kt
            .lock()
            .map(|g| g.contains_key(&key))
            .unwrap_or(false);
        let conv_present = self
            .linear_attn_conv_state_kt
            .lock()
            .map(|g| g.contains_key(&key))
            .unwrap_or(false);
        recurrent_present && conv_present
    }

    /// Acquire (or lazily create) a persistent scratch
    /// [`VulkanBuffer`] under the given role key, sized to at least
    /// `min_bytes`. The same buffer is returned on every subsequent
    /// call with the same role, so the resident decode block helpers
    /// pay zero allocation cost on the steady-state hot path.
    ///
    /// If a previously-cached buffer for the role is too small for
    /// the new `min_bytes` it is replaced.
    pub fn acquire_resident_scratch(
        &self,
        role: &'static str,
        min_bytes: u64,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let dev = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let mut g = self
            .resident_scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("resident scratch mutex poisoned"))?;
        if let Some(buf) = g.get(role) {
            if buf.size() >= min_bytes {
                return Ok(Arc::clone(buf));
            }
        }
        let buf = kiln_vulkan_kernel::VulkanBuffer::create_device_local(
            dev.device(),
            dev.device_local_mem_type(),
            min_bytes.max(4),
        )
        .with_context(|| format!("alloc resident scratch '{role}'"))?;
        let arc = Arc::new(buf);
        g.insert(role, Arc::clone(&arc));
        Ok(arc)
    }

    /// Host-visible variant of `acquire_resident_scratch`. Used by the
    /// native decode orchestrator to keep a persistent readback
    /// staging buffer (for logits) — folding the readback's
    /// `cmd_copy_buffer` into the main `CommandBatch` so the post-
    /// submit step is just a `map_memory` rather than a fresh queue
    /// submission.
    pub fn acquire_resident_scratch_host_visible(
        &self,
        role: &'static str,
        min_bytes: u64,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let dev = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let mut g = self
            .resident_scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("resident scratch mutex poisoned"))?;
        if let Some(buf) = g.get(role) {
            if buf.size() >= min_bytes {
                return Ok(Arc::clone(buf));
            }
        }
        let buf = kiln_vulkan_kernel::VulkanBuffer::create_host_visible(
            dev.device(),
            dev.host_visible_mem_type(),
            min_bytes.max(4),
        )
        .with_context(|| format!("alloc host-visible resident scratch '{role}'"))?;
        let arc = Arc::new(buf);
        g.insert(role, Arc::clone(&arc));
        Ok(arc)
    }

    /// kt-native f32 weight buffer cache: keys the buffer
    /// cache on the **kt** `TensorId` (stable for the model's lifetime) and
    /// extracts f32 bytes straight from kt storage on a miss — no candle
    /// bridge, so a cache hit (every token after the first) does zero copy
    /// work. (#1082)
    pub fn cached_f32_weight_buffer_kt(
        &self,
        weight: &kiln_tensor::Tensor,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let key = weight.id();
        {
            let cache = self
                .weight_cache_kt
                .lock()
                .map_err(|_| anyhow::anyhow!("Vulkan kt weight cache mutex poisoned"))?;
            if let Some(buffer) = cache.get(&key) {
                return Ok(Arc::clone(buffer));
            }
        }
        let weight_f32_data: Vec<f32> = weight
            .flatten_all()
            .context("kt weight flatten_all")?
            .to_dtype(kiln_tensor::DType::F32)
            .context("kt weight to f32")?
            .to_vec1::<f32>()
            .context("kt weight to_vec1 f32")?;
        let buffer = Arc::new(
            kiln_vulkan_kernel::kernels::upload_f32_buffer_from_slice(vk_device, &weight_f32_data)
                .context("upload kt f32 weight to Vulkan")?,
        );
        let mut cache = self
            .weight_cache_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("Vulkan kt weight cache mutex poisoned"))?;
        Ok(Arc::clone(cache.entry(key).or_insert(buffer)))
    }

    /// kt-native bf16-packed weight buffer cache. Stable-kt-id keying;
    /// extracts bf16 straight from kt storage on a miss. (#1082)
    pub fn cached_bf16_packed_weight_buffer_kt(
        &self,
        weight: &kiln_tensor::Tensor,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let key = weight.id();
        {
            let cache = self
                .bf16_packed_weight_cache_kt
                .lock()
                .map_err(|_| anyhow::anyhow!("Vulkan kt packed bf16 weight cache mutex poisoned"))?;
            if let Some(buffer) = cache.get(&key) {
                return Ok(Arc::clone(buffer));
            }
        }
        anyhow::ensure!(
            weight.dtype() == kiln_tensor::DType::BF16,
            "packed bf16 upload requires BF16 kt tensor, got {:?}",
            weight.dtype()
        );
        let weight_bf16_data: Vec<half::bf16> = weight
            .flatten_all()
            .context("kt bf16 weight flatten_all")?
            .to_vec1::<half::bf16>()
            .context("kt bf16 weight to_vec1")?;
        let buffer = Arc::new(
            kiln_vulkan_kernel::kernels::upload_bf16_packed_buffer_from_slice(
                vk_device,
                &weight_bf16_data,
            )
            .context("upload kt packed BF16 weight to Vulkan")?,
        );
        let mut cache = self
            .bf16_packed_weight_cache_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("Vulkan kt packed bf16 weight cache mutex poisoned"))?;
        Ok(Arc::clone(cache.entry(key).or_insert(buffer)))
    }

    /// kt-native: whether to use the bf16-packed linear-weight decode path.
    fn use_bf16_packed_linear_weight_kt(&self, weight: &kiln_tensor::Tensor) -> bool {
        self.bf16_packed_linear_weights_enabled && weight.dtype() == kiln_tensor::DType::BF16
    }

    fn use_bf16_packed_gdn_in_proj_weights_kt(&self, weights: &[&kiln_tensor::Tensor]) -> bool {
        self.bf16_packed_gdn_in_proj_weights_enabled
            && weights
                .iter()
                .all(|weight| weight.dtype() == kiln_tensor::DType::BF16)
    }

    fn use_bf16_packed_full_attn_qkv_weights_kt(
        &self,
        weights: &[&kiln_tensor::Tensor],
    ) -> bool {
        self.bf16_packed_full_attn_qkv_weights_enabled
            && weights
                .iter()
                .all(|weight| weight.dtype() == kiln_tensor::DType::BF16)
    }

    fn use_bf16_packed_mlp_decode_weights_kt(&self, weights: &[&kiln_tensor::Tensor]) -> bool {
        self.bf16_packed_mlp_decode_weights_enabled
            && weights
                .iter()
                .all(|weight| weight.dtype() == kiln_tensor::DType::BF16)
    }

    fn prewarm_f32_weight_kt(
        &self,
        name: &str,
        weight: &kiln_tensor::Tensor,
        count: &mut usize,
        bytes: &mut usize,
    ) -> Result<()> {
        self.cached_f32_weight_buffer_kt(weight)
            .with_context(|| format!("prewarm Vulkan decode weight {name}"))?;
        *count += 1;
        *bytes += weight.elem_count() * std::mem::size_of::<f32>();
        Ok(())
    }

    fn prewarm_bf16_packed_weight_kt(
        &self,
        name: &str,
        weight: &kiln_tensor::Tensor,
        count: &mut usize,
        bytes: &mut usize,
    ) -> Result<()> {
        self.cached_bf16_packed_weight_buffer_kt(weight)
            .with_context(|| format!("prewarm Vulkan packed BF16 decode weight {name}"))?;
        *count += 1;
        *bytes += weight.elem_count().div_ceil(2) * std::mem::size_of::<u32>();
        Ok(())
    }

    fn prewarm_linear_weight_kt(
        &self,
        name: &str,
        weight: &kiln_tensor::Tensor,
        f32_count: &mut usize,
        f32_bytes: &mut usize,
        bf16_count: &mut usize,
        bf16_bytes: &mut usize,
    ) -> Result<()> {
        if self.use_bf16_packed_linear_weight_kt(weight) {
            self.prewarm_bf16_packed_weight_kt(name, weight, bf16_count, bf16_bytes)
        } else {
            self.prewarm_f32_weight_kt(name, weight, f32_count, f32_bytes)
        }
    }

    fn prewarm_gdn_in_proj_weight_kt(
        &self,
        name: &str,
        weight: &kiln_tensor::Tensor,
        f32_count: &mut usize,
        f32_bytes: &mut usize,
        bf16_count: &mut usize,
        bf16_bytes: &mut usize,
    ) -> Result<()> {
        if self.use_bf16_packed_gdn_in_proj_weights_kt(&[weight]) {
            self.prewarm_bf16_packed_weight_kt(name, weight, bf16_count, bf16_bytes)
        } else {
            self.prewarm_f32_weight_kt(name, weight, f32_count, f32_bytes)
        }
    }

    fn prewarm_full_attn_qkv_weights_kt(
        &self,
        layer_idx: usize,
        q_weight_t: &kiln_tensor::Tensor,
        k_weight_t: &kiln_tensor::Tensor,
        v_weight_t: &kiln_tensor::Tensor,
        f32_count: &mut usize,
        f32_bytes: &mut usize,
        bf16_count: &mut usize,
        bf16_bytes: &mut usize,
    ) -> Result<()> {
        let weights = [
            ("q_proj_t", q_weight_t),
            ("k_proj_t", k_weight_t),
            ("v_proj_t", v_weight_t),
        ];
        if self.use_bf16_packed_full_attn_qkv_weights_kt(&[q_weight_t, k_weight_t, v_weight_t]) {
            for (suffix, weight) in weights {
                self.prewarm_bf16_packed_weight_kt(
                    &format!("layers.{layer_idx}.attention.{suffix}"),
                    weight,
                    bf16_count,
                    bf16_bytes,
                )?;
            }
        } else {
            for (suffix, weight) in weights {
                self.prewarm_f32_weight_kt(
                    &format!("layers.{layer_idx}.attention.{suffix}"),
                    weight,
                    f32_count,
                    f32_bytes,
                )?;
            }
        }
        Ok(())
    }

    fn prewarm_mlp_decode_weights_kt(
        &self,
        layer_idx: usize,
        gate_weight_t: &kiln_tensor::Tensor,
        up_weight_t: &kiln_tensor::Tensor,
        down_weight_t: &kiln_tensor::Tensor,
        f32_count: &mut usize,
        f32_bytes: &mut usize,
        bf16_count: &mut usize,
        bf16_bytes: &mut usize,
    ) -> Result<()> {
        let weights = [
            ("gate_proj_t", gate_weight_t),
            ("up_proj_t", up_weight_t),
            ("down_proj_t", down_weight_t),
        ];
        if self.use_bf16_packed_mlp_decode_weights_kt(&[gate_weight_t, up_weight_t, down_weight_t])
        {
            for (suffix, weight) in weights {
                self.prewarm_bf16_packed_weight_kt(
                    &format!("layers.{layer_idx}.mlp.{suffix}"),
                    weight,
                    bf16_count,
                    bf16_bytes,
                )?;
            }
            for (suffix, weight) in weights {
                self.prewarm_f32_weight_kt(
                    &format!("layers.{layer_idx}.mlp.{suffix}"),
                    weight,
                    f32_count,
                    f32_bytes,
                )?;
            }
        } else {
            for (suffix, weight) in weights {
                self.prewarm_f32_weight_kt(
                    &format!("layers.{layer_idx}.mlp.{suffix}"),
                    weight,
                    f32_count,
                    f32_bytes,
                )?;
            }
        }
        Ok(())
    }

    /// Dispatch FlashAttention-2 prefill kernel via Vulkan.
    fn flash_attn_prefill_vulkan(
        &self,
        q: &kiln_tensor::Tensor,
        k: &kiln_tensor::Tensor,
        v: &kiln_tensor::Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let Ok((batch, seq_len, num_heads, head_dim)) = q.dims4() else {
            return Ok(None);
        };
        let Ok((k_batch, kv_len, k_heads, k_head_dim)) = k.dims4() else {
            return Ok(None);
        };
        let Ok((v_batch, v_len, v_heads, v_head_dim)) = v.dims4() else {
            return Ok(None);
        };
        if head_dim > 256
            || kv_len != seq_len
            || k_batch != batch
            || v_batch != batch
            || v_len != seq_len
            || k_heads != num_heads
            || v_heads != num_heads
            || k_head_dim != head_dim
            || v_head_dim != head_dim
        {
            return Ok(None);
        }

        // Resident buffer-based SDPA for the common single-sequence causal
        // case: zero-copy bridge q/k/v → fused on-device `vk_sdpa_prefill`
        // (permute → batched matmul → scale → causal-mask → softmax → matmul,
        // all resident) → bridge back. No host round-trip. batch>1 (the kernel
        // flattens query rows to one sequence, so cross-batch attention would
        // be wrong) and non-causal fall through to the bytes path below.
        if causal
            && batch == 1
            && matches!(q.device(), kiln_tensor::Device::Vulkan(_))
            && matches!(k.device(), kiln_tensor::Device::Vulkan(_))
            && matches!(v.device(), kiln_tensor::Device::Vulkan(_))
        {
            match resident_sdpa_prefill_b1(q, k, v, seq_len, num_heads, head_dim, softmax_scale) {
                Ok(out) => return Ok(Some(out)),
                Err(e) => {
                    if std::env::var("KILN_VK_TRACE").is_ok() {
                        eprintln!("[vk] resident sdpa_prefill fell back to bytes: {e}");
                    }
                }
            }
        }

        let in_dtype = q.dtype();
        let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
        let k_data = kt_tensor_to_f32_bytes_with_shape(k)?.0;
        let v_data = kt_tensor_to_f32_bytes_with_shape(v)?.0;
        let out_data = kiln_vulkan_kernel::kernels::dispatch_sdpa_prefill_f32_bytes(
            vk_device,
            &q_data,
            &k_data,
            &v_data,
            batch,
            seq_len,
            num_heads,
            head_dim,
            softmax_scale,
            causal,
        )?;
        let out_f32 = kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, seq_len, num_heads, head_dim],
            kiln_tensor::DType::F32,
        )?;

        let out = if in_dtype == kiln_tensor::DType::F32 {
            out_f32
        } else {
            out_f32.to_dtype(in_dtype)?
        };
        // The SDPA result is currently materialized host-side (the bytes-based
        // kernel dispatch). Keep `attn_output` on q's compute device so the
        // downstream gate / o-proj run on-device instead of mismatching
        // (vulkan gate × cpu attn_output). NOTE: the q/k/v inputs are still
        // bounced to host bytes above — a buffer-resident SDPA dispatch
        // (zero-copy q/k/v + device-resident output) is the perf follow-up to
        // remove the host round-trip entirely.
        let out = if out.device() != q.device() {
            out.to_device(q.device())?
        } else {
            out
        };
        Ok(Some(out))
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
        if self.has_vulkan() { "vulkan" } else { "cpu" }
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

    fn materialize_gdn_recurrent_resident_state(&self, state_kt: &mut kiln_tensor::Tensor) -> Result<()> {
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
        *state_kt = kt_tensor_from_f32_bytes(
            &data,
            &state_dims,
            state_dtype,
        )?;
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

    fn dispatch_sgd_step(&self, param: &kiln_tensor::Tensor, grad: &kiln_tensor::Tensor, lr: f32) -> Result<bool> {
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
        if !self.has_vulkan()
            || !matches!(
                q.dtype(),
                kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
            )
        {
            return Ok(None);
        }
        self.flash_attn_prefill_vulkan(q, k, v, softmax_scale, causal)
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
        if !self.has_vulkan()
            || !self.paged_attn_decode_batch_enabled
            || q.dtype() != kiln_tensor::DType::F32
            || k_pool.dtype() != kiln_tensor::DType::F32
            || v_pool.dtype() != kiln_tensor::DType::F32
        {
            return Ok(None);
        }
        if !causal {
            return Ok(None);
        }
        if !matches!(q.device(), kiln_tensor::Device::Cpu)
            || !matches!(k_pool.device(), kiln_tensor::Device::Cpu)
            || !matches!(v_pool.device(), kiln_tensor::Device::Cpu)
            || !matches!(block_table.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }

        let Ok((batch, q_len, num_heads, head_dim)) = q.dims4() else {
            return Ok(None);
        };
        let Ok((total_slots, num_kv_heads, k_head_dim)) = k_pool.dims3() else {
            return Ok(None);
        };
        let Ok(v_dims) = v_pool.dims3() else {
            return Ok(None);
        };
        let Ok((bt_batch, max_blocks_per_seq)) = block_table.dims2() else {
            return Ok(None);
        };
        if batch == 0
            || q_len != 1
            || total_seqlen_k == 0
            || page_block_size == 0
            || head_dim > 256
            || k_head_dim != head_dim
            || v_dims != (total_slots, num_kv_heads, head_dim)
            || num_heads % num_kv_heads != 0
            || bt_batch != batch
            || total_seqlen_k.div_ceil(page_block_size) > max_blocks_per_seq
        {
            return Ok(None);
        }

        let block_data = block_table
            .flatten_all()
            .context("Vulkan paged decode: flatten block_table")?
            .to_dtype(kiln_tensor::DType::U32)
            .context("Vulkan paged decode: block_table to u32")?
            .to_vec1::<u32>()
            .context("Vulkan paged decode: read block_table")?;
        if block_data.len() != batch * max_blocks_per_seq {
            return Ok(None);
        }

        for row in 0..batch {
            let blocks_needed = total_seqlen_k.div_ceil(page_block_size).max(1);
            for block_idx in 0..blocks_needed {
                let block = block_data[row * max_blocks_per_seq + block_idx] as usize;
                let last_pos_in_block = if block_idx == blocks_needed - 1 {
                    total_seqlen_k - block_idx * page_block_size - 1
                } else {
                    page_block_size - 1
                };
                let last_slot = block
                    .checked_mul(page_block_size)
                    .and_then(|base| base.checked_add(last_pos_in_block))
                    .context("Vulkan paged decode slot index overflow")?;
                if last_slot >= total_slots {
                    return Ok(None);
                }
            }
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
        let k_pool_data = kt_tensor_to_f32_bytes_with_shape(k_pool)?.0;
        let v_pool_data = kt_tensor_to_f32_bytes_with_shape(v_pool)?.0;
        let seq_lens = vec![
            u32::try_from(total_seqlen_k)
                .context("Vulkan paged decode total_seqlen_k exceeds u32")?;
            batch
        ];

        let out_data = dispatch_vulkan_paged_decode_bytes(
            vk_device,
            &q_data,
            &k_pool_data,
            &v_pool_data,
            batch,
            num_heads,
            head_dim,
            total_slots,
            num_kv_heads,
            &block_data,
            &seq_lens,
            max_blocks_per_seq,
            page_block_size,
            softmax_scale,
        )
        .context("Vulkan paged decode batch-paged kernel failed")?;

        Ok(Some(kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, 1, num_heads, head_dim],
            kiln_tensor::DType::F32,
        )?))
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
        if !self.has_vulkan()
            || !self.paged_attn_decode_batch_enabled
            || q.dtype() != kiln_tensor::DType::F32
            || k_pool.dtype() != kiln_tensor::DType::F32
            || v_pool.dtype() != kiln_tensor::DType::F32
        {
            return Ok(None);
        }
        if !causal {
            return Ok(None);
        }
        if !matches!(q.device(), kiln_tensor::Device::Cpu)
            || !matches!(k_pool.device(), kiln_tensor::Device::Cpu)
            || !matches!(v_pool.device(), kiln_tensor::Device::Cpu)
            || !matches!(block_table.device(), kiln_tensor::Device::Cpu)
            || !matches!(seqused_k.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }
        if !paged_decode_gpu_gather_enabled() {
            anyhow::bail!("Vulkan paged decode GPU block-table gather disabled");
        }

        let Ok((batch, q_len, num_heads, head_dim)) = q.dims4() else {
            return Ok(None);
        };
        let Ok((total_slots, num_kv_heads, k_head_dim)) = k_pool.dims3() else {
            return Ok(None);
        };
        let Ok(v_dims) = v_pool.dims3() else {
            return Ok(None);
        };
        let Ok((bt_batch, max_blocks_per_seq)) = block_table.dims2() else {
            return Ok(None);
        };
        let Ok(seq_count) = seqused_k.dims1() else {
            return Ok(None);
        };
        if batch == 0
            || q_len != 1
            || head_dim > 256
            || k_head_dim != head_dim
            || v_dims != (total_slots, num_kv_heads, head_dim)
            || num_heads % num_kv_heads != 0
            || bt_batch != batch
            || seq_count != batch
            || page_block_size == 0
            || max_seqlen_k == 0
            || max_seqlen_k.div_ceil(page_block_size) > max_blocks_per_seq
        {
            return Ok(None);
        }

        let block_data = block_table
            .flatten_all()?
            .to_dtype(kiln_tensor::DType::U32)?
            .to_vec1::<u32>()?;
        let seq_i64 = seqused_k
            .flatten_all()?
            .to_dtype(kiln_tensor::DType::I64)?
            .to_vec1::<i64>()?;
        let mut seq_lens = Vec::with_capacity(batch);
        for row in 0..batch {
            let row_len = usize::try_from(seq_i64[row])
                .context("Vulkan paged decode seqused_k contains negative length")?;
            if row_len == 0 || row_len > max_seqlen_k {
                return Ok(None);
            }
            seq_lens.push(
                u32::try_from(row_len).context("Vulkan paged decode row length exceeds u32")?,
            );
        }
        // Bounds-check the block_table entries that the kernel will follow.
        // We don't want the shader to OOB-read the K/V pool, so reject any
        // out-of-range (block, offset) we can prove invalid from host state.
        // Only the slots actually visited (`pos < row_len`) need to be valid.
        for row in 0..batch {
            let row_len = seq_lens[row] as usize;
            let blocks_needed = row_len.div_ceil(page_block_size).max(1);
            for block_idx in 0..blocks_needed {
                let block = block_data[row * max_blocks_per_seq + block_idx] as usize;
                let last_pos_in_block = if block_idx == blocks_needed - 1 {
                    row_len - block_idx * page_block_size - 1
                } else {
                    page_block_size - 1
                };
                let last_slot = block
                    .checked_mul(page_block_size)
                    .and_then(|base| base.checked_add(last_pos_in_block))
                    .context("Vulkan paged decode slot index overflow")?;
                if last_slot >= total_slots {
                    return Ok(None);
                }
            }
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let q_data = kt_tensor_to_f32_bytes_with_shape(q)?.0;
        let k_pool_data = kt_tensor_to_f32_bytes_with_shape(k_pool)?.0;
        let v_pool_data = kt_tensor_to_f32_bytes_with_shape(v_pool)?.0;
        let out_data = dispatch_vulkan_paged_decode_bytes(
            vk_device,
            &q_data,
            &k_pool_data,
            &v_pool_data,
            batch,
            num_heads,
            head_dim,
            total_slots,
            num_kv_heads,
            &block_data,
            &seq_lens,
            max_blocks_per_seq,
            page_block_size,
            softmax_scale,
        )
        .context("paged_attn_decode_batch_paged kernel failed")?;
        Ok(Some(kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, 1, num_heads, head_dim],
            kiln_tensor::DType::F32,
        )?))
    }

    fn gdn_in_proj_decode(
        &self,
        x: &kiln_tensor::Tensor,
        in_proj_qkv_t: &kiln_tensor::Tensor,
        in_proj_z_t: &kiln_tensor::Tensor,
        in_proj_a_t: &kiln_tensor::Tensor,
        in_proj_b_t: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
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
            let out = kt_tensor_from_f32_bytes(
                &out_data,
                &[batch_d, 1, nv, dv],
                q_dtype,
            )?;
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
        let out = kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, 1, nv, dv],
            q_dtype,
        )?;
        if !skip_state_readback {
            if let Some(sd) = new_state_data {
                *state_kt = kt_tensor_from_f32_bytes(
                    &sd,
                    &state_dims,
                    state_dtype,
                )?;
            }
        }
        Ok(Some(out))
    }

    fn linear_decode(&self, x: &kiln_tensor::Tensor, weight_t: &kiln_tensor::Tensor) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.linear_decode_enabled || x.dtype() != kiln_tensor::DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu) || !matches!(weight_t.device(), kiln_tensor::Device::Cpu) {
            return Ok(None);
        }
        // (#1082) Fully kt-native: read shapes off the kt tensors, extract
        // f32 bytes straight from kt storage, and key the weight buffer cache
        // on the **stable** kt `TensorId`. The old path bridged BOTH x and the
        // (large) weight through `kt_logits_to_candle` every call — minting a
        // fresh candle id per token so the weight cache missed every step and
        // re-uploaded ~1 GB/token. Now the weight uploads exactly once.
        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
            return Ok(None);
        };
        if weight_hidden != hidden {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let row_count = batch * seq_len;
        // x is [batch, seq_len, hidden] contiguous F32; the kernel consumes a
        // flat [row_count, hidden] f32 buffer, so the [.,1,.] reshape the candle
        // path did is a no-op on the bytes — extract them straight from kt.
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let packed = self.use_bf16_packed_linear_weight_kt(weight_t);
        let weight_buf = if packed {
            self.cached_bf16_packed_weight_buffer_kt(weight_t)?
        } else {
            self.cached_f32_weight_buffer_kt(weight_t)?
        };
        let out_data = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            row_count,
            hidden,
            out_dim,
            packed,
        )
        .context("linear_decode kernel failed")?;
        Ok(Some(kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, seq_len, out_dim],
            kiln_tensor::DType::F32,
        )?))
    }

    fn linear_prefill_apply(&self, _x: &kiln_tensor::Tensor, _weight_t: &kiln_tensor::Tensor) -> Result<Option<kiln_tensor::Tensor>> {
        // (#1082) Decline. This hook previously routed the training-time
        // projection matmul through `VulkanLinearOp` (a
        // `candle_core::CustomOp1`) so candle's `loss.backward()` could
        // produce the input gradient. With the kt autograd tape
        // (`kiln_autograd`) as the sole grad producer that candle autograd
        // island is gone — the projection matmul is recorded onto the tape
        // by the portable kt matmul path in forward.rs, and
        // `Tape::backward()` produces the gradient. Returning `Ok(None)`
        // routes the caller to that kt-recorded path.
        //
        // NOTE: the forward-only inference linear kernel still lives in
        // `linear_decode` (declines tracked tensors); only the
        // autograd-wrapping prefill path is removed here.
        Ok(None)
    }

    fn linear_prefill_apply_offset(
        &self,
        x: &kiln_tensor::Tensor,
        full_weight_t: &kiln_tensor::Tensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<kiln_tensor::Tensor>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.linear_decode_enabled {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu) || !matches!(full_weight_t.device(), kiln_tensor::Device::Cpu) {
            return Ok(None);
        }
        // Only the bf16-packed kernel has an offset variant today; require
        // bf16 weights so the cached buffer matches the dispatch shader.
        if full_weight_t.dtype() != kiln_tensor::DType::BF16 {
            return Ok(None);
        }
        // (#1082) kt-native: the cached-weight offset kernel + FLOP-ceiling
        // sub-chunking run directly on the kt args (the FLCE caller owns its
        // own analytic backward, so this is forward-only).
        let Ok((_batch, _seq_len, hidden_x)) = x.dims3() else {
            return Ok(None);
        };
        let Ok((hidden_w, full_out_dim)) = full_weight_t.dims2() else {
            return Ok(None);
        };
        if hidden_x != hidden_w {
            return Ok(None);
        }
        if chunk_start + chunk_len > full_out_dim {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?
            .clone();
        let weight_buffer = self.cached_bf16_packed_weight_buffer_kt(full_weight_t)?;
        // Promote x to f32 for the kernel (kernel expects f32 input).
        let x_f32 = if x.dtype() == kiln_tensor::DType::F32 {
            x.clone()
        } else {
            x.to_dtype(kiln_tensor::DType::F32)?
        };
        let dims = x_f32.dims().to_vec();
        let row_count: usize = dims[..dims.len() - 1].iter().product();
        let dispatch_x = if dims.len() == 3 && dims[1] == 1 {
            x_f32
        } else {
            x_f32.reshape((row_count, 1usize, hidden_x))?
        };
        // Per-dispatch FLOP guard. FLCE chunks at chunk_size=4096 sit
        // right at the 20 GFLOP ceiling for T=918; longer T or larger
        // chunk_len passed by future callers would put a single submit
        // over the safety limit. Sub-chunk along the chunk_len dim so
        // each submit fits — that's strictly better than bailing to
        // FLCE's CPU fallback because each sub-chunk still uses the
        // same offset kernel with no re-upload of the weight buffer.
        let sub_chunk_len = if dispatch_exceeds_safety_ceiling(row_count, hidden_x, chunk_len) {
            max_chunk_dim_for_flop(row_count.saturating_mul(hidden_x)).min(chunk_len)
        } else {
            chunk_len
        };
        let out = if sub_chunk_len == chunk_len {
            let x_data = kt_tensor_to_f32_bytes_with_shape(&dispatch_x)?.0;
            let out_bytes =
                kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset_bytes(
                    vk_device.as_ref(),
                    &x_data,
                    weight_buffer.as_ref(),
                    row_count,
                    hidden_x,
                    chunk_len,
                    chunk_start,
                    full_out_dim,
                )
                .context("VulkanBackend: linear_prefill_apply_offset dispatch failed")?;
            kt_tensor_from_f32_bytes(
                &out_bytes,
                &[row_count, 1, chunk_len],
                kiln_tensor::DType::F32,
            )?
        } else {
            // One-shot trace so the operator can see when FLCE chunks
            // are themselves being sub-chunked. Combined with the
            // VulkanLinearOp chunking traces, gives a complete picture
            // of which paths are exceeding the safety ceiling.
            static FIRST_OFFSET_SUBCHUNK_LOGGED: std::sync::OnceLock<()> =
                std::sync::OnceLock::new();
            FIRST_OFFSET_SUBCHUNK_LOGGED.get_or_init(|| {
                let total_gflop = (2u64
                    .saturating_mul(row_count as u64)
                    .saturating_mul(hidden_x as u64)
                    .saturating_mul(chunk_len as u64)) as f64
                    / 1.0e9;
                let sub_count = chunk_len.div_ceil(sub_chunk_len);
                tracing::info!(
                    row_count,
                    hidden_x,
                    chunk_len,
                    full_out_dim,
                    total_gflop,
                    sub_chunk_len,
                    sub_count,
                    "linear_prefill_apply_offset first sub-chunked dispatch"
                );
            });
            // Walk chunk_len in sub_chunk_len-sized strides; concat
            // outputs along the last axis. Same kernel/buffer per
            // sub-dispatch, just different `chunk_start` offsets and
            // smaller `chunk_len` per submit.
            let mut sub_outputs: Vec<kiln_tensor::Tensor> = Vec::new();
            let mut sub_offset = 0usize;
            let x_data = kt_tensor_to_f32_bytes_with_shape(&dispatch_x)?.0;
            while sub_offset < chunk_len {
                let cur_len = (chunk_len - sub_offset).min(sub_chunk_len);
                let sub_bytes =
                    kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset_bytes(
                        vk_device.as_ref(),
                        &x_data,
                        weight_buffer.as_ref(),
                        row_count,
                        hidden_x,
                        cur_len,
                        chunk_start + sub_offset,
                        full_out_dim,
                    )
                    .with_context(|| {
                        format!(
                            "VulkanBackend: linear_prefill_apply_offset sub-chunk \
                         (sub_offset={sub_offset}, cur_len={cur_len}, \
                          chunk_start={chunk_start}, chunk_len={chunk_len}) failed"
                        )
                    })?;
                let sub = kt_tensor_from_f32_bytes(
                    &sub_bytes,
                    &[row_count, 1, cur_len],
                    kiln_tensor::DType::F32,
                )?;
                sub_outputs.push(sub);
                sub_offset += cur_len;
            }
            let sub_refs: Vec<&kiln_tensor::Tensor> = sub_outputs.iter().collect();
            kiln_tensor::ops::concat(&sub_refs, 2).context("offset sub-chunk concat")?
        };
        // Output from kernel is `[row_count, 1, chunk_len]`. Restore the
        // caller's leading dims with chunk_len in the last position.
        let mut out_dims = dims;
        *out_dims.last_mut().unwrap() = chunk_len;
        let reshaped = out.reshape(out_dims.as_slice())?;
        Ok(Some(reshaped))
    }

    fn supports_linear_decode_argmax(&self) -> bool {
        self.has_vulkan() && self.linear_decode_enabled
    }

    fn linear_decode_argmax(&self, x: &kiln_tensor::Tensor, weight_t: &kiln_tensor::Tensor) -> Result<Option<u32>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.linear_decode_enabled || x.dtype() != kiln_tensor::DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu) || !matches!(weight_t.device(), kiln_tensor::Device::Cpu) {
            return Ok(None);
        }
        // (#1082) Fully kt-native: the lm_head weight (the 778 MB table) was
        // re-bridged + re-uploaded per token under the candle-id cache; key on
        // the stable kt id so it uploads once.
        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        if batch != 1 || seq_len != 1 {
            return Ok(None);
        }
        let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
            return Ok(None);
        };
        if weight_hidden != hidden {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let token = if self.use_bf16_packed_linear_weight_kt(weight_t) {
            let weight_buf = self.cached_bf16_packed_weight_buffer_kt(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached_bf16_weights_bytes(
                vk_device,
                &x_data,
                &weight_buf,
                hidden,
                out_dim,
            )
        } else {
            let weight_buf = self.cached_f32_weight_buffer_kt(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached_bytes(
                vk_device,
                &x_data,
                &weight_buf,
                hidden,
                out_dim,
            )
        }
        .context("linear_decode_argmax kernel failed")?;
        Ok(Some(token))
    }

    fn supports_linear_decode_argmax_batch(&self) -> bool {
        self.has_vulkan() && self.linear_decode_enabled && self.linear_argmax_batch_enabled
    }

    fn supports_linear_decode_sample(&self, top_k: u32) -> bool {
        // The fused sample kernel only handles top_k in `1..=TOPK_SAMPLE_KERNEL_K_MAX`.
        // Larger requests fall back to the host sampler.
        self.has_vulkan()
            && self.linear_decode_enabled
            && top_k > 0
            && top_k <= kiln_vulkan_kernel::kernels::TOPK_SAMPLE_KERNEL_K_MAX
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
        // kt guards read directly off the kt args before the bridge.
        if !self.supports_linear_decode_sample(top_k) || x.dtype() != kiln_tensor::DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu) || !matches!(weight_t.device(), kiln_tensor::Device::Cpu) {
            return Ok(None);
        }
        // (#1082) Fully kt-native: lm_head weight keyed on the stable kt id.
        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        if batch != 1 || seq_len != 1 {
            return Ok(None);
        }
        let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
            return Ok(None);
        };
        if weight_hidden != hidden {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let packed_bf16 = self.use_bf16_packed_linear_weight_kt(weight_t);
        let weight_buf = if packed_bf16 {
            self.cached_bf16_packed_weight_buffer_kt(weight_t)?
        } else {
            self.cached_f32_weight_buffer_kt(weight_t)?
        };
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let token = kiln_vulkan_kernel::kernels::dispatch_linear_decode_sample_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            packed_bf16,
            hidden,
            out_dim,
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
        .context("fused linear_decode_sample dispatch failed")?;
        Ok(Some(token))
    }

    fn supports_linear_decode_sample_batch(&self, top_k: &[u32], temperatures: &[f32]) -> bool {
        self.has_vulkan()
            && self.linear_decode_enabled
            && top_k.len() == temperatures.len()
            && !top_k.is_empty()
            && top_k.iter().zip(temperatures.iter()).all(|(&k, &temp)| {
                let greedy = temp == 0.0 || (k == 1 && temp.is_finite() && temp > 0.0);
                greedy
                    || (temp.is_finite()
                        && temp > 0.0
                        && k > 0
                        && k <= kiln_vulkan_kernel::kernels::TOPK_SAMPLE_KERNEL_K_MAX)
            })
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
        if !self.supports_linear_decode_sample_batch(top_k, temperatures)
            || x.dtype() != kiln_tensor::DType::F32
        {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu)
            || !matches!(weight_t.device(), kiln_tensor::Device::Cpu)
        {
            return Ok(None);
        }
        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        if batch == 0 || seq_len != 1 {
            return Ok(None);
        }
        let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
            return Ok(None);
        };
        if weight_hidden != hidden {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let packed_bf16 = self.use_bf16_packed_linear_weight_kt(weight_t);
        let weight_buf = if packed_bf16 {
            self.cached_bf16_packed_weight_buffer_kt(weight_t)?
        } else {
            self.cached_f32_weight_buffer_kt(weight_t)?
        };
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let tokens = kiln_vulkan_kernel::kernels::dispatch_linear_decode_sample_batch_bytes(
            vk_device,
            &x_data,
            &weight_buf,
            packed_bf16,
            batch,
            hidden,
            out_dim,
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
        .context("fused linear_decode_sample_batch dispatch failed")?;
        Ok(Some(tokens))
    }

    fn linear_decode_argmax_batch(
        &self,
        x: &kiln_tensor::Tensor,
        weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<Vec<u32>>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan()
            || !self.linear_decode_enabled
            || !self.linear_argmax_batch_enabled
            || x.dtype() != kiln_tensor::DType::F32
        {
            return Ok(None);
        }
        if !matches!(x.device(), kiln_tensor::Device::Cpu) || !matches!(weight_t.device(), kiln_tensor::Device::Cpu) {
            return Ok(None);
        }
        // (#1082) Fully kt-native: lm_head weight keyed on the stable kt id.
        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        if batch == 0 || seq_len != 1 {
            return Ok(None);
        }
        let Ok((weight_hidden, out_dim)) = weight_t.dims2() else {
            return Ok(None);
        };
        if weight_hidden != hidden {
            return Ok(None);
        }

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let x_data = kt_tensor_to_f32_bytes_with_shape(x)?.0;
        let tokens = if self.use_bf16_packed_linear_weight_kt(weight_t) {
            let weight_buf = self.cached_bf16_packed_weight_buffer_kt(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached_bf16_weights_bytes(
                vk_device,
                &x_data,
                &weight_buf,
                batch,
                hidden,
                out_dim,
            )
        } else {
            let weight_buf = self.cached_f32_weight_buffer_kt(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached_bytes(
                vk_device,
                &x_data,
                &weight_buf,
                batch,
                hidden,
                out_dim,
            )
        }
        .context("linear_decode_argmax_batch kernel failed")?;
        Ok(Some(tokens))
    }

    fn prewarm_decode_weights(&self, weights: &GpuWeights) -> Result<()> {
        if !self.has_vulkan() || !self.weight_prewarm_enabled {
            return Ok(());
        }

        let start = std::time::Instant::now();
        let mut count = 0usize;
        let mut bytes = 0usize;
        let mut bf16_packed_count = 0usize;
        let mut bf16_packed_bytes = 0usize;

        self.prewarm_linear_weight_kt(
            "embed_tokens_t",
            &weights.embed_tokens_t,
            &mut count,
            &mut bytes,
            &mut bf16_packed_count,
            &mut bf16_packed_bytes,
        )?;

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            match &layer.attention {
                GpuAttentionWeights::Full(attn) => {
                    self.prewarm_full_attn_qkv_weights_kt(
                        layer_idx,
                        &attn.q_proj_t,
                        &attn.k_proj_t,
                        &attn.v_proj_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_linear_weight_kt(
                        &format!("layers.{layer_idx}.attention.o_proj_t"),
                        &attn.o_proj_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                }
                GpuAttentionWeights::Linear(attn) => {
                    self.prewarm_gdn_in_proj_weight_kt(
                        &format!("layers.{layer_idx}.attention.in_proj_qkv_t"),
                        &attn.in_proj_qkv_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_gdn_in_proj_weight_kt(
                        &format!("layers.{layer_idx}.attention.in_proj_z_t"),
                        &attn.in_proj_z_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_gdn_in_proj_weight_kt(
                        &format!("layers.{layer_idx}.attention.in_proj_a_t"),
                        &attn.in_proj_a_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_gdn_in_proj_weight_kt(
                        &format!("layers.{layer_idx}.attention.in_proj_b_t"),
                        &attn.in_proj_b_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_linear_weight_kt(
                        &format!("layers.{layer_idx}.attention.out_proj_t"),
                        &attn.out_proj_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                }
            }

            self.prewarm_mlp_decode_weights_kt(
                layer_idx,
                &layer.mlp.gate_proj_t,
                &layer.mlp.up_proj_t,
                &layer.mlp.down_proj_t,
                &mut count,
                &mut bytes,
                &mut bf16_packed_count,
                &mut bf16_packed_bytes,
            )?;
        }

        tracing::info!(
            weights = count,
            f32_cache_mb = bytes / (1024 * 1024),
            bf16_packed_weights = bf16_packed_count,
            bf16_packed_cache_mb = bf16_packed_bytes / (1024 * 1024),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Vulkan decode weight cache prewarmed"
        );
        Ok(())
    }

    /// Phase 4.x residency: drop the CPU storage of every
    /// pre-transposed weight cache (`*_proj_t`, `embed_tokens_t`)
    /// whose BF16-packed bytes are already resident in
    /// `bf16_packed_weight_cache_kt`. Replace each with a
    /// 1-element BF16 stub and re-key the cache so subsequent
    /// lookups against the new kt `TensorId` still find the same
    /// `Arc<VulkanBuffer>`.
    ///
    /// Saves ~6-7 GB peak RSS on Qwen3.5-4B training at T=918 — the
    /// transposed-cache copies are the dominant remaining
    /// CPU-side residency item documented in
    /// `docs/audits/candle_cpu_residency_2026-05-11.md`.
    ///
    /// Safe because:
    /// - The bf16-packed Vulkan code paths read the weight via the
    ///   `Arc<VulkanBuffer>` looked up in `bf16_packed_weight_cache_kt`.
    ///   They never re-read the CPU storage of the source tensor
    ///   after the buffer is cached.
    /// - `VulkanLinearOp::bwd` for BF16 weights routes through the
    ///   transposed Vulkan kernel (also buffer-backed). The F32
    ///   fallback bwd path that *does* read `self.weight_t` cannot
    ///   fire for BF16 weights.
    /// - Non-BF16 tensors and tensors not in the cache are skipped.
    fn drop_uploaded_bf16_weights(
        &self,
        weights: &mut crate::forward::GpuWeights,
        device: &kiln_tensor::Device,
    ) -> Result<usize> {
        if !self.has_vulkan() {
            return Ok(0);
        }
        // Broadcast-base for cheap shape-preserving stubs. Source has
        // 2 bytes of storage; broadcast_as(target_shape) creates views
        // with stride [0, 0] sharing the same backing storage. Each per-
        // weight stub costs ~24 bytes of metadata (Layout + Tensor
        // struct), not `hidden * out_dim * 2` bytes. The weights are
        // kt-typed (#1082 forward-flip), and the Vulkan buffer cache is
        // re-keyed directly from the old kt TensorId to the stub's kt
        // TensorId.
        let broadcast_base = kiln_tensor::Tensor::zeros(
            (1usize, 1usize),
            kiln_tensor::DType::BF16,
            kiln_tensor::Device::Cpu,
        )
        .context("drop_uploaded_bf16_weights: create broadcast base")?;
        let _ = device;
        let mut bf16_cache = self
            .bf16_packed_weight_cache_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("bf16 weight cache mutex poisoned"))?;
        let mut f32_cache = self
            .weight_cache_kt
            .lock()
            .map_err(|_| anyhow::anyhow!("f32 weight cache mutex poisoned"))?;

        // Per-tensor replacement closure. Returns true if the tensor
        // was stubbed (was BF16, rank-2, and in the cache).
        //
        // - Reads the original `[hidden, out_dim]` shape from `t.dims()`
        //   *before* replacement.
        // - Creates a shape-preserving stub by broadcasting the
        //   2-byte base to that shape (so downstream `weight_t.dims2()`
        //   reads continue to return the right shape, but the storage
        //   bytes drop to ~zero).
        // - Re-keys the packed cache and any F32 shadow cache entry so
        //   subsequent kt-native lookups by the stub's new TensorId still find
        //   the original `Arc<VulkanBuffer>`s.
        fn replace(
            t: &mut kiln_tensor::Tensor,
            bf16_cache: &mut std::collections::HashMap<
                kiln_tensor::TensorId,
                Arc<kiln_vulkan_kernel::VulkanBuffer>,
            >,
            f32_cache: &mut std::collections::HashMap<
                kiln_tensor::TensorId,
                Arc<kiln_vulkan_kernel::VulkanBuffer>,
            >,
            broadcast_base: &kiln_tensor::Tensor,
        ) -> bool {
            if t.dtype() != kiln_tensor::DType::BF16 {
                return false;
            }
            let dims = t.dims();
            if dims.len() != 2 {
                return false; // Only rank-2 transposed-cache tensors are stubbable.
            }
            let (d0, d1) = (dims[0], dims[1]);
            let old_id = t.id();
            let Some(bf16_buf) = bf16_cache.remove(&old_id) else {
                return false;
            };
            let f32_buf = f32_cache.remove(&old_id);
            let Ok(new_stub) = broadcast_base.broadcast_as((d0, d1)) else {
                bf16_cache.insert(old_id, bf16_buf); // restore on failure
                if let Some(buf) = f32_buf {
                    f32_cache.insert(old_id, buf);
                }
                return false;
            };
            let new_id = new_stub.id();
            *t = new_stub;
            bf16_cache.insert(new_id, bf16_buf);
            if let Some(buf) = f32_buf {
                f32_cache.insert(new_id, buf);
            }
            true
        }

        let mut stubbed = 0usize;

        // Intentionally NOT stubbing `weights.embed_tokens_t`:
        // `embedding_lookup_from_transposed_index` calls
        // `embed_tokens_t.index_select(idx, 1)` which reads the
        // tensor's data (not just shape), so a 1-element stub would
        // make the embedding lookup return garbage. The other `*_proj_t`
        // caches go through the kt TensorId → Arc<VulkanBuffer> packed cache,
        // so they only need shape/dtype metadata locally. Embedding savings
        // (~750 MB) are small
        // next to the per-layer transposes (~5-6 GB across 32 layers).

        // Per-layer attention + MLP transposes.
        for layer in weights.layers.iter_mut() {
            match &mut layer.attention {
                crate::forward::GpuAttentionWeights::Full(attn) => {
                    for t in [
                        &mut attn.q_proj_t,
                        &mut attn.k_proj_t,
                        &mut attn.v_proj_t,
                        &mut attn.o_proj_t,
                    ] {
                        if replace(t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                            stubbed += 1;
                        }
                    }
                    if let Some(qkv_t) = attn.qkv_proj_t.as_mut() {
                        if replace(qkv_t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                            stubbed += 1;
                        }
                    }
                }
                crate::forward::GpuAttentionWeights::Linear(attn) => {
                    for t in [
                        &mut attn.in_proj_qkv_t,
                        &mut attn.in_proj_z_t,
                        &mut attn.in_proj_a_t,
                        &mut attn.in_proj_b_t,
                        &mut attn.out_proj_t,
                    ] {
                        if replace(t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                            stubbed += 1;
                        }
                    }
                    if let Some(ab_t) = attn.in_proj_ab_t.as_mut() {
                        if replace(ab_t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                            stubbed += 1;
                        }
                    }
                }
            }
            for t in [
                &mut layer.mlp.gate_proj_t,
                &mut layer.mlp.up_proj_t,
                &mut layer.mlp.down_proj_t,
            ] {
                if replace(t, &mut bf16_cache, &mut f32_cache, &broadcast_base) {
                    stubbed += 1;
                }
            }
        }

        tracing::info!(
            stubbed,
            "dropped CPU storage of pre-transposed bf16 weight caches"
        );
        Ok(stubbed)
    }

    fn full_attn_qkv_decode(
        &self,
        x: &kiln_tensor::Tensor,
        q_weight_t: &kiln_tensor::Tensor,
        k_weight_t: &kiln_tensor::Tensor,
        v_weight_t: &kiln_tensor::Tensor,
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
        // kt guards read directly off the kt args before the bridge.
        if !self.has_vulkan() || !self.full_attn_qkv_enabled || x.dtype() != kiln_tensor::DType::F32 {
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
                    vk_device, &x_data, row_count, &gate_buf, &up_buf, &down_buf, hidden,
                    intermediate, out_dim,
                )
                .context("mlp_decode kernel failed")?
            } else if use_bf16_mlp_weights {
                let gate_buf = self.cached_bf16_packed_weight_buffer_kt(gate_weight_t)?;
                let up_buf = self.cached_bf16_packed_weight_buffer_kt(up_weight_t)?;
                let down_buf = self.cached_bf16_packed_weight_buffer_kt(down_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_weights_bytes(
                    vk_device, &x_data, row_count, &gate_buf, &up_buf, &down_buf, hidden,
                    intermediate, out_dim,
                )
                .context("mlp_decode kernel failed")?
            } else {
                let gate_buf = self.cached_f32_weight_buffer_kt(gate_weight_t)?;
                let up_buf = self.cached_f32_weight_buffer_kt(up_weight_t)?;
                let down_buf = self.cached_f32_weight_buffer_kt(down_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bytes(
                    vk_device, &x_data, row_count, &gate_buf, &up_buf, &down_buf, hidden,
                    intermediate, out_dim,
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
            || !matches!(q.dtype(), kiln_tensor::DType::BF16 | kiln_tensor::DType::F32)
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
            let out_no_seq = kt_tensor_from_f32_bytes(
                &out_data,
                &[batch, heads, dv],
                q_dtype,
            )?;
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
        let out = kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, heads, dv],
            q_dtype,
        )?
        .unsqueeze(1)?;
        if let Some(sd) = new_state_data {
            *state_kt = kt_tensor_from_f32_bytes(
                &sd,
                &state_dims,
                state_dtype,
            )?;
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
            || !matches!(q.dtype(), kiln_tensor::DType::F32 | kiln_tensor::DType::BF16)
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
        let out = kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, heads, dv],
            state_dtype,
        )?
        .unsqueeze(1)?;
        if let Some(sd) = new_state_data {
            *state_kt = kt_tensor_from_f32_bytes(
                &sd,
                &state_dims,
                state_dtype,
            )?;
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
        if !matches!(q.dtype(), kiln_tensor::DType::BF16 | kiln_tensor::DType::F32) {
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
                    &q_data, &k_data, &v_data, &beta_data, &g_data,
                    state_data_owned.as_deref(),
                    batch, heads, dk, dv,
                    resident_state,
                )
                .context("gdn_recurrent_step resident-state kernel failed")?;
            let out = kt_tensor_from_f32_bytes(
                &out_data,
                &[batch, heads, dv],
                q_dtype,
            )?;

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
        let out = kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, heads, dv],
            q_dtype,
        )?;
        if let Some(sd) = new_state_data {
            *state_kt = kt_tensor_from_f32_bytes(
                &sd,
                &state_dims,
                state_dtype,
            )?;
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
                            .map_err(|e| anyhow::anyhow!("gdn_chunkwise_forward: to_vec1 f32: {e}"))?;
                        kiln_vulkan_kernel::vk_tensor::VkTensor::from_f32_slice(
                            &data,
                            shape,
                            vk_device.clone(),
                        )
                    };
                (load(q)?, load(k)?, load(v)?, load(beta)?, load(g)?, load(state_kt)?)
            };

        let out_vk =
            if std::env::var("KILN_DISABLE_VULKAN_GDN_CHUNKWISE_SINGLE_SUBMIT").is_ok() {
                if kiln_core::env_flag::env_flag("KILN_VULKAN_GDN_CHUNKWISE_FALLBACK", false) {
                    tracing::warn!(
                        "single-submit Vulkan GDN chunkwise prefill disabled; falling back"
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
    ) -> Result<Option<(kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor, kiln_tensor::Tensor)>> {
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
                &g_data, &v_data, &kkt_data, &qkt_data, &ks_entry_data, &q_s_data,
                batch, heads, chunk, dv,
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
        let (batch, heads, chunk, dv) =
            (v_prime_dims[0], v_prime_dims[1], v_prime_dims[2], v_prime_dims[3]);
        let (out_data, p_out_data) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_scan_bytes(
                vk_device,
                &a_strict_data,
                &b_mask_data,
                &v_prime_data,
                &q_s_scaled_data,
                &beta_data,
                &decay_last_col_data,
                batch, heads, chunk, dv,
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
                vk_device, &g_data, &v_data, &kkt_data, &qkt_data, &ks_entry_data,
                &q_s_data, &beta_data, &k_t_data, &state_data,
                batch, heads, chunk, dk, dv,
            )
            .context("gdn_full_chunk_forward kernel failed")?;
        let out = kt_tensor_from_f32_bytes(
            &out_data,
            &[batch, heads, chunk, dv],
            kiln_tensor::DType::BF16,
        )?;
        *state_kt = kt_tensor_from_f32_bytes(
            &new_state_data,
            &state_dims,
            kiln_tensor::DType::BF16,
        )?;
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
        if !matches!(a.dtype(), kiln_tensor::DType::BF16 | kiln_tensor::DType::F32) {
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
        let beta =
            kt_tensor_from_f32_bytes(&beta_b, &out_shape, output_dtype)?;
        let g =
            kt_tensor_from_f32_bytes(&g_b, &out_shape, output_dtype)?;
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
        if !matches!(x.dtype(), kiln_tensor::DType::BF16 | kiln_tensor::DType::F32) {
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
        let out = kt_tensor_from_f32_bytes(
            &out_data,
            &out_shape,
            output_dtype,
        )?;
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
        if !matches!(x.dtype(), kiln_tensor::DType::BF16 | kiln_tensor::DType::F32) {
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
        let out =
            kt_tensor_from_f32_bytes(&out_data, &out_shape, kiln_tensor::DType::F32)?;
        *conv_state_kt = kt_tensor_from_f32_bytes(
            &state_data_out,
            &conv_state_shape,
            kiln_tensor::DType::F32,
        )?;
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
        if !matches!(x.dtype(), kiln_tensor::DType::BF16 | kiln_tensor::DType::F32) {
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
            let out = kt_tensor_from_f32_bytes(
                &out_data,
                x_dims,
                kiln_tensor::DType::F32,
            )?;
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
                let out = kt_tensor_from_f32_bytes(
                    &out_data,
                    x_dims,
                    kiln_tensor::DType::F32,
                )?;
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

/// Check if Vulkan is available on this system.
/// Uses a cheap probe (instance + physical-device enumeration only) cached
/// with OnceLock to avoid repeated checks.
pub fn vulkan_is_available() -> bool {
    static VULKAN_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VULKAN_AVAILABLE.get_or_init(kiln_vulkan_kernel::VulkanDevice::probe)
}

/// Return the selected Vulkan device name for diagnostics and benchmark output.
pub fn vulkan_device_name() -> Option<String> {
    static VULKAN_DEVICE_NAME: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    VULKAN_DEVICE_NAME
        .get_or_init(|| {
            kiln_vulkan_kernel::VulkanDevice::new()
                .ok()
                .map(|dev| dev.device_name().to_string())
        })
        .clone()
}

/// Precompile Vulkan custom kernels.
///
/// This verifies that the validated built-in SPIR-V modules load correctly and
/// that compute pipelines can be created. `VulkanBackend::new` warms the real
/// backend device; this standalone helper is only for background verification.
pub fn precompile_custom_kernels() -> Result<()> {
    let vk_device = match kiln_vulkan_kernel::VulkanDevice::new() {
        Ok(dev) => dev,
        Err(_) => return Ok(()),
    };
    kiln_vulkan_kernel::kernels::prewarm_builtin_pipelines(&vk_device)?;
    tracing::info!("Vulkan shader and pipeline verification complete");
    Ok(())
}

// (#1082) Vulkan residency / optimizer / `lora_delta_resident` tests.
//
// These exercise the resident-activation registry internals (register /
// update / has / evict / resolve), the on-device SGD + AdamW kernels, and the
// `lora_delta_resident` decline contract. The registry is now **kt-native** —
// it is keyed directly on the kt `TensorId` (`tensor.id()`), with byte
// extraction reading straight from kt storage (see
// `register_resident_activation` and friends above). There is no candle
// bridge anymore, so a kt tensor handed to `register_*` and then to `has_*` /
// `resolve_*` round-trips to the same registry key — which is what re-enables
// these tests under the kt-typed `BackendRuntime` trait.
//
// id-stability across an in-place content change (formerly provided by candle
// `Var::set`) is reproduced with kt `Tensor::slice_set` (dim-0 in-place
// overwrite that preserves the tensor's `TensorId` and bumps its version
// counter) — the kt analog of `Var::set`.
//
// `lora_delta_resident` was rewritten from on-device dispatch (a
// `candle_core::CustomOp3` autograd island) to an unconditional decline: the
// kt autograd tape (`kiln_autograd`) is now the sole grad producer, and the
// forward LoRA delta is recorded by the portable kt `compute_lora_delta` path
// in forward.rs. The former "dispatches on-device + reflects post-update
// weights" success test had no kt analog (its whole point was the removed
// dispatch path), so it was dropped; the surviving lora tests assert the new
// decline contract instead.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendRuntime;

    /// dispatch_sgd_step against two registry-resident F32 tensors —
    /// param := param - lr * grad, computed on-device, must match the
    /// CPU reference to f32 precision.
    #[test]
    fn dispatch_sgd_step_resident_round_trip() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 16usize;
        let lr = 0.01f32;
        let param_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let grad_data: Vec<f32> = (0..n).map(|i| ((i as i32 - 8) as f32) * 0.05).collect();
        let expected: Vec<f32> = param_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&p, &g)| p - lr * g)
            .collect();

        let param = kiln_tensor::Tensor::from_vec(param_data, (n,))?;
        let grad = kiln_tensor::Tensor::from_vec(grad_data, (n,))?;

        // Both must be resident before dispatch_sgd_step succeeds.
        backend.register_resident_activation(&param)?;
        backend.register_resident_activation(&grad)?;

        let dispatched = backend.dispatch_sgd_step(&param, &grad, lr)?;
        assert!(
            dispatched,
            "dispatch_sgd_step should succeed when both buffers are resident"
        );

        // Read back the updated param buffer from the registry.
        let param_buf = with_resident_registry(|cache| cache.get(&param.id()).cloned())
            .expect("param must still be in registry");
        let device = backend.vulkan_device.as_ref().unwrap();
        let updated_bytes = kiln_vulkan_kernel::VulkanBuffer::read_back(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &param_buf,
        )?;
        let updated: Vec<f32> = updated_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(updated.len(), n);
        for (i, (got, want)) in updated.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-7,
                "idx {i}: got {got:.9} want {want:.9}"
            );
        }

        backend.evict_resident_activation(&param);
        backend.evict_resident_activation(&grad);
        Ok(())
    }

    /// dispatch_sgd_step must return false (caller falls back to CPU)
    /// when the operands aren't both resident — exercises all four
    /// (resident? × resident?) combinations.
    #[test]
    fn dispatch_sgd_step_falls_back_when_not_resident() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = kiln_tensor::Tensor::from_vec(vec![1.0f32; 4], (4,))?;
        let g = kiln_tensor::Tensor::from_vec(vec![0.5f32; 4], (4,))?;
        // Neither registered — fall back.
        assert!(!backend.dispatch_sgd_step(&p, &g, 0.01)?);
        // Only param registered — fall back (grad missing).
        backend.register_resident_activation(&p)?;
        assert!(!backend.dispatch_sgd_step(&p, &g, 0.01)?);
        // Only grad registered — fall back (param missing).
        backend.evict_resident_activation(&p);
        backend.register_resident_activation(&g)?;
        assert!(!backend.dispatch_sgd_step(&p, &g, 0.01)?);
        backend.evict_resident_activation(&g);
        Ok(())
    }

    /// dispatch_sgd_step must error (not silently succeed or fall
    /// back) when shapes mismatch — that's a programmer bug worth
    /// surfacing immediately.
    #[test]
    fn dispatch_sgd_step_errors_on_shape_mismatch() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = kiln_tensor::Tensor::from_vec(vec![1.0f32; 4], (4,))?;
        let g = kiln_tensor::Tensor::from_vec(vec![0.5f32; 8], (8,))?;
        backend.register_resident_activation(&p)?;
        backend.register_resident_activation(&g)?;
        let err = backend.dispatch_sgd_step(&p, &g, 0.01).unwrap_err();
        assert!(
            err.to_string().contains("different element counts"),
            "unexpected error: {err}"
        );
        backend.evict_resident_activation(&p);
        backend.evict_resident_activation(&g);
        Ok(())
    }

    /// (#1082) `lora_delta_resident` declines (`Ok(None)`) even when A and B
    /// are resident in the registry.
    ///
    /// Formerly (`lora_delta_resident_matches_cpu_reference`) this asserted the
    /// hook dispatched the LoRA delta on-device (a `candle_core::CustomOp3`
    /// autograd island) and matched a CPU `(x @ A.T @ B.T) * scale` reference.
    /// That dispatch path was removed: the kt autograd tape (`kiln_autograd`)
    /// is the sole grad producer and the forward LoRA delta is recorded by the
    /// portable kt `compute_lora_delta` path in forward.rs. The hook now
    /// unconditionally declines, routing the caller to that kt-recorded path —
    /// and it must do so *even* when A and B are resident (residency is no
    /// longer a dispatch trigger). This is the inverse-condition partner of
    /// `lora_delta_resident_falls_back_when_not_resident`.
    #[test]
    fn lora_delta_resident_declines_even_when_resident() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        // Same LoRA-shape setup the old success test used: rank=4, in=8, out=6.
        let t = 5usize;
        let in_features = 8usize;
        let rank = 4usize;
        let out_features = 6usize;
        let scale = 0.5f32;

        let x_data: Vec<f32> = (0..t * in_features).map(|i| (i as f32) * 0.01).collect();
        let a_data: Vec<f32> = (0..rank * in_features).map(|i| (i as f32) * 0.02).collect();
        let b_data: Vec<f32> = (0..out_features * rank)
            .map(|i| (i as f32) * 0.03)
            .collect();

        let x_bf16 = kiln_tensor::Tensor::from_vec(x_data, (1, t, in_features))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        let a_bf16 = kiln_tensor::Tensor::from_vec(a_data, (rank, in_features))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        let b_bf16 = kiln_tensor::Tensor::from_vec(b_data, (out_features, rank))?
            .to_dtype(kiln_tensor::DType::BF16)?;

        // Register A and B in the registry — residency must NOT trigger a
        // dispatch under the kt decline contract.
        backend.register_resident_activation(&a_bf16)?;
        backend.register_resident_activation(&b_bf16)?;

        assert!(
            backend
                .lora_delta_resident(&x_bf16, &a_bf16, &b_bf16, scale)?
                .is_none(),
            "lora_delta_resident must decline even when A and B are resident \
             (kt tape is the sole grad producer; forward delta is recorded by \
             the portable compute_lora_delta path)"
        );

        backend.evict_resident_activation(&a_bf16);
        backend.evict_resident_activation(&b_bf16);
        Ok(())
    }

    /// lora_delta_resident must return Ok(None) when A or B is not
    /// registered — caller falls back to the portable kt path.
    #[test]
    fn lora_delta_resident_falls_back_when_not_resident() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let x =
            kiln_tensor::Tensor::from_vec(vec![0.0f32; 16], (1, 2, 8))?.to_dtype(kiln_tensor::DType::BF16)?;
        let a = kiln_tensor::Tensor::from_vec(vec![0.0f32; 32], (4, 8))?.to_dtype(kiln_tensor::DType::BF16)?;
        let b = kiln_tensor::Tensor::from_vec(vec![0.0f32; 24], (6, 4))?.to_dtype(kiln_tensor::DType::BF16)?;
        // Neither registered — fall back.
        assert!(backend.lora_delta_resident(&x, &a, &b, 0.5)?.is_none());
        // Only A registered — fall back.
        backend.register_resident_activation(&a)?;
        assert!(backend.lora_delta_resident(&x, &a, &b, 0.5)?.is_none());
        // Only B registered — fall back.
        backend.evict_resident_activation(&a);
        backend.register_resident_activation(&b)?;
        assert!(backend.lora_delta_resident(&x, &a, &b, 0.5)?.is_none());
        backend.evict_resident_activation(&b);
        Ok(())
    }

    /// dispatch_sgd_step on BF16 operands must NOW succeed (post-Phase
    /// 4.x bf16 SGD kernel) and produce results that match the F32
    /// reference computation to bf16 precision. This is the path
    /// that lets LoRA params (BF16 by convention) update on-device
    /// without the host re-upload round-trip.
    #[test]
    fn dispatch_sgd_step_bf16_resident_round_trip() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 32usize;
        let lr = 0.01f32;
        let p_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let g_data: Vec<f32> = (0..n).map(|i| ((i as i32 - 16) as f32) * 0.05).collect();
        // F32 reference for what BF16 SGD should produce.
        let expected_f32: Vec<f32> = p_data
            .iter()
            .zip(g_data.iter())
            .map(|(&p, &g)| p - lr * g)
            .collect();

        let p_f32 = kiln_tensor::Tensor::from_vec(p_data, (n,))?;
        let g_f32 = kiln_tensor::Tensor::from_vec(g_data, (n,))?;
        let p_bf16 = p_f32.to_dtype(kiln_tensor::DType::BF16)?;
        let g_bf16 = g_f32.to_dtype(kiln_tensor::DType::BF16)?;

        backend.register_resident_activation(&p_bf16)?;
        backend.register_resident_activation(&g_bf16)?;

        let dispatched = backend.dispatch_sgd_step(&p_bf16, &g_bf16, lr)?;
        assert!(
            dispatched,
            "BF16 dispatch_sgd_step must succeed when both operands are resident"
        );

        // Read the updated param buffer back via resolve.
        let resolved = backend
            .resolve_resident_activation(&p_bf16, &[n], kiln_tensor::DType::BF16)?
            .expect("must resolve");
        let updated_v: Vec<f32> = resolved
            .to_dtype(kiln_tensor::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, (got, want)) in updated_v.iter().zip(expected_f32.iter()).enumerate() {
            // BF16 has ~3 decimal digits of precision; tolerance reflects that.
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1e-3);
            assert!(
                abs < 5e-2 || rel < 5e-2,
                "idx {i}: got={got:.6} want={want:.6} abs={abs:e} rel={rel:e}"
            );
        }

        backend.evict_resident_activation(&p_bf16);
        backend.evict_resident_activation(&g_bf16);
        Ok(())
    }

    /// dispatch_adamw_step on registry-resident F32 operands must
    /// match a scalar reference of the decoupled-weight-decay AdamW
    /// math to f32 precision, after one optimizer step from
    /// `m=v=0`. Exercises the full param/grad/m/v round-trip plus
    /// the bias-correction precompute path.
    #[test]
    fn dispatch_adamw_step_resident_round_trip_f32() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 16usize;
        let lr = 0.01f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.01f32;
        let step: u32 = 1;

        let p_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.2).collect();
        let g_data: Vec<f32> = (0..n).map(|i| ((i as i32 - 8) as f32) * 0.03).collect();
        let m_data: Vec<f32> = vec![0.0; n];
        let v_data: Vec<f32> = vec![0.0; n];

        // Scalar reference (matches the shader math exactly).
        let bc1 = 1.0_f32 - beta1.powi(step as i32);
        let bc2 = 1.0_f32 - beta2.powi(step as i32);
        let expected: Vec<f32> = p_data
            .iter()
            .zip(g_data.iter())
            .map(|(&p, &g)| {
                let p_wd = p - lr * weight_decay * p;
                let m = beta1 * 0.0 + (1.0 - beta1) * g;
                let v = beta2 * 0.0 + (1.0 - beta2) * g * g;
                let m_hat = m / bc1.max(1e-20);
                let v_hat = v / bc2.max(1e-20);
                p_wd - lr * m_hat / (v_hat.sqrt() + eps)
            })
            .collect();

        let param = kiln_tensor::Tensor::from_vec(p_data, (n,))?;
        let grad = kiln_tensor::Tensor::from_vec(g_data, (n,))?;
        let m = kiln_tensor::Tensor::from_vec(m_data, (n,))?;
        let v = kiln_tensor::Tensor::from_vec(v_data, (n,))?;

        backend.register_resident_activation(&param)?;
        backend.register_resident_activation(&grad)?;
        backend.register_resident_activation(&m)?;
        backend.register_resident_activation(&v)?;

        let dispatched = backend.dispatch_adamw_step(
            &param,
            &grad,
            &m,
            &v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        )?;
        assert!(
            dispatched,
            "adamw_step must succeed when all four buffers are resident"
        );

        let resolved = backend
            .resolve_resident_activation(&param, &[n], kiln_tensor::DType::F32)?
            .expect("param must resolve after dispatch");
        let got: Vec<f32> = resolved.flatten_all()?.to_vec1::<f32>()?;
        for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "idx {i}: got={g:.9} want={w:.9}");
        }

        backend.evict_resident_activation(&param);
        backend.evict_resident_activation(&grad);
        backend.evict_resident_activation(&m);
        backend.evict_resident_activation(&v);
        Ok(())
    }

    /// Two-step BF16 AdamW round-trip: starts at m=v=0, runs
    /// `dispatch_adamw_step` twice with step=1 then step=2, and
    /// verifies the param ends up close to the bf16-precision
    /// reference. Catches bugs where bias-correction precompute or
    /// in-place buffer updates don't carry across steps.
    #[test]
    fn dispatch_adamw_step_resident_round_trip_bf16_two_step() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 32usize;
        let lr = 0.05f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.01f32;

        let p_data: Vec<f32> = (0..n).map(|i| ((i as i32 - 16) as f32) * 0.05).collect();
        let g_data: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.02).collect();

        // Reference: two AdamW steps on f32 (no bf16 quantization).
        let mut ref_p = p_data.clone();
        let mut ref_m = vec![0.0f32; n];
        let mut ref_v = vec![0.0f32; n];
        for step in 1u32..=2 {
            let bc1 = (1.0_f32 - beta1.powi(step as i32)).max(1e-20);
            let bc2 = (1.0_f32 - beta2.powi(step as i32)).max(1e-20);
            for i in 0..n {
                let g = g_data[i];
                let p_wd = ref_p[i] - lr * weight_decay * ref_p[i];
                let m_new = beta1 * ref_m[i] + (1.0 - beta1) * g;
                let v_new = beta2 * ref_v[i] + (1.0 - beta2) * g * g;
                let m_hat = m_new / bc1;
                let v_hat = v_new / bc2;
                ref_p[i] = p_wd - lr * m_hat / (v_hat.sqrt() + eps);
                ref_m[i] = m_new;
                ref_v[i] = v_new;
            }
        }

        let p_f32 = kiln_tensor::Tensor::from_vec(p_data, (n,))?;
        let g_f32 = kiln_tensor::Tensor::from_vec(g_data, (n,))?;
        let m_f32 = kiln_tensor::Tensor::from_vec(vec![0.0f32; n], (n,))?;
        let v_f32 = kiln_tensor::Tensor::from_vec(vec![0.0f32; n], (n,))?;
        let p_bf16 = p_f32.to_dtype(kiln_tensor::DType::BF16)?;
        let g_bf16 = g_f32.to_dtype(kiln_tensor::DType::BF16)?;
        let m_bf16 = m_f32.to_dtype(kiln_tensor::DType::BF16)?;
        let v_bf16 = v_f32.to_dtype(kiln_tensor::DType::BF16)?;

        backend.register_resident_activation(&p_bf16)?;
        backend.register_resident_activation(&g_bf16)?;
        backend.register_resident_activation(&m_bf16)?;
        backend.register_resident_activation(&v_bf16)?;

        for step in 1u32..=2 {
            let dispatched = backend.dispatch_adamw_step(
                &p_bf16,
                &g_bf16,
                &m_bf16,
                &v_bf16,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            )?;
            assert!(dispatched, "step {step}: adamw bf16 dispatch must succeed");
        }

        let resolved = backend
            .resolve_resident_activation(&p_bf16, &[n], kiln_tensor::DType::BF16)?
            .expect("param must resolve");
        let got: Vec<f32> = resolved
            .to_dtype(kiln_tensor::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, (g, w)) in got.iter().zip(ref_p.iter()).enumerate() {
            // bf16 mantissa ≈ 7 bits; loose tolerance per lane.
            let abs = (g - w).abs();
            let rel = abs / w.abs().max(1e-3);
            assert!(
                abs < 5e-2 || rel < 5e-2,
                "idx {i}: got={g:.6} want={w:.6} abs={abs:e} rel={rel:e}"
            );
        }

        backend.evict_resident_activation(&p_bf16);
        backend.evict_resident_activation(&g_bf16);
        backend.evict_resident_activation(&m_bf16);
        backend.evict_resident_activation(&v_bf16);
        Ok(())
    }

    /// dispatch_adamw_step falls back (returns false) when any of the
    /// four operand buffers isn't resident.
    #[test]
    fn dispatch_adamw_step_falls_back_when_not_resident() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = kiln_tensor::Tensor::from_vec(vec![1.0f32; 4], (4,))?;
        let g = kiln_tensor::Tensor::from_vec(vec![0.5f32; 4], (4,))?;
        let m = kiln_tensor::Tensor::from_vec(vec![0.0f32; 4], (4,))?;
        let v = kiln_tensor::Tensor::from_vec(vec![0.0f32; 4], (4,))?;
        // Nothing registered.
        let dispatched =
            backend.dispatch_adamw_step(&p, &g, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1)?;
        assert!(!dispatched);
        // Only param + m registered — v missing → fall back.
        backend.register_resident_activation(&p)?;
        backend.register_resident_activation(&m)?;
        let dispatched =
            backend.dispatch_adamw_step(&p, &g, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1)?;
        assert!(!dispatched);
        backend.evict_resident_activation(&p);
        backend.evict_resident_activation(&m);
        Ok(())
    }

    /// Lazy host-storage sync end-to-end. Register a param tensor, run an
    /// on-device SGD step against its registry buffer (which the trainer now
    /// does *without* writing the host tensor's storage), then verify that:
    ///   1. The param's host storage is STALE — `p.flatten_all()` still
    ///      matches the pre-step values (the kernel wrote the Vulkan registry
    ///      buffer, not the kt tensor's CPU storage).
    ///   2. The registry buffer is CURRENT — `resolve_resident_activation`
    ///      returns the post-step values.
    ///   3. After an explicit in-place sync (`p.slice_set(resolve(...))`,
    ///      the kt analog of candle `Var::set` — id-stable in-place overwrite),
    ///      the param's host storage matches the registry.
    /// This is the contract the lazy-sync flow relies on.
    #[test]
    fn lazy_sync_keeps_host_stale_until_explicit_sync() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 8usize;
        let lr = 0.1f32;
        let init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i as i32 - 4) as f32) * 0.05).collect();
        let expected: Vec<f32> = init
            .iter()
            .zip(grad.iter())
            .map(|(&p, &g)| p - lr * g)
            .collect();

        // kt has no `Var`; the param is a plain kt Tensor. Its `TensorId` is
        // stable (the registry keys on it) and `slice_set` mutates its storage
        // in place — exactly the id-stable, lazy-host-sync semantics candle's
        // `Var` provided here.
        let p = kiln_tensor::Tensor::from_vec(init.clone(), (n,))?;
        let g_tensor = kiln_tensor::Tensor::from_vec(grad, (n,))?;

        backend.register_resident_activation(&p)?;
        backend.register_resident_activation(&g_tensor)?;
        let dispatched = backend.dispatch_sgd_step(&p, &g_tensor, lr)?;
        assert!(dispatched);

        // (1) Host storage is still the initial values.
        let stale: Vec<f32> = p.flatten_all()?.to_vec1::<f32>()?;
        for (i, (s, w)) in stale.iter().zip(init.iter()).enumerate() {
            assert!(
                (s - w).abs() < 1e-7,
                "host storage must be stale post-dispatch: idx {i}: got {s}, init {w}"
            );
        }

        // (2) Registry has post-step values.
        let resolved = backend
            .resolve_resident_activation(&p, &[n], kiln_tensor::DType::F32)?
            .expect("must resolve after on-device dispatch");
        let resolved_v: Vec<f32> = resolved.flatten_all()?.to_vec1::<f32>()?;
        for (i, (r, w)) in resolved_v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - w).abs() < 1e-6,
                "registry must hold post-step values: idx {i}: got {r}, want {w}"
            );
        }

        // (3) After explicit in-place sync, host storage matches.
        p.slice_set(&resolved, 0, 0)?;
        let fresh: Vec<f32> = p.flatten_all()?.to_vec1::<f32>()?;
        for (i, (f, w)) in fresh.iter().zip(expected.iter()).enumerate() {
            assert!(
                (f - w).abs() < 1e-6,
                "host storage must match registry post-sync: idx {i}: got {f}, want {w}"
            );
        }

        backend.evict_resident_activation(&p);
        backend.evict_resident_activation(&g_tensor);
        Ok(())
    }

    /// dispatch_sgd_step still falls back when dtypes don't match
    /// (e.g. BF16 param but F32 grad). Mixed-precision SGD requires
    /// an F32 master copy that we don't maintain.
    #[test]
    fn dispatch_sgd_step_falls_back_on_dtype_mismatch() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = kiln_tensor::Tensor::from_vec(vec![1.0f32; 4], (4,))?.to_dtype(kiln_tensor::DType::BF16)?;
        let g = kiln_tensor::Tensor::from_vec(vec![0.5f32; 4], (4,))?; // F32
        backend.register_resident_activation(&p)?;
        backend.register_resident_activation(&g)?;
        let dispatched = backend.dispatch_sgd_step(&p, &g, 0.01)?;
        assert!(!dispatched, "dtype mismatch must fall back");
        backend.evict_resident_activation(&p);
        backend.evict_resident_activation(&g);
        Ok(())
    }

}

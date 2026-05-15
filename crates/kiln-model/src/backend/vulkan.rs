//! Vulkan backend: FlashAttention-2 and Gated DeltaNet fused kernels via Vulkan.
//!
//! candle-core 0.10.x has no native Vulkan device, so this backend manages
//! its own `vk::Device`. Normal inference still exposes a candle `Device::Cpu`
//! surface and may fall back to portable candle ops when a Vulkan backend method
//! declines a call. Vulkan-native SFT/GRPO training use the separate `VkTensor`
//! stack to keep weights, activations, loss, backward, and optimizer updates
//! resident on Vulkan buffers.
//!
//! `Ok(None)` responses route the caller to the portable candle path.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, TensorId};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::{BackendRuntime, TrainingCapabilities};
use crate::forward::{GpuAttentionWeights, GpuWeights};

/// Vulkan backend for Kiln.
///
/// Manages its own Vulkan device and dispatches compute shaders for
/// FlashAttention-2, Gated DeltaNet, and supporting operations.
#[derive(Debug)]
pub struct VulkanBackend {
    device: Device,
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
    /// Tensor path without re-checking.
    decode_resident_pool:
        OnceLock<Option<Arc<kiln_vulkan_kernel::DecodeResidentPool>>>,
    /// Cached f32 device-local buffers for immutable CPU weight tensors.
    ///
    /// This field must drop before `vulkan_device`: `VulkanBuffer` owns raw
    /// memory that must be freed before the logical Vulkan device is destroyed.
    weight_cache: Mutex<HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    /// Cached packed-bf16 device-local buffers for immutable CPU weights used
    /// by Vulkan transposed linear decode paths.
    bf16_packed_weight_cache: Mutex<HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    /// Vulkan device (owned, not from candle-core).
    ///
    /// `Arc` rather than `Box` so a `CustomOp1` impl that wants to dispatch
    /// a Vulkan kernel from inside `cpu_fwd` can capture a refcounted
    /// handle to the device — the candle CustomOp trait requires the op
    /// state to be `'static + Send + Sync`, which a borrow off `&self`
    /// can never satisfy.
    vulkan_device: Option<Arc<kiln_vulkan_kernel::VulkanDevice>>,
}

thread_local! {
    static RECURRENT_STATE_RESIDENT_SCOPE_DEPTH: Cell<usize> = const { Cell::new(0) };
    static RECURRENT_STATE_RESIDENT_CACHE: RefCell<HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>> =
        RefCell::new(HashMap::new());
}

/// General-purpose resident-activation registry keyed by candle
/// `TensorId`. Process-global (not thread-local) so worker threads
/// spawned by candle's internal parallelism, rayon, etc. see the
/// same registry as the thread that registered. Phase 3.1 of the
/// residency plan — the registry the `register_resident_activation`
/// / `evict_resident_activation` / `has_resident_activation` /
/// `update_resident_activation` / `resolve_resident_activation`
/// BackendRuntime hooks read and write.
///
/// Held behind a Mutex; per-access lock cost is negligible relative
/// to the Vulkan dispatches the registry feeds (~50µs+ each).
///
/// Separate from `RECURRENT_STATE_RESIDENT_CACHE` so the
/// GDN-specific hot path can keep its own thread-local
/// scope-limited lifecycle without growing accidental coupling to
/// non-recurrent activations.
static RESIDENT_ACTIVATION_REGISTRY: std::sync::OnceLock<
    std::sync::Mutex<HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
> = std::sync::OnceLock::new();

fn resident_registry()
-> &'static std::sync::Mutex<HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>> {
    RESIDENT_ACTIVATION_REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Helper: short, self-recovering accessor that wraps the registry's
/// mutex. Poison recovery returns the inner data so we never leave
/// the registry inaccessible just because some panicking code touched
/// it.
fn with_resident_registry<F, R>(f: F) -> R
where
    F: FnOnce(&mut HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>) -> R,
{
    let mut guard = resident_registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    f(&mut guard)
}

fn recurrent_state_resident_scope_active() -> bool {
    RECURRENT_STATE_RESIDENT_SCOPE_DEPTH.with(|depth| depth.get() > 0)
}

fn fused_gdn_resident_state_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_RESIDENT_STATE").is_err()
    })
}

/// When set, the multi-batch paged attention decode path walks the
/// block_table inside the Vulkan shader instead of compacting K/V on the
/// host with `Tensor::index_select`. Default: enabled. Disable via
/// `KILN_DISABLE_VULKAN_PAGED_DECODE_GPU_GATHER=1` to fall back to the
/// host-side gather path for parity comparisons.
fn paged_decode_gpu_gather_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_PAGED_DECODE_GPU_GATHER").is_err()
    })
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
/// Set `KILN_VULKAN_LINEAR=0` to opt back out for parity comparisons
/// or if a future regression makes the on-device path misbehave.
fn linear_prefill_apply_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| kiln_core::env_flag::env_flag("KILN_VULKAN_LINEAR", true))
}

fn enter_recurrent_state_resident_scope() {
    RECURRENT_STATE_RESIDENT_SCOPE_DEPTH.with(|depth| {
        depth.set(depth.get() + 1);
    });
}

fn exit_recurrent_state_resident_scope() {
    RECURRENT_STATE_RESIDENT_SCOPE_DEPTH.with(|depth| {
        let previous = depth.get();
        if previous == 0 {
            return;
        }
        let next = previous - 1;
        depth.set(next);
    });
}

impl VulkanBackend {
    pub fn new(device: Device) -> Self {
        let gdn_enabled = std::env::var("KILN_DISABLE_GDN_KERNEL").is_err();
        let gdn_prefill_in_proj_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_VULKAN_GDN_PREFILL_IN_PROJ").is_err();
        let gdn_gates_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATES").is_err();
        let gdn_gated_rms_norm_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM").is_err();
        // The fused full-chunk shader is parity-covered, but default-on A070
        // latency regressed on Strix Halo. Keep it available for explicit
        // tuning without changing the production route.
        let gdn_full_chunk_forward_enabled =
            gdn_enabled && std::env::var("KILN_ENABLE_VULKAN_GDN_FULL_CHUNK_FORWARD").is_ok();
        // forward_sub is opt-in only (default off): solve_tri shared-memory
        // layout is not yet validated against CPU parity and may exceed
        // maxComputeSharedMemorySize on many GPUs.
        //
        // Conv1d prefill now wins on Strix Halo, while single-token update
        // still regresses decode latency. Keep update opt-in and leave a
        // prefill rollback for driver/model-specific follow-up.
        let fused_conv1d_update_enabled = gdn_enabled
            && (std::env::var("KILN_ENABLE_VULKAN_FUSED_CONV1D").is_ok()
                || std::env::var("KILN_ENABLE_VULKAN_FUSED_CONV1D_UPDATE").is_ok());
        let fused_conv1d_prefill_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_VULKAN_FUSED_CONV1D_PREFILL").is_err();
        let conv1d_prefill_single_submit_enabled = fused_conv1d_prefill_enabled
            && std::env::var("KILN_DISABLE_VULKAN_CONV1D_PREFILL_SINGLE_SUBMIT").is_err();
        let gdn_forward_sub_enabled =
            gdn_enabled && std::env::var("KILN_ENABLE_VULKAN_GDN_FORWARD_SUB").is_ok();
        // The fused GDN decode path is validated, but for bs=1 it remains
        // run-to-run unstable on Strix Halo. Batch decode enables it by shape
        // in `gdn_decode_gates_recurrent_rmsnorm`; this env gates bs=1 only.
        let gdn_decode_fused_enabled =
            gdn_enabled && std::env::var("KILN_ENABLE_VULKAN_GDN_DECODE_FUSED").is_ok();
        let gdn_recurrent_unexpanded_qk_enabled = gdn_enabled
            && std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_UNEXPANDED_QK").is_err();
        let gdn_recurrent_qk_norm_unexpanded_enabled = gdn_recurrent_unexpanded_qk_enabled
            && std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_QK_NORM").is_err();
        let linear_decode_enabled = std::env::var("KILN_DISABLE_VULKAN_LINEAR_DECODE").is_err();
        let bf16_packed_linear_weights_enabled = linear_decode_enabled
            && std::env::var("KILN_DISABLE_VULKAN_BF16_PACKED_LINEAR_WEIGHTS").is_err();
        let bf16_packed_gdn_in_proj_weights_enabled = gdn_enabled
            && std::env::var("KILN_DISABLE_VULKAN_BF16_PACKED_GDN_IN_PROJ_WEIGHTS").is_err();
        let linear_argmax_batch_enabled =
            std::env::var("KILN_DISABLE_VULKAN_LINEAR_ARGMAX_BATCH").is_err();
        let full_attn_qkv_enabled = std::env::var("KILN_DISABLE_VULKAN_FULL_ATTN_QKV").is_err();
        let bf16_packed_full_attn_qkv_weights_enabled = full_attn_qkv_enabled
            && std::env::var("KILN_DISABLE_VULKAN_BF16_PACKED_FULL_ATTN_QKV_WEIGHTS").is_err();
        let paged_attn_decode_batch_enabled =
            std::env::var("KILN_DISABLE_VULKAN_PAGED_ATTN_DECODE_BATCH").is_err();
        // Full fused MLP decode is validated for single-token no-LoRA decode.
        // After descriptor-pool reuse and tiled projection kernels it is now
        // consistently faster than the split generic GEMV path on Strix Halo.
        let mlp_decode_enabled = std::env::var("KILN_DISABLE_VULKAN_MLP_DECODE").is_err();
        let bf16_packed_mlp_decode_weights_enabled = mlp_decode_enabled
            && std::env::var("KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS").is_err();
        let mlp_bf16_gate_up_f32_down_enabled = bf16_packed_mlp_decode_weights_enabled
            && std::env::var("KILN_DISABLE_VULKAN_MLP_BF16_GATE_UP_F32_DOWN").is_err();
        // The fused Vulkan MLP gate/up shader is validated, but on Strix Halo
        // it was slower than the generic cached GEMV path in short decode
        // benchmarks. Keep it opt-in until it is tiled/tuned.
        let mlp_gate_up_enabled = std::env::var("KILN_ENABLE_VULKAN_MLP_GATE_UP").is_ok();
        let weight_prewarm_enabled = std::env::var("KILN_DISABLE_VULKAN_WEIGHT_PREWARM").is_err();
        // Device-resident recurrent state is correct but regressed the live
        // Strix Halo batcher A/B in A129 because row/batch buffer copies cost
        // more than the saved readback/upload at the current batch shape.
        let recurrent_state_residency_enabled = gdn_enabled
            && std::env::var("KILN_ENABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE").is_ok()
            && std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE").is_err();
        // Default ON: every Vulkan build that brings up a logical device
        // wants to route decode through the resident path. Pool feasibility
        // is checked later at first use; if the device can't fit the ring
        // (Strix Halo near memory limit) the call site falls back
        // transparently to the per-call Tensor path and emits a one-time
        // tracing::warn! — exactly the contract spelled out in gate (b)
        // of docs/vk_resident_decode_plan.md.
        let resident_decode_enabled =
            kiln_core::env_flag::env_flag("KILN_VULKAN_RESIDENT_DECODE", true);

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

        Self {
            device,
            gdn_enabled,
            gdn_prefill_in_proj_enabled,
            gdn_gates_enabled,
            gdn_gated_rms_norm_enabled,
            gdn_full_chunk_forward_enabled,
            fused_conv1d_update_enabled,
            fused_conv1d_prefill_enabled,
            conv1d_prefill_single_submit_enabled,
            gdn_forward_sub_enabled,
            gdn_decode_fused_enabled,
            gdn_recurrent_unexpanded_qk_enabled,
            gdn_recurrent_qk_norm_unexpanded_enabled,
            linear_decode_enabled,
            linear_argmax_batch_enabled,
            full_attn_qkv_enabled,
            paged_attn_decode_batch_enabled,
            mlp_decode_enabled,
            mlp_gate_up_enabled,
            mlp_bf16_gate_up_f32_down_enabled,
            bf16_packed_linear_weights_enabled,
            bf16_packed_gdn_in_proj_weights_enabled,
            bf16_packed_full_attn_qkv_weights_enabled,
            bf16_packed_mlp_decode_weights_enabled,
            weight_prewarm_enabled,
            recurrent_state_residency_enabled,
            resident_decode_enabled,
            decode_resident_pool: OnceLock::new(),
            weight_cache: Mutex::new(HashMap::new()),
            bf16_packed_weight_cache: Mutex::new(HashMap::new()),
            vulkan_device,
        }
    }

    fn has_vulkan(&self) -> bool {
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
    /// per-call Tensor fallback does not re-probe on every decode
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
                             falling back to per-call Tensor path"
                        );
                        None
                    }
                }
            })
            .as_ref()
    }

    fn cached_f32_weight_buffer(
        &self,
        weight: &Tensor,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let key = weight.id();

        {
            let cache = self
                .weight_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("Vulkan weight cache mutex poisoned"))?;
            if let Some(buffer) = cache.get(&key) {
                return Ok(Arc::clone(buffer));
            }
        }

        let buffer = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(vk_device, weight)
            .context("upload GDN projection weight to Vulkan")?;
        let buffer = Arc::new(buffer);

        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Vulkan weight cache mutex poisoned"))?;
        Ok(Arc::clone(cache.entry(key).or_insert(buffer)))
    }

    fn use_bf16_packed_linear_weight(&self, weight: &Tensor) -> bool {
        self.bf16_packed_linear_weights_enabled && weight.dtype() == DType::BF16
    }

    fn use_bf16_packed_gdn_in_proj_weights(&self, weights: &[&Tensor]) -> bool {
        self.bf16_packed_gdn_in_proj_weights_enabled
            && weights.iter().all(|weight| weight.dtype() == DType::BF16)
    }

    fn use_bf16_packed_full_attn_qkv_weights(&self, weights: &[&Tensor]) -> bool {
        self.bf16_packed_full_attn_qkv_weights_enabled
            && weights.iter().all(|weight| weight.dtype() == DType::BF16)
    }

    fn use_bf16_packed_mlp_decode_weights(&self, weights: &[&Tensor]) -> bool {
        self.bf16_packed_mlp_decode_weights_enabled
            && weights.iter().all(|weight| weight.dtype() == DType::BF16)
    }

    fn cached_bf16_packed_weight_buffer(
        &self,
        weight: &Tensor,
    ) -> Result<Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let key = weight.id();

        {
            let cache = self
                .bf16_packed_weight_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("Vulkan packed bf16 weight cache mutex poisoned"))?;
            if let Some(buffer) = cache.get(&key) {
                return Ok(Arc::clone(buffer));
            }
        }

        let buffer =
            kiln_vulkan_kernel::kernels::upload_tensor_bf16_packed_buffer(vk_device, weight)
                .context("upload packed BF16 projection weight to Vulkan")?;
        let buffer = Arc::new(buffer);

        let mut cache = self
            .bf16_packed_weight_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("Vulkan packed bf16 weight cache mutex poisoned"))?;
        Ok(Arc::clone(cache.entry(key).or_insert(buffer)))
    }

    fn prewarm_f32_weight(
        &self,
        name: &str,
        weight: &Tensor,
        count: &mut usize,
        bytes: &mut usize,
    ) -> Result<()> {
        self.cached_f32_weight_buffer(weight)
            .with_context(|| format!("prewarm Vulkan decode weight {name}"))?;
        *count += 1;
        *bytes += weight.elem_count() * std::mem::size_of::<f32>();
        Ok(())
    }

    fn prewarm_bf16_packed_weight(
        &self,
        name: &str,
        weight: &Tensor,
        count: &mut usize,
        bytes: &mut usize,
    ) -> Result<()> {
        self.cached_bf16_packed_weight_buffer(weight)
            .with_context(|| format!("prewarm Vulkan packed BF16 decode weight {name}"))?;
        *count += 1;
        *bytes += weight.elem_count().div_ceil(2) * std::mem::size_of::<u32>();
        Ok(())
    }

    fn prewarm_linear_weight(
        &self,
        name: &str,
        weight: &Tensor,
        f32_count: &mut usize,
        f32_bytes: &mut usize,
        bf16_count: &mut usize,
        bf16_bytes: &mut usize,
    ) -> Result<()> {
        if self.use_bf16_packed_linear_weight(weight) {
            self.prewarm_bf16_packed_weight(name, weight, bf16_count, bf16_bytes)
        } else {
            self.prewarm_f32_weight(name, weight, f32_count, f32_bytes)
        }
    }

    fn prewarm_gdn_in_proj_weight(
        &self,
        name: &str,
        weight: &Tensor,
        f32_count: &mut usize,
        f32_bytes: &mut usize,
        bf16_count: &mut usize,
        bf16_bytes: &mut usize,
    ) -> Result<()> {
        if self.bf16_packed_gdn_in_proj_weights_enabled && weight.dtype() == DType::BF16 {
            self.prewarm_bf16_packed_weight(name, weight, bf16_count, bf16_bytes)
        } else {
            self.prewarm_f32_weight(name, weight, f32_count, f32_bytes)
        }
    }

    fn prewarm_full_attn_qkv_weights(
        &self,
        layer_idx: usize,
        q_weight_t: &Tensor,
        k_weight_t: &Tensor,
        v_weight_t: &Tensor,
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
        if self.use_bf16_packed_full_attn_qkv_weights(&[q_weight_t, k_weight_t, v_weight_t]) {
            for (suffix, weight) in weights {
                self.prewarm_bf16_packed_weight(
                    &format!("layers.{layer_idx}.attention.{suffix}"),
                    weight,
                    bf16_count,
                    bf16_bytes,
                )?;
            }
        } else {
            for (suffix, weight) in weights {
                self.prewarm_f32_weight(
                    &format!("layers.{layer_idx}.attention.{suffix}"),
                    weight,
                    f32_count,
                    f32_bytes,
                )?;
            }
        }
        Ok(())
    }

    fn prewarm_mlp_decode_weights(
        &self,
        layer_idx: usize,
        gate_weight_t: &Tensor,
        up_weight_t: &Tensor,
        down_weight_t: &Tensor,
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
        if self.use_bf16_packed_mlp_decode_weights(&[gate_weight_t, up_weight_t, down_weight_t]) {
            for (suffix, weight) in weights {
                self.prewarm_bf16_packed_weight(
                    &format!("layers.{layer_idx}.mlp.{suffix}"),
                    weight,
                    bf16_count,
                    bf16_bytes,
                )?;
            }
            for (suffix, weight) in weights {
                self.prewarm_f32_weight(
                    &format!("layers.{layer_idx}.mlp.{suffix}"),
                    weight,
                    f32_count,
                    f32_bytes,
                )?;
            }
        } else {
            for (suffix, weight) in weights {
                self.prewarm_f32_weight(
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
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (_b, _seq_len, _num_heads, head_dim) = q.dims4()?;
        // sdpa_prefill_f32.comp uses local_size_x=128. Larger head_dim
        // would need a multi-pass reduction the v1 kernel doesn't do.
        if head_dim > 128 {
            return Ok(None);
        }

        // Cast to F32 if needed — the kernel is F32-in/F32-out. The
        // BF16→F32 promotion is cheap relative to the SDPA compute
        // (e.g. T=918, H=16, dh=128 ≈ 7.5 MB to convert vs. 7 GFLOP
        // to compute) and matches what the candle CPU baseline did
        // implicitly via broadcast_matmul_cpu_compatible.
        let in_dtype = q.dtype();
        let q_f32 = if in_dtype == DType::F32 {
            q.clone()
        } else {
            q.to_dtype(DType::F32)?
        };
        let k_f32 = if in_dtype == DType::F32 {
            k.clone()
        } else {
            k.to_dtype(DType::F32)?
        };
        let v_f32 = if in_dtype == DType::F32 {
            v.clone()
        } else {
            v.to_dtype(DType::F32)?
        };

        let out_f32 = kiln_vulkan_kernel::kernels::dispatch_sdpa_prefill_f32(
            vk_device,
            &q_f32,
            &k_f32,
            &v_f32,
            softmax_scale,
            causal,
        )?;

        let out = if in_dtype == DType::F32 {
            out_f32
        } else {
            out_f32.to_dtype(in_dtype)?
        };
        Ok(Some(out))
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        if let Ok(mut cache) = self.weight_cache.lock() {
            cache.clear();
        }
    }
}

impl BackendRuntime for VulkanBackend {
    fn name(&self) -> &'static str {
        if self.has_vulkan() { "vulkan" } else { "cpu" }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn training_capabilities(&self) -> TrainingCapabilities {
        TrainingCapabilities {
            projection_training: "VulkanLinearOp CustomOp1 when enabled",
            flce_loss: "Vulkan offset matmul provider when enabled; FLCE remains chunked",
            rmsnorm_training: "Vulkan RMSNorm autograd path auto-gated by row count",
            resident_activation: "Vulkan buffer registry",
            lora_delta_training: "VulkanLoraOp CustomOp3 with registry-resident A/B",
            sgd_step: "Vulkan in-place registry update when operands are resident",
            adamw_step: "Vulkan in-place registry update when operands are resident",
            native_training: "vk_native_sft_train/vk_native_grpo_train enabled by default on Vulkan",
        }
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
        // Not yet implemented — returning false so callers don't skip
        // their preamble work only to get Ok(None) back.
        false
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

    fn materialize_gdn_recurrent_resident_state(&self, state: &mut Tensor) -> Result<()> {
        if !self.recurrent_state_residency_enabled {
            return Ok(());
        }
        let state_id = state.id();
        let resident_state =
            RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow_mut().remove(&state_id));
        let Some(resident_state) = resident_state else {
            return Ok(());
        };

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
        *state = kiln_vulkan_kernel::kernels::create_tensor_from_data(
            &data,
            state.dims().as_ref(),
            state.dtype(),
        )?;
        Ok(())
    }

    fn evict_gdn_recurrent_resident_state(&self, state: &Tensor) {
        if !self.recurrent_state_residency_enabled {
            return;
        }
        let state_id = state.id();
        RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
            cache.borrow_mut().remove(&state_id);
        });
    }

    fn has_gdn_recurrent_resident_state(&self, state: &Tensor) -> bool {
        if !self.recurrent_state_residency_enabled {
            return false;
        }
        let state_id = state.id();
        RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow().contains_key(&state_id))
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
    /// records the buffer under the tensor's `TensorId`. The caller
    /// owns lifecycle — Phase 3.2 will pair every register with a
    /// matching evict at the appropriate autograd boundary. Until then
    /// any caller using this hook must clean up explicitly to avoid
    /// leaking VRAM.
    fn register_resident_activation(&self, tensor: &Tensor) -> Result<()> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(());
        };
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
        let bytes = if tensor.dtype() == DType::BF16 {
            kiln_vulkan_kernel::kernels::extract_tensor_packed_bf16_bytes_pub(tensor)?.0
        } else {
            kiln_vulkan_kernel::kernels::extract_tensor_bytes(tensor)?.0
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

    fn evict_resident_activation(&self, tensor: &Tensor) {
        let id = tensor.id();
        with_resident_registry(|cache| {
            cache.remove(&id);
        });
    }

    fn update_resident_activation(&self, tensor: &Tensor) -> Result<()> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(());
        };
        let id = tensor.id();
        let buffer = with_resident_registry(|cache| cache.get(&id).cloned());
        let Some(buffer) = buffer else {
            // Not registered — caller probably skipped the registration
            // path. No-op.
            return Ok(());
        };
        // Same encoding choice as register_resident_activation.
        let bytes = if tensor.dtype() == DType::BF16 {
            kiln_vulkan_kernel::kernels::extract_tensor_packed_bf16_bytes_pub(tensor)?.0
        } else {
            kiln_vulkan_kernel::kernels::extract_tensor_bytes(tensor)?.0
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

    fn has_resident_activation(&self, tensor: &Tensor) -> bool {
        let id = tensor.id();
        with_resident_registry(|cache| cache.contains_key(&id))
    }

    fn resolve_resident_activation(
        &self,
        tensor: &Tensor,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Option<Tensor>> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(None);
        };
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
        // other dtypes hold F32 bytes. To avoid a `half` crate dep
        // (only enabled under the cuda feature), reconstruct BF16 by
        // bit-expanding each 16-bit lane into f32 (`bits << 16`) and
        // then casting back to BF16 via candle.
        let resolved = if dtype == DType::BF16 {
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
            Tensor::from_vec(f32_data, shape, &Device::Cpu)?.to_dtype(DType::BF16)?
        } else {
            kiln_vulkan_kernel::kernels::create_tensor_from_data(&bytes, shape, dtype)
                .context("resolve_resident_activation: create_tensor_from_data")?
        };
        Ok(Some(resolved))
    }

    fn dispatch_sgd_step(&self, param: &Tensor, grad: &Tensor, lr: f32) -> Result<bool> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(false);
        };
        // Both operands must be resident — no support for mixed
        // resident/CPU yet (would require a per-call upload that
        // defeats the purpose of the on-device update).
        let param_id = param.id();
        let grad_id = grad.id();
        let lookup = with_resident_registry(|cache| {
            cache
                .get(&param_id)
                .and_then(|p| cache.get(&grad_id).map(|g| (Arc::clone(p), Arc::clone(g))))
        });
        let Some((param_buf, grad_buf)) = lookup else {
            return Ok(false);
        };
        // Dispatch the dtype-appropriate kernel. Param and grad must
        // share dtype (mixed-precision SGD is a different design that
        // would need an F32 master copy). LoRA Vars are BF16 in
        // production; activations and intermediate buffers are F32.
        if param.dtype() != grad.dtype() {
            return Ok(false);
        }
        let n_elements: usize = param.shape().elem_count();
        if n_elements != grad.shape().elem_count() {
            anyhow::bail!(
                "dispatch_sgd_step: param ({:?}) and grad ({:?}) have different element counts",
                param.shape(),
                grad.shape(),
            );
        }
        static FIRST_SGD_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        FIRST_SGD_LOGGED.get_or_init(|| {
            tracing::info!(
                n_elements,
                lr,
                dtype = ?param.dtype(),
                "VulkanBackend::dispatch_sgd_step first call"
            );
        });
        match param.dtype() {
            DType::F32 => {
                kiln_vulkan_kernel::kernels::dispatch_sgd_step_f32(
                    vk_device, &param_buf, &grad_buf, n_elements, lr,
                )?;
                Ok(true)
            }
            DType::BF16 => {
                kiln_vulkan_kernel::kernels::dispatch_sgd_step_bf16(
                    vk_device, &param_buf, &grad_buf, n_elements, lr,
                )?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn dispatch_adamw_step(
        &self,
        param: &Tensor,
        grad: &Tensor,
        first_moment: &Tensor,
        second_moment: &Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<bool> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(false);
        };
        if step < 1 {
            anyhow::bail!("dispatch_adamw_step: step must be 1-indexed (>=1), got {step}");
        }
        if param.dtype() != grad.dtype()
            || param.dtype() != first_moment.dtype()
            || param.dtype() != second_moment.dtype()
        {
            anyhow::bail!(
                "dispatch_adamw_step: dtype mismatch (param={:?}, grad={:?}, m={:?}, v={:?})",
                param.dtype(),
                grad.dtype(),
                first_moment.dtype(),
                second_moment.dtype(),
            );
        }
        let n_elements: usize = param.shape().elem_count();
        if n_elements != grad.shape().elem_count()
            || n_elements != first_moment.shape().elem_count()
            || n_elements != second_moment.shape().elem_count()
        {
            anyhow::bail!(
                "dispatch_adamw_step: element count mismatch (param={}, grad={}, m={}, v={})",
                n_elements,
                grad.shape().elem_count(),
                first_moment.shape().elem_count(),
                second_moment.shape().elem_count(),
            );
        }
        let p_id = param.id();
        let g_id = grad.id();
        let m_id = first_moment.id();
        let v_id = second_moment.id();
        let bufs = with_resident_registry(|cache| {
            let p = cache.get(&p_id).map(Arc::clone)?;
            let g = cache.get(&g_id).map(Arc::clone)?;
            let m = cache.get(&m_id).map(Arc::clone)?;
            let v = cache.get(&v_id).map(Arc::clone)?;
            Some((p, g, m, v))
        });
        let Some((param_buf, grad_buf, m_buf, v_buf)) = bufs else {
            return Ok(false);
        };
        static FIRST_ADAMW_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        FIRST_ADAMW_LOGGED.get_or_init(|| {
            tracing::info!(
                n_elements,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
                dtype = ?param.dtype(),
                "VulkanBackend::dispatch_adamw_step first call"
            );
        });
        match param.dtype() {
            DType::F32 => {
                kiln_vulkan_kernel::kernels::dispatch_adamw_step_f32(
                    vk_device,
                    &param_buf,
                    &grad_buf,
                    &m_buf,
                    &v_buf,
                    n_elements,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    step,
                )?;
                Ok(true)
            }
            DType::BF16 => {
                kiln_vulkan_kernel::kernels::dispatch_adamw_step_bf16(
                    vk_device,
                    &param_buf,
                    &grad_buf,
                    &m_buf,
                    &v_buf,
                    n_elements,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    step,
                )?;
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn lora_delta_resident(
        &self,
        x: &Tensor,
        a: &Tensor,
        b: &Tensor,
        scale: f32,
    ) -> Result<Option<Tensor>> {
        let Some(vk_device) = self.vulkan_device.as_ref() else {
            return Ok(None);
        };
        if !self.has_vulkan() || !self.linear_decode_enabled {
            return Ok(None);
        }
        // Kernel constraint: weight buffer is bf16-packed.
        if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
            return Ok(None);
        }
        // Both A and B must be registry-resident.
        let a_id = a.id();
        let b_id = b.id();
        let bufs = with_resident_registry(|cache| {
            cache
                .get(&a_id)
                .and_then(|ab| cache.get(&b_id).map(|bb| (Arc::clone(ab), Arc::clone(bb))))
        });
        let Some((a_buf, b_buf)) = bufs else {
            return Ok(None);
        };
        // Shape inference. A is `[rank, in]`, B is `[out, rank]`. x is
        // `[..., in]`. delta is `[..., out]`.
        let Ok((rank_a, in_features)) = a.dims2() else {
            return Ok(None);
        };
        let Ok((out_features, rank_b)) = b.dims2() else {
            return Ok(None);
        };
        if rank_a != rank_b {
            return Ok(None);
        }
        let in_dtype = x.dtype();
        let x_dims = x.dims().to_vec();
        if x_dims.is_empty() || *x_dims.last().unwrap() != in_features {
            return Ok(None);
        }
        let row_count: usize = x_dims[..x_dims.len() - 1].iter().product();
        if row_count == 0 {
            return Ok(None);
        }
        // Construct the autograd-safe CustomOp3 wrapper. apply_op3
        // builds a backprop link from the returned Tensor through x,
        // a, and b — VulkanLoraOp::bwd computes analytic gradients
        // for all three. This lets the trainer's loss.backward()
        // produce real grad_A and grad_B instead of dropping them
        // (which is what the prior leaf-Tensor return did).
        let op = crate::backend::vulkan_lora_op::VulkanLoraOp {
            vk_device: Arc::clone(vk_device),
            a_buffer: Arc::clone(&a_buf),
            b_buffer: Arc::clone(&b_buf),
            rank: rank_a,
            in_features,
            out_features,
            scale,
            out_dtype: in_dtype,
        };
        let delta = x
            .apply_op3(a, b, op)
            .context("VulkanLoraOp apply_op3 failed")?;
        // One-shot trace so the operator can confirm the on-device
        // LoRA delta path engaged.
        static FIRST_LORA_DELTA_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        FIRST_LORA_DELTA_LOGGED.get_or_init(|| {
            tracing::info!(
                row_count,
                in_features,
                rank = rank_a,
                out_features,
                scale,
                "VulkanBackend::lora_delta_resident first call (CustomOp3 / autograd-safe)"
            );
        });
        Ok(Some(delta))
    }

    fn assemble_gdn_recurrent_resident_batch_rows(
        &self,
        rows: &[&Tensor],
        batch: &Tensor,
    ) -> Result<bool> {
        if !self.recurrent_state_residency_enabled
            || !recurrent_state_resident_scope_active()
            || !self.has_vulkan()
            || rows.is_empty()
        {
            return Ok(false);
        }
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
                || !matches!(row.device(), Device::Cpu)
            {
                return Ok(false);
            }
        }

        let row_buffers = RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
            let cache = cache.borrow();
            rows.iter()
                .map(|row| cache.get(&row.id()).cloned())
                .collect::<Option<Vec<_>>>()
        });
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
        RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
            cache.borrow_mut().insert(batch.id(), batch_buffer);
        });
        Ok(true)
    }

    fn scatter_gdn_recurrent_resident_batch_rows(
        &self,
        batch: &Tensor,
        destinations: &mut [&mut Tensor],
    ) -> Result<bool> {
        if !self.recurrent_state_residency_enabled
            || !recurrent_state_resident_scope_active()
            || !self.has_vulkan()
            || destinations.is_empty()
        {
            return Ok(false);
        }
        let Ok((batch_rows, heads, dk, dv)) = batch.dims4() else {
            return Ok(false);
        };
        if destinations.len() != batch_rows {
            return Ok(false);
        }
        let batch_buffer =
            RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow().get(&batch.id()).cloned());
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
            let old_id = dst.id();
            let placeholder = batch.narrow(0, row_idx, 1)?.contiguous()?;
            if placeholder.dtype() != batch.dtype()
                || placeholder.dims() != [1, heads, dk, dv]
                || !matches!(placeholder.device(), Device::Cpu)
            {
                return Ok(false);
            }
            **dst = placeholder;
            RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
                let mut cache = cache.borrow_mut();
                cache.remove(&old_id);
                cache.insert(dst.id(), row_buffer);
            });
        }
        RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
            cache.borrow_mut().remove(&batch.id());
        });

        Ok(true)
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
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if q.dtype() != DType::BF16 || !self.has_vulkan() {
            return Ok(None);
        }
        self.flash_attn_prefill_vulkan(q, k, v, softmax_scale, causal)
    }

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
        if !self.has_vulkan() {
            return Ok(None);
        }
        // TODO: Implement Vulkan paged decode dispatch
        Ok(None)
    }

    fn flash_attn_paged_decode_contiguous_batch_dyn_seqlen(
        &self,
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        block_table: &Tensor,
        seqused_k: &Tensor,
        max_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan()
            || !self.paged_attn_decode_batch_enabled
            || q.dtype() != DType::F32
            || k_pool.dtype() != DType::F32
            || v_pool.dtype() != DType::F32
        {
            return Ok(None);
        }
        if !causal {
            return Ok(None);
        }
        if !matches!(q.device(), Device::Cpu)
            || !matches!(k_pool.device(), Device::Cpu)
            || !matches!(v_pool.device(), Device::Cpu)
            || !matches!(block_table.device(), Device::Cpu)
            || !matches!(seqused_k.device(), Device::Cpu)
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
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        let seq_i32 = seqused_k
            .flatten_all()?
            .to_dtype(DType::I32)?
            .to_vec1::<i32>()?;
        let mut seq_lens = Vec::with_capacity(batch);
        for row in 0..batch {
            let row_len = usize::try_from(seq_i32[row])
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
        // Skip the host-side gather + candle index_select entirely when the
        // GPU-paged path is enabled (default on for batch > 1). The shader
        // walks `block_table` inline against the resident pool, so the host
        // never materializes a `batch * max_seqlen_k * num_kv_heads * head_dim`
        // compacted tensor and never runs a CPU index_select over the pool.
        if paged_decode_gpu_gather_enabled() && batch > 1 {
            let out = kiln_vulkan_kernel::kernels::dispatch_paged_attn_decode_batch_paged_f32(
                vk_device,
                q,
                k_pool,
                v_pool,
                &block_data,
                &seq_lens,
                batch,
                max_blocks_per_seq,
                page_block_size,
                softmax_scale,
            )
            .context("paged_attn_decode_batch_paged kernel failed")?;
            return Ok(Some(out));
        }

        // Single-row fallback (batch == 1) keeps the original compacted path
        // for now — the gather cost is negligible at batch=1 and the kernel
        // is well-tuned for that shape. Build the gather indices and slice
        // the pool.
        let mut gather_slots = Vec::with_capacity(batch * max_seqlen_k);
        for row in 0..batch {
            let row_len = seq_lens[row] as usize;
            for pos in 0..max_seqlen_k {
                if pos >= row_len {
                    gather_slots.push(0);
                    continue;
                }
                let block_idx = pos / page_block_size;
                let offset = pos % page_block_size;
                let block = block_data[row * max_blocks_per_seq + block_idx] as usize;
                let slot = block * page_block_size + offset;
                gather_slots
                    .push(u32::try_from(slot).context("Vulkan paged decode slot exceeds u32")?);
            }
        }
        let gather =
            Tensor::from_slice(gather_slots.as_slice(), batch * max_seqlen_k, q.device())?;
        let k_compact = k_pool
            .index_select(&gather, 0)?
            .reshape((batch, max_seqlen_k, num_kv_heads, head_dim))?
            .contiguous()?;
        let v_compact = v_pool
            .index_select(&gather, 0)?
            .reshape((batch, max_seqlen_k, num_kv_heads, head_dim))?
            .contiguous()?;

        let out = kiln_vulkan_kernel::kernels::dispatch_paged_attn_decode_batch_f32(
            vk_device,
            q,
            &k_compact,
            &v_compact,
            &seq_lens,
            softmax_scale,
        )
        .context("paged_attn_decode_batch kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_in_proj_decode(
        &self,
        x: &Tensor,
        in_proj_qkv_t: &Tensor,
        in_proj_z_t: &Tensor,
        in_proj_a_t: &Tensor,
        in_proj_b_t: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor)>> {
        if !self.has_vulkan() || !self.gdn_enabled || x.dtype() != DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu)
            || !matches!(in_proj_qkv_t.device(), Device::Cpu)
            || !matches!(in_proj_z_t.device(), Device::Cpu)
            || !matches!(in_proj_a_t.device(), Device::Cpu)
            || !matches!(in_proj_b_t.device(), Device::Cpu)
        {
            return Ok(None);
        }

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
        let dispatch_x = if seq_len == 1 {
            x.clone()
        } else {
            x.reshape((row_count, 1usize, hidden))?
        };

        let (qkv, z, a, b) = if self.use_bf16_packed_gdn_in_proj_weights(&[
            in_proj_qkv_t,
            in_proj_z_t,
            in_proj_a_t,
            in_proj_b_t,
        ]) {
            let qkv_buf = self.cached_bf16_packed_weight_buffer(in_proj_qkv_t)?;
            let z_buf = self.cached_bf16_packed_weight_buffer(in_proj_z_t)?;
            let a_buf = self.cached_bf16_packed_weight_buffer(in_proj_a_t)?;
            let b_buf = self.cached_bf16_packed_weight_buffer(in_proj_b_t)?;
            kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached_bf16_weights(
                vk_device,
                &dispatch_x,
                &qkv_buf,
                &z_buf,
                &a_buf,
                &b_buf,
                hidden,
                qkv_dim,
                z_dim,
                a_dim,
                b_dim,
            )
        } else {
            let qkv_buf = self.cached_f32_weight_buffer(in_proj_qkv_t)?;
            let z_buf = self.cached_f32_weight_buffer(in_proj_z_t)?;
            let a_buf = self.cached_f32_weight_buffer(in_proj_a_t)?;
            let b_buf = self.cached_f32_weight_buffer(in_proj_b_t)?;
            kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached(
                vk_device,
                &dispatch_x,
                &qkv_buf,
                &z_buf,
                &a_buf,
                &b_buf,
                hidden,
                qkv_dim,
                z_dim,
                a_dim,
                b_dim,
            )
        }
        .context("gdn_in_proj_decode kernel failed")?;
        let result = if seq_len == 1 {
            (qkv, z, a, b)
        } else {
            (
                qkv.reshape((batch, seq_len, qkv_dim))?,
                z.reshape((batch, seq_len, z_dim))?,
                a.reshape((batch, seq_len, a_dim))?,
                b.reshape((batch, seq_len, b_dim))?,
            )
        };
        Ok(Some(result))
    }

    fn gdn_decode_gates_recurrent_rmsnorm(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        state: &mut Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.gdn_enabled || q.dtype() != DType::F32 {
            return Ok(None);
        }
        if !matches!(q.device(), Device::Cpu)
            || !matches!(k.device(), Device::Cpu)
            || !matches!(v.device(), Device::Cpu)
            || !matches!(a.device(), Device::Cpu)
            || !matches!(b.device(), Device::Cpu)
            || !matches!(a_log.device(), Device::Cpu)
            || !matches!(dt_bias.device(), Device::Cpu)
            || !matches!(state.device(), Device::Cpu)
            || !matches!(z.device(), Device::Cpu)
            || !matches!(weight.device(), Device::Cpu)
        {
            return Ok(None);
        }
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
        let Ok((state_batch, state_nv, state_dk, state_dv)) = state.dims4() else {
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
            let state_id = state.id();
            let resident_state =
                RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow().get(&state_id).cloned());
            let (out, resident_state) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state(
                    vk_device,
                    q,
                    k,
                    v,
                    a,
                    b,
                    a_log,
                    dt_bias,
                    state,
                    z,
                    weight,
                    eps as f32,
                    resident_state,
                )
                .context("gdn_decode_gates_recurrent_rmsnorm resident-state kernel failed")?;
            RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
                cache.borrow_mut().insert(state_id, resident_state);
            });
            return Ok(Some(out));
        }
        let (out, new_state) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm(
                vk_device,
                q,
                k,
                v,
                a,
                b,
                a_log,
                dt_bias,
                state,
                z,
                weight,
                eps as f32,
                skip_state_readback,
            )
            .context("gdn_decode_gates_recurrent_rmsnorm kernel failed")?;
        if !skip_state_readback {
            *state = new_state;
        }
        Ok(Some(out))
    }

    fn linear_decode(&self, x: &Tensor, weight_t: &Tensor) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.linear_decode_enabled || x.dtype() != DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu) || !matches!(weight_t.device(), Device::Cpu) {
            return Ok(None);
        }

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
        let dispatch_x = if seq_len == 1 {
            x.clone()
        } else {
            x.reshape((row_count, 1usize, hidden))?
        };
        let out = if self.use_bf16_packed_linear_weight(weight_t) {
            let weight_buf = self.cached_bf16_packed_weight_buffer(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights(
                vk_device,
                &dispatch_x,
                &weight_buf,
                row_count,
                hidden,
                out_dim,
            )
        } else {
            let weight_buf = self.cached_f32_weight_buffer(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached(
                vk_device,
                &dispatch_x,
                &weight_buf,
                row_count,
                hidden,
                out_dim,
            )
        }
        .context("linear_decode kernel failed")?;
        let out = if seq_len == 1 {
            out
        } else {
            out.reshape((batch, seq_len, out_dim))?
        };
        Ok(Some(out))
    }

    fn linear_prefill_apply(&self, x: &Tensor, weight_t: &Tensor) -> Result<Option<Tensor>> {
        // Opt-in until end-to-end training parity has been validated on
        // production-sized payloads. The CustomOp1 itself is unit-test
        // covered (forward + backward parity) and per-tensor parity has
        // been verified on the actual Strix Halo device, but the
        // integration into every projection in forward.rs is the kind
        // of thing that benefits from staged rollout. Set
        // KILN_VULKAN_LINEAR=1 to enable.
        if !linear_prefill_apply_enabled() {
            return Ok(None);
        }
        // One-shot dispatch trace so the operator can confirm the
        // CustomOp path is actually firing without sprinkling per-call
        // logs through every projection.
        static FIRST_DISPATCH_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        FIRST_DISPATCH_LOGGED.get_or_init(|| {
            tracing::info!(
                x_dims = ?x.dims(),
                x_dtype = ?x.dtype(),
                weight_dims = ?weight_t.dims(),
                weight_dtype = ?weight_t.dtype(),
                "VulkanLinearOp::linear_prefill_apply first dispatch"
            );
        });
        if !self.has_vulkan() || !self.linear_decode_enabled {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu) || !matches!(weight_t.device(), Device::Cpu) {
            return Ok(None);
        }
        let Ok((_batch, _seq_len, hidden_x)) = x.dims3() else {
            return Ok(None);
        };
        let Ok((hidden_w, out_dim)) = weight_t.dims2() else {
            return Ok(None);
        };
        if hidden_x != hidden_w {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?
            .clone();
        let out_dtype = x.dtype();
        let (weight_buffer, layout) = if self.use_bf16_packed_linear_weight(weight_t) {
            (
                self.cached_bf16_packed_weight_buffer(weight_t)?,
                crate::backend::vulkan_linear_op::WeightLayout::Bf16Packed,
            )
        } else if weight_t.dtype() == DType::F32 {
            (
                self.cached_f32_weight_buffer(weight_t)?,
                crate::backend::vulkan_linear_op::WeightLayout::F32,
            )
        } else {
            return Ok(None);
        };
        // Per-dispatch FLOP guard — only relevant for the F32 weight
        // path. The BF16-packed path chunks oversized dispatches
        // internally inside `VulkanLinearOp::cpu_fwd` (and the
        // transposed bwd does the same), so it's safe to dispatch any
        // shape against the BF16 op. The F32 path has no offset kernel
        // and would queue a single oversized submit, which is what
        // hard-hung the host twice — bail to caller's CPU
        // `broadcast_matmul` for that case.
        let row_count: usize = x
            .dims()
            .iter()
            .take(x.dims().len().saturating_sub(1))
            .product();
        if layout == crate::backend::vulkan_linear_op::WeightLayout::F32
            && crate::backend::vulkan_linear_op::dispatch_exceeds_safety_ceiling(
                row_count, hidden_x, out_dim,
            )
        {
            return Ok(None);
        }
        let op = crate::backend::vulkan_linear_op::build_op(
            vk_device,
            weight_buffer,
            weight_t.clone(),
            layout,
            hidden_x,
            out_dim,
            out_dtype,
        );
        let out = x.apply_op1(op).context("VulkanLinearOp apply_op1 failed")?;
        Ok(Some(out))
    }

    fn linear_prefill_apply_offset(
        &self,
        x: &Tensor,
        full_weight_t: &Tensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.linear_decode_enabled {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu) || !matches!(full_weight_t.device(), Device::Cpu) {
            return Ok(None);
        }
        // Only the bf16-packed kernel has an offset variant today; require
        // bf16 weights so the cached buffer matches the dispatch shader.
        if full_weight_t.dtype() != DType::BF16 {
            return Ok(None);
        }
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
        let weight_buffer = self.cached_bf16_packed_weight_buffer(full_weight_t)?;
        // Promote x to f32 for the kernel (kernel expects f32 input).
        let x_f32 = if x.dtype() == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        let dims = x_f32.shape().dims().to_vec();
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
        let sub_chunk_len = if crate::backend::vulkan_linear_op::dispatch_exceeds_safety_ceiling(
            row_count, hidden_x, chunk_len,
        ) {
            crate::backend::vulkan_linear_op::max_chunk_dim_for_flop(
                row_count.saturating_mul(hidden_x),
            )
            .min(chunk_len)
        } else {
            chunk_len
        };
        let out = if sub_chunk_len == chunk_len {
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset(
                vk_device.as_ref(),
                &dispatch_x,
                weight_buffer.as_ref(),
                row_count,
                hidden_x,
                chunk_len,
                chunk_start,
                full_out_dim,
            )
            .context("VulkanBackend: linear_prefill_apply_offset dispatch failed")?
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
            let mut sub_outputs: Vec<Tensor> = Vec::new();
            let mut sub_offset = 0usize;
            while sub_offset < chunk_len {
                let cur_len = (chunk_len - sub_offset).min(sub_chunk_len);
                let sub =
                    kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset(
                        vk_device.as_ref(),
                        &dispatch_x,
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
                sub_outputs.push(sub);
                sub_offset += cur_len;
            }
            Tensor::cat(&sub_outputs, 2).context("offset sub-chunk concat")?
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

    fn linear_decode_argmax(&self, x: &Tensor, weight_t: &Tensor) -> Result<Option<u32>> {
        if !self.has_vulkan() || !self.linear_decode_enabled || x.dtype() != DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu) || !matches!(weight_t.device(), Device::Cpu) {
            return Ok(None);
        }

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
        let token = if self.use_bf16_packed_linear_weight(weight_t) {
            let weight_buf = self.cached_bf16_packed_weight_buffer(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached_bf16_weights(
                vk_device,
                x,
                &weight_buf,
                hidden,
                out_dim,
            )
        } else {
            let weight_buf = self.cached_f32_weight_buffer(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached(
                vk_device,
                x,
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
        x: &Tensor,
        weight_t: &Tensor,
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
        if !self.supports_linear_decode_sample(top_k) || x.dtype() != DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu) || !matches!(weight_t.device(), Device::Cpu) {
            return Ok(None);
        }
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
        let packed_bf16 = self.use_bf16_packed_linear_weight(weight_t);
        let weight_buf = if packed_bf16 {
            self.cached_bf16_packed_weight_buffer(weight_t)?
        } else {
            self.cached_f32_weight_buffer(weight_t)?
        };
        let token = kiln_vulkan_kernel::kernels::dispatch_linear_decode_sample(
            vk_device,
            x,
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

    fn linear_decode_argmax_batch(
        &self,
        x: &Tensor,
        weight_t: &Tensor,
    ) -> Result<Option<Vec<u32>>> {
        if !self.has_vulkan()
            || !self.linear_decode_enabled
            || !self.linear_argmax_batch_enabled
            || x.dtype() != DType::F32
        {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu) || !matches!(weight_t.device(), Device::Cpu) {
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
        let tokens = if self.use_bf16_packed_linear_weight(weight_t) {
            let weight_buf = self.cached_bf16_packed_weight_buffer(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached_bf16_weights(
                vk_device,
                x,
                &weight_buf,
                batch,
                hidden,
                out_dim,
            )
        } else {
            let weight_buf = self.cached_f32_weight_buffer(weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached(
                vk_device,
                x,
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

        self.prewarm_linear_weight(
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
                    self.prewarm_full_attn_qkv_weights(
                        layer_idx,
                        &attn.q_proj_t,
                        &attn.k_proj_t,
                        &attn.v_proj_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_linear_weight(
                        &format!("layers.{layer_idx}.attention.o_proj_t"),
                        &attn.o_proj_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                }
                GpuAttentionWeights::Linear(attn) => {
                    self.prewarm_gdn_in_proj_weight(
                        &format!("layers.{layer_idx}.attention.in_proj_qkv_t"),
                        &attn.in_proj_qkv_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_gdn_in_proj_weight(
                        &format!("layers.{layer_idx}.attention.in_proj_z_t"),
                        &attn.in_proj_z_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_gdn_in_proj_weight(
                        &format!("layers.{layer_idx}.attention.in_proj_a_t"),
                        &attn.in_proj_a_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_gdn_in_proj_weight(
                        &format!("layers.{layer_idx}.attention.in_proj_b_t"),
                        &attn.in_proj_b_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                    self.prewarm_linear_weight(
                        &format!("layers.{layer_idx}.attention.out_proj_t"),
                        &attn.out_proj_t,
                        &mut count,
                        &mut bytes,
                        &mut bf16_packed_count,
                        &mut bf16_packed_bytes,
                    )?;
                }
            }

            self.prewarm_mlp_decode_weights(
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

    /// Phase 4.x residency: drop the candle CPU storage of every
    /// pre-transposed weight cache (`*_proj_t`, `embed_tokens_t`)
    /// whose BF16-packed bytes are already resident in
    /// [`Self::bf16_packed_weight_cache`]. Replace each with a
    /// 1-element BF16 stub and re-key the cache so subsequent
    /// lookups against the new TensorId still find the same
    /// `Arc<VulkanBuffer>`.
    ///
    /// Saves ~6-7 GB peak RSS on Qwen3.5-4B training at T=918 — the
    /// transposed-cache copies are the dominant remaining
    /// candle-side residency item documented in
    /// `docs/audits/candle_cpu_residency_2026-05-11.md`.
    ///
    /// Safe because:
    /// - The bf16-packed Vulkan code paths read the weight via the
    ///   `Arc<VulkanBuffer>` looked up in `bf16_packed_weight_cache`.
    ///   They never re-read the candle storage of the source tensor
    ///   after the buffer is cached.
    /// - `VulkanLinearOp::bwd` for BF16 weights routes through the
    ///   transposed Vulkan kernel (also buffer-backed). The F32
    ///   fallback bwd path that *does* read `self.weight_t` cannot
    ///   fire for BF16 weights.
    /// - Non-BF16 tensors and tensors not in the cache are skipped.
    fn drop_uploaded_bf16_weights(
        &self,
        weights: &mut crate::forward::GpuWeights,
        device: &Device,
    ) -> Result<usize> {
        if !self.has_vulkan() {
            return Ok(0);
        }
        // Broadcast-base for cheap shape-preserving stubs. Source has
        // 2 bytes of storage; broadcast_as(target_shape) creates views
        // with stride [0, 0] sharing the same Arc<Storage>. Each per-
        // weight stub costs ~24 bytes of metadata (Layout + Tensor
        // struct), not `hidden * out_dim * 2` bytes.
        let broadcast_base = Tensor::zeros((1usize, 1usize), DType::BF16, device)
            .context("drop_uploaded_bf16_weights: create broadcast base")?;
        let mut cache = self
            .bf16_packed_weight_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("bf16 weight cache mutex poisoned"))?;

        // Per-tensor replacement closure. Returns true if the tensor
        // was stubbed (was BF16, rank-2, and in the cache).
        //
        // - Reads the original `[hidden, out_dim]` shape from `t.dims()`
        //   *before* replacement.
        // - Creates a shape-preserving stub by broadcasting the
        //   2-byte base to that shape (so downstream `weight_t.dims2()`
        //   reads continue to return the right shape, but the storage
        //   bytes drop to ~zero).
        // - Re-keys the cache so subsequent
        //   `cached_bf16_packed_weight_buffer(weight_t)` lookups by the
        //   new TensorId still find the original `Arc<VulkanBuffer>`.
        fn replace(
            t: &mut Tensor,
            cache: &mut std::collections::HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>,
            broadcast_base: &Tensor,
        ) -> bool {
            if t.dtype() != DType::BF16 {
                return false;
            }
            let dims = t.dims();
            if dims.len() != 2 {
                return false; // Only rank-2 transposed-cache tensors are stubbable.
            }
            let old_id = t.id();
            let Some(buf) = cache.remove(&old_id) else {
                return false;
            };
            let Ok(new_stub) = broadcast_base.broadcast_as((dims[0], dims[1])) else {
                cache.insert(old_id, buf); // restore on failure
                return false;
            };
            let new_id = new_stub.id();
            *t = new_stub;
            cache.insert(new_id, buf);
            true
        }

        let mut stubbed = 0usize;

        // Intentionally NOT stubbing `weights.embed_tokens_t`:
        // `embedding_lookup_from_transposed_index` calls
        // `embed_tokens_t.index_select(idx, 1)` which reads the
        // tensor's data (not just shape), so a 1-element stub would
        // make the embedding lookup return garbage. The other `*_proj_t`
        // caches go through `cached_bf16_packed_weight_buffer` (TensorId
        // → Arc<VulkanBuffer>) so they only need shape/dtype metadata
        // on the candle side. Embedding savings (~750 MB) are small
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
                        if replace(t, &mut cache, &broadcast_base) {
                            stubbed += 1;
                        }
                    }
                    if let Some(qkv_t) = attn.qkv_proj_t.as_mut() {
                        if replace(qkv_t, &mut cache, &broadcast_base) {
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
                        if replace(t, &mut cache, &broadcast_base) {
                            stubbed += 1;
                        }
                    }
                    if let Some(ab_t) = attn.in_proj_ab_t.as_mut() {
                        if replace(ab_t, &mut cache, &broadcast_base) {
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
                if replace(t, &mut cache, &broadcast_base) {
                    stubbed += 1;
                }
            }
        }

        tracing::info!(
            stubbed,
            "dropped candle CPU storage of pre-transposed bf16 weight caches"
        );
        Ok(stubbed)
    }

    fn full_attn_qkv_decode(
        &self,
        x: &Tensor,
        q_weight_t: &Tensor,
        k_weight_t: &Tensor,
        v_weight_t: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor)>> {
        if !self.has_vulkan() || !self.full_attn_qkv_enabled || x.dtype() != DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu)
            || !matches!(q_weight_t.device(), Device::Cpu)
            || !matches!(k_weight_t.device(), Device::Cpu)
            || !matches!(v_weight_t.device(), Device::Cpu)
        {
            return Ok(None);
        }

        let Ok((batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        // Multi-token (prefill-ish) shapes still go through the unfused
        // path: this kernel family is the single-token decode projection.
        // Batched single-token decode IS supported now via the `_batched`
        // dispatch — collapsing seq_len==1 across an arbitrary batch dim
        // into a single fused submit was the explicit scaling fix.
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
        let bf16 = self.use_bf16_packed_full_attn_qkv_weights(&[q_weight_t, k_weight_t, v_weight_t]);
        let out = if batch == 1 {
            if bf16 {
                let q_buf = self.cached_bf16_packed_weight_buffer(q_weight_t)?;
                let k_buf = self.cached_bf16_packed_weight_buffer(k_weight_t)?;
                let v_buf = self.cached_bf16_packed_weight_buffer(v_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_bf16_weights(
                    vk_device, x, &q_buf, &k_buf, &v_buf, hidden, q_dim, k_dim, v_dim,
                )
            } else {
                let q_buf = self.cached_f32_weight_buffer(q_weight_t)?;
                let k_buf = self.cached_f32_weight_buffer(k_weight_t)?;
                let v_buf = self.cached_f32_weight_buffer(v_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached(
                    vk_device, x, &q_buf, &k_buf, &v_buf, hidden, q_dim, k_dim, v_dim,
                )
            }
            .context("full_attn_qkv_decode kernel failed")?
        } else if bf16 {
            let q_buf = self.cached_bf16_packed_weight_buffer(q_weight_t)?;
            let k_buf = self.cached_bf16_packed_weight_buffer(k_weight_t)?;
            let v_buf = self.cached_bf16_packed_weight_buffer(v_weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_batched_bf16_weights(
                vk_device, x, &q_buf, &k_buf, &v_buf, batch, hidden, q_dim, k_dim, v_dim,
            )
            .context("full_attn_qkv_decode_batched_bf16w kernel failed")?
        } else {
            let q_buf = self.cached_f32_weight_buffer(q_weight_t)?;
            let k_buf = self.cached_f32_weight_buffer(k_weight_t)?;
            let v_buf = self.cached_f32_weight_buffer(v_weight_t)?;
            kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached_batched(
                vk_device, x, &q_buf, &k_buf, &v_buf, batch, hidden, q_dim, k_dim, v_dim,
            )
            .context("full_attn_qkv_decode_batched kernel failed")?
        };
        Ok(Some(out))
    }

    fn mlp_gate_up_decode(
        &self,
        x: &Tensor,
        gate_weight_t: &Tensor,
        up_weight_t: &Tensor,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.mlp_gate_up_enabled || x.dtype() != DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu)
            || !matches!(gate_weight_t.device(), Device::Cpu)
            || !matches!(up_weight_t.device(), Device::Cpu)
        {
            return Ok(None);
        }

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
        let gate_buf = self.cached_f32_weight_buffer(gate_weight_t)?;
        let up_buf = self.cached_f32_weight_buffer(up_weight_t)?;
        let row_count = batch * seq_len;
        let dispatch_x = if seq_len == 1 {
            x.clone()
        } else {
            x.reshape((row_count, 1usize, hidden))?
        };
        let out = kiln_vulkan_kernel::kernels::dispatch_mlp_gate_up_decode_cached(
            vk_device,
            &dispatch_x,
            &gate_buf,
            &up_buf,
            hidden,
            intermediate,
        )
        .context("mlp_gate_up_decode kernel failed")?;
        let out = if seq_len == 1 {
            out
        } else {
            out.reshape((batch, seq_len, intermediate))?
        };
        Ok(Some(out))
    }

    fn mlp_decode(
        &self,
        x: &Tensor,
        gate_weight_t: &Tensor,
        up_weight_t: &Tensor,
        down_weight_t: &Tensor,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.mlp_decode_enabled || x.dtype() != DType::F32 {
            return Ok(None);
        }
        if !matches!(x.device(), Device::Cpu)
            || !matches!(gate_weight_t.device(), Device::Cpu)
            || !matches!(up_weight_t.device(), Device::Cpu)
            || !matches!(down_weight_t.device(), Device::Cpu)
        {
            return Ok(None);
        }

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
        let dispatch_x = if seq_len == 1 {
            x.clone()
        } else {
            x.reshape((row_count, 1usize, hidden))?
        };
        let use_bf16_mlp_weights =
            self.use_bf16_packed_mlp_decode_weights(&[gate_weight_t, up_weight_t, down_weight_t]);
        let out =
            if row_count >= 8 && self.mlp_bf16_gate_up_f32_down_enabled && use_bf16_mlp_weights {
                let gate_buf = self.cached_bf16_packed_weight_buffer(gate_weight_t)?;
                let up_buf = self.cached_bf16_packed_weight_buffer(up_weight_t)?;
                let down_buf = self.cached_f32_weight_buffer(down_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_gate_up_f32_down(
                    vk_device,
                    &dispatch_x,
                    &gate_buf,
                    &up_buf,
                    &down_buf,
                    hidden,
                    intermediate,
                    out_dim,
                )
            } else if use_bf16_mlp_weights {
                let gate_buf = self.cached_bf16_packed_weight_buffer(gate_weight_t)?;
                let up_buf = self.cached_bf16_packed_weight_buffer(up_weight_t)?;
                let down_buf = self.cached_bf16_packed_weight_buffer(down_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached_bf16_weights(
                    vk_device,
                    &dispatch_x,
                    &gate_buf,
                    &up_buf,
                    &down_buf,
                    hidden,
                    intermediate,
                    out_dim,
                )
            } else {
                let gate_buf = self.cached_f32_weight_buffer(gate_weight_t)?;
                let up_buf = self.cached_f32_weight_buffer(up_weight_t)?;
                let down_buf = self.cached_f32_weight_buffer(down_weight_t)?;
                kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached(
                    vk_device,
                    &dispatch_x,
                    &gate_buf,
                    &up_buf,
                    &down_buf,
                    hidden,
                    intermediate,
                    out_dim,
                )
            }
            .context("mlp_decode kernel failed")?;
        let out = if seq_len == 1 {
            out
        } else {
            out.reshape((batch, seq_len, out_dim))?
        };
        Ok(Some(out))
    }

    fn gdn_forward_substitution(
        &self,
        a_strict: &Tensor,
        v_prime: &Tensor,
        beta: &Tensor,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if a_strict.dtype() != DType::BF16 {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let out = kiln_vulkan_kernel::kernels::dispatch_gdn_forward_substitution(
            vk_device, a_strict, v_prime, beta,
        )
        .context("gdn_forward_substitution kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_recurrent_prefill_native_head_last(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan()
            || !self.gdn_recurrent_unexpanded_qk_enabled
            || !matches!(q.dtype(), DType::BF16 | DType::F32)
        {
            return Ok(None);
        }
        if !matches!(q.device(), Device::Cpu)
            || !matches!(k.device(), Device::Cpu)
            || !matches!(v.device(), Device::Cpu)
            || !matches!(beta.device(), Device::Cpu)
            || !matches!(g.device(), Device::Cpu)
            || !matches!(state.device(), Device::Cpu)
        {
            return Ok(None);
        }
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
        let Ok((state_batch, state_heads, state_dk, state_dv)) = state.dims4() else {
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
            && state.dtype() == q.dtype()
        {
            let state_id = state.id();
            let resident_state =
                RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow().get(&state_id).cloned());
            let (out, resident_state) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_native_head_last_resident_state(
                    vk_device,
                    q,
                    k,
                    v,
                    beta,
                    g,
                    state,
                    resident_state,
                )
                .context("gdn_recurrent_step native-head resident-state Vulkan kernel failed")?;
            RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
                cache.borrow_mut().insert(state_id, resident_state);
            });
            return Ok(Some(out));
        }
        let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
        let (out, new_state) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_native_head_last_with_options(
                vk_device,
                q,
                k,
                v,
                beta,
                g,
                state,
                skip_state_readback,
            )
            .context("gdn_recurrent_step native-head Vulkan kernel failed")?;
        if let Some(new_state) = new_state {
            *state = new_state;
        }
        Ok(Some(out))
    }

    fn gdn_recurrent_qk_norm_prefill_native_head_last(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Tensor,
        q_scale: f64,
        qk_eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan()
            || !self.gdn_recurrent_qk_norm_unexpanded_enabled
            || !matches!(q.dtype(), DType::F32 | DType::BF16)
        {
            return Ok(None);
        }
        if !matches!(q.device(), Device::Cpu)
            || !matches!(k.device(), Device::Cpu)
            || !matches!(v.device(), Device::Cpu)
            || !matches!(beta.device(), Device::Cpu)
            || !matches!(g.device(), Device::Cpu)
            || !matches!(state.device(), Device::Cpu)
        {
            return Ok(None);
        }
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
        let (out, new_state) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_qk_norm_step_native_head_last_with_options(
                vk_device,
                q,
                k,
                v,
                beta,
                g,
                state,
                skip_state_readback,
            )
            .context("gdn_recurrent_qk_norm native-head Vulkan kernel failed")?;
        if let Some(new_state) = new_state {
            *state = new_state;
        }
        Ok(Some(out))
    }

    fn gdn_recurrent_step(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        beta: &Tensor,
        g: &Tensor,
        state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if !matches!(q.dtype(), DType::BF16 | DType::F32) {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        if self.recurrent_state_residency_enabled && recurrent_state_resident_scope_active() {
            let state_id = state.id();
            let resident_state =
                RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow().get(&state_id).cloned());

            let (out, resident_state) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_resident_state(
                    vk_device,
                    q,
                    k,
                    v,
                    beta,
                    g,
                    state,
                    resident_state,
                )
                .context("gdn_recurrent_step resident-state kernel failed")?;

            RECURRENT_STATE_RESIDENT_CACHE.with(|cache| {
                cache.borrow_mut().insert(state_id, resident_state);
            });
            return Ok(Some(out));
        }

        let skip_state_readback = crate::forward::vulkan_skip_gdn_state_readback_active();
        let (out, new_state) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_with_options(
                vk_device,
                q,
                k,
                v,
                beta,
                g,
                state,
                skip_state_readback,
            )
            .context("gdn_recurrent_step kernel failed")?;
        if let Some(new_state) = new_state {
            *state = new_state;
        }
        Ok(Some(out))
    }

    fn gdn_chunk_prep(
        &self,
        g: &Tensor,
        v: &Tensor,
        kkt: &Tensor,
        qkt: &Tensor,
        ks_entry: &Tensor,
        q_s: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)>> {
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if g.dtype() != DType::BF16 {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let result = kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_prep(
            vk_device, g, v, kkt, qkt, ks_entry, q_s,
        )
        .context("gdn_chunk_prep kernel failed")?;
        Ok(Some(result))
    }

    fn gdn_chunk_scan(
        &self,
        a_strict: &Tensor,
        b_mask: &Tensor,
        v_prime: &Tensor,
        q_s_scaled: &Tensor,
        beta: &Tensor,
        decay_last_col: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if a_strict.dtype() != DType::BF16 {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let result = kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_scan(
            vk_device,
            a_strict,
            b_mask,
            v_prime,
            q_s_scaled,
            beta,
            decay_last_col,
        )
        .context("gdn_chunk_scan kernel failed")?;
        Ok(Some(result))
    }

    fn gdn_full_chunk_forward(
        &self,
        g: &Tensor,
        v: &Tensor,
        kkt: &Tensor,
        qkt: &Tensor,
        ks_entry: &Tensor,
        q_s: &Tensor,
        beta: &Tensor,
        k_t: &Tensor,
        state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.gdn_enabled {
            return Ok(None);
        }
        if g.dtype() != DType::BF16 {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (out, new_state) = kiln_vulkan_kernel::kernels::dispatch_gdn_full_chunk_forward(
            vk_device, g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
        )
        .context("gdn_full_chunk_forward kernel failed")?;
        *state = new_state;
        Ok(Some(out))
    }

    fn gdn_gates(
        &self,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if !self.has_vulkan() || !self.gdn_gates_enabled {
            return Ok(None);
        }
        if !matches!(a.dtype(), DType::BF16 | DType::F32) {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let nv = a_log.elem_count();
        if dt_bias.elem_count() != nv {
            return Ok(None);
        }
        let a_log_buf = self.cached_f32_weight_buffer(a_log)?;
        let dt_bias_buf = self.cached_f32_weight_buffer(dt_bias)?;

        // Output shape matches input shape [B, T, nv]
        let out_shape = a.dims().as_ref().to_vec();
        let (beta, g) = kiln_vulkan_kernel::kernels::dispatch_gdn_gates_cached(
            vk_device,
            a,
            b,
            &a_log_buf,
            &dt_bias_buf,
            nv,
            &out_shape,
        )
        .context("gdn_gates kernel failed")?;
        Ok(Some((beta, g)))
    }

    fn gdn_gated_rms_norm(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.gdn_gated_rms_norm_enabled {
            return Ok(None);
        }
        if !matches!(x.dtype(), DType::BF16 | DType::F32) {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
        let hidden = weight.elem_count();
        if hidden == 0 || x.elem_count() % hidden != 0 {
            return Ok(None);
        }
        let weight_buf = self.cached_f32_weight_buffer(weight)?;

        // Output shape matches x shape
        let out_shape = x.dims().as_ref().to_vec();
        let out = kiln_vulkan_kernel::kernels::dispatch_gdn_gated_rms_norm_cached(
            vk_device,
            x,
            z,
            &weight_buf,
            hidden,
            eps as f32,
            &out_shape,
        )
        .context("gdn_gated_rms_norm kernel failed")?;
        Ok(Some(out))
    }

    fn causal_conv1d_update(
        &self,
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.fused_conv1d_update_enabled {
            return Ok(None);
        }
        if !matches!(x.dtype(), DType::BF16 | DType::F32) {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (out, new_state) = kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_update(
            vk_device,
            x,
            weight,
            conv_state,
            kernel_size,
        )
        .context("causal_conv1d_update kernel failed")?;
        *conv_state = new_state;
        Ok(Some(out))
    }

    fn causal_conv1d_prefill(
        &self,
        x: &Tensor,
        weight: &Tensor,
        conv_state: &mut Tensor,
        kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        if !self.has_vulkan() || !self.fused_conv1d_prefill_enabled {
            return Ok(None);
        }
        if !matches!(x.dtype(), DType::BF16 | DType::F32) {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (out, new_state) = if self.conv1d_prefill_single_submit_enabled {
            let weight_buf = self.cached_f32_weight_buffer(weight)?;
            kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill_cached_weight(
                vk_device,
                x,
                &weight_buf,
                conv_state,
                kernel_size,
            )
            .context("causal_conv1d_prefill cached-weight single-submit kernel failed")?
        } else {
            kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill(
                vk_device,
                x,
                weight,
                conv_state,
                kernel_size,
            )
            .context("causal_conv1d_prefill kernel failed")?
        };
        *conv_state = new_state;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendRuntime;
    use candle_core::Device;

    /// Round-trip test for the Phase 3.1 hooks. Registers a fresh
    /// activation, asserts `has_resident_activation` flips true,
    /// evicts it, asserts it flips back. Skipped if no Vulkan
    /// device — the hooks have no-op defaults so a CPU-only run
    /// would just always answer false.
    /// `update_resident_activation` must overwrite the registry
    /// buffer with the tensor's current bytes — the SGD path relies
    /// on this to keep `lora_delta_resident` reading current weights.
    /// Verifies BF16-packed encoding round-trips correctly through
    /// the update path too.
    #[test]
    fn update_resident_activation_overwrites_buffer() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        // Use a BF16 tensor — that's the LoRA Var case the production
        // path exercises. The update path's encoding choice depends
        // on dtype, so testing BF16 specifically (not just F32)
        // guards against regression in the dtype branch.
        let initial = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?
            .to_dtype(DType::BF16)?;
        backend.register_resident_activation(&initial)?;
        // Sanity: registered with initial values.
        let resolved = backend
            .resolve_resident_activation(&initial, &[2, 2], DType::BF16)?
            .expect("must resolve right after register");
        let init_v: Vec<f32> = resolved
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(init_v, vec![1.0, 2.0, 3.0, 4.0]);

        // Mutate the tensor's storage out-of-band — analogous to what
        // candle Var::set does. Use to_dtype roundtrip + a fresh tensor
        // since we can't mutate in place. The TensorId stays the same
        // because we update the same Var-equivalent reference.
        // Workaround: create a NEW tensor with the same TensorId by
        // using `.copy()` semantics — actually candle doesn't expose
        // that. So instead simulate the post-SGD state by registering
        // a different tensor (with a different id) and verify the
        // update-via-the-original-reference path still works on the
        // ORIGINAL id.
        //
        // Concretely: hand `update_resident_activation` a tensor whose
        // BYTES differ from what's in the buffer but whose .id() is
        // the original. We can do that via `Var::set`-like:
        // use the original Tensor object (.id() unchanged) and
        // overwrite its underlying storage by re-running update with
        // a tensor that has different DATA but the same shape. Since
        // update keys on tensor.id(), we have to use a Var to keep
        // the id stable across a content change.
        let v = candle_core::Var::from_tensor(&initial)?;
        let new_data = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (2, 2), &Device::Cpu)?
            .to_dtype(DType::BF16)?;
        v.set(&new_data)?;
        // v.as_tensor() now wraps the same TensorId as the original
        // Var construction — but Var wraps a Tensor that has its own
        // id, distinct from `initial`. So this test path actually
        // demonstrates that the update applies to whatever id we hand
        // it, not to the unchanged `initial`.
        //
        // Register v.as_tensor() and update it with newer data.
        backend.register_resident_activation(v.as_tensor())?;
        // Build "newer" data (v already holds new_data; resolve and
        // confirm the registry sees IT, not initial).
        let resolved_v = backend
            .resolve_resident_activation(v.as_tensor(), &[2, 2], DType::BF16)?
            .expect("v must resolve after register");
        let v_init_v: Vec<f32> = resolved_v
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(v_init_v, vec![10.0, 20.0, 30.0, 40.0]);

        // Now mutate v further and call update.
        let newer_data =
            Tensor::from_vec(vec![100.0f32, 200.0, 300.0, 400.0], (2, 2), &Device::Cpu)?
                .to_dtype(DType::BF16)?;
        v.set(&newer_data)?;
        backend.update_resident_activation(v.as_tensor())?;
        let resolved_after = backend
            .resolve_resident_activation(v.as_tensor(), &[2, 2], DType::BF16)?
            .expect("v must resolve after update");
        let after_v: Vec<f32> = resolved_after
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(after_v, vec![100.0, 200.0, 300.0, 400.0]);

        backend.evict_resident_activation(&initial);
        backend.evict_resident_activation(v.as_tensor());
        Ok(())
    }

    /// End-to-end Phase 4.1 chain: register A and B → call
    /// `lora_delta_resident` → mutate A via `Var::set` → call
    /// `update_resident_activation` → call `lora_delta_resident`
    /// again → second result must reflect the new A.
    ///
    /// This is the contract `sgd_step + update_resident_activation`
    /// relies on: the next forward inference pass after SGD must see
    /// the updated weights.
    #[test]
    fn lora_delta_resident_reflects_post_update_weights() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let in_features = 8usize;
        let rank = 4usize;
        let out_features = 6usize;
        let scale = 1.0f32;

        let x_data: Vec<f32> = (0..in_features).map(|i| (i as f32) * 0.1).collect();
        let a_init: Vec<f32> = (0..rank * in_features).map(|i| (i as f32) * 0.01).collect();
        let b_init: Vec<f32> = (0..out_features * rank)
            .map(|i| (i as f32) * 0.02)
            .collect();

        let x =
            Tensor::from_vec(x_data, (1, 1, in_features), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let a_var = candle_core::Var::from_tensor(
            &Tensor::from_vec(a_init, (rank, in_features), &Device::Cpu)?.to_dtype(DType::BF16)?,
        )?;
        let b_var = candle_core::Var::from_tensor(
            &Tensor::from_vec(b_init, (out_features, rank), &Device::Cpu)?.to_dtype(DType::BF16)?,
        )?;

        backend.register_resident_activation(a_var.as_tensor())?;
        backend.register_resident_activation(b_var.as_tensor())?;

        // First forward: gets the init delta.
        let delta_init = backend
            .lora_delta_resident(&x, a_var.as_tensor(), b_var.as_tensor(), scale)?
            .expect("must dispatch on-device when registered");

        // Mutate A — simulate what sgd_step does. New A bytes are
        // intentionally far from the init values so the resulting
        // delta will be visibly different.
        let a_post: Vec<f32> = (0..rank * in_features)
            .map(|i| 5.0 - (i as f32) * 0.05)
            .collect();
        let a_post_tensor =
            Tensor::from_vec(a_post, (rank, in_features), &Device::Cpu)?.to_dtype(DType::BF16)?;
        a_var.set(&a_post_tensor)?;
        // Critical: keep the registry in sync.
        backend.update_resident_activation(a_var.as_tensor())?;

        // Second forward: must use the new A bytes.
        let delta_post = backend
            .lora_delta_resident(&x, a_var.as_tensor(), b_var.as_tensor(), scale)?
            .expect("must dispatch on-device when registered");

        // The two deltas must differ — if update_resident_activation
        // were a no-op or used the wrong encoding, delta_post would
        // equal delta_init.
        let init_v: Vec<f32> = delta_init
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let post_v: Vec<f32> = delta_post
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(init_v.len(), post_v.len());
        let max_diff = init_v
            .iter()
            .zip(post_v.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.1,
            "delta should differ noticeably after A update; max_diff={max_diff}, \
             init={init_v:?}, post={post_v:?}"
        );

        // Compare delta_post against a CPU reference computed with
        // the new A bytes — they should match to bf16 precision.
        let a_post_round = a_var.as_tensor().to_dtype(DType::F32)?;
        let b_round = b_var.as_tensor().to_dtype(DType::F32)?;
        let x_f32 = x.to_dtype(DType::F32)?;
        let hidden = x_f32.broadcast_matmul(&a_post_round.t()?)?;
        let cpu_delta_post = hidden
            .broadcast_matmul(&b_round.t()?)?
            .to_dtype(DType::BF16)?;
        let cpu_post_v: Vec<f32> = cpu_delta_post
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, (vk, cpu)) in post_v.iter().zip(cpu_post_v.iter()).enumerate() {
            let abs = (vk - cpu).abs();
            let rel = abs / cpu.abs().max(1e-3);
            assert!(
                abs < 5e-2 || rel < 5e-2,
                "idx {i}: vk={vk:.6} cpu={cpu:.6} abs={abs:e} rel={rel:e}"
            );
        }

        backend.evict_resident_activation(a_var.as_tensor());
        backend.evict_resident_activation(b_var.as_tensor());
        Ok(())
    }

    /// `update_resident_activation` is a no-op when the tensor isn't
    /// registered — avoids surprising errors when caller is
    /// dtype-agnostic (e.g. a sgd_step that fires for both
    /// registered LoRA Vars and unregistered legacy Vars).
    #[test]
    fn update_resident_activation_noop_when_not_registered() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let t = Tensor::from_vec(vec![1.0f32; 4], (4,), &Device::Cpu)?;
        // Not registered — must not error.
        backend.update_resident_activation(&t)?;
        assert!(!backend.has_resident_activation(&t));
        Ok(())
    }

    /// Re-registration after eviction must work — the trainer's
    /// per-step lifecycle relies on this (training step N evicts
    /// boundaries, step N+1 re-registers fresh ones with new
    /// TensorIds, but conceptually the same lifecycle).
    #[test]
    fn resident_activation_re_register_after_evict() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?;
        backend.register_resident_activation(&t)?;
        assert!(backend.has_resident_activation(&t));
        backend.evict_resident_activation(&t);
        assert!(!backend.has_resident_activation(&t));
        // Re-register with the same TensorId — must succeed and
        // re-upload (the previous buffer was dropped at eviction).
        backend.register_resident_activation(&t)?;
        assert!(
            backend.has_resident_activation(&t),
            "tensor must be registered again after eviction"
        );
        // Resolve to confirm the bytes round-tripped correctly the
        // second time too.
        let resolved = backend
            .resolve_resident_activation(&t, &[2, 2], DType::F32)?
            .expect("must resolve after re-register");
        let data: Vec<f32> = resolved.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        backend.evict_resident_activation(&t);
        Ok(())
    }

    /// Empty-tensor (zero-byte) input must not panic the Vulkan
    /// allocator. Bails silently — `has_resident_activation` returns
    /// false and the caller falls through to its CPU path.
    #[test]
    fn register_resident_activation_handles_empty_tensor() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let empty: Tensor = Tensor::from_vec(Vec::<f32>::new(), (0,), &Device::Cpu)?;
        backend.register_resident_activation(&empty)?;
        assert!(
            !backend.has_resident_activation(&empty),
            "empty tensor must not be registered (zero-size driver issue)"
        );
        Ok(())
    }

    /// resolve_resident_activation must reconstruct a Tensor whose
    /// data matches the originally-registered tensor's bytes.
    /// Returns Ok(None) when the tensor isn't in the registry.
    #[test]
    fn resolve_resident_activation_round_trip() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let original_data = vec![1.5f32, -2.5, 3.25, -4.75];
        let t = Tensor::from_vec(original_data.clone(), (2, 2), &Device::Cpu)?;

        // Not registered yet → resolve returns None.
        let unresolved = backend.resolve_resident_activation(&t, &[2, 2], DType::F32)?;
        assert!(unresolved.is_none(), "unregistered tensor must not resolve");

        backend.register_resident_activation(&t)?;
        let resolved = backend
            .resolve_resident_activation(&t, &[2, 2], DType::F32)?
            .expect("must resolve once registered");
        assert_eq!(resolved.dims(), &[2, 2]);
        let resolved_data: Vec<f32> = resolved.flatten_all()?.to_vec1::<f32>()?;
        for (i, (got, want)) in resolved_data.iter().zip(original_data.iter()).enumerate() {
            assert!((got - want).abs() < 1e-9, "idx {i}: got {got} want {want}");
        }

        backend.evict_resident_activation(&t);
        // After eviction → resolve returns None again.
        let unresolved = backend.resolve_resident_activation(&t, &[2, 2], DType::F32)?;
        assert!(unresolved.is_none());
        Ok(())
    }

    /// dispatch_sgd_step against two registry-resident F32 tensors —
    /// param := param - lr * grad, computed on-device, must match the
    /// CPU reference to f32 precision.
    #[test]
    fn dispatch_sgd_step_resident_round_trip() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
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

        let param = Tensor::from_vec(param_data, (n,), &Device::Cpu)?;
        let grad = Tensor::from_vec(grad_data, (n,), &Device::Cpu)?;

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
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = Tensor::from_vec(vec![1.0f32; 4], (4,), &Device::Cpu)?;
        let g = Tensor::from_vec(vec![0.5f32; 4], (4,), &Device::Cpu)?;
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
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = Tensor::from_vec(vec![1.0f32; 4], (4,), &Device::Cpu)?;
        let g = Tensor::from_vec(vec![0.5f32; 8], (8,), &Device::Cpu)?;
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

    /// Vulkan lora_delta_resident must match the candle CPU
    /// `compute_lora_delta` (i.e. `(x @ A.T @ B.T) * scale`) to bf16
    /// numerics tolerance when A and B are registered.
    #[test]
    fn lora_delta_resident_matches_cpu_reference() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        // Small LoRA-shape: rank=4, in=8, out=6.
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

        let x = Tensor::from_vec(x_data, (1, t, in_features), &Device::Cpu)?;
        let a_f32 = Tensor::from_vec(a_data, (rank, in_features), &Device::Cpu)?;
        let b_f32 = Tensor::from_vec(b_data, (out_features, rank), &Device::Cpu)?;
        let a_bf16 = a_f32.to_dtype(DType::BF16)?;
        let b_bf16 = b_f32.to_dtype(DType::BF16)?;
        let x_bf16 = x.to_dtype(DType::BF16)?;

        // CPU baseline (manual, F32) — `compute_lora_delta` casts to
        // x.dtype() which would be BF16 here, but candle CPU doesn't
        // support BF16 matmul. The math we want to validate is
        // identical: (x @ A.T @ B.T) * scale, computed against the
        // same BF16-quantised A and B that the Vulkan path reads
        // from the registry (we round-trip through bf16 to match
        // the bytes the kernel sees).
        let a_round = a_bf16.to_dtype(DType::F32)?;
        let b_round = b_bf16.to_dtype(DType::F32)?;
        let hidden_cpu = x.broadcast_matmul(&a_round.t()?)?;
        let delta_cpu = hidden_cpu.broadcast_matmul(&b_round.t()?)?;
        let cpu_delta = (delta_cpu * scale as f64)?.to_dtype(DType::BF16)?;

        // Register A and B in the registry.
        backend.register_resident_activation(&a_bf16)?;
        backend.register_resident_activation(&b_bf16)?;

        // Vulkan path.
        let vk_delta = backend
            .lora_delta_resident(&x_bf16, &a_bf16, &b_bf16, scale)?
            .expect("lora_delta_resident must succeed when A and B are registered");

        assert_eq!(vk_delta.dims(), cpu_delta.dims());
        assert_eq!(vk_delta.dtype(), cpu_delta.dtype());
        let cpu_v: Vec<f32> = cpu_delta
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let vk_v: Vec<f32> = vk_delta
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        for (i, (c, v)) in cpu_v.iter().zip(vk_v.iter()).enumerate() {
            let abs = (c - v).abs();
            let rel = abs / c.abs().max(1e-3);
            assert!(
                abs < 5e-2 || rel < 5e-2,
                "idx {i}: cpu={c:.6} vk={v:.6} abs={abs:e} rel={rel:e}"
            );
        }

        backend.evict_resident_activation(&a_bf16);
        backend.evict_resident_activation(&b_bf16);
        Ok(())
    }

    /// lora_delta_resident must return Ok(None) when A or B is not
    /// registered — caller falls back to candle CPU.
    #[test]
    fn lora_delta_resident_falls_back_when_not_resident() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let x =
            Tensor::from_vec(vec![0.0f32; 16], (1, 2, 8), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let a = Tensor::from_vec(vec![0.0f32; 32], (4, 8), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let b = Tensor::from_vec(vec![0.0f32; 24], (6, 4), &Device::Cpu)?.to_dtype(DType::BF16)?;
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
    /// that lets LoRA Vars (BF16 by convention) update on-device
    /// without the candle CPU re-upload round-trip.
    #[test]
    fn dispatch_sgd_step_bf16_resident_round_trip() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
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

        let p_f32 = Tensor::from_vec(p_data, (n,), &Device::Cpu)?;
        let g_f32 = Tensor::from_vec(g_data, (n,), &Device::Cpu)?;
        let p_bf16 = p_f32.to_dtype(DType::BF16)?;
        let g_bf16 = g_f32.to_dtype(DType::BF16)?;

        backend.register_resident_activation(&p_bf16)?;
        backend.register_resident_activation(&g_bf16)?;

        let dispatched = backend.dispatch_sgd_step(&p_bf16, &g_bf16, lr)?;
        assert!(
            dispatched,
            "BF16 dispatch_sgd_step must succeed when both operands are resident"
        );

        // Read the updated param buffer back via resolve.
        let resolved = backend
            .resolve_resident_activation(&p_bf16, &[n], DType::BF16)?
            .expect("must resolve");
        let updated_v: Vec<f32> = resolved
            .to_dtype(DType::F32)?
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
        let backend = VulkanBackend::new(Device::Cpu);
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

        let param = Tensor::from_vec(p_data, (n,), &Device::Cpu)?;
        let grad = Tensor::from_vec(g_data, (n,), &Device::Cpu)?;
        let m = Tensor::from_vec(m_data, (n,), &Device::Cpu)?;
        let v = Tensor::from_vec(v_data, (n,), &Device::Cpu)?;

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
            .resolve_resident_activation(&param, &[n], DType::F32)?
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
        let backend = VulkanBackend::new(Device::Cpu);
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

        let p_f32 = Tensor::from_vec(p_data, (n,), &Device::Cpu)?;
        let g_f32 = Tensor::from_vec(g_data, (n,), &Device::Cpu)?;
        let m_f32 = Tensor::from_vec(vec![0.0f32; n], (n,), &Device::Cpu)?;
        let v_f32 = Tensor::from_vec(vec![0.0f32; n], (n,), &Device::Cpu)?;
        let p_bf16 = p_f32.to_dtype(DType::BF16)?;
        let g_bf16 = g_f32.to_dtype(DType::BF16)?;
        let m_bf16 = m_f32.to_dtype(DType::BF16)?;
        let v_bf16 = v_f32.to_dtype(DType::BF16)?;

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
            .resolve_resident_activation(&p_bf16, &[n], DType::BF16)?
            .expect("param must resolve");
        let got: Vec<f32> = resolved
            .to_dtype(DType::F32)?
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
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = Tensor::from_vec(vec![1.0f32; 4], (4,), &Device::Cpu)?;
        let g = Tensor::from_vec(vec![0.5f32; 4], (4,), &Device::Cpu)?;
        let m = Tensor::from_vec(vec![0.0f32; 4], (4,), &Device::Cpu)?;
        let v = Tensor::from_vec(vec![0.0f32; 4], (4,), &Device::Cpu)?;
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

    /// Lazy candle-storage sync end-to-end. Register a `Var`, run an
    /// on-device SGD step against its registry buffer (which the
    /// trainer now does *without* calling `var.set`), then verify
    /// that:
    ///   1. Candle storage is STALE — `var.as_tensor()` data still
    ///      matches the pre-step values.
    ///   2. The registry buffer is CURRENT — `resolve_resident_activation`
    ///      returns the post-step values.
    ///   3. After explicit `var.set(resolve(...))` (which is what
    ///      `TrainableLoraParams::sync_to_candle` does internally),
    ///      candle storage matches the registry.
    /// This is the contract the lazy-sync flow relies on.
    #[test]
    fn lazy_sync_keeps_candle_stale_until_explicit_sync() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
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

        let p_var =
            candle_core::Var::from_tensor(&Tensor::from_vec(init.clone(), (n,), &Device::Cpu)?)?;
        let g_tensor = Tensor::from_vec(grad, (n,), &Device::Cpu)?;

        backend.register_resident_activation(p_var.as_tensor())?;
        backend.register_resident_activation(&g_tensor)?;
        let dispatched = backend.dispatch_sgd_step(p_var.as_tensor(), &g_tensor, lr)?;
        assert!(dispatched);

        // (1) Candle storage is still the initial values.
        let stale: Vec<f32> = p_var.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        for (i, (s, w)) in stale.iter().zip(init.iter()).enumerate() {
            assert!(
                (s - w).abs() < 1e-7,
                "candle storage must be stale post-dispatch: idx {i}: got {s}, init {w}"
            );
        }

        // (2) Registry has post-step values.
        let resolved = backend
            .resolve_resident_activation(p_var.as_tensor(), &[n], DType::F32)?
            .expect("must resolve after on-device dispatch");
        let resolved_v: Vec<f32> = resolved.flatten_all()?.to_vec1::<f32>()?;
        for (i, (r, w)) in resolved_v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - w).abs() < 1e-6,
                "registry must hold post-step values: idx {i}: got {r}, want {w}"
            );
        }

        // (3) After explicit var.set, candle storage matches.
        p_var.set(&resolved)?;
        let fresh: Vec<f32> = p_var.as_tensor().flatten_all()?.to_vec1::<f32>()?;
        for (i, (f, w)) in fresh.iter().zip(expected.iter()).enumerate() {
            assert!(
                (f - w).abs() < 1e-6,
                "candle storage must match registry post-sync: idx {i}: got {f}, want {w}"
            );
        }

        backend.evict_resident_activation(p_var.as_tensor());
        backend.evict_resident_activation(&g_tensor);
        Ok(())
    }

    /// dispatch_sgd_step still falls back when dtypes don't match
    /// (e.g. BF16 param but F32 grad). Mixed-precision SGD requires
    /// an F32 master copy that we don't maintain.
    #[test]
    fn dispatch_sgd_step_falls_back_on_dtype_mismatch() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = Tensor::from_vec(vec![1.0f32; 4], (4,), &Device::Cpu)?.to_dtype(DType::BF16)?;
        let g = Tensor::from_vec(vec![0.5f32; 4], (4,), &Device::Cpu)?; // F32
        backend.register_resident_activation(&p)?;
        backend.register_resident_activation(&g)?;
        let dispatched = backend.dispatch_sgd_step(&p, &g, 0.01)?;
        assert!(!dispatched, "dtype mismatch must fall back");
        backend.evict_resident_activation(&p);
        backend.evict_resident_activation(&g);
        Ok(())
    }

    #[test]
    fn resident_activation_register_evict_round_trip() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        // The capability bit is true regardless of whether a Vulkan
        // device exists in the test environment — it advertises the
        // backend's *intent* to handle these hooks non-trivially.
        // Trainer call sites gate on this to avoid the per-call
        // extract_tensor_bytes overhead on CPU/Metal/CUDA backends.
        assert!(
            backend.supports_resident_activation(),
            "VulkanBackend must advertise resident-activation support"
        );
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping live registry test");
            return Ok(());
        }
        // Small synthetic tensor — no specific shape required, the
        // hook just uploads `extract_tensor_bytes(tensor).0` and
        // keys on `tensor.id()`.
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)?;
        assert!(
            !backend.has_resident_activation(&t),
            "fresh tensor must not be registered"
        );
        backend.register_resident_activation(&t)?;
        assert!(
            backend.has_resident_activation(&t),
            "tensor must be registered after register_resident_activation"
        );
        // Idempotency: re-registering the same tensor is a no-op,
        // not an error.
        backend.register_resident_activation(&t)?;
        assert!(backend.has_resident_activation(&t));
        backend.evict_resident_activation(&t);
        assert!(
            !backend.has_resident_activation(&t),
            "tensor must be unregistered after evict_resident_activation"
        );
        // Evicting again is also a no-op.
        backend.evict_resident_activation(&t);
        Ok(())
    }
}

//! Vulkan backend: FlashAttention-2 and Gated DeltaNet fused kernels via Vulkan.
//!
//! candle-core 0.10.x has no native Vulkan device, so this backend manages
//! its own `vk::Device` and copies tensor data through the CPU path at
//! kernel boundaries. The model's main forward pass (matmuls, MLP, lm_head)
//! runs on CPU; only GDN-specific kernel calls reach Vulkan. This means
//! each kernel call pays a CPU→GPU→CPU roundtrip, which is the primary
//! bottleneck for this backend. A native Vulkan tensor storage layer is needed
//! to keep tensors resident on GPU between kernel calls.
//!
//! `Ok(None)` responses route the caller to the portable candle path.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, TensorId};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::BackendRuntime;
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
    /// General-purpose resident-activation registry keyed by candle
    /// `TensorId`. Phase 3.1 of the residency plan — the registry the
    /// `register_resident_activation` / `evict_resident_activation` /
    /// `has_resident_activation` BackendRuntime hooks read and write.
    /// Separate from `RECURRENT_STATE_RESIDENT_CACHE` so the
    /// GDN-specific hot path can keep its own scope-limited lifecycle
    /// without growing accidental coupling to non-recurrent
    /// activations. Entries here are evicted explicitly by the caller
    /// (Phase 3.2 will add the trainer-side wiring for that).
    static RESIDENT_ACTIVATION_REGISTRY: RefCell<HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>> =
        RefCell::new(HashMap::new());
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

/// Read `KILN_VULKAN_LINEAR` env var. When enabled, the autograd-safe
/// `linear_prefill_apply` path wraps the existing Vulkan linear kernel in
/// a `CustomOp1` so training projections produce a tracked tensor whose
/// backward computes a real gradient instead of dropping it at the leaf
/// returned by the inference-shaped `linear_decode`.
///
/// Default: **disabled**. Two host hard-hangs were observed on Strix Halo
/// when this defaulted on with the original `/tmp/sft-data.jsonl` repro
/// (T≈918, vocab=152064): the lm_head training-time forward routes through
/// here, and a single dispatch of `[918, 2560] @ [2560, 152064]` queues
/// ~4.36M workgroups in one submit on a 40-CU APU. The kernel logs go
/// silent (no OOM, no AMDGPU reset, no panic), meaning the GPU/driver
/// didn't recover gracefully and the box had to be physically rebooted.
/// `VulkanLinearOp` now chunks oversized BF16-packed dispatches along
/// the output dim (forward) or batch dim (backward) so each per-chunk
/// submit stays under the FLOP ceiling and `queue_wait_idle()` between
/// chunks gives the display compositor preemption points. The F32
/// weight path has no offset kernel, so it still bails to CPU
/// `broadcast_matmul` for oversized shapes. Until the chunking has
/// been load-validated end-to-end on the original repro, the env var
/// is opt-in: set `KILN_VULKAN_LINEAR=1` (or `true`/`yes`) to enable.
/// Smaller-shape projections (T≤256) ran ~6% faster with this on
/// (111 s vs 118 s baseline) at bit-exact loss.
fn linear_prefill_apply_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| kiln_core::env_flag::env_flag("KILN_VULKAN_LINEAR", false))
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
            weight_cache: Mutex::new(HashMap::new()),
            bf16_packed_weight_cache: Mutex::new(HashMap::new()),
            vulkan_device,
        }
    }

    fn has_vulkan(&self) -> bool {
        self.vulkan_device.is_some()
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
        let q_f32 = if in_dtype == DType::F32 { q.clone() } else { q.to_dtype(DType::F32)? };
        let k_f32 = if in_dtype == DType::F32 { k.clone() } else { k.to_dtype(DType::F32)? };
        let v_f32 = if in_dtype == DType::F32 { v.clone() } else { v.to_dtype(DType::F32)? };

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

    fn supports_flash_attn_prefill(&self) -> bool {
        // The flash_attn.comp placeholder is replaced by the
        // sdpa_prefill_f32.comp kernel landed in commit dc4664ed.
        // The new kernel is parity-tested against CPU broadcast_matmul
        // + softmax at small N including the Qwen3.5-4B head_dim=128
        // shape, but has not been load-validated on a real training
        // forward yet, so the support flag is opt-in via
        // `KILN_VULKAN_SDPA=1` until we observe a clean run.
        if !self.has_vulkan() {
            return false;
        }
        kiln_core::env_flag::env_flag("KILN_VULKAN_SDPA", false)
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
        let already_registered =
            RESIDENT_ACTIVATION_REGISTRY.with(|cache| cache.borrow().contains_key(&id));
        if already_registered {
            return Ok(());
        }
        let bytes = kiln_vulkan_kernel::kernels::extract_tensor_bytes(tensor)?.0;
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
        RESIDENT_ACTIVATION_REGISTRY.with(|cache| {
            cache.borrow_mut().insert(id, buffer);
        });
        Ok(())
    }

    fn evict_resident_activation(&self, tensor: &Tensor) {
        let id = tensor.id();
        RESIDENT_ACTIVATION_REGISTRY.with(|cache| {
            cache.borrow_mut().remove(&id);
        });
    }

    fn has_resident_activation(&self, tensor: &Tensor) -> bool {
        let id = tensor.id();
        RESIDENT_ACTIVATION_REGISTRY.with(|cache| cache.borrow().contains_key(&id))
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
        let buffer = RESIDENT_ACTIVATION_REGISTRY.with(|cache| cache.borrow().get(&id).cloned());
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
        let resolved = kiln_vulkan_kernel::kernels::create_tensor_from_data(&bytes, shape, dtype)
            .context("resolve_resident_activation: create_tensor_from_data")?;
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
        let lookup = RESIDENT_ACTIVATION_REGISTRY.with(|cache| {
            let cache = cache.borrow();
            cache
                .get(&param_id)
                .and_then(|p| cache.get(&grad_id).map(|g| (Arc::clone(p), Arc::clone(g))))
        });
        let Some((param_buf, grad_buf)) = lookup else {
            return Ok(false);
        };
        // Both buffers are F32 (resident registry stores raw bytes —
        // the SGD shader assumes F32, matching the candle Var layout
        // for LoRA parameters).
        if param.dtype() != DType::F32 || grad.dtype() != DType::F32 {
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
        // One-shot trace so the operator can confirm the on-device
        // SGD path is engaging. Cumulatively the FIRST_*_LOGGED OneLocks
        // give a clear "registry → matmul chunking → SGD" signal in
        // the startup log without per-step spam.
        static FIRST_SGD_LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        FIRST_SGD_LOGGED.get_or_init(|| {
            tracing::info!(
                n_elements,
                lr,
                "VulkanBackend::dispatch_sgd_step first call"
            );
        });
        kiln_vulkan_kernel::kernels::dispatch_sgd_step_f32(
            vk_device,
            &param_buf,
            &grad_buf,
            n_elements,
            lr,
        )?;
        Ok(true)
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
        let mut gather_slots = Vec::with_capacity(batch * max_seqlen_k);
        for row in 0..batch {
            let row_len = usize::try_from(seq_i32[row])
                .context("Vulkan paged decode seqused_k contains negative length")?;
            if row_len == 0 || row_len > max_seqlen_k {
                return Ok(None);
            }
            seq_lens.push(
                u32::try_from(row_len).context("Vulkan paged decode row length exceeds u32")?,
            );
            for pos in 0..max_seqlen_k {
                if pos >= row_len {
                    gather_slots.push(0);
                    continue;
                }
                let block_idx = pos / page_block_size;
                let offset = pos % page_block_size;
                let block = block_data[row * max_blocks_per_seq + block_idx] as usize;
                let slot = block
                    .checked_mul(page_block_size)
                    .and_then(|base| base.checked_add(offset))
                    .context("Vulkan paged decode slot index overflow")?;
                if slot >= total_slots {
                    return Ok(None);
                }
                gather_slots
                    .push(u32::try_from(slot).context("Vulkan paged decode slot exceeds u32")?);
            }
        }

        let gather = Tensor::from_slice(gather_slots.as_slice(), batch * max_seqlen_k, q.device())?;
        let k_compact = k_pool
            .index_select(&gather, 0)?
            .reshape((batch, max_seqlen_k, num_kv_heads, head_dim))?
            .contiguous()?;
        let v_compact = v_pool
            .index_select(&gather, 0)?
            .reshape((batch, max_seqlen_k, num_kv_heads, head_dim))?
            .contiguous()?;

        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;
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
        let out = x
            .apply_op1(op)
            .context("VulkanLinearOp apply_op1 failed")?;
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
                    .saturating_mul(chunk_len as u64))
                    as f64
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
                let sub = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bf16_weights_offset(
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
        if batch != 1 || seq_len != 1 {
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
        let out =
            if self.use_bf16_packed_full_attn_qkv_weights(&[q_weight_t, k_weight_t, v_weight_t]) {
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
            .context("full_attn_qkv_decode kernel failed")?;
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
    use candle_core::Device;
    use crate::backend::BackendRuntime;

    /// Round-trip test for the Phase 3.1 hooks. Registers a fresh
    /// activation, asserts `has_resident_activation` flips true,
    /// evicts it, asserts it flips back. Skipped if no Vulkan
    /// device — the hooks have no-op defaults so a CPU-only run
    /// would just always answer false.
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
            assert!(
                (got - want).abs() < 1e-9,
                "idx {i}: got {got} want {want}"
            );
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
        assert!(dispatched, "dispatch_sgd_step should succeed when both buffers are resident");

        // Read back the updated param buffer from the registry.
        let param_buf = RESIDENT_ACTIVATION_REGISTRY
            .with(|cache| cache.borrow().get(&param.id()).cloned())
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

    /// dispatch_sgd_step must fall back (return Ok(false), not error)
    /// when operands have a non-F32 dtype the kernel doesn't handle.
    /// This matches the trait contract: false = "caller use the
    /// candle CPU path"; error = "I tried but it failed."
    #[test]
    fn dispatch_sgd_step_falls_back_on_non_f32_dtype() -> Result<()> {
        let backend = VulkanBackend::new(Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p_f32 = Tensor::from_vec(vec![1.0f32; 4], (4,), &Device::Cpu)?;
        let p_bf16 = p_f32.to_dtype(DType::BF16)?;
        let g_bf16 = p_f32.to_dtype(DType::BF16)?;
        backend.register_resident_activation(&p_bf16)?;
        backend.register_resident_activation(&g_bf16)?;
        let dispatched = backend.dispatch_sgd_step(&p_bf16, &g_bf16, 0.01)?;
        assert!(!dispatched, "BF16 operands must fall back to CPU");
        backend.evict_resident_activation(&p_bf16);
        backend.evict_resident_activation(&g_bf16);
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
        assert!(!backend.has_resident_activation(&t), "fresh tensor must not be registered");
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

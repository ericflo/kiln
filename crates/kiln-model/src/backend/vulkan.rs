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
    gdn_gates_enabled: bool,
    gdn_gated_rms_norm_enabled: bool,
    fused_conv1d_enabled: bool,
    gdn_forward_sub_enabled: bool,
    gdn_decode_fused_enabled: bool,
    linear_decode_enabled: bool,
    linear_argmax_batch_enabled: bool,
    full_attn_qkv_enabled: bool,
    mlp_decode_enabled: bool,
    mlp_gate_up_enabled: bool,
    weight_prewarm_enabled: bool,
    recurrent_state_residency_enabled: bool,
    /// Cached f32 device-local buffers for immutable CPU weight tensors.
    ///
    /// This field must drop before `vulkan_device`: `VulkanBuffer` owns raw
    /// memory that must be freed before the logical Vulkan device is destroyed.
    weight_cache: Mutex<HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>>,
    /// Vulkan device (owned, not from candle-core)
    vulkan_device: Option<Box<kiln_vulkan_kernel::VulkanDevice>>,
}

thread_local! {
    static RECURRENT_STATE_RESIDENT_SCOPE_DEPTH: Cell<usize> = const { Cell::new(0) };
    static RECURRENT_STATE_RESIDENT_CACHE: RefCell<HashMap<TensorId, Arc<kiln_vulkan_kernel::VulkanBuffer>>> =
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

fn enter_recurrent_state_resident_scope() {
    RECURRENT_STATE_RESIDENT_SCOPE_DEPTH.with(|depth| {
        let previous = depth.get();
        if previous == 0 {
            RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow_mut().clear());
        }
        depth.set(previous + 1);
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
        if next == 0 {
            RECURRENT_STATE_RESIDENT_CACHE.with(|cache| cache.borrow_mut().clear());
        }
    });
}

impl VulkanBackend {
    pub fn new(device: Device) -> Self {
        let gdn_enabled = std::env::var("KILN_DISABLE_GDN_KERNEL").is_err();
        let gdn_gates_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATES").is_err();
        let gdn_gated_rms_norm_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM").is_err();
        // forward_sub is opt-in only (default off): solve_tri shared-memory
        // layout is not yet validated against CPU parity and may exceed
        // maxComputeSharedMemorySize on many GPUs.
        //
        // conv1d remains opt-in while we A/B the corrected state-aware shader
        // against the CPU fallback on Strix Halo.
        let fused_conv1d_enabled =
            gdn_enabled && std::env::var("KILN_ENABLE_VULKAN_FUSED_CONV1D").is_ok();
        let gdn_forward_sub_enabled =
            gdn_enabled && std::env::var("KILN_ENABLE_VULKAN_GDN_FORWARD_SUB").is_ok();
        // The fused GDN decode path is validated, but for bs=1 it remains
        // run-to-run unstable on Strix Halo. Batch decode enables it by shape
        // in `gdn_decode_gates_recurrent_rmsnorm`; this env gates bs=1 only.
        let gdn_decode_fused_enabled =
            gdn_enabled && std::env::var("KILN_ENABLE_VULKAN_GDN_DECODE_FUSED").is_ok();
        let linear_decode_enabled = std::env::var("KILN_DISABLE_VULKAN_LINEAR_DECODE").is_err();
        let linear_argmax_batch_enabled =
            std::env::var("KILN_DISABLE_VULKAN_LINEAR_ARGMAX_BATCH").is_err();
        let full_attn_qkv_enabled = std::env::var("KILN_DISABLE_VULKAN_FULL_ATTN_QKV").is_err();
        // Full fused MLP decode is validated for single-token no-LoRA decode.
        // After descriptor-pool reuse and tiled projection kernels it is now
        // consistently faster than the split generic GEMV path on Strix Halo.
        let mlp_decode_enabled = std::env::var("KILN_DISABLE_VULKAN_MLP_DECODE").is_err();
        // The fused Vulkan MLP gate/up shader is validated, but on Strix Halo
        // it was slower than the generic cached GEMV path in short decode
        // benchmarks. Keep it opt-in until it is tiled/tuned.
        let mlp_gate_up_enabled = std::env::var("KILN_ENABLE_VULKAN_MLP_GATE_UP").is_ok();
        let weight_prewarm_enabled = std::env::var("KILN_DISABLE_VULKAN_WEIGHT_PREWARM").is_err();
        let recurrent_state_residency_enabled = gdn_enabled
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
                Some(Box::new(dev))
            }
            Err(e) => {
                tracing::warn!(error = %e, "Vulkan device initialization failed, falling back to CPU");
                None
            }
        };

        Self {
            device,
            gdn_enabled,
            gdn_gates_enabled,
            gdn_gated_rms_norm_enabled,
            fused_conv1d_enabled,
            gdn_forward_sub_enabled,
            gdn_decode_fused_enabled,
            linear_decode_enabled,
            linear_argmax_batch_enabled,
            full_attn_qkv_enabled,
            mlp_decode_enabled,
            mlp_gate_up_enabled,
            weight_prewarm_enabled,
            recurrent_state_residency_enabled,
            weight_cache: Mutex::new(HashMap::new()),
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

        let (b, seq_len, num_heads, head_dim) = q.dims4()?;

        // Only support head_dim=128 for now (matches CUDA kernel constraint)
        if head_dim != 128 {
            return Ok(None);
        }

        // Compile shader (cached after first call)
        let glsl_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../kiln-vulkan-kernel/csrc/shaders/flash_attn.comp"
        );
        let spirv = kiln_vulkan_kernel::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

        // Push constants: batch, seq_len, num_heads, head_dim, softmax_scale, causal
        let push_constants: [u32; 6] = [
            b as u32,
            seq_len as u32,
            num_heads as u32,
            head_dim as u32,
            softmax_scale.to_bits(),
            causal as u32,
        ];

        // Workgroup count: one group per (head, seq, batch)
        let workgroup_count = (num_heads as u32, seq_len as u32, b as u32);

        let output_shape = vec![b, seq_len, num_heads, head_dim];

        let out = kiln_vulkan_kernel::kernels::dispatch_kernel(
            vk_device,
            &spirv,
            &push_constants,
            workgroup_count,
            &[q, k, v],
            &output_shape,
            DType::BF16,
        )?;

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
        // flash_attn_prefill is a placeholder — only works for head_dim=128
        // and is missing scratch/LSE/causal-mask buffers. Return false
        // so callers don't skip their preamble work only to get Ok(None).
        false
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

    fn supports_gdn_chunk_prep(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_chunk_scan(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        // The fused full-chunk shader has not been validated against the
        // canonical prep + scan + state-update path. Keep Vulkan on the
        // correct split path until that kernel has parity coverage.
        false
    }

    fn supports_gdn_gates(&self) -> bool {
        self.has_vulkan() && self.gdn_gates_enabled
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        self.has_vulkan() && self.gdn_gated_rms_norm_enabled
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        // Opt-in until Strix Halo latency confirms the two extra Vulkan
        // dispatches beat the host fallback in the real decode loop.
        self.has_vulkan() && self.fused_conv1d_enabled
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
        // Same experimental status as _update.
        self.has_vulkan() && self.fused_conv1d_enabled
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

        let Ok((_batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        if seq_len != 1 {
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
        let qkv_buf = self.cached_f32_weight_buffer(in_proj_qkv_t)?;
        let z_buf = self.cached_f32_weight_buffer(in_proj_z_t)?;
        let a_buf = self.cached_f32_weight_buffer(in_proj_a_t)?;
        let b_buf = self.cached_f32_weight_buffer(in_proj_b_t)?;

        let result = kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached(
            vk_device, x, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim, z_dim, a_dim, b_dim,
        )
        .context("gdn_in_proj_decode kernel failed")?;
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
        if seq_len != 1 {
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
        let weight_buf = self.cached_f32_weight_buffer(weight_t)?;
        let out = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached(
            vk_device,
            x,
            &weight_buf,
            batch,
            hidden,
            out_dim,
        )
        .context("linear_decode kernel failed")?;
        Ok(Some(out))
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
        let weight_buf = self.cached_f32_weight_buffer(weight_t)?;
        let token = kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached(
            vk_device,
            x,
            &weight_buf,
            hidden,
            out_dim,
        )
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
        let weight_buf = self.cached_f32_weight_buffer(weight_t)?;
        let tokens = kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached(
            vk_device,
            x,
            &weight_buf,
            batch,
            hidden,
            out_dim,
        )
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

        self.prewarm_f32_weight(
            "embed_tokens_t",
            &weights.embed_tokens_t,
            &mut count,
            &mut bytes,
        )?;

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            match &layer.attention {
                GpuAttentionWeights::Full(attn) => {
                    self.prewarm_f32_weight(
                        &format!("layers.{layer_idx}.attention.q_proj_t"),
                        &attn.q_proj_t,
                        &mut count,
                        &mut bytes,
                    )?;
                    self.prewarm_f32_weight(
                        &format!("layers.{layer_idx}.attention.k_proj_t"),
                        &attn.k_proj_t,
                        &mut count,
                        &mut bytes,
                    )?;
                    self.prewarm_f32_weight(
                        &format!("layers.{layer_idx}.attention.v_proj_t"),
                        &attn.v_proj_t,
                        &mut count,
                        &mut bytes,
                    )?;
                    self.prewarm_f32_weight(
                        &format!("layers.{layer_idx}.attention.o_proj_t"),
                        &attn.o_proj_t,
                        &mut count,
                        &mut bytes,
                    )?;
                }
                GpuAttentionWeights::Linear(attn) => {
                    self.prewarm_f32_weight(
                        &format!("layers.{layer_idx}.attention.in_proj_qkv_t"),
                        &attn.in_proj_qkv_t,
                        &mut count,
                        &mut bytes,
                    )?;
                    self.prewarm_f32_weight(
                        &format!("layers.{layer_idx}.attention.in_proj_z_t"),
                        &attn.in_proj_z_t,
                        &mut count,
                        &mut bytes,
                    )?;
                    self.prewarm_f32_weight(
                        &format!("layers.{layer_idx}.attention.in_proj_a_t"),
                        &attn.in_proj_a_t,
                        &mut count,
                        &mut bytes,
                    )?;
                    self.prewarm_f32_weight(
                        &format!("layers.{layer_idx}.attention.in_proj_b_t"),
                        &attn.in_proj_b_t,
                        &mut count,
                        &mut bytes,
                    )?;
                    self.prewarm_f32_weight(
                        &format!("layers.{layer_idx}.attention.out_proj_t"),
                        &attn.out_proj_t,
                        &mut count,
                        &mut bytes,
                    )?;
                }
            }

            self.prewarm_f32_weight(
                &format!("layers.{layer_idx}.mlp.gate_proj_t"),
                &layer.mlp.gate_proj_t,
                &mut count,
                &mut bytes,
            )?;
            self.prewarm_f32_weight(
                &format!("layers.{layer_idx}.mlp.up_proj_t"),
                &layer.mlp.up_proj_t,
                &mut count,
                &mut bytes,
            )?;
            self.prewarm_f32_weight(
                &format!("layers.{layer_idx}.mlp.down_proj_t"),
                &layer.mlp.down_proj_t,
                &mut count,
                &mut bytes,
            )?;
        }

        tracing::info!(
            weights = count,
            f32_cache_mb = bytes / (1024 * 1024),
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
        let q_buf = self.cached_f32_weight_buffer(q_weight_t)?;
        let k_buf = self.cached_f32_weight_buffer(k_weight_t)?;
        let v_buf = self.cached_f32_weight_buffer(v_weight_t)?;
        let out = kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached(
            vk_device, x, &q_buf, &k_buf, &v_buf, hidden, q_dim, k_dim, v_dim,
        )
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

        let Ok((_batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        if seq_len != 1 {
            return Ok(None);
        }
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
        let out = kiln_vulkan_kernel::kernels::dispatch_mlp_gate_up_decode_cached(
            vk_device,
            x,
            &gate_buf,
            &up_buf,
            hidden,
            intermediate,
        )
        .context("mlp_gate_up_decode kernel failed")?;
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

        let Ok((_batch, seq_len, hidden)) = x.dims3() else {
            return Ok(None);
        };
        if seq_len != 1 {
            return Ok(None);
        }
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
        let gate_buf = self.cached_f32_weight_buffer(gate_weight_t)?;
        let up_buf = self.cached_f32_weight_buffer(up_weight_t)?;
        let down_buf = self.cached_f32_weight_buffer(down_weight_t)?;
        let out = kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached(
            vk_device,
            x,
            &gate_buf,
            &up_buf,
            &down_buf,
            hidden,
            intermediate,
            out_dim,
        )
        .context("mlp_decode kernel failed")?;
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

        let (out, new_state) = kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step(
            vk_device, q, k, v, beta, g, state,
        )
        .context("gdn_recurrent_step kernel failed")?;
        *state = new_state;
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
        if !self.has_vulkan() || !self.fused_conv1d_enabled {
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
        if !self.has_vulkan() || !self.fused_conv1d_enabled {
            return Ok(None);
        }
        if !matches!(x.dtype(), DType::BF16 | DType::F32) {
            return Ok(None);
        }
        let vk_device = self
            .vulkan_device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (out, new_state) = kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill(
            vk_device,
            x,
            weight,
            conv_state,
            kernel_size,
        )
        .context("causal_conv1d_prefill kernel failed")?;
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

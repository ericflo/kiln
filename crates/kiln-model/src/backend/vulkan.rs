//! Vulkan backend: FlashAttention-2 and Gated DeltaNet fused kernels via Vulkan.
//!
//! candle-core 0.10.x has no native Vulkan device, so this backend manages
//! its own `vk::Device` and copies tensor data through the CPU path at
//! kernel boundaries. This matches llama.cpp's Vulkan approach.
//!
//! `Ok(None)` responses route the caller to the portable candle path.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use super::BackendRuntime;

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
    /// Vulkan device (owned, not from candle-core)
    vulkan_device: Option<Box<kiln_vulkan_kernel::VulkanDevice>>,
}

impl VulkanBackend {
    pub fn new(device: Device) -> Self {
        let gdn_enabled = std::env::var("KILN_DISABLE_GDN_KERNEL").is_err();
        let gdn_gates_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATES").is_err();
        let gdn_gated_rms_norm_enabled = gdn_enabled
            && std::env::var("KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM").is_err();
        let fused_conv1d_enabled = std::env::var("KILN_DISABLE_FUSED_CONV1D").is_err();

        let vulkan_device = match kiln_vulkan_kernel::VulkanDevice::new() {
            Ok(dev) => {
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
            vulkan_device,
        }
    }

    fn has_vulkan(&self) -> bool {
        self.vulkan_device.is_some()
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
        let vk_device = self.vulkan_device.as_ref()
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

impl BackendRuntime for VulkanBackend {
    fn name(&self) -> &'static str {
        if self.has_vulkan() { "vulkan" } else { "cpu" }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn supports_flash_attn_prefill(&self) -> bool {
        self.has_vulkan()
    }

    fn supports_flash_attn_prefill_head_major(&self) -> bool {
        self.has_vulkan()
    }

    fn supports_flash_attn_paged_decode(&self) -> bool {
        // Not yet implemented — returning false so callers don't skip
        // their preamble work only to get Ok(None) back.
        false
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_chunk_prep(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_chunk_scan(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        self.has_vulkan() && self.gdn_enabled
    }

    fn supports_gdn_gates(&self) -> bool {
        self.has_vulkan() && self.gdn_gates_enabled
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        self.has_vulkan() && self.gdn_gated_rms_norm_enabled
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        self.has_vulkan() && self.fused_conv1d_enabled
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
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
        let vk_device = self.vulkan_device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let out = kiln_vulkan_kernel::kernels::dispatch_gdn_forward_substitution(
            vk_device, a_strict, v_prime, beta,
        ).context("gdn_forward_substitution kernel failed")?;
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
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        let vk_device = self.vulkan_device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (out, new_state) = kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step(
            vk_device, q, k, v, beta, g, state,
        ).context("gdn_recurrent_step kernel failed")?;
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
        let vk_device = self.vulkan_device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let result = kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_prep(
            vk_device, g, v, kkt, qkt, ks_entry, q_s,
        ).context("gdn_chunk_prep kernel failed")?;
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
        let vk_device = self.vulkan_device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let result = kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_scan(
            vk_device, a_strict, b_mask, v_prime, q_s_scaled, beta, decay_last_col,
        ).context("gdn_chunk_scan kernel failed")?;
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
        let vk_device = self.vulkan_device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (out, new_state) = kiln_vulkan_kernel::kernels::dispatch_gdn_full_chunk_forward(
            vk_device, g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state,
        ).context("gdn_full_chunk_forward kernel failed")?;
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
        if a.dtype() != DType::BF16 {
            return Ok(None);
        }
        let vk_device = self.vulkan_device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        // Output shape matches input shape [B, T, nv]
        let out_shape = a.dims().as_ref().to_vec();
        let (beta, g) = kiln_vulkan_kernel::kernels::dispatch_gdn_gates(
            vk_device, a, b, a_log, dt_bias, &out_shape,
        ).context("gdn_gates kernel failed")?;
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
        if x.dtype() != DType::BF16 {
            return Ok(None);
        }
        let vk_device = self.vulkan_device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        // Output shape matches x shape
        let out_shape = x.dims().as_ref().to_vec();
        let out = kiln_vulkan_kernel::kernels::dispatch_gdn_gated_rms_norm(
            vk_device, x, z, weight, eps as f32, &out_shape,
        ).context("gdn_gated_rms_norm kernel failed")?;
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
        if x.dtype() != DType::BF16 {
            return Ok(None);
        }
        let vk_device = self.vulkan_device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (out, new_state) = kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_update(
            vk_device, x, weight, conv_state, kernel_size,
        ).context("causal_conv1d_update kernel failed")?;
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
        if x.dtype() != DType::BF16 {
            return Ok(None);
        }
        let vk_device = self.vulkan_device.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Vulkan device not available"))?;

        let (out, new_state) = kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill(
            vk_device, x, weight, conv_state, kernel_size,
        ).context("causal_conv1d_prefill kernel failed")?;
        *conv_state = new_state;
        Ok(Some(out))
    }
}

/// Check if Vulkan is available on this system.
/// Uses a cached result to avoid the expensive VulkanDevice::new() call.
static VULKAN_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn vulkan_is_available() -> bool {
    *VULKAN_AVAILABLE.get_or_init(|| {
        kiln_vulkan_kernel::VulkanDevice::new().is_ok()
    })
}

/// Precompile Vulkan custom kernels (warm up pipeline cache).
///
/// Compiles all known GLSL shaders to SPIR-V at startup to avoid
/// compilation latency during the first inference request.
pub fn precompile_custom_kernels(_device: &Device) -> Result<()> {
    let shaders = [
        "gdn_gates",
        "gdn_gated_rms_norm",
        "causal_conv1d",
        "solve_tri",
        "gdn_recurrent_prefill",
        "gdn_chunk_prep",
        "gdn_full_chunk_forward",
        "gdn_chunk_scan",
        "flash_attn",
    ];

    let base = env!("CARGO_MANIFEST_DIR");
    let vulkan_base = format!("{}/../kiln-vulkan-kernel/csrc/shaders", base);

    for shader_name in &shaders {
        let glsl_path = format!("{}/{}.comp", vulkan_base, shader_name);
        match kiln_vulkan_kernel::pipeline::ShaderPipeline::compile_shader(&glsl_path) {
            Ok(_) => tracing::info!(shader = %shader_name, "precompiled Vulkan shader"),
            Err(e) => tracing::warn!(shader = %shader_name, error = %e, "failed to precompile Vulkan shader (will compile on first use)"),
        }
    }

    tracing::info!("Vulkan shader precompilation complete");
    Ok(())
}

//! CUDA backend: FlashAttention-2 and Gated DeltaNet fused kernels.
//!
//! Wraps the vendored `kiln-flash-attn` and `kiln-gdn-kernel` crates.
//! `Ok(None)` responses route the caller to the portable candle path.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use super::BackendRuntime;

#[derive(Debug)]
pub struct CudaBackend {
    device: Device,
    /// Cached at construction: reading env vars per decode step × 24 GDN layers
    /// shows up in decode NVTX captures. Env vars don't change at runtime.
    gdn_enabled: bool,
    /// Same pattern: cache the env-var read. The fused gates kernel is
    /// gated behind its own kill switch so it can be disabled independently.
    gdn_gates_enabled: bool,
}

impl CudaBackend {
    pub fn new(device: Device) -> Self {
        debug_assert!(device.is_cuda(), "CudaBackend created on non-CUDA device");
        let gdn_enabled = std::env::var("KILN_DISABLE_GDN_KERNEL").is_err();
        let gdn_gates_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATES").is_err();
        Self {
            device,
            gdn_enabled,
            gdn_gates_enabled,
        }
    }
}

impl BackendRuntime for CudaBackend {
    fn name(&self) -> &'static str {
        "cuda"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn supports_flash_attn_prefill(&self) -> bool {
        true
    }

    fn supports_flash_attn_paged_decode(&self) -> bool {
        true
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        self.gdn_enabled
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        self.gdn_enabled
    }

    fn flash_attn_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        // The vendored CUDA kernel hard-errors on non-BF16. Decline here so
        // the caller falls back to the portable path instead of bubbling a
        // hard error up for non-BF16 test configs.
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        let out = kiln_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
            .context("flash_attn kernel failed")?;
        Ok(Some(out))
    }

    fn flash_attn_paged_decode(
        &self,
        q: &Tensor,
        k_pool: &Tensor,
        v_pool: &Tensor,
        block_table: &Tensor,
        total_seqlen_k: usize,
        page_block_size: usize,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        let out = kiln_flash_attn::flash_attn_paged_decode(
            q,
            k_pool,
            v_pool,
            block_table,
            total_seqlen_k,
            page_block_size,
            softmax_scale,
            causal,
        )
        .context("flash_attn_paged_decode kernel failed")?;
        Ok(Some(out))
    }

    fn gdn_forward_substitution(
        &self,
        a_strict: &Tensor,
        v_prime: &Tensor,
        beta: &Tensor,
    ) -> Result<Option<Tensor>> {
        if a_strict.dtype() != DType::BF16 {
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_forward_substitution(a_strict, v_prime, beta)?;
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
        if q.dtype() != DType::BF16 {
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_recurrent_forward(q, k, v, beta, g, state)?;
        Ok(Some(out))
    }

    fn supports_gdn_gates(&self) -> bool {
        self.gdn_gates_enabled
    }

    fn gdn_gates(
        &self,
        a: &Tensor,
        b: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        // Decline quietly when the envelope isn't met — the candle reference
        // path handles non-CUDA, non-bf16, and oversized nv.
        if !kiln_gdn_kernel::gdn_gates_supports(a, b, a_log, dt_bias) {
            return Ok(None);
        }
        let (beta, g) = kiln_gdn_kernel::gdn_gates(a, b, a_log, dt_bias)
            .context("gdn_gates kernel failed")?;
        Ok(Some((beta, g)))
    }
}

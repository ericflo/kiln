//! CUDA backend implementation of [`BackendRuntime`].
//!
//! Wraps the two vendored CUDA kernel crates — `kiln-flash-attn` (FlashAttention-2
//! forward / paged-decode / backward) and `kiln-gdn-kernel` (Gated DeltaNet
//! forward-substitution + fused single-token recurrence) — with precondition
//! checks that decide whether the fused kernel can run. `Ok(None)` falls back
//! to the portable candle path in `forward.rs`.
//!
//! All preconditions mirror what `forward.rs` used to check inline inside
//! `#[cfg(feature = "cuda")]` blocks; the logic was lifted here verbatim as
//! part of the Phase 1 refactor.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use super::BackendRuntime;

#[derive(Debug)]
pub struct CudaBackend {
    device: Device,
}

impl CudaBackend {
    pub fn new(device: Device) -> Self {
        debug_assert!(device.is_cuda(), "CudaBackend created on non-CUDA device");
        Self { device }
    }

    fn gdn_disabled() -> bool {
        std::env::var("KILN_DISABLE_GDN_KERNEL").is_ok()
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
        !Self::gdn_disabled()
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        !Self::gdn_disabled()
    }

    fn flash_attn_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        if q.dtype() != DType::BF16 || !q.device().is_cuda() {
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
        if q.dtype() != DType::BF16 || !q.device().is_cuda() {
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
        if Self::gdn_disabled()
            || !a_strict.device().is_cuda()
            || a_strict.dtype() != DType::BF16
            || v_prime.dtype() != DType::BF16
            || beta.dtype() != DType::BF16
        {
            return Ok(None);
        }
        // Kernel envelope: C <= 128. Enforced by caller via explicit check
        // before invoking the trait; keep a defensive check here too.
        let c = a_strict.dim(2)?;
        if c > 128 {
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
        if Self::gdn_disabled()
            || !q.device().is_cuda()
            || q.dtype() != DType::BF16
            || state.dtype() != DType::BF16
        {
            return Ok(None);
        }
        let out = kiln_gdn_kernel::gdn_recurrent_forward(q, k, v, beta, g, state)?;
        Ok(Some(out))
    }
}

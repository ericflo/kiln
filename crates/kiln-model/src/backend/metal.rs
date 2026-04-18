//! Metal backend: candle's fused SDPA for the attention hot path, portable
//! fallback for everything else.
//!
//! candle-metal ships `candle_nn::ops::sdpa`, an MLX-style fused scaled-dot-
//! product attention kernel with native GQA, BF16, and head dims {32, 64, 72,
//! 80, 96, 128, 256, 512}. For Qwen3.5-4B's 8 full-attention layers (head_dim
//! 256, GQA ratio 4), this replaces the vendored CUDA FlashAttention-2 call on
//! the Wave 1 macOS target.
//!
//! GDN ops (`gdn_forward_substitution`, `gdn_recurrent_step`) return `Ok(None)`
//! today — the caller falls back to the per-token candle loop. Phase 2b (if
//! profiling shows it matters) adds a custom MSL kernel via
//! `candle_core::CustomOp1` or `candle-metal-kernels`.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

use super::BackendRuntime;

#[derive(Debug)]
pub struct MetalBackend {
    device: Device,
}

impl MetalBackend {
    pub fn new(device: Device) -> Self {
        debug_assert!(
            matches!(device, Device::Metal(_)),
            "MetalBackend created on non-Metal device"
        );
        Self { device }
    }
}

impl BackendRuntime for MetalBackend {
    fn name(&self) -> &'static str {
        "metal"
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn supports_flash_attn_prefill(&self) -> bool {
        true
    }

    /// FlashAttention prefill via `candle_nn::ops::sdpa`. Input layout matches
    /// the CUDA path: `[batch, seq_len, num_heads, head_dim]` bf16 contiguous.
    /// Callers have already GQA-expanded K/V; SDPA also accepts unexpanded
    /// GQA, but we pass the expanded tensors for shape symmetry with the CUDA
    /// path so later profiling compares apples-to-apples.
    fn flash_attn_prefill(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        softmax_scale: f32,
        causal: bool,
    ) -> Result<Option<Tensor>> {
        // candle's SDPA-on-Metal whitelists specific head_dims and q_seq>=1.
        // Decline (caller falls back to the portable path) rather than surfacing
        // a kernel error when the shape is unsupported — important for tiny
        // test configs and future model variants.
        if !metal_sdpa_supports_head_dim(q.dim(candle_core::D::Minus1)?) {
            return Ok(None);
        }

        // candle SDPA expects [batch, num_heads, seq_len, head_dim].
        let q_t = q.transpose(1, 2)?.contiguous()?;
        let k_t = k.transpose(1, 2)?.contiguous()?;
        let v_t = v.transpose(1, 2)?.contiguous()?;

        // `sdpa(q, k, v, mask, do_causal, scale, softcapping)`. softcapping=1.0
        // disables it. Kiln's prefill path is always causal.
        let out = candle_nn::ops::sdpa(&q_t, &k_t, &v_t, None, causal, softmax_scale, 1.0)
            .context("candle-metal sdpa failed")?;

        // Back to [batch, seq_len, num_heads, head_dim] to match CUDA output.
        let out = out.transpose(1, 2)?.contiguous()?;
        Ok(Some(out))
    }
}

/// Mirrors the head-dim whitelist in candle-nn's `Sdpa::custom_op3` on Metal
/// (see candle-nn 0.10.x `ops.rs`). Kept in sync by hand; if this drifts, the
/// fallback path absorbs the mismatch via a kernel error — safe but slow.
fn metal_sdpa_supports_head_dim(head_dim: usize) -> bool {
    matches!(head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256 | 512)
}

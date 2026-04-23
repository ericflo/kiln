//! Backend runtime abstraction for Kiln's platform-specific kernels.
//!
//! Most of the forward pass is expressed as `candle_core::Tensor` ops that
//! run on any candle device. A few ops — FlashAttention-2 forward /
//! paged-decode and the Gated DeltaNet fused recurrent + forward-substitution
//! kernels — have no candle equivalent and are implemented per-platform as
//! CUDA or (later) Metal kernels. This trait is the seam that lets the
//! forward pass dispatch those ops without threading `#[cfg(feature = "cuda")]`
//! gates through every call site.
//!
//! **`Option<Tensor>` return**: `Ok(None)` means "this backend declines this
//! call — fall back to the portable candle path". Matches the existing
//! `try_flash_attn_paged_decode` precondition-miss contract and extends it
//! to all kernel ops.
//!
//! **`supports_*` hints**: let the caller skip preamble work (e.g., a
//! `contiguous()` copy before the trait call) when the backend will decline
//! anyway. Intended to be constant-return for each concrete backend.

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::sync::Arc;

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

pub trait BackendRuntime: Send + Sync + std::fmt::Debug {
    /// Human-readable name (`"cuda"`, `"metal"`, `"cpu"`). Surfaced in
    /// `/health` and logs.
    fn name(&self) -> &'static str;

    /// The candle `Device` this backend drives. All tensors passed to trait
    /// methods must live on this device.
    fn device(&self) -> &Device;

    fn supports_flash_attn_prefill(&self) -> bool {
        false
    }

    fn supports_flash_attn_prefill_head_major(&self) -> bool {
        false
    }

    fn supports_flash_attn_paged_decode(&self) -> bool {
        false
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        false
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        false
    }

    fn supports_gdn_chunk_prep(&self) -> bool {
        false
    }

    fn supports_gdn_chunk_scan(&self) -> bool {
        false
    }

    fn supports_gdn_full_chunk_forward(&self) -> bool {
        false
    }

    /// FlashAttention-2 forward for prefill (no KV cache, seq_len > 1).
    ///
    /// `q`, `k`, `v`: `[batch, seq_len, num_heads, head_dim]` bf16 contiguous.
    /// Caller must GQA-expand K/V to match Q's head count. Returns
    /// `[batch, seq_len, num_heads, head_dim]` bf16.
    fn flash_attn_prefill(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// FlashAttention-2 forward for prefill with Q/K/V already in SDPA layout.
    ///
    /// `q`: `[batch, num_heads, seq_len, head_dim]` bf16 contiguous. `k` and
    /// `v`: `[batch, num_kv_heads, seq_len, head_dim]` bf16 contiguous.
    /// Backends may decline when they lack native GQA support. Returns
    /// `[batch, num_heads, seq_len, head_dim]` bf16.
    fn flash_attn_prefill_head_major(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _softmax_scale: f32,
        _causal: bool,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// FlashAttention-2 paged decode (single query token against paged K/V pool).
    ///
    /// `q`: `[batch, 1, num_heads, head_dim]` bf16. `k_pool`/`v_pool`:
    /// `[total_slots, num_kv_heads, head_dim]` bf16. `block_table`:
    /// `[batch, max_blocks_per_seq]` u32. Returns `[batch, 1, num_heads, head_dim]`.
    ///
    /// Returning `Ok(None)` is valid for backends that can't satisfy the
    /// call's preconditions (e.g. non-contiguous blocks, unsupported page
    /// size); callers fall back to `paged_cache.read + naive softmax`.
    #[allow(clippy::too_many_arguments)]
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
        Ok(None)
    }

    /// Gated DeltaNet chunkwise forward-substitution (prefill path).
    /// Computes `W = (I + A_strict)^{-1} (beta * V_prime)`.
    ///
    /// `a_strict`: `[B, H, C, C]` bf16 (strictly lower-triangular).
    /// `v_prime`: `[B, H, C, dv]` bf16. `beta`: `[B, H, C]` bf16.
    /// Returns `W: [B, H, C, dv]` bf16.
    ///
    /// Backend kernels may advertise narrower envelopes; callers enforce the
    /// shared `C <= 128` cap and implementations can return `None` for shapes
    /// they do not handle.
    fn gdn_forward_substitution(
        &self,
        _a_strict: &Tensor,
        _v_prime: &Tensor,
        _beta: &Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Gated DeltaNet single-token recurrent step (decode fast path).
    ///
    /// `q`, `k`: `[B, H, dk]` bf16. `v`: `[B, H, dv]` bf16.
    /// `beta`, `g`: `[B, H]` bf16. `state`: `[B, H, dk, dv]` bf16,
    /// mutated in place. Returns `out: [B, H, dv]` bf16.
    fn gdn_recurrent_step(
        &self,
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _beta: &Tensor,
        _g: &Tensor,
        _state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Fused GDN chunk-prep kernel (prefill outer recurrence).
    ///
    /// Collapses the 7+ candle op launches (cumsum, decay matrix, exp, masked
    /// scales, v_prime, q_s_scaled, decay_last_col, p_last) inside the
    /// chunkwise recurrence's inner loop into a single CUDA launch per
    /// (chunk × batch × head). Matmuls (KKT, QKT, ks_entry, q_s) stay on
    /// cuBLAS — this kernel consumes their outputs.
    ///
    /// `g`: `[B, H, C]` bf16. `v`: `[B, H, C, dv]` bf16.
    /// `kkt`, `qkt`: `[B, H, C, C]` bf16. `ks_entry`, `q_s`: `[B, H, C, dv]` bf16.
    ///
    /// Returns `(a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last)`:
    ///   - `a_strict`:       `[B, H, C, C]` bf16 — `kkt * decay * strict_lower`
    ///   - `b_mask`:         `[B, H, C, C]` bf16 — `qkt * decay * causal_lower`
    ///   - `v_prime`:        `[B, H, C, dv]` bf16 — `v - ks_entry * p`
    ///   - `q_s_scaled`:     `[B, H, C, dv]` bf16 — `q_s * p`
    ///   - `decay_last_col`: `[B, H, C]` bf16 — `exp(big_g[C-1] - big_g[i])`
    ///   - `p_last`:         `[B, H]` bf16 — `exp(big_g[C-1])`
    ///
    /// Returning `Ok(None)` is valid for backends that can't satisfy the
    /// envelope; callers fall back to the candle-op path.
    fn gdn_chunk_prep(
        &self,
        _g: &Tensor,
        _v: &Tensor,
        _kkt: &Tensor,
        _qkt: &Tensor,
        _ks_entry: &Tensor,
        _q_s: &Tensor,
    ) -> Result<Option<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)>> {
        Ok(None)
    }

    fn gdn_chunk_scan(
        &self,
        _a_strict: &Tensor,
        _b_mask: &Tensor,
        _v_prime: &Tensor,
        _q_s_scaled: &Tensor,
        _beta: &Tensor,
        _decay_last_col: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        Ok(None)
    }

    fn gdn_full_chunk_forward(
        &self,
        _g: &Tensor,
        _v: &Tensor,
        _kkt: &Tensor,
        _qkt: &Tensor,
        _ks_entry: &Tensor,
        _q_s: &Tensor,
        _beta: &Tensor,
        _k_t: &Tensor,
        _state: &mut Tensor,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    fn supports_gdn_gates(&self) -> bool {
        false
    }

    fn supports_gdn_gated_rms_norm(&self) -> bool {
        false
    }

    fn supports_causal_conv1d_update(&self) -> bool {
        false
    }

    fn supports_causal_conv1d_prefill(&self) -> bool {
        false
    }

    /// Fused single-step causal depthwise conv1d + state update + silu.
    ///
    /// Replaces the candle `to_f32 -> cat(state, x) -> sum(window * weight) ->
    /// narrow/contiguous -> silu` chain inside `kiln/gdn/conv` with one CUDA
    /// launch per (batch, channel).
    ///
    /// `x`: `[B, C, 1]` bf16 contiguous. `weight`: `[C, 1, K]` bf16 contiguous
    /// (or `[C, K]` equivalently — width stride = 1). `conv_state`:
    /// `[B, C, K-1]` F32, mutated in place to drop oldest col and append
    /// newest `x`. `kernel_size`: must be 4 for the current CUDA
    /// specialisation.
    ///
    /// Returns `Ok(Some(out))` with `out: [B, C, 1]` F32 (silu-fused), or
    /// `Ok(None)` when the backend declines (wrong dtype, wrong K, envelope
    /// violation, disabled via env kill switch). When `Some`, the caller must
    /// NOT apply `silu` again — it is fused into the kernel epilogue.
    fn causal_conv1d_update(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _conv_state: &mut Tensor,
        _kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Fused prefill causal depthwise conv1d + state update + silu.
    ///
    /// `x`: `[B, C, T]` bf16 contiguous with `T > 1`. `weight`: `[C, 1, K]`
    /// bf16 contiguous (or `[C, K]`). `conv_state`: `[B, C, K-1]` F32,
    /// mutated in place after all outputs have consumed the entry state.
    ///
    /// Returns `Ok(Some(out))` with `out: [B, C, T]` F32 (silu-fused), or
    /// `Ok(None)` when the backend declines. When `Some`, the caller must not
    /// apply `silu` again.
    fn causal_conv1d_prefill(
        &self,
        _x: &Tensor,
        _weight: &Tensor,
        _conv_state: &mut Tensor,
        _kernel_size: usize,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Fused GDN gate computation.
    ///
    /// Collapses the Step-6 `sigmoid(b)` + `-exp(A_log) * softplus(a + dt_bias)`
    /// chain into one CUDA launch. Inputs are bf16 tensors of shape
    /// `[B, T, nv]` for `a`, `b` and `[nv]` for `a_log`, `dt_bias`.
    /// Returns `(beta, g)`, both bf16 `[B, T, nv]`, or `Ok(None)` when
    /// the backend declines (wrong dtype, envelope violation, disabled).
    fn gdn_gates(
        &self,
        _a: &Tensor,
        _b: &Tensor,
        _a_log: &Tensor,
        _dt_bias: &Tensor,
    ) -> Result<Option<(Tensor, Tensor)>> {
        Ok(None)
    }

    /// Fused GDN gated RMSNorm.
    ///
    /// Computes `rms_norm(x, weight) * silu(z)` for Gated DeltaNet outputs.
    /// `x` and `z` are `[B, T, H, D]`, and `weight` is `[D]`.
    /// Returns a tensor with the same shape as `x`. Backends may return the
    /// model dtype directly; the call site already casts to the requested
    /// dtype after reshaping, matching the portable fallback.
    fn gdn_gated_rms_norm(
        &self,
        _x: &Tensor,
        _z: &Tensor,
        _weight: &Tensor,
        _eps: f64,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

/// Pick the right backend for a given candle device.
///
/// On Metal devices, `--features metal` uses Kiln's native candle-metal
/// backend and Metal kernels. The former MLX bridge was removed because it
/// only accelerated attention while paying Candle<->MLX host-copy overheads
/// and bypassing Kiln's Qwen3.5 GDN decode kernels.
pub fn for_device(device: &Device) -> Arc<dyn BackendRuntime> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => Arc::new(cuda::CudaBackend::new(device.clone())),
        #[cfg(feature = "metal")]
        Device::Metal(_) => Arc::new(metal::MetalBackend::new(device.clone())),
        _ => Arc::new(cpu::CpuBackend::new(device.clone())),
    }
}

//! Backend runtime abstraction for Kiln's platform-specific kernels.
//!
//! Most of Kiln's forward pass is expressed as `candle_core::Tensor` operations
//! that run on any candle device (CPU, CUDA, Metal). A small number of ops —
//! FlashAttention-2 forward/paged-decode and the Gated DeltaNet fused
//! recurrent / forward-substitution kernels — have no candle equivalent today
//! and are implemented as platform-specific CUDA (Wave 1) or Metal (Wave 2)
//! kernels. This trait is the seam that lets the forward pass dispatch those
//! four ops to the right backend without threading `#[cfg(feature = "cuda")]`
//! gates through every call site.
//!
//! Design principles:
//!
//! 1. **Every kernel method returns `Option<Tensor>`.** `Ok(None)` means "this
//!    backend declines — fall back to the portable candle path". The existing
//!    `try_flash_attn_paged_decode` already uses this pattern (returns `None`
//!    on precondition misses like non-contiguous blocks); the trait makes it
//!    universal.
//! 2. **`supports_*` hints** let the caller skip preamble work (e.g., a
//!    `contiguous()` copy before the trait call) when the backend will decline
//!    anyway.
//! 3. **No candle-op methods in the trait.** matmul / softmax / rms_norm /
//!    rope / conv1d / silu / sigmoid all run on candle Metal and CUDA today
//!    and stay as direct `Tensor` calls.
//! 4. **No graph capture in the trait yet.** `cuda_graph.rs` stays
//!    `#[cfg(feature = "cuda")]`-gated. A Metal analog (if it proves valuable)
//!    will be added in Phase 2; MLX's `mx.compile` lands in Wave 2.

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::sync::Arc;

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

/// A backend runtime providing platform-specific fast paths for the handful of
/// ops candle doesn't cover. Methods default to "decline" (`Ok(None)`) so a
/// minimal `CpuBackend` works anywhere.
pub trait BackendRuntime: Send + Sync + std::fmt::Debug {
    /// Human-readable backend name (`"cuda"`, `"metal"`, `"cpu"`). Surfaced in
    /// `/health` and logs.
    fn name(&self) -> &'static str;

    /// The candle `Device` this backend drives. All tensors passed to trait
    /// methods must live on this device.
    fn device(&self) -> &Device;

    /// If false, callers skip the preamble (e.g., contiguous copies) and go
    /// straight to the portable fallback.
    fn supports_flash_attn_prefill(&self) -> bool {
        false
    }

    /// Hint for the paged-decode fast path. Non-CUDA backends return `false`
    /// by default; Wave 2 Metal SDPA implementation can flip this to true.
    fn supports_flash_attn_paged_decode(&self) -> bool {
        false
    }

    fn supports_gdn_forward_substitution(&self) -> bool {
        false
    }

    fn supports_gdn_recurrent_step(&self) -> bool {
        false
    }

    /// FlashAttention-2 forward, used for prefill (seq_len > 1) when no KV
    /// cache is present.
    ///
    /// - `q`, `k`, `v`: `[batch, seq_len, num_heads, head_dim]` bf16,
    ///   contiguous. Caller is responsible for GQA-expanding K/V to match
    ///   Q's head count.
    /// - Returns `[batch, seq_len, num_heads, head_dim]` bf16.
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

    /// FlashAttention-2 paged decode (single query token, attention over a
    /// paged KV pool).
    ///
    /// - `q`: `[batch, 1, num_heads, head_dim]` bf16, contiguous.
    /// - `k_pool`, `v_pool`: `[total_slots, num_kv_heads, head_dim]` bf16.
    /// - `block_table`: `[batch, max_blocks_per_seq]` u32.
    /// - Returns `[batch, 1, num_heads, head_dim]` bf16.
    ///
    /// `Ok(None)` is the proper response when the backend can't satisfy
    /// preconditions (e.g., non-contiguous block table, unsupported page
    /// size). Callers fall back to `paged_cache.read(...) + naive softmax`.
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

    /// Gated DeltaNet chunkwise forward-substitution (prefill).
    /// Computes `W = (I + A_strict)^{-1} (beta * V_prime)` via forward
    /// substitution, fused into a single kernel block per `(batch, head)`.
    ///
    /// - `a_strict`: `[B, H, C, C]` bf16 (strictly lower-triangular).
    /// - `v_prime`:  `[B, H, C, dv]` bf16.
    /// - `beta`:     `[B, H, C]` bf16.
    /// - Returns `W: [B, H, C, dv]` bf16.
    ///
    /// Current kernel envelope: `C <= 128`, `dv <= 1024`. Outside the envelope
    /// return `Ok(None)` and let the caller run the per-token fallback.
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
    /// - `q`, `k`: `[B, H, dk]` bf16. `v`: `[B, H, dv]` bf16.
    /// - `beta`, `g`: `[B, H]` bf16.
    /// - `state`: `[B, H, dk, dv]` bf16, mutated in place.
    /// - Returns `out: [B, H, dv]` bf16.
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
}

/// Pick the right backend for a given candle device.
///
/// CPU and Metal devices both get a `CpuBackend` today — its methods return
/// `Ok(None)`, so the forward pass falls back to the portable candle
/// composition. Phase 2 will add a `MetalBackend` that implements the
/// attention / GDN ops via MSL kernels or `candle_nn::ops::sdpa`.
pub fn for_device(device: &Device) -> Arc<dyn BackendRuntime> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => Arc::new(cuda::CudaBackend::new(device.clone())),
        _ => Arc::new(cpu::CpuBackend::new(device.clone())),
    }
}

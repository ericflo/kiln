//! Fused Linear Cross-Entropy (FLCE) — kt-typed building blocks.
//!
//! Computes cross-entropy loss over a projected head without materializing
//! the full `[T, V]` logits tensor. The mathematical trick is the log-sum-exp
//! identity: `log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))`, which
//! lets us reduce over the vocab dimension in chunks while keeping only
//! per-row max + running sum-exp + the gathered correct logit.
//!
//! # (#1082) candle-free
//!
//! This crate is now 100% candle-free (the THIRD kernel-crate candle drop,
//! after `kiln-opd-loss-kernel` and `kiln-rmsnorm-kernel`). It ships only
//! the pure-`kiln_tensor` / `kiln_autograd` building blocks:
//!
//! - [`kt_api`] — the kt-typed forward
//!   ([`kt_api::fused_linear_cross_entropy_phase_b_kt`]) + manual backward
//!   ([`kt_api::fused_linear_cross_entropy_phase_b_backward_kt`]) over
//!   `kiln_tensor::Tensor` ops, plus the kt-typed
//!   [`kt_api::FlceMatmulProviderKt`] provider trait.
//! - [`kt_tape`] — the kt-tape entries
//!   ([`fused_linear_cross_entropy_phase_b_via_kt_tape`] and
//!   [`fused_linear_cross_entropy_phase_b_unit_grad_via_kt_tape`]) that record
//!   the FLCE backward onto a `kiln_autograd::Tape`.
//!
//! The candle-typed glue that the SFT/FLCE trainer needs — the pure-candle
//! Phase A reference, the Phase B candle `CustomOp1`, the `KtForwardOp1`
//! kt-forward-op shim, and the kt-tape production-caller adapter — moved UP
//! into `kiln-train::flce_candle_shim`, which legitimately keeps
//! `candle-core` (and already depends on `kiln-kt-bridge`). Those moved
//! paths call the kt entries this crate re-exports. The relocation kept the
//! FLCE math byte-identical — only the crate location changed.
//!
//! # Phase A vs Phase B (history)
//!
//! Phase A was a pure-candle reference that chunks the forward over the
//! vocab dim; its backward flowed through candle autograd, retaining chunk
//! intermediates (`logits_chunk`, `shifted`, `shifted.exp()`) for the entire
//! forward — at T=8192 with V=248320 this was ~23 GiB held live across 61
//! vocab chunks, which OOMed SFT on A6000 (see
//! `docs/audits/PHASE10_MODE_B_TRACE.md`). Phase B replaced the autograd
//! graph with a manual-backward that recomputes each vocab chunk on the fly,
//! storing only the scalar loss. The kt-typed forward/backward here
//! implement the same chunked log-sum-exp math, numerically equivalent to
//! the candle Phase A/B reference up to floating-point associativity in the
//! chunked reduction. The candle Phase A/B reference now lives in
//! `kiln-train::flce_candle_shim` as the parity oracle + `KILN_FLCE_PHASE_A`
//! escape hatch.
//!
//! # Target
//!
//! Phase 10 enables long-context SFT on Qwen3.5-4B on A6000. Preflight
//! (PR #235) showed the head materializes a `[T, V]` F32 logits tensor
//! that dominates peak VRAM at `T >= 8192` (OOM before reaching head).
//! FLCE is the prerequisite, not an optimization.

pub mod kt_api;
pub use kt_api::{
    FlceError, FlceMatmulProviderKt, FlceProviderKt,
    fused_linear_cross_entropy_phase_b_backward_unit_grad_kt,
    fused_linear_cross_entropy_phase_b_kt,
};

/// Phase 6a/CP-4 (#1082): parallel kt-tape entry that records the FLCE
/// backward onto a `kiln_autograd::Tape` directly (no candle CustomOp1
/// wrapper). Same kt-typed forward and backward, same envelope. See
/// `kt_tape.rs` for the pilot port rationale.
pub mod kt_tape;
pub use kt_tape::{
    CudaFlcePhaseBBackward, fused_linear_cross_entropy_phase_b_unit_grad_via_kt_tape,
    fused_linear_cross_entropy_phase_b_via_kt_tape,
};

/// Default chunk size along the vocab dimension.
///
/// The preflight math picked 4096 as a reasonable balance between kernel
/// launch overhead and peak intermediate footprint. For Qwen3.5-4B with
/// V=151936, a chunk of 4096 means ~37 chunks per forward — small enough
/// that per-chunk launch cost is absorbed.
///
/// Re-exported by the candle-typed surface in
/// `kiln-train::flce_candle_shim` so both `kiln_flce_kernel::DEFAULT_CHUNK_SIZE`
/// and `flce_candle_shim::DEFAULT_CHUNK_SIZE` resolve to the same value.
pub const DEFAULT_CHUNK_SIZE: usize = 4096;

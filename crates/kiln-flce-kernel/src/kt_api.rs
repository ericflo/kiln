//! `kiln_tensor::Tensor`-typed parallel surface for kiln-flce-kernel.
//!
//! # Status
//!
//! Phase 7 prep — same migration pattern as kiln-flash-attn,
//! kiln-conv1d-kernel, kiln-rmsnorm-kernel, kiln-gdn-kernel, and
//! kiln-marlin-gemm (see #1082 line 322 "kernel-crate kt-API ports").
//!
//! Unlike those crates, kiln-flce-kernel does **not** yet have any
//! raw-CUDA FFI to wrap: today's Phase A/B forward+backward run on
//! `candle_core::Tensor` ops (matmul, exp, max_keepdim, ...). The
//! kt-typed surface this module defines is therefore the *migration
//! target*: when Phase B's `forward_loss` / `backward_dhidden` are
//! eventually re-implemented over `kiln_tensor::Tensor` (the natural
//! next step once the [`crate::FlceMatmulProvider`] external
//! integrations migrate), the kt-typed entry points + provider trait
//! below are the public surface they will plug into.
//!
//! Until then this module ships only the parallel **trait** + **error
//! type**. Callers that already have a `kiln_tensor::Tensor`-typed
//! matmul implementation (the kiln-vulkan-kernel head-chunk path will,
//! per #1082 line 614) can implement [`FlceMatmulProviderKt`] today;
//! the candle-typed [`crate::FlceMatmulProvider`] stays in place for
//! the existing call sites and continues to compile until the full
//! Phase B rewrite lands.
//!
//! # Design — why no `KtTensor` entry point yet
//!
//! `fused_linear_cross_entropy_phase_b_kt` would need a candle-free
//! re-implementation of the chunked log-sum-exp reduction. That is the
//! bulk of the Phase B rewrite (~900 lines including the `CustomOp1`
//! adapter) and is intentionally out of scope here; this PR ships the
//! migration scaffolding so the rewrite can land incrementally without
//! breaking the candle-typed path.

use std::sync::Arc;

use kiln_tensor::Tensor as KtTensor;

/// Error type for the kiln-tensor-typed FLCE surface.
///
/// Mirrors `kiln-flash-attn::kt_api::FlashAttnError` — kept separate
/// from `anyhow::Error` (the candle-typed surface's error) so Phase 7
/// can delete candle without rewriting any kt-typed call site.
#[derive(Debug)]
pub enum FlceError {
    Msg(String),
}

impl std::fmt::Display for FlceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlceError::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for FlceError {}

impl FlceError {
    pub fn msg(s: impl Into<String>) -> Self {
        FlceError::Msg(s.into())
    }
}

/// `kiln_tensor::Tensor`-typed parallel of [`crate::FlceMatmulProvider`].
///
/// The candle-typed trait lives in `lib.rs` and continues to be used
/// by the production Phase B path. This kt-typed twin lets backend
/// crates (kiln-vulkan-kernel, kiln-mps, future kiln-cuda-kernel)
/// implement the chunk matmul over `kiln_tensor::Tensor` directly,
/// without having to round-trip through candle storage. When Phase B
/// is rewritten to run over `kiln_tensor::Tensor` end-to-end, the FLCE
/// chunk loop will call this trait instead of the candle-typed one.
///
/// # Contract
///
/// `lhs` is `[active, hidden]` F32. `full_rhs` is the original
/// `[hidden, vocab_size]` head_t in its original dtype. The chunk to
/// compute is `full_rhs[:, chunk_start .. chunk_start + chunk_len]`.
/// Expected output shape is `[active, chunk_len]` F32.
///
/// Returning `Ok(None)` is a signal that this provider declines the
/// chunk — the FLCE driver falls back to its native compute path for
/// that specific chunk. Returning `Err(_)` aborts the FLCE forward.
///
/// # Why `full_rhs` rather than a pre-narrowed chunk
///
/// Same reason as the candle-typed trait (see [`crate::FlceMatmulProvider`]):
/// threading the un-narrowed `full_rhs` + `(chunk_start, chunk_len)`
/// through to the provider lets implementations upload `full_rhs` to
/// a device buffer once and reuse the same buffer for every chunk
/// via offset-aware dispatch — the alternative (give the provider
/// the already-narrowed rhs Tensor) costs a fresh device-buffer
/// upload per chunk when the underlying per-tensor cache keys on
/// `TensorId`.
pub trait FlceMatmulProviderKt: Send + Sync + std::fmt::Debug {
    fn chunk_matmul(
        &self,
        lhs: &KtTensor,
        full_rhs: &KtTensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<KtTensor>, FlceError>;
}

/// Convenience boxed type used by future `_with_provider_kt` entry
/// points (analogous to [`crate::FlceProvider`]).
pub type FlceProviderKt = Arc<dyn FlceMatmulProviderKt>;

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: the kt-typed trait + error type compile and are
    /// Send + Sync. No behavioral assertions — there is no kt-typed
    /// entry point to exercise yet (see module docs).
    #[derive(Debug)]
    struct DeclineAllProvider;

    impl FlceMatmulProviderKt for DeclineAllProvider {
        fn chunk_matmul(
            &self,
            _lhs: &KtTensor,
            _full_rhs: &KtTensor,
            _chunk_start: usize,
            _chunk_len: usize,
        ) -> Result<Option<KtTensor>, FlceError> {
            Ok(None)
        }
    }

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn kt_provider_trait_compiles_and_is_send_sync() {
        _assert_send_sync::<DeclineAllProvider>();
        _assert_send_sync::<FlceProviderKt>();

        let provider: FlceProviderKt = Arc::new(DeclineAllProvider);
        // Smoke-format via Debug so Rust doesn't elide the trait object.
        let _ = format!("{provider:?}");
    }

    #[test]
    fn flce_error_displays_message() {
        let e = FlceError::msg("test message");
        assert_eq!(format!("{e}"), "test message");
        // std::error::Error impl is reachable.
        let _: &dyn std::error::Error = &e;
    }
}

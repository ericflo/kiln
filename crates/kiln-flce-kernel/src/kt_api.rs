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
//! Until then this module ships:
//!
//! 1. [`FlceMatmulProviderKt`] — kt-typed parallel of the candle
//!    [`crate::FlceMatmulProvider`] trait.
//! 2. [`FlceError`] — kt-typed error, independent of candle / anyhow.
//! 3. [`fused_linear_cross_entropy_phase_b_kt`] — kt-typed entry
//!    point **stub** that validates shapes and returns
//!    [`FlceError::NotYetImplemented`]. The signature is stable; the
//!    body is filled in by the Phase B kt-rewrite. Documenting the
//!    name now lets downstream call sites be ported in advance behind
//!    a feature flag.
//!
//! # Design — why a stub entry point
//!
//! `fused_linear_cross_entropy_phase_b_kt` would need a candle-free
//! re-implementation of the chunked log-sum-exp reduction. That is
//! the bulk of the Phase B rewrite (~900 lines including the
//! `CustomOp1` adapter) and is intentionally out of scope here. By
//! shipping the stub function, the next sub-agent doing the kt-rewrite
//! only needs to fill in the body — the public signature is already
//! frozen, doc-commented, and unit-tested (shape validation only).

use std::sync::Arc;

use kiln_tensor::Tensor as KtTensor;

/// Error type for the kiln-tensor-typed FLCE surface.
///
/// Mirrors `kiln-flash-attn::kt_api::FlashAttnError` — kept separate
/// from `anyhow::Error` (the candle-typed surface's error) so Phase 7
/// can delete candle without rewriting any kt-typed call site.
#[derive(Debug)]
pub enum FlceError {
    /// Generic message error for shape / dtype validation failures
    /// in the kt-typed entry points.
    Msg(String),
    /// The kt-typed entry point exists but its body has not yet been
    /// implemented. Returned today by
    /// [`fused_linear_cross_entropy_phase_b_kt`]; will be removed once
    /// the kt-typed Phase B forward/backward land.
    NotYetImplemented(&'static str),
}

impl std::fmt::Display for FlceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlceError::Msg(m) => f.write_str(m),
            FlceError::NotYetImplemented(name) => write!(
                f,
                "kt-flce: {name} is not yet implemented; use the candle-typed entry point",
            ),
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

/// kt-typed entry point for FLCE Phase B — **stub**.
///
/// Validates `hidden` and `head_t` shapes against the production
/// FLCE contract and returns [`FlceError::NotYetImplemented`]. When
/// the Phase B kt-rewrite lands, this function's body will be filled
/// in (re-implementing the chunked log-sum-exp reduction over
/// `kiln_tensor::Tensor` ops) without breaking the signature.
///
/// # Shape contract (matches the candle-typed entry point)
///
/// - `hidden`: `[1, seq_len, hidden_size]` post-final-RMSNorm
///   hidden states.
/// - `head_t`: `[hidden_size, vocab_size]` transposed lm_head weight
///   (matches kiln's `embed_tokens_t` layout — i.e. `W.T` where `W`
///   is the standard `[vocab_size, hidden_size]` lm_head).
/// - `input_ids`: token ids; `input_ids[1..]` are next-token targets
///   for `logits[..seq_len-1]`.
/// - `label_mask`: `[seq_len]` booleans; only positions where
///   `label_mask[i+1]` is true contribute to the loss.
/// - `chunk_size`: chunk size along the vocab dim.
///
/// # Returns
///
/// Today: always returns [`FlceError::NotYetImplemented`] after
/// passing shape validation. After the kt-rewrite: a scalar F32
/// [`KtTensor`] holding the mean cross-entropy over active positions.
pub fn fused_linear_cross_entropy_phase_b_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
) -> Result<KtTensor, FlceError> {
    let seq_len = input_ids.len();
    if label_mask.len() != seq_len {
        return Err(FlceError::msg(format!(
            "kt-flce: label_mask length {} does not match input_ids length {}",
            label_mask.len(),
            seq_len,
        )));
    }
    if chunk_size == 0 {
        return Err(FlceError::msg("kt-flce: chunk_size must be > 0"));
    }
    let hidden_dims = hidden.shape();
    if hidden_dims.len() != 3 {
        return Err(FlceError::msg(format!(
            "kt-flce: hidden must be 3-D [1, seq_len, hidden_size]; got {hidden_dims:?}",
        )));
    }
    if hidden_dims[0] != 1 {
        return Err(FlceError::msg(format!(
            "kt-flce: hidden batch dim must be 1; got {hidden_dims:?}",
        )));
    }
    let head_dims = head_t.shape();
    if head_dims.len() != 2 {
        return Err(FlceError::msg(format!(
            "kt-flce: head_t must be 2-D [hidden_size, vocab_size]; got {head_dims:?}",
        )));
    }
    if hidden_dims[2] != head_dims[0] {
        return Err(FlceError::msg(format!(
            "kt-flce: hidden hidden_size {} != head_t hidden_size {}",
            hidden_dims[2], head_dims[0],
        )));
    }
    // All shape preconditions pass; the body is filled in by the
    // Phase B kt-rewrite.
    Err(FlceError::NotYetImplemented(
        "fused_linear_cross_entropy_phase_b_kt",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{DType as KtDType, Tensor as KtTensorCtor};

    /// Smoke test: the kt-typed trait + error type compile and are
    /// Send + Sync.
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

    #[test]
    fn flce_error_not_yet_implemented_displays_name() {
        let e = FlceError::NotYetImplemented("foo_kt");
        let s = format!("{e}");
        assert!(s.contains("foo_kt"), "got: {s}");
        assert!(s.contains("not yet implemented"), "got: {s}");
    }

    fn dummy_hidden(seq_len: usize, hidden_size: usize) -> KtTensor {
        let n = seq_len * hidden_size;
        let data = vec![0.0f32; n];
        KtTensorCtor::from_vec(data, vec![1, seq_len, hidden_size]).expect("alloc hidden")
    }

    fn dummy_head_t(hidden_size: usize, vocab_size: usize) -> KtTensor {
        let n = hidden_size * vocab_size;
        let data = vec![0.0f32; n];
        KtTensorCtor::from_vec(data, vec![hidden_size, vocab_size]).expect("alloc head")
    }

    #[test]
    fn fused_linear_cross_entropy_phase_b_kt_validates_chunk_size_zero() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let ids = vec![0u32; 4];
        let mask = vec![true; 4];
        let err = fused_linear_cross_entropy_phase_b_kt(&h, &w, &ids, &mask, 0).unwrap_err();
        assert!(matches!(err, FlceError::Msg(_)));
        let s = format!("{err}");
        assert!(s.contains("chunk_size must be > 0"), "got: {s}");
        // Avoid an unused-import warning on DType when no other test
        // references it.
        let _: KtDType = KtDType::F32;
    }

    #[test]
    fn fused_linear_cross_entropy_phase_b_kt_validates_mask_length() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let ids = vec![0u32; 4];
        let mask = vec![true; 3]; // wrong length
        let err = fused_linear_cross_entropy_phase_b_kt(&h, &w, &ids, &mask, 4).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("label_mask length"), "got: {s}");
    }

    #[test]
    fn fused_linear_cross_entropy_phase_b_kt_validates_hidden_rank() {
        // 2-D hidden — must be 3-D.
        let h = KtTensor::from_vec(vec![0.0f32; 16], vec![4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let ids = vec![0u32; 4];
        let mask = vec![true; 4];
        let err = fused_linear_cross_entropy_phase_b_kt(&h, &w, &ids, &mask, 4).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden must be 3-D"), "got: {s}");
    }

    #[test]
    fn fused_linear_cross_entropy_phase_b_kt_validates_hidden_vs_head_hidden_size() {
        // hidden_size=8 but head_t hidden_size=4 — mismatch.
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(4, 16);
        let ids = vec![0u32; 4];
        let mask = vec![true; 4];
        let err = fused_linear_cross_entropy_phase_b_kt(&h, &w, &ids, &mask, 4).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden hidden_size"), "got: {s}");
    }

    #[test]
    fn fused_linear_cross_entropy_phase_b_kt_stub_returns_not_yet_implemented_on_valid_shapes() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let ids = vec![0u32; 4];
        let mask = vec![true; 4];
        let err = fused_linear_cross_entropy_phase_b_kt(&h, &w, &ids, &mask, 4).unwrap_err();
        assert!(
            matches!(err, FlceError::NotYetImplemented(_)),
            "expected NotYetImplemented, got: {err}"
        );
    }
}

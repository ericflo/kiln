//! `kiln_tensor::Tensor`-typed surface alongside the candle-typed
//! OPD top-K reverse-KL API.
//!
//! # Status
//!
//! Phase 7 prep — same migration pattern as kiln-flce-kernel,
//! kiln-flash-attn, kiln-conv1d-kernel, kiln-rmsnorm-kernel,
//! kiln-gdn-kernel, and kiln-marlin-gemm (see #1082 line 322
//! "kernel-crate kt-API ports").
//!
//! Like kiln-flce-kernel, kiln-opd-loss-kernel today's Phase A path
//! runs entirely on `candle_core::Tensor` ops (gather + matmul + log-
//! softmax) and Phase B wraps the analytic backward in a candle
//! [`CustomOp1`]. The raw CUDA FFI in `phase_b.rs` is only the inner
//! kernel dispatch and is still tightly coupled to candle storage
//! conversions (`CudaStorage`, `Storage::Cuda`, `BackpropOp::none`).
//!
//! The kt-typed surface this module defines is therefore the
//! *migration target*: when Phase B's `forward_inner` /
//! `backward_inner` are eventually re-implemented over
//! `kiln_tensor::Tensor` (the natural next step once
//! `kiln-flce-kernel::kt_api` lands its Phase B kt-rewrite), the
//! kt-typed entry points below are the public surface they will plug
//! into.
//!
//! Until then this module ships:
//!
//! 1. [`OpdLossError`] — kt-typed error, independent of candle /
//!    anyhow.
//! 2. [`opd_top_k_reverse_kl_kt`] — kt-typed scalar-mean entry
//!    point **stub** that validates shapes and returns
//!    [`OpdLossError::NotYetImplemented`].
//! 3. [`opd_top_k_reverse_kl_per_position_kt`] — kt-typed per-
//!    position entry point **stub**, same shape validation.
//!
//! Documenting both names now lets downstream call sites (the
//! kiln-train OPD trainer in particular) be ported in advance behind
//! a feature flag — the bodies are filled in by the Phase B
//! kt-rewrite sub-task. (#1082)
//!
//! # Design — why a stub
//!
//! Filling in either body requires a candle-free re-implementation
//! of the per-token gather + batched matmul + log-softmax reduction,
//! plus a kt-typed parallel of the [`crate::phase_b::OpdLossCustomOp`]
//! adapter (manual backward over kiln-autograd, which is still
//! evolving — see PR #1078/#1079 for the SFT/GRPO step structure).
//! That is the bulk of the Phase B rewrite (~1500 lines including
//! the autograd adapter) and is intentionally out of scope here. By
//! shipping the stub functions, the next sub-agent doing the
//! kt-rewrite only needs to fill in the body — the public signature
//! is already frozen, doc-commented, and unit-tested (shape
//! validation only).

use kiln_tensor::Tensor as KtTensor;

/// Error type for the kiln-tensor-typed OPD loss surface.
///
/// Mirrors `kiln-flce-kernel::kt_api::FlceError` — kept separate
/// from `anyhow::Error` (the candle-typed surface's error) so Phase 7
/// can delete candle without rewriting any kt-typed call site.
#[derive(Debug)]
pub enum OpdLossError {
    /// Generic message error for shape / dtype validation failures
    /// in the kt-typed entry points.
    Msg(String),
    /// The kt-typed entry point exists but its body has not yet been
    /// implemented. Returned today by [`opd_top_k_reverse_kl_kt`] and
    /// [`opd_top_k_reverse_kl_per_position_kt`]; will be removed once
    /// the kt-typed Phase B forward/backward land.
    NotYetImplemented(&'static str),
}

impl std::fmt::Display for OpdLossError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpdLossError::Msg(m) => f.write_str(m),
            OpdLossError::NotYetImplemented(name) => write!(
                f,
                "kt-opd-loss: {name} is not yet implemented; use the candle-typed entry point",
            ),
        }
    }
}

impl std::error::Error for OpdLossError {}

impl OpdLossError {
    pub fn msg(s: impl Into<String>) -> Self {
        OpdLossError::Msg(s.into())
    }
}

/// Validate the shape / dtype contract on the kt-typed entry points
/// and return the `(T, H, V, T_active, K)` quintuple for downstream
/// code. Mirrors the candle-typed `crate::validate_inputs` but uses
/// `kiln_tensor::Tensor::shape()` (slice) rather than candle's
/// `Tensor::dims()` (slice), and returns [`OpdLossError`] instead of
/// `anyhow::Error`.
fn validate_inputs_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<(usize, usize, usize, usize, usize), OpdLossError> {
    let hidden_dims = hidden.shape();
    if hidden_dims.len() != 3 {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: hidden must be 3-D [1, T, H]; got {hidden_dims:?}",
        )));
    }
    if hidden_dims[0] != 1 {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: hidden batch dim must be 1 (kiln trainer convention); got {hidden_dims:?}",
        )));
    }
    let seq_len = hidden_dims[1];
    let hidden_size = hidden_dims[2];

    let head_dims = head_t.shape();
    if head_dims.len() != 2 {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: head_t must be 2-D [H, V]; got {head_dims:?}",
        )));
    }
    if head_dims[0] != hidden_size {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: hidden_size mismatch: hidden has H={hidden_size} but head_t has H={}",
            head_dims[0],
        )));
    }
    let vocab_size = head_dims[1];

    if label_mask.len() != seq_len {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: label_mask length {} does not match T {seq_len}",
            label_mask.len(),
        )));
    }
    if top_k == 0 {
        return Err(OpdLossError::msg("kt-opd-loss: top_k must be > 0"));
    }

    let active_count = label_mask.iter().filter(|&&m| m).count();
    let expected_logits = active_count * top_k;
    if teacher_topk_indices.len() != expected_logits {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: teacher_topk_indices length {} != T_active * K = {active_count} * {top_k} = {expected_logits}",
            teacher_topk_indices.len(),
        )));
    }
    if teacher_topk_logprobs.len() != expected_logits {
        return Err(OpdLossError::msg(format!(
            "kt-opd-loss: teacher_topk_logprobs length {} != T_active * K = {active_count} * {top_k} = {expected_logits}",
            teacher_topk_logprobs.len(),
        )));
    }
    for (i, &idx) in teacher_topk_indices.iter().enumerate() {
        if (idx as usize) >= vocab_size {
            return Err(OpdLossError::msg(format!(
                "kt-opd-loss: teacher_topk_indices[{i}] = {idx} >= vocab_size {vocab_size}",
            )));
        }
    }

    Ok((seq_len, hidden_size, vocab_size, active_count, top_k))
}

/// kt-typed entry point for OPD scalar-mean reverse-KL — **stub**.
///
/// Validates `hidden`, `head_t`, and the teacher top-K tensors
/// against the production OPD contract and returns
/// [`OpdLossError::NotYetImplemented`]. When the Phase B kt-rewrite
/// lands, this function's body will be filled in (re-implementing
/// the gather + matmul + KL reduction over `kiln_tensor::Tensor`
/// ops + raw CUDA FFI) without breaking the signature.
///
/// # Shape contract (matches the candle-typed
/// [`crate::opd_top_k_reverse_kl`] entry point)
///
/// - `hidden`: `[1, T, H]` student hidden states.
/// - `head_t`: `[H, V]` transposed LM head (matches kiln's
///   `embed_tokens_t` layout).
/// - `teacher_topk_indices`: `[T_active * K]` row-major flat — the
///   teacher's top-K vocab indices at each active position.
/// - `teacher_topk_logprobs`: `[T_active * K]` row-major flat —
///   matching log-probabilities at those indices (log_softmax over
///   the full teacher vocab).
/// - `label_mask`: `[T]` booleans; the position-`t` logit contributes
///   when `label_mask[t]` is true. The number of active positions
///   must equal `T_active` and the order of active positions
///   left-to-right in `hidden` must match the row order of
///   `teacher_topk_indices`.
/// - `top_k`: K — the teacher's support size.
///
/// # Returns
///
/// Today: always returns [`OpdLossError::NotYetImplemented`] after
/// passing shape validation. After the kt-rewrite: a scalar F32
/// [`KtTensor`] holding the mean reverse KL over active positions.
pub fn opd_top_k_reverse_kl_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<KtTensor, OpdLossError> {
    let _ = validate_inputs_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;
    // All shape preconditions pass; the body is filled in by the
    // Phase B kt-rewrite.
    Err(OpdLossError::NotYetImplemented("opd_top_k_reverse_kl_kt"))
}

/// kt-typed entry point for OPD per-position reverse-KL — **stub**.
///
/// Same contract as [`opd_top_k_reverse_kl_kt`] but emits the
/// `[T_active]` f32 per-position KL vector instead of the scalar
/// mean. Used by the GRPO importance-sampling advantage construction
/// (`A_t = -KL_t`, §3.1 step 4 of the OPD grand plan).
///
/// # Returns
///
/// Today: always returns [`OpdLossError::NotYetImplemented`] after
/// passing shape validation. After the kt-rewrite: a 1-D F32
/// [`KtTensor`] of shape `[T_active]` holding the per-position
/// reverse KL.
pub fn opd_top_k_reverse_kl_per_position_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<KtTensor, OpdLossError> {
    let _ = validate_inputs_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;
    // All shape preconditions pass; the body is filled in by the
    // Phase B kt-rewrite.
    Err(OpdLossError::NotYetImplemented(
        "opd_top_k_reverse_kl_per_position_kt",
    ))
}

/// kt-typed parallel of [`crate::PerPositionMetrics`].
///
/// Three parallel `[T_active]` arrays carrying the per-position
/// distribution-alignment diagnostics (§3.8 of the OPD grand plan).
/// Lengths are all equal to the number of active positions.
///
/// Mirrors the candle-typed [`crate::PerPositionMetrics`] struct so
/// the Phase B kt-rewrite has a stable target type to populate.
#[derive(Debug, Clone, Default)]
pub struct PerPositionMetricsKt {
    /// Per-position student entropy over the teacher's K support, in
    /// nats. Higher = student less concentrated.
    pub student_entropy: Vec<f32>,
    /// Per-position teacher entropy over the same K support, in nats.
    pub teacher_entropy: Vec<f32>,
    /// Per-position reverse KL (same value the loss kernel emits).
    pub reverse_kl: Vec<f32>,
}

impl PerPositionMetricsKt {
    /// `[T_active]` of `|H(q) - H(p)|` per position.
    pub fn entropy_gap_vec(&self) -> Vec<f32> {
        self.student_entropy
            .iter()
            .zip(self.teacher_entropy.iter())
            .map(|(p, q)| (q - p).abs())
            .collect()
    }

    /// Mean over active positions of `|H(q) - H(p)|`. The scalar §3.8
    /// diagnostic.
    pub fn mean_entropy_gap(&self) -> f64 {
        if self.student_entropy.is_empty() {
            return 0.0;
        }
        let n = self.student_entropy.len() as f64;
        self.student_entropy
            .iter()
            .zip(self.teacher_entropy.iter())
            .map(|(p, q)| (q - p).abs() as f64)
            .sum::<f64>()
            / n
    }

    /// Mean per-position KL — matches what the trainer already tracks,
    /// but recomputed here so the metrics call doesn't depend on a
    /// separate loss pass.
    pub fn mean_reverse_kl(&self) -> f64 {
        if self.reverse_kl.is_empty() {
            return 0.0;
        }
        let n = self.reverse_kl.len() as f64;
        self.reverse_kl.iter().map(|&v| v as f64).sum::<f64>() / n
    }
}

/// kt-typed entry point for per-position distribution-alignment
/// metrics over the teacher's K support — **stub**.
///
/// Validates `hidden`, `head_t`, and the teacher top-K tensors
/// against the production OPD contract (same checks as
/// [`opd_top_k_reverse_kl_kt`]) and returns
/// [`OpdLossError::NotYetImplemented`]. When the Phase B kt-rewrite
/// lands, this function's body will be filled in (re-implementing
/// the gather + matmul + entropy / KL reduction over
/// `kiln_tensor::Tensor` ops + raw CUDA FFI) without breaking the
/// signature.
///
/// # Shape contract (matches the candle-typed
/// [`crate::compute_per_position_metrics`] entry point)
///
/// - `hidden`: `[1, T, H]` student hidden states.
/// - `head_t`: `[H, V]` transposed LM head.
/// - `teacher_topk_indices`: `[T_active * K]` row-major flat.
/// - `teacher_topk_logprobs`: `[T_active * K]` row-major flat.
/// - `label_mask`: `[T]` booleans selecting active positions.
/// - `top_k`: K — the teacher's support size.
///
/// # Returns
///
/// Today: always returns [`OpdLossError::NotYetImplemented`] after
/// passing shape validation. After the kt-rewrite: a populated
/// [`PerPositionMetricsKt`] with the three diagnostic vectors.
pub fn compute_per_position_metrics_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<PerPositionMetricsKt, OpdLossError> {
    let _ = validate_inputs_kt(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;
    // All shape preconditions pass; the body is filled in by the
    // Phase B kt-rewrite.
    Err(OpdLossError::NotYetImplemented(
        "compute_per_position_metrics_kt",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{DType as KtDType, Tensor as KtTensorCtor};

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
    fn opd_loss_error_displays_message() {
        let e = OpdLossError::msg("test message");
        assert_eq!(format!("{e}"), "test message");
        // std::error::Error impl is reachable.
        let _: &dyn std::error::Error = &e;
    }

    #[test]
    fn opd_loss_error_not_yet_implemented_displays_name() {
        let e = OpdLossError::NotYetImplemented("foo_kt");
        let s = format!("{e}");
        assert!(s.contains("foo_kt"), "got: {s}");
        assert!(s.contains("not yet implemented"), "got: {s}");
        // Avoid an unused-import warning on DType when no other test
        // references it.
        let _: KtDType = KtDType::F32;
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_hidden_rank() {
        // 2-D hidden — must be 3-D.
        let h = KtTensor::from_vec(vec![0.0f32; 16], vec![4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden must be 3-D"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_hidden_batch_dim() {
        // hidden batch dim must be 1.
        let h = KtTensor::from_vec(vec![0.0f32; 32], vec![2, 4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("batch dim must be 1"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_head_rank() {
        let h = dummy_hidden(4, 4);
        // 3-D head — must be 2-D.
        let w = KtTensor::from_vec(vec![0.0f32; 32], vec![4, 2, 4]).expect("alloc w");
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("head_t must be 2-D"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_hidden_vs_head_hidden_size() {
        // hidden_size=8 but head_t hidden_size=4 — mismatch.
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(4, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden_size mismatch"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_label_mask_length() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 3]; // wrong length
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("label_mask length"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_top_k_zero() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 0).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("top_k must be > 0"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_indices_length() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        // T_active = 4, K = 2, so expected length = 8; pass 6.
        let idx = vec![0u32; 6];
        let lp = vec![0.0f32; 8];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 2).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("teacher_topk_indices length"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_logprobs_length() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        // T_active = 4, K = 2, so expected length = 8; pass 6 for lp.
        let idx = vec![0u32; 8];
        let lp = vec![0.0f32; 6];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 2).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("teacher_topk_logprobs length"), "got: {s}");
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_validates_index_in_vocab() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        // vocab_size = 16, so index 99 is out of bounds.
        let idx = vec![99u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(
            s.contains(">= vocab_size") || s.contains("99"),
            "got: {s}"
        );
    }

    #[test]
    fn opd_top_k_reverse_kl_kt_stub_returns_not_yet_implemented_on_valid_shapes() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            opd_top_k_reverse_kl_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        assert!(
            matches!(err, OpdLossError::NotYetImplemented(_)),
            "expected NotYetImplemented, got: {err}"
        );
    }

    #[test]
    fn opd_top_k_reverse_kl_per_position_kt_stub_returns_not_yet_implemented_on_valid_shapes() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err = opd_top_k_reverse_kl_per_position_kt(&h, &w, &idx, &lp, &mask, 1)
            .unwrap_err();
        assert!(
            matches!(err, OpdLossError::NotYetImplemented(_)),
            "expected NotYetImplemented, got: {err}"
        );
    }

    #[test]
    fn opd_top_k_reverse_kl_per_position_kt_validates_hidden_rank() {
        // Same hidden rank validation applies to per-position entry.
        let h = KtTensor::from_vec(vec![0.0f32; 16], vec![4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err = opd_top_k_reverse_kl_per_position_kt(&h, &w, &idx, &lp, &mask, 1)
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden must be 3-D"), "got: {s}");
    }

    #[test]
    fn compute_per_position_metrics_kt_validates_hidden_rank() {
        let h = KtTensor::from_vec(vec![0.0f32; 16], vec![4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err =
            compute_per_position_metrics_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden must be 3-D"), "got: {s}");
    }

    #[test]
    fn compute_per_position_metrics_kt_stub_returns_not_yet_implemented_on_valid_shapes() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let idx = vec![0u32; 4];
        let lp = vec![0.0f32; 4];
        let mask = vec![true; 4];
        let err = compute_per_position_metrics_kt(&h, &w, &idx, &lp, &mask, 1).unwrap_err();
        assert!(
            matches!(err, OpdLossError::NotYetImplemented(_)),
            "expected NotYetImplemented, got: {err}"
        );
    }

    #[test]
    fn per_position_metrics_kt_entropy_gap_vec_smoke() {
        let m = PerPositionMetricsKt {
            student_entropy: vec![1.0, 0.5, 2.0],
            teacher_entropy: vec![1.5, 0.5, 1.0],
            reverse_kl: vec![0.1, 0.0, 0.5],
        };
        let gap = m.entropy_gap_vec();
        assert_eq!(gap, vec![0.5, 0.0, 1.0]);
    }

    #[test]
    fn per_position_metrics_kt_mean_helpers() {
        let m = PerPositionMetricsKt {
            student_entropy: vec![1.0, 0.5, 2.0],
            teacher_entropy: vec![1.5, 0.5, 1.0],
            reverse_kl: vec![0.1, 0.0, 0.5],
        };
        let mean_gap = m.mean_entropy_gap();
        assert!((mean_gap - 0.5).abs() < 1e-6, "got: {mean_gap}");
        let mean_kl = m.mean_reverse_kl();
        assert!((mean_kl - 0.2).abs() < 1e-6, "got: {mean_kl}");
    }

    #[test]
    fn per_position_metrics_kt_empty_means_are_zero() {
        let m = PerPositionMetricsKt::default();
        assert_eq!(m.mean_entropy_gap(), 0.0);
        assert_eq!(m.mean_reverse_kl(), 0.0);
        assert!(m.entropy_gap_vec().is_empty());
    }

}

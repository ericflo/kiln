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
//! This module ships:
//!
//! 1. [`FlceMatmulProviderKt`] — kt-typed parallel of the candle
//!    [`crate::FlceMatmulProvider`] trait.
//! 2. [`FlceError`] — kt-typed error, independent of candle / anyhow.
//! 3. [`fused_linear_cross_entropy_phase_b_kt`] — kt-typed forward
//!    entry point. Implements the chunked log-sum-exp reduction
//!    over [`kiln_tensor`] ops; mirrors the Phase A candle reference
//!    (see [`crate::fused_linear_cross_entropy`]) up to floating-
//!    point associativity in the chunked reduction.
//!
//! Backward (`backward_dhidden_kt`) is still TBD — autograd lives in
//! a separate crate and will plug into the existing dC chunk recipe
//! once kt-tensor has the necessary backward hooks.

use std::sync::Arc;

use kiln_tensor::{
    ops::{
        broadcast_to, exp, gather, index_select, ln, matmul, max_axis, mean_all, mul, scatter_add,
        sub, sum_axis, to_f32,
    },
    Error as KtError, Tensor as KtTensor,
};

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
    /// Underlying `kiln_tensor` op error surfaced from the chunked
    /// reduction body.
    Kt(KtError),
    /// The kt-typed entry point exists but its body has not yet been
    /// implemented. Reserved for future backward / extra entry
    /// points; the production forward
    /// [`fused_linear_cross_entropy_phase_b_kt`] no longer returns
    /// this variant.
    NotYetImplemented(&'static str),
}

impl std::fmt::Display for FlceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlceError::Msg(m) => f.write_str(m),
            FlceError::Kt(e) => write!(f, "kt-flce: {e}"),
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

impl From<KtError> for FlceError {
    fn from(e: KtError) -> Self {
        FlceError::Kt(e)
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

/// kt-typed entry point for FLCE Phase B forward.
///
/// Re-implements the candle Phase A chunked log-sum-exp reduction
/// (see [`crate::fused_linear_cross_entropy`]) over `kiln_tensor`
/// ops. Numerically equivalent up to floating-point associativity in
/// the chunked sum-exp accumulation; the per-chunk kernel sequence
/// (matmul → max → shift → exp → sum) is identical.
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
/// A scalar F32 [`KtTensor`] (rank-0 / shape `[]`) holding the mean
/// cross-entropy over active positions. Returns a scalar 0.0 tensor
/// if no positions are active or if `seq_len < 2`.
///
/// # Backward
///
/// Backward (`dhidden`) is not yet implemented in the kt-typed path —
/// it currently lives in the candle Phase B `CustomOp1` and will be
/// migrated once kt-tensor has the necessary autograd hooks. Until
/// then this entry point is forward-only.
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

    // Sub-2 seq lens have no targets to predict; return scalar 0.
    if seq_len < 2 {
        return zero_scalar();
    }

    let vocab_size = head_dims[1];

    // Gather active positions: indices into hidden[0, :seq_len-1] that
    // contribute to the loss (their corresponding shifted label is
    // unmasked). Mirrors `active_positions` in the candle reference.
    let shift_mask = &label_mask[1..]; // length seq_len - 1
    let shift_labels: Vec<u32> = input_ids[1..].to_vec();
    let active_positions: Vec<u32> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();

    if active_positions.is_empty() {
        return zero_scalar();
    }

    let num_active = active_positions.len();
    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i as usize])
        .collect();

    // Build `active_hidden` of shape `[num_active, hidden_size]` in F32.
    //
    // 1. Squeeze batch dim 0 (hidden was [1, seq_len, hidden_size]).
    // 2. Narrow seq dim 0 to the [0..seq_len-1] shift range.
    // 3. index_select along axis 0 with active_positions indices.
    // 4. Cast to F32.
    let hidden_2d = hidden.squeeze(0).map_err(FlceError::Kt)?;
    let shift_hidden = hidden_2d
        .narrow(0, 0, seq_len - 1)
        .map_err(FlceError::Kt)?
        .contiguous()
        .map_err(FlceError::Kt)?;
    let active_idx = KtTensor::from_vec(active_positions.clone(), vec![num_active])
        .map_err(FlceError::Kt)?;
    let active_hidden = index_select(&shift_hidden, 0, &active_idx).map_err(FlceError::Kt)?;
    let active_hidden_f32 = to_f32(&active_hidden).map_err(FlceError::Kt)?;
    let head_t_f32 = to_f32(head_t).map_err(FlceError::Kt)?;

    // Accumulators in F32 for numerical stability.
    //
    //   running_max[i]      = max_{j in [0, V_seen)} logits[i, j]    (shape [num_active])
    //   running_sumexp[i]   = sum_{j in [0, V_seen)} exp(logits[i, j] - running_max[i])
    //   correct_logit[i]    = logits[i, labels[i]]                    (shape [num_active])
    //
    // The candle reference keeps these as `keepdim` 2-D tensors so
    // broadcast_mul lines up. kt-tensor's `max_axis` / `sum_axis`
    // both collapse the reduced axis, so we instead store the 1-D
    // form and `unsqueeze(1)` + `broadcast_to` when we need to
    // broadcast across the chunk dim.
    let mut running_max: Option<KtTensor> = None; // 1-D [num_active]
    let mut running_sumexp: Option<KtTensor> = None; // 1-D [num_active]
    let mut correct_logit: Option<KtTensor> = None; // 1-D [num_active] F32

    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);

        // Head slice: [hidden_size, chunk_len], contiguous.
        let head_chunk = head_t_f32
            .narrow(1, chunk_start, chunk_len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;

        // Chunk logits: [num_active, chunk_len] F32. The one materialized
        // intermediate whose vocab-axis size is chunk_len instead of
        // vocab_size.
        let logits_chunk = matmul(&active_hidden_f32, &head_chunk).map_err(FlceError::Kt)?;

        // Per-row max within the chunk: 1-D [num_active] (axis-reduced,
        // collapsed). Broadcast back to 2-D before the shift.
        let chunk_max_1d = max_axis(&logits_chunk, 1).map_err(FlceError::Kt)?;

        // Update running_max + running_sumexp.
        let (new_max_1d, new_sumexp_1d) = match (running_max.as_ref(), running_sumexp.as_ref()) {
            (None, None) => {
                // First chunk: running_max = chunk_max,
                //              running_sumexp = sum(exp(chunk - chunk_max)).
                let chunk_max_2d = chunk_max_1d
                    .unsqueeze(1)
                    .map_err(FlceError::Kt)?
                    .contiguous()
                    .map_err(FlceError::Kt)?;
                let chunk_max_b = broadcast_to(&chunk_max_2d, &[num_active, chunk_len])
                    .map_err(FlceError::Kt)?;
                let shifted = sub(&logits_chunk, &chunk_max_b).map_err(FlceError::Kt)?;
                let exped = exp(&shifted).map_err(FlceError::Kt)?;
                let chunk_sumexp_1d = sum_axis(&exped, 1).map_err(FlceError::Kt)?;
                (chunk_max_1d, chunk_sumexp_1d)
            }
            (Some(prev_max), Some(prev_sumexp)) => {
                // new_max = max(prev_max, chunk_max)
                // prev_sumexp *= exp(prev_max - new_max)
                // chunk_sumexp = sum(exp(logits_chunk - new_max))
                // new_sumexp = prev_sumexp + chunk_sumexp
                let new_max_1d = elementwise_max(prev_max, &chunk_max_1d)?;
                let prev_scale_1d = exp(&sub(prev_max, &new_max_1d).map_err(FlceError::Kt)?)
                    .map_err(FlceError::Kt)?;
                let scaled_prev_1d = mul(prev_sumexp, &prev_scale_1d).map_err(FlceError::Kt)?;
                let new_max_2d = new_max_1d
                    .unsqueeze(1)
                    .map_err(FlceError::Kt)?
                    .contiguous()
                    .map_err(FlceError::Kt)?;
                let new_max_b = broadcast_to(&new_max_2d, &[num_active, chunk_len])
                    .map_err(FlceError::Kt)?;
                let shifted = sub(&logits_chunk, &new_max_b).map_err(FlceError::Kt)?;
                let exped = exp(&shifted).map_err(FlceError::Kt)?;
                let chunk_sumexp_1d = sum_axis(&exped, 1).map_err(FlceError::Kt)?;
                let new_sumexp_1d = kiln_tensor::ops::add(&scaled_prev_1d, &chunk_sumexp_1d)
                    .map_err(FlceError::Kt)?;
                (new_max_1d, new_sumexp_1d)
            }
            _ => unreachable!("running_max and running_sumexp are set together"),
        };
        running_max = Some(new_max_1d);
        running_sumexp = Some(new_sumexp_1d);

        // For each active row whose label falls inside this chunk,
        // gather the correct logit from `logits_chunk`.
        let chunk_end = chunk_start + chunk_len;
        let mut row_hits: Vec<u32> = Vec::new();
        let mut col_hits: Vec<u32> = Vec::new();
        for (row_idx, &label) in active_labels.iter().enumerate() {
            let label = label as usize;
            if label >= chunk_start && label < chunk_end {
                row_hits.push(row_idx as u32);
                col_hits.push((label - chunk_start) as u32);
            }
        }
        if !row_hits.is_empty() {
            let hits = row_hits.len();
            let row_idx_t = KtTensor::from_vec(row_hits.clone(), vec![hits])
                .map_err(FlceError::Kt)?;
            // Pick the rows from logits_chunk via index_select along axis 0:
            // shape [hits, chunk_len].
            let selected_rows = index_select(&logits_chunk, 0, &row_idx_t)
                .map_err(FlceError::Kt)?;
            // Gather one column per row: gather expects index tensor with
            // the same rank as the source; result shape matches indices'
            // shape, so we use [hits, 1] index tensor along axis 1.
            let col_idx_2d = KtTensor::from_vec(col_hits.clone(), vec![hits, 1])
                .map_err(FlceError::Kt)?;
            let gathered_2d =
                gather(&selected_rows, 1, &col_idx_2d).map_err(FlceError::Kt)?; // [hits, 1]
            let gathered_1d = gathered_2d.squeeze(1).map_err(FlceError::Kt)?; // [hits]

            // Scatter into `correct_logit` (shape [num_active]) using
            // scatter_add: each row in `active_labels` falls in exactly
            // one chunk, so each [num_active] slot is touched exactly
            // once → scatter_add is equivalent to a scatter.
            let scattered = scatter_add(&gathered_1d, 0, &row_idx_t, num_active)
                .map_err(FlceError::Kt)?;
            correct_logit = Some(match correct_logit.take() {
                Some(cur) => kiln_tensor::ops::add(&cur, &scattered).map_err(FlceError::Kt)?,
                None => scattered,
            });
        }

        chunk_start = chunk_end;
    }

    let running_max_1d = running_max
        .ok_or_else(|| FlceError::msg("kt-flce: vocab_size was 0"))?;
    let running_sumexp_1d = running_sumexp
        .ok_or_else(|| FlceError::msg("kt-flce: vocab_size was 0"))?;
    let correct_logit_1d = correct_logit.ok_or_else(|| {
        FlceError::msg("kt-flce: no labels fell inside any vocab chunk — label >= vocab_size?")
    })?;

    // log_sum_exp = running_max + log(running_sumexp). Both are 1-D
    // [num_active] F32.
    let log_sumexp = ln(&running_sumexp_1d).map_err(FlceError::Kt)?;
    let log_sum_exp = kiln_tensor::ops::add(&running_max_1d, &log_sumexp).map_err(FlceError::Kt)?;

    // Per-token loss = log_sum_exp - correct_logit. Mean over active rows.
    let per_token_loss = sub(&log_sum_exp, &correct_logit_1d).map_err(FlceError::Kt)?;
    let loss = mean_all(&per_token_loss).map_err(FlceError::Kt)?;
    Ok(loss)
}

/// Helper: build a rank-0 F32 scalar tensor holding 0.0.
fn zero_scalar() -> Result<KtTensor, FlceError> {
    KtTensor::from_vec(vec![0.0f32], vec![]).map_err(FlceError::Kt)
}

/// Elementwise max for two same-shape F32 tensors. kt-tensor has
/// `ops::maximum(a, b)`; the binary_minmax module re-exports it. Use
/// the public re-export to keep imports tight.
fn elementwise_max(a: &KtTensor, b: &KtTensor) -> Result<KtTensor, FlceError> {
    kiln_tensor::ops::maximum(a, b).map_err(FlceError::Kt)
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

    /// Forward smoke: zero-input hidden + zero head should give
    /// log(V) cross-entropy (uniform distribution).
    #[test]
    fn fused_linear_cross_entropy_phase_b_kt_uniform_loss() {
        let h = 8;
        let v = 16;
        let seq = 4;
        let hidden = dummy_hidden(seq, h); // all zeros
        let head = dummy_head_t(h, v); // all zeros
        let ids = vec![0u32, 1, 2, 3];
        let mask = vec![true; seq];
        let loss = fused_linear_cross_entropy_phase_b_kt(&hidden, &head, &ids, &mask, 4)
            .expect("forward");
        // Read scalar value.
        let storage = loss.storage();
        let cpu = storage
            .as_any()
            .downcast_ref::<kiln_tensor::CpuStorage>()
            .expect("scalar cpu");
        let bytes = cpu.as_bytes();
        let v_f32 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        // Uniform distribution loss = ln(V).
        let expected = (v as f32).ln();
        assert!(
            (v_f32 - expected).abs() < 1e-4,
            "loss {v_f32} != expected {expected}"
        );
    }

    /// Returns 0 when no labels are active (mask all false at shifted
    /// positions).
    #[test]
    fn fused_linear_cross_entropy_phase_b_kt_no_active_returns_zero() {
        let hidden = dummy_hidden(4, 8);
        let head = dummy_head_t(8, 16);
        let ids = vec![0u32, 1, 2, 3];
        // label_mask[1..] is the shifted mask; mark all false.
        let mask = vec![true, false, false, false];
        let loss = fused_linear_cross_entropy_phase_b_kt(&hidden, &head, &ids, &mask, 4)
            .expect("forward");
        assert_eq!(loss.shape(), &[] as &[usize]);
        let storage = loss.storage();
        let cpu = storage
            .as_any()
            .downcast_ref::<kiln_tensor::CpuStorage>()
            .expect("scalar cpu");
        let bytes = cpu.as_bytes();
        let v_f32 = f32::from_le_bytes(bytes[0..4].try_into().unwrap());
        assert_eq!(v_f32, 0.0);
    }

    /// Multi-chunk parity: same uniform-distribution input but split
    /// across two vocab chunks should give the same loss as a single
    /// chunk (math-equivalent up to floating-point associativity).
    #[test]
    fn fused_linear_cross_entropy_phase_b_kt_chunk_parity() {
        let h = 8;
        let v = 16;
        let seq = 4;
        let hidden = dummy_hidden(seq, h);
        let head = dummy_head_t(h, v);
        let ids = vec![0u32, 1, 2, 3];
        let mask = vec![true; seq];

        let l_single =
            scalar_value(fused_linear_cross_entropy_phase_b_kt(&hidden, &head, &ids, &mask, v)
                .unwrap());
        let l_multi =
            scalar_value(fused_linear_cross_entropy_phase_b_kt(&hidden, &head, &ids, &mask, 4)
                .unwrap());
        assert!(
            (l_single - l_multi).abs() < 1e-4,
            "single-chunk {l_single} != multi-chunk {l_multi}"
        );
    }

    fn scalar_value(t: KtTensor) -> f32 {
        let storage = t.storage();
        let cpu = storage
            .as_any()
            .downcast_ref::<kiln_tensor::CpuStorage>()
            .expect("scalar cpu");
        let bytes = cpu.as_bytes();
        f32::from_le_bytes(bytes[0..4].try_into().unwrap())
    }
}

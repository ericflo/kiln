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
//! 4. [`fused_linear_cross_entropy_phase_b_backward_kt`] — kt-typed
//!    manual backward producing `dhidden` from `grad_loss`. Mirrors
//!    the candle reference's `phase_b::backward_dhidden` step-by-step
//!    using the kt-typed parallels of every op in the candle code
//!    path. Together with the forward this closes the loop on running
//!    FLCE Phase B end-to-end over kt-tensor without dragging a
//!    `candle_core::CustomOp1` boundary into the autograd graph.

use std::sync::Arc;

use kiln_tensor::{
    ops::{
        broadcast_to, exp, gather, index_select, ln, matmul, max_axis, mean_all, mul, mul_scalar,
        scatter_add, sub, sum_axis, to_f32,
    },
    DType as KtDType, Device as KtDevice, Error as KtError, Tensor as KtTensor,
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

/// kt-typed FLCE Phase B backward — compute `dhidden` from `grad_loss`.
///
/// Manual two-pass backward mirroring the candle reference
/// (`crate::phase_b::backward_dhidden`):
///
/// 1. **Pass 1**: recompute `running_max` and `running_sumexp` chunk-by-chunk
///    (identical to forward, minus the `correct_logit` gather).
/// 2. **Pass 2**: for each chunk, recompute `softmax = exp(logits - running_max)
///    / running_sumexp`, build the one-hot label mask for that chunk, and
///    accumulate `dhidden_active += (softmax - one_hot) @ head_chunk.T *
///    grad_loss / N`. Finally scatter `dhidden_active` back into the
///    `[seq_len, hidden_size]` zero buffer, unsqueeze batch dim, cast to
///    the original `hidden` dtype.
///
/// # Shape contract
///
/// - `hidden`: same shape as forward input `[1, seq_len, hidden_size]`.
/// - `head_t`: `[hidden_size, vocab_size]`.
/// - `input_ids`: length `seq_len`.
/// - `label_mask`: length `seq_len`.
/// - `chunk_size`: same as forward (rerun the chunk loop).
/// - `grad_loss`: scalar (rank-0) F32 tensor; the seed gradient
///   `d_loss / d_loss` (typically 1.0).
///
/// # Returns
///
/// `dhidden` as a tensor with shape `[1, seq_len, hidden_size]` and the
/// **original** `hidden.dtype()`. Rows outside `active_positions` and
/// the `seq_len-1` row are zero.
///
/// # Numerical equivalence
///
/// Returns the same gradient as candle's `phase_b::backward_dhidden` up
/// to floating-point associativity in the chunked sum-exp accumulation.
/// The per-chunk kernel sequence is identical (matmul → max → shift →
/// exp → div → diff → matmul); all reductions and accumulators run in
/// F32.
///
/// # Device-agnostic
///
/// `grad_loss` is broadcast through `broadcast_to([num_active,
/// chunk_len])` after a `reshape([1, 1])`, so the backward never
/// pulls scalar values to host — when kt-tensor's elementwise ops
/// are wired up to CUDA, the backward continues to work without
/// changes.
pub fn fused_linear_cross_entropy_phase_b_backward_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
    grad_loss: &KtTensor,
) -> Result<KtTensor, FlceError> {
    let seq_len = input_ids.len();
    if label_mask.len() != seq_len {
        return Err(FlceError::msg(format!(
            "kt-flce-bwd: label_mask length {} does not match input_ids length {}",
            label_mask.len(),
            seq_len,
        )));
    }
    if chunk_size == 0 {
        return Err(FlceError::msg("kt-flce-bwd: chunk_size must be > 0"));
    }
    let hidden_dims = hidden.shape().to_vec();
    if hidden_dims.len() != 3 {
        return Err(FlceError::msg(format!(
            "kt-flce-bwd: hidden must be 3-D [1, seq_len, hidden_size]; got {hidden_dims:?}",
        )));
    }
    if hidden_dims[0] != 1 {
        return Err(FlceError::msg(format!(
            "kt-flce-bwd: hidden batch dim must be 1; got {hidden_dims:?}",
        )));
    }
    let head_dims = head_t.shape().to_vec();
    if head_dims.len() != 2 {
        return Err(FlceError::msg(format!(
            "kt-flce-bwd: head_t must be 2-D [hidden_size, vocab_size]; got {head_dims:?}",
        )));
    }
    if hidden_dims[2] != head_dims[0] {
        return Err(FlceError::msg(format!(
            "kt-flce-bwd: hidden hidden_size {} != head_t hidden_size {}",
            hidden_dims[2], head_dims[0],
        )));
    }

    let hidden_size = hidden_dims[2];
    let vocab_size = head_dims[1];
    let original_dtype = hidden.dtype();

    // seq_len < 2: no targets, gradient is zero everywhere.
    if seq_len < 2 {
        return zeros_like_hidden_in_dtype(&hidden_dims, original_dtype);
    }

    // Active row indices in the shifted positions, mirroring forward.
    let shift_mask = &label_mask[1..]; // length seq_len - 1
    let shift_labels: Vec<u32> = input_ids[1..].to_vec();
    let active_positions: Vec<u32> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();

    if active_positions.is_empty() {
        return zeros_like_hidden_in_dtype(&hidden_dims, original_dtype);
    }

    let num_active = active_positions.len();
    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i as usize])
        .collect();

    // Build active_hidden F32 the same way as forward.
    let hidden_2d = hidden.squeeze(0).map_err(FlceError::Kt)?;
    let shift_hidden = hidden_2d
        .narrow(0, 0, seq_len - 1)
        .map_err(FlceError::Kt)?
        .contiguous()
        .map_err(FlceError::Kt)?;
    // Derive the destination device from the input `hidden`'s storage
    // so that every downstream allocator + dispatch stays on-device.
    // `dispatch2` rejects mixed-device inputs (CPU + CUDA would error),
    // so allocating `active_idx` / accumulator / one_hot via the
    // CPU-only `from_vec` / `zeros_cpu` constructors would break the
    // chain the moment `hidden` lives on CUDA. `*_on` is the
    // device-parametric companion that routes to the matching backend.
    let device: KtDevice = hidden.device();
    let active_idx = KtTensor::from_vec_on(device, active_positions.clone(), vec![num_active])
        .map_err(FlceError::Kt)?;
    let active_hidden = index_select(&shift_hidden, 0, &active_idx).map_err(FlceError::Kt)?;
    let active_hidden_f32 = to_f32(&active_hidden).map_err(FlceError::Kt)?;
    let head_t_f32 = to_f32(head_t).map_err(FlceError::Kt)?;

    // -----------------------------------------------------------------
    // Pass 1: recompute running_max + running_sumexp.
    // -----------------------------------------------------------------
    let mut running_max: Option<KtTensor> = None; // 1-D [num_active] F32
    let mut running_sumexp: Option<KtTensor> = None; // 1-D [num_active] F32
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let head_chunk = head_t_f32
            .narrow(1, chunk_start, chunk_len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        let logits_chunk = matmul(&active_hidden_f32, &head_chunk).map_err(FlceError::Kt)?;
        let chunk_max_1d = max_axis(&logits_chunk, 1).map_err(FlceError::Kt)?;

        let (new_max_1d, new_sumexp_1d) = match (running_max.as_ref(), running_sumexp.as_ref()) {
            (None, None) => {
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
                let new_sumexp_1d =
                    kiln_tensor::ops::add(&scaled_prev_1d, &chunk_sumexp_1d).map_err(FlceError::Kt)?;
                (new_max_1d, new_sumexp_1d)
            }
            _ => unreachable!("running_max and running_sumexp are set together"),
        };
        running_max = Some(new_max_1d);
        running_sumexp = Some(new_sumexp_1d);
        chunk_start += chunk_len;
    }

    let running_max_1d = running_max
        .ok_or_else(|| FlceError::msg("kt-flce-bwd: vocab_size was 0"))?;
    let running_sumexp_1d = running_sumexp
        .ok_or_else(|| FlceError::msg("kt-flce-bwd: vocab_size was 0"))?;

    // -----------------------------------------------------------------
    // Pass 2: accumulate dhidden_active by chunk.
    // -----------------------------------------------------------------
    //
    // Reshape grad_loss to a 1x1 tensor up front so we can broadcast it
    // to `[num_active, chunk_len]` per-chunk via `broadcast_to`. The
    // candle reference uses `broadcast_mul(&grad_loss_f32)` directly on
    // a rank-0 scalar; kt-tensor's `broadcast_to` requires equal-rank
    // inputs so we promote the scalar to rank-2 once before the loop.
    let grad_loss_f32 = to_f32(grad_loss).map_err(FlceError::Kt)?;
    let grad_loss_2d = grad_loss_f32
        .reshape(vec![1, 1])
        .map_err(FlceError::Kt)?;
    let inv_n = 1.0f32 / (num_active as f32);

    // Broadcast running_max / running_sumexp to 2-D once (they don't
    // depend on chunk; we'll re-broadcast to chunk_len inside the loop).
    let running_max_2d = running_max_1d
        .unsqueeze(1)
        .map_err(FlceError::Kt)?
        .contiguous()
        .map_err(FlceError::Kt)?;
    let running_sumexp_2d = running_sumexp_1d
        .unsqueeze(1)
        .map_err(FlceError::Kt)?
        .contiguous()
        .map_err(FlceError::Kt)?;

    // Allocate the chunk accumulator on the same device as `hidden`
    // so the `kiln_tensor::ops::add(&dhidden_active, &chunk_contrib)`
    // call inside the loop stays on device and never has to round-trip
    // through CPU.
    let mut dhidden_active =
        KtTensor::zeros_on(device, vec![num_active, hidden_size], KtDType::F32)
            .map_err(FlceError::Kt)?;
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let chunk_end = chunk_start + chunk_len;

        let head_chunk = head_t_f32
            .narrow(1, chunk_start, chunk_len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        let logits_chunk = matmul(&active_hidden_f32, &head_chunk).map_err(FlceError::Kt)?;

        // softmax_chunk = exp(logits_chunk - running_max) / running_sumexp
        let max_b = broadcast_to(&running_max_2d, &[num_active, chunk_len])
            .map_err(FlceError::Kt)?;
        let shifted = sub(&logits_chunk, &max_b).map_err(FlceError::Kt)?;
        let exp_chunk = exp(&shifted).map_err(FlceError::Kt)?;
        let sumexp_b = broadcast_to(&running_sumexp_2d, &[num_active, chunk_len])
            .map_err(FlceError::Kt)?;
        let softmax_chunk =
            kiln_tensor::ops::div(&exp_chunk, &sumexp_b).map_err(FlceError::Kt)?;

        // Build one_hot mask: 1.0 wherever label == col, else 0.
        let mut one_hot_data: Vec<f32> = vec![0.0; num_active * chunk_len];
        for (row_idx, &label) in active_labels.iter().enumerate() {
            let label = label as usize;
            if label >= chunk_start && label < chunk_end {
                let col = label - chunk_start;
                one_hot_data[row_idx * chunk_len + col] = 1.0;
            }
        }
        // One-hot lives on the same device as `hidden` (and as
        // `softmax_chunk`, which already lives on-device because all
        // upstream tensors are derived from `hidden` / `head_t`). The
        // host-side `one_hot_data` Vec is staged on CPU then uploaded
        // via `from_vec_on` for CUDA, mirroring how `cuda_from_slice`
        // would do it manually.
        let one_hot =
            KtTensor::from_vec_on(device, one_hot_data, vec![num_active, chunk_len])
                .map_err(FlceError::Kt)?;

        // diff = softmax - one_hot
        let diff = sub(&softmax_chunk, &one_hot).map_err(FlceError::Kt)?;

        // scaled = diff * (1/N)
        let scaled = mul_scalar(&diff, inv_n).map_err(FlceError::Kt)?;

        // grad_logits_chunk = scaled * grad_loss
        //   Broadcast the [1, 1] grad_loss to [num_active, chunk_len]
        //   and multiply elementwise. Equivalent to candle's
        //   `scaled.broadcast_mul(&grad_loss_f32)` on a rank-0 scalar.
        let grad_loss_b = broadcast_to(&grad_loss_2d, &[num_active, chunk_len])
            .map_err(FlceError::Kt)?;
        let grad_logits_chunk = mul(&scaled, &grad_loss_b).map_err(FlceError::Kt)?;

        // chunk_contrib = grad_logits_chunk @ head_chunk.T   shape [num_active, hidden_size]
        let head_chunk_t = head_chunk
            .t()
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        let chunk_contrib =
            matmul(&grad_logits_chunk, &head_chunk_t).map_err(FlceError::Kt)?;

        dhidden_active =
            kiln_tensor::ops::add(&dhidden_active, &chunk_contrib).map_err(FlceError::Kt)?;

        chunk_start = chunk_end;
    }

    // Scatter dhidden_active back into a [seq_len, hidden_size] zero
    // buffer. active_indices live in [0..seq_len-1]; row seq_len-1
    // never contributed (we used hidden[..seq_len-1]) so its gradient
    // stays zero.
    let grad_hidden_2d =
        scatter_add(&dhidden_active, 0, &active_idx, seq_len).map_err(FlceError::Kt)?;

    // Restore the batch dim.
    let grad_hidden_3d = grad_hidden_2d.unsqueeze(0).map_err(FlceError::Kt)?;

    // Cast back to the original `hidden` dtype.
    let out = if original_dtype == KtDType::F32 {
        grad_hidden_3d
    } else {
        kiln_tensor::ops::cast(&grad_hidden_3d, original_dtype).map_err(FlceError::Kt)?
    };
    Ok(out)
}

/// Helper: build a zero `[1, seq_len, hidden_size]` tensor in the given
/// dtype. Used by the empty-mask / short-seq early-returns.
fn zeros_like_hidden_in_dtype(
    hidden_dims: &[usize],
    dtype: KtDType,
) -> Result<KtTensor, FlceError> {
    Ok(KtTensor::zeros_cpu(hidden_dims.to_vec(), dtype))
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

    fn read_f32_vec(t: &KtTensor) -> Vec<f32> {
        let storage = t.storage();
        let cpu = storage
            .as_any()
            .downcast_ref::<kiln_tensor::CpuStorage>()
            .expect("cpu storage");
        let bytes = cpu.as_bytes();
        bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Backward smoke: short-seq early-return produces zero grad.
    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_kt_short_seq_returns_zero() {
        let hidden = dummy_hidden(1, 4);
        let head = dummy_head_t(4, 8);
        let ids = vec![0u32];
        let mask = vec![true];
        let grad_loss = KtTensor::from_vec(vec![1.0f32], vec![]).unwrap();
        let g =
            fused_linear_cross_entropy_phase_b_backward_kt(&hidden, &head, &ids, &mask, 4, &grad_loss)
                .expect("backward");
        assert_eq!(g.shape(), &[1, 1, 4]);
        assert!(read_f32_vec(&g).iter().all(|&v| v == 0.0));
    }

    /// Backward smoke: empty mask produces zero grad with hidden shape.
    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_kt_no_active_returns_zero() {
        let hidden = dummy_hidden(4, 8);
        let head = dummy_head_t(8, 16);
        let ids = vec![0u32, 1, 2, 3];
        // Mask shifted positions all false.
        let mask = vec![true, false, false, false];
        let grad_loss = KtTensor::from_vec(vec![1.0f32], vec![]).unwrap();
        let g =
            fused_linear_cross_entropy_phase_b_backward_kt(&hidden, &head, &ids, &mask, 4, &grad_loss)
                .expect("backward");
        assert_eq!(g.shape(), &[1, 4, 8]);
        assert!(read_f32_vec(&g).iter().all(|&v| v == 0.0));
    }

    /// Uniform-input backward: with zero hidden + zero head, softmax is
    /// 1/V for every position, so for an active row with label = c:
    ///
    ///   d_loss / d_logits[i, j] = (1/N) * (1/V - 1{j==c})
    ///
    /// then d_loss / d_active_hidden[i, :] = sum_j (...) * head_t[:, j]
    /// which is zero because head_t is all zero. So the resulting
    /// dhidden should be all zero, just confirming end-to-end runs
    /// without panicking and the chunk-loop math is wired.
    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_kt_zero_inputs_zero_grad() {
        let h = 8;
        let v = 16;
        let seq = 4;
        let hidden = dummy_hidden(seq, h);
        let head = dummy_head_t(h, v);
        let ids = vec![0u32, 1, 2, 3];
        let mask = vec![true; seq];
        let grad_loss = KtTensor::from_vec(vec![1.0f32], vec![]).unwrap();
        let g = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden,
            &head,
            &ids,
            &mask,
            4,
            &grad_loss,
        )
        .expect("backward");
        assert_eq!(g.shape(), &[1, seq, h]);
        // head_t == 0 ⇒ chunk_contrib == 0 ⇒ dhidden_active == 0.
        for v in read_f32_vec(&g) {
            assert!(v.abs() < 1e-6, "expected zero grad, got {v}");
        }
    }

    /// Shape-validation: chunk_size = 0 errors.
    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_kt_validates_chunk_size_zero() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let ids = vec![0u32; 4];
        let mask = vec![true; 4];
        let grad_loss = KtTensor::from_vec(vec![1.0f32], vec![]).unwrap();
        let err =
            fused_linear_cross_entropy_phase_b_backward_kt(&h, &w, &ids, &mask, 0, &grad_loss)
                .unwrap_err();
        assert!(matches!(err, FlceError::Msg(_)));
        let s = format!("{err}");
        assert!(s.contains("chunk_size must be > 0"), "got: {s}");
    }

    /// Shape-validation: mask length mismatch errors.
    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_kt_validates_mask_length() {
        let h = dummy_hidden(4, 8);
        let w = dummy_head_t(8, 16);
        let ids = vec![0u32; 4];
        let mask = vec![true; 3];
        let grad_loss = KtTensor::from_vec(vec![1.0f32], vec![]).unwrap();
        let err =
            fused_linear_cross_entropy_phase_b_backward_kt(&h, &w, &ids, &mask, 4, &grad_loss)
                .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("label_mask length"), "got: {s}");
    }

    /// Hidden rank validation in backward.
    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_kt_validates_hidden_rank() {
        let h = KtTensor::from_vec(vec![0.0f32; 16], vec![4, 4]).expect("alloc h");
        let w = dummy_head_t(4, 8);
        let ids = vec![0u32; 4];
        let mask = vec![true; 4];
        let grad_loss = KtTensor::from_vec(vec![1.0f32], vec![]).unwrap();
        let err =
            fused_linear_cross_entropy_phase_b_backward_kt(&h, &w, &ids, &mask, 4, &grad_loss)
                .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("hidden must be 3-D"), "got: {s}");
    }

    /// Backward chunk-size parity: single vs multi chunks produce
    /// numerically equivalent gradients (up to FP associativity).
    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_kt_chunk_parity() {
        let seq_len = 8;
        let hidden_size = 4;
        let vocab_size = 16;

        let total_h = seq_len * hidden_size;
        let hidden_vec: Vec<f32> = (0..total_h)
            .map(|i| (i as f32 * 0.011).sin() * 0.3)
            .collect();
        let total_head = hidden_size * vocab_size;
        let head_vec: Vec<f32> = (0..total_head)
            .map(|i| ((i as f32 + 3.0) * 0.005).cos() * 0.2)
            .collect();
        let ids: Vec<u32> = (0..seq_len as u32)
            .map(|i| (i * 17 + 2) % vocab_size as u32)
            .collect();
        let mask: Vec<bool> = (0..seq_len).map(|i| i > 0).collect();

        let hidden_kt = KtTensor::from_vec(hidden_vec.clone(), vec![1, seq_len, hidden_size]).unwrap();
        let head_kt =
            KtTensor::from_vec(head_vec.clone(), vec![hidden_size, vocab_size]).unwrap();
        let grad_loss_kt = KtTensor::from_vec(vec![1.0f32], vec![]).unwrap();

        let g_single = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden_kt,
            &head_kt,
            &ids,
            &mask,
            vocab_size,
            &grad_loss_kt,
        )
        .unwrap();
        let g_multi = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden_kt,
            &head_kt,
            &ids,
            &mask,
            4,
            &grad_loss_kt,
        )
        .unwrap();

        let s = read_f32_vec(&g_single);
        let m = read_f32_vec(&g_multi);
        assert_eq!(s.len(), m.len());
        let mut max_abs = 0.0f32;
        let mut max_mag = 0.0f32;
        for (a, b) in s.iter().zip(m.iter()) {
            max_abs = max_abs.max((a - b).abs());
            max_mag = max_mag.max(a.abs());
        }
        let rel = if max_mag > 1e-6 { max_abs / max_mag } else { max_abs };
        assert!(
            max_abs < 1e-4 || rel < 1e-4,
            "chunk parity bwd: max_abs={max_abs:.2e} max_mag={max_mag:.6} rel={rel:.2e}"
        );
    }
}

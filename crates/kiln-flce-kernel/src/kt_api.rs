//! `kiln_tensor::Tensor`-typed FLCE surface for kiln-flce-kernel.
//!
//! # Status
//!
//! This is the production FLCE path. Originally drafted as Phase 7
//! (#1082) migration prep alongside the sibling kernel-crate kt-API
//! ports, it became the real surface when the Phase B forward+backward
//! were implemented over [`kiln_tensor`] ops; the candle-typed glue
//! (`flce_candle_shim`) that briefly lived in `kiln-train` was deleted
//! post-#1082, leaving this crate 100% candle-free.
//!
//! This module ships:
//!
//! 1. [`FlceMatmulProviderKt`] — kt-typed chunk-matmul provider trait
//!    (decline-a-chunk contract, see below).
//! 2. [`FlceError`] — error type independent of any other backend's error.
//! 3. [`fused_linear_cross_entropy_phase_b_kt`] — kt-typed forward
//!    entry point. Implements the chunked log-sum-exp reduction
//!    over [`kiln_tensor`] ops, mirroring the original candle Phase A
//!    approach up to floating-point associativity in the chunked reduction.
//! 4. [`fused_linear_cross_entropy_phase_b_backward_kt`] — kt-typed
//!    manual backward producing `dhidden` from `grad_loss`. Mirrors the
//!    candle reference's `phase_b::backward_dhidden` step-by-step
//!    using the kt-typed parallels of every op in the candle code
//!    path. Together with the forward this closes the loop on running
//!    FLCE Phase B end-to-end over kt-tensor.

use std::sync::Arc;

use kiln_tensor::{
    DType as KtDType, Device as KtDevice, Error as KtError, Tensor as KtTensor,
    ops::{
        broadcast_to, exp, index_select, ln, matmul, matmul_rhs_transposed, max_axis, mean_all,
        mul, mul_scalar, scatter_add, sub, sum_axis, to_f32,
    },
};

const DEFAULT_FLCE_ACTIVE_ROW_TILE: usize = 4096;

/// Error type for the kiln-tensor-typed FLCE surface.
///
/// Mirrors `kiln-flash-attn::kt_api::FlashAttnError`: self-contained,
/// with no dependency on any other backend's error type.
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

/// `kiln_tensor::Tensor`-typed chunk-matmul provider for the FLCE loop.
///
/// This trait lets backend crates (kiln-vulkan-kernel, kiln-mps, future
/// kiln-cuda-kernel) implement the chunk matmul over
/// `kiln_tensor::Tensor` directly, without having to round-trip through
/// candle storage. The production Phase B path already runs over
/// `kiln_tensor::Tensor` end-to-end via the kt-typed entry points below.
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
/// Same reason as the original candle-typed provider trait:
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

/// Convenience boxed type used by `_with_provider_kt` entry points.
pub type FlceProviderKt = Arc<dyn FlceMatmulProviderKt>;

/// Active shifted-token metadata produced by the FLCE forward pass.
///
/// The scalar SFT root and the checkpointed SFT tail both run a unit-seed
/// FLCE backward immediately after computing the loss value. Saving the
/// shifted active-row index tensor and labels lets that backward skip the
/// repeated long-context mask scan and device index upload.
#[derive(Debug, Clone)]
pub struct FlceActiveMetadata {
    active_idx: KtTensor,
    active_labels: Vec<u32>,
    running_max: Option<KtTensor>,
    running_sumexp: Option<KtTensor>,
}

/// kt-typed entry point for FLCE Phase B forward.
///
/// Implements the chunked log-sum-exp reduction over `kiln_tensor`
/// ops, mirroring the historical candle Phase A reference up to
/// floating-point associativity in
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
/// See [`fused_linear_cross_entropy_phase_b_backward_kt`] and
/// [`fused_linear_cross_entropy_phase_b_backward_unit_grad_kt`] for the
/// kt-typed manual backward entries.
pub fn fused_linear_cross_entropy_phase_b_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
) -> Result<KtTensor, FlceError> {
    let (loss, _) = fused_linear_cross_entropy_phase_b_with_metadata_kt(
        hidden, head_t, input_ids, label_mask, chunk_size,
    )?;
    Ok(loss)
}

/// Same as [`fused_linear_cross_entropy_phase_b_kt`], plus reusable active
/// shifted-token metadata for the common unit-root backward that follows.
#[doc(hidden)]
pub fn fused_linear_cross_entropy_phase_b_with_metadata_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
) -> Result<(KtTensor, Option<FlceActiveMetadata>), FlceError> {
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
        return Ok((zero_scalar()?, None));
    }

    let vocab_size = head_dims[1];

    let Some(mut active_metadata) =
        build_flce_active_metadata(hidden, input_ids, label_mask, vocab_size, "kt-flce")?
    else {
        return Ok((zero_scalar()?, None));
    };
    let num_active = active_metadata.active_labels.len();
    let active_labels = active_metadata.active_labels.as_slice();

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
    // Derive the destination device from the input `hidden`'s storage so
    // every index/accumulator tensor lands on the same backend. `dispatch2`
    // rejects mixed-device inputs (CPU index + CUDA logits would error), so
    // the CPU-only `from_vec` constructor used to break the chain the moment
    // `hidden` lives on CUDA — exactly the kt-substrate index-op gap that kept
    // the CUDA E2E tests `#[ignore]`-d. `from_vec_on` is the device-parametric
    // companion (#1082) that the already-correct backward
    // (`fused_linear_cross_entropy_phase_b_backward_kt`) and the H6 CE adapter
    // (`tape_forward::try_tape_cross_entropy_from_logits_kt`) both use.
    let device: KtDevice = hidden.device();
    let active_hidden =
        index_select(&shift_hidden, 0, &active_metadata.active_idx).map_err(FlceError::Kt)?;
    let active_hidden_f32 = to_f32(&active_hidden).map_err(FlceError::Kt)?;
    synchronize_flce_reduction_tensor("flce_active_hidden_f32_ready", &active_hidden_f32)?;
    let head_t_f32 = to_f32(head_t).map_err(FlceError::Kt)?;
    synchronize_flce_reduction_tensor("flce_head_t_f32_ready", &head_t_f32)?;

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
    let (running_max_1d, running_sumexp_1d, correct_logit_1d) =
        if let Some(tile) = flce_active_row_tile_len(device, num_active) {
            flce_forward_row_tiled_stats(
                &active_hidden_f32,
                &head_t_f32,
                active_labels,
                vocab_size,
                chunk_size,
                device,
                tile,
                "kt-flce",
            )?
        } else {
            flce_forward_full_active_stats(
                &active_hidden_f32,
                &head_t_f32,
                active_labels,
                vocab_size,
                chunk_size,
                device,
                "kt-flce",
            )?
        };
    synchronize_flce_reduction_tensor("flce_running_max", &running_max_1d)?;
    synchronize_flce_reduction_tensor("flce_running_sumexp", &running_sumexp_1d)?;
    synchronize_flce_reduction_tensor("flce_correct_logit", &correct_logit_1d)?;

    let loss = if flce_gpu_host_scalar_mean_enabled(&device) {
        mean_flce_loss_from_metadata_host_scalar(
            &running_max_1d,
            &running_sumexp_1d,
            &correct_logit_1d,
            device,
        )?
    } else {
        // log_sum_exp = running_max + log(running_sumexp). Both are 1-D
        // [num_active] F32.
        let log_sumexp = ln(&running_sumexp_1d).map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor("flce_log_sumexp", &log_sumexp)?;
        let log_sum_exp =
            kiln_tensor::ops::add(&running_max_1d, &log_sumexp).map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor("flce_log_sum_exp", &log_sum_exp)?;

        // Per-token loss = log_sum_exp - correct_logit. Mean over active rows.
        let per_token_loss = sub(&log_sum_exp, &correct_logit_1d).map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor("flce_per_token_loss", &per_token_loss)?;
        mean_all(&per_token_loss).map_err(FlceError::Kt)?
    };
    synchronize_flce_reduction_tensor("flce_loss", &loss)?;
    active_metadata.running_max = Some(running_max_1d);
    active_metadata.running_sumexp = Some(running_sumexp_1d);
    Ok((loss, Some(active_metadata)))
}

fn synchronize_flce_reduction_tensor(label: &str, tensor: &KtTensor) -> Result<(), FlceError> {
    let _ = label;
    match tensor.device() {
        #[cfg(feature = "rocm")]
        KtDevice::Rocm(device_index) => {
            if kiln_tensor::rocm_capture_arena_active() {
                Ok(())
            } else {
                kiln_tensor::rocm_synchronize_default_stream(device_index)
                    .map_err(|e| FlceError::msg(format!("kt-flce: synchronize {label}: {e}")))
            }
        }
        #[cfg(feature = "cuda")]
        KtDevice::Cuda(device_index) => kiln_tensor::cuda_synchronize_default_stream_for(
            device_index,
            kiln_tensor::CudaSyncReason::TensorHandoff,
        )
        .map_err(|e| FlceError::msg(format!("kt-flce: synchronize {label}: {e}"))),
        _ => Ok(()),
    }
}

fn flce_active_row_tile_len(device: KtDevice, rows: usize) -> Option<usize> {
    let tile = DEFAULT_FLCE_ACTIVE_ROW_TILE.min(rows).max(1);
    if device.is_gpu() && rows > tile {
        return Some(tile);
    }
    let _ = (device, rows, tile);
    None
}

fn flce_update_running_stats_for_chunk(
    running_max: Option<&KtTensor>,
    running_sumexp: Option<&KtTensor>,
    logits_chunk: &KtTensor,
    chunk_max_1d: KtTensor,
    rows: usize,
    chunk_len: usize,
) -> Result<(KtTensor, KtTensor), FlceError> {
    match (running_max, running_sumexp) {
        (None, None) => {
            let chunk_max_2d = chunk_max_1d
                .unsqueeze(1)
                .map_err(FlceError::Kt)?
                .contiguous()
                .map_err(FlceError::Kt)?;
            let chunk_max_b =
                broadcast_to(&chunk_max_2d, &[rows, chunk_len]).map_err(FlceError::Kt)?;
            let shifted = sub(logits_chunk, &chunk_max_b).map_err(FlceError::Kt)?;
            let exped = exp(&shifted).map_err(FlceError::Kt)?;
            let chunk_sumexp_1d = sum_axis(&exped, 1).map_err(FlceError::Kt)?;
            synchronize_flce_reduction_tensor("flce_chunk_sumexp_ready", &chunk_sumexp_1d)?;
            Ok((chunk_max_1d, chunk_sumexp_1d))
        }
        (Some(prev_max), Some(prev_sumexp)) => {
            let new_max_1d = elementwise_max(prev_max, &chunk_max_1d)?;
            let prev_scale_1d =
                exp(&sub(prev_max, &new_max_1d).map_err(FlceError::Kt)?).map_err(FlceError::Kt)?;
            let scaled_prev_1d = mul(prev_sumexp, &prev_scale_1d).map_err(FlceError::Kt)?;
            let new_max_2d = new_max_1d
                .unsqueeze(1)
                .map_err(FlceError::Kt)?
                .contiguous()
                .map_err(FlceError::Kt)?;
            let new_max_b = broadcast_to(&new_max_2d, &[rows, chunk_len]).map_err(FlceError::Kt)?;
            let shifted = sub(logits_chunk, &new_max_b).map_err(FlceError::Kt)?;
            let exped = exp(&shifted).map_err(FlceError::Kt)?;
            let chunk_sumexp_1d = sum_axis(&exped, 1).map_err(FlceError::Kt)?;
            synchronize_flce_reduction_tensor("flce_chunk_sumexp_ready", &chunk_sumexp_1d)?;
            let new_sumexp_1d =
                kiln_tensor::ops::add(&scaled_prev_1d, &chunk_sumexp_1d).map_err(FlceError::Kt)?;
            Ok((new_max_1d, new_sumexp_1d))
        }
        _ => unreachable!("running_max and running_sumexp are set together"),
    }
}

fn flce_correct_logit_for_chunk(
    logits_chunk: &KtTensor,
    active_labels: &[u32],
    chunk_start: usize,
    chunk_len: usize,
    device: KtDevice,
) -> Result<Option<KtTensor>, FlceError> {
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
    if row_hits.is_empty() {
        return Ok(None);
    }

    let rows = active_labels.len();
    let hits = row_hits.len();
    let row_idx_t =
        KtTensor::from_vec_on(device, row_hits.clone(), vec![hits]).map_err(FlceError::Kt)?;
    let selected_rows = index_select(logits_chunk, 0, &row_idx_t).map_err(FlceError::Kt)?;
    let flat_hit_idx: Vec<u32> = col_hits
        .iter()
        .enumerate()
        .map(|(r, &col)| (r as u32) * (chunk_len as u32) + col)
        .collect();
    let flat_hit_idx_t =
        KtTensor::from_vec_on(device, flat_hit_idx, vec![hits]).map_err(FlceError::Kt)?;
    let gathered_1d = selected_rows
        .contiguous()
        .map_err(FlceError::Kt)?
        .flatten_all()
        .map_err(FlceError::Kt)?
        .index_select(&flat_hit_idx_t, 0)
        .map_err(FlceError::Kt)?;
    let scattered = scatter_add(&gathered_1d, 0, &row_idx_t, rows).map_err(FlceError::Kt)?;
    synchronize_flce_reduction_tensor("flce_correct_logit_scatter", &scattered)?;
    Ok(Some(scattered))
}

fn flce_forward_full_active_stats(
    active_hidden_f32: &KtTensor,
    head_t_f32: &KtTensor,
    active_labels: &[u32],
    vocab_size: usize,
    chunk_size: usize,
    device: KtDevice,
    context: &str,
) -> Result<(KtTensor, KtTensor, KtTensor), FlceError> {
    let num_active = active_labels.len();
    let mut running_max: Option<KtTensor> = None;
    let mut running_sumexp: Option<KtTensor> = None;
    let mut correct_logit: Option<KtTensor> = None;

    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let head_chunk = head_t_f32
            .narrow(1, chunk_start, chunk_len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor("flce_head_chunk_ready", &head_chunk)?;

        let logits_chunk =
            flce_matmul_active_rows(active_hidden_f32, &head_chunk, "flce_logits_chunk")?;
        synchronize_flce_reduction_tensor("flce_logits_chunk_ready", &logits_chunk)?;

        let chunk_max_1d = max_axis(&logits_chunk, 1).map_err(FlceError::Kt)?;
        // `chunk_max_1d` is the numerical stabilizer for the online
        // log-sum-exp update below. Treat arbitrary-axis reductions as an
        // async producer boundary before broadcasting/reusing their output.
        synchronize_flce_reduction_tensor("flce_chunk_max_ready", &chunk_max_1d)?;
        let (new_max_1d, new_sumexp_1d) = flce_update_running_stats_for_chunk(
            running_max.as_ref(),
            running_sumexp.as_ref(),
            &logits_chunk,
            chunk_max_1d,
            num_active,
            chunk_len,
        )?;
        running_max = Some(new_max_1d);
        running_sumexp = Some(new_sumexp_1d);

        if let Some(scattered) = flce_correct_logit_for_chunk(
            &logits_chunk,
            active_labels,
            chunk_start,
            chunk_len,
            device,
        )? {
            correct_logit = Some(match correct_logit.take() {
                Some(cur) => kiln_tensor::ops::add(&cur, &scattered).map_err(FlceError::Kt)?,
                None => scattered,
            });
        }
        chunk_start += chunk_len;
    }

    let running_max =
        running_max.ok_or_else(|| FlceError::msg(format!("{context}: vocab_size was 0")))?;
    let running_sumexp =
        running_sumexp.ok_or_else(|| FlceError::msg(format!("{context}: vocab_size was 0")))?;
    let correct_logit = correct_logit.ok_or_else(|| {
        FlceError::msg(format!(
            "{context}: no labels fell inside any vocab chunk — label >= vocab_size?"
        ))
    })?;
    Ok((running_max, running_sumexp, correct_logit))
}

// Judgment keep (round 66): the flat argument list mirrors the per-tile
// kernel inputs (tensors, labels, dims, tile, device) — a parameter
// struct would obscure that 1:1 correspondence.
#[allow(clippy::too_many_arguments)]
fn flce_forward_row_tiled_stats(
    active_hidden_f32: &KtTensor,
    head_t_f32: &KtTensor,
    active_labels: &[u32],
    vocab_size: usize,
    chunk_size: usize,
    device: KtDevice,
    tile: usize,
    context: &str,
) -> Result<(KtTensor, KtTensor, KtTensor), FlceError> {
    let num_active = active_labels.len();
    let running_max_all =
        KtTensor::zeros_on(device, vec![num_active], KtDType::F32).map_err(FlceError::Kt)?;
    let running_sumexp_all =
        KtTensor::zeros_on(device, vec![num_active], KtDType::F32).map_err(FlceError::Kt)?;
    let correct_logit_all =
        KtTensor::zeros_on(device, vec![num_active], KtDType::F32).map_err(FlceError::Kt)?;

    let mut row_start = 0usize;
    while row_start < num_active {
        let row_len = (num_active - row_start).min(tile);
        let active_tile = active_hidden_f32
            .narrow(0, row_start, row_len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        let labels_tile = &active_labels[row_start..row_start + row_len];

        let (running_max_tile, running_sumexp_tile, correct_logit_tile) =
            flce_forward_full_active_stats(
                &active_tile,
                head_t_f32,
                labels_tile,
                vocab_size,
                chunk_size,
                device,
                context,
            )?;

        running_max_all
            .slice_set(&running_max_tile, 0usize, row_start)
            .map_err(FlceError::Kt)?;
        running_sumexp_all
            .slice_set(&running_sumexp_tile, 0usize, row_start)
            .map_err(FlceError::Kt)?;
        correct_logit_all
            .slice_set(&correct_logit_tile, 0usize, row_start)
            .map_err(FlceError::Kt)?;
        row_start += row_len;
    }

    Ok((running_max_all, running_sumexp_all, correct_logit_all))
}

// Judgment keep (round 66): same mirrored-kernel-input argument list as
// the forward driver above.
#[allow(clippy::too_many_arguments)]
fn flce_backward_row_tiled_dhidden(
    active_hidden_f32: &KtTensor,
    head_t_f32: &KtTensor,
    active_labels: &[u32],
    running_max_1d: &KtTensor,
    running_sumexp_1d: &KtTensor,
    vocab_size: usize,
    chunk_size: usize,
    grad_scale: f32,
    hidden_size: usize,
    device: KtDevice,
    tile: usize,
) -> Result<KtTensor, FlceError> {
    let num_active = active_labels.len();
    let dhidden_active = KtTensor::zeros_on(device, vec![num_active, hidden_size], KtDType::F32)
        .map_err(FlceError::Kt)?;

    let mut row_start = 0usize;
    while row_start < num_active {
        let row_len = (num_active - row_start).min(tile);
        let labels_tile = &active_labels[row_start..row_start + row_len];
        let active_tile = active_hidden_f32
            .narrow(0, row_start, row_len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        let running_max_tile = running_max_1d
            .narrow(0, row_start, row_len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        let running_sumexp_tile = running_sumexp_1d
            .narrow(0, row_start, row_len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        let running_max_2d = running_max_tile
            .unsqueeze(1)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        let running_sumexp_2d = running_sumexp_tile
            .unsqueeze(1)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;

        let mut dhidden_tile = KtTensor::zeros_on(device, vec![row_len, hidden_size], KtDType::F32)
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
            let logits_chunk = matmul(&active_tile, &head_chunk).map_err(FlceError::Kt)?;
            synchronize_flce_reduction_tensor("flce_bwd_row_tile_logits_chunk", &logits_chunk)?;

            let max_b =
                broadcast_to(&running_max_2d, &[row_len, chunk_len]).map_err(FlceError::Kt)?;
            let shifted = sub(&logits_chunk, &max_b).map_err(FlceError::Kt)?;
            let exp_chunk = exp(&shifted).map_err(FlceError::Kt)?;
            let sumexp_b =
                broadcast_to(&running_sumexp_2d, &[row_len, chunk_len]).map_err(FlceError::Kt)?;
            let softmax_chunk =
                kiln_tensor::ops::div(&exp_chunk, &sumexp_b).map_err(FlceError::Kt)?;
            let grad_logits_softmax =
                mul_scalar(&softmax_chunk, grad_scale).map_err(FlceError::Kt)?;
            let softmax_contrib =
                matmul_rhs_transposed(&grad_logits_softmax, &head_chunk).map_err(FlceError::Kt)?;
            synchronize_flce_reduction_tensor(
                "flce_bwd_row_tile_softmax_contrib",
                &softmax_contrib,
            )?;

            let mut row_hits: Vec<u32> = Vec::new();
            let mut rel_hits: Vec<u32> = Vec::new();
            for (row_idx, &label) in labels_tile.iter().enumerate() {
                let label = label as usize;
                if label >= chunk_start && label < chunk_end {
                    row_hits.push(row_idx as u32);
                    rel_hits.push((label - chunk_start) as u32);
                }
            }
            let chunk_contrib = if row_hits.is_empty() {
                softmax_contrib
            } else {
                let hits = row_hits.len();
                let row_idx_t =
                    KtTensor::from_vec_on(device, row_hits, vec![hits]).map_err(FlceError::Kt)?;
                let rel_idx_t =
                    KtTensor::from_vec_on(device, rel_hits, vec![hits]).map_err(FlceError::Kt)?;
                let selected_head_cols =
                    index_select(&head_chunk, 1, &rel_idx_t).map_err(FlceError::Kt)?;
                let selected_head_rows = selected_head_cols
                    .t()
                    .map_err(FlceError::Kt)?
                    .contiguous()
                    .map_err(FlceError::Kt)?;
                let selected_weighted =
                    mul_scalar(&selected_head_rows, grad_scale).map_err(FlceError::Kt)?;
                let selected_contrib = scatter_add(&selected_weighted, 0, &row_idx_t, row_len)
                    .map_err(FlceError::Kt)?;
                sub(&softmax_contrib, &selected_contrib).map_err(FlceError::Kt)?
            };

            dhidden_tile =
                kiln_tensor::ops::add(&dhidden_tile, &chunk_contrib).map_err(FlceError::Kt)?;
            chunk_start = chunk_end;
        }

        dhidden_active
            .slice_set(&dhidden_tile, 0usize, row_start)
            .map_err(FlceError::Kt)?;
        row_start += row_len;
    }

    Ok(dhidden_active)
}

fn flce_matmul_active_rows(
    lhs: &KtTensor,
    rhs: &KtTensor,
    label: &str,
) -> Result<KtTensor, FlceError> {
    let rows = lhs.shape().first().copied().unwrap_or(0);
    let device = lhs.device();
    let Some(tile) = flce_active_row_tile_len(device, rows) else {
        let out = matmul(lhs, rhs).map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor(label, &out)?;
        return Ok(out);
    };

    let mut tiles = Vec::new();
    let mut start = 0usize;
    while start < rows {
        let len = (rows - start).min(tile);
        let lhs_tile = lhs
            .narrow(0, start, len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor(label, &lhs_tile)?;
        let out_tile = matmul(&lhs_tile, rhs).map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor(label, &out_tile)?;
        tiles.push(out_tile);
        start += len;
    }

    if tiles.len() == 1 {
        return tiles
            .pop()
            .ok_or_else(|| FlceError::msg(format!("kt-flce: {label} produced no row tiles")));
    }
    let refs: Vec<&KtTensor> = tiles.iter().collect();
    let out = KtTensor::cat(&refs, 0).map_err(FlceError::Kt)?;
    synchronize_flce_reduction_tensor(label, &out)?;
    Ok(out)
}

fn flce_matmul_rhs_transposed_active_rows(
    lhs: &KtTensor,
    rhs: &KtTensor,
    label: &str,
) -> Result<KtTensor, FlceError> {
    let rows = lhs.shape().first().copied().unwrap_or(0);
    let device = lhs.device();
    let Some(tile) = flce_active_row_tile_len(device, rows) else {
        let out = matmul_rhs_transposed(lhs, rhs).map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor(label, &out)?;
        return Ok(out);
    };

    let mut tiles = Vec::new();
    let mut start = 0usize;
    while start < rows {
        let len = (rows - start).min(tile);
        let lhs_tile = lhs
            .narrow(0, start, len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor(label, &lhs_tile)?;
        let out_tile = matmul_rhs_transposed(&lhs_tile, rhs).map_err(FlceError::Kt)?;
        synchronize_flce_reduction_tensor(label, &out_tile)?;
        tiles.push(out_tile);
        start += len;
    }

    if tiles.len() == 1 {
        return tiles
            .pop()
            .ok_or_else(|| FlceError::msg(format!("kt-flce: {label} produced no row tiles")));
    }
    let refs: Vec<&KtTensor> = tiles.iter().collect();
    let out = KtTensor::cat(&refs, 0).map_err(FlceError::Kt)?;
    synchronize_flce_reduction_tensor(label, &out)?;
    Ok(out)
}

fn flce_gpu_host_scalar_mean_enabled(device: &KtDevice) -> bool {
    #[cfg(any(feature = "cuda", feature = "rocm"))]
    {
        match device {
            #[cfg(feature = "cuda")]
            KtDevice::Cuda(_) => true,
            #[cfg(feature = "rocm")]
            KtDevice::Rocm(_) => true,
            _ => false,
        }
    }
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    {
        let _ = device;
        false
    }
}

fn mean_flce_loss_from_metadata_host_scalar(
    running_max: &KtTensor,
    running_sumexp: &KtTensor,
    correct_logit: &KtTensor,
    device: KtDevice,
) -> Result<KtTensor, FlceError> {
    let running_max = flce_host_f32_values("running_max", running_max)?;
    let running_sumexp = flce_host_f32_values("running_sumexp", running_sumexp)?;
    let correct_logit = flce_host_f32_values("correct_logit", correct_logit)?;
    if running_max.len() != running_sumexp.len() || running_max.len() != correct_logit.len() {
        return Err(FlceError::msg(format!(
            "kt-flce: metadata length mismatch for scalar loss: running_max={} running_sumexp={} correct_logit={}",
            running_max.len(),
            running_sumexp.len(),
            correct_logit.len()
        )));
    }
    if running_max.is_empty() {
        return Err(FlceError::msg(
            "kt-flce: cannot compute mean loss over zero active rows",
        ));
    }

    let mut sum = 0.0f32;
    for idx in 0..running_max.len() {
        let max = running_max[idx];
        let sumexp = running_sumexp[idx];
        let correct = correct_logit[idx];
        if !max.is_finite() {
            return Err(FlceError::msg(format!(
                "kt-flce: non-finite running_max before scalar mean at active row {idx}: {max}"
            )));
        }
        if !sumexp.is_finite() || sumexp <= 0.0 {
            return Err(FlceError::msg(format!(
                "kt-flce: invalid running_sumexp before scalar mean at active row {idx}: {sumexp}"
            )));
        }
        if !correct.is_finite() {
            return Err(FlceError::msg(format!(
                "kt-flce: non-finite correct_logit before scalar mean at active row {idx}: {correct}"
            )));
        }

        let loss = max + sumexp.ln() - correct;
        if !loss.is_finite() {
            return Err(FlceError::msg(format!(
                "kt-flce: non-finite scalar loss term at active row {idx}: {loss}"
            )));
        }
        sum += loss;
    }

    let mean = sum / running_max.len() as f32;
    KtTensor::from_vec_on(device, vec![mean], vec![]).map_err(FlceError::Kt)
}

fn flce_host_f32_values(label: &str, tensor: &KtTensor) -> Result<Vec<f32>, FlceError> {
    synchronize_flce_reduction_tensor(label, tensor)?;
    tensor
        .to_device(KtDevice::Cpu)
        .map_err(FlceError::Kt)?
        .to_dtype(KtDType::F32)
        .map_err(FlceError::Kt)?
        .contiguous()
        .map_err(FlceError::Kt)?
        .to_vec::<f32>()
        .map_err(FlceError::Kt)
}

/// kt-typed FLCE Phase B backward — compute `dhidden` from `grad_loss`.
///
/// Manual two-pass backward mirroring the historical candle reference
/// (`phase_b::backward_dhidden`, last at
/// `kiln_train::flce_candle_shim::backward_dhidden` before that module
/// was removed):
///
/// 1. **Pass 1**: recompute `running_max` and `running_sumexp` chunk-by-chunk
///    (identical to forward, minus the `correct_logit` gather).
/// 2. **Pass 2**: for each chunk, recompute `softmax = exp(logits - running_max)
///    / running_sumexp`, accumulate the dense softmax term, then subtract the
///    sparse selected-row correction for labels in that chunk. This computes
///    `dhidden_active += (softmax - one_hot) @ head_chunk.T * grad_loss / N`
///    without materializing the `[active, chunk]` one-hot tile. Finally scatter
///    `dhidden_active` back into the
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
    fused_linear_cross_entropy_phase_b_backward_impl_kt(
        hidden,
        head_t,
        input_ids,
        label_mask,
        chunk_size,
        Some(grad_loss),
        None,
    )
}

/// kt-typed FLCE Phase B backward for the common tape-root seed
/// `d loss / d loss = 1`.
///
/// This is equivalent to calling
/// [`fused_linear_cross_entropy_phase_b_backward_kt`] with a scalar F32 one
/// tensor, but it avoids allocating that seed tensor and avoids the scalar
/// device-to-host read used by the generic seeded path. The trainer's analytic
/// SFT checkpoint tail always uses this unit seed; the generic entry point
/// remains available for composed tape roots with non-unit upstream gradients.
pub fn fused_linear_cross_entropy_phase_b_backward_unit_grad_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
) -> Result<KtTensor, FlceError> {
    fused_linear_cross_entropy_phase_b_backward_impl_kt(
        hidden, head_t, input_ids, label_mask, chunk_size, None, None,
    )
}

/// Unit-root FLCE backward that reuses metadata returned by
/// [`fused_linear_cross_entropy_phase_b_with_metadata_kt`].
#[doc(hidden)]
pub fn fused_linear_cross_entropy_phase_b_backward_unit_grad_with_metadata_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
    active_metadata: &FlceActiveMetadata,
) -> Result<KtTensor, FlceError> {
    fused_linear_cross_entropy_phase_b_backward_impl_kt(
        hidden,
        head_t,
        input_ids,
        label_mask,
        chunk_size,
        None,
        Some(active_metadata),
    )
}

fn fused_linear_cross_entropy_phase_b_backward_impl_kt(
    hidden: &KtTensor,
    head_t: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
    grad_loss: Option<&KtTensor>,
    active_metadata: Option<&FlceActiveMetadata>,
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
    let device: KtDevice = hidden.device();

    // seq_len < 2: no targets, gradient is zero everywhere.
    if seq_len < 2 {
        return zeros_like_hidden_in_dtype(&hidden_dims, original_dtype);
    }

    let active_metadata_owned;
    let active_metadata = match active_metadata {
        Some(metadata) => {
            validate_flce_active_metadata(metadata, device, vocab_size, "kt-flce-bwd")?;
            metadata
        }
        None => {
            active_metadata_owned = build_flce_active_metadata(
                hidden,
                input_ids,
                label_mask,
                vocab_size,
                "kt-flce-bwd",
            )?;
            match active_metadata_owned.as_ref() {
                Some(metadata) => metadata,
                None => return zeros_like_hidden_in_dtype(&hidden_dims, original_dtype),
            }
        }
    };
    let num_active = active_metadata.active_labels.len();
    let active_labels = active_metadata.active_labels.as_slice();
    let active_idx = &active_metadata.active_idx;

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
    let active_hidden = index_select(&shift_hidden, 0, active_idx).map_err(FlceError::Kt)?;
    let active_hidden_f32 = to_f32(&active_hidden).map_err(FlceError::Kt)?;
    let head_t_f32 = to_f32(head_t).map_err(FlceError::Kt)?;

    // -----------------------------------------------------------------
    // Pass 1: use forward softmax stats when available, otherwise
    // recompute running_max + running_sumexp.
    // -----------------------------------------------------------------
    let (running_max_1d, running_sumexp_1d) = match (
        active_metadata.running_max.as_ref(),
        active_metadata.running_sumexp.as_ref(),
    ) {
        (Some(saved_max), Some(saved_sumexp)) => {
            if saved_max.dtype() != KtDType::F32 {
                return Err(FlceError::msg(format!(
                    "kt-flce-bwd: saved running_max dtype {} != F32",
                    saved_max.dtype()
                )));
            }
            if saved_sumexp.dtype() != KtDType::F32 {
                return Err(FlceError::msg(format!(
                    "kt-flce-bwd: saved running_sumexp dtype {} != F32",
                    saved_sumexp.dtype()
                )));
            }
            if saved_max.device() != device {
                return Err(FlceError::msg(format!(
                    "kt-flce-bwd: saved running_max device {} != hidden device {}",
                    saved_max.device().short_name(),
                    device.short_name()
                )));
            }
            if saved_sumexp.device() != device {
                return Err(FlceError::msg(format!(
                    "kt-flce-bwd: saved running_sumexp device {} != hidden device {}",
                    saved_sumexp.device().short_name(),
                    device.short_name()
                )));
            }
            if saved_max.shape() != [num_active] {
                return Err(FlceError::msg(format!(
                    "kt-flce-bwd: saved running_max shape {:?} != [{num_active}]",
                    saved_max.shape()
                )));
            }
            if saved_sumexp.shape() != [num_active] {
                return Err(FlceError::msg(format!(
                    "kt-flce-bwd: saved running_sumexp shape {:?} != [{num_active}]",
                    saved_sumexp.shape()
                )));
            }
            (saved_max.clone(), saved_sumexp.clone())
        }
        (None, None) => {
            let (running_max, running_sumexp, _) =
                if let Some(tile) = flce_active_row_tile_len(device, num_active) {
                    flce_forward_row_tiled_stats(
                        &active_hidden_f32,
                        &head_t_f32,
                        active_labels,
                        vocab_size,
                        chunk_size,
                        device,
                        tile,
                        "kt-flce-bwd",
                    )?
                } else {
                    flce_forward_full_active_stats(
                        &active_hidden_f32,
                        &head_t_f32,
                        active_labels,
                        vocab_size,
                        chunk_size,
                        device,
                        "kt-flce-bwd",
                    )?
                };
            (running_max, running_sumexp)
        }
        _ => {
            return Err(FlceError::msg(
                "kt-flce-bwd: active metadata has incomplete saved softmax stats",
            ));
        }
    };

    // -----------------------------------------------------------------
    // Pass 2: accumulate dhidden_active by chunk.
    // -----------------------------------------------------------------
    //
    // Fold the scalar dL/dloss seed into the per-active-row mean scale once.
    // This avoids allocating and multiplying dense `[active, chunk]` and
    // `[hits, hidden]` grad-loss broadcast tensors inside the chunk loop.
    let grad_loss_scalar = match grad_loss {
        Some(grad_loss) => to_f32(grad_loss)
            .map_err(FlceError::Kt)?
            .to_scalar::<f32>()
            .map_err(FlceError::Kt)?,
        None => 1.0,
    };
    let grad_scale = grad_loss_scalar / (num_active as f32);

    if let Some(tile) = flce_active_row_tile_len(device, num_active) {
        let dhidden_active = flce_backward_row_tiled_dhidden(
            &active_hidden_f32,
            &head_t_f32,
            active_labels,
            &running_max_1d,
            &running_sumexp_1d,
            vocab_size,
            chunk_size,
            grad_scale,
            hidden_size,
            device,
            tile,
        )?;
        let grad_hidden_2d =
            scatter_add(&dhidden_active, 0, active_idx, seq_len).map_err(FlceError::Kt)?;
        let grad_hidden_3d = grad_hidden_2d.unsqueeze(0).map_err(FlceError::Kt)?;
        let out = if original_dtype == KtDType::F32 {
            grad_hidden_3d
        } else {
            kiln_tensor::ops::cast(&grad_hidden_3d, original_dtype).map_err(FlceError::Kt)?
        };
        return Ok(out);
    }

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

    #[cfg(feature = "cuda")]
    if matches!(device, KtDevice::Cuda(_)) {
        let active_labels_t =
            KtTensor::from_vec_on(device, active_labels.to_vec(), vec![num_active])
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
            let logits_chunk =
                flce_matmul_active_rows(&active_hidden_f32, &head_chunk, "flce_bwd_logits_chunk")?;

            kiln_tensor::cuda_flce_grad_logits_chunk_inplace(
                &logits_chunk,
                &active_labels_t,
                &running_max_1d,
                &running_sumexp_1d,
                chunk_start,
                grad_scale,
            )
            .map_err(FlceError::Kt)?;

            let chunk_contrib = flce_matmul_rhs_transposed_active_rows(
                &logits_chunk,
                &head_chunk,
                "flce_bwd_chunk_contrib",
            )?;
            dhidden_active =
                kiln_tensor::ops::add(&dhidden_active, &chunk_contrib).map_err(FlceError::Kt)?;

            chunk_start = chunk_end;
        }

        let grad_hidden_2d =
            scatter_add(&dhidden_active, 0, active_idx, seq_len).map_err(FlceError::Kt)?;
        let grad_hidden_3d = grad_hidden_2d.unsqueeze(0).map_err(FlceError::Kt)?;
        let out = if original_dtype == KtDType::F32 {
            grad_hidden_3d
        } else {
            kiln_tensor::ops::cast(&grad_hidden_3d, original_dtype).map_err(FlceError::Kt)?
        };
        return Ok(out);
    }

    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let chunk_end = chunk_start + chunk_len;

        let head_chunk = head_t_f32
            .narrow(1, chunk_start, chunk_len)
            .map_err(FlceError::Kt)?
            .contiguous()
            .map_err(FlceError::Kt)?;
        let logits_chunk =
            flce_matmul_active_rows(&active_hidden_f32, &head_chunk, "flce_bwd_logits_chunk")?;

        // softmax_chunk = exp(logits_chunk - running_max) / running_sumexp
        let max_b =
            broadcast_to(&running_max_2d, &[num_active, chunk_len]).map_err(FlceError::Kt)?;
        let shifted = sub(&logits_chunk, &max_b).map_err(FlceError::Kt)?;
        let exp_chunk = exp(&shifted).map_err(FlceError::Kt)?;
        let sumexp_b =
            broadcast_to(&running_sumexp_2d, &[num_active, chunk_len]).map_err(FlceError::Kt)?;
        let softmax_chunk = kiln_tensor::ops::div(&exp_chunk, &sumexp_b).map_err(FlceError::Kt)?;

        // softmax contribution: softmax * (grad_loss / N)
        let grad_logits_softmax = mul_scalar(&softmax_chunk, grad_scale).map_err(FlceError::Kt)?;

        // softmax_contrib = grad_logits_softmax @ head_chunk.T
        // shape [num_active, hidden_size]
        let softmax_contrib = flce_matmul_rhs_transposed_active_rows(
            &grad_logits_softmax,
            &head_chunk,
            "flce_bwd_softmax_contrib",
        )?;

        // one-hot contribution: select the `head_chunk.T` row for each label
        // in this chunk and scatter it into the matching active row.
        let mut row_hits: Vec<u32> = Vec::new();
        let mut rel_hits: Vec<u32> = Vec::new();
        for (row_idx, &label) in active_labels.iter().enumerate() {
            let label = label as usize;
            if label >= chunk_start && label < chunk_end {
                row_hits.push(row_idx as u32);
                rel_hits.push((label - chunk_start) as u32);
            }
        }
        let chunk_contrib = if row_hits.is_empty() {
            softmax_contrib
        } else {
            let hits = row_hits.len();
            let row_idx_t =
                KtTensor::from_vec_on(device, row_hits, vec![hits]).map_err(FlceError::Kt)?;
            let rel_idx_t =
                KtTensor::from_vec_on(device, rel_hits, vec![hits]).map_err(FlceError::Kt)?;
            let selected_head_cols =
                index_select(&head_chunk, 1, &rel_idx_t).map_err(FlceError::Kt)?;
            let selected_head_rows = selected_head_cols
                .t()
                .map_err(FlceError::Kt)?
                .contiguous()
                .map_err(FlceError::Kt)?;
            let selected_weighted =
                mul_scalar(&selected_head_rows, grad_scale).map_err(FlceError::Kt)?;
            let selected_contrib = scatter_add(&selected_weighted, 0, &row_idx_t, num_active)
                .map_err(FlceError::Kt)?;
            sub(&softmax_contrib, &selected_contrib).map_err(FlceError::Kt)?
        };

        dhidden_active =
            kiln_tensor::ops::add(&dhidden_active, &chunk_contrib).map_err(FlceError::Kt)?;

        chunk_start = chunk_end;
    }

    // Scatter dhidden_active back into a [seq_len, hidden_size] zero
    // buffer. active_indices live in [0..seq_len-1]; row seq_len-1
    // never contributed (we used hidden[..seq_len-1]) so its gradient
    // stays zero.
    let grad_hidden_2d =
        scatter_add(&dhidden_active, 0, active_idx, seq_len).map_err(FlceError::Kt)?;

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

fn build_flce_active_metadata(
    hidden: &KtTensor,
    input_ids: &[u32],
    label_mask: &[bool],
    vocab_size: usize,
    context: &str,
) -> Result<Option<FlceActiveMetadata>, FlceError> {
    if input_ids.len() < 2 {
        return Ok(None);
    }

    let mut active_positions = Vec::new();
    let mut active_labels = Vec::new();
    for (shift_idx, &is_active) in label_mask[1..].iter().enumerate() {
        if !is_active {
            continue;
        }
        let label = input_ids[shift_idx + 1];
        if label as usize >= vocab_size {
            return Err(FlceError::msg(format!(
                "{context}: label {label} >= vocab_size {vocab_size}"
            )));
        }
        active_positions.push(shift_idx as u32);
        active_labels.push(label);
    }

    if active_positions.is_empty() {
        return Ok(None);
    }

    let num_active = active_positions.len();
    let active_idx = KtTensor::from_vec_on(hidden.device(), active_positions, vec![num_active])
        .map_err(FlceError::Kt)?;
    Ok(Some(FlceActiveMetadata {
        active_idx,
        active_labels,
        running_max: None,
        running_sumexp: None,
    }))
}

fn validate_flce_active_metadata(
    metadata: &FlceActiveMetadata,
    device: KtDevice,
    vocab_size: usize,
    context: &str,
) -> Result<(), FlceError> {
    let num_active = metadata.active_labels.len();
    if num_active == 0 {
        return Err(FlceError::msg(format!(
            "{context}: active metadata must contain at least one row"
        )));
    }
    if metadata.active_idx.dtype() != KtDType::U32 {
        return Err(FlceError::msg(format!(
            "{context}: active_idx dtype {} != U32",
            metadata.active_idx.dtype()
        )));
    }
    if metadata.active_idx.device() != device {
        return Err(FlceError::msg(format!(
            "{context}: active_idx device {} != hidden device {}",
            metadata.active_idx.device().short_name(),
            device.short_name()
        )));
    }
    if metadata.active_idx.shape() != [num_active] {
        return Err(FlceError::msg(format!(
            "{context}: active_idx shape {:?} != [{num_active}]",
            metadata.active_idx.shape()
        )));
    }
    for &label in &metadata.active_labels {
        if label as usize >= vocab_size {
            return Err(FlceError::msg(format!(
                "{context}: label {label} >= vocab_size {vocab_size}"
            )));
        }
    }
    Ok(())
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
        let loss =
            fused_linear_cross_entropy_phase_b_kt(&hidden, &head, &ids, &mask, 4).expect("forward");
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
        let loss =
            fused_linear_cross_entropy_phase_b_kt(&hidden, &head, &ids, &mask, 4).expect("forward");
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

        let l_single = scalar_value(
            fused_linear_cross_entropy_phase_b_kt(&hidden, &head, &ids, &mask, v).unwrap(),
        );
        let l_multi = scalar_value(
            fused_linear_cross_entropy_phase_b_kt(&hidden, &head, &ids, &mask, 4).unwrap(),
        );
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

    // Device-agnostic read for GPU tensors — used only by the
    // `cuda`-gated tests below (e.g.
    // fused_linear_cross_entropy_phase_b_backward_kt_cuda_sparse_chunk_runs),
    // so it looks dead under default features.
    #[allow(dead_code)]
    fn read_f32_vec_any(t: &KtTensor) -> Vec<f32> {
        t.to_dtype(KtDType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec()
            .unwrap()
    }

    /// Backward smoke: short-seq early-return produces zero grad.
    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_kt_short_seq_returns_zero() {
        let hidden = dummy_hidden(1, 4);
        let head = dummy_head_t(4, 8);
        let ids = vec![0u32];
        let mask = vec![true];
        let grad_loss = KtTensor::from_vec(vec![1.0f32], vec![]).unwrap();
        let g = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden, &head, &ids, &mask, 4, &grad_loss,
        )
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
        let g = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden, &head, &ids, &mask, 4, &grad_loss,
        )
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
            &hidden, &head, &ids, &mask, 4, &grad_loss,
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

        let hidden_kt =
            KtTensor::from_vec(hidden_vec.clone(), vec![1, seq_len, hidden_size]).unwrap();
        let head_kt = KtTensor::from_vec(head_vec.clone(), vec![hidden_size, vocab_size]).unwrap();
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
        let rel = if max_mag > 1e-6 {
            max_abs / max_mag
        } else {
            max_abs
        };
        assert!(
            max_abs < 1e-4 || rel < 1e-4,
            "chunk parity bwd: max_abs={max_abs:.2e} max_mag={max_mag:.6} rel={rel:.2e}"
        );
    }

    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_unit_grad_matches_seed_one() {
        let seq_len = 8;
        let hidden_size = 4;
        let vocab_size = 16;

        let hidden_vec: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32 * 0.017).sin() * 0.2)
            .collect();
        let head_vec: Vec<f32> = (0..hidden_size * vocab_size)
            .map(|i| ((i as f32 + 11.0) * 0.007).cos() * 0.15)
            .collect();
        let ids: Vec<u32> = (0..seq_len as u32)
            .map(|i| (i * 13 + 3) % vocab_size as u32)
            .collect();
        let mask: Vec<bool> = (0..seq_len).map(|i| i != 0 && i != 3).collect();

        let hidden = KtTensor::from_vec(hidden_vec, vec![1, seq_len, hidden_size]).unwrap();
        let head = KtTensor::from_vec(head_vec, vec![hidden_size, vocab_size]).unwrap();
        let grad_loss = KtTensor::from_vec(vec![1.0f32], vec![]).unwrap();

        let seeded = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden, &head, &ids, &mask, 4, &grad_loss,
        )
        .unwrap();
        let unit = fused_linear_cross_entropy_phase_b_backward_unit_grad_kt(
            &hidden, &head, &ids, &mask, 4,
        )
        .unwrap();

        let seeded = read_f32_vec(&seeded);
        let unit = read_f32_vec(&unit);
        assert_eq!(seeded.len(), unit.len());
        for (i, (a, b)) in seeded.iter().zip(unit.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "unit grad mismatch at {i}: seeded={a} unit={b}"
            );
        }
    }

    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_reuses_forward_metadata() {
        let seq_len = 8;
        let hidden_size = 4;
        let vocab_size = 16;

        let hidden_vec: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32 * 0.021).sin() * 0.2)
            .collect();
        let head_vec: Vec<f32> = (0..hidden_size * vocab_size)
            .map(|i| ((i as f32 + 7.0) * 0.009).cos() * 0.15)
            .collect();
        let ids: Vec<u32> = (0..seq_len as u32)
            .map(|i| (i * 11 + 5) % vocab_size as u32)
            .collect();
        let mask: Vec<bool> = (0..seq_len).map(|i| i != 0 && i != 4).collect();

        let hidden = KtTensor::from_vec(hidden_vec, vec![1, seq_len, hidden_size]).unwrap();
        let head = KtTensor::from_vec(head_vec, vec![hidden_size, vocab_size]).unwrap();

        let (_loss, metadata) =
            fused_linear_cross_entropy_phase_b_with_metadata_kt(&hidden, &head, &ids, &mask, 4)
                .unwrap();
        let metadata = metadata.expect("active metadata");
        assert!(
            metadata.running_max.is_some(),
            "forward metadata should retain running max for unit-root backward"
        );
        assert!(
            metadata.running_sumexp.is_some(),
            "forward metadata should retain running sumexp for unit-root backward"
        );
        let regular = fused_linear_cross_entropy_phase_b_backward_unit_grad_kt(
            &hidden, &head, &ids, &mask, 4,
        )
        .unwrap();
        let reused = fused_linear_cross_entropy_phase_b_backward_unit_grad_with_metadata_kt(
            &hidden, &head, &ids, &mask, 4, &metadata,
        )
        .unwrap();

        let regular = read_f32_vec(&regular);
        let reused = read_f32_vec(&reused);
        assert_eq!(regular.len(), reused.len());
        for (i, (a, b)) in regular.iter().zip(reused.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-6,
                "metadata grad mismatch at {i}: regular={a} reused={b}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_linear_cross_entropy_phase_b_backward_kt_cuda_sparse_chunk_runs() {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!("[FLCE-CUDA] no CUDA device; skipping");
            return;
        }

        let device = KtDevice::Cuda(0);
        let seq_len = 7;
        let hidden_size = 5;
        let vocab_size = 17;
        let hidden_vec: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| ((i as f32 + 1.0) * 0.019).sin() * 0.25)
            .collect();
        let head_vec: Vec<f32> = (0..hidden_size * vocab_size)
            .map(|i| ((i as f32 + 5.0) * 0.013).cos() * 0.2)
            .collect();
        let ids: Vec<u32> = vec![1, 5, 9, 13, 16, 4, 7];
        let mask: Vec<bool> = vec![false, true, true, false, true, false, true];
        let hidden = KtTensor::from_vec_on(device, hidden_vec, vec![1, seq_len, hidden_size])
            .expect("cuda hidden");
        let head = KtTensor::from_vec_on(device, head_vec, vec![hidden_size, vocab_size])
            .expect("cuda head");
        let grad_loss =
            KtTensor::from_vec_on(device, vec![1.0f32], vec![]).expect("cuda grad_loss");

        let g_single = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden, &head, &ids, &mask, vocab_size, &grad_loss,
        )
        .expect("single chunk cuda backward");
        let g_multi = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden, &head, &ids, &mask, 4, &grad_loss,
        )
        .expect("multi chunk cuda backward");
        let g_unit = fused_linear_cross_entropy_phase_b_backward_unit_grad_kt(
            &hidden, &head, &ids, &mask, 4,
        )
        .expect("unit-seed cuda backward");

        let single = read_f32_vec_any(&g_single);
        let multi = read_f32_vec_any(&g_multi);
        let unit = read_f32_vec_any(&g_unit);
        assert_eq!(single.len(), multi.len());
        assert_eq!(multi.len(), unit.len());
        for (i, (a, b)) in single.iter().zip(multi.iter()).enumerate() {
            let tol = 2e-4f32.max(2e-4 * a.abs());
            assert!(
                (a - b).abs() <= tol,
                "cuda sparse FLCE bwd drift at {i}: single={a} multi={b} tol={tol}"
            );
        }
        for (i, (a, b)) in multi.iter().zip(unit.iter()).enumerate() {
            let tol = 2e-4f32.max(2e-4 * a.abs());
            assert!(
                (a - b).abs() <= tol,
                "cuda unit-seed FLCE bwd drift at {i}: seeded={a} unit={b} tol={tol}"
            );
        }
    }
}

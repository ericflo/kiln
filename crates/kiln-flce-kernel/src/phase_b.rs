//! FLCE **Phase B** — manual-backward [`CustomOp1`] that closes the
//! T=8192 SFT autograd-retention bottleneck on A6000.
//!
//! # Why a CustomOp1
//!
//! Phase A keeps `logits_chunk`, `shifted`, and `shifted.exp()` live for
//! candle autograd because the scalar loss depends on `running_sumexp`,
//! which is a sum-tree across all chunks' `shifted.exp()` nodes. At T=8192
//! with V=248320 / chunk=4096 = 61 chunks, that is 61 × 3 × 127.81 MiB ≈
//! **23 GiB of intermediates pinned in the forward graph**. PR #646 traced
//! the OOM allocation directly to this pattern (see
//! `docs/audits/PHASE10_MODE_B_TRACE.md`).
//!
//! Phase B replaces the autograd graph with a [`CustomOp1`] whose
//! `cpu_fwd`/`cuda_fwd` runs the chunked forward in a function-local scope,
//! producing only the scalar loss as output. All chunk intermediates are
//! dropped on return because nothing outside the function references them.
//! The op's `bwd()` recomputes the chunks to produce `dhidden` without
//! retaining a forward graph between iterations: each chunk's
//! intermediates are local to its loop iteration and dropped before the
//! next chunk runs. Estimated peak-VRAM saving at T=8192: ~22 GiB.
//!
//! # Inputs treated as op state, not autograd inputs
//!
//! Only `hidden` is an autograd input — the LM head (`head_t`) is frozen
//! during LoRA training, and `input_ids` / `label_mask` are integer/bool
//! data. We capture them by value into the [`FlceCustomOp`] struct so the
//! op closes over them.
//!
//! # Math
//!
//! Loss is identical to Phase A:
//! ```text
//! loss = (1/N) * sum_i (log_sum_exp(logits_i) - logits[i, label_i])
//! ```
//! Backward (per-row, per-vocab-element):
//! ```text
//! d_loss / d_logits[i,j] = (1/N) * (softmax[i,j] - 1{j == label_i})
//! d_loss / d_active_hidden[i, :] = sum_j (d_loss/d_logits[i,j]) * head_t[:, j]
//!                                = (1/N) * (softmax_i - one_hot_label_i) @ head_t.T
//! ```
//! Chunked: accumulate `dhidden_active += (softmax_chunk - one_hot_chunk) @ head_chunk.T`
//! across the 61 chunks. The first pass through the loop computes the
//! global `running_max` and `running_sumexp` (needed to evaluate
//! `softmax = exp(logits - global_max) / global_sumexp` on the second
//! pass) — same recompute trick as the forward.

use anyhow::{Context, Result, anyhow};
#[cfg(feature = "cuda")]
use candle_core::backend::BackendStorage;
use candle_core::op::BackpropOp;
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, DType, Device, Layout, Shape, Storage, Tensor, D,
};

use crate::DEFAULT_CHUNK_SIZE;

/// Phase B entry point: chunked FLCE with a manual-backward [`CustomOp1`].
///
/// Behaves identically to [`crate::fused_linear_cross_entropy`] up to
/// floating-point associativity in the reduction across chunks, but routes
/// the autograd graph through a custom op so chunk intermediates do not
/// pin ~23 GiB of VRAM at T=8192 SFT.
///
/// See module docs for the reasoning + math.
pub fn fused_linear_cross_entropy_phase_b(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
) -> Result<Tensor> {
    let seq_len = input_ids.len();
    if seq_len < 2 {
        return Tensor::new(0.0f32, device).context("allocate zero loss scalar for seq_len < 2");
    }
    if label_mask.len() != seq_len {
        return Err(anyhow!(
            "label_mask length {} does not match input_ids length {}",
            label_mask.len(),
            seq_len,
        ));
    }
    if chunk_size == 0 {
        return Err(anyhow!("chunk_size must be > 0"));
    }
    let head_dims = head_t.dims();
    if head_dims.len() != 2 {
        return Err(anyhow!(
            "head_t must be 2-D [hidden_size, vocab_size]; got {:?}",
            head_dims
        ));
    }
    let hidden_dims = hidden.dims();
    if hidden_dims.len() != 3 {
        return Err(anyhow!(
            "hidden must be 3-D [1, seq_len, hidden_size]; got {:?}",
            hidden_dims
        ));
    }
    if hidden_dims[0] != 1 {
        return Err(anyhow!(
            "hidden batch dim must be 1; got {:?}",
            hidden_dims
        ));
    }
    if hidden_dims[2] != head_dims[0] {
        return Err(anyhow!(
            "hidden hidden_size {} != head_t hidden_size {}",
            hidden_dims[2],
            head_dims[0],
        ));
    }

    // Short-circuit when no positions are active. Phase A returns a zero
    // tensor with no autograd parent in this case; Phase B does the same so
    // calling .backward() on the result is a no-op for `hidden`.
    let active_count = label_mask[1..].iter().filter(|&&m| m).count();
    if active_count == 0 {
        return Tensor::new(0.0f32, device).context("allocate zero loss scalar (no active rows)");
    }

    // Apply the custom op. The op closes over head_t / input_ids / label_mask /
    // chunk_size; only `hidden` is the autograd input.
    let op = FlceCustomOp {
        head_t: head_t.clone(),
        input_ids: input_ids.to_vec(),
        label_mask: label_mask.to_vec(),
        chunk_size,
    };
    hidden.apply_op1(op).map_err(Into::into)
}

/// CustomOp1 wrapper for Phase B. `apply_op1(hidden)` -> scalar f32 loss.
struct FlceCustomOp {
    /// `[hidden_size, vocab_size]` transposed lm_head — frozen during LoRA
    /// training, so it is captured here as op state rather than an autograd
    /// input.
    head_t: Tensor,
    /// Token ids for the sequence (length `seq_len`).
    input_ids: Vec<u32>,
    /// Loss mask aligned with `input_ids`; positions where `label_mask[i+1]`
    /// is true contribute to the loss.
    label_mask: Vec<bool>,
    /// Chunk size along the vocab dim. Use [`DEFAULT_CHUNK_SIZE`] unless
    /// tuning.
    chunk_size: usize,
}

impl CustomOp1 for FlceCustomOp {
    fn name(&self) -> &'static str {
        "kiln-flce-phase-b"
    }

    fn cpu_fwd(
        &self,
        s_hidden: &CpuStorage,
        l_hidden: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        // Build a leaf hidden tensor from the input storage so we can run
        // candle ops over it. The clone is a single Vec memcpy (~80 MiB at
        // T=8192 H=2560 bf16) — paid once vs the ~23 GiB autograd-retention
        // cost it eliminates.
        let storage = Storage::Cpu(s_hidden.clone());
        let hidden_shape = Shape::from(l_hidden.shape().dims());
        let hidden_leaf =
            Tensor::from_storage(storage, hidden_shape, BackpropOp::none(), false);

        let loss = forward_loss(
            &hidden_leaf,
            &self.head_t,
            &self.input_ids,
            &self.label_mask,
            self.chunk_size,
        )
        .map_err(|e| candle_core::Error::Msg(format!("flce phase b cpu_fwd: {e:#}")))?;

        Ok((CpuStorage::F32(vec![loss]), Shape::from(())))
    }

    fn cuda_fwd(
        &self,
        s_hidden: &CudaStorage,
        l_hidden: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (s_hidden, l_hidden);
            return Err(candle_core::Error::Msg(
                "flce phase b cuda_fwd: kiln-flce-kernel built without `cuda` feature".into(),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            // `try_clone` invokes a single-launch device-to-device copy
            // (`Clone.map(...)` in the cuda backend) — ~40 MiB at T=8192
            // H=2560 bf16. Avoids reallocating the input.
            let storage = Storage::Cuda(s_hidden.try_clone(l_hidden)?);
            let hidden_shape = Shape::from(l_hidden.shape().dims());
            let hidden_leaf =
                Tensor::from_storage(storage, hidden_shape, BackpropOp::none(), false);

            let loss_value = forward_loss(
                &hidden_leaf,
                &self.head_t,
                &self.input_ids,
                &self.label_mask,
                self.chunk_size,
            )
            .map_err(|e| candle_core::Error::Msg(format!("flce phase b cuda_fwd: {e:#}")))?;

            let device = s_hidden.device();
            let out_slice = device.clone_htod(&[loss_value])?;
            Ok((
                CudaStorage::wrap_cuda_slice(out_slice, device.clone()),
                Shape::from(()),
            ))
        }
    }

    fn bwd(
        &self,
        hidden: &Tensor,
        _loss: &Tensor,
        grad_loss: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        backward_dhidden(
            hidden,
            &self.head_t,
            &self.input_ids,
            &self.label_mask,
            self.chunk_size,
            grad_loss,
        )
        .map(Some)
        .map_err(|e| candle_core::Error::Msg(format!("flce phase b bwd: {e:#}")))
    }
}

/// Forward implementation that runs over a leaf hidden tensor and returns
/// only the scalar loss value (host f32). Chunk intermediates are local to
/// this function and dropped on return — the whole point of Phase B.
///
/// `running_max` / `running_sumexp` / `correct_logit` are `.detach()`-ed
/// after each accumulation step so candle autograd does not build a chain
/// of dependencies through the chunk loop. (The leaf input has no autograd
/// parent so the graph has nowhere to extend back to anyway, but detaching
/// avoids a `O(num_chunks)` accumulator chain that could otherwise hold
/// per-chunk tensors live until the final `mean_all` runs.)
fn forward_loss(
    hidden_leaf: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
) -> Result<f32> {
    let device = hidden_leaf.device();
    let seq_len = input_ids.len();
    debug_assert!(seq_len >= 2);
    debug_assert_eq!(label_mask.len(), seq_len);

    let hidden_2d = hidden_leaf
        .squeeze(0)
        .context("squeeze batch dim from hidden")?;
    let shift_hidden = hidden_2d
        .narrow(0, 0, seq_len - 1)
        .context("narrow shift_hidden")?;
    let shift_labels: &[u32] = &input_ids[1..];
    let shift_mask: &[bool] = &label_mask[1..];

    let active_positions: Vec<u32> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    debug_assert!(!active_positions.is_empty(), "caller short-circuits empty");

    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i as usize])
        .collect();
    let num_active = active_positions.len();

    let active_indices = Tensor::new(active_positions.as_slice(), device)
        .context("build active position indices")?;
    let active_hidden = shift_hidden
        .index_select(&active_indices, 0)
        .context("gather active_hidden rows")?;

    let active_hidden_f32 = active_hidden.to_dtype(DType::F32)?;
    let head_t_f32 = head_t.to_dtype(DType::F32)?;
    let vocab_size = head_t.dim(1)?;

    let mut running_max: Option<Tensor> = None;
    let mut running_sumexp: Option<Tensor> = None;
    let mut correct_logit: Option<Tensor> = None;

    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);

        // Materialize a contiguous head chunk — `narrow` on a `[H, V]`
        // tensor preserves stride `[V, 1]` along the V axis, which CUDA
        // matmul rejects. This matches Phase A's fix (PR #631).
        let head_chunk = head_t_f32
            .narrow(1, chunk_start, chunk_len)
            .context("slice head_t chunk")?
            .contiguous()
            .context("contiguous head_t chunk for matmul")?;

        let logits_chunk = active_hidden_f32
            .matmul(&head_chunk)
            .context("matmul active_hidden_f32 @ head_chunk")?;

        let chunk_max = logits_chunk
            .max_keepdim(D::Minus1)
            .context("max_keepdim on logits_chunk")?;

        let (new_max, new_sumexp) = match (running_max.as_ref(), running_sumexp.as_ref()) {
            (None, None) => {
                let shifted = (&logits_chunk
                    - chunk_max.broadcast_as(logits_chunk.shape())?)?;
                let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                (chunk_max.detach(), chunk_sumexp.detach())
            }
            (Some(prev_max), Some(prev_sumexp)) => {
                let new_max = prev_max.maximum(&chunk_max)?;
                let prev_scale = (prev_max - &new_max)?.exp()?;
                let scaled_prev = prev_sumexp.broadcast_mul(&prev_scale)?;
                let shifted =
                    (&logits_chunk - new_max.broadcast_as(logits_chunk.shape())?)?;
                let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                let new_sumexp = (scaled_prev + chunk_sumexp)?;
                (new_max.detach(), new_sumexp.detach())
            }
            _ => unreachable!("running_max and running_sumexp are set together"),
        };
        running_max = Some(new_max);
        running_sumexp = Some(new_sumexp);

        // Gather correct logits for labels falling inside this chunk.
        let chunk_end = chunk_start + chunk_len;
        let mut chunk_hits: Vec<(u32, u32)> = Vec::new();
        for (row_idx, &label) in active_labels.iter().enumerate() {
            let label = label as usize;
            if label >= chunk_start && label < chunk_end {
                chunk_hits.push((row_idx as u32, (label - chunk_start) as u32));
            }
        }
        if !chunk_hits.is_empty() {
            let rows: Vec<u32> = chunk_hits.iter().map(|&(r, _)| r).collect();
            let cols: Vec<u32> = chunk_hits.iter().map(|&(_, c)| c).collect();
            let row_idx = Tensor::new(rows.as_slice(), device)?;
            let col_idx_2d = Tensor::new(cols.as_slice(), device)?.unsqueeze(1)?;
            let selected_rows = logits_chunk.index_select(&row_idx, 0)?;
            let gathered = selected_rows.gather(&col_idx_2d, 1)?.squeeze(1)?;
            let mut cur = match correct_logit.take() {
                Some(t) => t,
                None => Tensor::zeros(num_active, DType::F32, device)?,
            };
            cur = cur.index_add(&row_idx, &gathered, 0)?;
            correct_logit = Some(cur.detach());
        }

        chunk_start = chunk_end;
    }

    let running_max = running_max.ok_or_else(|| anyhow!("vocab_size was 0"))?;
    let running_sumexp = running_sumexp.ok_or_else(|| anyhow!("vocab_size was 0"))?;
    let correct_logit = correct_logit.ok_or_else(|| {
        anyhow!("no labels fell inside any vocab chunk — label >= vocab_size?")
    })?;

    let log_sum_exp = (running_max.squeeze(D::Minus1)?
        + running_sumexp.squeeze(D::Minus1)?.log()?)?;
    let per_token_loss = (log_sum_exp - correct_logit)?;
    let loss = per_token_loss.mean_all()?;
    Ok(loss.to_scalar::<f32>()?)
}

/// Backward implementation. Runs the chunk loop twice:
///
/// 1. **Pass 1**: recompute `running_max` and `running_sumexp` (needed by
///    pass 2 for `softmax = exp(logits - global_max) / global_sumexp`).
///    Identical to the forward chunk loop minus the `correct_logit` gather.
/// 2. **Pass 2**: recompute each chunk's `softmax`, build the one-hot
///    label mask for that chunk, and accumulate
///    `dhidden_active += (softmax - one_hot) @ head_chunk.T * grad_loss / N`.
///    `dhidden_active` is `.detach()`-ed after each accumulation so the
///    autograd graph for the chunk loop does not pin
///    per-chunk `softmax`/`one_hot`/`chunk_contrib` tensors live across
///    iterations.
///
/// Returns `dhidden` as a `[1, seq_len, hidden_size]` tensor with the
/// original `hidden.dtype()`. Rows outside `active_positions` and the
/// `seq_len-1` row are zero.
fn backward_dhidden(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
    grad_loss: &Tensor,
) -> Result<Tensor> {
    let device = hidden.device();
    let dtype = hidden.dtype();
    let seq_len = input_ids.len();
    debug_assert!(seq_len >= 2);
    debug_assert_eq!(label_mask.len(), seq_len);

    let hidden_dims = hidden.dims();
    let hidden_size = hidden_dims[2];

    let hidden_2d = hidden.squeeze(0)?;
    let shift_hidden = hidden_2d.narrow(0, 0, seq_len - 1)?;
    let shift_labels: &[u32] = &input_ids[1..];
    let shift_mask: &[bool] = &label_mask[1..];

    let active_positions: Vec<u32> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();

    if active_positions.is_empty() {
        // Loss was 0 (no active rows); gradient is zero everywhere.
        return Ok(Tensor::zeros(hidden.shape(), dtype, device)?);
    }

    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i as usize])
        .collect();
    let num_active = active_positions.len();

    let active_indices = Tensor::new(active_positions.as_slice(), device)?;
    let active_hidden = shift_hidden.index_select(&active_indices, 0)?;
    let active_hidden_f32 = active_hidden.to_dtype(DType::F32)?;
    let head_t_f32 = head_t.to_dtype(DType::F32)?;
    let vocab_size = head_t.dim(1)?;

    // Pass 1: recompute running_max + running_sumexp.
    let mut running_max: Option<Tensor> = None;
    let mut running_sumexp: Option<Tensor> = None;
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let head_chunk = head_t_f32
            .narrow(1, chunk_start, chunk_len)?
            .contiguous()?;
        let logits_chunk = active_hidden_f32.matmul(&head_chunk)?;
        let chunk_max = logits_chunk.max_keepdim(D::Minus1)?;
        let (new_max, new_sumexp) = match (running_max.as_ref(), running_sumexp.as_ref()) {
            (None, None) => {
                let shifted =
                    (&logits_chunk - chunk_max.broadcast_as(logits_chunk.shape())?)?;
                let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                (chunk_max.detach(), chunk_sumexp.detach())
            }
            (Some(prev_max), Some(prev_sumexp)) => {
                let new_max = prev_max.maximum(&chunk_max)?;
                let prev_scale = (prev_max - &new_max)?.exp()?;
                let scaled_prev = prev_sumexp.broadcast_mul(&prev_scale)?;
                let shifted =
                    (&logits_chunk - new_max.broadcast_as(logits_chunk.shape())?)?;
                let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                let new_sumexp = (scaled_prev + chunk_sumexp)?;
                (new_max.detach(), new_sumexp.detach())
            }
            _ => unreachable!(),
        };
        running_max = Some(new_max);
        running_sumexp = Some(new_sumexp);
        chunk_start += chunk_len;
    }
    let running_max = running_max.ok_or_else(|| anyhow!("vocab_size was 0"))?;
    let running_sumexp = running_sumexp.ok_or_else(|| anyhow!("vocab_size was 0"))?;

    // Pass 2: accumulate dhidden_active by chunk.
    let grad_loss_f32 = grad_loss.to_dtype(DType::F32)?;
    let inv_n = 1.0f64 / (num_active as f64);

    let mut dhidden_active = Tensor::zeros((num_active, hidden_size), DType::F32, device)?;
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let head_chunk = head_t_f32
            .narrow(1, chunk_start, chunk_len)?
            .contiguous()?;
        let logits_chunk = active_hidden_f32.matmul(&head_chunk)?;
        let shifted = (&logits_chunk - running_max.broadcast_as(logits_chunk.shape())?)?;
        let exp_chunk = shifted.exp()?;
        let softmax_chunk =
            exp_chunk.broadcast_div(&running_sumexp.broadcast_as(logits_chunk.shape())?)?;

        // One-hot mask for label hits inside this chunk.
        let chunk_end = chunk_start + chunk_len;
        let mut one_hot_data: Vec<f32> = vec![0.0; num_active * chunk_len];
        for (row_idx, &label) in active_labels.iter().enumerate() {
            let label = label as usize;
            if label >= chunk_start && label < chunk_end {
                let col = label - chunk_start;
                one_hot_data[row_idx * chunk_len + col] = 1.0;
            }
        }
        let one_hot = Tensor::from_vec(one_hot_data, (num_active, chunk_len), device)?;

        // grad_logits_chunk = (softmax - one_hot) * (grad_loss / N)
        let diff = (softmax_chunk - one_hot)?;
        let scaled = diff.affine(inv_n, 0.0)?;
        let grad_logits_chunk = scaled.broadcast_mul(&grad_loss_f32)?;

        // chunk_contrib = grad_logits_chunk @ head_chunk.T  (shape [num_active, hidden_size])
        let head_chunk_t = head_chunk.t()?.contiguous()?;
        let chunk_contrib = grad_logits_chunk.matmul(&head_chunk_t)?;

        dhidden_active = (&dhidden_active + chunk_contrib)?.detach();

        chunk_start = chunk_end;
    }

    // Scatter dhidden_active back into a [seq_len, hidden_size] zero buffer.
    // active_indices live in [0..seq_len-1]; row seq_len-1 of `hidden`
    // never contributed to a logit (we used hidden[..seq_len-1]) so its
    // gradient stays zero.
    let mut grad_hidden_2d = Tensor::zeros((seq_len, hidden_size), DType::F32, device)?;
    grad_hidden_2d = grad_hidden_2d.index_add(&active_indices, &dhidden_active, 0)?;

    let grad_hidden_3d = grad_hidden_2d.unsqueeze(0)?;
    let dhidden = grad_hidden_3d.to_dtype(dtype)?;
    Ok(dhidden)
}

/// Re-export so callers that don't tune chunk_size can use the same default
/// as Phase A.
#[allow(dead_code)]
pub(crate) const _DEFAULT_CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;

    /// Naive softmax cross-entropy loss against the full materialized logits.
    /// Used as the gradient oracle for [`backward_dhidden`] parity.
    fn naive_softmax_ce(
        hidden: &Tensor,
        head_t: &Tensor,
        input_ids: &[u32],
        label_mask: &[bool],
        device: &Device,
    ) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let hidden_2d = hidden.squeeze(0)?;
        let shift_hidden = hidden_2d.narrow(0, 0, seq_len - 1)?;
        let shift_labels: Vec<u32> = input_ids[1..].to_vec();
        let shift_mask: Vec<bool> = label_mask[1..].to_vec();

        let active_positions: Vec<u32> = shift_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
            .collect();
        if active_positions.is_empty() {
            return Ok(Tensor::new(0.0f32, device)?);
        }
        let indices = Tensor::new(active_positions.as_slice(), device)?;
        let active_hidden = shift_hidden.index_select(&indices, 0)?;
        let logits = active_hidden
            .to_dtype(DType::F32)?
            .matmul(&head_t.to_dtype(DType::F32)?)?;
        let log_sum_exp = logits.log_sum_exp(D::Minus1)?;
        let active_labels: Vec<u32> = active_positions
            .iter()
            .map(|&i| shift_labels[i as usize])
            .collect();
        let labels_tensor = Tensor::new(active_labels.as_slice(), device)?;
        let labels_2d = labels_tensor.unsqueeze(1)?;
        let correct_logits = logits.gather(&labels_2d, 1)?.squeeze(1)?;
        let per_token_loss = (log_sum_exp - correct_logits)?;
        let loss = per_token_loss.mean_all()?;
        Ok(loss)
    }

    fn random_case(
        seq_len: usize,
        hidden_size: usize,
        vocab_size: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Vec<u32>, Vec<bool>)> {
        let total_h = seq_len * hidden_size;
        let hidden_vec: Vec<f32> = (0..total_h)
            .map(|i| (i as f32 * 0.013).sin() * 0.5)
            .collect();
        let hidden = Tensor::from_vec(hidden_vec, (1, seq_len, hidden_size), device)?;
        let total_head = hidden_size * vocab_size;
        let head_vec: Vec<f32> = (0..total_head)
            .map(|i| ((i as f32 + 7.0) * 0.007).cos() * 0.25)
            .collect();
        let head_t = Tensor::from_vec(head_vec, (hidden_size, vocab_size), device)?;
        let input_ids: Vec<u32> = (0..seq_len as u32)
            .map(|i| (i * 31 + 5) % vocab_size as u32)
            .collect();
        let label_mask: Vec<bool> = (0..seq_len).map(|i| i > 0 && i % 2 == 1).collect();
        Ok((hidden, head_t, input_ids, label_mask))
    }

    #[test]
    fn cpu_phase_a_vs_phase_b_forward() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_t, ids, mask) = random_case(16, 8, 64, &device)?;

        let phase_a =
            crate::fused_linear_cross_entropy(&hidden, &head_t, &ids, &mask, &device, 16)?;
        let phase_b =
            fused_linear_cross_entropy_phase_b(&hidden, &head_t, &ids, &mask, &device, 16)?;

        let a = phase_a.to_scalar::<f32>()?;
        let b = phase_b.to_scalar::<f32>()?;
        let abs = (a - b).abs();
        assert!(
            abs < 1e-5,
            "Phase A vs Phase B forward parity: a={a:.6} b={b:.6} abs={abs:.2e}"
        );
        Ok(())
    }

    #[test]
    fn cpu_phase_b_uneven_chunks() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_t, ids, mask) = random_case(12, 6, 73, &device)?;

        let phase_a =
            crate::fused_linear_cross_entropy(&hidden, &head_t, &ids, &mask, &device, 16)?;
        let phase_b =
            fused_linear_cross_entropy_phase_b(&hidden, &head_t, &ids, &mask, &device, 16)?;

        let a = phase_a.to_scalar::<f32>()?;
        let b = phase_b.to_scalar::<f32>()?;
        assert!((a - b).abs() < 1e-5, "uneven chunks: a={a:.6} b={b:.6}");
        Ok(())
    }

    #[test]
    fn cpu_phase_b_single_chunk() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_t, ids, mask) = random_case(10, 4, 32, &device)?;

        let phase_a =
            crate::fused_linear_cross_entropy(&hidden, &head_t, &ids, &mask, &device, 128)?;
        let phase_b =
            fused_linear_cross_entropy_phase_b(&hidden, &head_t, &ids, &mask, &device, 128)?;

        let a = phase_a.to_scalar::<f32>()?;
        let b = phase_b.to_scalar::<f32>()?;
        assert!((a - b).abs() < 1e-6, "single-chunk: a={a:.6} b={b:.6}");
        Ok(())
    }

    #[test]
    fn cpu_phase_b_bf16() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_t, ids, mask) = random_case(16, 8, 64, &device)?;
        let hidden_bf = hidden.to_dtype(DType::BF16)?;
        let head_bf = head_t.to_dtype(DType::BF16)?;

        let phase_a = crate::fused_linear_cross_entropy(
            &hidden_bf, &head_bf, &ids, &mask, &device, 16,
        )?;
        let phase_b = fused_linear_cross_entropy_phase_b(
            &hidden_bf, &head_bf, &ids, &mask, &device, 16,
        )?;

        let a = phase_a.to_scalar::<f32>()?;
        let b = phase_b.to_scalar::<f32>()?;
        assert!((a - b).abs() < 1e-2, "bf16: a={a:.6} b={b:.6}");
        Ok(())
    }

    #[test]
    fn cpu_phase_b_empty_mask_returns_zero() -> Result<()> {
        let device = Device::Cpu;
        let (hidden, head_t, ids, _) = random_case(8, 4, 16, &device)?;
        let all_false = vec![false; ids.len()];
        let loss = fused_linear_cross_entropy_phase_b(
            &hidden, &head_t, &ids, &all_false, &device, 4,
        )?;
        assert_eq!(loss.to_scalar::<f32>()?, 0.0);
        Ok(())
    }

    /// Backward parity: gradient at hidden from naive softmax CE backward
    /// vs Phase B's manual `bwd()`. This is the load-bearing correctness
    /// test — if this passes, training step gradients match Phase A's.
    #[test]
    fn cpu_phase_b_backward_parity_f32() -> Result<()> {
        let device = Device::Cpu;
        let (hidden_init, head_t, ids, mask) = random_case(16, 8, 64, &device)?;

        // Naive backward: full logits, candle autograd.
        let hidden_var_a = Var::from_tensor(&hidden_init)?;
        let loss_naive =
            naive_softmax_ce(hidden_var_a.as_tensor(), &head_t, &ids, &mask, &device)?;
        let grads_naive = loss_naive.backward()?;
        let grad_hidden_naive = grads_naive
            .get(hidden_var_a.as_tensor())
            .ok_or_else(|| anyhow!("no naive grad for hidden"))?
            .clone();

        // Phase B backward: CustomOp1.
        let hidden_var_b = Var::from_tensor(&hidden_init)?;
        let loss_phase_b = fused_linear_cross_entropy_phase_b(
            hidden_var_b.as_tensor(),
            &head_t,
            &ids,
            &mask,
            &device,
            16,
        )?;
        let grads_phase_b = loss_phase_b.backward()?;
        let grad_hidden_phase_b = grads_phase_b
            .get(hidden_var_b.as_tensor())
            .ok_or_else(|| anyhow!("no phase-b grad for hidden"))?
            .clone();

        // Compare loss values first as a sanity check.
        let loss_n = loss_naive.to_scalar::<f32>()?;
        let loss_b = loss_phase_b.to_scalar::<f32>()?;
        assert!(
            (loss_n - loss_b).abs() < 1e-5,
            "loss mismatch: naive={loss_n:.6} phase_b={loss_b:.6}"
        );

        // Compare gradients element-wise.
        let diff = (&grad_hidden_naive - &grad_hidden_phase_b)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let max_naive = grad_hidden_naive.abs()?.max_all()?.to_scalar::<f32>()?;
        let rel = if max_naive > 1e-6 {
            diff / max_naive
        } else {
            diff
        };
        assert!(
            diff < 1e-4 || rel < 1e-4,
            "grad parity failed: max_abs={diff:.2e} max_naive={max_naive:.6} rel={rel:.2e}",
        );
        Ok(())
    }

    /// Backward parity at bf16 — production training dtype. Tighter tolerance
    /// than the forward parity test because bf16 + f32 accumulation is what
    /// the trainer uses.
    #[test]
    fn cpu_phase_b_backward_parity_bf16() -> Result<()> {
        let device = Device::Cpu;
        let (hidden_init_f32, head_t, ids, mask) = random_case(16, 8, 64, &device)?;
        let hidden_init = hidden_init_f32.to_dtype(DType::BF16)?;
        let head_bf = head_t.to_dtype(DType::BF16)?;

        let hidden_var_a = Var::from_tensor(&hidden_init)?;
        let loss_naive =
            naive_softmax_ce(hidden_var_a.as_tensor(), &head_bf, &ids, &mask, &device)?;
        let grads_naive = loss_naive.backward()?;
        let grad_hidden_naive = grads_naive
            .get(hidden_var_a.as_tensor())
            .ok_or_else(|| anyhow!("no naive grad"))?
            .clone();

        let hidden_var_b = Var::from_tensor(&hidden_init)?;
        let loss_phase_b = fused_linear_cross_entropy_phase_b(
            hidden_var_b.as_tensor(),
            &head_bf,
            &ids,
            &mask,
            &device,
            16,
        )?;
        let grads_phase_b = loss_phase_b.backward()?;
        let grad_hidden_phase_b = grads_phase_b
            .get(hidden_var_b.as_tensor())
            .ok_or_else(|| anyhow!("no phase-b grad"))?
            .clone();

        let diff = (&grad_hidden_naive.to_dtype(DType::F32)?
            - &grad_hidden_phase_b.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let max_naive = grad_hidden_naive
            .to_dtype(DType::F32)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        let rel = if max_naive > 1e-6 {
            diff / max_naive
        } else {
            diff
        };
        assert!(
            diff < 1e-2 || rel < 1e-2,
            "bf16 grad parity failed: max_abs={diff:.2e} rel={rel:.2e}",
        );
        Ok(())
    }
}

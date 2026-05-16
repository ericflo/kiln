//! OPD top-K reverse-KL — **Phase B** custom op.
//!
//! Mirrors `kiln-flce-kernel`'s Phase B layout: a [`CustomOp1`] whose
//! forward runs the chunked gather + matmul + KL pass in a function-local
//! scope, returning only the scalar (or per-position) loss tensor. The
//! `bwd()` recomputes the gather + matmul to derive `dhidden` analytically,
//! so autograd does not retain the `[T_active, H, K]` head-gather
//! intermediate across iterations.
//!
//! # Why
//!
//! At T_active = 4096, K = 32, H = 2560 the head-gather alone is
//! ~335 M f32 elements = **~1.3 GB**. Phase A's autograd graph holds this
//! tensor (and its softmax-numerator broadcast) live for the entire
//! forward, which is wasteful at long-context OPD. Phase B drops it on
//! function return and recomputes it during backward — same correctness
//! tradeoff that FLCE Phase B makes (PR #646).
//!
//! # Analytic backward
//!
//! Let `s_logits[t, k] = (hidden_active[t] @ head_t[:, idx[t, k]])` be the
//! K student logits at position `t`. The renormalised softmax over the
//! K support is `p_hat[t, k] = softmax(s_logits[t])`. The renormalised
//! teacher log-probs over the same support are
//! `log_q_hat[t, k] = teacher_topk_logprobs[t, k] - lse(teacher_topk_logprobs[t])`
//! — these are **constants** with respect to `hidden`.
//!
//! ```text
//! KL_t = sum_k p_hat[t,k] * (log_p_hat[t,k] - log_q_hat[t,k])
//! ```
//!
//! Differentiating with respect to `s_logits[t, k]`:
//!
//! ```text
//! d KL_t / d s_logits[t, j]
//!   = p_hat[t, j] * (log_p_hat[t, j] - log_q_hat[t, j])
//!     + sum_k p_hat[t, k] * (d log_p_hat[t, k] / d s_logits[t, j])
//!   = p_hat[t, j] * (log_p_hat[t, j] - log_q_hat[t, j] - KL_t)
//! ```
//!
//! Using `d log_p_hat[t, k] / d s_logits[t, j] = delta_{kj} - p_hat[t, j]`
//! and `sum_k p_hat[t, k] * (delta_{kj} - p_hat[t, j]) = p_hat[t, j] -
//! p_hat[t, j] = 0` only when the entropy-of-softmax cancellation collapses
//! the residual, leaving `p_hat[t,j] * (log_p_hat[t,j] - log_q_hat[t,j] -
//! E_p[log p_hat - log q_hat])`.
//!
//! The mean-loss scaling adds a `1 / T_active * grad_loss` prefactor:
//!
//! ```text
//! d_s_logits[t, j] = (grad_loss / T_active)
//!                    * p_hat[t, j]
//!                    * (log_p_hat[t, j] - log_q_hat[t, j] - KL_t)
//! ```
//!
//! And `d_hidden_active[t, :] = sum_k d_s_logits[t, k] * head_t[:, idx[t, k]]`.
//!
//! For [`opd_top_k_reverse_kl_phase_b_per_position`] (no mean reduction),
//! `grad_loss` is a `[T_active]` upstream gradient and the `1 / T_active`
//! factor drops:
//!
//! ```text
//! d_s_logits[t, j] = grad_loss[t]
//!                    * p_hat[t, j]
//!                    * (log_p_hat[t, j] - log_q_hat[t, j] - KL_t)
//! ```

use anyhow::{Context, Result, anyhow};
use candle_core::op::BackpropOp;
use candle_core::{
    CpuStorage, CustomOp1, D, DType, Device, Layout, Shape, Storage, Tensor,
};

#[cfg(feature = "cuda")]
use candle_core::{
    CudaStorage,
    backend::{BackendDevice, BackendStorage},
};

use crate::{log_softmax_last, DEFAULT_CHUNK_SIZE};

/// Phase B entry point — scalar mean reverse-KL. Behaviourally identical
/// to [`crate::opd_top_k_reverse_kl_phase_a`] up to f32 associativity.
pub fn opd_top_k_reverse_kl_phase_b(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
    chunk_size: usize,
) -> Result<Tensor> {
    apply_op(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
        device,
        chunk_size,
        OpdLossOutput::ScalarMean,
    )
}

/// Phase B entry point — per-position reverse-KL. Returns a `[T_active]`
/// f32 tensor for the GRPO importance-sampling advantage construction
/// (§3.1 step 4 of the grand plan). Empty tensor when no positions are
/// active.
pub fn opd_top_k_reverse_kl_phase_b_per_position(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
    chunk_size: usize,
) -> Result<Tensor> {
    apply_op(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
        device,
        chunk_size,
        OpdLossOutput::PerPosition,
    )
}

fn apply_op(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
    chunk_size: usize,
    output_mode: OpdLossOutput,
) -> Result<Tensor> {
    if chunk_size == 0 {
        return Err(anyhow!("chunk_size must be > 0"));
    }
    let seq_len = hidden.dim(1)?;
    if label_mask.len() != seq_len {
        return Err(anyhow!(
            "label_mask length {} != T {}",
            label_mask.len(),
            seq_len
        ));
    }
    let active_count = label_mask.iter().filter(|&&m| m).count();
    if active_count == 0 {
        return match output_mode {
            OpdLossOutput::ScalarMean => Tensor::new(0.0f32, device)
                .context("zero scalar loss (no active rows)"),
            OpdLossOutput::PerPosition => Tensor::zeros((0,), DType::F32, device)
                .context("empty per-position KL"),
        };
    }
    let hidden_contig = hidden
        .contiguous()
        .context("force-contiguous hidden for OPD Phase B")?;
    let op = OpdLossCustomOp {
        head_t: head_t.clone(),
        teacher_topk_indices: teacher_topk_indices.to_vec(),
        teacher_topk_logprobs: teacher_topk_logprobs.to_vec(),
        label_mask: label_mask.to_vec(),
        top_k,
        chunk_size,
        output_mode,
    };
    hidden_contig.apply_op1(op).map_err(Into::into)
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum OpdLossOutput {
    /// Mean over active positions; result is `Shape::scalar()`.
    ScalarMean,
    /// Per-position vector of shape `[T_active]`.
    PerPosition,
}

/// `CustomOp1` for OPD top-K reverse KL. `apply_op1(hidden) -> loss`.
///
/// Captures `head_t`, the teacher's top-K indices + logprobs, the
/// `label_mask`, and `top_k` by value so they live with the op
/// instance — `hidden` is the only autograd input.
#[derive(Debug)]
pub struct OpdLossCustomOp {
    /// `[H, V]` transposed LM head — frozen during LoRA training.
    pub(crate) head_t: Tensor,
    /// Length `T_active * K`, row-major. Teacher top-K vocab indices.
    pub(crate) teacher_topk_indices: Vec<u32>,
    /// Length `T_active * K`, row-major. Teacher logprobs at those
    /// indices (log_softmax over the full teacher vocab).
    pub(crate) teacher_topk_logprobs: Vec<f32>,
    /// Length `T`, true at positions that contribute to the loss.
    pub(crate) label_mask: Vec<bool>,
    /// K — the teacher's support size.
    pub(crate) top_k: usize,
    /// Bound on the temporary `[chunk_T, K]` intermediate.
    pub(crate) chunk_size: usize,
    /// Whether the op outputs a scalar mean or per-position vector.
    pub(crate) output_mode: OpdLossOutput,
}

impl CustomOp1 for OpdLossCustomOp {
    fn name(&self) -> &'static str {
        "kiln-opd-loss-phase-b"
    }

    fn cpu_fwd(
        &self,
        s_hidden: &CpuStorage,
        l_hidden: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let storage = Storage::Cpu(s_hidden.clone());
        let hidden_shape = Shape::from(l_hidden.shape().dims());
        let hidden_leaf = Tensor::from_storage(storage, hidden_shape, BackpropOp::none(), false);
        let (loss_vec, out_shape) = self
            .forward_inner(&hidden_leaf)
            .map_err(|e| candle_core::Error::Msg(format!("opd-loss phase b cpu_fwd: {e:#}")))?;
        Ok((CpuStorage::F32(loss_vec), out_shape))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_hidden: &CudaStorage,
        l_hidden: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        let storage = Storage::Cuda(s_hidden.try_clone(l_hidden)?);
        let hidden_shape = Shape::from(l_hidden.shape().dims());
        let hidden_leaf =
            Tensor::from_storage(storage, hidden_shape, BackpropOp::none(), false);
        let (loss_vec, out_shape) = self
            .forward_inner(&hidden_leaf)
            .map_err(|e| candle_core::Error::Msg(format!("opd-loss phase b cuda_fwd: {e:#}")))?;
        let device = s_hidden.device();
        let out_slice = device.clone_htod(&loss_vec)?;
        Ok((
            CudaStorage::wrap_cuda_slice(out_slice, device.clone()),
            out_shape,
        ))
    }

    fn bwd(
        &self,
        hidden: &Tensor,
        _loss: &Tensor,
        grad_loss: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        self.backward_inner(hidden, grad_loss)
            .map(Some)
            .map_err(|e| candle_core::Error::Msg(format!("opd-loss phase b bwd: {e:#}")))
    }
}

impl OpdLossCustomOp {
    /// Forward implementation that runs over the leaf `hidden` tensor and
    /// returns the loss as a host f32 vector + output shape. The chunk
    /// intermediates (`head_gather`, `s_logits`, `log_p_hat`, etc.) are
    /// local to this function and dropped on return.
    ///
    /// Returns:
    /// - `ScalarMean`: `(vec![mean_kl], Shape::scalar())`.
    /// - `PerPosition`: `(per_position_kl_f32_vec, Shape::from(T_active))`.
    fn forward_inner(&self, hidden_leaf: &Tensor) -> Result<(Vec<f32>, Shape)> {
        let device = hidden_leaf.device();
        let seq_len = hidden_leaf.dim(1)?;
        debug_assert_eq!(self.label_mask.len(), seq_len);

        let active_positions: Vec<u32> = self
            .label_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
            .collect();
        let active_count = active_positions.len();
        debug_assert!(active_count > 0, "caller short-circuits empty");

        let active_indices = Tensor::new(active_positions.as_slice(), device)
            .context("build active position indices")?;
        let hidden_2d = hidden_leaf
            .squeeze(0)
            .context("squeeze hidden batch dim")?;
        let active_hidden = hidden_2d
            .index_select(&active_indices, 0)
            .context("gather active rows from hidden")?;
        let active_hidden_f32 = active_hidden
            .to_dtype(DType::F32)
            .context("cast hidden to f32")?;
        let head_t_f32 = self
            .head_t
            .to_dtype(DType::F32)
            .context("cast head_t to f32")?;
        let hidden_size = self.head_t.dim(0)?;

        let mut per_position: Vec<f32> = Vec::with_capacity(active_count);

        let mut start = 0usize;
        while start < active_count {
            let len = self.chunk_size.min(active_count - start);
            let end = start + len;
            let chunk_kl = self
                .compute_chunk_kl(
                    &active_hidden_f32,
                    &head_t_f32,
                    hidden_size,
                    start,
                    len,
                    device,
                )
                .with_context(|| format!("compute chunk KL for [{}, {})", start, end))?;
            per_position.extend_from_slice(&chunk_kl);
            start = end;
        }

        match self.output_mode {
            OpdLossOutput::ScalarMean => {
                let sum: f32 = per_position.iter().sum();
                let mean = sum / (active_count as f32);
                Ok((vec![mean], Shape::from(())))
            }
            OpdLossOutput::PerPosition => Ok((per_position, Shape::from(active_count))),
        }
    }

    /// Compute per-position KL for one chunk along the active-token axis.
    /// Returns a length-`len` f32 vector on the host.
    fn compute_chunk_kl(
        &self,
        active_hidden_f32: &Tensor,
        head_t_f32: &Tensor,
        hidden_size: usize,
        start: usize,
        len: usize,
        device: &Device,
    ) -> Result<Vec<f32>> {
        let k = self.top_k;
        // Pull this chunk's indices + logprobs.
        let chunk_indices = &self.teacher_topk_indices[start * k..(start + len) * k];
        let chunk_q_logprobs = &self.teacher_topk_logprobs[start * k..(start + len) * k];

        // [H, len * K]
        let flat_indices = Tensor::new(chunk_indices, device)?;
        let head_gather = head_t_f32.index_select(&flat_indices, 1)?;
        let head_3d = head_gather
            .reshape((hidden_size, len, k))?
            .permute((1, 0, 2))?
            .contiguous()?;

        // [len, 1, H] @ [len, H, K] -> [len, 1, K]
        let chunk_hidden = active_hidden_f32
            .narrow(0, start, len)?
            .unsqueeze(1)?;
        let s_logits = chunk_hidden.matmul(&head_3d)?.squeeze(1)?;

        let q_logprobs = Tensor::from_vec(chunk_q_logprobs.to_vec(), (len, k), device)?;
        let log_p_hat = log_softmax_last(&s_logits)?;
        let log_q_hat = log_softmax_last(&q_logprobs)?;
        let p_hat = log_p_hat.exp()?;
        let diff = (&log_p_hat - &log_q_hat)?;
        let per_pos = (p_hat * diff)?.sum(D::Minus1)?;
        Ok(per_pos.to_vec1::<f32>()?)
    }

    /// Backward: recompute the gather + matmul and emit
    /// `d_hidden` analytically. Returns a `[1, T, H]` tensor matching
    /// `hidden`'s dtype, zero outside active positions.
    fn backward_inner(&self, hidden: &Tensor, grad_loss: &Tensor) -> Result<Tensor> {
        let device = hidden.device();
        let dtype = hidden.dtype();
        let seq_len = hidden.dim(1)?;
        let hidden_size = hidden.dim(2)?;
        debug_assert_eq!(self.label_mask.len(), seq_len);

        let active_positions: Vec<u32> = self
            .label_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
            .collect();
        let active_count = active_positions.len();
        if active_count == 0 {
            return Ok(Tensor::zeros(hidden.shape(), dtype, device)?);
        }

        let active_indices = Tensor::new(active_positions.as_slice(), device)?;
        let hidden_2d = hidden.squeeze(0)?;
        let active_hidden = hidden_2d.index_select(&active_indices, 0)?;
        let active_hidden_f32 = active_hidden.to_dtype(DType::F32)?;
        let head_t_f32 = self.head_t.to_dtype(DType::F32)?;
        let grad_loss_f32 = grad_loss.to_dtype(DType::F32)?;
        let inv_n = match self.output_mode {
            OpdLossOutput::ScalarMean => 1.0_f64 / (active_count as f64),
            OpdLossOutput::PerPosition => 1.0_f64,
        };

        // Accumulate dhidden over chunks, since each chunk's intermediate
        // is the dominant memory cost.
        let mut d_hidden_active =
            Tensor::zeros((active_count, hidden_size), DType::F32, device)?;

        let mut start = 0usize;
        while start < active_count {
            let len = self.chunk_size.min(active_count - start);
            let chunk_dh = self.compute_chunk_dhidden(
                &active_hidden_f32,
                &head_t_f32,
                &grad_loss_f32,
                hidden_size,
                start,
                len,
                inv_n,
                device,
            )?;
            // Scatter chunk_dh rows back into the active-row buffer at
            // positions [start, start+len). index_add over a contiguous
            // range is just an addition into the right slice.
            let chunk_row_indices: Vec<u32> =
                ((start as u32)..((start + len) as u32)).collect();
            let row_idx = Tensor::new(chunk_row_indices.as_slice(), device)?;
            d_hidden_active = d_hidden_active.index_add(&row_idx, &chunk_dh, 0)?.detach();
            start += len;
        }

        // Scatter active rows back into a [T, H] zero buffer.
        let mut d_hidden_2d = Tensor::zeros((seq_len, hidden_size), DType::F32, device)?;
        d_hidden_2d = d_hidden_2d.index_add(&active_indices, &d_hidden_active, 0)?;
        let d_hidden_3d = d_hidden_2d.unsqueeze(0)?;
        Ok(d_hidden_3d.to_dtype(dtype)?)
    }

    fn compute_chunk_dhidden(
        &self,
        active_hidden_f32: &Tensor,
        head_t_f32: &Tensor,
        grad_loss_f32: &Tensor,
        hidden_size: usize,
        start: usize,
        len: usize,
        inv_n: f64,
        device: &Device,
    ) -> Result<Tensor> {
        let k = self.top_k;
        let chunk_indices = &self.teacher_topk_indices[start * k..(start + len) * k];
        let chunk_q_logprobs = &self.teacher_topk_logprobs[start * k..(start + len) * k];

        // Recompute the per-chunk head-gather (dropped after forward).
        let flat_indices = Tensor::new(chunk_indices, device)?;
        let head_gather = head_t_f32.index_select(&flat_indices, 1)?;
        let head_3d = head_gather
            .reshape((hidden_size, len, k))?
            .permute((1, 0, 2))?
            .contiguous()?;

        // Recompute s_logits + softmaxes.
        let chunk_hidden = active_hidden_f32
            .narrow(0, start, len)?
            .unsqueeze(1)?;
        let s_logits = chunk_hidden.matmul(&head_3d)?.squeeze(1)?;
        let q_logprobs = Tensor::from_vec(chunk_q_logprobs.to_vec(), (len, k), device)?;
        let log_p_hat = log_softmax_last(&s_logits)?;
        let log_q_hat = log_softmax_last(&q_logprobs)?;
        let p_hat = log_p_hat.exp()?;
        let diff = (&log_p_hat - &log_q_hat)?;
        // KL_t = sum_k p_hat * diff, broadcast back for the per-position subtraction.
        let kl_per_pos = (p_hat.clone() * diff.clone())?.sum_keepdim(D::Minus1)?;
        let inner = (diff - kl_per_pos.broadcast_as(p_hat.shape())?)?;
        let d_s_unscaled = (p_hat * inner)?; // [len, K]

        // Apply mean-loss factor and upstream gradient.
        // For ScalarMean: grad_loss is scalar; multiply by inv_n / T_active.
        // For PerPosition: grad_loss is [T_active]; multiply by per-position scalar.
        let d_s = match self.output_mode {
            OpdLossOutput::ScalarMean => {
                // (grad_loss * inv_n) is a scalar.
                let scale = grad_loss_f32.affine(inv_n, 0.0)?;
                d_s_unscaled.broadcast_mul(&scale)?
            }
            OpdLossOutput::PerPosition => {
                let chunk_grad = grad_loss_f32
                    .narrow(0, start, len)?
                    .unsqueeze(1)?; // [len, 1]
                d_s_unscaled.broadcast_mul(&chunk_grad)?
            }
        };

        // d_hidden_chunk = d_s @ head_3d.T   shape [len, K] @ [len, K, H] -> [len, 1, H]
        // head_3d is [len, H, K]; its inverse-permuted slice is [len, K, H].
        let head_3d_kh = head_3d.transpose(1, 2)?.contiguous()?; // [len, K, H]
        let d_s_3d = d_s.unsqueeze(1)?; // [len, 1, K]
        let d_h_chunk = d_s_3d.matmul(&head_3d_kh)?.squeeze(1)?; // [len, H]
        Ok(d_h_chunk)
    }
}

/// Public default chunk size, re-exported by `lib.rs`. Mirrors FLCE.
#[allow(dead_code)]
pub(crate) const _DEFAULT_CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE;

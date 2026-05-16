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
use candle_core::{CudaStorage, backend::BackendStorage};

use crate::{log_softmax_last, DEFAULT_CHUNK_SIZE};

// FFI declarations for the raw CUDA fused kernel (§9.2 of the grand
// plan). These are linked in only when the `cuda` feature is active —
// the `build.rs` compiles `csrc/opd_topk_kl.cu` and produces
// `libkiln_opd_loss_kernel.a` which Cargo links into the binary.
#[cfg(feature = "cuda")]
unsafe extern "C" {
    fn kiln_opd_topk_kl_fwd_bf16(
        hidden: *const core::ffi::c_void,
        head_t: *const core::ffi::c_void,
        topk_indices: *const core::ffi::c_void,
        topk_lp_q: *const core::ffi::c_void,
        kl_out: *mut core::ffi::c_void,
        t_active: i32,
        hidden_size: i32,
        vocab_size: i32,
        top_k: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_opd_topk_kl_fwd_f32(
        hidden: *const core::ffi::c_void,
        head_t: *const core::ffi::c_void,
        topk_indices: *const core::ffi::c_void,
        topk_lp_q: *const core::ffi::c_void,
        kl_out: *mut core::ffi::c_void,
        t_active: i32,
        hidden_size: i32,
        vocab_size: i32,
        top_k: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_opd_topk_kl_bwd_bf16(
        hidden: *const core::ffi::c_void,
        head_t: *const core::ffi::c_void,
        topk_indices: *const core::ffi::c_void,
        topk_lp_q: *const core::ffi::c_void,
        grad_loss: *const core::ffi::c_void,
        scale_factor: f32,
        d_hidden: *mut core::ffi::c_void,
        t_active: i32,
        hidden_size: i32,
        vocab_size: i32,
        top_k: i32,
        output_mode: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_opd_topk_kl_bwd_f32(
        hidden: *const core::ffi::c_void,
        head_t: *const core::ffi::c_void,
        topk_indices: *const core::ffi::c_void,
        topk_lp_q: *const core::ffi::c_void,
        grad_loss: *const core::ffi::c_void,
        scale_factor: f32,
        d_hidden: *mut core::ffi::c_void,
        t_active: i32,
        hidden_size: i32,
        vocab_size: i32,
        top_k: i32,
        output_mode: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_opd_topk_metrics_bf16(
        hidden: *const core::ffi::c_void,
        head_t: *const core::ffi::c_void,
        topk_indices: *const core::ffi::c_void,
        topk_lp_q: *const core::ffi::c_void,
        metrics_out: *mut core::ffi::c_void,
        t_active: i32,
        hidden_size: i32,
        vocab_size: i32,
        top_k: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_opd_topk_metrics_f32(
        hidden: *const core::ffi::c_void,
        head_t: *const core::ffi::c_void,
        topk_indices: *const core::ffi::c_void,
        topk_lp_q: *const core::ffi::c_void,
        metrics_out: *mut core::ffi::c_void,
        t_active: i32,
        hidden_size: i32,
        vocab_size: i32,
        top_k: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Returns `true` when the fused CUDA kernel supports the requested
/// `(top_k, dtype)` combination. K ∈ {16, 32} is the milestone-5 fast
/// path (§6 default is 32; the kernel hits that with 1024 threads per
/// block, the Ampere max). Other K values fall back to the candle
/// reference path on CUDA storage.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_kernel_supports(top_k: usize, dtype: DType) -> bool {
    let dtype_ok = matches!(dtype, DType::F32 | DType::BF16);
    dtype_ok && (top_k == 16 || top_k == 32)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub(crate) fn cuda_kernel_supports(_top_k: usize, _dtype: DType) -> bool {
    false
}

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

        // Route through the raw fused CUDA kernel when the dtype + K are
        // in the milestone-5 fast-path set AND the kill switch isn't on.
        // Otherwise fall through to the candle reference path (same as
        // CPU but on CUDA storage) — this preserves correctness for all
        // dtype/K combinations while we incrementally widen the kernel.
        let route_kernel = !crate::kernel_disabled()
            && cuda_kernel_supports(self.top_k, hidden_leaf.dtype());

        let (loss_vec, out_shape) = if route_kernel {
            match self.cuda_kernel_forward(&hidden_leaf) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        "opd-loss CUDA kernel failed, falling back to candle: {e:#}"
                    );
                    self.forward_inner(&hidden_leaf).map_err(|e2| {
                        candle_core::Error::Msg(format!(
                            "opd-loss phase b cuda_fwd fallback: {e2:#}"
                        ))
                    })?
                }
            }
        } else {
            self.forward_inner(&hidden_leaf).map_err(|e| {
                candle_core::Error::Msg(format!("opd-loss phase b cuda_fwd: {e:#}"))
            })?
        };
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
        // CUDA fast path mirrors cuda_fwd's gating: route through the
        // raw kernel when (a) we're on CUDA, (b) K and dtype are in the
        // supported envelope, and (c) the kill-switch isn't on. On any
        // decline / failure path we fall through to the analytic candle
        // backward (which is the parity oracle for the kernel).
        let on_cuda = matches!(hidden.device(), Device::Cuda(_));
        let route_kernel = on_cuda
            && !crate::kernel_disabled()
            && cuda_kernel_supports(self.top_k, hidden.dtype());

        #[cfg(feature = "cuda")]
        {
            if route_kernel {
                match self.cuda_kernel_backward(hidden, grad_loss) {
                    Ok(dh) => return Ok(Some(dh)),
                    Err(e) => {
                        tracing::warn!(
                            "opd-loss CUDA backward kernel failed, falling back to candle: {e:#}"
                        );
                    }
                }
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = route_kernel;
        }

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

    /// CUDA fast-path forward: gather active rows on device, upload
    /// teacher tensors, and call the fused `kiln_opd_topk_kl_fwd_*`
    /// kernel. Output is `[T_active]` f32 per-position KL; we then
    /// optionally mean-reduce on host for the `ScalarMean` mode.
    ///
    /// Falls within milestone 5's K ∈ {16, 32} dtype ∈ {f32, bf16}
    /// envelope. Out-of-envelope cases are caught by
    /// [`cuda_kernel_supports`] before this method is called.
    #[cfg(feature = "cuda")]
    fn cuda_kernel_forward(
        &self,
        hidden_leaf: &Tensor,
    ) -> Result<(Vec<f32>, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let device = match hidden_leaf.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Err(anyhow!("cuda_kernel_forward called with non-CUDA device")),
        };
        let dtype = hidden_leaf.dtype();
        if !cuda_kernel_supports(self.top_k, dtype) {
            return Err(anyhow!(
                "cuda_kernel_supports false for (top_k={}, dtype={:?})",
                self.top_k,
                dtype
            ));
        }

        let active_positions: Vec<u32> = self
            .label_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
            .collect();
        let active_count = active_positions.len();
        if active_count == 0 {
            return match self.output_mode {
                OpdLossOutput::ScalarMean => Ok((vec![0.0], Shape::from(()))),
                OpdLossOutput::PerPosition => Ok((Vec::new(), Shape::from(0_usize))),
            };
        }

        // Gather active rows on device. `index_select` over a contiguous
        // hidden tensor on CUDA storage executes via candle's CUDA
        // backend — no host round-trip.
        let active_indices = Tensor::new(active_positions.as_slice(), hidden_leaf.device())
            .context("upload active indices")?;
        let hidden_2d = hidden_leaf.squeeze(0).context("squeeze hidden")?;
        let active_hidden = hidden_2d
            .index_select(&active_indices, 0)
            .context("gather active rows")?
            .contiguous()
            .context("contiguous active hidden")?;
        let head_t_contig = self
            .head_t
            .contiguous()
            .context("contiguous head_t for CUDA kernel")?;

        let hidden_size = head_t_contig.dim(0)?;
        let vocab_size = head_t_contig.dim(1)?;
        if active_hidden.dim(1)? != hidden_size {
            return Err(anyhow!(
                "shape mismatch: active_hidden has H={} but head_t has H={}",
                active_hidden.dim(1)?,
                hidden_size
            ));
        }

        // Upload teacher tensors to device.
        let topk_idx_dev = Tensor::new(
            self.teacher_topk_indices.as_slice(),
            hidden_leaf.device(),
        )
        .context("upload topk indices")?
        .reshape((active_count, self.top_k))
        .context("reshape topk indices")?;
        let topk_lp_q_dev = Tensor::new(
            self.teacher_topk_logprobs.as_slice(),
            hidden_leaf.device(),
        )
        .context("upload topk logprobs")?
        .reshape((active_count, self.top_k))
        .context("reshape topk logprobs")?;

        // Allocate output `[active_count]` f32 buffer on device.
        let out_slice = device
            .alloc_zeros::<f32>(active_count)
            .map_err(|e| anyhow!("alloc opd kl output: {e}"))?;
        let stream = device.cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        // Extract device pointers. Each tensor's storage must be
        // contiguous (asserted upstream).
        let (hidden_storage, hidden_layout) = active_hidden.storage_and_layout();
        let (head_storage, head_layout) = head_t_contig.storage_and_layout();
        let (idx_storage, idx_layout) = topk_idx_dev.storage_and_layout();
        let (lpq_storage, lpq_layout) = topk_lp_q_dev.storage_and_layout();
        let hidden_cuda = match &*hidden_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("active_hidden not on CUDA")),
        };
        let head_cuda = match &*head_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("head_t not on CUDA")),
        };
        let idx_cuda = match &*idx_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("topk_idx not on CUDA")),
        };
        let lpq_cuda = match &*lpq_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("topk_lp_q not on CUDA")),
        };

        let status = unsafe {
            match dtype {
                DType::F32 => {
                    let hidden_slice = hidden_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("hidden as f32 slice: {e}"))?
                        .slice(hidden_layout.start_offset()..);
                    let head_slice = head_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("head_t as f32 slice: {e}"))?
                        .slice(head_layout.start_offset()..);
                    let idx_slice = idx_cuda
                        .as_cuda_slice::<u32>()
                        .map_err(|e| anyhow!("topk_idx as u32 slice: {e}"))?
                        .slice(idx_layout.start_offset()..);
                    let lpq_slice = lpq_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("topk_lp_q as f32 slice: {e}"))?
                        .slice(lpq_layout.start_offset()..);

                    let (h_ptr, _g1) = hidden_slice.device_ptr(&stream);
                    let (head_ptr, _g2) = head_slice.device_ptr(&stream);
                    let (idx_ptr, _g3) = idx_slice.device_ptr(&stream);
                    let (lpq_ptr, _g4) = lpq_slice.device_ptr(&stream);
                    let (out_ptr, _g5) = out_slice.device_ptr(&stream);

                    kiln_opd_topk_kl_fwd_f32(
                        h_ptr as *const _,
                        head_ptr as *const _,
                        idx_ptr as *const _,
                        lpq_ptr as *const _,
                        out_ptr as *mut _,
                        active_count as i32,
                        hidden_size as i32,
                        vocab_size as i32,
                        self.top_k as i32,
                        raw_stream,
                    )
                }
                DType::BF16 => {
                    use half::bf16;
                    let hidden_slice = hidden_cuda
                        .as_cuda_slice::<bf16>()
                        .map_err(|e| anyhow!("hidden as bf16 slice: {e}"))?
                        .slice(hidden_layout.start_offset()..);
                    let head_slice = head_cuda
                        .as_cuda_slice::<bf16>()
                        .map_err(|e| anyhow!("head_t as bf16 slice: {e}"))?
                        .slice(head_layout.start_offset()..);
                    let idx_slice = idx_cuda
                        .as_cuda_slice::<u32>()
                        .map_err(|e| anyhow!("topk_idx as u32 slice: {e}"))?
                        .slice(idx_layout.start_offset()..);
                    let lpq_slice = lpq_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("topk_lp_q as f32 slice: {e}"))?
                        .slice(lpq_layout.start_offset()..);

                    let (h_ptr, _g1) = hidden_slice.device_ptr(&stream);
                    let (head_ptr, _g2) = head_slice.device_ptr(&stream);
                    let (idx_ptr, _g3) = idx_slice.device_ptr(&stream);
                    let (lpq_ptr, _g4) = lpq_slice.device_ptr(&stream);
                    let (out_ptr, _g5) = out_slice.device_ptr(&stream);

                    kiln_opd_topk_kl_fwd_bf16(
                        h_ptr as *const _,
                        head_ptr as *const _,
                        idx_ptr as *const _,
                        lpq_ptr as *const _,
                        out_ptr as *mut _,
                        active_count as i32,
                        hidden_size as i32,
                        vocab_size as i32,
                        self.top_k as i32,
                        raw_stream,
                    )
                }
                other => {
                    return Err(anyhow!(
                        "cuda_kernel_forward: unsupported dtype {other:?}"
                    ));
                }
            }
        };
        if status != 0 {
            return Err(anyhow!(
                "kiln_opd_topk_kl_fwd_* returned status {status}"
            ));
        }

        // Download per-position KL to host via a candle Tensor wrapper.
        // `wrap_cuda_slice` doesn't move data; the subsequent
        // `to_vec1::<f32>()` performs the D2H copy through candle's
        // own backend path.
        let out_storage =
            CudaStorage::wrap_cuda_slice(out_slice, device.clone());
        let out_tensor = Tensor::from_storage(
            Storage::Cuda(out_storage),
            Shape::from(active_count),
            BackpropOp::none(),
            false,
        );
        let host_kl: Vec<f32> = out_tensor
            .to_vec1::<f32>()
            .context("D2H copy of kl output")?;

        match self.output_mode {
            OpdLossOutput::ScalarMean => {
                let sum: f32 = host_kl.iter().sum();
                let mean = sum / (active_count as f32);
                Ok((vec![mean], Shape::from(())))
            }
            OpdLossOutput::PerPosition => Ok((host_kl, Shape::from(active_count))),
        }
    }

    /// CUDA fast-path backward — computes `d_hidden` analytically via
    /// the fused `kiln_opd_topk_kl_bwd_*` kernel. Mirrors the active-row
    /// gather + teacher tensor upload pattern from `cuda_kernel_forward`,
    /// then scatters the K-by-token gradient back into the full
    /// `[1, T, H]` shape that the candle autograd contract requires.
    ///
    /// `grad_loss` shape:
    /// - `ScalarMean` output mode: a 0-dim tensor (or 1-element 1-D).
    ///   The kernel multiplies by `1/T_active * grad_loss[0]`.
    /// - `PerPosition` output mode: a 1-D `[T_active]` tensor. The
    ///   kernel multiplies position-wise.
    #[cfg(feature = "cuda")]
    fn cuda_kernel_backward(
        &self,
        hidden: &Tensor,
        grad_loss: &Tensor,
    ) -> Result<Tensor> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let device = match hidden.device() {
            Device::Cuda(d) => d.clone(),
            _ => return Err(anyhow!("cuda_kernel_backward called with non-CUDA device")),
        };
        let dtype = hidden.dtype();
        if !cuda_kernel_supports(self.top_k, dtype) {
            return Err(anyhow!(
                "cuda_kernel_supports false for (top_k={}, dtype={:?})",
                self.top_k,
                dtype
            ));
        }

        let seq_len = hidden.dim(1)?;
        let hidden_size = hidden.dim(2)?;
        if seq_len != self.label_mask.len() {
            return Err(anyhow!(
                "seq_len {} != label_mask len {}",
                seq_len,
                self.label_mask.len()
            ));
        }

        let active_positions: Vec<u32> = self
            .label_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
            .collect();
        let active_count = active_positions.len();
        if active_count == 0 {
            return Ok(Tensor::zeros(hidden.shape(), dtype, hidden.device())?);
        }

        // Gather active rows on device.
        let active_indices = Tensor::new(active_positions.as_slice(), hidden.device())
            .context("upload active indices")?;
        let hidden_2d = hidden.squeeze(0).context("squeeze hidden")?;
        let active_hidden = hidden_2d
            .index_select(&active_indices, 0)
            .context("gather active rows")?
            .contiguous()
            .context("contiguous active hidden")?;
        let head_t_contig = self
            .head_t
            .contiguous()
            .context("contiguous head_t for CUDA backward")?;
        if active_hidden.dim(1)? != hidden_size {
            return Err(anyhow!(
                "shape mismatch: active_hidden H={} but hidden H={}",
                active_hidden.dim(1)?,
                hidden_size
            ));
        }
        let vocab_size = head_t_contig.dim(1)?;

        // Upload teacher tensors.
        let topk_idx_dev = Tensor::new(
            self.teacher_topk_indices.as_slice(),
            hidden.device(),
        )
        .context("upload topk indices")?
        .reshape((active_count, self.top_k))
        .context("reshape topk indices")?;
        let topk_lp_q_dev = Tensor::new(
            self.teacher_topk_logprobs.as_slice(),
            hidden.device(),
        )
        .context("upload topk logprobs")?
        .reshape((active_count, self.top_k))
        .context("reshape topk logprobs")?;

        // Normalise grad_loss to a 1-D f32 tensor on device, shape
        // {ScalarMean: [1], PerPosition: [active_count]}. The kernel
        // reads grad_loss[0] in ScalarMean and grad_loss[t] in
        // PerPosition.
        let grad_loss_f32 = grad_loss
            .to_dtype(DType::F32)
            .context("cast grad_loss to f32")?;
        let (grad_loss_dev, output_mode_i32, scale_factor) = match self.output_mode {
            OpdLossOutput::ScalarMean => {
                // grad_loss is a scalar (Shape::scalar()). Reshape to [1].
                let g = grad_loss_f32
                    .reshape(1)
                    .context("reshape ScalarMean grad_loss to [1]")?
                    .contiguous()?;
                let scale = 1.0_f32 / (active_count as f32);
                (g, 0_i32, scale)
            }
            OpdLossOutput::PerPosition => {
                let dims = grad_loss_f32.dims().to_vec();
                if dims.len() != 1 || dims[0] != active_count {
                    return Err(anyhow!(
                        "PerPosition grad_loss must have shape [{active_count}], got {dims:?}"
                    ));
                }
                (grad_loss_f32.contiguous()?, 1_i32, 1.0_f32)
            }
        };

        // Allocate the `[active_count, hidden_size]` output buffer on
        // device in the same dtype as hidden.
        let d_hidden_active_storage: Storage = match dtype {
            DType::F32 => {
                let slice = device
                    .alloc_zeros::<f32>(active_count * hidden_size)
                    .map_err(|e| anyhow!("alloc d_hidden_active f32: {e}"))?;
                Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, device.clone()))
            }
            DType::BF16 => {
                use half::bf16;
                let slice = device
                    .alloc_zeros::<bf16>(active_count * hidden_size)
                    .map_err(|e| anyhow!("alloc d_hidden_active bf16: {e}"))?;
                Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, device.clone()))
            }
            other => return Err(anyhow!("unsupported dtype {other:?}")),
        };
        let d_hidden_active = Tensor::from_storage(
            d_hidden_active_storage,
            Shape::from((active_count, hidden_size)),
            BackpropOp::none(),
            false,
        );

        let stream = device.cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        // Pull device pointers out of every input/output tensor.
        let (hidden_storage, hidden_layout) = active_hidden.storage_and_layout();
        let (head_storage, head_layout) = head_t_contig.storage_and_layout();
        let (idx_storage, idx_layout) = topk_idx_dev.storage_and_layout();
        let (lpq_storage, lpq_layout) = topk_lp_q_dev.storage_and_layout();
        let (gl_storage, gl_layout) = grad_loss_dev.storage_and_layout();
        let (dh_storage, dh_layout) = d_hidden_active.storage_and_layout();

        let hidden_cuda = match &*hidden_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("active_hidden not on CUDA")),
        };
        let head_cuda = match &*head_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("head_t not on CUDA")),
        };
        let idx_cuda = match &*idx_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("topk_idx not on CUDA")),
        };
        let lpq_cuda = match &*lpq_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("topk_lp_q not on CUDA")),
        };
        let gl_cuda = match &*gl_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("grad_loss not on CUDA")),
        };
        let dh_cuda = match &*dh_storage {
            Storage::Cuda(c) => c,
            _ => return Err(anyhow!("d_hidden not on CUDA")),
        };

        let status = unsafe {
            match dtype {
                DType::F32 => {
                    let h_s = hidden_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("hidden f32: {e}"))?
                        .slice(hidden_layout.start_offset()..);
                    let head_s = head_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("head_t f32: {e}"))?
                        .slice(head_layout.start_offset()..);
                    let i_s = idx_cuda
                        .as_cuda_slice::<u32>()
                        .map_err(|e| anyhow!("idx u32: {e}"))?
                        .slice(idx_layout.start_offset()..);
                    let l_s = lpq_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("lpq f32: {e}"))?
                        .slice(lpq_layout.start_offset()..);
                    let g_s = gl_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("grad_loss f32: {e}"))?
                        .slice(gl_layout.start_offset()..);
                    let d_s = dh_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("d_hidden f32: {e}"))?
                        .slice(dh_layout.start_offset()..);

                    let (h_ptr, _g1) = h_s.device_ptr(&stream);
                    let (head_ptr, _g2) = head_s.device_ptr(&stream);
                    let (i_ptr, _g3) = i_s.device_ptr(&stream);
                    let (l_ptr, _g4) = l_s.device_ptr(&stream);
                    let (g_ptr, _g5) = g_s.device_ptr(&stream);
                    let (d_ptr, _g6) = d_s.device_ptr(&stream);

                    kiln_opd_topk_kl_bwd_f32(
                        h_ptr as *const _,
                        head_ptr as *const _,
                        i_ptr as *const _,
                        l_ptr as *const _,
                        g_ptr as *const _,
                        scale_factor,
                        d_ptr as *mut _,
                        active_count as i32,
                        hidden_size as i32,
                        vocab_size as i32,
                        self.top_k as i32,
                        output_mode_i32,
                        raw_stream,
                    )
                }
                DType::BF16 => {
                    use half::bf16;
                    let h_s = hidden_cuda
                        .as_cuda_slice::<bf16>()
                        .map_err(|e| anyhow!("hidden bf16: {e}"))?
                        .slice(hidden_layout.start_offset()..);
                    let head_s = head_cuda
                        .as_cuda_slice::<bf16>()
                        .map_err(|e| anyhow!("head_t bf16: {e}"))?
                        .slice(head_layout.start_offset()..);
                    let i_s = idx_cuda
                        .as_cuda_slice::<u32>()
                        .map_err(|e| anyhow!("idx u32: {e}"))?
                        .slice(idx_layout.start_offset()..);
                    let l_s = lpq_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("lpq f32: {e}"))?
                        .slice(lpq_layout.start_offset()..);
                    let g_s = gl_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|e| anyhow!("grad_loss f32: {e}"))?
                        .slice(gl_layout.start_offset()..);
                    let d_s = dh_cuda
                        .as_cuda_slice::<bf16>()
                        .map_err(|e| anyhow!("d_hidden bf16: {e}"))?
                        .slice(dh_layout.start_offset()..);

                    let (h_ptr, _g1) = h_s.device_ptr(&stream);
                    let (head_ptr, _g2) = head_s.device_ptr(&stream);
                    let (i_ptr, _g3) = i_s.device_ptr(&stream);
                    let (l_ptr, _g4) = l_s.device_ptr(&stream);
                    let (g_ptr, _g5) = g_s.device_ptr(&stream);
                    let (d_ptr, _g6) = d_s.device_ptr(&stream);

                    kiln_opd_topk_kl_bwd_bf16(
                        h_ptr as *const _,
                        head_ptr as *const _,
                        i_ptr as *const _,
                        l_ptr as *const _,
                        g_ptr as *const _,
                        scale_factor,
                        d_ptr as *mut _,
                        active_count as i32,
                        hidden_size as i32,
                        vocab_size as i32,
                        self.top_k as i32,
                        output_mode_i32,
                        raw_stream,
                    )
                }
                other => return Err(anyhow!("unsupported dtype {other:?}")),
            }
        };
        if status != 0 {
            return Err(anyhow!("kiln_opd_topk_kl_bwd_* returned status {status}"));
        }

        // Scatter the `[active_count, hidden_size]` gradient back into
        // a `[seq_len, hidden_size]` zero buffer, then unsqueeze to
        // `[1, seq_len, hidden_size]`. The active_indices select
        // exactly the rows we just wrote.
        let zeros_2d = Tensor::zeros((seq_len, hidden_size), dtype, hidden.device())
            .context("zeros [seq_len, hidden_size]")?;
        let d_hidden_2d = zeros_2d
            .index_add(&active_indices, &d_hidden_active, 0)
            .context("scatter d_hidden_active back into [seq_len, hidden_size]")?;
        let d_hidden_3d = d_hidden_2d
            .unsqueeze(0)
            .context("unsqueeze to [1, seq_len, hidden_size]")?;
        Ok(d_hidden_3d)
    }
}

/// Output of [`compute_per_position_metrics`]: three parallel
/// `[T_active]` arrays carrying the per-position distribution-alignment
/// diagnostics. Lengths are all equal to the number of active positions.
#[derive(Debug, Clone)]
pub struct PerPositionMetrics {
    /// Per-position student entropy over the teacher's K support.
    pub student_entropy: Vec<f32>,
    /// Per-position teacher entropy over the same K support.
    pub teacher_entropy: Vec<f32>,
    /// Per-position reverse KL (same value the loss kernel emits).
    pub reverse_kl: Vec<f32>,
}

impl PerPositionMetrics {
    /// `[T_active]` of `|H(q) - H(p)|` per position.
    pub fn entropy_gap_vec(&self) -> Vec<f32> {
        self.student_entropy
            .iter()
            .zip(self.teacher_entropy.iter())
            .map(|(p, q)| (q - p).abs())
            .collect()
    }

    /// Mean over active positions of `|H(q) - H(p)|`. Used as the
    /// scalar §3.8 diagnostic.
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

/// Compute distribution-alignment metrics per-position over the
/// teacher's K support. CUDA-only fast path (`top_k ∈ {16, 32}`,
/// dtype ∈ {f32, bf16}); other configurations fall back to a candle
/// reference computed in this function.
///
/// Caller convention matches the loss kernel: `hidden` is `[1, T, H]`,
/// `head_t` is `[H, V]`, `teacher_topk_*` are flattened
/// `[T_active * K]`, and `label_mask[t] == true` for positions that
/// contribute. Output is in `T_active` order matching the mask
/// left-to-right.
pub fn compute_per_position_metrics(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<PerPositionMetrics> {
    let active_positions: Vec<u32> = label_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    let active_count = active_positions.len();
    if active_count == 0 {
        return Ok(PerPositionMetrics {
            student_entropy: Vec::new(),
            teacher_entropy: Vec::new(),
            reverse_kl: Vec::new(),
        });
    }
    let on_cuda = matches!(hidden.device(), Device::Cuda(_));
    let dtype = hidden.dtype();
    let route_kernel = on_cuda
        && !crate::kernel_disabled()
        && cuda_kernel_supports(top_k, dtype);

    #[cfg(feature = "cuda")]
    {
        if route_kernel {
            match cuda_compute_per_position_metrics(
                hidden,
                head_t,
                teacher_topk_indices,
                teacher_topk_logprobs,
                &active_positions,
                top_k,
            ) {
                Ok(m) => return Ok(m),
                Err(e) => {
                    tracing::warn!(
                        "opd metrics CUDA kernel failed, falling back to candle: {e:#}"
                    );
                }
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = route_kernel;
    }

    // Candle reference path. Reuses the same active-row gather and
    // per-token logit computation as the Phase A loss kernel; reads
    // intermediates and produces the three metrics on host.
    candle_reference_per_position_metrics(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        &active_positions,
        top_k,
    )
}

fn candle_reference_per_position_metrics(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    active_positions: &[u32],
    top_k: usize,
) -> Result<PerPositionMetrics> {
    let device = hidden.device();
    let active_count = active_positions.len();
    let active_indices = Tensor::new(active_positions, device)?;
    let hidden_2d = hidden.squeeze(0)?;
    let active_hidden = hidden_2d
        .index_select(&active_indices, 0)?
        .to_dtype(DType::F32)?;
    let head_t_f32 = head_t.to_dtype(DType::F32)?;
    let hidden_size = head_t.dim(0)?;

    // Per-position gather of K head columns, then matmul.
    let flat_idx = Tensor::new(teacher_topk_indices, device)?;
    let head_gather = head_t_f32.index_select(&flat_idx, 1)?;
    let head_3d = head_gather
        .reshape((hidden_size, active_count, top_k))?
        .permute((1, 0, 2))?
        .contiguous()?;
    let lhs = active_hidden.unsqueeze(1)?;
    let s_logits = lhs.matmul(&head_3d)?.squeeze(1)?;
    let q_lp = Tensor::from_vec(teacher_topk_logprobs.to_vec(), (active_count, top_k), device)?;
    let log_p_hat = crate::log_softmax_last(&s_logits)?;
    let log_q_hat = crate::log_softmax_last(&q_lp)?;
    let p_hat = log_p_hat.exp()?;
    let q_hat = log_q_hat.exp()?;

    // H(p) = -sum_k p_hat * log_p_hat, H(q) = -sum_k q_hat * log_q_hat, KL = sum_k p_hat * (log_p_hat - log_q_hat)
    let h_p = (&p_hat * &log_p_hat)?
        .sum(D::Minus1)?
        .affine(-1.0, 0.0)?;
    let h_q = (&q_hat * &log_q_hat)?
        .sum(D::Minus1)?
        .affine(-1.0, 0.0)?;
    let diff = (&log_p_hat - &log_q_hat)?;
    let kl = (p_hat * diff)?.sum(D::Minus1)?;
    Ok(PerPositionMetrics {
        student_entropy: h_p.to_vec1()?,
        teacher_entropy: h_q.to_vec1()?,
        reverse_kl: kl.to_vec1()?,
    })
}

#[cfg(feature = "cuda")]
fn cuda_compute_per_position_metrics(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    active_positions: &[u32],
    top_k: usize,
) -> Result<PerPositionMetrics> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let device = match hidden.device() {
        Device::Cuda(d) => d.clone(),
        _ => return Err(anyhow!("non-CUDA device")),
    };
    let dtype = hidden.dtype();
    let active_count = active_positions.len();
    debug_assert!(active_count > 0);

    let active_indices = Tensor::new(active_positions, hidden.device())?;
    let hidden_2d = hidden.squeeze(0)?;
    let active_hidden = hidden_2d
        .index_select(&active_indices, 0)?
        .contiguous()?;
    let head_t_contig = head_t.contiguous()?;
    let hidden_size = head_t_contig.dim(0)?;
    let vocab_size = head_t_contig.dim(1)?;

    let topk_idx_dev = Tensor::new(teacher_topk_indices, hidden.device())?
        .reshape((active_count, top_k))?;
    let topk_lp_q_dev = Tensor::new(teacher_topk_logprobs, hidden.device())?
        .reshape((active_count, top_k))?;

    // Output buffer: [T_active, 3] f32 — Hp, Hq, KL per row.
    let out_slice = device
        .alloc_zeros::<f32>(active_count * 3)
        .map_err(|e| anyhow!("alloc metrics output: {e}"))?;
    let stream = device.cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let (h_storage, h_layout) = active_hidden.storage_and_layout();
    let (head_storage, head_layout) = head_t_contig.storage_and_layout();
    let (i_storage, i_layout) = topk_idx_dev.storage_and_layout();
    let (l_storage, l_layout) = topk_lp_q_dev.storage_and_layout();
    let h_c = match &*h_storage {
        Storage::Cuda(c) => c,
        _ => return Err(anyhow!("active_hidden not on CUDA")),
    };
    let head_c = match &*head_storage {
        Storage::Cuda(c) => c,
        _ => return Err(anyhow!("head_t not on CUDA")),
    };
    let i_c = match &*i_storage {
        Storage::Cuda(c) => c,
        _ => return Err(anyhow!("idx not on CUDA")),
    };
    let l_c = match &*l_storage {
        Storage::Cuda(c) => c,
        _ => return Err(anyhow!("lpq not on CUDA")),
    };

    let status = unsafe {
        match dtype {
            DType::F32 => {
                let h_s = h_c
                    .as_cuda_slice::<f32>()
                    .map_err(|e| anyhow!("hidden f32: {e}"))?
                    .slice(h_layout.start_offset()..);
                let head_s = head_c
                    .as_cuda_slice::<f32>()
                    .map_err(|e| anyhow!("head_t f32: {e}"))?
                    .slice(head_layout.start_offset()..);
                let i_s = i_c
                    .as_cuda_slice::<u32>()
                    .map_err(|e| anyhow!("idx: {e}"))?
                    .slice(i_layout.start_offset()..);
                let l_s = l_c
                    .as_cuda_slice::<f32>()
                    .map_err(|e| anyhow!("lpq: {e}"))?
                    .slice(l_layout.start_offset()..);

                let (h_ptr, _g1) = h_s.device_ptr(&stream);
                let (head_ptr, _g2) = head_s.device_ptr(&stream);
                let (i_ptr, _g3) = i_s.device_ptr(&stream);
                let (l_ptr, _g4) = l_s.device_ptr(&stream);
                let (out_ptr, _g5) = out_slice.device_ptr(&stream);

                kiln_opd_topk_metrics_f32(
                    h_ptr as *const _,
                    head_ptr as *const _,
                    i_ptr as *const _,
                    l_ptr as *const _,
                    out_ptr as *mut _,
                    active_count as i32,
                    hidden_size as i32,
                    vocab_size as i32,
                    top_k as i32,
                    raw_stream,
                )
            }
            DType::BF16 => {
                use half::bf16;
                let h_s = h_c
                    .as_cuda_slice::<bf16>()
                    .map_err(|e| anyhow!("hidden bf16: {e}"))?
                    .slice(h_layout.start_offset()..);
                let head_s = head_c
                    .as_cuda_slice::<bf16>()
                    .map_err(|e| anyhow!("head_t bf16: {e}"))?
                    .slice(head_layout.start_offset()..);
                let i_s = i_c
                    .as_cuda_slice::<u32>()
                    .map_err(|e| anyhow!("idx: {e}"))?
                    .slice(i_layout.start_offset()..);
                let l_s = l_c
                    .as_cuda_slice::<f32>()
                    .map_err(|e| anyhow!("lpq: {e}"))?
                    .slice(l_layout.start_offset()..);

                let (h_ptr, _g1) = h_s.device_ptr(&stream);
                let (head_ptr, _g2) = head_s.device_ptr(&stream);
                let (i_ptr, _g3) = i_s.device_ptr(&stream);
                let (l_ptr, _g4) = l_s.device_ptr(&stream);
                let (out_ptr, _g5) = out_slice.device_ptr(&stream);

                kiln_opd_topk_metrics_bf16(
                    h_ptr as *const _,
                    head_ptr as *const _,
                    i_ptr as *const _,
                    l_ptr as *const _,
                    out_ptr as *mut _,
                    active_count as i32,
                    hidden_size as i32,
                    vocab_size as i32,
                    top_k as i32,
                    raw_stream,
                )
            }
            other => return Err(anyhow!("unsupported dtype {other:?}")),
        }
    };
    if status != 0 {
        return Err(anyhow!("opd metrics kernel status {status}"));
    }
    // Download flattened result.
    let out_storage = CudaStorage::wrap_cuda_slice(out_slice, device.clone());
    let out_tensor = Tensor::from_storage(
        Storage::Cuda(out_storage),
        Shape::from((active_count, 3)),
        BackpropOp::none(),
        false,
    );
    let flat: Vec<Vec<f32>> = out_tensor.to_vec2()?;
    let mut h_p = Vec::with_capacity(active_count);
    let mut h_q = Vec::with_capacity(active_count);
    let mut kl = Vec::with_capacity(active_count);
    for row in flat {
        h_p.push(row[0]);
        h_q.push(row[1]);
        kl.push(row[2]);
    }
    Ok(PerPositionMetrics {
        student_entropy: h_p,
        teacher_entropy: h_q,
        reverse_kl: kl,
    })
}

/// Public default chunk size, re-exported by `lib.rs`. Mirrors FLCE.
#[allow(dead_code)]
pub(crate) const _DEFAULT_CHUNK_SIZE: usize = DEFAULT_CHUNK_SIZE;

//! Fused OPD top-K reverse-KL loss on `VkTensor`.
//!
//! Vulkan port of the CUDA `kiln_opd_topk_kl_fwd/bwd_{bf16,f32}` kernels
//! (see `crates/kiln-opd-loss-kernel/csrc/opd_topk_kl.cu`).
//!
//! Computes the per-token reverse KL between the student's distribution
//! (gathered from `hidden @ weight^T` at the teacher's top-K vocab support)
//! and the teacher's distribution (renormalised over the same K support).
//!
//! Used by the Vulkan-native OPD trainer per §9.2 + §9.10 of
//! `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`.
//! The low-level `dispatch_*_resident` entry points take raw `&VulkanBuffer`
//! arguments so the OPD loss can be folded into a single training-step
//! command submission alongside the resident-decode pattern that PR #1030
//! established for inference.
//!
//! # Tensor shape + dtype contract (matches `flce.rs`)
//!
//! - `hidden`: `[T_active, H]` F32. Already gathered to active positions.
//! - `weight`: `[V, H]` F32 or BF16-packed (vocab-row-major; this is the
//!   canonical kiln-train LM-head layout). Column `c = weight[c * H + h]`.
//! - `teacher_topk_indices`: host `[T_active * K]` u32. Top-K vocab indices
//!   from the teacher at each active position.
//! - `teacher_topk_logprobs`: host `[T_active * K]` f32. Teacher logprobs at
//!   the same indices (full-vocab `log_softmax`; the kernel renormalises
//!   over the K support internally).
//! - `top_k`: 16 or 32 (matches CUDA kernel's supported set).
//!
//! # Output
//!
//! - `vk_opd_top_k_reverse_kl_loss` → scalar `[1]` F32 mean reverse-KL,
//!   with a `grad_fn` that emits `d_hidden` via the fused backward kernel
//!   when `vk_backward()` walks the tape.
//! - `vk_opd_top_k_reverse_kl_per_position` → `[T_active]` F32 per-token
//!   reverse-KL (no mean reduction; used by the GRPO importance-sampling
//!   advantage path).
//! - `vk_opd_top_k_metrics` → `[T_active, 3]` F32: columns `(H(p_hat),
//!   H(q_hat), KL_t)` for the §3.8 distribution-alignment diagnostics.

use crate::vk_ops::dispatch_simple;
use crate::vk_ops::reduce::vk_mean_all;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::Result;
use ash::vk;
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn upload_u32(device: &Arc<VulkanDevice>, data: &[u32]) -> Result<Arc<VulkanBuffer>> {
    let bytes: Vec<u8> = data.iter().flat_map(|i| i.to_le_bytes()).collect();
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes.len().max(4) as u64,
    )?;
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &buf,
        &bytes,
    )?;
    Ok(Arc::new(buf))
}

fn upload_f32(device: &Arc<VulkanDevice>, data: &[f32]) -> Result<Arc<VulkanBuffer>> {
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes.len().max(4) as u64,
    )?;
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &buf,
        &bytes,
    )?;
    Ok(Arc::new(buf))
}

/// Validate the (hidden, weight) shapes + dtypes accepted by the kernel.
/// Mirrors `flce::validate_lm_head_inputs` plus the K ∈ {16, 32} envelope.
fn validate_opd_inputs(
    hidden: &VkTensor,
    weight: &VkTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    top_k: usize,
) -> Result<(usize, usize, usize)> {
    anyhow::ensure!(
        hidden.shape().len() == 2 && weight.shape().len() == 2,
        "vk_opd: hidden and weight must be rank-2"
    );
    anyhow::ensure!(
        hidden.dtype() == VkDType::F32,
        "vk_opd: hidden must be F32 (got {:?})",
        hidden.dtype()
    );
    anyhow::ensure!(
        matches!(weight.dtype(), VkDType::F32 | VkDType::Bf16),
        "vk_opd: weight must be F32 or BF16 (got {:?})",
        weight.dtype()
    );
    let num_active = hidden.shape()[0];
    let hidden_size = hidden.shape()[1];
    let vocab = weight.shape()[0];
    anyhow::ensure!(
        weight.shape()[1] == hidden_size,
        "vk_opd: weight inner-dim {} != hidden_dim {hidden_size}",
        weight.shape()[1]
    );
    anyhow::ensure!(top_k == 16 || top_k == 32, "vk_opd: top_k must be 16 or 32 (got {top_k})");
    let expected = num_active * top_k;
    anyhow::ensure!(
        teacher_topk_indices.len() == expected,
        "vk_opd: teacher_topk_indices.len() {} != num_active * top_k = {expected}",
        teacher_topk_indices.len()
    );
    anyhow::ensure!(
        teacher_topk_logprobs.len() == expected,
        "vk_opd: teacher_topk_logprobs.len() {} != num_active * top_k = {expected}",
        teacher_topk_logprobs.len()
    );
    for (i, &idx) in teacher_topk_indices.iter().enumerate() {
        anyhow::ensure!(
            (idx as usize) < vocab,
            "vk_opd: teacher_topk_indices[{i}] = {idx} >= vocab {vocab}"
        );
    }
    Ok((num_active, hidden_size, vocab))
}

/// Resident-buffer forward dispatch (§9.10).
///
/// Takes raw buffer handles + scalar push constants; writes per-token
/// reverse-KL into `kl_out_handle` (caller-allocated, length `num_active`
/// floats). Used by both the high-level `vk_opd_top_k_reverse_kl_loss`
/// path and external callers that want to fold this dispatch into a larger
/// command submission.
pub fn dispatch_opd_topk_kl_fwd_resident(
    device: &VulkanDevice,
    hidden_handle: vk::Buffer,
    weight_handle: vk::Buffer,
    weight_is_bf16: bool,
    topk_idx_handle: vk::Buffer,
    topk_lp_q_handle: vk::Buffer,
    kl_out_handle: vk::Buffer,
    num_active: u32,
    hidden_size: u32,
    vocab_size: u32,
    top_k: u32,
) -> Result<()> {
    debug_assert!(top_k == 16 || top_k == 32);
    if num_active == 0 {
        return Ok(());
    }
    let shader = if weight_is_bf16 {
        "vk_opd_topk_kl_fwd_bf16w"
    } else {
        "vk_opd_topk_kl_fwd_f32"
    };
    let push = [hidden_size, vocab_size, num_active, top_k];
    dispatch_simple(
        device,
        shader,
        &[
            hidden_handle,
            weight_handle,
            topk_idx_handle,
            topk_lp_q_handle,
            kl_out_handle,
        ],
        &push,
        num_active,
    )
}

/// Resident-buffer backward dispatch (§9.10).
///
/// Computes `d_hidden = ∂L/∂hidden` into `d_hidden_handle` (caller-allocated,
/// `num_active * hidden_size` f32 elements). `output_mode = 0` means the
/// forward emitted a scalar mean and `grad_loss_handle` points to a single
/// f32; `output_mode = 1` means per-position and `grad_loss_handle` is
/// `num_active` floats. `scale_factor` is `1 / num_active` for mode 0 and
/// `1.0` for mode 1 (caller multiplies in).
pub fn dispatch_opd_topk_kl_bwd_resident(
    device: &VulkanDevice,
    hidden_handle: vk::Buffer,
    weight_handle: vk::Buffer,
    weight_is_bf16: bool,
    topk_idx_handle: vk::Buffer,
    topk_lp_q_handle: vk::Buffer,
    grad_loss_handle: vk::Buffer,
    d_hidden_handle: vk::Buffer,
    num_active: u32,
    hidden_size: u32,
    vocab_size: u32,
    top_k: u32,
    output_mode: u32,
    scale_factor: f32,
) -> Result<()> {
    debug_assert!(top_k == 16 || top_k == 32);
    debug_assert!(output_mode == 0 || output_mode == 1);
    if num_active == 0 {
        return Ok(());
    }
    let shader = if weight_is_bf16 {
        "vk_opd_topk_kl_bwd_bf16w"
    } else {
        "vk_opd_topk_kl_bwd_f32"
    };
    let push = [
        hidden_size,
        vocab_size,
        num_active,
        top_k,
        output_mode,
        scale_factor.to_bits(),
    ];
    dispatch_simple(
        device,
        shader,
        &[
            hidden_handle,
            weight_handle,
            topk_idx_handle,
            topk_lp_q_handle,
            grad_loss_handle,
            d_hidden_handle,
        ],
        &push,
        num_active,
    )
}

/// Resident-buffer metrics dispatch (§3.8).
///
/// Emits `[num_active, 3]` f32 rows: `(H(p_hat), H(q_hat), KL_t)`.
pub fn dispatch_opd_topk_metrics_resident(
    device: &VulkanDevice,
    hidden_handle: vk::Buffer,
    weight_handle: vk::Buffer,
    weight_is_bf16: bool,
    topk_idx_handle: vk::Buffer,
    topk_lp_q_handle: vk::Buffer,
    metrics_out_handle: vk::Buffer,
    num_active: u32,
    hidden_size: u32,
    vocab_size: u32,
    top_k: u32,
) -> Result<()> {
    debug_assert!(top_k == 16 || top_k == 32);
    if num_active == 0 {
        return Ok(());
    }
    let shader = if weight_is_bf16 {
        "vk_opd_topk_metrics_bf16w"
    } else {
        "vk_opd_topk_metrics_f32"
    };
    let push = [hidden_size, vocab_size, num_active, top_k];
    dispatch_simple(
        device,
        shader,
        &[
            hidden_handle,
            weight_handle,
            topk_idx_handle,
            topk_lp_q_handle,
            metrics_out_handle,
        ],
        &push,
        num_active,
    )
}

/// State captured by the forward pass for use by the analytic backward.
/// Owns the uploaded teacher tensors so the backward kernel can re-read
/// them without another host-side allocation.
#[derive(Debug)]
struct OpdLossState {
    weight: VkTensor,
    topk_idx_buf: Arc<VulkanBuffer>,
    topk_lp_q_buf: Arc<VulkanBuffer>,
    num_active: usize,
    hidden_size: usize,
    vocab: usize,
    top_k: usize,
    output_mode: OpdLossOutputMode,
    inputs: [VkTensor; 1],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpdLossOutputMode {
    /// Forward returned a `[1]` scalar mean.
    ScalarMean,
    /// Forward returned a `[T_active]` per-position vector.
    PerPosition,
}

#[derive(Debug)]
pub struct OpdLossBackward {
    state: OpdLossState,
}

impl OpdLossBackward {
    fn run_backward(&self, grad_out: &VkTensor) -> Result<VkTensor> {
        let hidden = &self.state.inputs[0];
        let device = hidden.device();
        let dtype = hidden.dtype();
        anyhow::ensure!(dtype == VkDType::F32, "OPD bwd: hidden must be F32");
        let n = self.state.num_active * self.state.hidden_size;
        let d_hidden = alloc_f32(device, n)?;

        if self.state.num_active == 0 {
            // No active tokens → the backward kernel would launch zero
            // workgroups and write nothing. Zero-init the empty buffer
            // anyway so the caller's downstream accumulators see a clean
            // tensor in case the empty path is interleaved with non-empty
            // ones.
            return Ok(VkTensor::from_buffer(
                d_hidden,
                vec![self.state.num_active, self.state.hidden_size],
                VkDType::F32,
                Arc::clone(device),
            ));
        }

        // grad_loss buffer + scale factor.
        // ScalarMean: caller passes a scalar; scale = 1 / num_active.
        // PerPosition: caller passes [num_active]; scale = 1.0.
        let (grad_buf, output_mode, scale_factor): (Arc<VulkanBuffer>, u32, f32) =
            match self.state.output_mode {
                OpdLossOutputMode::ScalarMean => {
                    // grad_out is a scalar VkTensor (shape `[1]`).
                    anyhow::ensure!(
                        grad_out.num_elements() == 1,
                        "OPD bwd ScalarMean: grad_out must be scalar [1], got {:?}",
                        grad_out.shape()
                    );
                    (
                        Arc::clone(grad_out.buffer()),
                        0,
                        1.0_f32 / (self.state.num_active as f32),
                    )
                }
                OpdLossOutputMode::PerPosition => {
                    anyhow::ensure!(
                        grad_out.shape() == [self.state.num_active],
                        "OPD bwd PerPosition: grad_out must be [num_active]={}, got {:?}",
                        self.state.num_active,
                        grad_out.shape()
                    );
                    (Arc::clone(grad_out.buffer()), 1, 1.0_f32)
                }
            };

        let weight_is_bf16 = self.state.weight.dtype() == VkDType::Bf16;
        dispatch_opd_topk_kl_bwd_resident(
            device,
            hidden.buffer().handle(),
            self.state.weight.buffer().handle(),
            weight_is_bf16,
            self.state.topk_idx_buf.handle(),
            self.state.topk_lp_q_buf.handle(),
            grad_buf.handle(),
            d_hidden.handle(),
            self.state.num_active as u32,
            self.state.hidden_size as u32,
            self.state.vocab as u32,
            self.state.top_k as u32,
            output_mode,
            scale_factor,
        )?;

        Ok(VkTensor::from_buffer(
            d_hidden,
            vec![self.state.num_active, self.state.hidden_size],
            VkDType::F32,
            Arc::clone(device),
        ))
    }
}

impl VkBackwardOp for OpdLossBackward {
    fn op_name(&self) -> &'static str {
        "opd_topk_kl"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.state.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let dh = self.run_backward(grad_out)?;
        Ok(vec![Some(dh)])
    }
}

/// Scalar-mean OPD loss. The autograd tape attaches `OpdLossBackward` so
/// the gradient w.r.t. `hidden` flows analytically through the fused bwd
/// kernel — no autodiff recompute of the gather + matmul.
pub fn vk_opd_top_k_reverse_kl_loss(
    hidden: &VkTensor,
    weight: &VkTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    top_k: usize,
) -> Result<VkTensor> {
    apply_op(
        hidden,
        weight,
        teacher_topk_indices,
        teacher_topk_logprobs,
        top_k,
        OpdLossOutputMode::ScalarMean,
    )
}

/// Per-position OPD reverse-KL. Returns a `[T_active]` F32 tensor used by
/// the GRPO importance-sampling advantage path (`A_t = -KL_t`).
pub fn vk_opd_top_k_reverse_kl_per_position(
    hidden: &VkTensor,
    weight: &VkTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    top_k: usize,
) -> Result<VkTensor> {
    apply_op(
        hidden,
        weight,
        teacher_topk_indices,
        teacher_topk_logprobs,
        top_k,
        OpdLossOutputMode::PerPosition,
    )
}

fn apply_op(
    hidden: &VkTensor,
    weight: &VkTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    top_k: usize,
    mode: OpdLossOutputMode,
) -> Result<VkTensor> {
    let (num_active, hidden_size, vocab) =
        validate_opd_inputs(hidden, weight, teacher_topk_indices, teacher_topk_logprobs, top_k)?;
    let device = hidden.device();
    if num_active == 0 {
        // Empty mask: return a zero scalar / empty per-position vector.
        return match mode {
            OpdLossOutputMode::ScalarMean => {
                let buf = alloc_f32(device, 1)?;
                let push = [1u32, 0.0_f32.to_bits()];
                dispatch_simple(device, "vk_fill_f32", &[buf.handle()], &push, 1)?;
                Ok(VkTensor::from_buffer(buf, vec![1], VkDType::F32, Arc::clone(device)))
            }
            OpdLossOutputMode::PerPosition => {
                let buf = alloc_f32(device, 0.max(1))?;
                Ok(VkTensor::from_buffer(buf, vec![0], VkDType::F32, Arc::clone(device)))
            }
        };
    }

    // Upload teacher tensors once; the same Arc<VulkanBuffer> is shared
    // with `OpdLossBackward` so the backward kernel reads them in place.
    let topk_idx_buf = upload_u32(device, teacher_topk_indices)?;
    let topk_lp_q_buf = upload_f32(device, teacher_topk_logprobs)?;

    // Per-position KL output.
    let per_pos_buf = alloc_f32(device, num_active)?;
    let weight_is_bf16 = weight.dtype() == VkDType::Bf16;
    dispatch_opd_topk_kl_fwd_resident(
        device,
        hidden.buffer().handle(),
        weight.buffer().handle(),
        weight_is_bf16,
        topk_idx_buf.handle(),
        topk_lp_q_buf.handle(),
        per_pos_buf.handle(),
        num_active as u32,
        hidden_size as u32,
        vocab as u32,
        top_k as u32,
    )?;

    let per_pos_tensor =
        VkTensor::from_buffer(per_pos_buf, vec![num_active], VkDType::F32, Arc::clone(device));

    let state = OpdLossState {
        weight: weight.clone(),
        topk_idx_buf,
        topk_lp_q_buf,
        num_active,
        hidden_size,
        vocab,
        top_k,
        output_mode: mode,
        inputs: [hidden.clone()],
    };

    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if hidden.requires_grad() {
        Some(Arc::new(OpdLossBackward { state }))
    } else {
        None
    };

    match mode {
        OpdLossOutputMode::ScalarMean => {
            let mean = vk_mean_all(&per_pos_tensor)?;
            // `vk_mean_all` returns a `[1]` tensor; we override its grad_fn
            // (which would have backproped via the mean's own backward) with
            // ours, which is the analytic OPD path. This mirrors the
            // `vk_flce_loss` trick.
            Ok(VkTensor::from_op(
                Arc::clone(mean.buffer()),
                vec![1],
                VkDType::F32,
                Arc::clone(device),
                grad_fn,
            ))
        }
        OpdLossOutputMode::PerPosition => Ok(VkTensor::from_op(
            Arc::clone(per_pos_tensor.buffer()),
            vec![num_active],
            VkDType::F32,
            Arc::clone(device),
            grad_fn,
        )),
    }
}

/// Per-position distribution-alignment metrics (§3.8).
///
/// Returns a `[T_active, 3]` F32 `VkTensor` with columns
/// `(H(p_hat), H(q_hat), KL_t)`. Detached — no autograd link, since the
/// metrics call runs at validation cadence, not every training step.
pub fn vk_opd_top_k_metrics(
    hidden: &VkTensor,
    weight: &VkTensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    top_k: usize,
) -> Result<VkTensor> {
    let (num_active, hidden_size, vocab) =
        validate_opd_inputs(hidden, weight, teacher_topk_indices, teacher_topk_logprobs, top_k)?;
    let device = hidden.device();
    if num_active == 0 {
        let buf = alloc_f32(device, 1)?;
        return Ok(VkTensor::from_buffer(buf, vec![0, 3], VkDType::F32, Arc::clone(device)));
    }

    let topk_idx_buf = upload_u32(device, teacher_topk_indices)?;
    let topk_lp_q_buf = upload_f32(device, teacher_topk_logprobs)?;
    let out_buf = alloc_f32(device, num_active * 3)?;
    let weight_is_bf16 = weight.dtype() == VkDType::Bf16;
    dispatch_opd_topk_metrics_resident(
        device,
        hidden.buffer().handle(),
        weight.buffer().handle(),
        weight_is_bf16,
        topk_idx_buf.handle(),
        topk_lp_q_buf.handle(),
        out_buf.handle(),
        num_active as u32,
        hidden_size as u32,
        vocab as u32,
        top_k as u32,
    )?;

    Ok(VkTensor::from_buffer(
        out_buf,
        vec![num_active, 3],
        VkDType::F32,
        Arc::clone(device),
    ))
}

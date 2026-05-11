//! Fused Linear Cross-Entropy (FLCE) on VkTensor.
//!
//! Computes mean cross-entropy loss over `num_active` rows of a hidden
//! state matmul against the LM head weight, *without* materializing
//! the full `[num_active, vocab]` logits tensor. The vocab dim is
//! processed in chunks of `chunk_len` columns.
//!
//! Forward (per row): online log-sum-exp over chunks + a gathered
//! "correct logit". Per-row loss = log_sum_exp - correct_logit; total
//! loss = mean over rows.
//!
//! Backward: produces gradient w.r.t. the hidden state by recomputing
//! per-chunk logits → softmax-minus-onehot → matmul-with-W-chunk
//! (yielding [num_active, hidden] partials that sum to dL/dhidden).
//!
//! Inputs:
//!   hidden: [num_active, hidden_dim]  (already pre-gathered to active tokens)
//!   weight: [vocab, hidden_dim]       (LM head; frozen for SFT/LoRA)
//!   labels: [num_active] u32          (target token ids)
//!
//! Output:
//!   loss: scalar VkTensor (shape [1]) with `requires_grad=true` and
//!   a backward op closing over hidden, weight, labels, and the
//!   per-row global_max / global_sumexp.

use crate::vk_ops::dispatch_simple;
use crate::vk_ops::matmul::{vk_matmul, vk_matmul_no_grad};
use crate::vk_ops::reduce::vk_mean_all;
use crate::vk_ops::shape::vk_transpose_2d_no_grad;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n * 4).max(4);
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes as u64,
    )
    .context("vk_flce: alloc f32")?;
    Ok(Arc::new(buf))
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

/// Computed in `flce_forward_workspace`, used by the backward op.
struct FlceState {
    weight_t: VkTensor, // [hidden, vocab] - transposed weight
    labels_buf: Arc<VulkanBuffer>,
    global_max: Arc<VulkanBuffer>,
    global_sumexp: Arc<VulkanBuffer>,
    num_active: usize,
    vocab: usize,
    hidden_dim: usize,
    chunk_len: usize,
}

fn run_flce_forward(
    hidden: &VkTensor,
    weight_t: &VkTensor,
    labels_buf: &Arc<VulkanBuffer>,
    num_active: usize,
    hidden_dim: usize,
    vocab: usize,
    chunk_len: usize,
) -> Result<(VkTensor, Arc<VulkanBuffer>, Arc<VulkanBuffer>)> {
    let device = hidden.device();
    let global_max = alloc_f32(device, num_active)?;
    let global_sumexp = alloc_f32(device, num_active)?;
    let correct = alloc_f32(device, num_active)?;
    // Initialize correct to 0
    {
        let workgroups = ((num_active + 255) / 256) as u32;
        let push = [num_active as u32, 0.0_f32.to_bits()];
        dispatch_simple(
            device,
            "vk_fill_f32",
            &[correct.handle()],
            &push,
            workgroups,
        )?;
    }

    let mut chunk_off = 0usize;
    let mut first = true;
    while chunk_off < vocab {
        let cur_len = chunk_len.min(vocab - chunk_off);
        // Slice the transposed weight column-block [hidden_dim, cur_len].
        // Since weight_t is row-major [hidden_dim, vocab], the column
        // block at offset chunk_off is contiguous *within rows* but
        // strided across rows. The simplest correct path is to
        // materialize a contiguous chunk buffer via a 2D transpose-copy
        // shader. For Phase E correctness we read it back to host and
        // re-upload — slower but correct; replace with strided dispatch
        // later.
        let chunk_w_t = extract_weight_t_chunk(weight_t, chunk_off, cur_len)?;
        // logits_chunk = hidden @ chunk_w_t  → [num_active, cur_len]
        let logits_chunk = vk_matmul_no_grad(hidden, &chunk_w_t)?;
        // chunk stats
        let chunk_max = alloc_f32(device, num_active)?;
        let chunk_sumexp = alloc_f32(device, num_active)?;
        let push = [num_active as u32, cur_len as u32];
        dispatch_simple(
            device,
            "vk_flce_chunk_stats_f32",
            &[
                logits_chunk.buffer().handle(),
                chunk_max.handle(),
                chunk_sumexp.handle(),
            ],
            &push,
            num_active as u32,
        )?;
        // combine into global
        let combine_push = [num_active as u32, if first { 1 } else { 0 }];
        dispatch_simple(
            device,
            "vk_flce_log_sum_exp_combine_f32",
            &[
                chunk_max.handle(),
                chunk_sumexp.handle(),
                global_max.handle(),
                global_sumexp.handle(),
            ],
            &combine_push,
            ((num_active + 255) / 256) as u32,
        )?;
        // gather correct logit if in this chunk
        let gather_push = [num_active as u32, cur_len as u32, chunk_off as u32];
        dispatch_simple(
            device,
            "vk_flce_gather_correct_f32",
            &[
                labels_buf.handle(),
                logits_chunk.buffer().handle(),
                correct.handle(),
            ],
            &gather_push,
            ((num_active + 255) / 256) as u32,
        )?;
        chunk_off += cur_len;
        first = false;
    }

    // per_row_loss = log(global_sumexp) + global_max - correct
    let per_row = alloc_f32(device, num_active)?;
    let push = [num_active as u32];
    dispatch_simple(
        device,
        "vk_flce_per_token_loss_f32",
        &[
            global_max.handle(),
            global_sumexp.handle(),
            correct.handle(),
            per_row.handle(),
        ],
        &push,
        ((num_active + 255) / 256) as u32,
    )?;

    let per_row_tensor = VkTensor::from_buffer(
        per_row,
        vec![num_active],
        VkDType::F32,
        Arc::clone(device),
    );
    Ok((per_row_tensor, global_max, global_sumexp))
}

/// Materialize a contiguous F32 chunk of weight_t covering vocab columns
/// `[chunk_off, chunk_off + cur_len)` of shape `[hidden_dim, cur_len]`.
///
/// For Phase E we go through a CPU readback + re-upload. A dedicated
/// `vk_slice_2d` shader is a small Phase G optimization.
fn extract_weight_t_chunk(
    weight_t: &VkTensor,
    chunk_off: usize,
    cur_len: usize,
) -> Result<VkTensor> {
    let dev = weight_t.device();
    let hidden_dim = weight_t.shape()[0];
    let vocab = weight_t.shape()[1];
    anyhow::ensure!(
        chunk_off + cur_len <= vocab,
        "flce chunk OOB: {chunk_off}+{cur_len} > {vocab}"
    );
    let full = weight_t.to_vec_f32()?;
    let mut chunk = vec![0.0_f32; hidden_dim * cur_len];
    for h in 0..hidden_dim {
        for c in 0..cur_len {
            chunk[h * cur_len + c] = full[h * vocab + chunk_off + c];
        }
    }
    let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        bytes.len().max(4) as u64,
    )?;
    VulkanBuffer::upload_data(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &buf,
        &bytes,
    )?;
    Ok(VkTensor::from_buffer(
        Arc::new(buf),
        vec![hidden_dim, cur_len],
        VkDType::F32,
        Arc::clone(dev),
    ))
}

#[derive(Debug)]
pub struct FlceBackward {
    // captured by closure-ish state for backward
    pub weight: VkTensor,
    pub labels: Vec<u32>,
    pub global_max: Arc<VulkanBuffer>,
    pub global_sumexp: Arc<VulkanBuffer>,
    pub num_active: usize,
    pub vocab: usize,
    pub hidden_dim: usize,
    pub chunk_len: usize,
    pub inputs: [VkTensor; 1], // hidden
}

impl VkBackwardOp for FlceBackward {
    fn op_name(&self) -> &'static str {
        "flce"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let hidden = &self.inputs[0];
        let device = hidden.device();
        // grad_out is a scalar from mean(per_row_loss). The chunk
        // gradient uses scale = grad_out_value / num_active. Read back
        // the scalar grad to host (cheap).
        let go = grad_out.to_vec_f32()?[0];
        let scale = go / (self.num_active as f32);
        // accumulator dL/dhidden in F32 [num_active, hidden_dim]
        let grad_hidden = alloc_f32(device, self.num_active * self.hidden_dim)?;
        // initialize to zero
        {
            let workgroups =
                ((self.num_active * self.hidden_dim + 255) / 256) as u32;
            let push = [
                (self.num_active * self.hidden_dim) as u32,
                0.0_f32.to_bits(),
            ];
            dispatch_simple(
                device,
                "vk_fill_f32",
                &[grad_hidden.handle()],
                &push,
                workgroups,
            )?;
        }
        // labels buffer (re-upload from saved Vec — small, cheap)
        let labels_buf = upload_u32(device, &self.labels)?;
        // weight_t once
        let weight_t = vk_transpose_2d_no_grad(&self.weight)?;

        let mut chunk_off = 0usize;
        while chunk_off < self.vocab {
            let cur_len = self.chunk_len.min(self.vocab - chunk_off);
            // Recompute logits_chunk
            let chunk_w_t = extract_weight_t_chunk(&weight_t, chunk_off, cur_len)?;
            let logits_chunk = vk_matmul_no_grad(hidden, &chunk_w_t)?;
            // Convert logits_chunk in-place to grad_logits_chunk
            let push = [
                self.num_active as u32,
                cur_len as u32,
                chunk_off as u32,
                scale.to_bits(),
            ];
            let workgroups =
                (((self.num_active * cur_len) + 255) / 256) as u32;
            dispatch_simple(
                device,
                "vk_flce_grad_chunk_f32",
                &[
                    labels_buf.handle(),
                    self.global_max.handle(),
                    self.global_sumexp.handle(),
                    logits_chunk.buffer().handle(),
                ],
                &push,
                workgroups,
            )?;
            // d hidden_chunk = grad_logits_chunk @ W_chunk
            //  shape: [num_active, cur_len] @ [cur_len, hidden_dim]
            //  → need W_chunk = weight[chunk_off:chunk_off+cur_len, :]
            //    which is contiguous in original (vocab-major) layout.
            let w_chunk = extract_weight_chunk_rows(&self.weight, chunk_off, cur_len)?;
            let dh_chunk = vk_matmul_no_grad(&logits_chunk, &w_chunk)?;
            // accumulate into grad_hidden
            let workgroups =
                (((self.num_active * self.hidden_dim) + 255) / 256) as u32;
            let push = [
                (self.num_active * self.hidden_dim) as u32,
                0u32, // OP_ADD
            ];
            // We need to add dh_chunk into grad_hidden (existing in
            // place). vk_elementwise_binary_f32 writes to a third
            // buffer; we then copy back. Use a temp buffer.
            let tmp = alloc_f32(device, self.num_active * self.hidden_dim)?;
            dispatch_simple(
                device,
                "vk_elementwise_binary_f32",
                &[
                    grad_hidden.handle(),
                    dh_chunk.buffer().handle(),
                    tmp.handle(),
                ],
                &push,
                workgroups,
            )?;
            // Now copy tmp back to grad_hidden (cheap GPU→GPU via a fill
            // shader? No — easier: swap roles by re-using the
            // accumulator buffer. To avoid extra alloc per chunk, the
            // cleanest path is to bind grad_hidden as out for next
            // iteration. We just swap the Arc bindings.
            // Implementing: grad_hidden = tmp (move).
            // But we keep grad_hidden as the final, so copy via
            // element-wise add with zeros into grad_hidden.
            let zero = alloc_f32(device, self.num_active * self.hidden_dim)?;
            {
                let wg = workgroups;
                let push_zero = [
                    (self.num_active * self.hidden_dim) as u32,
                    0.0_f32.to_bits(),
                ];
                dispatch_simple(
                    device,
                    "vk_fill_f32",
                    &[zero.handle()],
                    &push_zero,
                    wg,
                )?;
            }
            let push_add = [
                (self.num_active * self.hidden_dim) as u32,
                0u32, // ADD
            ];
            dispatch_simple(
                device,
                "vk_elementwise_binary_f32",
                &[tmp.handle(), zero.handle(), grad_hidden.handle()],
                &push_add,
                workgroups,
            )?;
            chunk_off += cur_len;
        }

        Ok(vec![Some(VkTensor::from_buffer(
            grad_hidden,
            vec![self.num_active, self.hidden_dim],
            VkDType::F32,
            Arc::clone(device),
        ))])
    }
}

fn extract_weight_chunk_rows(
    weight: &VkTensor,
    chunk_off: usize,
    cur_len: usize,
) -> Result<VkTensor> {
    // weight has shape [vocab, hidden_dim]; we want rows
    // [chunk_off, chunk_off + cur_len). That's a contiguous slice.
    let dev = weight.device();
    let hidden_dim = weight.shape()[1];
    let full = weight.to_vec_f32()?;
    let start = chunk_off * hidden_dim;
    let end = start + cur_len * hidden_dim;
    let slice = &full[start..end];
    let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        bytes.len().max(4) as u64,
    )?;
    VulkanBuffer::upload_data(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &buf,
        &bytes,
    )?;
    Ok(VkTensor::from_buffer(
        Arc::new(buf),
        vec![cur_len, hidden_dim],
        VkDType::F32,
        Arc::clone(dev),
    ))
}

/// Default FLCE chunk length (matches the existing kiln-flce-kernel
/// chunking heuristic).
pub const FLCE_DEFAULT_CHUNK: usize = 4096;

pub fn vk_flce_loss(
    hidden: &VkTensor,
    weight: &VkTensor,
    labels: &[u32],
    chunk_len: usize,
) -> Result<VkTensor> {
    anyhow::ensure!(
        hidden.shape().len() == 2 && weight.shape().len() == 2,
        "vk_flce: hidden and weight must be rank-2"
    );
    anyhow::ensure!(
        hidden.dtype() == VkDType::F32 && weight.dtype() == VkDType::F32,
        "vk_flce: F32-only"
    );
    let num_active = hidden.shape()[0];
    let hidden_dim = hidden.shape()[1];
    let vocab = weight.shape()[0];
    anyhow::ensure!(
        weight.shape()[1] == hidden_dim,
        "vk_flce: weight inner-dim {} != hidden_dim {hidden_dim}",
        weight.shape()[1]
    );
    anyhow::ensure!(
        labels.len() == num_active,
        "vk_flce: labels.len() {} != num_active {num_active}",
        labels.len()
    );

    let dev = hidden.device();
    let labels_buf = upload_u32(dev, labels)?;
    // Materialize weight.T once (cheap relative to total cost).
    let weight_t = vk_transpose_2d_no_grad(weight)?;

    let (per_row, global_max, global_sumexp) = run_flce_forward(
        hidden,
        &weight_t,
        &labels_buf,
        num_active,
        hidden_dim,
        vocab,
        chunk_len,
    )?;

    // loss = mean(per_row) — but use vk_mean_all which has its own
    // backward; we override with our analytic FLCE backward instead.
    let loss = vk_mean_all(&per_row)?;
    // Build a fresh VkTensor cloning loss buffer but with our
    // FlceBackward replacing the mean's backward chain. We attach the
    // backward directly to `hidden` for clarity.
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if hidden.requires_grad() {
        Some(Arc::new(FlceBackward {
            weight: weight.clone(),
            labels: labels.to_vec(),
            global_max,
            global_sumexp,
            num_active,
            vocab,
            hidden_dim,
            chunk_len,
            inputs: [hidden.clone()],
        }))
    } else {
        None
    };
    // Important: we cannot easily intercept the mean's autograd. So we
    // return a "scalar wrapper" VkTensor whose grad_fn IS our analytic
    // FlceBackward. The user calls `vk_backward(&loss)` which seeds
    // grad=1 and calls our backward directly.
    Ok(VkTensor::from_op(
        Arc::clone(loss.buffer()),
        vec![1],
        VkDType::F32,
        Arc::clone(dev),
        grad_fn,
    ))
}

#[allow(dead_code)]
fn _unused_keep_vk_matmul_export(_a: &VkTensor, _b: &VkTensor) -> Option<()> {
    let _ = vk_matmul;
    None
}

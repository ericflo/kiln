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
use anyhow::Result;
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

/// Computed in `flce_forward_workspace`, used by the backward op.
struct FlceState {
    weight_t: VkTensor, // [hidden, vocab] - transposed weight
    labels_buf: Arc<VulkanBuffer>,
    global_max: Arc<VulkanBuffer>,
    global_sumexp: Arc<VulkanBuffer>,
    num_active: usize,
    vocab: usize,
    _hidden_dim: usize,
    chunk_len: usize,
}

fn run_flce_forward(
    hidden: &VkTensor,
    weight: &VkTensor,
    labels_buf: &Arc<VulkanBuffer>,
    num_active: usize,
    _hidden_dim: usize,
    vocab: usize,
    chunk_len: usize,
) -> Result<(
    VkTensor,
    Arc<VulkanBuffer>,
    Arc<VulkanBuffer>,
    Arc<VulkanBuffer>,
)> {
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
        // Materialize only this vocab chunk, then transpose that small
        // block. Avoids a full `[hidden_dim, vocab]` F32 LM-head
        // transpose, which costs multiple GiB at Qwen3.5 vocab size.
        let w_chunk = extract_weight_chunk_rows(weight, chunk_off, cur_len)?;
        let chunk_w_t = vk_transpose_2d_no_grad(&w_chunk)?;
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

    let per_row_tensor =
        VkTensor::from_buffer(per_row, vec![num_active], VkDType::F32, Arc::clone(device));
    Ok((per_row_tensor, global_max, global_sumexp, correct))
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
            let workgroups = ((self.num_active * self.hidden_dim + 255) / 256) as u32;
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
        let mut chunk_off = 0usize;
        while chunk_off < self.vocab {
            let cur_len = self.chunk_len.min(self.vocab - chunk_off);
            // Recompute logits_chunk
            let w_chunk = extract_weight_chunk_rows(&self.weight, chunk_off, cur_len)?;
            let chunk_w_t = vk_transpose_2d_no_grad(&w_chunk)?;
            let logits_chunk = vk_matmul_no_grad(hidden, &chunk_w_t)?;
            // Convert logits_chunk in-place to grad_logits_chunk
            let push = [
                self.num_active as u32,
                cur_len as u32,
                chunk_off as u32,
                scale.to_bits(),
            ];
            let workgroups = (((self.num_active * cur_len) + 255) / 256) as u32;
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
            //  → W_chunk is contiguous in original (vocab-major) layout.
            let dh_chunk = vk_matmul_no_grad(&logits_chunk, &w_chunk)?;
            // accumulate into grad_hidden
            let workgroups = (((self.num_active * self.hidden_dim) + 255) / 256) as u32;
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
                dispatch_simple(device, "vk_fill_f32", &[zero.handle()], &push_zero, wg)?;
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

#[derive(Debug)]
pub struct GrpoBackward {
    pub weight: VkTensor,
    pub labels: Vec<u32>,
    pub global_max: Arc<VulkanBuffer>,
    pub global_sumexp: Arc<VulkanBuffer>,
    pub coeff: Arc<VulkanBuffer>,
    pub num_active: usize,
    pub vocab: usize,
    pub hidden_dim: usize,
    pub chunk_len: usize,
    pub inputs: [VkTensor; 1], // hidden
}

impl VkBackwardOp for GrpoBackward {
    fn op_name(&self) -> &'static str {
        "grpo"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let hidden = &self.inputs[0];
        let device = hidden.device();
        let go = grad_out.to_vec_f32()?[0];
        let scale = go / (self.num_active as f32);

        let grad_hidden = alloc_f32(device, self.num_active * self.hidden_dim)?;
        {
            let workgroups = ((self.num_active * self.hidden_dim + 255) / 256) as u32;
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

        let labels_buf = upload_u32(device, &self.labels)?;
        let mut chunk_off = 0usize;
        while chunk_off < self.vocab {
            let cur_len = self.chunk_len.min(self.vocab - chunk_off);
            let w_chunk = extract_weight_chunk_rows(&self.weight, chunk_off, cur_len)?;
            let chunk_w_t = vk_transpose_2d_no_grad(&w_chunk)?;
            let logits_chunk = vk_matmul_no_grad(hidden, &chunk_w_t)?;

            let push = [
                self.num_active as u32,
                cur_len as u32,
                chunk_off as u32,
                scale.to_bits(),
            ];
            let workgroups = (((self.num_active * cur_len) + 255) / 256) as u32;
            dispatch_simple(
                device,
                "vk_grpo_grad_chunk_f32",
                &[
                    labels_buf.handle(),
                    self.global_max.handle(),
                    self.global_sumexp.handle(),
                    self.coeff.handle(),
                    logits_chunk.buffer().handle(),
                ],
                &push,
                workgroups,
            )?;

            let dh_chunk = vk_matmul_no_grad(&logits_chunk, &w_chunk)?;
            let workgroups = (((self.num_active * self.hidden_dim) + 255) / 256) as u32;
            let push = [
                (self.num_active * self.hidden_dim) as u32,
                0u32, // OP_ADD
            ];
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
            let zero = alloc_f32(device, self.num_active * self.hidden_dim)?;
            {
                let push_zero = [
                    (self.num_active * self.hidden_dim) as u32,
                    0.0_f32.to_bits(),
                ];
                dispatch_simple(
                    device,
                    "vk_fill_f32",
                    &[zero.handle()],
                    &push_zero,
                    workgroups,
                )?;
            }
            dispatch_simple(
                device,
                "vk_elementwise_binary_f32",
                &[tmp.handle(), zero.handle(), grad_hidden.handle()],
                &push,
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
    // weight is [vocab, hidden_dim]; row-slicing rows
    // [chunk_off, chunk_off + cur_len) is a contiguous byte range, which
    // we express as a flat narrow on the second dim of [1, vocab*hidden].
    // Pure GPU; replaces a 2.5 GB CPU readback per chunk (Qwen3.5-4B
    // vocab=248K × hidden=2560 × 4 bytes).
    let hidden_dim = weight.shape()[1];
    let vocab = weight.shape()[0];
    let flat = crate::vk_ops::shape::vk_reshape(weight, &[1, vocab * hidden_dim])?;
    let sliced = crate::vk_ops::narrow::vk_narrow_lastdim_no_grad(
        &flat,
        chunk_off * hidden_dim,
        cur_len * hidden_dim,
    )?;
    crate::vk_ops::shape::vk_reshape(&sliced, &[cur_len, hidden_dim])
}

/// Conservative fallback FLCE vocab-chunk length for legacy callers that
/// cannot provide tensor shape/device limits. Production Vulkan training uses
/// `flce_recommended_chunk_len_for_tensors` instead.
pub const FLCE_DEFAULT_CHUNK: usize = 128;
const FLCE_MIN_CHUNK: usize = 1;
const FLCE_MAX_AUTO_CHUNK: usize = 4096;
const FLCE_FALLBACK_SCRATCH_BUDGET_BYTES: u64 = 64 * 1024 * 1024;
const FLCE_MIN_SCRATCH_BUDGET_BYTES: u64 = 16 * 1024 * 1024;
const FLCE_MAX_SCRATCH_BUDGET_BYTES: u64 = 512 * 1024 * 1024;

/// Returns the active FLCE chunk length, honoring `KILN_VK_FLCE_CHUNK_LEN`
/// when set.
///
/// Prefer `flce_recommended_chunk_len_for_tensors` for new Vulkan-native
/// training code so ordinary use is shape/device-aware without manual tuning.
pub fn flce_active_chunk_len() -> usize {
    std::env::var("KILN_VK_FLCE_CHUNK_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(FLCE_DEFAULT_CHUNK)
}

fn env_flce_chunk_len() -> Option<usize> {
    std::env::var("KILN_VK_FLCE_CHUNK_LEN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
}

fn floor_power_of_two(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        1usize << (usize::BITS - 1 - n.leading_zeros())
    }
}

/// Shape/limit-only chunking heuristic. Public for non-Vulkan preflight tests;
/// runtime code should call `flce_recommended_chunk_len_for_tensors`.
#[doc(hidden)]
pub fn flce_recommended_chunk_len_from_limits(
    num_active: usize,
    hidden_dim: usize,
    vocab: usize,
    device_local_heap_bytes: u64,
    max_workgroups_x: u32,
) -> usize {
    if vocab == 0 {
        return FLCE_MIN_CHUNK;
    }

    let active = num_active.max(1) as u64;
    let hidden = hidden_dim.max(1) as u64;
    // Per vocab column, the peak chunk-dependent buffers are the active
    // logits plus the LM-head slice and its transpose. Fixed per-token hidden
    // accumulators are modeled by preflight, not by this per-chunk chooser.
    let bytes_per_vocab_col = 4u64.saturating_mul(active.saturating_add(2 * hidden));
    let scratch_budget = if device_local_heap_bytes > 0 {
        (device_local_heap_bytes / 128)
            .clamp(FLCE_MIN_SCRATCH_BUDGET_BYTES, FLCE_MAX_SCRATCH_BUDGET_BYTES)
    } else {
        FLCE_FALLBACK_SCRATCH_BUDGET_BYTES
    };
    let by_memory = (scratch_budget / bytes_per_vocab_col).max(1) as usize;

    // 1D per-token shaders dispatch ceil(num_active * chunk_len / 256)
    // workgroups. Keep the automatic chunk inside the device's x-axis limit.
    let max_invocations = (max_workgroups_x as u64).saturating_mul(256).max(256);
    let by_dispatch = (max_invocations / active).max(1) as usize;

    let raw = vocab
        .min(FLCE_MAX_AUTO_CHUNK)
        .min(by_memory)
        .min(by_dispatch)
        .max(FLCE_MIN_CHUNK);
    let safe = floor_power_of_two(raw).max(FLCE_MIN_CHUNK).min(vocab);
    env_flce_chunk_len()
        .map(|forced| forced.clamp(FLCE_MIN_CHUNK, safe))
        .unwrap_or(safe)
}

pub fn flce_recommended_chunk_len(
    device: &VulkanDevice,
    num_active: usize,
    hidden_dim: usize,
    vocab: usize,
) -> usize {
    flce_recommended_chunk_len_from_limits(
        num_active,
        hidden_dim,
        vocab,
        device.device_local_heap_bytes(),
        device.max_compute_work_group_count(0),
    )
}

pub fn flce_recommended_chunk_len_for_tensors(hidden: &VkTensor, weight: &VkTensor) -> usize {
    if hidden.shape().len() != 2 || weight.shape().len() != 2 {
        return flce_active_chunk_len();
    }
    flce_recommended_chunk_len(
        hidden.device(),
        hidden.shape()[0],
        hidden.shape()[1],
        weight.shape()[0],
    )
}

fn validate_lm_head_inputs(
    hidden: &VkTensor,
    weight: &VkTensor,
    labels: &[u32],
) -> Result<(usize, usize, usize)> {
    anyhow::ensure!(
        hidden.shape().len() == 2 && weight.shape().len() == 2,
        "vk lm-head loss/logprob: hidden and weight must be rank-2"
    );
    anyhow::ensure!(
        hidden.dtype() == VkDType::F32 && weight.dtype() == VkDType::F32,
        "vk lm-head loss/logprob: F32-only"
    );
    let num_active = hidden.shape()[0];
    let hidden_dim = hidden.shape()[1];
    let vocab = weight.shape()[0];
    anyhow::ensure!(
        weight.shape()[1] == hidden_dim,
        "vk lm-head loss/logprob: weight inner-dim {} != hidden_dim {hidden_dim}",
        weight.shape()[1]
    );
    anyhow::ensure!(
        labels.len() == num_active,
        "vk lm-head loss/logprob: labels.len() {} != num_active {num_active}",
        labels.len()
    );
    Ok((num_active, hidden_dim, vocab))
}

pub fn vk_selected_log_probs(
    hidden: &VkTensor,
    weight: &VkTensor,
    labels: &[u32],
    chunk_len: usize,
) -> Result<VkTensor> {
    let (num_active, hidden_dim, vocab) = validate_lm_head_inputs(hidden, weight, labels)?;
    let chunk_len = if chunk_len == 0 {
        flce_recommended_chunk_len(hidden.device(), num_active, hidden_dim, vocab)
    } else {
        chunk_len.clamp(FLCE_MIN_CHUNK, vocab)
    };
    let dev = hidden.device();
    let labels_buf = upload_u32(dev, labels)?;
    let (_per_row, global_max, global_sumexp, correct) = run_flce_forward(
        hidden,
        weight,
        &labels_buf,
        num_active,
        hidden_dim,
        vocab,
        chunk_len,
    )?;

    let out = alloc_f32(dev, num_active)?;
    let push = [num_active as u32];
    dispatch_simple(
        dev,
        "vk_selected_logprob_f32",
        &[
            global_max.handle(),
            global_sumexp.handle(),
            correct.handle(),
            out.handle(),
        ],
        &push,
        ((num_active + 255) / 256) as u32,
    )?;
    Ok(VkTensor::from_buffer(
        out,
        vec![num_active],
        VkDType::F32,
        Arc::clone(dev),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn vk_grpo_loss(
    hidden: &VkTensor,
    weight: &VkTensor,
    labels: &[u32],
    ref_log_probs: &VkTensor,
    advantage: f32,
    clip_epsilon: f32,
    kl_coeff: f32,
    chunk_len: usize,
) -> Result<VkTensor> {
    let (num_active, hidden_dim, vocab) = validate_lm_head_inputs(hidden, weight, labels)?;
    let chunk_len = if chunk_len == 0 {
        flce_recommended_chunk_len(hidden.device(), num_active, hidden_dim, vocab)
    } else {
        chunk_len.clamp(FLCE_MIN_CHUNK, vocab)
    };
    anyhow::ensure!(
        ref_log_probs.shape() == [num_active],
        "vk_grpo_loss: ref_log_probs shape {:?} != [{num_active}]",
        ref_log_probs.shape()
    );
    anyhow::ensure!(
        ref_log_probs.dtype() == VkDType::F32,
        "vk_grpo_loss: ref_log_probs must be F32"
    );

    let dev = hidden.device();
    let labels_buf = upload_u32(dev, labels)?;
    let (_ce_per_row, global_max, global_sumexp, correct) = run_flce_forward(
        hidden,
        weight,
        &labels_buf,
        num_active,
        hidden_dim,
        vocab,
        chunk_len,
    )?;

    let per_row = alloc_f32(dev, num_active)?;
    let coeff = alloc_f32(dev, num_active)?;
    let push = [
        num_active as u32,
        advantage.to_bits(),
        clip_epsilon.to_bits(),
        kl_coeff.to_bits(),
    ];
    dispatch_simple(
        dev,
        "vk_grpo_per_token_f32",
        &[
            global_max.handle(),
            global_sumexp.handle(),
            correct.handle(),
            ref_log_probs.buffer().handle(),
            per_row.handle(),
            coeff.handle(),
        ],
        &push,
        ((num_active + 255) / 256) as u32,
    )?;

    let per_row_tensor =
        VkTensor::from_buffer(per_row, vec![num_active], VkDType::F32, Arc::clone(dev));
    let loss = vk_mean_all(&per_row_tensor)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if hidden.requires_grad() {
        Some(Arc::new(GrpoBackward {
            weight: weight.clone(),
            labels: labels.to_vec(),
            global_max,
            global_sumexp,
            coeff,
            num_active,
            vocab,
            hidden_dim,
            chunk_len,
            inputs: [hidden.clone()],
        }))
    } else {
        None
    };
    Ok(VkTensor::from_op(
        Arc::clone(loss.buffer()),
        vec![1],
        VkDType::F32,
        Arc::clone(dev),
        grad_fn,
    ))
}

pub fn vk_flce_loss(
    hidden: &VkTensor,
    weight: &VkTensor,
    labels: &[u32],
    chunk_len: usize,
) -> Result<VkTensor> {
    let (num_active, hidden_dim, vocab) = validate_lm_head_inputs(hidden, weight, labels)?;
    let chunk_len = if chunk_len == 0 {
        flce_recommended_chunk_len(hidden.device(), num_active, hidden_dim, vocab)
    } else {
        chunk_len.clamp(FLCE_MIN_CHUNK, vocab)
    };

    let dev = hidden.device();
    let labels_buf = upload_u32(dev, labels)?;

    let (per_row, global_max, global_sumexp, _correct) = run_flce_forward(
        hidden,
        weight,
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

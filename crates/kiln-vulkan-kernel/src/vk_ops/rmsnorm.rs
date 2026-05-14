//! RMSNorm for `VkTensor`.
//!
//! Wraps the existing `qwen_rmsnorm_forward.comp` / `qwen_rmsnorm_backward.comp`
//! shaders. Both are F32-only. Weight is treated as frozen (no `dW`),
//! matching the existing training path's contract — Qwen3.5 base RMSNorm
//! weights are not LoRA-adapted.
//!
//! Layout: x has shape `[..., hidden]`; weight has shape `[hidden]`.
//! The forward computes `(1 + w) * x / sqrt(mean(x^2) + eps)`.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::Result;
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn dispatch_rmsnorm_forward(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    weight: &VulkanBuffer,
    out: &VulkanBuffer,
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let workgroups = rows as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_rmsnorm_forward: rows {workgroups} > device limit {limit}"
    );
    let push_constants = [rows as u32, hidden as u32, eps.to_bits()];
    dispatch_simple(
        device,
        "qwen_rmsnorm_forward",
        &[x.handle(), weight.handle(), out.handle()],
        &push_constants,
        workgroups,
    )
}

fn dispatch_rmsnorm_backward(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    weight: &VulkanBuffer,
    grad_y: &VulkanBuffer,
    grad_x: &VulkanBuffer,
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let workgroups = rows as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_rmsnorm_backward: rows {workgroups} > device limit {limit}"
    );
    let push_constants = [rows as u32, hidden as u32, eps.to_bits()];
    dispatch_simple(
        device,
        "qwen_rmsnorm_backward",
        &[
            x.handle(),
            weight.handle(),
            grad_y.handle(),
            grad_x.handle(),
        ],
        &push_constants,
        workgroups,
    )
}

fn check_rmsnorm_shapes(x: &VkTensor, weight: &VkTensor) -> Result<(usize, usize)> {
    anyhow::ensure!(
        x.dtype() == VkDType::F32 && weight.dtype() == VkDType::F32,
        "vk_rmsnorm: F32-only (got x={:?}, w={:?})",
        x.dtype(),
        weight.dtype()
    );
    let dims = x.shape();
    anyhow::ensure!(!dims.is_empty(), "vk_rmsnorm: x must have at least 1 dim");
    let hidden = *dims.last().unwrap();
    let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
    anyhow::ensure!(
        weight.shape() == [hidden],
        "vk_rmsnorm: weight shape {:?} does not match hidden {}",
        weight.shape(),
        hidden
    );
    Ok((rows, hidden))
}

#[derive(Debug)]
pub struct RmsNormBackward {
    pub eps: f32,
    pub inputs: [VkTensor; 2], // [x, weight]
}

impl VkBackwardOp for RmsNormBackward {
    fn op_name(&self) -> &'static str {
        "rms_norm"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let x = &self.inputs[0];
        let weight = &self.inputs[1];
        let (rows, hidden) = check_rmsnorm_shapes(x, weight)?;
        let grad_x_buf = alloc_f32(x.device(), rows * hidden)?;
        dispatch_rmsnorm_backward(
            x.device(),
            x.buffer(),
            weight.buffer(),
            grad_out.buffer(),
            &grad_x_buf,
            rows,
            hidden,
            self.eps,
        )?;
        let grad_x = VkTensor::from_buffer(
            grad_x_buf,
            x.shape().to_vec(),
            VkDType::F32,
            Arc::clone(x.device()),
        );
        // Weight is frozen (no gradient). Return None for that input.
        Ok(vec![Some(grad_x), None])
    }
}

pub fn vk_rmsnorm_no_grad(x: &VkTensor, weight: &VkTensor, eps: f32) -> Result<VkTensor> {
    let (rows, hidden) = check_rmsnorm_shapes(x, weight)?;
    let out = alloc_f32(x.device(), rows * hidden)?;
    dispatch_rmsnorm_forward(
        x.device(),
        x.buffer(),
        weight.buffer(),
        &out,
        rows,
        hidden,
        eps,
    )?;
    Ok(VkTensor::from_buffer(
        out,
        x.shape().to_vec(),
        VkDType::F32,
        Arc::clone(x.device()),
    ))
}

pub fn vk_rmsnorm(x: &VkTensor, weight: &VkTensor, eps: f32) -> Result<VkTensor> {
    let out = vk_rmsnorm_no_grad(x, weight, eps)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if x.requires_grad() || weight.requires_grad() {
        Some(Arc::new(RmsNormBackward {
            eps,
            inputs: [x.clone(), weight.clone()],
        }))
    } else {
        None
    };
    Ok(VkTensor::from_op(
        Arc::clone(out.buffer()),
        out.shape().to_vec(),
        out.dtype(),
        Arc::clone(out.device()),
        grad_fn,
    ))
}

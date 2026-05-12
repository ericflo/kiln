//! SiLU activation for `VkTensor`.
//!
//! `silu(x) = x * sigmoid(x)`. Backward saves `x` (not `y`) — the
//! gradient `dy = sigmoid(x) * (1 + x * (1 - sigmoid(x)))`.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn dispatch_silu_fwd(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    out: &VulkanBuffer,
    n: usize,
) -> Result<()> {
    let workgroups = ((n + 255) / 256) as u32;
    let push = [n as u32];
    dispatch_simple(
        device,
        "vk_silu_f32",
        &[x.handle(), out.handle()],
        &push,
        workgroups,
    )
}

fn dispatch_silu_bwd(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    grad_out: &VulkanBuffer,
    grad_in: &VulkanBuffer,
    n: usize,
) -> Result<()> {
    let workgroups = ((n + 255) / 256) as u32;
    let push = [n as u32];
    dispatch_simple(
        device,
        "vk_silu_bwd_f32",
        &[x.handle(), grad_out.handle(), grad_in.handle()],
        &push,
        workgroups,
    )
}

pub fn vk_silu_no_grad(x: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(x.dtype() == VkDType::F32, "vk_silu: F32 only");
    let n = x.num_elements();
    let out = alloc_f32(x.device(), n)?;
    dispatch_silu_fwd(x.device(), x.buffer(), &out, n)?;
    Ok(VkTensor::from_buffer(
        out,
        x.shape().to_vec(),
        VkDType::F32,
        Arc::clone(x.device()),
    ))
}

#[derive(Debug)]
pub struct SiluBackward {
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for SiluBackward {
    fn op_name(&self) -> &'static str {
        "silu"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let x = &self.inputs[0];
        let n = x.num_elements();
        let grad_buf = alloc_f32(x.device(), n)?;
        dispatch_silu_bwd(x.device(), x.buffer(), grad_out.buffer(), &grad_buf, n)?;
        Ok(vec![Some(VkTensor::from_buffer(
            grad_buf,
            x.shape().to_vec(),
            VkDType::F32,
            Arc::clone(x.device()),
        ))])
    }
}

pub fn vk_silu(x: &VkTensor) -> Result<VkTensor> {
    let out = vk_silu_no_grad(x)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if x.requires_grad() {
        Some(Arc::new(SiluBackward {
            inputs: [x.clone()],
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

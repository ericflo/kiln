//! Sigmoid for VkTensor. y = 1 / (1 + exp(-x)). Backward saves `y`
//! (not `x`) so backward is `dx = grad_out * y * (1 - y)`.

use crate::vk_ops::dispatch_simple;
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
    .context("vk_sigmoid: alloc")?;
    Ok(Arc::new(buf))
}

fn dispatch_fwd(device: &VulkanDevice, x: &VulkanBuffer, out: &VulkanBuffer, n: usize) -> Result<()> {
    let workgroups = ((n + 255) / 256) as u32;
    let push = [n as u32];
    dispatch_simple(
        device,
        "vk_sigmoid_f32",
        &[x.handle(), out.handle()],
        &push,
        workgroups,
    )
}

fn dispatch_bwd(
    device: &VulkanDevice,
    y: &VulkanBuffer,
    grad_out: &VulkanBuffer,
    grad_in: &VulkanBuffer,
    n: usize,
) -> Result<()> {
    let workgroups = ((n + 255) / 256) as u32;
    let push = [n as u32];
    dispatch_simple(
        device,
        "vk_sigmoid_bwd_f32",
        &[y.handle(), grad_out.handle(), grad_in.handle()],
        &push,
        workgroups,
    )
}

pub fn vk_sigmoid_no_grad(x: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(x.dtype() == VkDType::F32, "vk_sigmoid: F32-only");
    let n = x.num_elements();
    let out = alloc_f32(x.device(), n)?;
    dispatch_fwd(x.device(), x.buffer(), &out, n)?;
    Ok(VkTensor::from_buffer(
        out,
        x.shape().to_vec(),
        VkDType::F32,
        Arc::clone(x.device()),
    ))
}

#[derive(Debug)]
pub struct SigmoidBackward {
    pub y: VkTensor,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for SigmoidBackward {
    fn op_name(&self) -> &'static str {
        "sigmoid"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let x = &self.inputs[0];
        let n = x.num_elements();
        let grad_buf = alloc_f32(x.device(), n)?;
        dispatch_bwd(x.device(), self.y.buffer(), grad_out.buffer(), &grad_buf, n)?;
        Ok(vec![Some(VkTensor::from_buffer(
            grad_buf,
            x.shape().to_vec(),
            VkDType::F32,
            Arc::clone(x.device()),
        ))])
    }
}

pub fn vk_sigmoid(x: &VkTensor) -> Result<VkTensor> {
    let out = vk_sigmoid_no_grad(x)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if x.requires_grad() {
        Some(Arc::new(SigmoidBackward {
            y: out.clone(),
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

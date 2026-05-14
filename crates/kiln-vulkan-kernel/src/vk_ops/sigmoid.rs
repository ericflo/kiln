//! Sigmoid for VkTensor. y = 1 / (1 + exp(-x)). Backward saves `y`
//! (not `x`) so backward is `dx = grad_out * y * (1 - y)`.

use crate::vk_ops::{dispatch_simple, for_each_1d_tile, vk_exp_tile_elements};
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::Result;
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn dispatch_fwd(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    out: &VulkanBuffer,
    n: usize,
) -> Result<()> {
    let tile_elements = vk_exp_tile_elements();
    if n <= tile_elements {
        let workgroups = ((n + 255) / 256) as u32;
        let push = [n as u32];
        return dispatch_simple(
            device,
            "vk_sigmoid_f32",
            &[x.handle(), out.handle()],
            &push,
            workgroups,
        );
    }
    for_each_1d_tile(n, tile_elements, |offset, len| {
        let workgroups = ((len + 255) / 256) as u32;
        let push = [len as u32, offset as u32];
        dispatch_simple(
            device,
            "vk_sigmoid_f32_offset",
            &[x.handle(), out.handle()],
            &push,
            workgroups,
        )
    })
}

fn dispatch_bwd(
    device: &VulkanDevice,
    y: &VulkanBuffer,
    grad_out: &VulkanBuffer,
    grad_in: &VulkanBuffer,
    n: usize,
) -> Result<()> {
    let tile_elements = vk_exp_tile_elements();
    if n <= tile_elements {
        let workgroups = ((n + 255) / 256) as u32;
        let push = [n as u32];
        return dispatch_simple(
            device,
            "vk_sigmoid_bwd_f32",
            &[y.handle(), grad_out.handle(), grad_in.handle()],
            &push,
            workgroups,
        );
    }
    for_each_1d_tile(n, tile_elements, |offset, len| {
        let workgroups = ((len + 255) / 256) as u32;
        let push = [len as u32, offset as u32];
        dispatch_simple(
            device,
            "vk_sigmoid_bwd_f32_offset",
            &[y.handle(), grad_out.handle(), grad_in.handle()],
            &push,
            workgroups,
        )
    })
}

fn dispatch_mul_sigmoid_gate_fwd(
    device: &VulkanDevice,
    a: &VulkanBuffer,
    gate: &VulkanBuffer,
    out: &VulkanBuffer,
    n: usize,
) -> Result<()> {
    let tile_elements = vk_exp_tile_elements();
    if n <= tile_elements {
        let workgroups = ((n + 255) / 256) as u32;
        let push = [n as u32];
        return dispatch_simple(
            device,
            "vk_mul_sigmoid_gate_f32",
            &[a.handle(), gate.handle(), out.handle()],
            &push,
            workgroups,
        );
    }
    for_each_1d_tile(n, tile_elements, |offset, len| {
        let workgroups = ((len + 255) / 256) as u32;
        let push = [len as u32, offset as u32];
        dispatch_simple(
            device,
            "vk_mul_sigmoid_gate_f32_offset",
            &[a.handle(), gate.handle(), out.handle()],
            &push,
            workgroups,
        )
    })
}

fn dispatch_mul_sigmoid_gate_bwd(
    device: &VulkanDevice,
    a: &VulkanBuffer,
    gate: &VulkanBuffer,
    grad_out: &VulkanBuffer,
    grad_a: &VulkanBuffer,
    grad_gate: &VulkanBuffer,
    n: usize,
) -> Result<()> {
    let tile_elements = vk_exp_tile_elements();
    if n <= tile_elements {
        let workgroups = ((n + 255) / 256) as u32;
        let push = [n as u32];
        return dispatch_simple(
            device,
            "vk_mul_sigmoid_gate_bwd_f32",
            &[
                a.handle(),
                gate.handle(),
                grad_out.handle(),
                grad_a.handle(),
                grad_gate.handle(),
            ],
            &push,
            workgroups,
        );
    }
    for_each_1d_tile(n, tile_elements, |offset, len| {
        let workgroups = ((len + 255) / 256) as u32;
        let push = [len as u32, offset as u32];
        dispatch_simple(
            device,
            "vk_mul_sigmoid_gate_bwd_f32_offset",
            &[
                a.handle(),
                gate.handle(),
                grad_out.handle(),
                grad_a.handle(),
                grad_gate.handle(),
            ],
            &push,
            workgroups,
        )
    })
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

#[derive(Debug)]
pub struct MulSigmoidGateBackward {
    pub inputs: [VkTensor; 2],
}

impl VkBackwardOp for MulSigmoidGateBackward {
    fn op_name(&self) -> &'static str {
        "mul_sigmoid_gate"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let a = &self.inputs[0];
        let gate = &self.inputs[1];
        let n = a.num_elements();
        let grad_a = alloc_f32(a.device(), n)?;
        let grad_gate = alloc_f32(a.device(), n)?;
        dispatch_mul_sigmoid_gate_bwd(
            a.device(),
            a.buffer(),
            gate.buffer(),
            grad_out.buffer(),
            &grad_a,
            &grad_gate,
            n,
        )?;
        Ok(vec![
            Some(VkTensor::from_buffer(
                grad_a,
                a.shape().to_vec(),
                VkDType::F32,
                Arc::clone(a.device()),
            )),
            Some(VkTensor::from_buffer(
                grad_gate,
                gate.shape().to_vec(),
                VkDType::F32,
                Arc::clone(gate.device()),
            )),
        ])
    }
}

pub fn vk_mul_sigmoid_gate(a: &VkTensor, gate: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(
        a.dtype() == VkDType::F32 && gate.dtype() == VkDType::F32,
        "vk_mul_sigmoid_gate: F32-only"
    );
    anyhow::ensure!(
        a.shape() == gate.shape(),
        "vk_mul_sigmoid_gate: shape mismatch {:?} vs {:?}",
        a.shape(),
        gate.shape()
    );
    let n = a.num_elements();
    let out = alloc_f32(a.device(), n)?;
    dispatch_mul_sigmoid_gate_fwd(a.device(), a.buffer(), gate.buffer(), &out, n)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if a.requires_grad() || gate.requires_grad() {
        Some(Arc::new(MulSigmoidGateBackward {
            inputs: [a.clone(), gate.clone()],
        }))
    } else {
        None
    };
    Ok(VkTensor::from_op(
        out,
        a.shape().to_vec(),
        VkDType::F32,
        Arc::clone(a.device()),
        grad_fn,
    ))
}

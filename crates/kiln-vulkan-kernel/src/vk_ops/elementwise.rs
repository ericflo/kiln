//! Element-wise binary ops (add/sub/mul/div) for `VkTensor`.
//!
//! Same-shape only — broadcasting is staged as an explicit op (Phase A.6).
//! F32 only for Phase A; BF16 variants come with the cast op (Phase A.5).
//!
//! Forward dispatches `vk_elementwise_binary_f32.comp` with an op_code
//! push constant.

use crate::vk_ops::dispatch_simple;
use crate::vk_ops::reduce::vk_neg_no_grad;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_DIV: u32 = 3;

fn dispatch_binary(
    device: &VulkanDevice,
    a: &VulkanBuffer,
    b: &VulkanBuffer,
    out: &VulkanBuffer,
    n_elements: usize,
    op_code: u32,
) -> Result<()> {
    anyhow::ensure!(
        n_elements > 0,
        "vk_elementwise_binary: n_elements must be > 0"
    );
    let workgroups = ((n_elements + 255) / 256) as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_elementwise_binary: workgroups {workgroups} > device limit {limit}"
    );
    let push_constants: [u32; 2] = [n_elements as u32, op_code];
    dispatch_simple(
        device,
        "vk_elementwise_binary_f32",
        &[a.handle(), b.handle(), out.handle()],
        &push_constants,
        workgroups,
    )
}

fn alloc_like(a: &VkTensor) -> Result<Arc<VulkanBuffer>> {
    let dev = a.device();
    let bytes = (a.num_elements() * a.dtype().byte_size()).max(1);
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        bytes as u64,
    )
    .context("vk_elementwise: alloc output buffer")?;
    Ok(Arc::new(buf))
}

fn check_same_shape_f32(a: &VkTensor, b: &VkTensor) -> Result<()> {
    anyhow::ensure!(
        a.shape() == b.shape(),
        "vk_elementwise: shape mismatch {:?} vs {:?}",
        a.shape(),
        b.shape()
    );
    anyhow::ensure!(
        a.dtype() == VkDType::F32 && b.dtype() == VkDType::F32,
        "vk_elementwise: Phase A supports F32 only (got {:?} / {:?})",
        a.dtype(),
        b.dtype()
    );
    Ok(())
}

// ---- backward op structs ----

#[derive(Debug)]
pub struct AddBackward {
    inputs: [VkTensor; 2],
}

impl VkBackwardOp for AddBackward {
    fn op_name(&self) -> &'static str {
        "add"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // d(a + b)/da = 1, d(a + b)/db = 1
        Ok(vec![Some(grad_out.clone()), Some(grad_out.clone())])
    }
}

#[derive(Debug)]
pub struct SubBackward {
    inputs: [VkTensor; 2],
}

impl VkBackwardOp for SubBackward {
    fn op_name(&self) -> &'static str {
        "sub"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // d(a - b)/da = 1, d(a - b)/db = -1
        let neg = vk_neg_no_grad(grad_out)?;
        Ok(vec![Some(grad_out.clone()), Some(neg)])
    }
}

#[derive(Debug)]
pub struct MulBackward {
    inputs: [VkTensor; 2],
}

impl VkBackwardOp for MulBackward {
    fn op_name(&self) -> &'static str {
        "mul"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // d(a * b)/da = b, d(a * b)/db = a
        let grad_a = vk_mul_no_grad(grad_out, &self.inputs[1])?;
        let grad_b = vk_mul_no_grad(grad_out, &self.inputs[0])?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
}

#[derive(Debug)]
pub struct DivBackward {
    inputs: [VkTensor; 2],
}

impl VkBackwardOp for DivBackward {
    fn op_name(&self) -> &'static str {
        "div"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // d(a / b)/da = 1/b ; d(a / b)/db = -a / b^2
        let a = &self.inputs[0];
        let b = &self.inputs[1];
        let grad_a = vk_div_no_grad(grad_out, b)?;
        let b_sq = vk_mul_no_grad(b, b)?;
        let neg_a = vk_neg_no_grad(a)?;
        let factor = vk_div_no_grad(&neg_a, &b_sq)?;
        let grad_b = vk_mul_no_grad(grad_out, &factor)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
}

// ---- forward ops (with autograd) ----

fn elementwise_forward_with_grad<B: VkBackwardOp + 'static>(
    a: &VkTensor,
    b: &VkTensor,
    op_code: u32,
    make_grad_fn: impl FnOnce() -> B,
) -> Result<VkTensor> {
    check_same_shape_f32(a, b)?;
    let out_buf = alloc_like(a)?;
    dispatch_binary(
        a.device(),
        a.buffer(),
        b.buffer(),
        &out_buf,
        a.num_elements(),
        op_code,
    )?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> =
        if a.requires_grad() || b.requires_grad() {
            Some(Arc::new(make_grad_fn()))
        } else {
            None
        };
    Ok(VkTensor::from_op(
        out_buf,
        a.shape().to_vec(),
        a.dtype(),
        Arc::clone(a.device()),
        grad_fn,
    ))
}

pub fn vk_add(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    elementwise_forward_with_grad(a, b, OP_ADD, || AddBackward {
        inputs: [a.clone(), b.clone()],
    })
}

pub fn vk_sub(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    elementwise_forward_with_grad(a, b, OP_SUB, || SubBackward {
        inputs: [a.clone(), b.clone()],
    })
}

pub fn vk_mul(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    elementwise_forward_with_grad(a, b, OP_MUL, || MulBackward {
        inputs: [a.clone(), b.clone()],
    })
}

pub fn vk_div(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    elementwise_forward_with_grad(a, b, OP_DIV, || DivBackward {
        inputs: [a.clone(), b.clone()],
    })
}

// ---- no-grad variants for autograd-internal use ----

pub fn vk_add_no_grad(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    check_same_shape_f32(a, b)?;
    let out_buf = alloc_like(a)?;
    dispatch_binary(
        a.device(),
        a.buffer(),
        b.buffer(),
        &out_buf,
        a.num_elements(),
        OP_ADD,
    )?;
    Ok(VkTensor::from_buffer(
        out_buf,
        a.shape().to_vec(),
        a.dtype(),
        Arc::clone(a.device()),
    ))
}

pub fn vk_mul_no_grad(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    check_same_shape_f32(a, b)?;
    let out_buf = alloc_like(a)?;
    dispatch_binary(
        a.device(),
        a.buffer(),
        b.buffer(),
        &out_buf,
        a.num_elements(),
        OP_MUL,
    )?;
    Ok(VkTensor::from_buffer(
        out_buf,
        a.shape().to_vec(),
        a.dtype(),
        Arc::clone(a.device()),
    ))
}

pub fn vk_sub_no_grad(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    check_same_shape_f32(a, b)?;
    let out_buf = alloc_like(a)?;
    dispatch_binary(
        a.device(),
        a.buffer(),
        b.buffer(),
        &out_buf,
        a.num_elements(),
        OP_SUB,
    )?;
    Ok(VkTensor::from_buffer(
        out_buf,
        a.shape().to_vec(),
        a.dtype(),
        Arc::clone(a.device()),
    ))
}

pub fn vk_div_no_grad(a: &VkTensor, b: &VkTensor) -> Result<VkTensor> {
    check_same_shape_f32(a, b)?;
    let out_buf = alloc_like(a)?;
    dispatch_binary(
        a.device(),
        a.buffer(),
        b.buffer(),
        &out_buf,
        a.num_elements(),
        OP_DIV,
    )?;
    Ok(VkTensor::from_buffer(
        out_buf,
        a.shape().to_vec(),
        a.dtype(),
        Arc::clone(a.device()),
    ))
}

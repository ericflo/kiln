//! RoPE (rotary position embedding) for `VkTensor`.
//!
//! Input shape `[rows, num_heads, head_dim]`. The first `rotary_dim`
//! elements of each head are rotated by the precomputed cos/sin
//! tables (shape `[rows, rotary_dim/2]`); the tail copies through.
//!
//! Backward is the inverse rotation (= rotation by -theta), applied
//! by the same kernel structure with opposite signs.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn check_rope_shape(
    x: &VkTensor,
    cos: &VkTensor,
    sin: &VkTensor,
    rotary_dim: usize,
) -> Result<(usize, usize, usize)> {
    anyhow::ensure!(x.dtype() == VkDType::F32, "vk_rope: F32-only");
    anyhow::ensure!(cos.dtype() == VkDType::F32, "vk_rope cos: F32-only");
    anyhow::ensure!(sin.dtype() == VkDType::F32, "vk_rope sin: F32-only");
    anyhow::ensure!(
        x.shape().len() == 3,
        "vk_rope: input must be rank-3 [rows, heads, head_dim] (got {:?})",
        x.shape()
    );
    let rows = x.shape()[0];
    let heads = x.shape()[1];
    let head_dim = x.shape()[2];
    anyhow::ensure!(
        rotary_dim <= head_dim && rotary_dim % 2 == 0,
        "vk_rope: rotary_dim={rotary_dim} must be <= head_dim={head_dim} and even"
    );
    let half = rotary_dim / 2;
    anyhow::ensure!(
        cos.shape() == [rows, half],
        "vk_rope: cos shape {:?} != [{rows}, {half}]",
        cos.shape()
    );
    anyhow::ensure!(
        sin.shape() == [rows, half],
        "vk_rope: sin shape {:?} != [{rows}, {half}]",
        sin.shape()
    );
    Ok((rows, heads, head_dim))
}

fn dispatch_rope(
    device: &VulkanDevice,
    shader: &str,
    x: &VulkanBuffer,
    cos: &VulkanBuffer,
    sin: &VulkanBuffer,
    out: &VulkanBuffer,
    rows: usize,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<()> {
    let total = rows * heads * head_dim;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [
        rows as u32,
        heads as u32,
        head_dim as u32,
        rotary_dim as u32,
    ];
    dispatch_simple(
        device,
        shader,
        &[x.handle(), cos.handle(), sin.handle(), out.handle()],
        &push,
        workgroups,
    )
}

pub fn vk_rope_no_grad(
    x: &VkTensor,
    cos: &VkTensor,
    sin: &VkTensor,
    rotary_dim: usize,
) -> Result<VkTensor> {
    let (rows, heads, head_dim) = check_rope_shape(x, cos, sin, rotary_dim)?;
    let out = alloc_f32(x.device(), rows * heads * head_dim)?;
    dispatch_rope(
        x.device(),
        "vk_rope_f32",
        x.buffer(),
        cos.buffer(),
        sin.buffer(),
        &out,
        rows,
        heads,
        head_dim,
        rotary_dim,
    )?;
    Ok(VkTensor::from_buffer(
        out,
        x.shape().to_vec(),
        VkDType::F32,
        Arc::clone(x.device()),
    ))
}

#[derive(Debug)]
pub struct RopeBackward {
    pub cos: VkTensor,
    pub sin: VkTensor,
    pub rotary_dim: usize,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for RopeBackward {
    fn op_name(&self) -> &'static str {
        "rope"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let x = &self.inputs[0];
        let (rows, heads, head_dim) = check_rope_shape(x, &self.cos, &self.sin, self.rotary_dim)?;
        let grad_buf = alloc_f32(x.device(), rows * heads * head_dim)?;
        dispatch_rope(
            x.device(),
            "vk_rope_bwd_f32",
            grad_out.buffer(),
            self.cos.buffer(),
            self.sin.buffer(),
            &grad_buf,
            rows,
            heads,
            head_dim,
            self.rotary_dim,
        )?;
        Ok(vec![Some(VkTensor::from_buffer(
            grad_buf,
            x.shape().to_vec(),
            VkDType::F32,
            Arc::clone(x.device()),
        ))])
    }
}

pub fn vk_rope(
    x: &VkTensor,
    cos: &VkTensor,
    sin: &VkTensor,
    rotary_dim: usize,
) -> Result<VkTensor> {
    let out = vk_rope_no_grad(x, cos, sin, rotary_dim)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if x.requires_grad() {
        Some(Arc::new(RopeBackward {
            cos: cos.clone(),
            sin: sin.clone(),
            rotary_dim,
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

//! Last-dimension L2 normalization for GDN Q/K:
//! `y = scale * x / sqrt(sum(x^2) + eps)`.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn check_l2norm_shape(x: &VkTensor) -> Result<(usize, usize)> {
    anyhow::ensure!(x.dtype() == VkDType::F32, "vk_l2_norm: F32-only");
    let dims = x.shape();
    anyhow::ensure!(!dims.is_empty(), "vk_l2_norm: x must have rank >= 1");
    let hidden = *dims.last().unwrap();
    anyhow::ensure!(
        hidden > 0 && hidden <= 256,
        "vk_l2_norm: hidden dim {hidden} exceeds shader cap 256"
    );
    let rows = dims[..dims.len() - 1].iter().product::<usize>().max(1);
    Ok((rows, hidden))
}

fn dispatch_l2norm_forward(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    out: &VulkanBuffer,
    rows: usize,
    hidden: usize,
    scale: f32,
    eps: f32,
) -> Result<()> {
    let workgroups = rows as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_l2_norm_forward: rows {workgroups} > device limit {limit}"
    );
    let push = [rows as u32, hidden as u32, scale.to_bits(), eps.to_bits()];
    dispatch_simple(
        device,
        "vk_l2_norm_lastdim_f32",
        &[x.handle(), out.handle()],
        &push,
        workgroups,
    )
}

fn dispatch_l2norm_backward(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    grad_out: &VulkanBuffer,
    grad_x: &VulkanBuffer,
    rows: usize,
    hidden: usize,
    scale: f32,
    eps: f32,
) -> Result<()> {
    let workgroups = rows as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_l2_norm_backward: rows {workgroups} > device limit {limit}"
    );
    let push = [rows as u32, hidden as u32, scale.to_bits(), eps.to_bits()];
    dispatch_simple(
        device,
        "vk_l2_norm_lastdim_bwd_f32",
        &[x.handle(), grad_out.handle(), grad_x.handle()],
        &push,
        workgroups,
    )
}

#[derive(Debug)]
struct L2NormBackward {
    scale: f32,
    eps: f32,
    inputs: [VkTensor; 1],
}

impl VkBackwardOp for L2NormBackward {
    fn op_name(&self) -> &'static str {
        "l2_norm_lastdim"
    }

    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let x = &self.inputs[0];
        let (rows, hidden) = check_l2norm_shape(x)?;
        anyhow::ensure!(
            grad_out.shape() == x.shape(),
            "vk_l2_norm backward: grad shape mismatch {:?} vs {:?}",
            grad_out.shape(),
            x.shape()
        );
        let grad_x_buf = alloc_f32(x.device(), x.num_elements())?;
        dispatch_l2norm_backward(
            x.device(),
            x.buffer(),
            grad_out.buffer(),
            &grad_x_buf,
            rows,
            hidden,
            self.scale,
            self.eps,
        )
        .context("vk_l2_norm backward dispatch")?;
        Ok(vec![Some(VkTensor::from_buffer(
            grad_x_buf,
            x.shape().to_vec(),
            VkDType::F32,
            Arc::clone(x.device()),
        ))])
    }
}

pub fn vk_l2_norm_lastdim_no_grad(x: &VkTensor, scale: f32, eps: f32) -> Result<VkTensor> {
    let (rows, hidden) = check_l2norm_shape(x)?;
    let out = alloc_f32(x.device(), x.num_elements())?;
    dispatch_l2norm_forward(x.device(), x.buffer(), &out, rows, hidden, scale, eps)
        .context("vk_l2_norm forward dispatch")?;
    Ok(VkTensor::from_buffer(
        out,
        x.shape().to_vec(),
        VkDType::F32,
        Arc::clone(x.device()),
    ))
}

pub fn vk_l2_norm_lastdim(x: &VkTensor, scale: f32, eps: f32) -> Result<VkTensor> {
    let out = vk_l2_norm_lastdim_no_grad(x, scale, eps)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if x.requires_grad() {
        Some(Arc::new(L2NormBackward {
            scale,
            eps,
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

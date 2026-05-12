//! Autograd-aware permutes for the attention block.
//!
//! - `vk_permute_rh_to_hr(t)`: `[rows, heads, head_dim] → [heads, rows, head_dim]`
//! - `vk_permute_hr_to_rh(t)`: `[heads, rows, head_dim] → [rows, heads, head_dim]`
//! - `vk_repeat_kv_heads(t, groups)`: `[heads_kv, rows, head_dim] →
//!   [heads_kv * groups, rows, head_dim]` (broadcasts each KV head
//!   to multiple Q heads for GQA). Backward sums groups together.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn dispatch_three_dim(
    device: &VulkanDevice,
    shader: &str,
    src: &VulkanBuffer,
    dst: &VulkanBuffer,
    dim0: usize,
    dim1: usize,
    dim2: usize,
) -> Result<()> {
    let total = dim0 * dim1 * dim2;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [dim0 as u32, dim1 as u32, dim2 as u32];
    dispatch_simple(
        device,
        shader,
        &[src.handle(), dst.handle()],
        &push,
        workgroups,
    )
}

// ---- [rows, heads, head_dim] ↔ [heads, rows, head_dim] ----

pub fn vk_permute_rh_to_hr_no_grad(t: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(t.dtype() == VkDType::F32, "vk_permute: F32-only");
    anyhow::ensure!(t.shape().len() == 3, "vk_permute_rh_to_hr: rank-3 required");
    let rows = t.shape()[0];
    let heads = t.shape()[1];
    let head_dim = t.shape()[2];
    let out = alloc_f32(t.device(), rows * heads * head_dim)?;
    dispatch_three_dim(
        t.device(),
        "vk_permute_rh_to_hr_f32",
        t.buffer(),
        &out,
        rows,
        heads,
        head_dim,
    )?;
    Ok(VkTensor::from_buffer(
        out,
        vec![heads, rows, head_dim],
        VkDType::F32,
        Arc::clone(t.device()),
    ))
}

pub fn vk_permute_hr_to_rh_no_grad(t: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(t.dtype() == VkDType::F32, "vk_permute: F32-only");
    anyhow::ensure!(t.shape().len() == 3, "vk_permute_hr_to_rh: rank-3 required");
    let heads = t.shape()[0];
    let rows = t.shape()[1];
    let head_dim = t.shape()[2];
    let out = alloc_f32(t.device(), heads * rows * head_dim)?;
    dispatch_three_dim(
        t.device(),
        "vk_permute_hr_to_rh_f32",
        t.buffer(),
        &out,
        heads,
        rows,
        head_dim,
    )?;
    Ok(VkTensor::from_buffer(
        out,
        vec![rows, heads, head_dim],
        VkDType::F32,
        Arc::clone(t.device()),
    ))
}

#[derive(Debug)]
pub struct PermuteRhToHrBackward {
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for PermuteRhToHrBackward {
    fn op_name(&self) -> &'static str {
        "permute_rh_to_hr"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        Ok(vec![Some(vk_permute_hr_to_rh_no_grad(grad_out)?)])
    }
}

pub fn vk_permute_rh_to_hr(t: &VkTensor) -> Result<VkTensor> {
    let out = vk_permute_rh_to_hr_no_grad(t)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(PermuteRhToHrBackward {
            inputs: [t.clone()],
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
pub struct PermuteHrToRhBackward {
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for PermuteHrToRhBackward {
    fn op_name(&self) -> &'static str {
        "permute_hr_to_rh"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        Ok(vec![Some(vk_permute_rh_to_hr_no_grad(grad_out)?)])
    }
}

pub fn vk_permute_hr_to_rh(t: &VkTensor) -> Result<VkTensor> {
    let out = vk_permute_hr_to_rh_no_grad(t)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(PermuteHrToRhBackward {
            inputs: [t.clone()],
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

// ---- KV head repeat for GQA ----

fn dispatch_kv_repeat(
    device: &VulkanDevice,
    src: &VulkanBuffer,
    dst: &VulkanBuffer,
    heads_kv: usize,
    groups: usize,
    rows: usize,
    head_dim: usize,
) -> Result<()> {
    let total = heads_kv * groups * rows * head_dim;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [heads_kv as u32, groups as u32, rows as u32, head_dim as u32];
    dispatch_simple(
        device,
        "vk_repeat_kv_heads_f32",
        &[src.handle(), dst.handle()],
        &push,
        workgroups,
    )
}

fn dispatch_kv_sum(
    device: &VulkanDevice,
    src: &VulkanBuffer,
    dst: &VulkanBuffer,
    heads_kv: usize,
    groups: usize,
    rows: usize,
    head_dim: usize,
) -> Result<()> {
    let total = heads_kv * rows * head_dim;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [heads_kv as u32, groups as u32, rows as u32, head_dim as u32];
    dispatch_simple(
        device,
        "vk_sum_kv_groups_f32",
        &[src.handle(), dst.handle()],
        &push,
        workgroups,
    )
}

pub fn vk_repeat_kv_heads_no_grad(t: &VkTensor, groups: usize) -> Result<VkTensor> {
    anyhow::ensure!(t.dtype() == VkDType::F32, "vk_repeat_kv_heads: F32-only");
    anyhow::ensure!(
        t.shape().len() == 3,
        "vk_repeat_kv_heads: rank-3 required"
    );
    anyhow::ensure!(groups >= 1, "vk_repeat_kv_heads: groups >= 1");
    let heads_kv = t.shape()[0];
    let rows = t.shape()[1];
    let head_dim = t.shape()[2];
    if groups == 1 {
        return Ok(t.clone());
    }
    let out = alloc_f32(t.device(), heads_kv * groups * rows * head_dim)?;
    dispatch_kv_repeat(t.device(), t.buffer(), &out, heads_kv, groups, rows, head_dim)?;
    Ok(VkTensor::from_buffer(
        out,
        vec![heads_kv * groups, rows, head_dim],
        VkDType::F32,
        Arc::clone(t.device()),
    ))
}

#[derive(Debug)]
pub struct RepeatKvHeadsBackward {
    pub heads_kv: usize,
    pub groups: usize,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for RepeatKvHeadsBackward {
    fn op_name(&self) -> &'static str {
        "repeat_kv_heads"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        if self.groups == 1 {
            return Ok(vec![Some(grad_out.clone())]);
        }
        let t = &self.inputs[0];
        let rows = t.shape()[1];
        let head_dim = t.shape()[2];
        let grad_buf = alloc_f32(t.device(), self.heads_kv * rows * head_dim)?;
        dispatch_kv_sum(
            t.device(),
            grad_out.buffer(),
            &grad_buf,
            self.heads_kv,
            self.groups,
            rows,
            head_dim,
        )?;
        Ok(vec![Some(VkTensor::from_buffer(
            grad_buf,
            vec![self.heads_kv, rows, head_dim],
            VkDType::F32,
            Arc::clone(t.device()),
        ))])
    }
}

pub fn vk_repeat_kv_heads(t: &VkTensor, groups: usize) -> Result<VkTensor> {
    let out = vk_repeat_kv_heads_no_grad(t, groups)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() && groups > 1 {
        Some(Arc::new(RepeatKvHeadsBackward {
            heads_kv: t.shape()[0],
            groups,
            inputs: [t.clone()],
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

//! Reverse cumulative sum along the last axis.
//!
//! `out[..., t] = Σ_{s ≥ t} in[..., s]`. Used in the backward of
//! cumsum (which appears in chunk_prep's G[t] = cumsum(g)[t]).

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

/// Reverse-cumsum the last axis of `t`.
pub fn vk_reverse_cumsum_no_grad(t: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(t.dtype() == VkDType::F32, "vk_reverse_cumsum: F32 only");
    let dims = t.shape();
    let cols = *dims.last().context("empty shape")?;
    let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
    let device = t.device();
    let out = alloc_f32(device, rows * cols)?;

    let workgroups = rows as u32;
    let push = [rows as u32, cols as u32];
    dispatch_simple(
        device,
        "vk_reverse_cumsum",
        &[t.buffer().handle(), out.handle()],
        &push,
        workgroups,
    )?;

    Ok(VkTensor::from_buffer(
        out,
        dims.to_vec(),
        VkDType::F32,
        Arc::clone(device),
    ))
}

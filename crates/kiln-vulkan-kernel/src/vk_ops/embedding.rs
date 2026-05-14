//! Embedding lookup for VkTensor.
//!
//! Token-id gather from a `[vocab, hidden]` weight table. Phase E
//! variant operates on F32 weights; a BF16-packed variant is also
//! provided for compatibility with the existing weight upload path.
//!
//! Backward (scatter-add into the weight table) is gated behind a
//! `requires_grad` check on the weight; for SFT/LoRA training the
//! base embed_tokens is frozen, so we typically return `None` for
//! the weight gradient.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::Result;
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

/// Upload a `Vec<u32>` token-id buffer to GPU. Returns a leaf VkTensor
/// with `dtype = F32` *as a placeholder* — the buffer actually holds u32
/// values; we wrap it to reuse the buffer-management surface. Callers
/// must use the returned tensor only as the `ids` argument to
/// `vk_embedding_lookup`. (We use F32 dtype here to keep the dtype enum
/// surface small for Phase E; real u32 dtype support is a Phase H
/// cleanup.)
pub fn upload_u32_ids(device: &Arc<VulkanDevice>, ids: &[u32]) -> Result<VkTensor> {
    let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
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
    Ok(VkTensor::from_buffer(
        Arc::new(buf),
        vec![ids.len()],
        VkDType::F32, // placeholder; data is u32
        Arc::clone(device),
    ))
}

pub fn vk_embedding_lookup_f32(
    weight: &VkTensor,
    ids: &VkTensor,
    vocab: usize,
    hidden: usize,
) -> Result<VkTensor> {
    anyhow::ensure!(
        weight.dtype() == VkDType::F32,
        "vk_embedding_lookup_f32: weight must be F32"
    );
    let num_tokens = ids.num_elements();
    let out = alloc_f32(weight.device(), num_tokens * hidden)?;
    let workgroups = ((num_tokens * hidden + 255) / 256) as u32;
    let push = [num_tokens as u32, hidden as u32, vocab as u32];
    dispatch_simple(
        weight.device(),
        "vk_embedding_lookup_f32",
        &[
            ids.buffer().handle(),
            weight.buffer().handle(),
            out.handle(),
        ],
        &push,
        workgroups,
    )?;
    let out_tensor = VkTensor::from_buffer(
        out,
        vec![num_tokens, hidden],
        VkDType::F32,
        Arc::clone(weight.device()),
    );
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if weight.requires_grad() {
        // Scatter-add bwd not implemented yet — embed_tokens is frozen
        // in our SFT/LoRA flow. If a user requests gradient through
        // embed, we panic so the missing path is obvious.
        anyhow::bail!("vk_embedding_lookup: scatter-add backward not implemented");
    } else {
        None
    };
    Ok(VkTensor::from_op(
        Arc::clone(out_tensor.buffer()),
        out_tensor.shape().to_vec(),
        out_tensor.dtype(),
        Arc::clone(out_tensor.device()),
        grad_fn,
    ))
}

pub fn vk_embedding_lookup_bf16(
    weight: &VkTensor,
    ids: &VkTensor,
    vocab: usize,
    hidden: usize,
) -> Result<VkTensor> {
    anyhow::ensure!(
        weight.dtype() == VkDType::Bf16,
        "vk_embedding_lookup_bf16: weight must be BF16"
    );
    let num_tokens = ids.num_elements();
    let out = alloc_f32(weight.device(), num_tokens * hidden)?;
    let workgroups = ((num_tokens * hidden + 255) / 256) as u32;
    let push = [num_tokens as u32, hidden as u32, vocab as u32];
    dispatch_simple(
        weight.device(),
        "vk_embedding_lookup_bf16w_f32",
        &[
            ids.buffer().handle(),
            weight.buffer().handle(),
            out.handle(),
        ],
        &push,
        workgroups,
    )?;
    Ok(VkTensor::from_buffer(
        out,
        vec![num_tokens, hidden],
        VkDType::F32,
        Arc::clone(weight.device()),
    ))
}

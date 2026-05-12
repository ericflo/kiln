//! Row-gather (index_select on dim 0) for VkTensor with autograd.
//!
//! Used to filter `[T, hidden]` model output to only the
//! label-mask=true rows before FLCE. Backward scatters the
//! gradient back into a zero-padded `[T, hidden]` buffer.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn upload_indices(device: &Arc<VulkanDevice>, indices: &[u32]) -> Result<Arc<VulkanBuffer>> {
    let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
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
    Ok(Arc::new(buf))
}

fn dispatch_fwd(
    device: &VulkanDevice,
    src: &VulkanBuffer,
    indices: &VulkanBuffer,
    dst: &VulkanBuffer,
    n_out: usize,
    dim: usize,
    n_rows_in: usize,
) -> Result<()> {
    let total = n_out * dim;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [n_out as u32, dim as u32, n_rows_in as u32];
    dispatch_simple(
        device,
        "vk_index_select_rows_f32",
        &[src.handle(), indices.handle(), dst.handle()],
        &push,
        workgroups,
    )
}

fn dispatch_bwd(
    device: &VulkanDevice,
    grad_out: &VulkanBuffer,
    indices: &VulkanBuffer,
    grad_in: &VulkanBuffer,
    n_out: usize,
    dim: usize,
    n_rows_in: usize,
) -> Result<()> {
    let total = n_out * dim;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [n_out as u32, dim as u32, n_rows_in as u32];
    dispatch_simple(
        device,
        "vk_index_select_rows_bwd_f32",
        &[grad_out.handle(), indices.handle(), grad_in.handle()],
        &push,
        workgroups,
    )
}

#[derive(Debug)]
pub struct IndexSelectRowsBackward {
    pub indices: Arc<VulkanBuffer>,
    pub n_out: usize,
    pub dim: usize,
    pub n_rows_in: usize,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for IndexSelectRowsBackward {
    fn op_name(&self) -> &'static str {
        "index_select_rows"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let device = self.inputs[0].device();
        let total_in = self.n_rows_in * self.dim;
        let grad_buf = alloc_f32(device, total_in)?;
        // zero-fill first
        let push_zero = [total_in as u32, 0.0_f32.to_bits()];
        dispatch_simple(
            device,
            "vk_fill_f32",
            &[grad_buf.handle()],
            &push_zero,
            ((total_in + 255) / 256) as u32,
        )?;
        dispatch_bwd(
            device,
            grad_out.buffer(),
            &self.indices,
            &grad_buf,
            self.n_out,
            self.dim,
            self.n_rows_in,
        )?;
        Ok(vec![Some(VkTensor::from_buffer(
            grad_buf,
            vec![self.n_rows_in, self.dim],
            VkDType::F32,
            Arc::clone(device),
        ))])
    }
}

/// Gather rows of a `[n_rows_in, dim]` F32 VkTensor by `indices`.
/// Returns `[indices.len(), dim]` F32 with autograd.
pub fn vk_index_select_rows(t: &VkTensor, indices: &[u32]) -> Result<VkTensor> {
    anyhow::ensure!(t.dtype() == VkDType::F32, "vk_index_select_rows: F32-only");
    anyhow::ensure!(
        t.shape().len() == 2,
        "vk_index_select_rows: rank-2 input required"
    );
    let n_rows_in = t.shape()[0];
    let dim = t.shape()[1];
    let n_out = indices.len();
    let device = t.device();

    let indices_buf = upload_indices(device, indices)?;
    let out_buf = alloc_f32(device, n_out * dim)?;
    dispatch_fwd(
        device,
        t.buffer(),
        &indices_buf,
        &out_buf,
        n_out,
        dim,
        n_rows_in,
    )?;

    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(IndexSelectRowsBackward {
            indices: Arc::clone(&indices_buf),
            n_out,
            dim,
            n_rows_in,
            inputs: [t.clone()],
        }))
    } else {
        None
    };

    Ok(VkTensor::from_op(
        out_buf,
        vec![n_out, dim],
        VkDType::F32,
        Arc::clone(device),
        grad_fn,
    ))
}

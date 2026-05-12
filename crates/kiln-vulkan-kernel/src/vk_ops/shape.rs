//! Shape ops: `vk_reshape` (metadata-only), `vk_transpose_2d`
//! (physical move), and `vk_contiguous` (no-op, we are always
//! C-contiguous).
//!
//! Phase A starts with the minimum surface needed for matmul (Phase
//! B): reshape for batching, transpose for `W.T` access. `narrow` and
//! `index_select` are deferred to the phase that introduces them
//! (Phase E for cross-entropy label gather).

use crate::vk_ops::dispatch_simple_2d;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

// ---- reshape (metadata-only) ----

#[derive(Debug)]
pub struct ReshapeBackward {
    pub input_shape: Vec<usize>,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for ReshapeBackward {
    fn op_name(&self) -> &'static str {
        "reshape"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // Inverse reshape: same underlying buffer, original shape.
        let grad_in = VkTensor::from_buffer(
            Arc::clone(grad_out.buffer()),
            self.input_shape.clone(),
            grad_out.dtype(),
            Arc::clone(grad_out.device()),
        );
        Ok(vec![Some(grad_in)])
    }
}

pub fn vk_reshape(t: &VkTensor, new_shape: &[usize]) -> Result<VkTensor> {
    let old_n: usize = t.shape().iter().product();
    let new_n: usize = new_shape.iter().product();
    anyhow::ensure!(
        old_n == new_n,
        "vk_reshape: element count mismatch {:?} -> {:?}",
        t.shape(),
        new_shape
    );
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(ReshapeBackward {
            input_shape: t.shape().to_vec(),
            inputs: [t.clone()],
        }))
    } else {
        None
    };
    Ok(VkTensor::from_op(
        Arc::clone(t.buffer()),
        new_shape.to_vec(),
        t.dtype(),
        Arc::clone(t.device()),
        grad_fn,
    ))
}

// ---- contiguous (no-op for now) ----

/// We currently maintain C-contiguous storage invariantly. `vk_contiguous`
/// is a no-op clone for API symmetry with candle; revisit when we add
/// strided views.
pub fn vk_contiguous(t: &VkTensor) -> Result<VkTensor> {
    Ok(t.clone())
}

// ---- transpose 2D (physical move) ----

fn dispatch_transpose_2d_f32(
    device: &VulkanDevice,
    src: &VulkanBuffer,
    dst: &VulkanBuffer,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let wg_x = ((cols + 15) / 16) as u32;
    let wg_y = ((rows + 15) / 16) as u32;
    let limit_x = device.max_compute_work_group_count(0);
    let limit_y = device.max_compute_work_group_count(1);
    anyhow::ensure!(
        wg_x <= limit_x && wg_y <= limit_y,
        "vk_transpose_2d: workgroups ({wg_x},{wg_y}) > device limits ({limit_x},{limit_y})"
    );
    let push_constants = [rows as u32, cols as u32];
    dispatch_simple_2d(
        device,
        "vk_transpose_2d_f32",
        &[src.handle(), dst.handle()],
        &push_constants,
        (wg_x, wg_y),
    )
}

pub fn vk_transpose_2d_no_grad(t: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(
        t.shape().len() == 2,
        "vk_transpose_2d: rank-2 input required (got {:?})",
        t.shape()
    );
    anyhow::ensure!(
        t.dtype() == VkDType::F32,
        "vk_transpose_2d: Phase A is F32-only"
    );
    let rows = t.shape()[0];
    let cols = t.shape()[1];
    let n = rows * cols;
    let dev = t.device();
    let bytes = (n * 4).max(4);
    let buf =
        VulkanBuffer::create_device_local(dev.device(), dev.device_local_mem_type(), bytes as u64)
            .context("vk_transpose_2d: alloc output buffer")?;
    let buf = Arc::new(buf);
    dispatch_transpose_2d_f32(dev, t.buffer(), &buf, rows, cols)?;
    Ok(VkTensor::from_buffer(
        buf,
        vec![cols, rows],
        VkDType::F32,
        Arc::clone(dev),
    ))
}

#[derive(Debug)]
pub struct Transpose2dBackward {
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for Transpose2dBackward {
    fn op_name(&self) -> &'static str {
        "transpose_2d"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // d/dx (x.T) = grad_out.T
        let grad_in = vk_transpose_2d_no_grad(grad_out)?;
        Ok(vec![Some(grad_in)])
    }
}

pub fn vk_transpose_2d(t: &VkTensor) -> Result<VkTensor> {
    let out = vk_transpose_2d_no_grad(t)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(Transpose2dBackward {
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

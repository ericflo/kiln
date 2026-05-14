//! Softmax along the last dim for `VkTensor`.
//!
//! Numerically stable F32 implementation: max → exp(x - max) → sum →
//! divide. Backward analytic: `dx_i = y_i * (dy_i - sum_j(y_j * dy_j))`.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::Result;
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn check_softmax_shape(x: &VkTensor) -> Result<(usize, usize)> {
    anyhow::ensure!(
        x.dtype() == VkDType::F32,
        "vk_softmax: F32-only (got {:?})",
        x.dtype()
    );
    let dims = x.shape();
    anyhow::ensure!(!dims.is_empty(), "vk_softmax: empty shape");
    let cols = *dims.last().unwrap();
    let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
    Ok((rows, cols))
}

fn dispatch_softmax_fwd(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    out: &VulkanBuffer,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let workgroups = rows as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_softmax_fwd: rows {workgroups} > device limit {limit}"
    );
    let push_constants = [rows as u32, cols as u32];
    dispatch_simple(
        device,
        "vk_softmax_lastdim_f32",
        &[x.handle(), out.handle()],
        &push_constants,
        workgroups,
    )
}

fn dispatch_softmax_bwd(
    device: &VulkanDevice,
    y: &VulkanBuffer,
    grad_out: &VulkanBuffer,
    grad_in: &VulkanBuffer,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let workgroups = rows as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_softmax_bwd: rows {workgroups} > device limit {limit}"
    );
    let push_constants = [rows as u32, cols as u32];
    dispatch_simple(
        device,
        "vk_softmax_lastdim_bwd_f32",
        &[y.handle(), grad_out.handle(), grad_in.handle()],
        &push_constants,
        workgroups,
    )
}

pub fn vk_softmax_lastdim_no_grad(x: &VkTensor) -> Result<VkTensor> {
    let (rows, cols) = check_softmax_shape(x)?;
    let out = alloc_f32(x.device(), rows * cols)?;
    dispatch_softmax_fwd(x.device(), x.buffer(), &out, rows, cols)?;
    Ok(VkTensor::from_buffer(
        out,
        x.shape().to_vec(),
        VkDType::F32,
        Arc::clone(x.device()),
    ))
}

#[derive(Debug)]
pub struct SoftmaxLastDimBackward {
    /// The forward output `y` (= softmax(x)). Required for backward.
    pub y: VkTensor,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for SoftmaxLastDimBackward {
    fn op_name(&self) -> &'static str {
        "softmax_lastdim"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let (rows, cols) = check_softmax_shape(&self.y)?;
        let grad_x_buf = alloc_f32(self.y.device(), rows * cols)?;
        dispatch_softmax_bwd(
            self.y.device(),
            self.y.buffer(),
            grad_out.buffer(),
            &grad_x_buf,
            rows,
            cols,
        )?;
        let grad_x = VkTensor::from_buffer(
            grad_x_buf,
            self.y.shape().to_vec(),
            VkDType::F32,
            Arc::clone(self.y.device()),
        );
        Ok(vec![Some(grad_x)])
    }
}

pub fn vk_softmax_lastdim(x: &VkTensor) -> Result<VkTensor> {
    let out = vk_softmax_lastdim_no_grad(x)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if x.requires_grad() {
        Some(Arc::new(SoftmaxLastDimBackward {
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

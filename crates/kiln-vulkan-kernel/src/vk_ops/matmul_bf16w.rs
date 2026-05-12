//! Frozen-base-weight matmul for VkTensor.
//!
//! Wraps the existing inference kernels
//! `linear_decode_batched_transposed_bf16w` (forward) and
//! `linear_decode_batched_bf16w` (backward dx) so they
//! work on plain `Arc<VulkanBuffer>` activations without going
//! through candle Tensor wrapping.
//!
//! Forward: `out_f32 = x_f32 @ W_bf16.T` with shapes
//!   x:   [batch, hidden]  (F32)
//!   W:   row-major bf16-packed [out_dim, hidden]
//!   out: [batch, out_dim]  (F32)
//!
//! Backward: `dx = dy_f32 @ W_bf16`. Base weight is FROZEN — no dW
//! computed.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn dispatch_fwd(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    weight: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<()> {
    let col_groups = (out_dim + 31) / 32;
    let workgroups = (col_groups * batch) as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_matmul_bf16w fwd: workgroups {workgroups} > limit {limit}"
    );
    let push = [hidden as u32, out_dim as u32, batch as u32];
    dispatch_simple(
        device,
        "linear_decode_batched_transposed_bf16w",
        &[x.handle(), weight.handle(), out.handle()],
        &push,
        workgroups,
    )
}

fn bf16w_row_tile_len() -> usize {
    std::env::var("KILN_VK_BF16W_ROW_TILE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(256)
}

fn dispatch_fwd_rows(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    weight: &VulkanBuffer,
    out: &VulkanBuffer,
    row_offset: usize,
    rows: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<()> {
    let col_groups = (out_dim + 31) / 32;
    let workgroups = (col_groups * rows) as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_matmul_bf16w fwd rows: workgroups {workgroups} > limit {limit}"
    );
    let push = [
        hidden as u32,
        out_dim as u32,
        rows as u32,
        row_offset as u32,
    ];
    dispatch_simple(
        device,
        "vk_matmul_bf16w_fwd_rows",
        &[x.handle(), weight.handle(), out.handle()],
        &push,
        workgroups,
    )
}

fn dispatch_fwd_tiled(
    device: &VulkanDevice,
    x: &VulkanBuffer,
    weight: &VulkanBuffer,
    out: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<()> {
    let col_groups = (out_dim + 31) / 32;
    let limit = device.max_compute_work_group_count(0) as usize;
    let max_rows_by_limit = (limit / col_groups.max(1)).max(1);
    let tile = bf16w_row_tile_len().min(max_rows_by_limit).max(1);
    if batch <= tile {
        return dispatch_fwd(device, x, weight, out, batch, hidden, out_dim);
    }
    for row_offset in (0..batch).step_by(tile) {
        let rows = (batch - row_offset).min(tile);
        dispatch_fwd_rows(device, x, weight, out, row_offset, rows, hidden, out_dim)?;
    }
    Ok(())
}

fn dispatch_bwd(
    device: &VulkanDevice,
    grad_out: &VulkanBuffer,
    weight: &VulkanBuffer,
    grad_in: &VulkanBuffer,
    batch: usize,
    out_dim: usize, // forward's out_dim = bwd's k_dim
    hidden: usize,  // forward's hidden  = bwd's n_dim
) -> Result<()> {
    let col_groups = (hidden + 31) / 32;
    let workgroups = (col_groups * batch) as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_matmul_bf16w bwd: workgroups {workgroups} > limit {limit}"
    );
    let push = [out_dim as u32, hidden as u32, batch as u32];
    dispatch_simple(
        device,
        "linear_decode_batched_bf16w",
        &[grad_out.handle(), weight.handle(), grad_in.handle()],
        &push,
        workgroups,
    )
}

fn dispatch_bwd_rows(
    device: &VulkanDevice,
    grad_out: &VulkanBuffer,
    weight: &VulkanBuffer,
    grad_in: &VulkanBuffer,
    row_offset: usize,
    rows: usize,
    out_dim: usize,
    hidden: usize,
) -> Result<()> {
    let col_groups = (hidden + 31) / 32;
    let workgroups = (col_groups * rows) as u32;
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroups <= limit,
        "vk_matmul_bf16w bwd rows: workgroups {workgroups} > limit {limit}"
    );
    let push = [
        out_dim as u32,
        hidden as u32,
        rows as u32,
        row_offset as u32,
    ];
    dispatch_simple(
        device,
        "vk_matmul_bf16w_bwd_rows",
        &[grad_out.handle(), weight.handle(), grad_in.handle()],
        &push,
        workgroups,
    )
}

fn dispatch_bwd_tiled(
    device: &VulkanDevice,
    grad_out: &VulkanBuffer,
    weight: &VulkanBuffer,
    grad_in: &VulkanBuffer,
    batch: usize,
    out_dim: usize,
    hidden: usize,
) -> Result<()> {
    let col_groups = (hidden + 31) / 32;
    let limit = device.max_compute_work_group_count(0) as usize;
    let max_rows_by_limit = (limit / col_groups.max(1)).max(1);
    let tile = bf16w_row_tile_len().min(max_rows_by_limit).max(1);
    if batch <= tile {
        return dispatch_bwd(device, grad_out, weight, grad_in, batch, out_dim, hidden);
    }
    for row_offset in (0..batch).step_by(tile) {
        let rows = (batch - row_offset).min(tile);
        dispatch_bwd_rows(
            device, grad_out, weight, grad_in, row_offset, rows, out_dim, hidden,
        )?;
    }
    Ok(())
}

#[derive(Debug)]
pub struct MatmulBf16wBackward {
    pub weight: VkTensor, // BF16-packed [out_dim, hidden]
    pub batch: usize,
    pub hidden: usize,
    pub out_dim: usize,
    pub inputs: [VkTensor; 1], // x (F32)
}

impl VkBackwardOp for MatmulBf16wBackward {
    fn op_name(&self) -> &'static str {
        "matmul_bf16w"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let x = &self.inputs[0];
        let grad_in_buf = alloc_f32(x.device(), self.batch * self.hidden)?;
        dispatch_bwd_tiled(
            x.device(),
            grad_out.buffer(),
            self.weight.buffer(),
            &grad_in_buf,
            self.batch,
            self.out_dim,
            self.hidden,
        )?;
        Ok(vec![Some(VkTensor::from_buffer(
            grad_in_buf,
            vec![self.batch, self.hidden],
            VkDType::F32,
            Arc::clone(x.device()),
        ))])
    }
}

/// Forward `out = x @ W.T` where `W` is a frozen BF16-packed weight.
///
/// `x` shape: `[batch, hidden]` F32. `weight` shape: `[out_dim, hidden]`
/// BF16. Returns `[batch, out_dim]` F32.
pub fn vk_matmul_bf16w(x: &VkTensor, weight: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(
        x.dtype() == VkDType::F32,
        "vk_matmul_bf16w: x must be F32 (got {:?})",
        x.dtype()
    );
    anyhow::ensure!(
        weight.dtype() == VkDType::Bf16,
        "vk_matmul_bf16w: weight must be Bf16 (got {:?})",
        weight.dtype()
    );
    anyhow::ensure!(
        x.shape().len() == 2 && weight.shape().len() == 2,
        "vk_matmul_bf16w: rank-2 inputs required"
    );
    let batch = x.shape()[0];
    let hidden = x.shape()[1];
    let out_dim = weight.shape()[0];
    anyhow::ensure!(
        weight.shape()[1] == hidden,
        "vk_matmul_bf16w: weight inner-dim {} != hidden {}",
        weight.shape()[1],
        hidden
    );

    let out = alloc_f32(x.device(), batch * out_dim)?;
    dispatch_fwd_tiled(
        x.device(),
        x.buffer(),
        weight.buffer(),
        &out,
        batch,
        hidden,
        out_dim,
    )?;

    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if x.requires_grad() {
        Some(Arc::new(MatmulBf16wBackward {
            weight: weight.clone(),
            batch,
            hidden,
            out_dim,
            inputs: [x.clone()],
        }))
    } else {
        None
    };

    Ok(VkTensor::from_op(
        out,
        vec![batch, out_dim],
        VkDType::F32,
        Arc::clone(x.device()),
        grad_fn,
    ))
}

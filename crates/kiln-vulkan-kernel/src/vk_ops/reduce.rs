//! Reductions: `vk_sum` (scalar), `vk_mean` (scalar), and the unary
//! helpers `vk_ones_like` / `vk_zeros_like` / `vk_neg_no_grad` /
//! `vk_scale_no_grad` used by autograd.
//!
//! All scalar reductions use a two-pass tree reduction with F32
//! accumulator. The first pass dispatches `n_workgroups` workgroups
//! (each reducing ~256 elements), the second pass reduces the
//! partials to one. For Phase A this is F32-only.

use crate::vk_ops::elementwise::vk_sub_no_grad;
use crate::vk_ops::{dispatch_simple, for_each_1d_tile, vk_1d_tile_elements};
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32_buffer(device: &Arc<VulkanDevice>, n_elements: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n_elements)
}

// ---- fill / scale / neg ----

pub(crate) fn dispatch_fill(
    device: &VulkanDevice,
    out: &VulkanBuffer,
    n_elements: usize,
    value: f32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "vk_fill: n_elements must be > 0");
    let tile_elements = vk_1d_tile_elements();
    if n_elements <= tile_elements {
        let workgroups = ((n_elements + 255) / 256) as u32;
        let limit = device.max_compute_work_group_count(0);
        anyhow::ensure!(
            workgroups <= limit,
            "vk_fill: workgroups {workgroups} > device limit {limit}"
        );
        let push_constants: [u32; 2] = [n_elements as u32, value.to_bits()];
        return dispatch_simple(
            device,
            "vk_fill_f32",
            &[out.handle()],
            &push_constants,
            workgroups,
        );
    }
    for_each_1d_tile(n_elements, tile_elements, |offset, len| {
        let workgroups = ((len + 255) / 256) as u32;
        let push_constants: [u32; 3] = [len as u32, value.to_bits(), offset as u32];
        dispatch_simple(
            device,
            "vk_fill_f32_offset",
            &[out.handle()],
            &push_constants,
            workgroups,
        )
    })
}

pub fn vk_full_like(t: &VkTensor, value: f32) -> Result<VkTensor> {
    anyhow::ensure!(
        t.dtype() == VkDType::F32,
        "vk_full_like: Phase A is F32-only (got {:?})",
        t.dtype()
    );
    let out = alloc_f32_buffer(t.device(), t.num_elements())?;
    dispatch_fill(t.device(), &out, t.num_elements(), value)?;
    Ok(VkTensor::from_buffer(
        out,
        t.shape().to_vec(),
        VkDType::F32,
        Arc::clone(t.device()),
    ))
}

pub fn vk_ones_like(t: &VkTensor) -> Result<VkTensor> {
    vk_full_like(t, 1.0)
}

pub fn vk_zeros_like(t: &VkTensor) -> Result<VkTensor> {
    vk_full_like(t, 0.0)
}

pub fn vk_neg_no_grad(t: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(
        t.dtype() == VkDType::F32,
        "vk_neg_no_grad: Phase A is F32-only"
    );
    let zeros = vk_zeros_like(t)?;
    vk_sub_no_grad(&zeros, t)
}

// ---- broadcast scalar ----

fn dispatch_broadcast_scalar(
    device: &VulkanDevice,
    scalar_in: &VulkanBuffer,
    out: &VulkanBuffer,
    n_elements: usize,
    scale: f32,
) -> Result<()> {
    anyhow::ensure!(
        n_elements > 0,
        "vk_broadcast_scalar: n_elements must be > 0"
    );
    let tile_elements = vk_1d_tile_elements();
    if n_elements <= tile_elements {
        let workgroups = ((n_elements + 255) / 256) as u32;
        let limit = device.max_compute_work_group_count(0);
        anyhow::ensure!(
            workgroups <= limit,
            "vk_broadcast_scalar: workgroups {workgroups} > device limit {limit}"
        );
        let push_constants: [u32; 2] = [n_elements as u32, scale.to_bits()];
        return dispatch_simple(
            device,
            "vk_broadcast_scalar_f32",
            &[scalar_in.handle(), out.handle()],
            &push_constants,
            workgroups,
        );
    }
    for_each_1d_tile(n_elements, tile_elements, |offset, len| {
        let workgroups = ((len + 255) / 256) as u32;
        let push_constants: [u32; 3] = [len as u32, scale.to_bits(), offset as u32];
        dispatch_simple(
            device,
            "vk_broadcast_scalar_f32_offset",
            &[scalar_in.handle(), out.handle()],
            &push_constants,
            workgroups,
        )
    })
}

/// Broadcast a scalar-shaped (1-element) VkTensor to `target_shape`,
/// multiplied by `scale`. Used by `SumBackward` / `MeanBackward`.
pub fn vk_broadcast_scalar_to_no_grad(
    scalar: &VkTensor,
    target_shape: &[usize],
    scale: f32,
) -> Result<VkTensor> {
    anyhow::ensure!(
        scalar.num_elements() == 1,
        "vk_broadcast_scalar: source must be 1-element (got shape {:?})",
        scalar.shape()
    );
    anyhow::ensure!(
        scalar.dtype() == VkDType::F32,
        "vk_broadcast_scalar: F32 only"
    );
    let nelem: usize = target_shape.iter().product();
    let out = alloc_f32_buffer(scalar.device(), nelem)?;
    dispatch_broadcast_scalar(scalar.device(), scalar.buffer(), &out, nelem, scale)?;
    Ok(VkTensor::from_buffer(
        out,
        target_shape.to_vec(),
        VkDType::F32,
        Arc::clone(scalar.device()),
    ))
}

// ---- reduce sum ----

fn dispatch_reduce_sum_pass(
    device: &VulkanDevice,
    src: &VulkanBuffer,
    dst: &VulkanBuffer,
    n_elements: usize,
    n_workgroups: u32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "vk_reduce_sum: n_elements must be > 0");
    let limit = device.max_compute_work_group_count(0);
    anyhow::ensure!(
        n_workgroups <= limit,
        "vk_reduce_sum: workgroups {n_workgroups} > device limit {limit}"
    );
    let push_constants: [u32; 1] = [n_elements as u32];
    dispatch_simple(
        device,
        "vk_reduce_sum_f32",
        &[src.handle(), dst.handle()],
        &push_constants,
        n_workgroups,
    )
}

/// Reduce-sum all elements to a scalar (1-element) F32 VkTensor. No
/// autograd link.
pub fn vk_sum_all_no_grad(t: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(t.dtype() == VkDType::F32, "vk_sum_all: F32 only");
    let n = t.num_elements();
    anyhow::ensure!(n > 0, "vk_sum_all: empty tensor");
    let device = t.device();

    // Choose first-pass workgroup count: at most 1024 partials (caps
    // second pass at one workgroup of 256 reading 4 elem per lane via
    // grid-stride within the same kernel — fine).
    let max_partials: usize = 1024;
    let first_pass_workgroups: usize = max_partials.min(((n + 255) / 256).max(1));

    let partials_buf = alloc_f32_buffer(device, first_pass_workgroups)?;
    dispatch_reduce_sum_pass(
        device,
        t.buffer(),
        &partials_buf,
        n,
        first_pass_workgroups as u32,
    )?;

    if first_pass_workgroups == 1 {
        return Ok(VkTensor::from_buffer(
            partials_buf,
            vec![1],
            VkDType::F32,
            Arc::clone(device),
        ));
    }

    // Second pass: one workgroup reduces the partials.
    let final_buf = alloc_f32_buffer(device, 1)?;
    dispatch_reduce_sum_pass(device, &partials_buf, &final_buf, first_pass_workgroups, 1)?;
    Ok(VkTensor::from_buffer(
        final_buf,
        vec![1],
        VkDType::F32,
        Arc::clone(device),
    ))
}

// ---- autograd: sum + mean ----

#[derive(Debug)]
pub struct SumAllBackward {
    pub input_shape: Vec<usize>,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for SumAllBackward {
    fn op_name(&self) -> &'static str {
        "sum_all"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // d sum(x) / dx_i = 1; broadcast grad scalar to input shape.
        let grad_in = vk_broadcast_scalar_to_no_grad(grad_out, &self.input_shape, 1.0)?;
        Ok(vec![Some(grad_in)])
    }
}

#[derive(Debug)]
pub struct MeanAllBackward {
    pub input_shape: Vec<usize>,
    pub inv_n: f32,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for MeanAllBackward {
    fn op_name(&self) -> &'static str {
        "mean_all"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // d mean(x) / dx_i = 1/n; broadcast scaled grad to input shape.
        let grad_in = vk_broadcast_scalar_to_no_grad(grad_out, &self.input_shape, self.inv_n)?;
        Ok(vec![Some(grad_in)])
    }
}

pub fn vk_sum_all(t: &VkTensor) -> Result<VkTensor> {
    let out = vk_sum_all_no_grad(t)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(SumAllBackward {
            input_shape: t.shape().to_vec(),
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

pub fn vk_mean_all(t: &VkTensor) -> Result<VkTensor> {
    let n = t.num_elements();
    anyhow::ensure!(n > 0, "vk_mean_all: empty tensor");
    let inv_n = 1.0 / (n as f32);
    let summed = vk_sum_all_no_grad(t)?;
    // mean = sum * (1/n); reuse broadcast_scalar with scale=inv_n,
    // target_shape=[1] (effectively a 1-elem scale).
    let mean = vk_broadcast_scalar_to_no_grad(&summed, &[1], inv_n)?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(MeanAllBackward {
            input_shape: t.shape().to_vec(),
            inv_n,
            inputs: [t.clone()],
        }))
    } else {
        None
    };
    Ok(VkTensor::from_op(
        Arc::clone(mean.buffer()),
        vec![1],
        VkDType::F32,
        Arc::clone(t.device()),
        grad_fn,
    ))
}

//! Narrow (slice) along the last dim of a VkTensor with autograd.
//!
//! `vk_narrow_lastdim(t, start, len)` returns a fresh `[..., len]`
//! tensor whose data is `t[..., start..start+len]`. Backward
//! scatters the grad back into a zero-padded full-shape buffer.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn outer_inner(t: &VkTensor) -> (usize, usize) {
    let dims = t.shape();
    let inner = *dims.last().unwrap_or(&1);
    let outer: usize = dims[..dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>()
        .max(1);
    (outer, inner)
}

fn dispatch_fwd(
    device: &VulkanDevice,
    src: &VulkanBuffer,
    dst: &VulkanBuffer,
    outer: usize,
    inner_in: usize,
    start: usize,
    len: usize,
) -> Result<()> {
    let total = outer * len;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [outer as u32, inner_in as u32, start as u32, len as u32];
    dispatch_simple(
        device,
        "vk_narrow_lastdim_f32",
        &[src.handle(), dst.handle()],
        &push,
        workgroups,
    )
}

fn dispatch_bwd(
    device: &VulkanDevice,
    grad_out: &VulkanBuffer,
    grad_in: &VulkanBuffer,
    outer: usize,
    inner_in: usize,
    start: usize,
    len: usize,
) -> Result<()> {
    let total = outer * len;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [outer as u32, inner_in as u32, start as u32, len as u32];
    dispatch_simple(
        device,
        "vk_narrow_lastdim_bwd_f32",
        &[grad_out.handle(), grad_in.handle()],
        &push,
        workgroups,
    )
}

pub fn vk_narrow_lastdim_no_grad(t: &VkTensor, start: usize, len: usize) -> Result<VkTensor> {
    anyhow::ensure!(t.dtype() == VkDType::F32, "vk_narrow: F32-only");
    let (outer, inner_in) = outer_inner(t);
    anyhow::ensure!(
        start + len <= inner_in,
        "vk_narrow: slice [{start}, {}) out of bounds inner={inner_in}",
        start + len
    );
    let out_buf = alloc_f32(t.device(), outer * len)?;
    dispatch_fwd(t.device(), t.buffer(), &out_buf, outer, inner_in, start, len)?;
    let mut new_shape = t.shape().to_vec();
    *new_shape.last_mut().unwrap() = len;
    Ok(VkTensor::from_buffer(
        out_buf,
        new_shape,
        VkDType::F32,
        Arc::clone(t.device()),
    ))
}

#[derive(Debug)]
pub struct NarrowLastDimBackward {
    pub start: usize,
    pub len: usize,
    pub inner_in: usize,
    pub outer: usize,
    pub input_shape: Vec<usize>,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for NarrowLastDimBackward {
    fn op_name(&self) -> &'static str {
        "narrow_lastdim"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let device = self.inputs[0].device();
        let total_in = self.outer * self.inner_in;
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
            &grad_buf,
            self.outer,
            self.inner_in,
            self.start,
            self.len,
        )?;
        Ok(vec![Some(VkTensor::from_buffer(
            grad_buf,
            self.input_shape.clone(),
            VkDType::F32,
            Arc::clone(device),
        ))])
    }
}

/// Scatter `src` into a contiguous slice along the last dim of `dst`,
/// in place: `dst[..., start..start+len] = src`. The non-slice region
/// is untouched.
///
/// `dst` and `src` must have the same outer dims (the leading `len-1`
/// axes), differ only on the last dim (`dst` last == `inner_in`,
/// `src` last == `len`), and both must be F32.
///
/// This is the same shader the autograd backward of `vk_narrow_lastdim`
/// uses to scatter-grad into a zero-padded buffer; here we expose it
/// for callers (e.g. GDN chunkwise backward) that pre-allocate a
/// full-shape destination once and write per-chunk slices into it on
/// GPU instead of round-tripping through CPU vec splatting.
pub fn vk_scatter_to_lastdim_slice_inplace(
    dst: &VkTensor,
    src: &VkTensor,
    start: usize,
    len: usize,
) -> Result<()> {
    anyhow::ensure!(
        dst.dtype() == VkDType::F32 && src.dtype() == VkDType::F32,
        "vk_scatter_to_lastdim_slice: F32-only"
    );
    let (outer_dst, inner_in) = outer_inner(dst);
    let (outer_src, inner_src) = outer_inner(src);
    anyhow::ensure!(
        outer_src == outer_dst,
        "vk_scatter_to_lastdim_slice: outer mismatch dst={outer_dst} src={outer_src}"
    );
    anyhow::ensure!(
        inner_src == len,
        "vk_scatter_to_lastdim_slice: src last dim {inner_src} != len {len}"
    );
    anyhow::ensure!(
        start + len <= inner_in,
        "vk_scatter_to_lastdim_slice: slice [{start}, {}) out of bounds inner={inner_in}",
        start + len
    );
    dispatch_bwd(
        dst.device(),
        src.buffer(),
        dst.buffer(),
        outer_dst,
        inner_in,
        start,
        len,
    )
}

pub fn vk_narrow_lastdim(t: &VkTensor, start: usize, len: usize) -> Result<VkTensor> {
    let out = vk_narrow_lastdim_no_grad(t, start, len)?;
    let (outer, inner_in) = outer_inner(t);
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(NarrowLastDimBackward {
            start,
            len,
            inner_in,
            outer,
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

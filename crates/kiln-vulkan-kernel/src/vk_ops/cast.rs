//! F32 ↔ BF16 dtype conversion for VkTensors.
//!
//! BF16 storage is a contiguous u16 sequence interpreted by shaders as
//! `u32[]` with 2 BF16 lanes per word. F32 storage is the obvious
//! contiguous 4-byte-per-element layout.
//!
//! For autograd, casts are treated as identity in the gradient
//! direction (cast back to the source dtype). This is the standard
//! mixed-precision convention.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_for(
    device: &Arc<VulkanDevice>,
    n_elements: usize,
    dtype: VkDType,
) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n_elements * dtype.byte_size()).max(dtype.byte_size());
    // BF16 buffers are interpreted as u32[]; round up to 4-byte multiple.
    let bytes = if matches!(dtype, VkDType::Bf16) {
        ((bytes + 3) / 4) * 4
    } else {
        bytes
    };
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes as u64,
    )
    .context("vk_cast: alloc output buffer")?;
    Ok(Arc::new(buf))
}

pub fn vk_cast_f32_to_bf16_no_grad(t: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(
        t.dtype() == VkDType::F32,
        "vk_cast_f32_to_bf16: input must be F32 (got {:?})",
        t.dtype()
    );
    let n = t.num_elements();
    anyhow::ensure!(n > 0, "vk_cast: empty tensor");
    let out = alloc_for(t.device(), n, VkDType::Bf16)?;
    let total_words = (n + 1) / 2;
    let workgroups = ((total_words + 255) / 256) as u32;
    let push_constants = [n as u32];
    dispatch_simple(
        t.device(),
        "vk_cast_f32_to_bf16",
        &[t.buffer().handle(), out.handle()],
        &push_constants,
        workgroups,
    )?;
    Ok(VkTensor::from_buffer(
        out,
        t.shape().to_vec(),
        VkDType::Bf16,
        Arc::clone(t.device()),
    ))
}

pub fn vk_cast_bf16_to_f32_no_grad(t: &VkTensor) -> Result<VkTensor> {
    anyhow::ensure!(
        t.dtype() == VkDType::Bf16,
        "vk_cast_bf16_to_f32: input must be BF16 (got {:?})",
        t.dtype()
    );
    let n = t.num_elements();
    anyhow::ensure!(n > 0, "vk_cast: empty tensor");
    let out = alloc_for(t.device(), n, VkDType::F32)?;
    let workgroups = ((n + 255) / 256) as u32;
    let push_constants = [n as u32];
    dispatch_simple(
        t.device(),
        "vk_cast_bf16_to_f32",
        &[t.buffer().handle(), out.handle()],
        &push_constants,
        workgroups,
    )?;
    Ok(VkTensor::from_buffer(
        out,
        t.shape().to_vec(),
        VkDType::F32,
        Arc::clone(t.device()),
    ))
}

#[derive(Debug)]
pub struct CastBackward {
    pub source_dtype: VkDType,
    pub inputs: [VkTensor; 1],
}

impl VkBackwardOp for CastBackward {
    fn op_name(&self) -> &'static str {
        "cast"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        // Cast grad back to the source dtype (identity gradient passthrough
        // up to a precision change).
        let grad_in = if grad_out.dtype() == self.source_dtype {
            grad_out.clone()
        } else {
            match (grad_out.dtype(), self.source_dtype) {
                (VkDType::F32, VkDType::Bf16) => vk_cast_f32_to_bf16_no_grad(grad_out)?,
                (VkDType::Bf16, VkDType::F32) => vk_cast_bf16_to_f32_no_grad(grad_out)?,
                (a, b) => anyhow::bail!("CastBackward: unsupported {:?} -> {:?}", a, b),
            }
        };
        Ok(vec![Some(grad_in)])
    }
}

pub fn vk_cast(t: &VkTensor, dst: VkDType) -> Result<VkTensor> {
    if t.dtype() == dst {
        return Ok(t.clone());
    }
    let out = match (t.dtype(), dst) {
        (VkDType::F32, VkDType::Bf16) => vk_cast_f32_to_bf16_no_grad(t)?,
        (VkDType::Bf16, VkDType::F32) => vk_cast_bf16_to_f32_no_grad(t)?,
        _ => anyhow::bail!("vk_cast: unsupported {:?} -> {:?}", t.dtype(), dst),
    };
    let grad_fn: Option<Arc<dyn VkBackwardOp>> = if t.requires_grad() {
        Some(Arc::new(CastBackward {
            source_dtype: t.dtype(),
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

//! Causal mask and in-place scalar scaling for VkTensor attention.
//!
//! `vk_causal_mask_inplace(scores, kv_offset)`: rewrites positions
//! `(q, k)` where `k > q + kv_offset` to `-1e30`. Operates in-place.
//!
//! `vk_scale_inplace(t, scale)`: multiplies every element by `scale`
//! in-place. Used before softmax to apply `1/sqrt(head_dim)`.

use crate::vk_ops::dispatch_simple;
use crate::vk_ops::elementwise::vk_add_no_grad;
use crate::vk_ops::reduce::vk_zeros_like;
use crate::vk_tensor::{VkDType, VkTensor};
use anyhow::Result;

/// In-place additive causal mask on `[batch_heads, q_len, k_len]`.
pub fn vk_causal_mask_inplace(scores: &VkTensor, kv_offset: usize) -> Result<()> {
    anyhow::ensure!(
        scores.dtype() == VkDType::F32,
        "vk_causal_mask: F32-only"
    );
    anyhow::ensure!(
        scores.shape().len() == 3,
        "vk_causal_mask: rank-3 required [batch_heads, q_len, k_len] (got {:?})",
        scores.shape()
    );
    let bh = scores.shape()[0];
    let q_len = scores.shape()[1];
    let k_len = scores.shape()[2];
    let total = bh * q_len * k_len;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [bh as u32, q_len as u32, k_len as u32, kv_offset as u32];
    dispatch_simple(
        scores.device(),
        "vk_causal_mask_add_f32",
        &[scores.buffer().handle()],
        &push,
        workgroups,
    )
}

/// In-place scalar multiply.
pub fn vk_scale_inplace(t: &VkTensor, scale: f32) -> Result<()> {
    anyhow::ensure!(t.dtype() == VkDType::F32, "vk_scale_inplace: F32-only");
    let n = t.num_elements();
    let workgroups = ((n + 255) / 256) as u32;
    let push = [n as u32, scale.to_bits()];
    dispatch_simple(
        t.device(),
        "vk_scale_inplace_f32",
        &[t.buffer().handle()],
        &push,
        workgroups,
    )
}

/// Out-of-place scalar multiply: returns a fresh VkTensor = t * scale.
/// Implemented as copy(zeros + t) then in-place scale.
pub fn vk_scale_no_grad(t: &VkTensor, scale: f32) -> Result<VkTensor> {
    let zeros = vk_zeros_like(t)?;
    let copy = vk_add_no_grad(t, &zeros)?;
    vk_scale_inplace(&copy, scale)?;
    Ok(copy)
}

/// Autograd-aware scalar multiply: `out = t * scale`. Backward is
/// `grad_in = grad_out * scale`.
pub fn vk_scale(t: &VkTensor, scale: f32) -> Result<VkTensor> {
    use crate::vk_tensor::VkBackwardOp;
    use std::sync::Arc;
    let out = vk_scale_no_grad(t, scale)?;
    if !t.requires_grad() {
        return Ok(out);
    }
    let grad_fn: Arc<dyn VkBackwardOp> = Arc::new(ScaleBackward {
        scale,
        inputs: [t.clone()],
    });
    Ok(VkTensor::from_op(
        Arc::clone(out.buffer()),
        out.shape().to_vec(),
        out.dtype(),
        Arc::clone(out.device()),
        Some(grad_fn),
    ))
}

#[derive(Debug)]
struct ScaleBackward {
    scale: f32,
    inputs: [VkTensor; 1],
}

impl crate::vk_tensor::VkBackwardOp for ScaleBackward {
    fn op_name(&self) -> &'static str {
        "scale"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(
        &self,
        grad_out: &VkTensor,
    ) -> anyhow::Result<Vec<Option<VkTensor>>> {
        let g = vk_scale_no_grad(grad_out, self.scale)?;
        Ok(vec![Some(g)])
    }
}

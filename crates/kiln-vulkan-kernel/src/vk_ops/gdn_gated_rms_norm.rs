//! GDN gated RMSNorm: out = (x / rms(x)) · silu(z) · weight.
//!
//! Wraps the existing inference shader `gdn_gated_rms_norm.comp`.
//!
//! Phase 2 ships the forward wrapper. The autograd-aware variant
//! (composing RMSNorm-bwd, SiLU-bwd-on-z, elementwise-mul-bwd) lands
//! in Phase 4 alongside `vk_gdn_gated_rms_norm_bwd.comp`.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n * 4).max(4);
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes as u64,
    )
    .context("vk_gdn_gated_rms_norm: alloc f32")?;
    Ok(Arc::new(buf))
}

/// Forward gated RMSNorm.
///
/// Shapes:
///   x:       [rows, hidden]   F32   (typically [B*T, nv*dv])
///   z:       [rows, hidden]   F32
///   weight:  [hidden]         F32
/// Returns:
///   out:     [rows, hidden]   F32
///
/// Note: the existing inference shader uses `weight` directly (NOT
/// `1 + weight` like Llama-style RMSNorm). Callers must pass the raw
/// learned scale.
pub fn vk_gdn_gated_rms_norm_no_grad(
    x: &VkTensor,
    z: &VkTensor,
    weight: &VkTensor,
    eps: f32,
) -> Result<VkTensor> {
    anyhow::ensure!(x.dtype() == VkDType::F32, "vk_gdn_gated_rms_norm: F32 only");
    anyhow::ensure!(z.dtype() == VkDType::F32, "vk_gdn_gated_rms_norm: F32 only");
    anyhow::ensure!(
        weight.dtype() == VkDType::F32,
        "vk_gdn_gated_rms_norm: F32 weight only"
    );
    anyhow::ensure!(
        x.shape() == z.shape(),
        "vk_gdn_gated_rms_norm: x/z shape mismatch ({:?} vs {:?})",
        x.shape(),
        z.shape()
    );
    let dims = x.shape();
    let hidden = *dims
        .last()
        .context("vk_gdn_gated_rms_norm: empty shape")?;
    anyhow::ensure!(
        weight.num_elements() == hidden,
        "vk_gdn_gated_rms_norm: weight size {} != hidden {}",
        weight.num_elements(),
        hidden
    );
    let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);

    let device = x.device();
    let total = rows * hidden;
    let out = alloc_f32(device, total)?;

    let workgroups = rows as u32;
    let push = [rows as u32, hidden as u32, eps.to_bits()];
    dispatch_simple(
        device,
        "gdn_gated_rms_norm",
        &[
            x.buffer().handle(),
            z.buffer().handle(),
            weight.buffer().handle(),
            out.handle(),
        ],
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

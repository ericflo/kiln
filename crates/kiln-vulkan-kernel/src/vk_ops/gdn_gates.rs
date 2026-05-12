//! GDN gates: β = sigmoid(b), g = -exp(A_log) · softplus(a + dt_bias).
//!
//! Wraps the existing inference shader `gdn_gates.comp`.
//!
//! Phase 2 ships the forward wrapper. The autograd-aware variant
//! (with `GatesBackward`) lands in Phase 4 alongside
//! `vk_gdn_gates_bwd.comp`.

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
    .context("vk_gdn_gates: alloc f32")?;
    Ok(Arc::new(buf))
}

/// Forward GDN gates.
///
/// Shapes:
///   a:          [B, T, nv]   F32
///   b:          [B, T, nv]   F32
///   a_log:      [nv]         F32
///   dt_bias:    [nv]         F32
/// Returns:
///   beta_out:   [B, T, nv]   F32  (= sigmoid(b))
///   g_out:      [B, T, nv]   F32  (= -exp(a_log) · softplus(a + dt_bias))
pub fn vk_gdn_gates_no_grad(
    a: &VkTensor,
    b: &VkTensor,
    a_log: &VkTensor,
    dt_bias: &VkTensor,
    nv: usize,
) -> Result<(VkTensor, VkTensor)> {
    anyhow::ensure!(a.dtype() == VkDType::F32, "vk_gdn_gates: F32 only");
    anyhow::ensure!(b.dtype() == VkDType::F32, "vk_gdn_gates: F32 only");
    anyhow::ensure!(a_log.dtype() == VkDType::F32, "vk_gdn_gates: F32 only");
    anyhow::ensure!(dt_bias.dtype() == VkDType::F32, "vk_gdn_gates: F32 only");
    anyhow::ensure!(
        a.num_elements() == b.num_elements(),
        "vk_gdn_gates: a/b element-count mismatch"
    );
    anyhow::ensure!(
        a_log.num_elements() == nv,
        "vk_gdn_gates: a_log size {} != nv {}",
        a_log.num_elements(),
        nv
    );
    anyhow::ensure!(
        dt_bias.num_elements() == nv,
        "vk_gdn_gates: dt_bias size {} != nv {}",
        dt_bias.num_elements(),
        nv
    );
    let total = a.num_elements();
    anyhow::ensure!(
        total % nv == 0,
        "vk_gdn_gates: total {} not divisible by nv {}",
        total,
        nv
    );

    let device = a.device();
    let beta_out = alloc_f32(device, total)?;
    let g_out = alloc_f32(device, total)?;

    let workgroups = ((total + 255) / 256) as u32;
    let push = [total as u32, nv as u32];
    dispatch_simple(
        device,
        "gdn_gates",
        &[
            a.buffer().handle(),
            b.buffer().handle(),
            a_log.buffer().handle(),
            dt_bias.buffer().handle(),
            beta_out.handle(),
            g_out.handle(),
        ],
        &push,
        workgroups,
    )?;

    Ok((
        VkTensor::from_buffer(beta_out, a.shape().to_vec(), VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(g_out, a.shape().to_vec(), VkDType::F32, Arc::clone(device)),
    ))
}

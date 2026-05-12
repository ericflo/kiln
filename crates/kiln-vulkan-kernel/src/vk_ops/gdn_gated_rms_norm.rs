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

/// CPU backward for gated RMSNorm.
///
/// Forward: out_i = x_i · r_inv · silu(z_i) · w_i, with r = sqrt(mean(x²)+ε).
///
/// Returns (d_x, d_z, d_w).
///
/// Implementation note: Phase 4 v1 is CPU-only (small sizes per row,
/// one row per token-step). A GLSL shader replacement is queued for
/// later optimization.
pub fn vk_gdn_gated_rms_norm_bwd_no_grad(
    d_out: &VkTensor, // [rows, hidden]
    x: &VkTensor,     // [rows, hidden]
    z: &VkTensor,
    weight: &VkTensor, // [hidden]
    eps: f32,
) -> Result<(VkTensor, VkTensor, VkTensor)> {
    let dims = x.shape();
    let hidden = *dims.last().context("empty shape")?;
    let rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
    let device = x.device();

    let dout_data = d_out.to_vec_f32()?;
    let x_data = x.to_vec_f32()?;
    let z_data = z.to_vec_f32()?;
    let w_data = weight.to_vec_f32()?;

    let mut d_x = vec![0.0_f32; rows * hidden];
    let mut d_z = vec![0.0_f32; rows * hidden];
    let mut d_w = vec![0.0_f32; hidden];

    for r in 0..rows {
        let base = r * hidden;
        // Compute r_inv
        let mut sum_sq = 0.0_f32;
        for c in 0..hidden {
            sum_sq += x_data[base + c] * x_data[base + c];
        }
        let m = sum_sq / (hidden as f32);
        let rms = (m + eps).sqrt();
        let r_inv = 1.0 / rms;
        // Per-row reduction: Σ_j d_out_j · x_j · silu(z_j) · w_j
        let mut row_dot = 0.0_f32;
        // Cache silu and silu' for this row
        let mut silu = vec![0.0_f32; hidden];
        let mut silu_d = vec![0.0_f32; hidden];
        for c in 0..hidden {
            let zv = z_data[base + c];
            let sig = if zv >= 0.0 {
                1.0 / (1.0 + (-zv).exp())
            } else {
                let e = zv.exp();
                e / (1.0 + e)
            };
            let s = zv * sig;
            silu[c] = s;
            silu_d[c] = sig + s * (1.0 - sig); // d/dz [z·sigmoid(z)]
            row_dot += dout_data[base + c] * x_data[base + c] * s * w_data[c];
        }
        let inv3 = r_inv * r_inv * r_inv;
        for c in 0..hidden {
            let dout_v = dout_data[base + c];
            let xv = x_data[base + c];
            let g = silu[c] * w_data[c];
            // d_x:
            d_x[base + c] = dout_v * r_inv * g - inv3 * xv / (hidden as f32) * row_dot;
            // d_z:
            d_z[base + c] = dout_v * xv * r_inv * w_data[c] * silu_d[c];
            // d_w (accumulated across rows):
            d_w[c] += dout_v * xv * r_inv * silu[c];
        }
    }

    let mk_buf = |data: &[f32], shape: Vec<usize>| -> Result<VkTensor> {
        let buf = alloc_f32(device, data.len())?;
        let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        VulkanBuffer::upload_data(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &buf,
            &raw,
        )?;
        Ok(VkTensor::from_buffer(
            buf,
            shape,
            VkDType::F32,
            Arc::clone(device),
        ))
    };

    Ok((
        mk_buf(&d_x, dims.to_vec())?,
        mk_buf(&d_z, dims.to_vec())?,
        mk_buf(&d_w, vec![hidden])?,
    ))
}

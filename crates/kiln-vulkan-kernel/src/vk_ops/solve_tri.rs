//! Forward-substitution triangular solve that produces
//! W = (I + diag(β) · A_strict)^{-1} · diag(β) · V'.
//!
//! NOTE: The existing inference shader `solve_tri.comp` requests 192 KB
//! of shared memory (16384 + 32768 floats), exceeding most devices'
//! per-workgroup shared-memory limit. It SIGFPEs at pipeline creation
//! on Strix Halo. Until a fresh shader is written (Phase 4 critical
//! path), Phase 3 falls back to a CPU implementation: read inputs to
//! host, compute forward sub, upload W. Slow but correct, and the
//! sizes are small (C ≤ 64, dv ≤ 256, ~16 KB per layer per chunk) so
//! the readback cost is bounded.

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
    .context("vk_solve_tri: alloc")?;
    Ok(Arc::new(buf))
}

/// Solve W = (I + diag(β)·A_strict)^{-1} · diag(β) · V' by forward sub.
///
/// Inputs:
///   a_strict: [B, nv, C, C]    F32  (strict-lower triangular)
///   v_prime:  [B, nv, C, dv]   F32
///   beta:     [B, nv, C]       F32
/// Output:
///   w:        [B, nv, C, dv]   F32
pub fn vk_solve_tri_no_grad(
    a_strict: &VkTensor,
    v_prime: &VkTensor,
    beta: &VkTensor,
    batch: usize,
    heads: usize,
    chunk: usize,
    dv: usize,
) -> Result<VkTensor> {
    anyhow::ensure!(a_strict.dtype() == VkDType::F32, "vk_solve_tri: F32");
    anyhow::ensure!(v_prime.dtype() == VkDType::F32, "vk_solve_tri: F32");
    anyhow::ensure!(beta.dtype() == VkDType::F32, "vk_solve_tri: F32");
    anyhow::ensure!(
        a_strict.num_elements() == batch * heads * chunk * chunk,
        "vk_solve_tri: a_strict shape"
    );
    anyhow::ensure!(
        v_prime.num_elements() == batch * heads * chunk * dv,
        "vk_solve_tri: v_prime shape"
    );
    anyhow::ensure!(
        beta.num_elements() == batch * heads * chunk,
        "vk_solve_tri: beta shape"
    );
    anyhow::ensure!(
        chunk <= 128 && dv <= 256,
        "vk_solve_tri: shader caps require chunk≤128, dv≤256"
    );

    let device = a_strict.device();
    // CPU forward substitution. Inputs to host:
    let a_data = a_strict.to_vec_f32()?;
    let v_data = v_prime.to_vec_f32()?;
    let b_data = beta.to_vec_f32()?;
    let mut w = vec![0.0_f32; batch * heads * chunk * dv];
    for bh in 0..batch * heads {
        let a_base = bh * chunk * chunk;
        let v_base = bh * chunk * dv;
        let beta_base = bh * chunk;
        for t in 0..chunk {
            let beta_t = b_data[beta_base + t];
            let a_row = a_base + t * chunk;
            for d in 0..dv {
                let mut acc = 0.0_f32;
                for j in 0..t {
                    acc += a_data[a_row + j] * w[v_base + j * dv + d];
                }
                let vp = v_data[v_base + t * dv + d];
                w[v_base + t * dv + d] = beta_t * (vp - acc);
            }
        }
    }
    let out = alloc_f32(device, w.len())?;
    let raw: Vec<u8> = w.iter().flat_map(|f| f.to_le_bytes()).collect();
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &out,
        &raw,
    )?;

    Ok(VkTensor::from_buffer(
        out,
        vec![batch, heads, chunk, dv],
        VkDType::F32,
        Arc::clone(device),
    ))
}

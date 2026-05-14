//! Forward-substitution triangular solve that produces
//! W = (I + diag(β) · A_strict)^{-1} · diag(β) · V'.
//!
//! Uses `vk_solve_tri_v2.comp` — a fresh GLSL shader with bounded
//! shared memory (32 KB total: 64×64 sA + 64×DV_PER_WG=64 sW). Replaces
//! the inference codebase's `solve_tri.comp` which requested 192 KB
//! of shared memory and SIGFPEd at pipeline creation on Strix Halo.
//!
//! The dispatch is 2D: one workgroup per (B*H, dv-tile). For
//! C ≤ 64 and dv ≤ 256 (the Qwen3.5-4B envelope) we tile dv in
//! 64-element chunks. Larger dv just runs more workgroups.
//!
//! A CPU fallback is kept around (`vk_solve_tri_cpu_fallback`) for
//! debugging or when the shader path is disabled via env.

use crate::vk_ops::dispatch_simple_2d;
use crate::vk_tensor::{VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::Result;
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
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
    let out = alloc_f32(device, batch * heads * chunk * dv)?;

    // Use CPU fallback if explicitly disabled via env (debug aid).
    if std::env::var("KILN_VK_SOLVE_TRI_CPU").is_ok() {
        return vk_solve_tri_cpu_fallback(a_strict, v_prime, beta, out, batch, heads, chunk, dv);
    }

    // GPU path: vk_solve_tri_v2.comp (32 KB shared mem, well within
    // Strix Halo's per-workgroup cap).
    let dv_per_wg = 64u32;
    let dv_tiles = (dv as u32 + dv_per_wg - 1) / dv_per_wg;
    let push = [batch as u32, heads as u32, chunk as u32, dv as u32];
    dispatch_simple_2d(
        device,
        "vk_solve_tri_v2",
        &[
            a_strict.buffer().handle(),
            v_prime.buffer().handle(),
            beta.buffer().handle(),
            out.handle(),
        ],
        &push,
        ((batch * heads) as u32, dv_tiles),
    )?;

    Ok(VkTensor::from_buffer(
        out,
        vec![batch, heads, chunk, dv],
        VkDType::F32,
        Arc::clone(device),
    ))
}

/// CPU forward-substitution fallback. Used when the GPU shader path
/// is disabled or for testing.
fn vk_solve_tri_cpu_fallback(
    a_strict: &VkTensor,
    v_prime: &VkTensor,
    beta: &VkTensor,
    out: Arc<VulkanBuffer>,
    batch: usize,
    heads: usize,
    chunk: usize,
    dv: usize,
) -> Result<VkTensor> {
    let device = a_strict.device();
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

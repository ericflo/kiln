//! Wrapper for the existing inference shader `gdn_chunk_prep.comp`.
//!
//! Bindings (12, all F32):
//!   in:  g, v, kkt, qkt, ks_entry, q_s
//!   out: a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last
//!
//! Push constants: (batch, heads, chunk, dv).
//!
//! Phase 3 forward-only wrapper. The autograd-aware variant + the
//! backward shader (`vk_gdn_chunk_prep_bwd.comp`) ship in Phase 4.

use crate::vk_ops::dispatch_simple;
use crate::vk_tensor::{VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

/// Output bundle: (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last).
pub struct GdnChunkPrepOutput {
    pub a_strict: VkTensor,       // [B, nv, C, C] F32
    pub b_mask: VkTensor,         // [B, nv, C, C] F32
    pub v_prime: VkTensor,        // [B, nv, C, dv] F32
    pub q_s_scaled: VkTensor,     // [B, nv, C, dv] F32
    pub decay_last_col: VkTensor, // [B, nv, C] F32
    pub p_last: VkTensor,         // [B, nv] F32
}

/// Forward chunk_prep.
///
/// All inputs are F32. Caller has already computed:
///   kkt      = k_c @ k_c^T
///   qkt      = q_c @ k_c^T
///   ks_entry = k_c @ S_in
///   q_s      = q_c @ S_in
/// via `vk_matmul_batched` etc.
pub fn vk_gdn_chunk_prep_no_grad(
    g: &VkTensor,        // [B, nv, C]
    v: &VkTensor,        // [B, nv, C, dv]
    kkt: &VkTensor,      // [B, nv, C, C]
    qkt: &VkTensor,      // [B, nv, C, C]
    ks_entry: &VkTensor, // [B, nv, C, dv]
    q_s: &VkTensor,      // [B, nv, C, dv]
    batch: usize,
    heads: usize,
    chunk: usize,
    dv: usize,
) -> Result<GdnChunkPrepOutput> {
    let device = g.device();
    for t in [g, v, kkt, qkt, ks_entry, q_s] {
        anyhow::ensure!(t.dtype() == VkDType::F32, "vk_gdn_chunk_prep: F32 only");
    }
    anyhow::ensure!(g.num_elements() == batch * heads * chunk, "g size");
    anyhow::ensure!(v.num_elements() == batch * heads * chunk * dv, "v size");
    anyhow::ensure!(
        kkt.num_elements() == batch * heads * chunk * chunk,
        "kkt size"
    );
    anyhow::ensure!(
        qkt.num_elements() == batch * heads * chunk * chunk,
        "qkt size"
    );
    anyhow::ensure!(
        ks_entry.num_elements() == batch * heads * chunk * dv,
        "ks_entry size"
    );
    anyhow::ensure!(q_s.num_elements() == batch * heads * chunk * dv, "q_s size");

    let bh = batch * heads;
    let a_strict = alloc_f32(device, bh * chunk * chunk)?;
    let b_mask = alloc_f32(device, bh * chunk * chunk)?;
    let v_prime = alloc_f32(device, bh * chunk * dv)?;
    let q_s_scaled = alloc_f32(device, bh * chunk * dv)?;
    let decay_last_col = alloc_f32(device, bh * chunk)?;
    let p_last = alloc_f32(device, bh)?;

    let per_bh = chunk * chunk + chunk * dv + chunk + 1;
    let total = bh * per_bh;
    let workgroups = ((total + 255) / 256) as u32;
    let push = [batch as u32, heads as u32, chunk as u32, dv as u32];

    dispatch_simple(
        device,
        "gdn_chunk_prep",
        &[
            g.buffer().handle(),
            v.buffer().handle(),
            kkt.buffer().handle(),
            qkt.buffer().handle(),
            ks_entry.buffer().handle(),
            q_s.buffer().handle(),
            a_strict.handle(),
            b_mask.handle(),
            v_prime.handle(),
            q_s_scaled.handle(),
            decay_last_col.handle(),
            p_last.handle(),
        ],
        &push,
        workgroups,
    )?;

    Ok(GdnChunkPrepOutput {
        a_strict: VkTensor::from_buffer(
            a_strict,
            vec![batch, heads, chunk, chunk],
            VkDType::F32,
            Arc::clone(device),
        ),
        b_mask: VkTensor::from_buffer(
            b_mask,
            vec![batch, heads, chunk, chunk],
            VkDType::F32,
            Arc::clone(device),
        ),
        v_prime: VkTensor::from_buffer(
            v_prime,
            vec![batch, heads, chunk, dv],
            VkDType::F32,
            Arc::clone(device),
        ),
        q_s_scaled: VkTensor::from_buffer(
            q_s_scaled,
            vec![batch, heads, chunk, dv],
            VkDType::F32,
            Arc::clone(device),
        ),
        decay_last_col: VkTensor::from_buffer(
            decay_last_col,
            vec![batch, heads, chunk],
            VkDType::F32,
            Arc::clone(device),
        ),
        p_last: VkTensor::from_buffer(p_last, vec![batch, heads], VkDType::F32, Arc::clone(device)),
    })
}

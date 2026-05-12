//! Vulkan-native chunkwise GDN forward (mirrors candle's
//! `gdn_chunkwise_recurrence` in forward.rs:4679).
//!
//! Inputs:
//!   q: [B, nv, T, dk]   F32
//!   k: [B, nv, T, dk]   F32
//!   v: [B, nv, T, dv]   F32
//!   beta: [B, nv, T]    F32
//!   g: [B, nv, T]       F32
//!   state: [B, nv, dk, dv]  F32   (mutated in place per chunk)
//!
//! Returns: out [B, nv, T, dv] F32.
//!
//! Each chunk runs:
//!   1. Compute kkt, qkt, ks_entry, q_s via batched matmuls
//!   2. Run vk_gdn_chunk_prep → a_strict, b_mask, v_prime, q_s_scaled,
//!      decay_last_col, p_last
//!   3. Run vk_solve_tri → W
//!   4. out_chunk = q_s_scaled + b_mask · W
//!   5. State update: S_new = p_last·S + k^T · (decay_last_col·W)
//!
//! Phase 3 is forward-only. The autograd-aware variant + cross-chunk
//! reverse iteration ship in Phase 5.

use crate::vk_ops::elementwise::{vk_add_no_grad, vk_mul_no_grad};
use crate::vk_ops::gdn_chunk_prep::{vk_gdn_chunk_prep_no_grad, GdnChunkPrepOutput};
use crate::vk_ops::mask::vk_scale_no_grad;
use crate::vk_ops::matmul_batched::{vk_matmul_batched_no_grad, vk_transpose_batched_2d_no_grad};
use crate::vk_ops::narrow::vk_narrow_lastdim_no_grad;
use crate::vk_ops::shape::vk_reshape;
use crate::vk_ops::solve_tri::vk_solve_tri_no_grad;
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
    .context("vk_gdn_chunkwise: alloc")?;
    Ok(Arc::new(buf))
}

/// Slice along time dim (dim=2 of a [B, nv, T, ...] tensor) for a
/// contiguous chunk. Implemented as a fresh allocation + element-wise
/// copy via dispatch_simple. No autograd attached (Phase 3 forward only).
fn time_narrow_no_grad(
    t: &VkTensor,
    t_start: usize,
    t_len: usize,
    last_axis: usize,
) -> Result<VkTensor> {
    // Reshape to flatten (B, nv) into a single outer dim, treating
    // [B*nv*T, last_axis] as 2D, then narrow on axis 1's slice
    // covering rows [t_start * last_axis .. (t_start + t_len) * last_axis)
    // for each (B*nv) outer element.
    //
    // Simplest correct path: reshape to [B*nv, T, last_axis] then for
    // each outer take rows t_start..t_start+t_len. We don't have a
    // dim=1 narrow shader yet, so we do this via raw buffer copy: for
    // each b_h in [0, B*nv), copy contiguous bytes
    //   src[b_h * T * last_axis + t_start * last_axis ..
    //       b_h * T * last_axis + (t_start + t_len) * last_axis]
    // into
    //   dst[b_h * t_len * last_axis ..
    //       (b_h + 1) * t_len * last_axis].
    //
    // Because `last_axis` may be 1 (g, beta) we keep the math general.
    let dims = t.shape();
    debug_assert!(dims.len() >= 3, "time_narrow: rank >= 3");
    let bh: usize = dims[..dims.len() - 2].iter().product();
    let t_total = dims[dims.len() - 2];
    debug_assert_eq!(dims[dims.len() - 1], last_axis);
    debug_assert!(t_start + t_len <= t_total);

    // Reshape input to [B*nv, T*last_axis]; chunk_prep wants [B, nv, C, last]
    // so output shape = dims[..-2] + [t_len, last_axis].
    let input_2d = vk_reshape(t, &[bh, t_total * last_axis])?;
    let out_n = bh * t_len * last_axis;
    let out_buf = alloc_f32(t.device(), out_n)?;

    // Copy via the existing vk_narrow shader: dst[i, 0..t_len*last_axis] =
    // src[i, t_start*last_axis .. (t_start+t_len)*last_axis]. That's
    // exactly what vk_narrow_lastdim does on a 2D input.
    let narrowed = vk_narrow_lastdim_no_grad(
        &input_2d,
        t_start * last_axis,
        t_len * last_axis,
    )?;
    // Copy bytes from narrowed buffer into out_buf to detach dependency
    let bytes = narrowed.to_vec_f32()?;
    let raw: Vec<u8> = bytes.iter().flat_map(|f| f.to_le_bytes()).collect();
    VulkanBuffer::upload_data(
        t.device().device(),
        t.device().host_visible_mem_type(),
        t.device().queue(),
        t.device().queue_family_index(),
        &out_buf,
        &raw,
    )?;

    let mut out_shape = dims[..dims.len() - 2].to_vec();
    out_shape.push(t_len);
    out_shape.push(last_axis);
    Ok(VkTensor::from_buffer(
        out_buf,
        out_shape,
        VkDType::F32,
        Arc::clone(t.device()),
    ))
}

/// Time-narrow a [B, nv, T] tensor (g or beta) → [B, nv, t_len].
fn time_narrow_3d_no_grad(t: &VkTensor, t_start: usize, t_len: usize) -> Result<VkTensor> {
    let dims = t.shape();
    debug_assert_eq!(dims.len(), 3);
    let bh = dims[0] * dims[1];
    let t_total = dims[2];
    let input_2d = vk_reshape(t, &[bh, t_total])?;
    let narrowed = vk_narrow_lastdim_no_grad(&input_2d, t_start, t_len)?;
    let out_n = bh * t_len;
    let out_buf = alloc_f32(t.device(), out_n)?;
    let bytes = narrowed.to_vec_f32()?;
    let raw: Vec<u8> = bytes.iter().flat_map(|f| f.to_le_bytes()).collect();
    VulkanBuffer::upload_data(
        t.device().device(),
        t.device().host_visible_mem_type(),
        t.device().queue(),
        t.device().queue_family_index(),
        &out_buf,
        &raw,
    )?;
    Ok(VkTensor::from_buffer(
        out_buf,
        vec![dims[0], dims[1], t_len],
        VkDType::F32,
        Arc::clone(t.device()),
    ))
}

/// Concatenate along time axis (dim=2 of [B, nv, T, last_axis]).
/// Used to assemble the per-chunk outputs into the final [B, nv, T, dv]
/// tensor. CPU concat through readback (Phase 3 acceptable).
fn concat_time(chunks: &[VkTensor], last_axis: usize) -> Result<VkTensor> {
    anyhow::ensure!(!chunks.is_empty(), "concat_time: empty input");
    let dims = chunks[0].shape();
    debug_assert!(dims.len() == 4 && dims[3] == last_axis);
    let batch = dims[0];
    let nv = dims[1];
    let total_t: usize = chunks.iter().map(|c| c.shape()[2]).sum();
    let out_n = batch * nv * total_t * last_axis;
    let mut out = vec![0.0_f32; out_n];

    let mut t_off = 0;
    for chunk in chunks {
        let cd = chunk.shape();
        let c_t = cd[2];
        let data = chunk.to_vec_f32()?;
        for b in 0..batch {
            for h in 0..nv {
                for ct in 0..c_t {
                    let src_base = ((b * nv + h) * c_t + ct) * last_axis;
                    let dst_base = ((b * nv + h) * total_t + (t_off + ct)) * last_axis;
                    out[dst_base..dst_base + last_axis]
                        .copy_from_slice(&data[src_base..src_base + last_axis]);
                }
            }
        }
        t_off += c_t;
    }

    let device = chunks[0].device();
    let out_buf = alloc_f32(device, out_n)?;
    let raw: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &out_buf,
        &raw,
    )?;
    Ok(VkTensor::from_buffer(
        out_buf,
        vec![batch, nv, total_t, last_axis],
        VkDType::F32,
        Arc::clone(device),
    ))
}

/// Per-chunk forward step.
///
/// Returns (out_chunk [B, nv, C, dv], w [B, nv, C, dv]).
fn chunk_forward_no_grad(
    q_c: &VkTensor,    // [B, nv, C, dk]
    k_c: &VkTensor,    // [B, nv, C, dk]
    v_c: &VkTensor,    // [B, nv, C, dv]
    beta_c: &VkTensor, // [B, nv, C]
    g_c: &VkTensor,    // [B, nv, C]
    state: &VkTensor,  // [B, nv, dk, dv]
    batch: usize,
    nv: usize,
    chunk: usize,
    dk: usize,
    dv: usize,
) -> Result<(VkTensor, VkTensor, VkTensor, VkTensor)> {
    // Reshape from [B, nv, C, dk] → [B*nv, C, dk] for batched matmul.
    let bh = batch * nv;
    let q_3 = vk_reshape(q_c, &[bh, chunk, dk])?;
    let k_3 = vk_reshape(k_c, &[bh, chunk, dk])?;
    let v_3 = vk_reshape(v_c, &[bh, chunk, dv])?;
    let s_3 = vk_reshape(state, &[bh, dk, dv])?;

    // k^T: [B*nv, dk, C]
    let k_t = vk_transpose_batched_2d_no_grad(&k_3)?;
    // ks_entry = k @ S → [B*nv, C, dv]
    let ks_entry = vk_matmul_batched_no_grad(&k_3, &s_3)?;
    // q_s = q @ S → [B*nv, C, dv]
    let q_s = vk_matmul_batched_no_grad(&q_3, &s_3)?;
    // kkt = k @ k^T → [B*nv, C, C]
    let kkt = vk_matmul_batched_no_grad(&k_3, &k_t)?;
    // qkt = q @ k^T → [B*nv, C, C]
    let qkt = vk_matmul_batched_no_grad(&q_3, &k_t)?;

    // Reshape back to [B, nv, C, ...] for chunk_prep input
    let v_4 = vk_reshape(&v_3, &[batch, nv, chunk, dv])?;
    let kkt_4 = vk_reshape(&kkt, &[batch, nv, chunk, chunk])?;
    let qkt_4 = vk_reshape(&qkt, &[batch, nv, chunk, chunk])?;
    let ks_entry_4 = vk_reshape(&ks_entry, &[batch, nv, chunk, dv])?;
    let q_s_4 = vk_reshape(&q_s, &[batch, nv, chunk, dv])?;
    // chunk_prep takes g flat [B, nv, C]; ours is already that shape

    let prep: GdnChunkPrepOutput = vk_gdn_chunk_prep_no_grad(
        g_c,
        &v_4,
        &kkt_4,
        &qkt_4,
        &ks_entry_4,
        &q_s_4,
        batch,
        nv,
        chunk,
        dv,
    )?;

    // Solve W = (I + diag(β)·A_strict)^{-1} · diag(β) · V'
    let w_4 = vk_solve_tri_no_grad(
        &prep.a_strict,
        &prep.v_prime,
        beta_c,
        batch,
        nv,
        chunk,
        dv,
    )?;

    // out_chunk = q_s_scaled + b_mask @ W
    //   b_mask: [B, nv, C, C], W: [B, nv, C, dv] → matmul → [B, nv, C, dv]
    let bmask_3 = vk_reshape(&prep.b_mask, &[bh, chunk, chunk])?;
    let w_3 = vk_reshape(&w_4, &[bh, chunk, dv])?;
    let intra_3 = vk_matmul_batched_no_grad(&bmask_3, &w_3)?;
    let intra_4 = vk_reshape(&intra_3, &[batch, nv, chunk, dv])?;
    let out_chunk = vk_add_no_grad(&prep.q_s_scaled, &intra_4)?;

    // w_weighted = W * decay_last_col_u, where decay_last_col is [B, nv, C]
    // and w is [B, nv, C, dv]. Broadcast multiply along the dv axis.
    // decay_last_col is per-(B, nv, c), so w_weighted[b, nv, c, d] =
    //   w[b, nv, c, d] * decay_last_col[b, nv, c].
    let decay_dlc_4 = vk_reshape(&prep.decay_last_col, &[batch, nv, chunk, 1])?;
    // We don't have full broadcast multiply yet, so expand decay_dlc_4
    // along the dv axis manually.
    let decay_expanded = expand_lastdim(&decay_dlc_4, dv)?;
    let w_weighted_4 = vk_mul_no_grad(&w_4, &decay_expanded)?;

    Ok((out_chunk, w_weighted_4, prep.p_last, k_t))
}

/// Expand a [..., 1] tensor to [..., N] by repeating the last axis.
fn expand_lastdim(t: &VkTensor, n: usize) -> Result<VkTensor> {
    let dims = t.shape();
    debug_assert_eq!(dims[dims.len() - 1], 1);
    let outer: usize = dims[..dims.len() - 1].iter().product();
    let device = t.device();
    let data = t.to_vec_f32()?;
    let mut out = Vec::with_capacity(outer * n);
    for &v in &data {
        for _ in 0..n {
            out.push(v);
        }
    }
    let out_buf = alloc_f32(device, outer * n)?;
    let raw: Vec<u8> = out.iter().flat_map(|f| f.to_le_bytes()).collect();
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &out_buf,
        &raw,
    )?;
    let mut out_shape = dims[..dims.len() - 1].to_vec();
    out_shape.push(n);
    Ok(VkTensor::from_buffer(
        out_buf,
        out_shape,
        VkDType::F32,
        Arc::clone(device),
    ))
}

/// State update: S_new = p_last·S + k^T · w_weighted.
fn state_update(
    state: &VkTensor,    // [B, nv, dk, dv]
    p_last: &VkTensor,   // [B, nv]
    k_t: &VkTensor,      // [B*nv, dk, C]
    w_weighted: &VkTensor, // [B, nv, C, dv]
    batch: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    chunk: usize,
) -> Result<VkTensor> {
    // p_last broadcast to state shape
    let p_data = p_last.to_vec_f32()?;
    let s_data = state.to_vec_f32()?;
    let mut s_scaled = vec![0.0_f32; batch * nv * dk * dv];
    for b in 0..batch {
        for h in 0..nv {
            let p = p_data[b * nv + h];
            let off = (b * nv + h) * dk * dv;
            for i in 0..dk * dv {
                s_scaled[off + i] = s_data[off + i] * p;
            }
        }
    }
    let device = state.device();
    let s_scaled_buf = alloc_f32(device, batch * nv * dk * dv)?;
    let raw: Vec<u8> = s_scaled.iter().flat_map(|f| f.to_le_bytes()).collect();
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &s_scaled_buf,
        &raw,
    )?;
    let s_scaled_t = VkTensor::from_buffer(
        s_scaled_buf,
        vec![batch, nv, dk, dv],
        VkDType::F32,
        Arc::clone(device),
    );

    // delta_state = k_t @ w_weighted, k_t: [B*nv, dk, C], w_weighted reshape to [B*nv, C, dv]
    let bh = batch * nv;
    let w_3 = vk_reshape(w_weighted, &[bh, chunk, dv])?;
    let delta_3 = vk_matmul_batched_no_grad(k_t, &w_3)?;
    let delta_4 = vk_reshape(&delta_3, &[batch, nv, dk, dv])?;
    vk_add_no_grad(&s_scaled_t, &delta_4)
}

/// Forward chunkwise GDN recurrence.
///
/// Mutates `state` in place (returned via the new buffer reference);
/// the caller must keep the original Arc<VulkanBuffer> alive separately
/// if needed.
pub fn vk_gdn_chunkwise_forward_no_grad(
    q: &VkTensor,
    k: &VkTensor,
    v: &VkTensor,
    beta: &VkTensor,
    g: &VkTensor,
    state: &mut VkTensor,
    chunk_size: usize,
) -> Result<VkTensor> {
    let dims = q.shape();
    anyhow::ensure!(dims.len() == 4, "vk_gdn_chunkwise_forward: q rank-4");
    let batch = dims[0];
    let nv = dims[1];
    let seq_len = dims[2];
    let dk = dims[3];
    let dv = v.shape()[3];

    let _ = vk_scale_no_grad; // avoid unused warning
    let full_chunks = seq_len / chunk_size;
    let tail = seq_len - full_chunks * chunk_size;
    let total_chunks = full_chunks + if tail > 0 { 1 } else { 0 };
    let mut out_chunks: Vec<VkTensor> = Vec::with_capacity(total_chunks);

    for ci in 0..total_chunks {
        let is_tail = ci >= full_chunks;
        let c = if is_tail { tail } else { chunk_size };
        let t_start = ci * chunk_size;

        let q_c = time_narrow_no_grad(q, t_start, c, dk)?;
        let k_c = time_narrow_no_grad(k, t_start, c, dk)?;
        let v_c = time_narrow_no_grad(v, t_start, c, dv)?;
        let beta_c = time_narrow_3d_no_grad(beta, t_start, c)?;
        let g_c = time_narrow_3d_no_grad(g, t_start, c)?;

        let (out_chunk, w_weighted, p_last, k_t) = chunk_forward_no_grad(
            &q_c, &k_c, &v_c, &beta_c, &g_c, state, batch, nv, c, dk, dv,
        )?;

        out_chunks.push(out_chunk);

        // Update state in place
        let new_state = state_update(state, &p_last, &k_t, &w_weighted, batch, nv, dk, dv, c)?;
        *state = new_state;
    }

    concat_time(&out_chunks, dv)
}

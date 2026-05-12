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
use crate::vk_ops::gdn_chunk_bwd::{
    vk_gdn_chunk_prep_bwd_no_grad, vk_gdn_chunk_scan_bwd_no_grad, vk_gdn_state_exit_bwd_no_grad,
    vk_solve_tri_transpose_no_grad,
};
use crate::vk_ops::gdn_chunk_prep::{vk_gdn_chunk_prep_no_grad, GdnChunkPrepOutput};
use crate::vk_ops::mask::vk_scale_no_grad;
use crate::vk_ops::matmul_batched::{vk_matmul_batched_no_grad, vk_transpose_batched_2d_no_grad};
use crate::vk_ops::narrow::vk_narrow_lastdim_no_grad;
use crate::vk_ops::shape::vk_reshape;
use crate::vk_ops::solve_tri::vk_solve_tri_no_grad;
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
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
/// contiguous chunk. Uses vk_narrow_lastdim_no_grad on a [B*nv,
/// T*last_axis] reshape — the shader is a pure GPU copy. No CPU
/// round-trip.
fn time_narrow_no_grad(
    t: &VkTensor,
    t_start: usize,
    t_len: usize,
    last_axis: usize,
) -> Result<VkTensor> {
    let dims = t.shape();
    debug_assert!(dims.len() >= 3, "time_narrow: rank >= 3");
    let bh: usize = dims[..dims.len() - 2].iter().product();
    let t_total = dims[dims.len() - 2];
    debug_assert_eq!(dims[dims.len() - 1], last_axis);
    debug_assert!(t_start + t_len <= t_total);

    let input_2d = vk_reshape(t, &[bh, t_total * last_axis])?;
    let narrowed = vk_narrow_lastdim_no_grad(
        &input_2d,
        t_start * last_axis,
        t_len * last_axis,
    )?;
    let mut out_shape = dims[..dims.len() - 2].to_vec();
    out_shape.push(t_len);
    out_shape.push(last_axis);
    vk_reshape(&narrowed, &out_shape)
}

/// Time-narrow a [B, nv, T] tensor (g or beta) → [B, nv, t_len].
/// Pure GPU copy via vk_narrow_lastdim_no_grad.
fn time_narrow_3d_no_grad(t: &VkTensor, t_start: usize, t_len: usize) -> Result<VkTensor> {
    let dims = t.shape();
    debug_assert_eq!(dims.len(), 3);
    let bh = dims[0] * dims[1];
    let t_total = dims[2];
    let input_2d = vk_reshape(t, &[bh, t_total])?;
    let narrowed = vk_narrow_lastdim_no_grad(&input_2d, t_start, t_len)?;
    vk_reshape(&narrowed, &[dims[0], dims[1], t_len])
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
    // and w is [B, nv, C, dv]. GPU broadcast multiply along the dv axis.
    let decay_dlc_4 = vk_reshape(&prep.decay_last_col, &[batch, nv, chunk, 1])?;
    let w_weighted_4 = vk_broadcast_mul_lastdim_no_grad(&w_4, &decay_dlc_4, dv)?;

    Ok((out_chunk, w_weighted_4, prep.p_last, k_t))
}

/// GPU broadcast multiply: out[..., n] = a[..., n] · b[..., 0].
/// Used for the per-position decay scaling in the chunkwise forward.
fn vk_broadcast_mul_lastdim_no_grad(a: &VkTensor, b: &VkTensor, n: usize) -> Result<VkTensor> {
    use crate::vk_ops::dispatch_simple;
    let dims_a = a.shape();
    let dims_b = b.shape();
    debug_assert_eq!(dims_a[dims_a.len() - 1], n);
    debug_assert_eq!(dims_b[dims_b.len() - 1], 1);
    let total = a.num_elements();
    let device = a.device();
    let out_buf = alloc_f32(device, total)?;
    let push = [total as u32, n as u32];
    dispatch_simple(
        device,
        "vk_broadcast_mul_lastdim",
        &[a.buffer().handle(), b.buffer().handle(), out_buf.handle()],
        &push,
        ((total as u32 + 255) / 256) as u32,
    )?;
    Ok(VkTensor::from_buffer(
        out_buf,
        dims_a.to_vec(),
        VkDType::F32,
        Arc::clone(device),
    ))
}

/// State update: S_new = p_last·S + k^T · w_weighted.
/// Pure GPU: vk_broadcast_mul_lastdim for the p_last broadcast,
/// vk_matmul_batched + vk_add for the rest.
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
    // s_scaled[bn, ix] = state[bn, ix] · p_last[bn]   (broadcast over dk*dv)
    let bh = batch * nv;
    let s_2d = vk_reshape(state, &[bh, dk * dv])?;
    let p_2d = vk_reshape(p_last, &[bh, 1])?;
    let s_scaled_2d = vk_broadcast_mul_lastdim_no_grad(&s_2d, &p_2d, dk * dv)?;
    let s_scaled_4d = vk_reshape(&s_scaled_2d, &[batch, nv, dk, dv])?;

    // delta_state = k_t @ w_weighted
    let w_3 = vk_reshape(w_weighted, &[bh, chunk, dv])?;
    let delta_3 = vk_matmul_batched_no_grad(k_t, &w_3)?;
    let delta_4 = vk_reshape(&delta_3, &[batch, nv, dk, dv])?;
    vk_add_no_grad(&s_scaled_4d, &delta_4)
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

// ---------------------------------------------------------------------------
// Phase 5: autograd-aware vk_gdn_chunkwise + GdnChunkwiseBackward op
// ---------------------------------------------------------------------------

/// Per-chunk saved tensors needed for backward.
#[derive(Clone)]
struct ChunkSaved {
    a_strict: VkTensor,
    b_mask: VkTensor,
    v_prime: VkTensor,
    q_s_scaled: VkTensor,
    decay_last_col: VkTensor,
    p_last: VkTensor,
    w: VkTensor,
    s_in_snapshot: VkTensor, // S_in for this chunk (pre-update)
    q_c: VkTensor,
    k_c: VkTensor,
    beta_c: VkTensor,
    g_c: VkTensor,
    kkt: VkTensor,
    qkt: VkTensor,
    ks_entry: VkTensor,
    q_s: VkTensor,
    v_c: VkTensor,
    chunk_len: usize,
}

#[derive(Clone)]
struct GdnChunkwiseBackward {
    chunks: Vec<ChunkSaved>,
    inputs: [VkTensor; 5], // q, k, v, beta, g
    batch: usize,
    nv: usize,
    seq_len: usize,
    dk: usize,
    dv: usize,
    chunk_size: usize,
}

impl std::fmt::Debug for GdnChunkwiseBackward {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GdnChunkwiseBackward")
            .field("batch", &self.batch)
            .field("nv", &self.nv)
            .field("seq_len", &self.seq_len)
            .field("dk", &self.dk)
            .field("dv", &self.dv)
            .field("chunk_size", &self.chunk_size)
            .field("num_chunks", &self.chunks.len())
            .finish()
    }
}

fn alloc_zeroed_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    let bytes = (n * 4).max(4);
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes as u64,
    )
    .context("alloc_zeroed_f32")?;
    let zeros = vec![0u8; bytes];
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &buf,
        &zeros,
    )?;
    Ok(Arc::new(buf))
}

fn upload_f32(device: &Arc<VulkanDevice>, data: &[f32], shape: Vec<usize>) -> Result<VkTensor> {
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
}

impl GdnChunkwiseBackward {
    /// CPU adjoint of the matmul-prep stage:
    ///   kkt      = k_c · k_c^T          shape [B, nv, C, C]
    ///   qkt      = q_c · k_c^T          shape [B, nv, C, C]
    ///   ks_entry = k_c · S_in           shape [B, nv, C, dv]
    ///   q_s      = q_c · S_in           shape [B, nv, C, dv]
    /// Given d_kkt, d_qkt, d_ks_entry, d_q_s, accumulates dq, dk, dS_in.
    fn matmul_prep_bwd(
        &self,
        ci: usize,
        d_kkt: &[f32],
        d_qkt: &[f32],
        d_ks_entry: &[f32],
        d_q_s: &[f32],
        dq_accum: &mut [f32],
        dk_accum: &mut [f32],
        ds_in: &mut [f32],
    ) {
        let chunk = self.chunks[ci].chunk_len;
        let q_data = self.chunks[ci].q_c.to_vec_f32().unwrap();
        let k_data = self.chunks[ci].k_c.to_vec_f32().unwrap();
        let s_data = self.chunks[ci].s_in_snapshot.to_vec_f32().unwrap();

        let bh = self.batch * self.nv;
        for bi in 0..bh {
            let q_base = bi * chunk * self.dk;
            let k_base = bi * chunk * self.dk;
            let s_base = bi * self.dk * self.dv;
            let cc_base = bi * chunk * chunk;
            let cv_base = bi * chunk * self.dv;

            // kkt[t,i] = Σ_kk k[t,kk] · k[i,kk]
            // dk[t,kk] += Σ_i d_kkt[t,i] · k[i,kk]
            // dk[i,kk] += Σ_t d_kkt[t,i] · k[t,kk]
            for t in 0..chunk {
                for kk in 0..self.dk {
                    let mut acc = 0.0_f32;
                    for i in 0..chunk {
                        acc += d_kkt[cc_base + t * chunk + i] * k_data[k_base + i * self.dk + kk];
                    }
                    dk_accum[k_base + t * self.dk + kk] += acc;
                }
            }
            for i in 0..chunk {
                for kk in 0..self.dk {
                    let mut acc = 0.0_f32;
                    for t in 0..chunk {
                        acc += d_kkt[cc_base + t * chunk + i] * k_data[k_base + t * self.dk + kk];
                    }
                    dk_accum[k_base + i * self.dk + kk] += acc;
                }
            }

            // qkt[t,i] = Σ_kk q[t,kk] · k[i,kk]
            // dq[t,kk] += Σ_i d_qkt[t,i] · k[i,kk]
            // dk[i,kk] += Σ_t d_qkt[t,i] · q[t,kk]
            for t in 0..chunk {
                for kk in 0..self.dk {
                    let mut acc = 0.0_f32;
                    for i in 0..chunk {
                        acc += d_qkt[cc_base + t * chunk + i] * k_data[k_base + i * self.dk + kk];
                    }
                    dq_accum[q_base + t * self.dk + kk] += acc;
                }
            }
            for i in 0..chunk {
                for kk in 0..self.dk {
                    let mut acc = 0.0_f32;
                    for t in 0..chunk {
                        acc += d_qkt[cc_base + t * chunk + i] * q_data[q_base + t * self.dk + kk];
                    }
                    dk_accum[k_base + i * self.dk + kk] += acc;
                }
            }

            // ks_entry[t,d] = Σ_kk k[t,kk] · S[kk,d]
            // dk[t,kk] += Σ_d d_ks_entry[t,d] · S[kk,d]
            // dS[kk,d] += Σ_t d_ks_entry[t,d] · k[t,kk]
            for t in 0..chunk {
                for kk in 0..self.dk {
                    let mut acc = 0.0_f32;
                    for d in 0..self.dv {
                        acc +=
                            d_ks_entry[cv_base + t * self.dv + d] * s_data[s_base + kk * self.dv + d];
                    }
                    dk_accum[k_base + t * self.dk + kk] += acc;
                }
            }
            for kk in 0..self.dk {
                for d in 0..self.dv {
                    let mut acc = 0.0_f32;
                    for t in 0..chunk {
                        acc +=
                            d_ks_entry[cv_base + t * self.dv + d] * k_data[k_base + t * self.dk + kk];
                    }
                    ds_in[s_base + kk * self.dv + d] += acc;
                }
            }

            // q_s[t,d] = Σ_kk q[t,kk] · S[kk,d]
            // dq[t,kk] += Σ_d d_q_s[t,d] · S[kk,d]
            // dS[kk,d] += Σ_t d_q_s[t,d] · q[t,kk]
            for t in 0..chunk {
                for kk in 0..self.dk {
                    let mut acc = 0.0_f32;
                    for d in 0..self.dv {
                        acc += d_q_s[cv_base + t * self.dv + d] * s_data[s_base + kk * self.dv + d];
                    }
                    dq_accum[q_base + t * self.dk + kk] += acc;
                }
            }
            for kk in 0..self.dk {
                for d in 0..self.dv {
                    let mut acc = 0.0_f32;
                    for t in 0..chunk {
                        acc += d_q_s[cv_base + t * self.dv + d] * q_data[q_base + t * self.dk + kk];
                    }
                    ds_in[s_base + kk * self.dv + d] += acc;
                }
            }
        }
    }
}

impl VkBackwardOp for GdnChunkwiseBackward {
    fn op_name(&self) -> &'static str {
        "gdn_chunkwise"
    }
    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }
    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        let device = self.inputs[0].device();
        let dout_full = grad_out.to_vec_f32()?;
        let bh = self.batch * self.nv;

        // Output gradient buffers (full T, accumulated across chunks)
        let mut dq = vec![0.0_f32; self.batch * self.nv * self.seq_len * self.dk];
        let mut dk = vec![0.0_f32; self.batch * self.nv * self.seq_len * self.dk];
        let mut dv_buf = vec![0.0_f32; self.batch * self.nv * self.seq_len * self.dv];
        let mut dbeta = vec![0.0_f32; self.batch * self.nv * self.seq_len];
        let mut dg = vec![0.0_f32; self.batch * self.nv * self.seq_len];

        // Cross-chunk dS (carried in reverse: dS_out of chunk c = dS_in of chunk c+1)
        let mut d_s_carry: Option<Vec<f32>> = None;

        for ci in (0..self.chunks.len()).rev() {
            let chunk = self.chunks[ci].chunk_len;
            let t_off = ci * self.chunk_size;

            // Slice d_out for this chunk: shape [B, nv, chunk, dv]
            let mut d_out_chunk = vec![0.0_f32; bh * chunk * self.dv];
            for bi in 0..bh {
                for tt in 0..chunk {
                    for d in 0..self.dv {
                        let src_t = t_off + tt;
                        d_out_chunk[bi * chunk * self.dv + tt * self.dv + d] =
                            dout_full[bi * self.seq_len * self.dv + src_t * self.dv + d];
                    }
                }
            }
            let d_out_t = upload_f32(device, &d_out_chunk, vec![self.batch, self.nv, chunk, self.dv])?;

            // chunk_scan_bwd → dq_s_scaled, db_mask, dW (initial)
            let (dq_s_scaled, db_mask, mut d_w) = vk_gdn_chunk_scan_bwd_no_grad(
                &d_out_t,
                &self.chunks[ci].b_mask,
                &self.chunks[ci].w,
                self.batch,
                self.nv,
                chunk,
                self.dv,
            )?;

            // Initialize per-chunk grad accumulators in CPU buffers
            let mut dq_local = vec![0.0_f32; bh * chunk * self.dk];
            let mut dk_local = vec![0.0_f32; bh * chunk * self.dk];
            let mut dv_local = vec![0.0_f32; bh * chunk * self.dv];
            let mut dbeta_local = vec![0.0_f32; bh * chunk];
            let mut dg_local = vec![0.0_f32; bh * chunk];

            // Chunk-prep grad accumulators
            let mut d_decay_last_col_acc = vec![0.0_f32; bh * chunk];
            let mut d_p_last_acc = vec![0.0_f32; bh];
            let mut d_w_acc = d_w.to_vec_f32()?;

            // state_exit_bwd if not the last chunk (i.e. dS_carry is Some)
            if let Some(ds_carry) = d_s_carry.as_ref() {
                let ds_t = upload_f32(
                    device,
                    ds_carry,
                    vec![self.batch, self.nv, self.dk, self.dv],
                )?;
                let (_d_s_in, d_w_extra, d_k_extra, d_decay_extra, d_p_last_extra) =
                    vk_gdn_state_exit_bwd_no_grad(
                        &ds_t,
                        &self.chunks[ci].decay_last_col,
                        &self.chunks[ci].k_c,
                        &self.chunks[ci].w,
                        &self.chunks[ci].s_in_snapshot,
                        &self.chunks[ci].p_last,
                        self.batch,
                        self.nv,
                        chunk,
                        self.dk,
                        self.dv,
                    )?;
                // Add extras to accumulators
                let dwe = d_w_extra.to_vec_f32()?;
                let dke = d_k_extra.to_vec_f32()?;
                let dde = d_decay_extra.to_vec_f32()?;
                let dpe = d_p_last_extra.to_vec_f32()?;
                for i in 0..d_w_acc.len() {
                    d_w_acc[i] += dwe[i];
                }
                for i in 0..dk_local.len() {
                    dk_local[i] += dke[i];
                }
                for i in 0..d_decay_last_col_acc.len() {
                    d_decay_last_col_acc[i] += dde[i];
                }
                for i in 0..d_p_last_acc.len() {
                    d_p_last_acc[i] += dpe[i];
                }
                // Note: d_s_in from state_exit (= p_last·dS_carry) goes into the
                // existing d_s_in computed below by chunk_prep_bwd. We capture
                // separately and add at the end.
            }
            let d_w_acc_t = upload_f32(
                device,
                &d_w_acc,
                vec![self.batch, self.nv, chunk, self.dv],
            )?;

            // Triangular-solve adjoint:
            //   dr = M^T \ d_w (d_w accumulated above)
            //   dV' = diag(beta) · dr      (elementwise per (t, d))
            //   dbeta = Σ_d (V'[t,d] · dr[t,d])  (collapsed across dv)
            //   dA_strict[t, i] = -beta[t] · dr[t,d] · W[i,d]   (sum across d, strict-lower)
            let dr = vk_solve_tri_transpose_no_grad(
                &self.chunks[ci].a_strict,
                &self.chunks[ci].beta_c,
                &d_w_acc_t,
                self.batch,
                self.nv,
                chunk,
                self.dv,
            )?;
            let dr_data = dr.to_vec_f32()?;
            let beta_data = self.chunks[ci].beta_c.to_vec_f32()?;
            let vp_data = self.chunks[ci].v_prime.to_vec_f32()?;
            let w_data = self.chunks[ci].w.to_vec_f32()?;
            let mut d_v_prime = vec![0.0_f32; bh * chunk * self.dv];
            let mut d_a_strict = vec![0.0_f32; bh * chunk * chunk];
            for bi in 0..bh {
                let cc = bi * chunk * chunk;
                let cv = bi * chunk * self.dv;
                let bb = bi * chunk;
                for t in 0..chunk {
                    let beta_t = beta_data[bb + t];
                    // dV'[t,d] = beta[t] · dr[t,d]
                    for d in 0..self.dv {
                        d_v_prime[cv + t * self.dv + d] = beta_t * dr_data[cv + t * self.dv + d];
                    }
                    // dbeta[t] += Σ_d v_prime[t,d] · dr[t,d]
                    let mut acc_beta = 0.0_f32;
                    for d in 0..self.dv {
                        acc_beta += vp_data[cv + t * self.dv + d] * dr_data[cv + t * self.dv + d];
                    }
                    // dbeta also gets a contribution from W = beta · (V' - A·W), but W's
                    // dependence on β has already been captured via dr (since dr = ∂L/∂(βV')).
                    // Net: dbeta[t] = Σ_d V'[t,d] · dr[t,d]  +  contribution from
                    //                 dA_strict back into β:
                    //   ∂W/∂β_t at the diagonal: W[t] = β·(V' - Σ A·W)
                    //                  ≡ ∂L/∂β_t · (V'[t] - Σ_{i<t} A[t,i]·W[i])
                    //   That's just ∂L/∂(β·something) absorbed by dr already.
                    dbeta_local[bb + t] += acc_beta;
                    // dA_strict[t,i] for i<t: -beta[t] · dr[t,d] · W[i,d] (sum over d)
                    for i in 0..t {
                        let mut acc = 0.0_f32;
                        for d in 0..self.dv {
                            acc += dr_data[cv + t * self.dv + d] * w_data[cv + i * self.dv + d];
                        }
                        d_a_strict[cc + t * chunk + i] += -beta_t * acc;
                    }
                }
            }
            let d_v_prime_t = upload_f32(
                device,
                &d_v_prime,
                vec![self.batch, self.nv, chunk, self.dv],
            )?;
            let d_a_strict_t = upload_f32(
                device,
                &d_a_strict,
                vec![self.batch, self.nv, chunk, chunk],
            )?;
            let d_decay_last_col_t = upload_f32(
                device,
                &d_decay_last_col_acc,
                vec![self.batch, self.nv, chunk],
            )?;
            let d_p_last_t = upload_f32(device, &d_p_last_acc, vec![self.batch, self.nv])?;

            // chunk_prep_bwd: produces d_g, d_v, d_kkt, d_qkt, d_ks_entry, d_q_s
            let (dg_chunk, dv_chunk, d_kkt, d_qkt, d_ks_entry, d_q_s) = vk_gdn_chunk_prep_bwd_no_grad(
                &d_a_strict_t,
                &db_mask,
                &d_v_prime_t,
                &dq_s_scaled,
                &d_decay_last_col_t,
                &d_p_last_t,
                &self.chunks[ci].g_c,
                &self.chunks[ci].v_c,
                &self.chunks[ci].kkt,
                &self.chunks[ci].qkt,
                &self.chunks[ci].ks_entry,
                &self.chunks[ci].q_s,
                self.batch,
                self.nv,
                chunk,
                self.dv,
            )?;
            // dv_local = dv_chunk
            let dv_chunk_data = dv_chunk.to_vec_f32()?;
            for i in 0..dv_local.len() {
                dv_local[i] += dv_chunk_data[i];
            }
            // dg_local = dg_chunk
            let dg_chunk_data = dg_chunk.to_vec_f32()?;
            for i in 0..dg_local.len() {
                dg_local[i] += dg_chunk_data[i];
            }

            // matmul_prep_bwd: routes d_kkt, d_qkt, d_ks_entry, d_q_s into dq, dk, dS_in
            let mut ds_in_local = vec![0.0_f32; bh * self.dk * self.dv];
            let d_kkt_data = d_kkt.to_vec_f32()?;
            let d_qkt_data = d_qkt.to_vec_f32()?;
            let d_ks_data = d_ks_entry.to_vec_f32()?;
            let d_qs_data = d_q_s.to_vec_f32()?;
            self.matmul_prep_bwd(
                ci,
                &d_kkt_data,
                &d_qkt_data,
                &d_ks_data,
                &d_qs_data,
                &mut dq_local,
                &mut dk_local,
                &mut ds_in_local,
            );

            // Add dS_in from state_exit (p_last · dS_carry)
            if let Some(ref ds_carry) = d_s_carry {
                let pl_data = self.chunks[ci].p_last.to_vec_f32()?;
                for bi in 0..bh {
                    let p = pl_data[bi];
                    let s_base = bi * self.dk * self.dv;
                    for ix in 0..self.dk * self.dv {
                        ds_in_local[s_base + ix] += p * ds_carry[s_base + ix];
                    }
                }
            }

            // Carry for next iteration (going to chunk ci-1):
            // ds_in for this chunk becomes ds_carry for chunk ci-1
            d_s_carry = Some(ds_in_local);

            // Splat per-chunk grads into full-T accumulators
            for bi in 0..bh {
                let q_base = bi * chunk * self.dk;
                let v_base = bi * chunk * self.dv;
                let bb = bi * chunk;
                for tt in 0..chunk {
                    let src_t = t_off + tt;
                    let q_full = (bi * self.seq_len + src_t) * self.dk;
                    let v_full = (bi * self.seq_len + src_t) * self.dv;
                    for kk in 0..self.dk {
                        dq[q_full + kk] += dq_local[q_base + tt * self.dk + kk];
                        dk[q_full + kk] += dk_local[q_base + tt * self.dk + kk];
                    }
                    for d in 0..self.dv {
                        dv_buf[v_full + d] += dv_local[v_base + tt * self.dv + d];
                    }
                    dbeta[bi * self.seq_len + src_t] += dbeta_local[bb + tt];
                    dg[bi * self.seq_len + src_t] += dg_local[bb + tt];
                }
            }
        }

        // The first chunk's dS_in is dropped (initial state is not trained).
        let dq_t = upload_f32(
            device,
            &dq,
            vec![self.batch, self.nv, self.seq_len, self.dk],
        )?;
        let dk_t = upload_f32(
            device,
            &dk,
            vec![self.batch, self.nv, self.seq_len, self.dk],
        )?;
        let dv_t = upload_f32(
            device,
            &dv_buf,
            vec![self.batch, self.nv, self.seq_len, self.dv],
        )?;
        let dbeta_t = upload_f32(device, &dbeta, vec![self.batch, self.nv, self.seq_len])?;
        let dg_t = upload_f32(device, &dg, vec![self.batch, self.nv, self.seq_len])?;

        Ok(vec![Some(dq_t), Some(dk_t), Some(dv_t), Some(dbeta_t), Some(dg_t)])
    }
}

/// Autograd-aware vk_gdn_chunkwise: forward + saves intermediates for
/// the GdnChunkwiseBackward op. Output VkTensor carries `requires_grad`
/// when any of (q, k, v, beta, g) does.
pub fn vk_gdn_chunkwise(
    q: &VkTensor,
    k: &VkTensor,
    v: &VkTensor,
    beta: &VkTensor,
    g: &VkTensor,
    state: &mut VkTensor,
    chunk_size: usize,
) -> Result<VkTensor> {
    let dims = q.shape();
    anyhow::ensure!(dims.len() == 4, "vk_gdn_chunkwise: q rank-4");
    let batch = dims[0];
    let nv = dims[1];
    let seq_len = dims[2];
    let dk = dims[3];
    let dv = v.shape()[3];

    let needs_grad = q.requires_grad()
        || k.requires_grad()
        || v.requires_grad()
        || beta.requires_grad()
        || g.requires_grad();

    let full_chunks = seq_len / chunk_size;
    let tail = seq_len - full_chunks * chunk_size;
    let total_chunks = full_chunks + if tail > 0 { 1 } else { 0 };

    let mut out_chunks: Vec<VkTensor> = Vec::with_capacity(total_chunks);
    let mut saved: Vec<ChunkSaved> = Vec::with_capacity(total_chunks);

    for ci in 0..total_chunks {
        let is_tail = ci >= full_chunks;
        let c = if is_tail { tail } else { chunk_size };
        let t_start = ci * chunk_size;

        let q_c = time_narrow_no_grad(q, t_start, c, dk)?;
        let k_c = time_narrow_no_grad(k, t_start, c, dk)?;
        let v_c = time_narrow_no_grad(v, t_start, c, dv)?;
        let beta_c = time_narrow_3d_no_grad(beta, t_start, c)?;
        let g_c = time_narrow_3d_no_grad(g, t_start, c)?;

        // Snapshot S_in BEFORE update for this chunk's backward
        let s_in_snap = {
            let s_data = state.to_vec_f32()?;
            upload_f32(state.device(), &s_data, state.shape().to_vec())?
        };

        // Compute kkt/qkt/ks_entry/q_s separately so we can save them
        let bh = batch * nv;
        let q_3 = vk_reshape(&q_c, &[bh, c, dk])?;
        let k_3 = vk_reshape(&k_c, &[bh, c, dk])?;
        let v_3 = vk_reshape(&v_c, &[bh, c, dv])?;
        let s_3 = vk_reshape(state, &[bh, dk, dv])?;
        let k_t = vk_transpose_batched_2d_no_grad(&k_3)?;
        let ks_entry = vk_matmul_batched_no_grad(&k_3, &s_3)?;
        let q_s = vk_matmul_batched_no_grad(&q_3, &s_3)?;
        let kkt = vk_matmul_batched_no_grad(&k_3, &k_t)?;
        let qkt = vk_matmul_batched_no_grad(&q_3, &k_t)?;

        let v_4 = vk_reshape(&v_3, &[batch, nv, c, dv])?;
        let kkt_4 = vk_reshape(&kkt, &[batch, nv, c, c])?;
        let qkt_4 = vk_reshape(&qkt, &[batch, nv, c, c])?;
        let ks_entry_4 = vk_reshape(&ks_entry, &[batch, nv, c, dv])?;
        let q_s_4 = vk_reshape(&q_s, &[batch, nv, c, dv])?;

        let prep = vk_gdn_chunk_prep_no_grad(
            &g_c, &v_4, &kkt_4, &qkt_4, &ks_entry_4, &q_s_4, batch, nv, c, dv,
        )?;
        let w_4 = vk_solve_tri_no_grad(&prep.a_strict, &prep.v_prime, &beta_c, batch, nv, c, dv)?;

        // out_chunk = q_s_scaled + b_mask @ W
        let bmask_3 = vk_reshape(&prep.b_mask, &[bh, c, c])?;
        let w_3 = vk_reshape(&w_4, &[bh, c, dv])?;
        let intra_3 = vk_matmul_batched_no_grad(&bmask_3, &w_3)?;
        let intra_4 = vk_reshape(&intra_3, &[batch, nv, c, dv])?;
        let out_chunk = vk_add_no_grad(&prep.q_s_scaled, &intra_4)?;
        out_chunks.push(out_chunk);

        // State update for next chunk: S = p_last · S + k^T · (decay·W)
        let decay_dlc_4 = vk_reshape(&prep.decay_last_col, &[batch, nv, c, 1])?;
        let w_weighted_4 = vk_broadcast_mul_lastdim_no_grad(&w_4, &decay_dlc_4, dv)?;
        let new_state = state_update(state, &prep.p_last, &k_t, &w_weighted_4, batch, nv, dk, dv, c)?;

        saved.push(ChunkSaved {
            a_strict: prep.a_strict.clone(),
            b_mask: prep.b_mask.clone(),
            v_prime: prep.v_prime.clone(),
            q_s_scaled: prep.q_s_scaled.clone(),
            decay_last_col: prep.decay_last_col.clone(),
            p_last: prep.p_last.clone(),
            w: w_4.clone(),
            s_in_snapshot: s_in_snap,
            q_c: q_c.clone(),
            k_c: k_c.clone(),
            beta_c: beta_c.clone(),
            g_c: g_c.clone(),
            kkt: kkt_4.clone(),
            qkt: qkt_4.clone(),
            ks_entry: ks_entry_4.clone(),
            q_s: q_s_4.clone(),
            v_c: v_4.clone(),
            chunk_len: c,
        });

        *state = new_state;
    }

    let out = concat_time(&out_chunks, dv)?;

    if !needs_grad {
        return Ok(out);
    }

    let grad_fn: Arc<dyn VkBackwardOp> = Arc::new(GdnChunkwiseBackward {
        chunks: saved,
        inputs: [q.clone(), k.clone(), v.clone(), beta.clone(), g.clone()],
        batch,
        nv,
        seq_len,
        dk,
        dv,
        chunk_size,
    });
    Ok(VkTensor::from_op(
        Arc::clone(out.buffer()),
        out.shape().to_vec(),
        out.dtype(),
        Arc::clone(out.device()),
        Some(grad_fn),
    ))
}

#[allow(dead_code)]
fn _silence_unused() {
    let _: fn(&Arc<VulkanDevice>, usize) -> Result<Arc<VulkanBuffer>> = alloc_zeroed_f32;
    let _: fn(&Arc<VulkanDevice>, &[f32], Vec<usize>) -> Result<VkTensor> = upload_f32;
}

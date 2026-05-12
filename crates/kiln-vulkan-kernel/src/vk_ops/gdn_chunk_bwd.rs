//! CPU backward kernels for the GDN chunkwise stages.
//!
//! Phase 4 v1: all functions are CPU implementations driven by VkTensor
//! readback / upload. The math is analytic from `docs/vk_native_gdn.md`.
//! Phase 7 will replace the hot ones with GLSL shaders.
//!
//! Functions:
//!   vk_solve_tri_transpose_no_grad  — adjoint of forward sub
//!   vk_gdn_chunk_scan_bwd_no_grad   — backward of out = q_s_scaled + b_mask·W
//!   vk_gdn_state_exit_bwd_no_grad   — backward of S_new = p_last·S + k^T·(decay·W)
//!   vk_gdn_chunk_prep_bwd_no_grad   — backward of all chunk_prep outputs

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
    .context("gdn_chunk_bwd: alloc")?;
    Ok(Arc::new(buf))
}

fn upload(device: &Arc<VulkanDevice>, data: &[f32], shape: Vec<usize>) -> Result<VkTensor> {
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

/// Solve M^T · dr = dW where M = (I + diag(β) · A_strict).
///
/// Upper-triangular back-substitution. Uses the GPU shader
/// `vk_solve_tri_transpose.comp` with bounded shared memory (32 KB
/// at chunk=64, DV_PER_WG=64).
///
/// Inputs:
///   a_strict: [B, nv, C, C]  (strict-lower)
///   beta:     [B, nv, C]
///   dW:       [B, nv, C, dv]
/// Output:
///   dr:       [B, nv, C, dv]
pub fn vk_solve_tri_transpose_no_grad(
    a_strict: &VkTensor,
    beta: &VkTensor,
    dw: &VkTensor,
    batch: usize,
    nv: usize,
    chunk: usize,
    dv: usize,
) -> Result<VkTensor> {
    let device = a_strict.device();
    let out = alloc_f32(device, batch * nv * chunk * dv)?;

    if std::env::var("KILN_VK_SOLVE_TRI_TRANSPOSE_CPU").is_ok() {
        return cpu_solve_tri_transpose(a_strict, beta, dw, out, batch, nv, chunk, dv);
    }

    let dv_per_wg = 64u32;
    let dv_tiles = (dv as u32 + dv_per_wg - 1) / dv_per_wg;
    let push = [batch as u32, nv as u32, chunk as u32, dv as u32];
    crate::vk_ops::dispatch_simple_2d(
        device,
        "vk_solve_tri_transpose",
        &[
            a_strict.buffer().handle(),
            beta.buffer().handle(),
            dw.buffer().handle(),
            out.handle(),
        ],
        &push,
        ((batch * nv) as u32, dv_tiles),
    )?;
    Ok(VkTensor::from_buffer(
        out,
        vec![batch, nv, chunk, dv],
        VkDType::F32,
        Arc::clone(device),
    ))
}

/// CPU fallback for vk_solve_tri_transpose, accessible via
/// KILN_VK_SOLVE_TRI_TRANSPOSE_CPU=1 for debugging.
fn cpu_solve_tri_transpose(
    a_strict: &VkTensor,
    beta: &VkTensor,
    dw: &VkTensor,
    out: Arc<VulkanBuffer>,
    batch: usize,
    nv: usize,
    chunk: usize,
    dv: usize,
) -> Result<VkTensor> {
    let device = a_strict.device();
    let a = a_strict.to_vec_f32()?;
    let b = beta.to_vec_f32()?;
    let dw_d = dw.to_vec_f32()?;
    let mut dr = vec![0.0_f32; batch * nv * chunk * dv];
    for bh in 0..batch * nv {
        let a_base = bh * chunk * chunk;
        let v_base = bh * chunk * dv;
        let b_base = bh * chunk;
        for t in (0..chunk).rev() {
            for d in 0..dv {
                let mut acc = 0.0_f32;
                for i in (t + 1)..chunk {
                    acc += b[b_base + i] * a[a_base + i * chunk + t] * dr[v_base + i * dv + d];
                }
                dr[v_base + t * dv + d] = dw_d[v_base + t * dv + d] - acc;
            }
        }
    }
    let raw: Vec<u8> = dr.iter().flat_map(|f| f.to_le_bytes()).collect();
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
        vec![batch, nv, chunk, dv],
        VkDType::F32,
        Arc::clone(device),
    ))
}

/// Backward of the chunk_scan stage:
///   out = q_s_scaled + b_mask · W
/// Given d_out, b_mask, W, produces dq_s_scaled, db_mask, dW.
///
/// GPU-native implementation using vk_matmul_batched composition.
pub fn vk_gdn_chunk_scan_bwd_no_grad(
    d_out: &VkTensor,   // [B, nv, C, dv]
    b_mask: &VkTensor,  // [B, nv, C, C]
    w: &VkTensor,       // [B, nv, C, dv]
    batch: usize,
    nv: usize,
    chunk: usize,
    dv: usize,
) -> Result<(VkTensor, VkTensor, VkTensor)> {
    use crate::vk_ops::matmul_batched::{vk_matmul_batched_no_grad, vk_transpose_batched_2d_no_grad};
    use crate::vk_ops::shape::vk_reshape;

    let device = d_out.device();
    if std::env::var("KILN_VK_GDN_CHUNK_SCAN_BWD_CPU").is_ok() {
        return cpu_chunk_scan_bwd(d_out, b_mask, w, batch, nv, chunk, dv);
    }

    // dq_s_scaled = d_out (just clone/copy — no allocation needed semantically,
    // but we produce a fresh VkTensor with the same data via add-zero or
    // re-upload).
    // Simpler: clone the buffer Arc — the result IS d_out.
    let dq_s_scaled = VkTensor::from_buffer(
        Arc::clone(d_out.buffer()),
        d_out.shape().to_vec(),
        VkDType::F32,
        Arc::clone(device),
    );

    // Reshape to 3D for vk_matmul_batched
    let bh = batch * nv;
    let dout_3 = vk_reshape(d_out, &[bh, chunk, dv])?;
    let bmask_3 = vk_reshape(b_mask, &[bh, chunk, chunk])?;
    let w_3 = vk_reshape(w, &[bh, chunk, dv])?;

    // dW = b_mask^T @ d_out, where b_mask is [C, C] (rows=t, cols=i)
    //   dW[i, d] = Σ_t b_mask[t, i] · d_out[t, d]
    //   b_mask^T: swap (t, i) → (i, t), [C, C] → [C, C]
    //   dW = transpose(b_mask) @ d_out
    let bmask_t = vk_transpose_batched_2d_no_grad(&bmask_3)?;
    let dw_3 = vk_matmul_batched_no_grad(&bmask_t, &dout_3)?;
    let d_w = vk_reshape(&dw_3, &[batch, nv, chunk, dv])?;

    // db_mask = d_out @ W^T, where W is [C, dv]
    //   db_mask[t, i] = Σ_d d_out[t, d] · W[i, d]
    //   W^T: [dv, C]
    let w_t = vk_transpose_batched_2d_no_grad(&w_3)?;
    let dbm_3 = vk_matmul_batched_no_grad(&dout_3, &w_t)?;
    let db_mask = vk_reshape(&dbm_3, &[batch, nv, chunk, chunk])?;

    Ok((dq_s_scaled, db_mask, d_w))
}

/// CPU fallback for vk_gdn_chunk_scan_bwd (debug aid).
fn cpu_chunk_scan_bwd(
    d_out: &VkTensor,
    b_mask: &VkTensor,
    w: &VkTensor,
    batch: usize,
    nv: usize,
    chunk: usize,
    dv: usize,
) -> Result<(VkTensor, VkTensor, VkTensor)> {
    let device = d_out.device();
    let dout = d_out.to_vec_f32()?;
    let bm = b_mask.to_vec_f32()?;
    let wd = w.to_vec_f32()?;
    let mut dq_s_scaled = vec![0.0_f32; batch * nv * chunk * dv];
    let mut db_mask = vec![0.0_f32; batch * nv * chunk * chunk];
    let mut d_w = vec![0.0_f32; batch * nv * chunk * dv];
    for bh in 0..batch * nv {
        let v_base = bh * chunk * dv;
        let m_base = bh * chunk * chunk;
        for t in 0..chunk {
            for d in 0..dv {
                dq_s_scaled[v_base + t * dv + d] = dout[v_base + t * dv + d];
            }
        }
        for t in 0..chunk {
            for i in 0..chunk {
                let mut acc = 0.0_f32;
                for d in 0..dv {
                    acc += dout[v_base + t * dv + d] * wd[v_base + i * dv + d];
                }
                db_mask[m_base + t * chunk + i] = acc;
            }
        }
        for i in 0..chunk {
            for d in 0..dv {
                let mut acc = 0.0_f32;
                for t in 0..chunk {
                    acc += bm[m_base + t * chunk + i] * dout[v_base + t * dv + d];
                }
                d_w[v_base + i * dv + d] = acc;
            }
        }
    }
    Ok((
        upload(device, &dq_s_scaled, vec![batch, nv, chunk, dv])?,
        upload(device, &db_mask, vec![batch, nv, chunk, chunk])?,
        upload(device, &d_w, vec![batch, nv, chunk, dv])?,
    ))
}

/// Backward of the state-exit stage. GPU composition via existing
/// vk_matmul_batched + elementwise ops:
///
///   dS_in       = p_last · dS_exit        (scalar broadcast)
///   d_p_last    = Σ S_in[i,j] · dS_exit[i,j]
///   tmp_dW      = k @ dS_exit              [B*nv, C, dv]   (NO decay)
///   dW_extra    = tmp_dW · decay_last_col_broadcast(dv)
///   tmp_dk      = W @ dS_exit^T            [B*nv, C, dk]   (NO decay)
///   dk_extra    = tmp_dk · decay_last_col_broadcast(dk)
///   d_decay[i]  = Σ_kk k[i,kk] · tmp_dk[i,kk]
///   dG[C-1]     += d_p_last  (composed by chunkwise op)
///   dG[i]       -= d_decay[i] (composed by chunkwise op)
///
/// CPU fallback via KILN_VK_GDN_STATE_EXIT_BWD_CPU=1.
pub fn vk_gdn_state_exit_bwd_no_grad(
    d_s_exit: &VkTensor,       // [B, nv, dk, dv]
    decay_last_col: &VkTensor, // [B, nv, C]
    k_chunk: &VkTensor,        // [B, nv, C, dk]
    w: &VkTensor,              // [B, nv, C, dv]
    s_in: &VkTensor,           // [B, nv, dk, dv]
    p_last: &VkTensor,         // [B, nv]
    batch: usize,
    nv: usize,
    chunk: usize,
    dk: usize,
    dv: usize,
) -> Result<(VkTensor, VkTensor, VkTensor, VkTensor, VkTensor)> {
    if std::env::var("KILN_VK_GDN_STATE_EXIT_BWD_CPU").is_ok() {
        return cpu_state_exit_bwd(
            d_s_exit,
            decay_last_col,
            k_chunk,
            w,
            s_in,
            p_last,
            batch,
            nv,
            chunk,
            dk,
            dv,
        );
    }

    let device = d_s_exit.device();
    let d_s_in = alloc_f32(device, batch * nv * dk * dv)?;
    let d_w = alloc_f32(device, batch * nv * chunk * dv)?;
    let d_k = alloc_f32(device, batch * nv * chunk * dk)?;
    let d_decay = alloc_f32(device, batch * nv * chunk)?;
    let d_p_last = alloc_f32(device, batch * nv)?;

    let workgroups = (batch * nv) as u32;
    let push = [batch as u32, nv as u32, chunk as u32, dk as u32, dv as u32];
    crate::vk_ops::dispatch_simple(
        device,
        "vk_gdn_state_exit_bwd",
        &[
            d_s_exit.buffer().handle(),
            decay_last_col.buffer().handle(),
            k_chunk.buffer().handle(),
            w.buffer().handle(),
            s_in.buffer().handle(),
            p_last.buffer().handle(),
            d_s_in.handle(),
            d_w.handle(),
            d_k.handle(),
            d_decay.handle(),
            d_p_last.handle(),
        ],
        &push,
        workgroups,
    )?;

    Ok((
        VkTensor::from_buffer(d_s_in, vec![batch, nv, dk, dv], VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_w, vec![batch, nv, chunk, dv], VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_k, vec![batch, nv, chunk, dk], VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_decay, vec![batch, nv, chunk], VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_p_last, vec![batch, nv], VkDType::F32, Arc::clone(device)),
    ))
}

fn cpu_state_exit_bwd(
    d_s_exit: &VkTensor,
    decay_last_col: &VkTensor,
    k_chunk: &VkTensor,
    w: &VkTensor,
    s_in: &VkTensor,
    p_last: &VkTensor,
    batch: usize,
    nv: usize,
    chunk: usize,
    dk: usize,
    dv: usize,
) -> Result<(VkTensor, VkTensor, VkTensor, VkTensor, VkTensor)> {
    let device = d_s_exit.device();
    let dse = d_s_exit.to_vec_f32()?;
    let dlc = decay_last_col.to_vec_f32()?;
    let kc = k_chunk.to_vec_f32()?;
    let wd = w.to_vec_f32()?;
    let s = s_in.to_vec_f32()?;
    let pl = p_last.to_vec_f32()?;
    let mut d_s_in = vec![0.0_f32; batch * nv * dk * dv];
    let mut d_w = vec![0.0_f32; batch * nv * chunk * dv];
    let mut d_k = vec![0.0_f32; batch * nv * chunk * dk];
    let mut d_decay = vec![0.0_f32; batch * nv * chunk];
    let mut d_p_last = vec![0.0_f32; batch * nv];
    for bh in 0..batch * nv {
        let s_base = bh * dk * dv;
        let k_base = bh * chunk * dk;
        let w_base = bh * chunk * dv;
        let dlc_base = bh * chunk;
        let p = pl[bh];
        for ix in 0..dk * dv {
            d_s_in[s_base + ix] += p * dse[s_base + ix];
        }
        let mut acc_dp = 0.0_f32;
        for ix in 0..dk * dv {
            acc_dp += s[s_base + ix] * dse[s_base + ix];
        }
        d_p_last[bh] += acc_dp;
        for i in 0..chunk {
            let dlc_i = dlc[dlc_base + i];
            let mut k_dot_dse = vec![0.0_f32; dv];
            for d in 0..dv {
                let mut acc = 0.0_f32;
                for kk in 0..dk {
                    acc += kc[k_base + i * dk + kk] * dse[s_base + kk * dv + d];
                }
                k_dot_dse[d] = acc;
            }
            for d in 0..dv {
                d_w[w_base + i * dv + d] += dlc_i * k_dot_dse[d];
            }
            for kk in 0..dk {
                let mut acc = 0.0_f32;
                for d in 0..dv {
                    acc += wd[w_base + i * dv + d] * dse[s_base + kk * dv + d];
                }
                d_k[k_base + i * dk + kk] += dlc_i * acc;
            }
            let mut acc_dec = 0.0_f32;
            for kk in 0..dk {
                for d in 0..dv {
                    acc_dec += kc[k_base + i * dk + kk]
                        * wd[w_base + i * dv + d]
                        * dse[s_base + kk * dv + d];
                }
            }
            d_decay[dlc_base + i] += acc_dec;
        }
    }
    Ok((
        upload(device, &d_s_in, vec![batch, nv, dk, dv])?,
        upload(device, &d_w, vec![batch, nv, chunk, dv])?,
        upload(device, &d_k, vec![batch, nv, chunk, dk])?,
        upload(device, &d_decay, vec![batch, nv, chunk])?,
        upload(device, &d_p_last, vec![batch, nv])?,
    ))
}

/// Backward of the chunk_prep stage. GPU shader path with bounded
/// shared memory (3 × 64 floats = 768 bytes); CPU fallback for
/// debugging.
///
/// Returns: d_g, d_v, d_kkt, d_qkt, d_ks_entry, d_q_s.
pub fn vk_gdn_chunk_prep_bwd_no_grad(
    d_a_strict: &VkTensor,
    d_b_mask: &VkTensor,
    d_v_prime: &VkTensor,
    d_q_s_scaled: &VkTensor,
    d_decay_last_col: &VkTensor,
    d_p_last: &VkTensor,
    g: &VkTensor,
    v: &VkTensor,
    kkt: &VkTensor,
    qkt: &VkTensor,
    ks_entry: &VkTensor,
    q_s: &VkTensor,
    batch: usize,
    nv: usize,
    chunk: usize,
    dv: usize,
) -> Result<(VkTensor, VkTensor, VkTensor, VkTensor, VkTensor, VkTensor)> {
    if std::env::var("KILN_VK_GDN_CHUNK_PREP_BWD_CPU").is_ok() {
        return cpu_chunk_prep_bwd(
            d_a_strict,
            d_b_mask,
            d_v_prime,
            d_q_s_scaled,
            d_decay_last_col,
            d_p_last,
            g,
            v,
            kkt,
            qkt,
            ks_entry,
            q_s,
            batch,
            nv,
            chunk,
            dv,
        );
    }
    anyhow::ensure!(chunk <= 64, "vk_gdn_chunk_prep_bwd: chunk ≤ 64 (shader cap)");

    let device = g.device();
    let bh = batch * nv;
    let d_g_buf = alloc_f32(device, bh * chunk)?;
    let d_v_buf = alloc_f32(device, bh * chunk * dv)?;
    let d_kkt_buf = alloc_f32(device, bh * chunk * chunk)?;
    let d_qkt_buf = alloc_f32(device, bh * chunk * chunk)?;
    let d_ks_buf = alloc_f32(device, bh * chunk * dv)?;
    let d_qs_buf = alloc_f32(device, bh * chunk * dv)?;

    let workgroups = bh as u32;
    let push = [bh as u32, chunk as u32, dv as u32];
    crate::vk_ops::dispatch_simple(
        device,
        "vk_gdn_chunk_prep_bwd",
        &[
            d_a_strict.buffer().handle(),
            d_b_mask.buffer().handle(),
            d_v_prime.buffer().handle(),
            d_q_s_scaled.buffer().handle(),
            d_decay_last_col.buffer().handle(),
            d_p_last.buffer().handle(),
            g.buffer().handle(),
            v.buffer().handle(),
            kkt.buffer().handle(),
            qkt.buffer().handle(),
            ks_entry.buffer().handle(),
            q_s.buffer().handle(),
            d_g_buf.handle(),
            d_v_buf.handle(),
            d_kkt_buf.handle(),
            d_qkt_buf.handle(),
            d_ks_buf.handle(),
            d_qs_buf.handle(),
        ],
        &push,
        workgroups,
    )?;

    Ok((
        VkTensor::from_buffer(d_g_buf, vec![batch, nv, chunk], VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_v_buf, vec![batch, nv, chunk, dv], VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_kkt_buf, vec![batch, nv, chunk, chunk], VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_qkt_buf, vec![batch, nv, chunk, chunk], VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_ks_buf, vec![batch, nv, chunk, dv], VkDType::F32, Arc::clone(device)),
        VkTensor::from_buffer(d_qs_buf, vec![batch, nv, chunk, dv], VkDType::F32, Arc::clone(device)),
    ))
}

fn cpu_chunk_prep_bwd(
    d_a_strict: &VkTensor,      // [B, nv, C, C]
    d_b_mask: &VkTensor,        // [B, nv, C, C]
    d_v_prime: &VkTensor,       // [B, nv, C, dv]
    d_q_s_scaled: &VkTensor,    // [B, nv, C, dv]
    d_decay_last_col: &VkTensor, // [B, nv, C]
    d_p_last: &VkTensor,        // [B, nv]
    g: &VkTensor,               // [B, nv, C]
    v: &VkTensor,               // [B, nv, C, dv]
    kkt: &VkTensor,             // [B, nv, C, C]
    qkt: &VkTensor,             // [B, nv, C, C]
    ks_entry: &VkTensor,        // [B, nv, C, dv]
    q_s: &VkTensor,             // [B, nv, C, dv]
    batch: usize,
    nv: usize,
    chunk: usize,
    dv: usize,
) -> Result<(VkTensor, VkTensor, VkTensor, VkTensor, VkTensor, VkTensor)> {
    let device = g.device();
    // Read inputs
    let das = d_a_strict.to_vec_f32()?;
    let dbm = d_b_mask.to_vec_f32()?;
    let dvp = d_v_prime.to_vec_f32()?;
    let dqss = d_q_s_scaled.to_vec_f32()?;
    let ddec = d_decay_last_col.to_vec_f32()?;
    let dpl = d_p_last.to_vec_f32()?;
    let gd = g.to_vec_f32()?;
    let vd = v.to_vec_f32()?;
    let kkd = kkt.to_vec_f32()?;
    let qkd = qkt.to_vec_f32()?;
    let kse = ks_entry.to_vec_f32()?;
    let qsd = q_s.to_vec_f32()?;

    let mut d_g = vec![0.0_f32; batch * nv * chunk]; // wrt G[t] then reverse-cumsum→g
    let mut d_v = vec![0.0_f32; batch * nv * chunk * dv];
    let mut d_kkt = vec![0.0_f32; batch * nv * chunk * chunk];
    let mut d_qkt = vec![0.0_f32; batch * nv * chunk * chunk];
    let mut d_ks_entry = vec![0.0_f32; batch * nv * chunk * dv];
    let mut d_q_s = vec![0.0_f32; batch * nv * chunk * dv];

    for bh in 0..batch * nv {
        let cv_base = bh * chunk * dv;
        let cc_base = bh * chunk * chunk;
        let c_base = bh * chunk;
        // Compute G[t] = cumsum(g)[t]
        let mut big_g = vec![0.0_f32; chunk];
        let mut acc = 0.0_f32;
        for t in 0..chunk {
            acc += gd[c_base + t];
            big_g[t] = acc;
        }
        let p: Vec<f32> = big_g.iter().map(|x| x.exp()).collect();
        // big_g_last = big_g[C-1], p_last = p[C-1]
        let g_last = big_g[chunk - 1];

        // ---- v_prime branch:  v_prime[t,d] = v[t,d] - p[t] · ks_entry[t,d]
        for t in 0..chunk {
            for d in 0..dv {
                let dv_v = dvp[cv_base + t * dv + d];
                d_v[cv_base + t * dv + d] += dv_v;
                d_ks_entry[cv_base + t * dv + d] += -p[t] * dv_v;
                // d/d p[t] component: -ks_entry[t,d] · dv_v  ⇒ contributes to d_g via dG[t]
                // dG[t] += -ks_entry[t,d] · p[t] · dv_v   (since d p[t] / d G[t] = p[t])
                d_g[c_base + t] += -kse[cv_base + t * dv + d] * p[t] * dv_v;
            }
        }

        // ---- q_s_scaled branch: q_s_scaled[t,d] = p[t] · q_s[t,d]
        for t in 0..chunk {
            for d in 0..dv {
                let dqs_v = dqss[cv_base + t * dv + d];
                d_q_s[cv_base + t * dv + d] += p[t] * dqs_v;
                d_g[c_base + t] += qsd[cv_base + t * dv + d] * p[t] * dqs_v;
            }
        }

        // ---- a_strict branch: a_strict[t,i] = exp(G[t]-G[i]) · kkt[t,i] for i<t else 0
        for t in 0..chunk {
            for i in 0..t {
                let decay = (big_g[t] - big_g[i]).exp();
                let das_v = das[cc_base + t * chunk + i];
                d_kkt[cc_base + t * chunk + i] += decay * das_v;
                // d/d G[t] = +decay · kkt; d/d G[i] = -decay · kkt
                let term = decay * kkd[cc_base + t * chunk + i] * das_v;
                d_g[c_base + t] += term;
                d_g[c_base + i] -= term;
            }
        }

        // ---- b_mask branch: b_mask[t,i] = exp(G[t]-G[i]) · qkt[t,i] for i ≤ t
        for t in 0..chunk {
            for i in 0..=t {
                let decay = (big_g[t] - big_g[i]).exp();
                let dbm_v = dbm[cc_base + t * chunk + i];
                d_qkt[cc_base + t * chunk + i] += decay * dbm_v;
                let term = decay * qkd[cc_base + t * chunk + i] * dbm_v;
                d_g[c_base + t] += term;
                d_g[c_base + i] -= term;
            }
        }

        // ---- decay_last_col branch: decay_last_col[i] = exp(G[C-1] - G[i])
        for i in 0..chunk {
            let decay = (g_last - big_g[i]).exp();
            let ddec_v = ddec[c_base + i];
            // d/dG[C-1] = +decay · ddec_v
            d_g[c_base + chunk - 1] += decay * ddec_v;
            // d/dG[i] = -decay · ddec_v  (skip when i == C-1: cancels)
            d_g[c_base + i] -= decay * ddec_v;
        }

        // ---- p_last branch: p_last = exp(G[C-1])
        let dpl_v = dpl[bh];
        d_g[c_base + chunk - 1] += p[chunk - 1] * dpl_v;

        // d_g currently holds dG[t]. Reverse-cumsum to get dg[t]:
        //   dg[t] = Σ_{s≥t} dG[s]
        // (since G[t] = Σ_{s≤t} g[s], the gradient w.r.t. g[t] is the
        // sum of dG[s] for s ≥ t — same identity used everywhere.)
        let mut acc_g = 0.0_f32;
        let mut tmp = vec![0.0_f32; chunk];
        for t in (0..chunk).rev() {
            acc_g += d_g[c_base + t];
            tmp[t] = acc_g;
        }
        for t in 0..chunk {
            d_g[c_base + t] = tmp[t];
        }
    }

    Ok((
        upload(device, &d_g, vec![batch, nv, chunk])?,
        upload(device, &d_v, vec![batch, nv, chunk, dv])?,
        upload(device, &d_kkt, vec![batch, nv, chunk, chunk])?,
        upload(device, &d_qkt, vec![batch, nv, chunk, chunk])?,
        upload(device, &d_ks_entry, vec![batch, nv, chunk, dv])?,
        upload(device, &d_q_s, vec![batch, nv, chunk, dv])?,
    ))
}

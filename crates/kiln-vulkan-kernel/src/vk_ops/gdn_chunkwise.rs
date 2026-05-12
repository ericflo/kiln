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
    crate::buffer_pool::pool_alloc_f32(device, n)
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

/// Concatenate along time axis (dim=2 of [B, nv, T, last_axis]) on the
/// GPU. Pre-allocates one [B, nv, T, last_axis] output and scatters
/// each chunk into its time slice via `vk_scatter_to_lastdim_slice_inplace`.
/// No CPU readbacks (was previously per-chunk to_vec_f32 + reupload).
fn concat_time(chunks: &[VkTensor], last_axis: usize) -> Result<VkTensor> {
    anyhow::ensure!(!chunks.is_empty(), "concat_time: empty input");
    let dims = chunks[0].shape();
    debug_assert!(dims.len() == 4 && dims[3] == last_axis);
    let batch = dims[0];
    let nv = dims[1];
    let total_t: usize = chunks.iter().map(|c| c.shape()[2]).sum();
    let bh = batch * nv;
    let out_n = bh * total_t * last_axis;
    let device = chunks[0].device();
    let out_buf = alloc_f32(device, out_n)?;
    let out_full = VkTensor::from_buffer(
        out_buf,
        vec![batch, nv, total_t, last_axis],
        VkDType::F32,
        Arc::clone(device),
    );
    // View as [bh, total_t * last_axis] so a chunk slice is contiguous.
    let dst_view = vk_reshape(&out_full, &[bh, total_t * last_axis])?;
    let mut t_off = 0usize;
    for chunk in chunks {
        let c_t = chunk.shape()[2];
        let chunk_view = vk_reshape(chunk, &[bh, c_t * last_axis])?;
        crate::vk_ops::narrow::vk_scatter_to_lastdim_slice_inplace(
            &dst_view,
            &chunk_view,
            t_off * last_axis,
            c_t * last_axis,
        )?;
        t_off += c_t;
    }
    Ok(out_full)
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

/// Upload a constant-filled F32 VkTensor.
fn upload_f32_full(device: &Arc<VulkanDevice>, value: f32, shape: &[usize]) -> Result<VkTensor> {
    let n: usize = shape.iter().product();
    let data = vec![value; n];
    upload_f32(device, &data, shape.to_vec())
}

/// Zero out the upper-triangular (j >= t) entries of a [B, nv, C, C]
/// tensor on GPU. CPU pattern: build a [C, C] mask once, broadcast-mul.
/// This impl uploads the mask once and reuses via vk_mul.
fn strict_lower_mask_4d(t: &VkTensor, chunk: usize) -> Result<VkTensor> {
    let dims = t.shape();
    debug_assert_eq!(dims.len(), 4);
    debug_assert_eq!(dims[2], chunk);
    debug_assert_eq!(dims[3], chunk);
    let device = t.device();
    let mut mask = vec![0.0_f32; chunk * chunk];
    for i in 0..chunk {
        for j in 0..i {
            mask[i * chunk + j] = 1.0;
        }
    }
    let mask_2d = upload_f32(device, &mask, vec![chunk, chunk])?;
    // broadcast to [B, nv, C, C] via reshape + mul. mask_2d is [C, C].
    // vk_mul_no_grad requires same shape; reshape t to [B*nv*C, C] and
    // broadcast mask via the same shape (it doesn't broadcast — we
    // need to repeat the mask rows). Simpler: just mul with reshape.
    let bh_c = dims[0] * dims[1] * dims[2];
    // Replicate the mask to [bh_c, C] by treating it as a tile. The
    // pattern we want: mask[r * C + j] = mask_2d[r % C, j]. Quickest:
    // build the full mask buffer once.
    let mut full = vec![0.0_f32; dims[0] * dims[1] * chunk * chunk];
    for i in 0..(dims[0] * dims[1]) {
        let off = i * chunk * chunk;
        for j in 0..(chunk * chunk) {
            full[off + j] = mask[j];
        }
    }
    let mask_full = upload_f32(device, &full, dims.to_vec())?;
    let _ = mask_2d;
    crate::vk_ops::elementwise::vk_mul_no_grad(t, &mask_full)
}

/// Zero-init upload helper.
fn upload_f32_zeros(
    device: &Arc<VulkanDevice>,
    n: usize,
    shape: Vec<usize>,
) -> Result<VkTensor> {
    let data = vec![0.0_f32; n];
    upload_f32(device, &data, shape)
}

impl GdnChunkwiseBackward {
    /// GPU adjoint of the matmul-prep stage:
    ///   kkt      = k_c · k_c^T          shape [B, nv, C, C]
    ///   qkt      = q_c · k_c^T          shape [B, nv, C, C]
    ///   ks_entry = k_c · S_in           shape [B, nv, C, dv]
    ///   q_s      = q_c · S_in           shape [B, nv, C, dv]
    ///
    /// Returns (dq, dk, dS_in) as VkTensors with shapes
    /// [B, nv, C, dk], [B, nv, C, dk], [B, nv, dk, dv] — all GPU.
    /// Replaces the prior CPU loops + readbacks (3 × to_vec_f32 +
    /// O(B·nv·C²·dk + B·nv·C·dk·dv) per chunk).
    #[allow(clippy::too_many_arguments)]
    fn matmul_prep_bwd_gpu(
        &self,
        ci: usize,
        d_kkt: &VkTensor,      // [B, nv, C, C]
        d_qkt: &VkTensor,      // [B, nv, C, C]
        d_ks_entry: &VkTensor, // [B, nv, C, dv]
        d_q_s: &VkTensor,      // [B, nv, C, dv]
    ) -> Result<(VkTensor, VkTensor, VkTensor)> {
        let chunk = self.chunks[ci].chunk_len;
        let bh = self.batch * self.nv;
        let dk = self.dk;
        let dv = self.dv;
        let q = vk_reshape(&self.chunks[ci].q_c, &[bh, chunk, dk])?; // [bh, C, dk]
        let k = vk_reshape(&self.chunks[ci].k_c, &[bh, chunk, dk])?;
        let s = vk_reshape(&self.chunks[ci].s_in_snapshot, &[bh, dk, dv])?;
        let kkt3 = vk_reshape(d_kkt, &[bh, chunk, chunk])?;
        let qkt3 = vk_reshape(d_qkt, &[bh, chunk, chunk])?;
        let ks3 = vk_reshape(d_ks_entry, &[bh, chunk, dv])?;
        let qs3 = vk_reshape(d_q_s, &[bh, chunk, dv])?;
        let q_t = vk_transpose_batched_2d_no_grad(&q)?; // [bh, dk, C]
        let k_t = vk_transpose_batched_2d_no_grad(&k)?;
        let s_t = vk_transpose_batched_2d_no_grad(&s)?; // [bh, dv, dk]
        let kkt_t = vk_transpose_batched_2d_no_grad(&kkt3)?;
        let qkt_t = vk_transpose_batched_2d_no_grad(&qkt3)?;
        let ks_t = vk_transpose_batched_2d_no_grad(&ks3)?; // [bh, dv, C]
        let qs_t = vk_transpose_batched_2d_no_grad(&qs3)?;

        // dk contributions: from kkt (2x), qkt (1x via transpose), ks_entry
        let dk_kkt_a = vk_matmul_batched_no_grad(&kkt3, &k)?; // [bh, C, dk]
        let dk_kkt_b = vk_matmul_batched_no_grad(&kkt_t, &k)?;
        let dk_qkt = vk_matmul_batched_no_grad(&qkt_t, &q)?;
        let dk_ks = vk_matmul_batched_no_grad(&ks3, &s_t)?; // [bh, C, dk]
        let dk1 = vk_add_no_grad(&dk_kkt_a, &dk_kkt_b)?;
        let dk2 = vk_add_no_grad(&dk1, &dk_qkt)?;
        let dk_sum = vk_add_no_grad(&dk2, &dk_ks)?;
        let dk_out = vk_reshape(&dk_sum, &[self.batch, self.nv, chunk, dk])?;

        // dq contributions: from qkt (d_qkt @ k) and q_s (d_q_s @ S^T)
        let dq_qkt = vk_matmul_batched_no_grad(&qkt3, &k)?;
        let dq_qs = vk_matmul_batched_no_grad(&qs3, &s_t)?;
        let dq_sum = vk_add_no_grad(&dq_qkt, &dq_qs)?;
        let dq_out = vk_reshape(&dq_sum, &[self.batch, self.nv, chunk, dk])?;

        // dS contributions: from ks_entry (k^T @ d_ks_entry) and q_s (q^T @ d_q_s)
        let ds_ks = vk_matmul_batched_no_grad(&k_t, &ks3)?; // [bh, dk, dv]
        let ds_qs = vk_matmul_batched_no_grad(&q_t, &qs3)?;
        let ds_sum = vk_add_no_grad(&ds_ks, &ds_qs)?;
        let ds_out = vk_reshape(&ds_sum, &[self.batch, self.nv, dk, dv])?;
        // suppress unused
        let _ = ks_t;
        let _ = qs_t;

        Ok((dq_out, dk_out, ds_out))
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
        let bh = self.batch * self.nv;

        // GPU dout slicing: reshape grad_out [B, nv, T, dv] →
        // [bh, T*dv] so vk_narrow_lastdim can carve out chunk-sized
        // slices without a CPU readback.
        let dout_2d = vk_reshape(grad_out, &[bh, self.seq_len * self.dv])?;

        // Output gradient buffers (full T) — pre-allocated as GPU
        // tensors. Each (b, h, t) position is written by exactly one
        // chunk's scatter (chunks tile [0..T) without overlap), so no
        // zero-init is needed. Per-chunk grads are scattered into the
        // corresponding time slice on the GPU — no CPU readback.
        let dq_full = upload_f32_zeros(
            device,
            self.batch * self.nv * self.seq_len * self.dk,
            vec![self.batch, self.nv, self.seq_len, self.dk],
        )?;
        let dk_full = upload_f32_zeros(
            device,
            self.batch * self.nv * self.seq_len * self.dk,
            vec![self.batch, self.nv, self.seq_len, self.dk],
        )?;
        let dv_full = upload_f32_zeros(
            device,
            self.batch * self.nv * self.seq_len * self.dv,
            vec![self.batch, self.nv, self.seq_len, self.dv],
        )?;
        let dbeta_full = upload_f32_zeros(
            device,
            self.batch * self.nv * self.seq_len,
            vec![self.batch, self.nv, self.seq_len],
        )?;
        let dg_full = upload_f32_zeros(
            device,
            self.batch * self.nv * self.seq_len,
            vec![self.batch, self.nv, self.seq_len],
        )?;
        // Flat views for scattering: [bh, T * last] / [bh, T].
        let dq_view = vk_reshape(&dq_full, &[bh, self.seq_len * self.dk])?;
        let dk_view = vk_reshape(&dk_full, &[bh, self.seq_len * self.dk])?;
        let dv_view = vk_reshape(&dv_full, &[bh, self.seq_len * self.dv])?;
        let dbeta_view = vk_reshape(&dbeta_full, &[bh, self.seq_len])?;
        let dg_view = vk_reshape(&dg_full, &[bh, self.seq_len])?;

        // Cross-chunk dS (carried in reverse: dS_out of chunk c = dS_in of chunk c+1)
        let mut d_s_carry: Option<VkTensor> = None;

        for ci in (0..self.chunks.len()).rev() {
            let chunk = self.chunks[ci].chunk_len;
            let t_off = ci * self.chunk_size;

            // GPU slice grad_out[..., t_off..t_off+chunk, :] →
            // [B, nv, chunk, dv] without CPU touch.
            let dout_chunk_2d =
                vk_narrow_lastdim_no_grad(&dout_2d, t_off * self.dv, chunk * self.dv)?;
            let d_out_t = vk_reshape(
                &dout_chunk_2d,
                &[self.batch, self.nv, chunk, self.dv],
            )?;

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

            // Per-chunk grad outputs are produced as GPU tensors and
            // scattered into the dq_full/dk_full/... buffers below — no
            // CPU vec accumulators. (Each chunk writes a disjoint time
            // slice, so a single scatter per output is sufficient.)

            // Chunk-prep grad accumulators — kept as GPU VkTensors
            // throughout (was CPU readbacks in v1).
            let mut d_w_acc_t = d_w.clone(); // [B, nv, C, dv]
            let mut d_decay_last_col_acc_t = upload_f32_zeros(
                device,
                bh * chunk,
                vec![self.batch, self.nv, chunk],
            )?;
            let mut d_p_last_acc_t =
                upload_f32_zeros(device, bh, vec![self.batch, self.nv])?;
            // dk_state_extra_t: optional extra dk contribution from state_exit_bwd
            let mut dk_state_extra_t: Option<VkTensor> = None;

            if let Some(ds_carry_t) = d_s_carry.as_ref() {
                let (_d_s_in, d_w_extra, d_k_extra, d_decay_extra, d_p_last_extra) =
                    vk_gdn_state_exit_bwd_no_grad(
                        ds_carry_t,
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
                // GPU adds (no readbacks)
                d_w_acc_t = vk_add_no_grad(&d_w_acc_t, &d_w_extra)?;
                d_decay_last_col_acc_t =
                    vk_add_no_grad(&d_decay_last_col_acc_t, &d_decay_extra)?;
                d_p_last_acc_t = vk_add_no_grad(&d_p_last_acc_t, &d_p_last_extra)?;
                dk_state_extra_t = Some(d_k_extra);
            }

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
            // Solve_tri adjoint on GPU (replaces 4 CPU readbacks + nested loops):
            //   dV'[t,d]    = β[t] · dr[t,d]                (broadcast multiply)
            //   dβ_chunk[t] = Σ_d V'[t,d] · dr[t,d]         (reduce via matmul·ones)
            //   dA_strict[t,i<t] = -β[t] · (dr @ W^T)[t,i]  (matmul + scale + mask)
            // dV' = β · dr (β broadcasts over dv axis)
            let dr_2d = vk_reshape(&dr, &[bh * chunk, self.dv])?;
            let beta_2d = vk_reshape(&self.chunks[ci].beta_c, &[bh * chunk, 1])?;
            let d_v_prime_2d =
                vk_broadcast_mul_lastdim_no_grad(&dr_2d, &beta_2d, self.dv)?;
            let d_v_prime_t = vk_reshape(
                &d_v_prime_2d,
                &[self.batch, self.nv, chunk, self.dv],
            )?;

            // dβ_chunk[t] = Σ_d V'[t,d] · dr[t,d] — kept on GPU and
            // scattered directly into dbeta_full below.
            let prod = vk_mul_no_grad(&self.chunks[ci].v_prime, &dr)?;
            let prod_3d = vk_reshape(&prod, &[1, bh * chunk, self.dv])?;
            let ones_dv = upload_f32_full(device, 1.0_f32, &[1, self.dv, 1])?;
            let dbeta_3d = vk_matmul_batched_no_grad(&prod_3d, &ones_dv)?;
            let dbeta_chunk = vk_reshape(&dbeta_3d, &[bh, chunk])?;
            crate::vk_ops::narrow::vk_scatter_to_lastdim_slice_inplace(
                &dbeta_view,
                &dbeta_chunk,
                t_off,
                chunk,
            )?;

            // dA_strict[t,i] = -β[t] · (dr @ W^T)[t,i] for i < t else 0
            let dr_3 = vk_reshape(&dr, &[bh, chunk, self.dv])?;
            let w_3 = vk_reshape(&self.chunks[ci].w, &[bh, chunk, self.dv])?;
            let w_t = vk_transpose_batched_2d_no_grad(&w_3)?;
            let dr_w_t = vk_matmul_batched_no_grad(&dr_3, &w_t)?;
            let dr_w_t_2d = vk_reshape(&dr_w_t, &[bh * chunk, chunk])?;
            let beta_neg_2d = vk_scale_no_grad(&beta_2d, -1.0_f32)?;
            let scaled_2d =
                vk_broadcast_mul_lastdim_no_grad(&dr_w_t_2d, &beta_neg_2d, chunk)?;
            let scaled =
                vk_reshape(&scaled_2d, &[self.batch, self.nv, chunk, chunk])?;
            let d_a_strict_t = strict_lower_mask_4d(&scaled, chunk)?;
            // chunk_prep_bwd: produces d_g, d_v, d_kkt, d_qkt, d_ks_entry, d_q_s
            let (dg_chunk, dv_chunk, d_kkt, d_qkt, d_ks_entry, d_q_s) = vk_gdn_chunk_prep_bwd_no_grad(
                &d_a_strict_t,
                &db_mask,
                &d_v_prime_t,
                &dq_s_scaled,
                &d_decay_last_col_acc_t,
                &d_p_last_acc_t,
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
            // Scatter dv_chunk and dg_chunk into the full-T buffers on
            // the GPU (no readback).
            let dv_chunk_2d = vk_reshape(&dv_chunk, &[bh, chunk * self.dv])?;
            crate::vk_ops::narrow::vk_scatter_to_lastdim_slice_inplace(
                &dv_view,
                &dv_chunk_2d,
                t_off * self.dv,
                chunk * self.dv,
            )?;
            let dg_chunk_2d = vk_reshape(&dg_chunk, &[bh, chunk])?;
            crate::vk_ops::narrow::vk_scatter_to_lastdim_slice_inplace(
                &dg_view,
                &dg_chunk_2d,
                t_off,
                chunk,
            )?;

            // matmul_prep_bwd: routes d_kkt, d_qkt, d_ks_entry, d_q_s into dq, dk, dS_in
            // GPU path: 8 batched matmuls + adds, no CPU loops (was the dominant
            // per-chunk CPU cost when this was a CPU loop).
            let (dq_gpu, dk_gpu_a, ds_in_gpu) =
                self.matmul_prep_bwd_gpu(ci, &d_kkt, &d_qkt, &d_ks_entry, &d_q_s)?;
            // Fold in state_exit's dk extra (if any) on GPU
            let dk_gpu = if let Some(extra) = dk_state_extra_t.as_ref() {
                vk_add_no_grad(&dk_gpu_a, extra)?
            } else {
                dk_gpu_a
            };
            // Compute dS_in = ds_in_gpu + p_last · dS_carry on GPU (if carry exists)
            let ds_in_gpu_total = if let Some(carry_t) = d_s_carry.as_ref() {
                let pl_2d = vk_reshape(&self.chunks[ci].p_last, &[bh, 1])?;
                let carry_2d = vk_reshape(carry_t, &[bh, self.dk * self.dv])?;
                let scaled_2d = vk_broadcast_mul_lastdim_no_grad(
                    &carry_2d, &pl_2d, self.dk * self.dv,
                )?;
                let scaled = vk_reshape(
                    &scaled_2d,
                    &[self.batch, self.nv, self.dk, self.dv],
                )?;
                vk_add_no_grad(&ds_in_gpu, &scaled)?
            } else {
                ds_in_gpu
            };

            // Scatter dq_chunk, dk_chunk into full-T buffers on the GPU.
            let dq_chunk_2d = vk_reshape(&dq_gpu, &[bh, chunk * self.dk])?;
            crate::vk_ops::narrow::vk_scatter_to_lastdim_slice_inplace(
                &dq_view,
                &dq_chunk_2d,
                t_off * self.dk,
                chunk * self.dk,
            )?;
            let dk_chunk_2d = vk_reshape(&dk_gpu, &[bh, chunk * self.dk])?;
            crate::vk_ops::narrow::vk_scatter_to_lastdim_slice_inplace(
                &dk_view,
                &dk_chunk_2d,
                t_off * self.dk,
                chunk * self.dk,
            )?;

            // Carry for next iteration: dS_in becomes dS_exit of prior chunk.
            // Keep as GPU VkTensor.
            d_s_carry = Some(ds_in_gpu_total);
        }

        // The first chunk's dS_in is dropped (initial state is not trained).
        Ok(vec![
            Some(dq_full),
            Some(dk_full),
            Some(dv_full),
            Some(dbeta_full),
            Some(dg_full),
        ])
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

        // Snapshot S_in BEFORE update for this chunk's backward.
        // state.clone() is an Arc bump on the underlying buffer — the
        // pre-update buffer stays alive via this snapshot even after
        // `*state = new_state` swaps in a fresh buffer below. No CPU
        // readback (was a 2 MB roundtrip per chunk per layer).
        let s_in_snap = state.clone();

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

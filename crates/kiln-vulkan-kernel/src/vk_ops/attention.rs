//! GQA scaled-dot-product attention block (prefill, causal) for VkTensor.
//!
//! Composed end-to-end of autograd-tracked primitives so backward
//! flows naturally to Q, K, V inputs.
//!
//! Inputs:
//!   q: [rows, heads_q,  head_dim]
//!   k: [rows, heads_kv, head_dim]
//!   v: [rows, heads_kv, head_dim]
//! Output:
//!   out: [rows, heads_q, head_dim]
//!
//! Pipeline:
//!   1. permute_rh_to_hr → [heads_q, rows, head_dim] / [heads_kv, ...]
//!   2. repeat_kv_heads(K, groups) and (V, groups) → [heads_q, ...]
//!   3. K_t = transpose_batched(K_perm)  → [heads_q, head_dim, rows]
//!   4. scores = Q_perm @ K_t            → [heads_q, rows, rows]
//!   5. scale_inplace(scores, 1/sqrt(head_dim))
//!   6. causal_mask_inplace(scores)
//!   7. attn = softmax(scores, last_dim)
//!   8. out_perm = attn @ V_bcast        → [heads_q, rows, head_dim]
//!   9. permute_hr_to_rh(out_perm)       → [rows, heads_q, head_dim]
//!
//! For Phase C, mask + scale are applied in-place on the autograd-
//! tracked scores buffer. The mask is correct because scores entering
//! softmax with -1e30 produce ~0 attn weights (gradient ~0 too). The
//! scale is folded into Q*K.T linearly; gradients pass through scaled
//! by `scale` correctly via the matmul backward chain (since the
//! scaled scores is what flows into softmax/attn → backward dy_scale =
//! dy_score * scale, recovered by softmax+matmul backward through
//! the same scaled value).

use crate::vk_ops::mask::{vk_causal_mask_inplace, vk_scale_inplace};
use crate::vk_ops::matmul_batched::{vk_matmul_batched, vk_transpose_batched_2d};
use crate::vk_ops::permute::{vk_permute_hr_to_rh, vk_permute_rh_to_hr, vk_repeat_kv_heads};
use crate::vk_ops::shape::vk_reshape;
use crate::vk_ops::softmax::vk_softmax_lastdim;
use crate::vk_ops::{dispatch_simple_3d, for_each_1d_tile};
use crate::vk_tensor::{VkBackwardOp, VkDType, VkTensor};
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use std::sync::Arc;

fn alloc_f32(device: &Arc<VulkanDevice>, n: usize) -> Result<Arc<VulkanBuffer>> {
    crate::buffer_pool::pool_alloc_f32(device, n)
}

fn flash_rows_tile(rows: usize) -> usize {
    std::env::var("KILN_VK_FLASH_ROWS_TILE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(256)
        .clamp(1, rows.max(1))
}

fn flash_row_work_budget(rows: usize, row_tile: usize) -> usize {
    std::env::var("KILN_VK_FLASH_ROW_WORK_TILE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or_else(|| row_tile.saturating_mul(rows.max(1)))
        .max(1)
}

fn for_each_flash_query_tile(
    rows: usize,
    row_tile: usize,
    mut f: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    let budget = flash_row_work_budget(rows, row_tile);
    let mut row_start = 0usize;
    while row_start < rows {
        let denom = (row_start + 1).max(1);
        let rows_chunk = (budget / denom).clamp(1, row_tile).min(rows - row_start);
        f(row_start, rows_chunk)?;
        row_start += rows_chunk;
    }
    Ok(())
}

fn for_each_flash_key_tile(
    rows: usize,
    row_tile: usize,
    mut f: impl FnMut(usize, usize) -> Result<()>,
) -> Result<()> {
    let budget = flash_row_work_budget(rows, row_tile);
    let mut key_start = 0usize;
    while key_start < rows {
        let denom = (rows - key_start).max(1);
        let rows_chunk = (budget / denom).clamp(1, row_tile).min(rows - key_start);
        f(key_start, rows_chunk)?;
        key_start += rows_chunk;
    }
    Ok(())
}

pub fn vk_sdpa_prefill(q: &VkTensor, k: &VkTensor, v: &VkTensor, scale: f32) -> Result<VkTensor> {
    anyhow::ensure!(
        q.shape().len() == 3 && k.shape().len() == 3 && v.shape().len() == 3,
        "vk_sdpa_prefill: rank-3 inputs required"
    );
    anyhow::ensure!(
        q.dtype() == VkDType::F32 && k.dtype() == VkDType::F32 && v.dtype() == VkDType::F32,
        "vk_sdpa_prefill: F32-only"
    );
    let rows = q.shape()[0];
    let heads_q = q.shape()[1];
    let head_dim = q.shape()[2];
    let heads_kv = k.shape()[1];
    anyhow::ensure!(
        k.shape() == [rows, heads_kv, head_dim],
        "k shape {:?} mismatch [{rows}, {heads_kv}, {head_dim}]",
        k.shape()
    );
    anyhow::ensure!(
        v.shape() == [rows, heads_kv, head_dim],
        "v shape {:?} mismatch",
        v.shape()
    );
    anyhow::ensure!(
        heads_q % heads_kv == 0,
        "heads_q ({heads_q}) must be a multiple of heads_kv ({heads_kv})"
    );
    let groups = heads_q / heads_kv;

    // [rows, heads, head_dim] → [heads, rows, head_dim]
    let q_perm = vk_permute_rh_to_hr(q)?;
    let k_perm = vk_permute_rh_to_hr(k)?;
    let v_perm = vk_permute_rh_to_hr(v)?;

    // GQA broadcast
    let k_bcast = vk_repeat_kv_heads(&k_perm, groups)?;
    let v_bcast = vk_repeat_kv_heads(&v_perm, groups)?;

    // K.T per-batch: [heads_q, rows, head_dim] → [heads_q, head_dim, rows]
    let k_t = vk_transpose_batched_2d(&k_bcast)?;

    // scores = Q @ K.T
    let scores = vk_matmul_batched(&q_perm, &k_t)?;

    // scale + causal mask in place (operate on the same autograd buffer)
    vk_scale_inplace(&scores, scale)?;
    vk_causal_mask_inplace(&scores, 0)?;

    // softmax along k_len (last) dim
    let attn = vk_softmax_lastdim(&scores)?;

    // out = attn @ V_bcast
    let out_perm = vk_matmul_batched(&attn, &v_bcast)?;

    // [heads, rows, head_dim] → [rows, heads, head_dim]
    vk_permute_hr_to_rh(&out_perm)
}

/// Convenience wrapper for callers with flattened `[rows, heads*head_dim]`.
pub fn vk_sdpa_prefill_flat(
    q_flat: &VkTensor,
    k_flat: &VkTensor,
    v_flat: &VkTensor,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    scale: f32,
) -> Result<VkTensor> {
    let rows = q_flat.shape()[0];
    anyhow::ensure!(q_flat.shape() == [rows, heads_q * head_dim]);
    anyhow::ensure!(k_flat.shape() == [rows, heads_kv * head_dim]);
    anyhow::ensure!(v_flat.shape() == [rows, heads_kv * head_dim]);
    let q = vk_reshape(q_flat, &[rows, heads_q, head_dim])?;
    let k = vk_reshape(k_flat, &[rows, heads_kv, head_dim])?;
    let v = vk_reshape(v_flat, &[rows, heads_kv, head_dim])?;
    let out = vk_sdpa_prefill(&q, &k, &v, scale)?;
    vk_reshape(&out, &[rows, heads_q * head_dim])
}

fn check_flash_sdpa_flat_inputs(
    q_flat: &VkTensor,
    k_flat: &VkTensor,
    v_flat: &VkTensor,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
) -> Result<usize> {
    anyhow::ensure!(
        q_flat.dtype() == VkDType::F32
            && k_flat.dtype() == VkDType::F32
            && v_flat.dtype() == VkDType::F32,
        "vk_flash_sdpa_prefill_flat: F32-only"
    );
    anyhow::ensure!(
        q_flat.shape().len() == 2 && k_flat.shape().len() == 2 && v_flat.shape().len() == 2,
        "vk_flash_sdpa_prefill_flat: flat rank-2 inputs required"
    );
    let rows = q_flat.shape()[0];
    anyhow::ensure!(rows > 0, "vk_flash_sdpa_prefill_flat: rows > 0 required");
    anyhow::ensure!(
        heads_q > 0 && heads_kv > 0,
        "vk_flash_sdpa_prefill_flat: heads > 0"
    );
    anyhow::ensure!(
        heads_q % heads_kv == 0,
        "vk_flash_sdpa_prefill_flat: heads_q ({heads_q}) must be a multiple of heads_kv ({heads_kv})"
    );
    anyhow::ensure!(
        head_dim > 0 && head_dim <= 256,
        "vk_flash_sdpa_prefill_flat: head_dim {head_dim} exceeds shader cap 256"
    );
    anyhow::ensure!(q_flat.shape() == [rows, heads_q * head_dim]);
    anyhow::ensure!(k_flat.shape() == [rows, heads_kv * head_dim]);
    anyhow::ensure!(v_flat.shape() == [rows, heads_kv * head_dim]);
    Ok(rows)
}

/// Exact causal GQA SDPA forward without materializing `[T, T]` scores.
///
/// The returned pair is `(out, lse)`, where `out` has shape
/// `[rows, heads_q * head_dim]` and `lse` has shape `[rows, heads_q]`.
#[allow(clippy::too_many_arguments)]
pub fn vk_flash_sdpa_prefill_flat_no_grad(
    q_flat: &VkTensor,
    k_flat: &VkTensor,
    v_flat: &VkTensor,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    scale: f32,
) -> Result<(VkTensor, VkTensor)> {
    let rows = check_flash_sdpa_flat_inputs(q_flat, k_flat, v_flat, heads_q, heads_kv, head_dim)?;
    let device = q_flat.device();
    let out_buf = alloc_f32(device, rows * heads_q * head_dim)?;
    let lse_buf = alloc_f32(device, rows * heads_q)?;
    let push = [
        rows as u32,
        heads_q as u32,
        heads_kv as u32,
        head_dim as u32,
        scale.to_bits(),
    ];
    let row_tile = flash_rows_tile(rows);
    if rows <= row_tile {
        dispatch_simple_3d(
            device,
            "vk_flash_sdpa_fwd_f32",
            &[
                q_flat.buffer().handle(),
                k_flat.buffer().handle(),
                v_flat.buffer().handle(),
                out_buf.handle(),
                lse_buf.handle(),
            ],
            &push,
            (rows as u32, heads_q as u32, 1),
        )?;
    } else {
        for_each_flash_query_tile(rows, row_tile, |row_start, rows_chunk| {
            let push = [
                rows as u32,
                row_start as u32,
                rows_chunk as u32,
                heads_q as u32,
                heads_kv as u32,
                head_dim as u32,
                scale.to_bits(),
            ];
            dispatch_simple_3d(
                device,
                "vk_flash_sdpa_fwd_f32_offset",
                &[
                    q_flat.buffer().handle(),
                    k_flat.buffer().handle(),
                    v_flat.buffer().handle(),
                    out_buf.handle(),
                    lse_buf.handle(),
                ],
                &push,
                (rows_chunk as u32, heads_q as u32, 1),
            )
        })?;
    }
    Ok((
        VkTensor::from_buffer(
            out_buf,
            vec![rows, heads_q * head_dim],
            VkDType::F32,
            Arc::clone(device),
        ),
        VkTensor::from_buffer(
            lse_buf,
            vec![rows, heads_q],
            VkDType::F32,
            Arc::clone(device),
        ),
    ))
}

fn vk_flash_sdpa_delta(
    grad_out: &VkTensor,
    out: &VkTensor,
    rows: usize,
    heads_q: usize,
    head_dim: usize,
) -> Result<VkTensor> {
    let device = grad_out.device();
    let delta_buf = alloc_f32(device, rows * heads_q)?;
    let row_tile = flash_rows_tile(rows);
    if rows <= row_tile {
        let push = [rows as u32, heads_q as u32, head_dim as u32];
        dispatch_simple_3d(
            device,
            "vk_flash_sdpa_delta_f32",
            &[
                grad_out.buffer().handle(),
                out.buffer().handle(),
                delta_buf.handle(),
            ],
            &push,
            (rows as u32, heads_q as u32, 1),
        )?;
    } else {
        for_each_1d_tile(rows, row_tile, |row_start, rows_chunk| {
            let push = [
                rows as u32,
                row_start as u32,
                rows_chunk as u32,
                heads_q as u32,
                head_dim as u32,
            ];
            dispatch_simple_3d(
                device,
                "vk_flash_sdpa_delta_f32_offset",
                &[
                    grad_out.buffer().handle(),
                    out.buffer().handle(),
                    delta_buf.handle(),
                ],
                &push,
                (rows_chunk as u32, heads_q as u32, 1),
            )
        })?;
    }
    Ok(VkTensor::from_buffer(
        delta_buf,
        vec![rows, heads_q],
        VkDType::F32,
        Arc::clone(device),
    ))
}

#[allow(clippy::too_many_arguments)]
fn vk_flash_sdpa_bwd_dq(
    q: &VkTensor,
    k: &VkTensor,
    v: &VkTensor,
    grad_out: &VkTensor,
    lse: &VkTensor,
    delta: &VkTensor,
    rows: usize,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    scale: f32,
) -> Result<VkTensor> {
    let device = q.device();
    let dq_buf = alloc_f32(device, rows * heads_q * head_dim)?;
    let push = [
        rows as u32,
        heads_q as u32,
        heads_kv as u32,
        head_dim as u32,
        scale.to_bits(),
    ];
    let row_tile = flash_rows_tile(rows);
    if rows <= row_tile {
        dispatch_simple_3d(
            device,
            "vk_flash_sdpa_bwd_dq_f32",
            &[
                q.buffer().handle(),
                k.buffer().handle(),
                v.buffer().handle(),
                grad_out.buffer().handle(),
                lse.buffer().handle(),
                delta.buffer().handle(),
                dq_buf.handle(),
            ],
            &push,
            (rows as u32, heads_q as u32, 1),
        )?;
    } else {
        for_each_flash_query_tile(rows, row_tile, |row_start, rows_chunk| {
            let push = [
                rows as u32,
                row_start as u32,
                rows_chunk as u32,
                heads_q as u32,
                heads_kv as u32,
                head_dim as u32,
                scale.to_bits(),
            ];
            dispatch_simple_3d(
                device,
                "vk_flash_sdpa_bwd_dq_f32_offset",
                &[
                    q.buffer().handle(),
                    k.buffer().handle(),
                    v.buffer().handle(),
                    grad_out.buffer().handle(),
                    lse.buffer().handle(),
                    delta.buffer().handle(),
                    dq_buf.handle(),
                ],
                &push,
                (rows_chunk as u32, heads_q as u32, 1),
            )
        })?;
    }
    Ok(VkTensor::from_buffer(
        dq_buf,
        vec![rows, heads_q * head_dim],
        VkDType::F32,
        Arc::clone(device),
    ))
}

#[allow(clippy::too_many_arguments)]
fn vk_flash_sdpa_bwd_dkdv(
    q: &VkTensor,
    k: &VkTensor,
    v: &VkTensor,
    grad_out: &VkTensor,
    lse: &VkTensor,
    delta: &VkTensor,
    rows: usize,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    scale: f32,
) -> Result<(VkTensor, VkTensor)> {
    let device = q.device();
    let dk_buf = alloc_f32(device, rows * heads_kv * head_dim)?;
    let dv_buf = alloc_f32(device, rows * heads_kv * head_dim)?;
    let push = [
        rows as u32,
        heads_q as u32,
        heads_kv as u32,
        head_dim as u32,
        scale.to_bits(),
    ];
    let row_tile = flash_rows_tile(rows);
    if rows <= row_tile {
        dispatch_simple_3d(
            device,
            "vk_flash_sdpa_bwd_dkdv_f32",
            &[
                q.buffer().handle(),
                k.buffer().handle(),
                v.buffer().handle(),
                grad_out.buffer().handle(),
                lse.buffer().handle(),
                delta.buffer().handle(),
                dk_buf.handle(),
                dv_buf.handle(),
            ],
            &push,
            (rows as u32, heads_kv as u32, 1),
        )?;
    } else {
        for_each_flash_key_tile(rows, row_tile, |row_start, rows_chunk| {
            let push = [
                rows as u32,
                row_start as u32,
                rows_chunk as u32,
                heads_q as u32,
                heads_kv as u32,
                head_dim as u32,
                scale.to_bits(),
            ];
            dispatch_simple_3d(
                device,
                "vk_flash_sdpa_bwd_dkdv_f32_offset",
                &[
                    q.buffer().handle(),
                    k.buffer().handle(),
                    v.buffer().handle(),
                    grad_out.buffer().handle(),
                    lse.buffer().handle(),
                    delta.buffer().handle(),
                    dk_buf.handle(),
                    dv_buf.handle(),
                ],
                &push,
                (rows_chunk as u32, heads_kv as u32, 1),
            )
        })?;
    }
    Ok((
        VkTensor::from_buffer(
            dk_buf,
            vec![rows, heads_kv * head_dim],
            VkDType::F32,
            Arc::clone(device),
        ),
        VkTensor::from_buffer(
            dv_buf,
            vec![rows, heads_kv * head_dim],
            VkDType::F32,
            Arc::clone(device),
        ),
    ))
}

#[derive(Debug)]
struct FlashSdpaBackward {
    out: VkTensor,
    lse: VkTensor,
    rows: usize,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    scale: f32,
    inputs: [VkTensor; 3],
}

impl VkBackwardOp for FlashSdpaBackward {
    fn op_name(&self) -> &'static str {
        "flash_sdpa_prefill"
    }

    fn input_refs(&self) -> &[VkTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &VkTensor) -> Result<Vec<Option<VkTensor>>> {
        anyhow::ensure!(
            grad_out.shape() == [self.rows, self.heads_q * self.head_dim],
            "flash_sdpa_prefill backward: grad shape mismatch {:?}",
            grad_out.shape()
        );
        let delta =
            vk_flash_sdpa_delta(grad_out, &self.out, self.rows, self.heads_q, self.head_dim)
                .context("flash_sdpa_prefill backward delta")?;
        let dq = vk_flash_sdpa_bwd_dq(
            &self.inputs[0],
            &self.inputs[1],
            &self.inputs[2],
            grad_out,
            &self.lse,
            &delta,
            self.rows,
            self.heads_q,
            self.heads_kv,
            self.head_dim,
            self.scale,
        )
        .context("flash_sdpa_prefill backward dQ")?;
        let (dk, dv) = vk_flash_sdpa_bwd_dkdv(
            &self.inputs[0],
            &self.inputs[1],
            &self.inputs[2],
            grad_out,
            &self.lse,
            &delta,
            self.rows,
            self.heads_q,
            self.heads_kv,
            self.head_dim,
            self.scale,
        )
        .context("flash_sdpa_prefill backward dK/dV")?;
        Ok(vec![Some(dq), Some(dk), Some(dv)])
    }
}

/// Exact causal GQA SDPA with a custom memory-bounded Vulkan backward.
#[allow(clippy::too_many_arguments)]
pub fn vk_flash_sdpa_prefill_flat(
    q_flat: &VkTensor,
    k_flat: &VkTensor,
    v_flat: &VkTensor,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    scale: f32,
) -> Result<VkTensor> {
    let rows = check_flash_sdpa_flat_inputs(q_flat, k_flat, v_flat, heads_q, heads_kv, head_dim)?;
    let (out, lse) = vk_flash_sdpa_prefill_flat_no_grad(
        q_flat, k_flat, v_flat, heads_q, heads_kv, head_dim, scale,
    )?;
    let grad_fn: Option<Arc<dyn VkBackwardOp>> =
        if q_flat.requires_grad() || k_flat.requires_grad() || v_flat.requires_grad() {
            Some(Arc::new(FlashSdpaBackward {
                out: out.clone(),
                lse,
                rows,
                heads_q,
                heads_kv,
                head_dim,
                scale,
                inputs: [q_flat.clone(), k_flat.clone(), v_flat.clone()],
            }))
        } else {
            None
        };
    Ok(VkTensor::from_op(
        Arc::clone(out.buffer()),
        out.shape().to_vec(),
        out.dtype(),
        Arc::clone(out.device()),
        grad_fn,
    ))
}

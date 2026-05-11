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
use crate::vk_ops::permute::{
    vk_permute_hr_to_rh, vk_permute_rh_to_hr, vk_repeat_kv_heads,
};
use crate::vk_ops::shape::vk_reshape;
use crate::vk_ops::softmax::vk_softmax_lastdim;
use crate::vk_tensor::{VkDType, VkTensor};
use anyhow::Result;

pub fn vk_sdpa_prefill(
    q: &VkTensor,
    k: &VkTensor,
    v: &VkTensor,
    scale: f32,
) -> Result<VkTensor> {
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

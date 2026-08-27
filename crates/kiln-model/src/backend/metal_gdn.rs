//! Metal Gated DeltaNet operation-family helpers.
//!
//! This module owns GDN support gates and command encoding for Q/K norm,
//! fused QKV conv/norm, gates/decay, recurrent decode and prefill, chunked
//! forward-substitution, and GDN-specific prefill conv splitting. The backend
//! facade in `metal` keeps trait-level routing and re-exports public helpers
//! used by forward code.

use anyhow::{Context, Result};

use super::metal_config::*;
use super::metal_core::{kt_metal, kt_metal_alloc};
use super::metal_pipeline::*;
use kiln_tensor::metal_types::buffer_o_kt;

pub(super) fn metal_gdn_gates_supports(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(a.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_log.device(), kiln_tensor::Device::Metal(_))
        || !matches!(dt_bias.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if a.dtype() != kiln_tensor::DType::BF16
        || b.dtype() != kiln_tensor::DType::BF16
        || a_log.dtype() != kiln_tensor::DType::F32
        || dt_bias.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if a.shape() != b.shape() {
        return false;
    }
    let Some(&nv) = a.dims().last() else {
        return false;
    };
    if nv == 0 || nv > 256 {
        return false;
    }
    if a_log.dims() != [nv] || dt_bias.dims() != [nv] {
        return false;
    }
    a.elem_count() > 0
}

pub(crate) fn metal_gdn_gates_decay_supports(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> bool {
    !metal_gdn_prefill_decay_recurrent_disabled() && metal_gdn_gates_supports(a, b, a_log, dt_bias)
}

pub(crate) fn metal_gdn_prefill_ab_in_proj_supports(
    x: &kiln_tensor::Tensor,
    in_proj_ab_t: &kiln_tensor::Tensor,
    nv: usize,
) -> bool {
    if metal_gdn_prefill_ab_in_proj_disabled() {
        return false;
    }
    if x.dtype() != kiln_tensor::DType::BF16 || in_proj_ab_t.dtype() != kiln_tensor::DType::BF16 {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(in_proj_ab_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous() || !in_proj_ab_t.is_contiguous() || nv == 0 {
        return false;
    }
    let Ok((_batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Some(ab_dim) = nv.checked_mul(2) else {
        return false;
    };
    seq_len > 1 && in_proj_ab_t.dims() == [hidden, ab_dim]
}

pub(crate) fn metal_gdn_gates_decay_ab_supports(
    ab: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    nv: usize,
) -> bool {
    if metal_gdn_prefill_ab_in_proj_disabled() || metal_gdn_prefill_decay_recurrent_disabled() {
        return false;
    }
    if ab.dtype() != kiln_tensor::DType::BF16
        || a_log.dtype() != kiln_tensor::DType::F32
        || dt_bias.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(ab.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_log.device(), kiln_tensor::Device::Metal(_))
        || !matches!(dt_bias.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !ab.is_contiguous() || nv == 0 || nv > 256 {
        return false;
    }
    let Ok((batch, seq_len, channels)) = ab.dims3() else {
        return false;
    };
    let Some(ab_dim) = nv.checked_mul(2) else {
        return false;
    };
    let Some(total) = batch.checked_mul(seq_len).and_then(|n| n.checked_mul(nv)) else {
        return false;
    };
    channels == ab_dim
        && total > 0
        && total <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && a_log.dims() == [nv]
        && dt_bias.dims() == [nv]
}

pub(super) fn metal_gdn_forward_substitution_supports(
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(a_strict.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v_prime.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    let dtype = a_strict.dtype();
    if (dtype != kiln_tensor::DType::BF16 && dtype != kiln_tensor::DType::F32)
        || v_prime.dtype() != dtype
        || beta.dtype() != dtype
    {
        return false;
    }
    let Ok((batch, heads, chunk, chunk_cols)) = a_strict.dims4() else {
        return false;
    };
    let Ok((b_v, h_v, c_v, dv)) = v_prime.dims4() else {
        return false;
    };
    let Ok((b_b, h_b, c_b)) = beta.dims3() else {
        return false;
    };

    chunk == chunk_cols
        && (b_v, h_v, c_v) == (batch, heads, chunk)
        && (b_b, h_b, c_b) == (batch, heads, chunk)
        && chunk > 0
        && chunk <= 64
        && dv > 0
        && dv <= 128
}

pub(super) fn metal_gdn_chunk_prep_supports(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(kkt.device(), kiln_tensor::Device::Metal(_))
        || !matches!(qkt.device(), kiln_tensor::Device::Metal(_))
        || !matches!(ks_entry.device(), kiln_tensor::Device::Metal(_))
        || !matches!(q_s.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if g.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || kkt.dtype() != kiln_tensor::DType::BF16
        || qkt.dtype() != kiln_tensor::DType::BF16
        || ks_entry.dtype() != kiln_tensor::DType::BF16
        || q_s.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    // Full chunks only: this keeps speculative multi-token verification on
    // the already-stable portable (kt) path while accelerating long prompt
    // prefill.
    if chunk != 64 {
        return false;
    }
    let Ok((b_v, h_v, c_v, dv)) = v.dims4() else {
        return false;
    };
    if (b_v, h_v, c_v) != (batch, heads, chunk) || dv == 0 || dv > 128 {
        return false;
    }
    let Ok((b_kkt, h_kkt, c_kkt, c2_kkt)) = kkt.dims4() else {
        return false;
    };
    if (b_kkt, h_kkt, c_kkt, c2_kkt) != (batch, heads, chunk, chunk) {
        return false;
    }
    let Ok((b_qkt, h_qkt, c_qkt, c2_qkt)) = qkt.dims4() else {
        return false;
    };
    if (b_qkt, h_qkt, c_qkt, c2_qkt) != (batch, heads, chunk, chunk) {
        return false;
    }
    let Ok((b_ks, h_ks, c_ks, dv_ks)) = ks_entry.dims4() else {
        return false;
    };
    if (b_ks, h_ks, c_ks, dv_ks) != (batch, heads, chunk, dv) {
        return false;
    }
    let Ok((b_qs, h_qs, c_qs, dv_qs)) = q_s.dims4() else {
        return false;
    };
    (b_qs, h_qs, c_qs, dv_qs) == (batch, heads, chunk, dv)
}

pub(super) fn metal_gdn_full_chunk_forward_supports(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !metal_gdn_chunk_prep_supports(g, v, kkt, qkt, ks_entry, q_s)
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if beta.dtype() != kiln_tensor::DType::BF16
        || k_t.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    let Ok((_, _, _, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, h_beta, c_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_kt, h_kt, dk, c_kt)) = k_t.dims4() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_beta, h_beta, c_beta) == (batch, heads, chunk)
        && (b_kt, h_kt, c_kt) == (batch, heads, chunk)
        && (b_state, h_state, dk_state, dv_state) == (batch, heads, dk, dv)
        && chunk == 64
        && dk > 0
        && dk <= 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

fn metal_gdn_full_chunk_forward_strided_inputs_support(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    heads: usize,
) -> bool {
    fn flat_batch_head_ok(stride: &[usize], heads: usize) -> bool {
        stride.len() >= 2
            && stride[1] > 0
            && heads
                .checked_mul(stride[1])
                .is_some_and(|expected| stride[0] == expected)
    }

    fn stride_u32_ok(stride: usize) -> bool {
        stride > 0 && stride <= u32::MAX as usize
    }

    let g_stride = g.layout().strides();
    let v_stride = v.layout().strides();
    let beta_stride = beta.layout().strides();
    let kt_stride = k_t.layout().strides();

    if g_stride.len() != 3 || v_stride.len() != 4 || beta_stride.len() != 3 || kt_stride.len() != 4
    {
        return false;
    }
    if !flat_batch_head_ok(g_stride, heads)
        || !flat_batch_head_ok(v_stride, heads)
        || !flat_batch_head_ok(beta_stride, heads)
        || !flat_batch_head_ok(kt_stride, heads)
    {
        return false;
    }

    // This path only needs to support a time-window narrow. Keep the value
    // dimension contiguous so the per-value lane remains coalesced.
    v_stride[3] == 1
        && [
            g_stride[1],
            g_stride[2],
            v_stride[1],
            v_stride[2],
            v_stride[3],
            beta_stride[1],
            beta_stride[2],
            kt_stride[1],
            kt_stride[2],
            kt_stride[3],
        ]
        .into_iter()
        .all(stride_u32_ok)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(super) fn metal_gdn_full_chunk_forward_head_last_supports(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    t_start: usize,
    seq_len: usize,
) -> bool {
    if !metal_gdn_full_chunk_forward_supports(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state)
        || !matches!(out.device(), kiln_tensor::Device::Metal(_))
        || out.dtype() != kiln_tensor::DType::BF16
        || !out.is_contiguous()
    {
        return false;
    }
    let Ok((batch, heads, chunk)) = g.dims3() else {
        return false;
    };
    let Ok((_, _, _, dv)) = v.dims4() else {
        return false;
    };
    out.dims4()
        .is_ok_and(|dims| dims == (batch, seq_len, heads, dv))
        && chunk == 64
        && t_start <= seq_len
        && t_start + chunk <= seq_len
        && metal_gdn_full_chunk_forward_strided_inputs_support(g, v, beta, k_t, heads)
}

pub(super) fn metal_gdn_recurrent_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || beta.dtype() != kiln_tensor::DType::BF16
        || g.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, heads, dk)) = q.dims3() else {
        return false;
    };
    let Ok((b_k, h_k, dk_k)) = k.dims3() else {
        return false;
    };
    let Ok((b_v, h_v, dv)) = v.dims3() else {
        return false;
    };
    let Ok((b_b, h_b)) = beta.dims2() else {
        return false;
    };
    let Ok((b_g, h_g)) = g.dims2() else {
        return false;
    };
    let Ok((b_s, h_s, dk_s, dv_s)) = state.dims4() else {
        return false;
    };
    (b_k, h_k, dk_k) == (batch, heads, dk)
        && (b_v, h_v) == (batch, heads)
        && (b_b, h_b) == (batch, heads)
        && (b_g, h_g) == (batch, heads)
        && (b_s, h_s, dk_s, dv_s) == (batch, heads, dk, dv)
        && dk <= 256
        && dv <= 1024
}

const METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN: usize = 2048;

pub(super) fn metal_gdn_recurrent_prefill_head_last_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || beta.dtype() != kiln_tensor::DType::BF16
        || g.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, q_heads, seq_len, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, h_k, t_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, v_heads, t_v, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, h_beta, t_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_g, h_g, t_g)) = g.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_k, h_k, t_k, dk_k) == (batch, q_heads, seq_len, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_beta, h_beta, t_beta) == (batch, v_heads, seq_len)
        && (b_g, h_g, t_g) == (batch, v_heads, seq_len)
        && (b_state, h_state, dk_state, dv_state) == (batch, v_heads, dk, dv)
        && v_heads >= q_heads
        && v_heads % q_heads == 0
        && seq_len > 1
        && seq_len <= METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN
        && dk == 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

pub(super) fn metal_gdn_recurrent_prefill_native_head_last_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(beta.device(), kiln_tensor::Device::Metal(_))
        || !matches!(g.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || beta.dtype() != kiln_tensor::DType::BF16
        || g.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, t_k, h_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, t_v, value_heads, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_beta, t_beta, h_beta)) = beta.dims3() else {
        return false;
    };
    let Ok((b_g, t_g, h_g)) = g.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_k, t_k, h_k, dk_k) == (batch, seq_len, q_heads, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_beta, t_beta, h_beta) == (batch, seq_len, value_heads)
        && (b_g, t_g, h_g) == (batch, seq_len, value_heads)
        && (b_state, h_state, dk_state, dv_state) == (batch, value_heads, dk, dv)
        && value_heads >= q_heads
        && value_heads % q_heads == 0
        && seq_len >= 1
        && seq_len <= METAL_GDN_RECURRENT_PREFILL_MAX_SEQ_LEN
        && dk == 128
        && dv > 0
        && dv <= 128
        && state.is_contiguous()
}

pub(crate) fn metal_gdn_recurrent_prefill_native_head_last_decay_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    decay: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    !metal_gdn_prefill_decay_recurrent_disabled()
        && metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, decay, state)
}

pub(crate) fn metal_gdn_decode_gates_recurrent_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
) -> bool {
    if metal_gdn_decode_gates_recurrent_disabled()
        || !matches!(q.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !matches!(k.device(), kiln_tensor::Device::Metal(_))
        || !matches!(v.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_log.device(), kiln_tensor::Device::Metal(_))
        || !matches!(dt_bias.device(), kiln_tensor::Device::Metal(_))
        || !matches!(state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::BF16
        || k.dtype() != kiln_tensor::DType::BF16
        || v.dtype() != kiln_tensor::DType::BF16
        || a.dtype() != kiln_tensor::DType::BF16
        || b.dtype() != kiln_tensor::DType::BF16
        || a_log.dtype() != kiln_tensor::DType::F32
        || dt_bias.dtype() != kiln_tensor::DType::BF16
        || state.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, t_k, h_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, t_v, value_heads, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_a, t_a, h_a)) = a.dims3() else {
        return false;
    };
    let Ok((b_b, t_b, h_b)) = b.dims3() else {
        return false;
    };
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    let Some(batch_heads) = batch.checked_mul(value_heads) else {
        return false;
    };
    batch > 0
        && seq_len == 1
        && (b_k, t_k, h_k, dk_k) == (batch, seq_len, q_heads, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_a, t_a, h_a) == (batch, seq_len, value_heads)
        && (b_b, t_b, h_b) == (batch, seq_len, value_heads)
        && a_log.dims() == [value_heads]
        && dt_bias.dims() == [value_heads]
        && (b_state, h_state, dk_state, dv_state) == (batch, value_heads, dk, dv)
        && q_heads > 0
        && value_heads > q_heads
        && value_heads % q_heads == 0
        && dk == 128
        && dv == 128
        && batch_heads <= u32::MAX as usize
        && q_heads <= u32::MAX as usize
        && value_heads <= u32::MAX as usize
        && state.is_contiguous()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_rmsnorm_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
) -> bool {
    if metal_gdn_decode_gates_recurrent_rmsnorm_disabled() {
        return false;
    }
    if !metal_gdn_decode_gates_recurrent_supports(q, k, v, a, b, a_log, dt_bias, state) {
        return false;
    }
    metal_gated_rms_norm_supports(v, z, weight)
}

pub(super) fn metal_gated_rms_norm_supports(
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
) -> bool {
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(z.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    // x/z are BF16 activations. The norm weight follows the model dtype, which
    // for Qwen3.5 GDN is BF16 (same as the CUDA `gdn_gated_rms_norm_supports_kt`
    // contract). Accept F32 too for callers that pre-promote — the kernel casts
    // the weight to F32 internally either way.
    if x.dtype() != kiln_tensor::DType::BF16
        || z.dtype() != kiln_tensor::DType::BF16
        || !matches!(
            weight.dtype(),
            kiln_tensor::DType::BF16 | kiln_tensor::DType::F32
        )
    {
        return false;
    }
    let Ok((batch, seq_len, heads, hidden)) = x.dims4() else {
        return false;
    };
    let Ok((z_batch, z_seq_len, z_heads, z_hidden)) = z.dims4() else {
        return false;
    };
    if (z_batch, z_seq_len, z_heads, z_hidden) != (batch, seq_len, heads, hidden) {
        return false;
    }
    weight.dims() == [hidden] && hidden <= 1024
}

pub(crate) fn metal_gdn_qk_norm_supports(q: &kiln_tensor::Tensor, k: &kiln_tensor::Tensor) -> bool {
    if metal_gdn_qk_norm_disabled() {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::F32 || k.dtype() != kiln_tensor::DType::F32 {
        return false;
    }
    let Some(hidden) = q.dims().last().copied() else {
        return false;
    };
    q.rank() >= 1 && q.dims() == k.dims() && hidden <= 8192
}

pub(crate) fn metal_gdn_qk_norm_gqa_supports(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    nv: usize,
) -> bool {
    if metal_gdn_qk_norm_disabled() {
        return false;
    }
    if !matches!(q.device(), kiln_tensor::Device::Metal(_))
        || !matches!(k.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if q.dtype() != kiln_tensor::DType::F32 || k.dtype() != kiln_tensor::DType::F32 {
        return false;
    }
    let Ok((_, _, nk, hidden)) = q.dims4() else {
        return false;
    };
    q.dims() == k.dims()
        && nk > 0
        && nv > nk
        && nv % nk == 0
        && hidden <= 8192
        && nv <= u32::MAX as usize
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_qkv_conv_norm_supports(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> bool {
    if metal_gdn_qkv_conv_norm_disabled() || metal_gdn_qk_norm_disabled() {
        return false;
    }
    if kernel_size != 4 || dk != 128 || dv != 128 {
        return false;
    }
    if !matches!(mixed_qkv.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
        || !matches!(conv_state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if mixed_qkv.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    let Ok((batch, seq_len, channels)) = mixed_qkv.dims3() else {
        return false;
    };
    let qk_dim = nk.saturating_mul(dk);
    let v_dim = nv.saturating_mul(dv);
    let Some(expected_channels) = qk_dim.checked_mul(2).and_then(|n| n.checked_add(v_dim)) else {
        return false;
    };
    let weight_ok = match weight.rank() {
        3 => weight
            .dims3()
            .is_ok_and(|(c, one, k)| c == channels && one == 1 && k == kernel_size),
        2 => weight
            .dims2()
            .is_ok_and(|(c, k)| c == channels && k == kernel_size),
        _ => false,
    };
    let Some(rows) = nk
        .checked_add(nk)
        .and_then(|n| n.checked_add(nv))
        .and_then(|n| n.checked_mul(batch))
    else {
        return false;
    };
    batch > 0
        && seq_len == 1
        && channels == expected_channels
        && nk > 0
        && nv > nk
        && nv % nk == 0
        && weight_ok
        && conv_state
            .dims3()
            .is_ok_and(|(b, c, k)| (b, c, k) == (batch, channels, kernel_size - 1))
        && channels <= u32::MAX as usize
        && nk <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && rows <= u32::MAX as usize
}

pub(super) fn metal_gdn_in_proj_decode_supports(
    x: &kiln_tensor::Tensor,
    qkv_t: &kiln_tensor::Tensor,
    z_t: &kiln_tensor::Tensor,
    a_t: &kiln_tensor::Tensor,
    b_t: &kiln_tensor::Tensor,
) -> bool {
    if x.dtype() != kiln_tensor::DType::BF16
        || qkv_t.dtype() != kiln_tensor::DType::BF16
        || z_t.dtype() != kiln_tensor::DType::BF16
        || a_t.dtype() != kiln_tensor::DType::BF16
        || b_t.dtype() != kiln_tensor::DType::BF16
    {
        return false;
    }
    if !matches!(x.device(), kiln_tensor::Device::Metal(_))
        || !matches!(qkv_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(z_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(a_t.device(), kiln_tensor::Device::Metal(_))
        || !matches!(b_t.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !x.is_contiguous()
        || !qkv_t.is_contiguous()
        || !z_t.is_contiguous()
        || !a_t.is_contiguous()
        || !b_t.is_contiguous()
    {
        return false;
    }
    let Ok((batch, seq_len, hidden)) = x.dims3() else {
        return false;
    };
    let Ok((qkv_hidden, qkv_dim)) = qkv_t.dims2() else {
        return false;
    };
    let Ok((z_hidden, z_dim)) = z_t.dims2() else {
        return false;
    };
    let Ok((a_hidden, nv)) = a_t.dims2() else {
        return false;
    };
    let Ok((b_hidden, b_nv)) = b_t.dims2() else {
        return false;
    };
    let Some(total) = qkv_dim
        .checked_add(z_dim)
        .and_then(|n| n.checked_add(nv))
        .and_then(|n| n.checked_add(b_nv))
    else {
        return false;
    };
    let Some(dispatch_total) = total.checked_mul(batch) else {
        return false;
    };

    batch > 0
        && seq_len == 1
        && hidden == qkv_hidden
        && hidden == z_hidden
        && hidden == a_hidden
        && hidden == b_hidden
        && nv == b_nv
        && hidden <= u32::MAX as usize
        && qkv_dim <= u32::MAX as usize
        && z_dim <= u32::MAX as usize
        && nv <= u32::MAX as usize
        && total <= u32::MAX as usize
        && dispatch_total <= u32::MAX as usize
}

pub(super) fn metal_gdn_in_proj_decode_bf16(
    x: &kiln_tensor::Tensor,
    qkv_t: &kiln_tensor::Tensor,
    z_t: &kiln_tensor::Tensor,
    a_t: &kiln_tensor::Tensor,
    b_t: &kiln_tensor::Tensor,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_in_proj_decode_supports(x, qkv_t, z_t, a_t, b_t),
        "metal gdn in-proj supports only BF16 [B,1,H] x [H,*] on Metal"
    );
    let (batch, _, hidden) = x.dims3()?;
    let (_, qkv_dim) = qkv_t.dims2()?;
    let (_, z_dim) = z_t.dims2()?;
    let (_, nv) = a_t.dims2()?;
    let output_total = qkv_dim + z_dim + (nv * 2);
    let row_grouping_enabled = batch >= 3 && !metal_gdn_in_proj_row_pair_disabled();
    let row_triple_enabled =
        batch == 3 && row_grouping_enabled && !metal_gdn_in_proj_row_triple_disabled();
    let row_quad_enabled =
        row_grouping_enabled && batch >= 8 && !metal_gdn_in_proj_row_quad_disabled();
    let row_group_size = if row_triple_enabled {
        3usize
    } else if row_quad_enabled {
        4usize
    } else if row_grouping_enabled {
        2usize
    } else {
        1usize
    };
    let x_metal = kt_metal(&x)?;
    // The kernel writes every output element exactly once. Keep bs=1 on one
    // backing allocation, but use separate batch outputs so each `[B,1,N]`
    // tensor remains contiguous for the following fused decode kernels.
    let (qkv_out, z_out, a_out, b_out) = if batch == 1 {
        let proj_out = kt_metal_alloc(
            x_metal,
            kiln_tensor::DType::BF16,
            &[1usize, 1usize, output_total],
        )?;
        (
            proj_out.narrow(2, 0, qkv_dim)?,
            proj_out.narrow(2, qkv_dim, z_dim)?,
            proj_out.narrow(2, qkv_dim + z_dim, nv)?,
            proj_out.narrow(2, qkv_dim + z_dim + nv, nv)?,
        )
    } else {
        (
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, qkv_dim])?,
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, z_dim])?,
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, nv])?,
            kt_metal_alloc(x_metal, kiln_tensor::DType::BF16, &[batch, 1usize, nv])?,
        )
    };

    let companion = x_metal.companion()?;
    let pipeline = metal_gdn_in_proj_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_in_proj_decode_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let qkv_metal = kt_metal(&qkv_t)?;
        let z_metal = kt_metal(&z_t)?;
        let a_metal = kt_metal(&a_t)?;
        let b_metal = kt_metal(&b_t)?;
        let qkv_o_metal = kt_metal(&qkv_out)?;
        let z_o_metal = kt_metal(&z_out)?;
        let a_o_metal = kt_metal(&a_out)?;
        let b_o_metal = kt_metal(&b_out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        // The kt-typed helper reads `start_offset()` + `size_in_bytes()`
        // off the kt Layout/DType; everything else is bit-identical.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let qkv_buf = buffer_o_kt(qkv_metal.buffer().as_ref(), qkv_t.layout(), qkv_t.dtype());
        let z_buf = buffer_o_kt(z_metal.buffer().as_ref(), z_t.layout(), z_t.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a_t.layout(), a_t.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b_t.layout(), b_t.dtype());
        let qkv_o_buf = buffer_o_kt(
            qkv_o_metal.buffer().as_ref(),
            qkv_out.layout(),
            qkv_out.dtype(),
        );
        let z_o_buf = buffer_o_kt(z_o_metal.buffer().as_ref(), z_out.layout(), z_out.dtype());
        let a_o_buf = buffer_o_kt(a_o_metal.buffer().as_ref(), a_out.layout(), a_out.dtype());
        let b_o_buf = buffer_o_kt(b_o_metal.buffer().as_ref(), b_out.layout(), b_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(qkv_buf.buffer), qkv_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qkv_o_buf.buffer), qkv_o_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(z_o_buf.buffer), z_o_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(a_o_buf.buffer), a_o_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(b_o_buf.buffer), b_o_buf.offset_in_bytes);

        let serial_vector_mode = batch == 1
            && !metal_gdn_in_proj_serial_vector_load_disabled()
            && qkv_dim % 2 == 0
            && z_dim % 2 == 0
            && qkv_buf.offset_in_bytes % 4 == 0
            && z_buf.offset_in_bytes % 4 == 0;
        let serial_x2_mode = serial_vector_mode
            && !metal_gdn_in_proj_serial_x2_load_disabled()
            && hidden % 2 == 0
            && x_buf.offset_in_bytes % 4 == 0;
        let dispatch_cols = if serial_vector_mode {
            (qkv_dim / 2) + (z_dim / 2) + (nv * 2)
        } else if batch == 1 {
            output_total
        } else {
            qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + (nv * 2)
        };
        let dispatch_rows = batch.div_ceil(row_group_size);
        let dispatch_total = dispatch_rows * dispatch_cols;

        let hidden_u32 = hidden as u32;
        let qkv_dim_u32 = qkv_dim as u32;
        let z_dim_u32 = z_dim as u32;
        let nv_u32 = nv as u32;
        let batch_u32 = batch as u32;
        let row_pair_mode_u32 = if serial_x2_mode {
            7
        } else if serial_vector_mode {
            6
        } else if row_group_size == 1 {
            0
        } else {
            row_group_size as u32
        };
        encoder.set_bytes(9, &hidden_u32);
        encoder.set_bytes(10, &qkv_dim_u32);
        encoder.set_bytes(11, &z_dim_u32);
        encoder.set_bytes(12, &nv_u32);
        encoder.set_bytes(13, &batch_u32);
        encoder.set_bytes(14, &row_pair_mode_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: dispatch_total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((qkv_out, z_out, a_out, b_out))
}

pub(crate) fn metal_gdn_qk_norm_f32_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    q_scale: f32,
    eps: f32,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    let dims = q.dims().to_vec();
    let hidden = *dims
        .last()
        .context("metal gdn qk norm requires rank >= 1 input")?;
    anyhow::ensure!(q.dims() == k.dims(), "metal gdn qk norm shape mismatch");
    anyhow::ensure!(hidden <= 8192, "metal gdn qk norm hidden dim > 8192");
    let rows: usize = dims[..dims.len() - 1].iter().product();
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal gdn qk norm shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    // The kernel writes every Q/K element for every row.
    let q_metal = kt_metal(&q)?;
    let q_out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, dims.as_slice())?;
    let k_out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, dims.as_slice())?;

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_qk_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_qk_norm_f32_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let qo_metal = kt_metal(&q_out)?;
        let ko_metal = kt_metal(&k_out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let qo_buf = buffer_o_kt(qo_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(ko_buf.buffer), ko_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &hidden_u32);
        encoder.set_bytes(6, &q_scale);
        encoder.set_bytes(7, &eps);
        encoder.set_bytes(8, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out))
}

pub(crate) fn metal_gdn_qk_norm_gqa_f32_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    nv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_gdn_qk_norm_gqa_supports(q, k, nv),
        "metal gdn qk norm gqa unsupported shape"
    );
    let (batch, seq_len, nk, hidden) = q.dims4()?;
    let gqa_ratio = nv / nk;
    let rows = batch * seq_len * nk;
    anyhow::ensure!(
        rows <= u32::MAX as usize
            && nk <= u32::MAX as usize
            && hidden <= u32::MAX as usize
            && gqa_ratio <= u32::MAX as usize,
        "metal gdn qk norm gqa shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    // Each source head writes all replicated value-head outputs.
    let q_metal = kt_metal(&q)?;
    let q_out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, nv, hidden],
    )?;
    let k_out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, nv, hidden],
    )?;

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_qk_norm_gqa_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_qk_norm_gqa_f32_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let qo_metal = kt_metal(&q_out)?;
        let ko_metal = kt_metal(&k_out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let qo_buf = buffer_o_kt(qo_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(ko_buf.buffer), ko_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let nk_u32 = nk as u32;
        let nv_u32 = nv as u32;
        let hidden_u32 = hidden as u32;
        let gqa_ratio_u32 = gqa_ratio as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &nk_u32);
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &hidden_u32);
        encoder.set_bytes(8, &gqa_ratio_u32);
        encoder.set_bytes(9, &q_scale);
        encoder.set_bytes(10, &eps);
        encoder.set_bytes(11, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_qkv_conv_norm_bf16(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_decode_qkv_conv_norm_supports(
            mixed_qkv,
            weight,
            conv_state,
            kernel_size,
            nk,
            dk,
            nv,
            dv
        ),
        "metal gdn decode qkv conv/norm unsupported shape"
    );
    let (batch, _, channels) = mixed_qkv.dims3()?;
    let rows_per_batch = nk + nk + nv;
    let rows = batch * rows_per_batch;
    anyhow::ensure!(
        rows <= u32::MAX as usize,
        "metal gdn decode qkv conv/norm shape too large"
    );

    let mixed_qkv = mixed_qkv.contiguous()?;
    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal gdn decode qkv conv/norm weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }

    // The kernel writes every unexpanded Q/K and V element, and updates each
    // convolution state channel exactly once.
    let mixed_qkv_metal = kt_metal(&mixed_qkv)?;
    let q_out = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, nk, dk],
    )?;
    let k_out = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, nk, dk],
    )?;
    let v_out = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, 1usize, nv, dv],
    )?;

    let companion = mixed_qkv_metal.companion()?;
    let pipeline = metal_gdn_decode_qkv_conv_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_qkv_conv_norm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let x_metal = kt_metal(&mixed_qkv)?;
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(&conv_state)?;
        let qo_metal = kt_metal(&q_out)?;
        let ko_metal = kt_metal(&k_out)?;
        let vo_metal = kt_metal(&v_out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let x_buf = buffer_o_kt(
            x_metal.buffer().as_ref(),
            mixed_qkv.layout(),
            mixed_qkv.dtype(),
        );
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let qo_buf = buffer_o_kt(qo_metal.buffer().as_ref(), q_out.layout(), q_out.dtype());
        let ko_buf = buffer_o_kt(ko_metal.buffer().as_ref(), k_out.layout(), k_out.dtype());
        let vo_buf = buffer_o_kt(vo_metal.buffer().as_ref(), v_out.layout(), v_out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qo_buf.buffer), qo_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ko_buf.buffer), ko_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(vo_buf.buffer), vo_buf.offset_in_bytes);

        let nk_u32 = nk as u32;
        let nv_u32 = nv as u32;
        encoder.set_bytes(6, &nk_u32);
        encoder.set_bytes(7, &nv_u32);
        encoder.set_bytes(8, &q_scale);
        encoder.set_bytes(9, &eps);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: rows_per_batch,
            height: batch,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q_out, k_out, v_out))
}

pub(super) fn metal_gdn_gates_bf16(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    let shape = a.dims().to_vec();
    let nv = *shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates requires at least rank-1 input"))?;
    let total = a.elem_count();
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates input too large"
    );
    anyhow::ensure!(nv <= u32::MAX as usize, "metal gdn_gates nv too large");

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_metal = kt_metal(&a)?;
    // The gates kernel writes every beta/g element.
    let beta = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, &shape)?;
    let g = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, &shape)?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_gates_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(g_buf.buffer), g_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((beta, g))
}

pub(crate) fn metal_gdn_prefill_ab_in_proj_bf16(
    x: &kiln_tensor::Tensor,
    in_proj_ab_t: &kiln_tensor::Tensor,
    nv: usize,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_prefill_ab_in_proj_supports(x, in_proj_ab_t, nv),
        "metal gdn prefill A/B in-proj unsupported shape"
    );
    let ab = x
        .broadcast_matmul(in_proj_ab_t)
        .context("metal gdn prefill A/B in-proj matmul")?;
    let a = ab.narrow(2, 0, nv)?;
    let b = ab.narrow(2, nv, nv)?;
    Ok((ab, a, b))
}

pub(crate) fn metal_gdn_gates_decay_bf16(
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_gdn_gates_decay_supports(a, b, a_log, dt_bias),
        "metal gdn_gates decay unsupported shape"
    );
    let shape = a.dims().to_vec();
    let nv = *shape
        .last()
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates decay requires at least rank-1 input"))?;
    let total = a.elem_count();
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates decay input too large"
    );
    anyhow::ensure!(
        nv <= u32::MAX as usize,
        "metal gdn_gates decay nv too large"
    );

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let a_metal = kt_metal(&a)?;
    let beta = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, shape.as_slice())?;
    let decay = kt_metal_alloc(a_metal, kiln_tensor::DType::BF16, shape.as_slice())?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_gates_decay_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_decay_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let beta_metal = kt_metal(&beta)?;
        let decay_metal = kt_metal(&decay)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let decay_buf = buffer_o_kt(decay_metal.buffer().as_ref(), decay.layout(), decay.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(decay_buf.buffer), decay_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(6, &nv_u32);
        encoder.set_bytes(7, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((beta, decay))
}

pub(crate) fn metal_gdn_gates_decay_ab_bf16(
    ab: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    nv: usize,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    anyhow::ensure!(
        metal_gdn_gates_decay_ab_supports(ab, a_log, dt_bias, nv),
        "metal gdn_gates decay A/B unsupported shape"
    );
    let (batch, seq_len, _channels) = ab.dims3()?;
    let total = batch
        .checked_mul(seq_len)
        .and_then(|n| n.checked_mul(nv))
        .ok_or_else(|| anyhow::anyhow!("metal gdn_gates decay A/B input too large"))?;
    anyhow::ensure!(
        total <= u32::MAX as usize,
        "metal gdn_gates decay A/B input too large"
    );
    anyhow::ensure!(
        nv <= u32::MAX as usize,
        "metal gdn_gates decay A/B nv too large"
    );

    let ab = ab.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let shape = vec![batch, seq_len, nv];
    let ab_metal = kt_metal(&ab)?;
    let beta = kt_metal_alloc(ab_metal, kiln_tensor::DType::BF16, shape.as_slice())?;
    let decay = kt_metal_alloc(ab_metal, kiln_tensor::DType::BF16, shape.as_slice())?;

    let companion = ab_metal.companion()?;
    let pipeline = metal_gdn_gates_decay_ab_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_gates_decay_ab_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let beta_metal = kt_metal(&beta)?;
        let decay_metal = kt_metal(&decay)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let ab_buf = buffer_o_kt(ab_metal.buffer().as_ref(), ab.layout(), ab.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let decay_buf = buffer_o_kt(decay_metal.buffer().as_ref(), decay.layout(), decay.dtype());

        encoder.set_buffer(0, Some(ab_buf.buffer), ab_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(decay_buf.buffer), decay_buf.offset_in_bytes);

        let nv_u32 = nv as u32;
        let total_u32 = total as u32;
        encoder.set_bytes(5, &nv_u32);
        encoder.set_bytes(6, &total_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: total,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok((beta, decay))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_decode_gates_recurrent_supports(q, k, v, a, b, a_log, dt_bias, state),
        "metal gdn decode gates+recurrent unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn decode gates+recurrent shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let q_metal = kt_metal(&q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_decode_gates_recurrent_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_gates_recurrent_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let a_metal = kt_metal(&a)?;
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let state_metal = kt_metal(&state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        encoder.set_bytes(9, &batch_heads_u32);
        encoder.set_bytes(10, &dk_u32);
        encoder.set_bytes(11, &dv_u32);
        encoder.set_bytes(12, &value_heads_u32);
        encoder.set_bytes(13, &q_heads_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_decode_gates_recurrent_rmsnorm_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    a: &kiln_tensor::Tensor,
    b: &kiln_tensor::Tensor,
    a_log: &kiln_tensor::Tensor,
    dt_bias: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_decode_gates_recurrent_rmsnorm_supports(
            q, k, v, a, b, a_log, dt_bias, state, z, weight
        ),
        "metal gdn decode gates+recurrent+rmsnorm unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn decode gates+recurrent+rmsnorm shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;
    let z = z.contiguous()?;
    let weight = weight.contiguous()?;
    let q_metal = kt_metal(&q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_decode_gates_recurrent_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let a_metal = kt_metal(&a)?;
        let b_metal = kt_metal(&b)?;
        let al_metal = kt_metal(&a_log)?;
        let dt_metal = kt_metal(&dt_bias)?;
        let state_metal = kt_metal(&state)?;
        let z_metal = kt_metal(&z)?;
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
        let al_buf = buffer_o_kt(al_metal.buffer().as_ref(), a_log.layout(), a_log.dtype());
        let dt_buf = buffer_o_kt(
            dt_metal.buffer().as_ref(),
            dt_bias.layout(),
            dt_bias.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let z_buf = buffer_o_kt(z_metal.buffer().as_ref(), z.layout(), z.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(al_buf.buffer), al_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(dt_buf.buffer), dt_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(10, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        encoder.set_bytes(11, &batch_heads_u32);
        encoder.set_bytes(12, &dk_u32);
        encoder.set_bytes(13, &dv_u32);
        encoder.set_bytes(14, &value_heads_u32);
        encoder.set_bytes(15, &q_heads_u32);
        encoder.set_bytes(16, &eps);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: dv,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(super) fn metal_gated_rms_norm_bf16(
    x: &kiln_tensor::Tensor,
    z: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    eps: f32,
) -> Result<kiln_tensor::Tensor> {
    let (batch, seq_len, heads, hidden) = x.dims4()?;
    let rows = batch
        .checked_mul(seq_len)
        .and_then(|v| v.checked_mul(heads))
        .context("metal gated rmsnorm row count overflow")?;
    anyhow::ensure!(
        rows <= u32::MAX as usize && hidden <= u32::MAX as usize,
        "metal gated rmsnorm shape too large"
    );
    anyhow::ensure!(hidden <= 1024, "metal gated rmsnorm hidden dim > 1024");

    let x = x.contiguous()?;
    let z = z.contiguous()?;
    // The kernel binds `weight` as a `device const float*`. The norm weight is
    // BF16 for Qwen3.5 (matches the CUDA path); promote to F32 here so the
    // kernel's buffer type is satisfied. F32 weights pass through unchanged.
    let weight = weight.contiguous()?.to_dtype(kiln_tensor::DType::F32)?;
    let x_metal = kt_metal(&x)?;
    // The kernel writes every hidden element for every row.
    let out = kt_metal_alloc(
        x_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, heads, hidden],
    )?;

    if rows == 0 {
        return Ok(out);
    }

    let companion = x_metal.companion()?;
    let pipeline = metal_gated_rms_norm_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gated_rmsnorm_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let z_metal = kt_metal(&z)?;
        let w_metal = kt_metal(&weight)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 rmsnorm-family: `buffer_o` → `buffer_o_kt`.
        let x_buf = buffer_o_kt(x_metal.buffer().as_ref(), x.layout(), x.dtype());
        let z_buf = buffer_o_kt(z_metal.buffer().as_ref(), z.layout(), z.dtype());
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(z_buf.buffer), z_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let rows_u32 = rows as u32;
        let hidden_u32 = hidden as u32;
        let threads = hidden.next_power_of_two().clamp(32, 1024);
        let threads_u32 = threads as u32;
        encoder.set_bytes(4, &rows_u32);
        encoder.set_bytes(5, &hidden_u32);
        encoder.set_bytes(6, &eps);
        encoder.set_bytes(7, &threads_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: threads,
            height: rows,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(super) fn metal_gdn_forward_substitution_bf16(
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    let (batch, heads, chunk_size, _) = a_strict.dims4()?;
    let dv = v_prime.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn forward-substitution shape too large"
    );

    let a_strict = a_strict.contiguous()?;
    let v_prime = v_prime.contiguous()?;
    let beta = beta.contiguous()?;
    // The kernel writes every chunk/value element.
    let a_metal = kt_metal(&a_strict)?;
    let out = kt_metal_alloc(
        a_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_forward_substitution_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_forward_substitution_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v_prime)?;
        let beta_metal = kt_metal(&beta)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(
            a_metal.buffer().as_ref(),
            a_strict.layout(),
            a_strict.dtype(),
        );
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_prime.layout(), v_prime.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(4, &batch_heads_u32);
        encoder.set_bytes(5, &chunk_size_u32);
        encoder.set_bytes(6, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(super) fn metal_gdn_forward_substitution_f32(
    a_strict: &kiln_tensor::Tensor,
    v_prime: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    let (batch, heads, chunk_size, _) = a_strict.dims4()?;
    let dv = v_prime.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn forward-substitution f32 shape too large"
    );

    let a_strict = a_strict.contiguous()?;
    let v_prime = v_prime.contiguous()?;
    let beta = beta.contiguous()?;
    let a_metal = kt_metal(&a_strict)?;
    let out = kt_metal_alloc(
        a_metal,
        kiln_tensor::DType::F32,
        &[batch, heads, chunk_size, dv],
    )?;

    let companion = a_metal.companion()?;
    let pipeline = metal_gdn_forward_substitution_f32_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_forward_substitution_f32");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v_prime)?;
        let beta_metal = kt_metal(&beta)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let a_buf = buffer_o_kt(
            a_metal.buffer().as_ref(),
            a_strict.layout(),
            a_strict.dtype(),
        );
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v_prime.layout(), v_prime.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(4, &batch_heads_u32);
        encoder.set_bytes(5, &chunk_size_u32);
        encoder.set_bytes(6, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(super) fn metal_gdn_chunk_prep_bf16(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    let (batch, heads, chunk_size) = g.dims3()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && chunk_size <= u32::MAX as usize
            && dv <= u32::MAX as usize,
        "metal gdn chunk-prep shape too large"
    );

    let g = g.contiguous()?;
    let v = v.contiguous()?;
    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;
    let g_metal = kt_metal(&g)?;
    // The prep kernel fills each temporary completely before any consumer sees it.
    let a_strict = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, chunk_size],
    )?;
    let b_mask = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, chunk_size],
    )?;
    let v_prime = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;
    let q_s_scaled = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;
    let decay_last_col = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size],
    )?;
    let p_last = kt_metal_alloc(g_metal, kiln_tensor::DType::BF16, &[batch, heads])?;

    let companion = g_metal.companion()?;
    let pipeline = metal_gdn_chunk_prep_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_chunk_prep_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v)?;
        let kkt_metal = kt_metal(&kkt)?;
        let qkt_metal = kt_metal(&qkt)?;
        let ks_metal = kt_metal(&ks_entry)?;
        let qs_metal = kt_metal(&q_s)?;
        let a_metal = kt_metal(&a_strict)?;
        let b_metal = kt_metal(&b_mask)?;
        let vp_metal = kt_metal(&v_prime)?;
        let qss_metal = kt_metal(&q_s_scaled)?;
        let dl_metal = kt_metal(&decay_last_col)?;
        let pl_metal = kt_metal(&p_last)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kkt_buf = buffer_o_kt(kkt_metal.buffer().as_ref(), kkt.layout(), kkt.dtype());
        let qkt_buf = buffer_o_kt(qkt_metal.buffer().as_ref(), qkt.layout(), qkt.dtype());
        let ks_buf = buffer_o_kt(
            ks_metal.buffer().as_ref(),
            ks_entry.layout(),
            ks_entry.dtype(),
        );
        let qs_buf = buffer_o_kt(qs_metal.buffer().as_ref(), q_s.layout(), q_s.dtype());
        let a_buf = buffer_o_kt(
            a_metal.buffer().as_ref(),
            a_strict.layout(),
            a_strict.dtype(),
        );
        let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b_mask.layout(), b_mask.dtype());
        let vp_buf = buffer_o_kt(
            vp_metal.buffer().as_ref(),
            v_prime.layout(),
            v_prime.dtype(),
        );
        let qss_buf = buffer_o_kt(
            qss_metal.buffer().as_ref(),
            q_s_scaled.layout(),
            q_s_scaled.dtype(),
        );
        let dl_buf = buffer_o_kt(
            dl_metal.buffer().as_ref(),
            decay_last_col.layout(),
            decay_last_col.dtype(),
        );
        let pl_buf = buffer_o_kt(pl_metal.buffer().as_ref(), p_last.layout(), p_last.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(a_buf.buffer), a_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(b_buf.buffer), b_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(vp_buf.buffer), vp_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(qss_buf.buffer), qss_buf.offset_in_bytes);
        encoder.set_buffer(10, Some(dl_buf.buffer), dl_buf.offset_in_bytes);
        encoder.set_buffer(11, Some(pl_buf.buffer), pl_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let chunk_size_u32 = chunk_size as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(12, &batch_heads_u32);
        encoder.set_bytes(13, &chunk_size_u32);
        encoder.set_bytes(14, &dv_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((
        a_strict,
        b_mask,
        v_prime,
        q_s_scaled,
        decay_last_col,
        p_last,
    ))
}

pub(super) fn metal_gdn_full_chunk_forward_bf16(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_full_chunk_forward_supports(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state),
        "metal gdn full-chunk unsupported shape"
    );
    let (batch, heads, chunk_size) = g.dims3()?;
    let (_, _, dk, _) = k_t.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize && dk <= u32::MAX as usize && dv <= u32::MAX as usize,
        "metal gdn full-chunk shape too large"
    );

    let g = g.contiguous()?;
    let v = v.contiguous()?;
    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;
    let beta = beta.contiguous()?;
    let k_t = k_t.contiguous()?;
    let g_metal = kt_metal(&g)?;
    // The full-chunk kernel writes every output token/head/value element.
    let out = kt_metal_alloc(
        g_metal,
        kiln_tensor::DType::BF16,
        &[batch, heads, chunk_size, dv],
    )?;

    let companion = g_metal.companion()?;
    let pipeline = metal_gdn_full_chunk_forward_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_full_chunk_forward_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v)?;
        let kkt_metal = kt_metal(&kkt)?;
        let qkt_metal = kt_metal(&qkt)?;
        let ks_metal = kt_metal(&ks_entry)?;
        let qs_metal = kt_metal(&q_s)?;
        let beta_metal = kt_metal(&beta)?;
        let kt_metal_storage = kt_metal(&k_t)?;
        let state_metal = kt_metal(&state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kkt_buf = buffer_o_kt(kkt_metal.buffer().as_ref(), kkt.layout(), kkt.dtype());
        let qkt_buf = buffer_o_kt(qkt_metal.buffer().as_ref(), qkt.layout(), qkt.dtype());
        let ks_buf = buffer_o_kt(
            ks_metal.buffer().as_ref(),
            ks_entry.layout(),
            ks_entry.dtype(),
        );
        let qs_buf = buffer_o_kt(qs_metal.buffer().as_ref(), q_s.layout(), q_s.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let kt_buf = buffer_o_kt(
            kt_metal_storage.buffer().as_ref(),
            k_t.layout(),
            k_t.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let output_mode_u32 = 0u32;
        let t_start_u32 = 0u32;
        let seq_len_u32 = chunk_size as u32;
        let heads_u32 = heads as u32;
        let g_stride = g.layout().strides();
        let v_stride = v.layout().strides();
        let beta_stride = beta.layout().strides();
        let kt_stride = k_t.layout().strides();
        let g_bh_stride_u32 = g_stride[1] as u32;
        let g_t_stride_u32 = g_stride[2] as u32;
        let v_bh_stride_u32 = v_stride[1] as u32;
        let v_t_stride_u32 = v_stride[2] as u32;
        let v_d_stride_u32 = v_stride[3] as u32;
        let beta_bh_stride_u32 = beta_stride[1] as u32;
        let beta_t_stride_u32 = beta_stride[2] as u32;
        let kt_bh_stride_u32 = kt_stride[1] as u32;
        let kt_k_stride_u32 = kt_stride[2] as u32;
        let kt_t_stride_u32 = kt_stride[3] as u32;
        encoder.set_bytes(10, &batch_heads_u32);
        encoder.set_bytes(11, &dk_u32);
        encoder.set_bytes(12, &dv_u32);
        encoder.set_bytes(13, &output_mode_u32);
        encoder.set_bytes(14, &t_start_u32);
        encoder.set_bytes(15, &seq_len_u32);
        encoder.set_bytes(16, &heads_u32);
        encoder.set_bytes(17, &g_bh_stride_u32);
        encoder.set_bytes(18, &g_t_stride_u32);
        encoder.set_bytes(19, &v_bh_stride_u32);
        encoder.set_bytes(20, &v_t_stride_u32);
        encoder.set_bytes(21, &v_d_stride_u32);
        encoder.set_bytes(22, &beta_bh_stride_u32);
        encoder.set_bytes(23, &beta_t_stride_u32);
        encoder.set_bytes(24, &kt_bh_stride_u32);
        encoder.set_bytes(25, &kt_k_stride_u32);
        encoder.set_bytes(26, &kt_t_stride_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(super) fn metal_gdn_full_chunk_forward_head_last_into_bf16(
    g: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    kkt: &kiln_tensor::Tensor,
    qkt: &kiln_tensor::Tensor,
    ks_entry: &kiln_tensor::Tensor,
    q_s: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    k_t: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
    out: &kiln_tensor::Tensor,
    t_start: usize,
    seq_len: usize,
) -> Result<()> {
    anyhow::ensure!(
        metal_gdn_full_chunk_forward_head_last_supports(
            g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state, out, t_start, seq_len,
        ),
        "metal gdn full-chunk head-last unsupported shape"
    );
    let (batch, heads, _) = g.dims3()?;
    let (_, _, dk, _) = k_t.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && t_start <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && heads <= u32::MAX as usize,
        "metal gdn full-chunk head-last shape too large"
    );

    let kkt = kkt.contiguous()?;
    let qkt = qkt.contiguous()?;
    let ks_entry = ks_entry.contiguous()?;
    let q_s = q_s.contiguous()?;

    let g_metal = kt_metal(&g)?;
    let companion = g_metal.companion()?;
    let pipeline = metal_gdn_full_chunk_forward_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_full_chunk_forward_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let v_metal = kt_metal(&v)?;
        let kkt_metal = kt_metal(&kkt)?;
        let qkt_metal = kt_metal(&qkt)?;
        let ks_metal = kt_metal(&ks_entry)?;
        let qs_metal = kt_metal(&q_s)?;
        let beta_metal = kt_metal(&beta)?;
        let kt_metal_storage = kt_metal(&k_t)?;
        let state_metal = kt_metal(&state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let kkt_buf = buffer_o_kt(kkt_metal.buffer().as_ref(), kkt.layout(), kkt.dtype());
        let qkt_buf = buffer_o_kt(qkt_metal.buffer().as_ref(), qkt.layout(), qkt.dtype());
        let ks_buf = buffer_o_kt(
            ks_metal.buffer().as_ref(),
            ks_entry.layout(),
            ks_entry.dtype(),
        );
        let qs_buf = buffer_o_kt(qs_metal.buffer().as_ref(), q_s.layout(), q_s.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let kt_buf = buffer_o_kt(
            kt_metal_storage.buffer().as_ref(),
            k_t.layout(),
            k_t.dtype(),
        );
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(kkt_buf.buffer), kkt_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(qkt_buf.buffer), qkt_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(ks_buf.buffer), ks_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(qs_buf.buffer), qs_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(7, Some(kt_buf.buffer), kt_buf.offset_in_bytes);
        encoder.set_buffer(8, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(9, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let output_mode_u32 = 1u32;
        let t_start_u32 = t_start as u32;
        let seq_len_u32 = seq_len as u32;
        let heads_u32 = heads as u32;
        let g_stride = g.layout().strides();
        let v_stride = v.layout().strides();
        let beta_stride = beta.layout().strides();
        let kt_stride = k_t.layout().strides();
        let g_bh_stride_u32 = g_stride[1] as u32;
        let g_t_stride_u32 = g_stride[2] as u32;
        let v_bh_stride_u32 = v_stride[1] as u32;
        let v_t_stride_u32 = v_stride[2] as u32;
        let v_d_stride_u32 = v_stride[3] as u32;
        let beta_bh_stride_u32 = beta_stride[1] as u32;
        let beta_t_stride_u32 = beta_stride[2] as u32;
        let kt_bh_stride_u32 = kt_stride[1] as u32;
        let kt_k_stride_u32 = kt_stride[2] as u32;
        let kt_t_stride_u32 = kt_stride[3] as u32;
        encoder.set_bytes(10, &batch_heads_u32);
        encoder.set_bytes(11, &dk_u32);
        encoder.set_bytes(12, &dv_u32);
        encoder.set_bytes(13, &output_mode_u32);
        encoder.set_bytes(14, &t_start_u32);
        encoder.set_bytes(15, &seq_len_u32);
        encoder.set_bytes(16, &heads_u32);
        encoder.set_bytes(17, &g_bh_stride_u32);
        encoder.set_bytes(18, &g_t_stride_u32);
        encoder.set_bytes(19, &v_bh_stride_u32);
        encoder.set_bytes(20, &v_t_stride_u32);
        encoder.set_bytes(21, &v_d_stride_u32);
        encoder.set_bytes(22, &beta_bh_stride_u32);
        encoder.set_bytes(23, &beta_t_stride_u32);
        encoder.set_bytes(24, &kt_bh_stride_u32);
        encoder.set_bytes(25, &kt_k_stride_u32);
        encoder.set_bytes(26, &kt_t_stride_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(())
}

pub(super) fn metal_gdn_recurrent_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    let (batch, heads, dk) = q.dims3()?;
    let dv = v.dim(2)?;
    let batch_heads = batch * heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize && dk <= u32::MAX as usize && dv <= u32::MAX as usize,
        "metal gdn recurrent shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    // The recurrent kernel writes every batch/head/value element.
    let out = kt_metal_alloc(q_metal, kiln_tensor::DType::BF16, &[batch, heads, dv])?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &dk_u32);
        encoder.set_bytes(9, &dv_u32);

        let threads_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(super) fn metal_gdn_recurrent_prefill_head_last_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_head_last_supports(q, k, v, beta, g, state),
        "metal gdn recurrent prefill unsupported shape"
    );
    let (batch, q_heads, seq_len, dk) = q.dims4()?;
    let (_, value_heads, _, _) = v.dims4()?;
    let dv = v.dim(3)?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent prefill shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    // SAFETY: the kernel dispatch covers every (batch, token, value-head, dv)
    // output element exactly once via `gid=batch_head*dv+d` and the token loop.
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_prefill_head_last_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 0u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(super) fn metal_gdn_recurrent_prefill_native_head_last_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    g: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_native_head_last_supports(q, k, v, beta, g, state),
        "metal gdn recurrent native prefill unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent native prefill shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    // SAFETY: the kernel dispatch covers every (batch, token, value-head, dv)
    // output element exactly once via `gid=batch_head*dv+d` and the token loop.
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_prefill_head_last_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_native_head_last_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let g_metal = kt_metal(&g)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let g_buf = buffer_o_kt(g_metal.buffer().as_ref(), g.layout(), g.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(g_buf.buffer), g_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 1u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

pub(crate) fn metal_gdn_recurrent_prefill_native_head_last_decay_bf16(
    q: &kiln_tensor::Tensor,
    k: &kiln_tensor::Tensor,
    v: &kiln_tensor::Tensor,
    beta: &kiln_tensor::Tensor,
    decay: &kiln_tensor::Tensor,
    state: &mut kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    anyhow::ensure!(
        metal_gdn_recurrent_prefill_native_head_last_decay_supports(q, k, v, beta, decay, state),
        "metal gdn recurrent native prefill decay unsupported shape"
    );
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;
    let batch_heads = batch * value_heads;
    anyhow::ensure!(
        batch_heads <= u32::MAX as usize
            && seq_len <= u32::MAX as usize
            && dk <= u32::MAX as usize
            && dv <= u32::MAX as usize
            && value_heads <= u32::MAX as usize
            && q_heads <= u32::MAX as usize,
        "metal gdn recurrent native prefill decay shape too large"
    );

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let decay = decay.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }
    let q_metal = kt_metal(&q)?;
    let out = kt_metal_alloc(
        q_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, value_heads, dv],
    )?;

    let companion = q_metal.companion()?;
    let pipeline = metal_gdn_recurrent_prefill_head_last_decay_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_recurrent_prefill_native_head_last_decay_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;
        let beta_metal = kt_metal(&beta)?;
        let decay_metal = kt_metal(&decay)?;
        let state_metal = kt_metal(state)?;
        let out_metal = kt_metal(&out)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());
        let beta_buf = buffer_o_kt(beta_metal.buffer().as_ref(), beta.layout(), beta.dtype());
        let decay_buf = buffer_o_kt(decay_metal.buffer().as_ref(), decay.layout(), decay.dtype());
        let state_buf = buffer_o_kt(state_metal.buffer().as_ref(), state.layout(), state.dtype());
        let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());

        encoder.set_buffer(0, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(v_buf.buffer), v_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(beta_buf.buffer), beta_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(decay_buf.buffer), decay_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(state_buf.buffer), state_buf.offset_in_bytes);
        encoder.set_buffer(6, Some(out_buf.buffer), out_buf.offset_in_bytes);

        let batch_heads_u32 = batch_heads as u32;
        let seq_len_u32 = seq_len as u32;
        let dk_u32 = dk as u32;
        let dv_u32 = dv as u32;
        let value_heads_u32 = value_heads as u32;
        let q_heads_u32 = q_heads as u32;
        let input_mode_u32 = 1u32;
        encoder.set_bytes(7, &batch_heads_u32);
        encoder.set_bytes(8, &seq_len_u32);
        encoder.set_bytes(9, &dk_u32);
        encoder.set_bytes(10, &dv_u32);
        encoder.set_bytes(11, &value_heads_u32);
        encoder.set_bytes(12, &q_heads_u32);
        encoder.set_bytes(13, &input_mode_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch_heads * dv,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok(out)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_prefill_qkv_conv_split_supports(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> bool {
    if metal_gdn_prefill_qkv_conv_split_disabled() {
        return false;
    }
    if mixed_qkv.dtype() != kiln_tensor::DType::BF16
        || weight.dtype() != kiln_tensor::DType::BF16
        || conv_state.dtype() != kiln_tensor::DType::F32
    {
        return false;
    }
    if !matches!(mixed_qkv.device(), kiln_tensor::Device::Metal(_))
        || !matches!(weight.device(), kiln_tensor::Device::Metal(_))
        || !matches!(conv_state.device(), kiln_tensor::Device::Metal(_))
    {
        return false;
    }
    if !mixed_qkv.is_contiguous() || !weight.is_contiguous() || !conv_state.is_contiguous() {
        return false;
    }
    let Ok((batch, seq_len, channels)) = mixed_qkv.dims3() else {
        return false;
    };
    let Ok((s_batch, s_channels, s_width)) = conv_state.dims3() else {
        return false;
    };
    let qk_dim = nk.saturating_mul(dk);
    let v_dim = nv.saturating_mul(dv);
    let Some(expected_channels) = qk_dim.checked_mul(2).and_then(|n| n.checked_add(v_dim)) else {
        return false;
    };
    let weight_ok = match weight.rank() {
        2 => weight.dims() == [channels, kernel_size],
        3 => weight.dims() == [channels, 1, kernel_size],
        _ => false,
    };
    batch <= u32::MAX as usize
        && seq_len > 1
        && seq_len <= u32::MAX as usize
        && channels == expected_channels
        && channels <= u32::MAX as usize
        && qk_dim <= u32::MAX as usize
        && v_dim <= u32::MAX as usize
        && kernel_size == 4
        && (s_batch, s_channels, s_width) == (batch, channels, 3)
        && weight_ok
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn metal_gdn_prefill_qkv_conv_split_bf16_f32_k4(
    mixed_qkv: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    conv_state: &mut kiln_tensor::Tensor,
    kernel_size: usize,
    nk: usize,
    dk: usize,
    nv: usize,
    dv: usize,
) -> Result<(
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
    kiln_tensor::Tensor,
)> {
    anyhow::ensure!(
        metal_gdn_prefill_qkv_conv_split_supports(
            mixed_qkv,
            weight,
            conv_state,
            kernel_size,
            nk,
            dk,
            nv,
            dv,
        ),
        "metal gdn prefill qkv conv-split unsupported shape"
    );
    let (batch, seq_len, channels) = mixed_qkv.dims3()?;
    let qk_dim = nk * dk;
    let v_dim = nv * dv;

    let weight = match weight.rank() {
        3 => weight.reshape((channels, kernel_size))?,
        2 => weight.clone(),
        r => anyhow::bail!("metal gdn prefill qkv conv-split weight rank must be 2 or 3, got {r}"),
    }
    .contiguous()?;
    if !conv_state.is_contiguous() {
        *conv_state = conv_state.contiguous()?;
    }
    let mixed_qkv_metal = kt_metal(mixed_qkv)?;
    let q = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::F32,
        &[batch, seq_len, nk, dk],
    )?;
    let k = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::F32,
        &[batch, seq_len, nk, dk],
    )?;
    let v = kt_metal_alloc(
        mixed_qkv_metal,
        kiln_tensor::DType::BF16,
        &[batch, seq_len, nv, dv],
    )?;

    let companion = mixed_qkv_metal.companion()?;
    let pipeline = metal_gdn_prefill_qkv_conv_split_pipeline(&*companion)?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4");
    encoder.set_compute_pipeline_state(&pipeline);

    {
        let w_metal = kt_metal(&weight)?;
        let s_metal = kt_metal(conv_state)?;
        let q_metal = kt_metal(&q)?;
        let k_metal = kt_metal(&k)?;
        let v_metal = kt_metal(&v)?;

        // #1082 Step 4 gdn-family: `buffer_o` → `buffer_o_kt`.
        let x_buf = buffer_o_kt(
            mixed_qkv_metal.buffer().as_ref(),
            mixed_qkv.layout(),
            mixed_qkv.dtype(),
        );
        let w_buf = buffer_o_kt(w_metal.buffer().as_ref(), weight.layout(), weight.dtype());
        let s_buf = buffer_o_kt(
            s_metal.buffer().as_ref(),
            conv_state.layout(),
            conv_state.dtype(),
        );
        let q_buf = buffer_o_kt(q_metal.buffer().as_ref(), q.layout(), q.dtype());
        let k_buf = buffer_o_kt(k_metal.buffer().as_ref(), k.layout(), k.dtype());
        let v_buf = buffer_o_kt(v_metal.buffer().as_ref(), v.layout(), v.dtype());

        encoder.set_buffer(0, Some(x_buf.buffer), x_buf.offset_in_bytes);
        encoder.set_buffer(1, Some(w_buf.buffer), w_buf.offset_in_bytes);
        encoder.set_buffer(2, Some(s_buf.buffer), s_buf.offset_in_bytes);
        encoder.set_buffer(3, Some(q_buf.buffer), q_buf.offset_in_bytes);
        encoder.set_buffer(4, Some(k_buf.buffer), k_buf.offset_in_bytes);
        encoder.set_buffer(5, Some(v_buf.buffer), v_buf.offset_in_bytes);

        let batch_u32 = batch as u32;
        let seq_len_u32 = seq_len as u32;
        let channels_u32 = channels as u32;
        let qk_dim_u32 = qk_dim as u32;
        let v_dim_u32 = v_dim as u32;
        let threads = seq_len.next_power_of_two().clamp(32, 256);
        let threads_u32 = threads as u32;
        encoder.set_bytes(6, &batch_u32);
        encoder.set_bytes(7, &seq_len_u32);
        encoder.set_bytes(8, &channels_u32);
        encoder.set_bytes(9, &qk_dim_u32);
        encoder.set_bytes(10, &v_dim_u32);
        encoder.set_bytes(11, &threads_u32);

        let threadgroups_per_grid = objc2_metal::MTLSize {
            width: batch * channels,
            height: 1,
            depth: 1,
        };
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(threadgroups_per_grid, threads_per_threadgroup);
    }

    Ok((q, k, v))
}

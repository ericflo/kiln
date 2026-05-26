//! `kiln_tensor::Tensor`-typed surface for the GDN forward-
//! substitution kernel (the namesake hot path).
//!
//! Phase 7 prep — same pattern as kiln-flash-attn (#1316/#1317),
//! kiln-conv1d-kernel (#1318), kiln-rmsnorm-kernel (#1319),
//! kiln-marlin-gemm (#1320). Same FFI; only the Rust shell types
//! switch from `candle_core::Tensor` to `kiln_tensor::Tensor`.
//!
//! Only `gdn_forward_substitution` ships in this PR; the remaining
//! 16 FFI entry points (recurrent variants, chunk_prep, chunk_scan,
//! full_chunk_forward, etc.) follow the same template.

use kiln_kt_bridge::BridgeError;
use kiln_tensor::{CudaStorage, DType as KtDType, Device as KtDevice, Tensor as KtTensor};

use crate::{
    kiln_gdn_decode_gates_recurrent_bf16, kiln_gdn_decode_gates_recurrent_vf32_bf16,
    kiln_gdn_decode_qk_norm_gates_recurrent_bf16,
    kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vbf16_bf16,
    kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vf32_bf16,
    kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16,
    kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vbf16_bf16,
    kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vf32_bf16,
    kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_vf32_bf16,
    kiln_gdn_chunk_prep, kiln_gdn_chunk_scan, kiln_gdn_decode_qk_norm_gates_recurrent_vf32_bf16,
    kiln_gdn_forward_substitution, kiln_gdn_full_chunk_forward,
    kiln_gdn_full_chunk_forward_multiblock, kiln_gdn_gated_rms_norm_bf16, kiln_gdn_gates_bf16,
    kiln_gdn_gates_bf16_f32_bf16_params, kiln_gdn_gates_bf16_f32_params, kiln_gdn_recurrent_forward,
};

#[derive(Debug)]
pub enum GdnError {
    Msg(String),
}

impl std::fmt::Display for GdnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GdnError::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for GdnError {}

impl From<BridgeError> for GdnError {
    fn from(e: BridgeError) -> Self {
        GdnError::Msg(e.message)
    }
}

fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), GdnError> {
    Ok(kiln_kt_bridge::cuda_storage_and_byte_offset(t, expected, name)?)
}

fn alloc_cuda_tensor(
    source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, GdnError> {
    Ok(kiln_kt_bridge::alloc_cuda_tensor(source, dtype, shape)?)
}

/// `gdn_forward_substitution` over `kiln_tensor::Tensor` operands.
///
/// Shapes:
/// - `a_strict`: BF16 `[B, H, C, C]` (square in last two dims)
/// - `v_prime`:  BF16 `[B, H, C, dv]`
/// - `beta`:     BF16 `[B, H, C]`
///
/// Returns BF16 `W` of shape `[B, H, C, dv]`.
pub fn gdn_forward_substitution_kt(
    a_strict: &KtTensor,
    v_prime: &KtTensor,
    beta: &KtTensor,
) -> Result<KtTensor, GdnError> {
    let a_shape = a_strict.shape();
    if a_shape.len() != 4 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: a_strict must be rank-4, got {a_shape:?}"
        )));
    }
    let (b, h, c, c2) = (a_shape[0], a_shape[1], a_shape[2], a_shape[3]);
    if c != c2 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: a_strict must be square on last two axes, got [_, _, {c}, {c2}]"
        )));
    }
    let vp_shape = v_prime.shape();
    if vp_shape.len() != 4 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: v_prime must be rank-4, got {vp_shape:?}"
        )));
    }
    let dv = vp_shape[3];
    if (vp_shape[0], vp_shape[1], vp_shape[2]) != (b, h, c) {
        return Err(GdnError::Msg(format!(
            "kt-gdn: v_prime {vp_shape:?} mismatch with a_strict prefix [{b}, {h}, {c}]"
        )));
    }
    let bt_shape = beta.shape();
    if bt_shape.len() != 3 || (bt_shape[0], bt_shape[1], bt_shape[2]) != (b, h, c) {
        return Err(GdnError::Msg(format!(
            "kt-gdn: beta {bt_shape:?} mismatch with [{b}, {h}, {c}]"
        )));
    }
    if c > 128 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk_size must be <= 128, got {c}"
        )));
    }
    if dv > 1024 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: dv must be <= 1024, got {dv}"
        )));
    }

    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_strict, KtDType::BF16, "a_strict")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a_strict, KtDType::BF16, "a_strict")?;
    let vp_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_prime, KtDType::BF16, "v_prime")?;
    let (vp_st, _) = cuda_storage_and_byte_offset(v_prime, KtDType::BF16, "v_prime")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let (bt_st, _) = cuda_storage_and_byte_offset(beta, KtDType::BF16, "beta")?;

    let w_out = alloc_cuda_tensor(a_st, KtDType::BF16, vec![b, h, c, dv])?;
    let w_ptr = kiln_kt_bridge::cuda_output_device_ptr(&w_out);
    

    let raw_stream = a_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_forward_substitution(
            a_ptr as *const _,
            vp_ptr as *const _,
            bt_ptr as *const _,
            w_ptr as *mut _,
            (b * h) as i32,
            c as i32,
            dv as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!("kt-gdn: FFI returned {status}")));
    }
    Ok(w_out)
}

/// `gdn_recurrent_forward` over `kiln_tensor::Tensor` operands.
///
/// Single-token decode path. Inputs: q/k/v/beta/g BF16, state BF16
/// `[B, H, dk, dv]` (mutated in place). Returns BF16 `[B, H, dv]`.
///
/// `state` is borrowed `&KtTensor`; the FFI mutates its underlying
/// CUDA buffer through the raw device pointer (same idiom as the
/// candle-typed wrapper which takes `&mut Tensor`).
pub fn gdn_recurrent_forward_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    beta: &KtTensor,
    g: &KtTensor,
    state: &KtTensor,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: recurrent q must be [B, H, dk], got {q_shape:?}"
        )));
    }
    let (b, h, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    if k.shape() != q_shape {
        return Err(GdnError::Msg(format!(
            "kt-gdn: recurrent k {:?} != q {q_shape:?}",
            k.shape()
        )));
    }
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != b || v_shape[1] != h {
        return Err(GdnError::Msg(format!(
            "kt-gdn: recurrent v {v_shape:?} must be [{b}, {h}, dv]"
        )));
    }
    let dv = v_shape[2];
    if beta.shape() != [b, h] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: recurrent beta {:?} != [{b}, {h}]",
            beta.shape()
        )));
    }
    if g.shape() != [b, h] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: recurrent g {:?} != [{b}, {h}]",
            g.shape()
        )));
    }
    let st_shape = state.shape();
    if st_shape.len() != 4 || (st_shape[0], st_shape[1], st_shape[2], st_shape[3]) != (b, h, dk, dv)
    {
        return Err(GdnError::Msg(format!(
            "kt-gdn: recurrent state {st_shape:?} != [{b}, {h}, {dk}, {dv}]"
        )));
    }
    if dk > 256 {
        return Err(GdnError::Msg(format!("kt-gdn: dk must be <= 256, got {dk}")));
    }
    if dv > 1024 {
        return Err(GdnError::Msg(format!("kt-gdn: dv must be <= 1024, got {dv}")));
    }

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let (bt_st, _) = cuda_storage_and_byte_offset(beta, KtDType::BF16, "beta")?;
    let g_ptr = kiln_kt_bridge::cuda_input_device_ptr(g, KtDType::BF16, "g")?;
    let (g_st, _) = cuda_storage_and_byte_offset(g, KtDType::BF16, "g")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, h, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_recurrent_forward(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            bt_ptr as *const _,
            g_ptr as *const _,
            s_ptr as *mut _,
            o_ptr as *mut _,
            (b * h) as i32,
            dk as i32,
            dv as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!("kt-gdn: recurrent FFI returned {status}")));
    }
    Ok(out)
}

/// `gdn_decode_qk_norm_gates_recurrent_rmsnorm` (BF16 q/v variant)
/// over `kiln_tensor::Tensor` operands.
///
/// The hottest decode path on Qwen3.5-4B. Fuses QK-norm + delta-net
/// gate update + recurrent state advance + RMS-norm of the value
/// projection in one kernel launch.
///
/// Shapes:
/// - `q`, `k`: BF16 `[B, q_heads, dk]`
/// - `v`:     BF16 `[B, value_heads, dv]`
/// - `a`, `b`, `a_log`, `dt_bias`: BF16 (per-head scalars; shape
///   `[B, q_heads]` for a/b/a_log, `[q_heads]` for dt_bias)
/// - `state`: BF16 `[B, value_heads, dk, dv]` — mutated in place
/// - `z`:     BF16 `[B, value_heads, dv]` gate input
/// - `weight`: F32 `[dv]` RMS-norm gain
///
/// Returns BF16 `[B, value_heads, dv]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    z: &KtTensor,
    weight: &KtTensor,
    q_scale: f32,
    qk_eps: f32,
    rms_eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    if k.shape() != q_shape {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode k {:?} != q {q_shape:?}",
            k.shape()
        )));
    }
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let (z_st, _) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (w_st, _) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            z_ptr as *const _,
            w_ptr as *const _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            q_scale,
            qk_eps,
            rms_eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_qk_norm_rmsnorm FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `gdn_full_chunk_forward` over `kiln_tensor::Tensor` operands.
///
/// The prefill chunk hot path. Takes 8 BF16 inputs (g/v/kkt/qkt/
/// ks_entry/q_s/beta/k_t) plus an in-place BF16 `state` tensor; the
/// kernel writes `out_chunk` BF16 `[B, H, C, dv]`.
///
/// Envelope:
/// - `g`:        `[B, H, C]`
/// - `v`:        `[B, H, C, dv]`
/// - `kkt`,`qkt`: `[B, H, C, C]`
/// - `ks_entry`,`q_s`: `[B, H, C, dv]`
/// - `beta`:     `[B, H, C]`
/// - `k_t`:      `[B, H, dk, C]` (`dk <= 128`)
/// - `state`:    `[B, H, dk, dv]` — mutated in place
///
/// Returns BF16 `[B, H, C, dv]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_full_chunk_forward_kt(
    g: &KtTensor,
    v: &KtTensor,
    kkt: &KtTensor,
    qkt: &KtTensor,
    ks_entry: &KtTensor,
    q_s: &KtTensor,
    beta: &KtTensor,
    k_t: &KtTensor,
    state: &KtTensor,
) -> Result<KtTensor, GdnError> {
    let g_shape = g.shape();
    if g_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk g must be [B, H, C], got {g_shape:?}"
        )));
    }
    let (b, h, c) = (g_shape[0], g_shape[1], g_shape[2]);

    let v_shape = v.shape();
    if v_shape.len() != 4
        || (v_shape[0], v_shape[1], v_shape[2]) != (b, h, c)
    {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk v {v_shape:?} != [{b}, {h}, {c}, dv]"
        )));
    }
    let dv = v_shape[3];

    if kkt.shape() != [b, h, c, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk kkt {:?} != [{b}, {h}, {c}, {c}]",
            kkt.shape()
        )));
    }
    if qkt.shape() != [b, h, c, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk qkt {:?} != [{b}, {h}, {c}, {c}]",
            qkt.shape()
        )));
    }
    if ks_entry.shape() != [b, h, c, dv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk ks_entry {:?} != [{b}, {h}, {c}, {dv}]",
            ks_entry.shape()
        )));
    }
    if q_s.shape() != [b, h, c, dv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk q_s {:?} != [{b}, {h}, {c}, {dv}]",
            q_s.shape()
        )));
    }
    if beta.shape() != [b, h, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk beta {:?} != [{b}, {h}, {c}]",
            beta.shape()
        )));
    }
    let kt_shape = k_t.shape();
    if kt_shape.len() != 4
        || (kt_shape[0], kt_shape[1], kt_shape[3]) != (b, h, c)
        || kt_shape[2] == 0
        || kt_shape[2] > 128
    {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk k_t {kt_shape:?} != [{b}, {h}, dk in 1..=128, {c}]"
        )));
    }
    let dk = kt_shape[2];
    if state.shape() != [b, h, dk, dv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk state {:?} != [{b}, {h}, {dk}, {dv}]",
            state.shape()
        )));
    }

    let g_ptr = kiln_kt_bridge::cuda_input_device_ptr(g, KtDType::BF16, "g")?;
    let (g_st, _) = cuda_storage_and_byte_offset(g, KtDType::BF16, "g")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let kkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(kkt, KtDType::BF16, "kkt")?;
    let (kkt_st, _) = cuda_storage_and_byte_offset(kkt, KtDType::BF16, "kkt")?;
    let qkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(qkt, KtDType::BF16, "qkt")?;
    let (qkt_st, _) = cuda_storage_and_byte_offset(qkt, KtDType::BF16, "qkt")?;
    let ks_ptr = kiln_kt_bridge::cuda_input_device_ptr(ks_entry, KtDType::BF16, "ks_entry")?;
    let (ks_st, _) = cuda_storage_and_byte_offset(ks_entry, KtDType::BF16, "ks_entry")?;
    let qs_ptr = kiln_kt_bridge::cuda_input_device_ptr(q_s, KtDType::BF16, "q_s")?;
    let (qs_st, _) = cuda_storage_and_byte_offset(q_s, KtDType::BF16, "q_s")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let (bt_st, _) = cuda_storage_and_byte_offset(beta, KtDType::BF16, "beta")?;
    let kt_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_t, KtDType::BF16, "k_t")?;
    let (kt_st, _) = cuda_storage_and_byte_offset(k_t, KtDType::BF16, "k_t")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = g_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_full_chunk_forward(
            g_ptr as *const _,
            v_ptr as *const _,
            kkt_ptr as *const _,
            qkt_ptr as *const _,
            ks_ptr as *const _,
            qs_ptr as *const _,
            bt_ptr as *const _,
            kt_ptr as *const _,
            s_ptr as *mut _,
            o_ptr as *mut _,
            (b * h) as i32,
            c as i32,
            dk as i32,
            dv as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: full_chunk_forward FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `gdn_decode_gates_recurrent_bf16` over kt operands.
///
/// Decode path WITHOUT QK-norm (the rmsnorm-free variant). Same
/// shape contract as decode_qk_norm_*_rmsnorm but takes a single
/// `eps` and a `weight` for the value norm.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_gates_recurrent_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    z: &KtTensor,
    weight: &KtTensor,
    eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-gates q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-gates v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let (z_st, _) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (w_st, _) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_gates_recurrent_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            z_ptr as *const _,
            w_ptr as *const _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_gates_recurrent FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `gdn_decode_qk_norm_gates_recurrent_bf16` over kt operands.
///
/// QK-norm variant without the rmsnorm output gating. Takes
/// `(q_scale, qk_eps)` instead of `(q_scale, qk_eps, rms_eps)` +
/// `weight`. Returns BF16 `[B, value_heads, dv]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    q_scale: f32,
    qk_eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-qk-norm q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-qk-norm v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_qk_norm_gates_recurrent_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            q_scale,
            qk_eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_qk_norm_gates_recurrent FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `kiln_gdn_decode_gates_recurrent_vf32_bf16` over kt operands.
///
/// `vf32` variant: `v` is `F32 [B, value_heads, dv]`; q/k stay BF16.
/// Returns BF16 `[B, value_heads, dv]`. Useful when v-cache or
/// projection runs at higher precision while q/k stay BF16.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_gates_recurrent_vf32_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    z: &KtTensor,
    weight: &KtTensor,
    eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-gates-vf32 q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-gates-vf32 v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let (z_st, _) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (w_st, _) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_gates_recurrent_vf32_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            z_ptr as *const _,
            w_ptr as *const _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_gates_recurrent_vf32 FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `kiln_gdn_decode_qk_norm_gates_recurrent_vf32_bf16` — qk-norm + vf32.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_vf32_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    q_scale: f32,
    qk_eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-qk-norm-vf32 q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-qk-norm-vf32 v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_qk_norm_gates_recurrent_vf32_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            q_scale,
            qk_eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_qk_norm_gates_recurrent_vf32 FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vf32_bf16` — qk-norm with q+v in F32.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_qf32_vf32_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    q_scale: f32,
    qk_eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-qk-norm-qf32-vf32 q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-qk-norm-qf32-vf32 v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::F32, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vf32_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            q_scale,
            qk_eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_qk_norm_gates_recurrent_qf32_vf32 FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vbf16_bf16` — qk-norm with q in F32, v in BF16.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_qf32_vbf16_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    q_scale: f32,
    qk_eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-qk-norm-qf32-vbf16 q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-qk-norm-qf32-vbf16 v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::F32, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vbf16_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            q_scale,
            qk_eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_qk_norm_gates_recurrent_qf32_vbf16 FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_vf32_bf16` — qk-norm + rmsnorm + vf32.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_rmsnorm_vf32_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    z: &KtTensor,
    weight: &KtTensor,
    q_scale: f32,
    qk_eps: f32,
    rms_eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-rmsnorm-vf32 q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-rmsnorm-vf32 v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let (z_st, _) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (w_st, _) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_vf32_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            z_ptr as *const _,
            w_ptr as *const _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            q_scale,
            qk_eps,
            rms_eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_rmsnorm_vf32 FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vf32_bf16` — qk-norm + rmsnorm + qf32 + vf32.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vf32_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    z: &KtTensor,
    weight: &KtTensor,
    q_scale: f32,
    qk_eps: f32,
    rms_eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-rmsnorm-qf32-vf32 q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-rmsnorm-qf32-vf32 v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::F32, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let (z_st, _) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (w_st, _) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vf32_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            z_ptr as *const _,
            w_ptr as *const _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            q_scale,
            qk_eps,
            rms_eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_rmsnorm_qf32_vf32 FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vbf16_bf16` — qk-norm + rmsnorm + qf32 + vbf16.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vbf16_bf16_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b_param: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    z: &KtTensor,
    weight: &KtTensor,
    q_scale: f32,
    qk_eps: f32,
    rms_eps: f32,
) -> Result<KtTensor, GdnError> {
    let q_shape = q.shape();
    if q_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-rmsnorm-qf32-vbf16 q must be [B, q_heads, dk], got {q_shape:?}"
        )));
    }
    let (batch, q_heads, dk) = (q_shape[0], q_shape[1], q_shape[2]);
    let v_shape = v.shape();
    if v_shape.len() != 3 || v_shape[0] != batch {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode-rmsnorm-qf32-vbf16 v {v_shape:?} != [{batch}, value_heads, dv]"
        )));
    }
    let value_heads = v_shape[1];
    let dv = v_shape[2];

    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::F32, "q")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let (k_st, _) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let (bp_st, _) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let (z_st, _) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (w_st, _) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = q_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vbf16_bf16(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            a_ptr as *const _,
            bp_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            s_ptr as *mut _,
            z_ptr as *const _,
            w_ptr as *const _,
            o_ptr as *mut _,
            batch as i32,
            q_heads as i32,
            value_heads as i32,
            dk as i32,
            dv as i32,
            q_scale,
            qk_eps,
            rms_eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: decode_rmsnorm_qf32_vbf16 FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `gdn_full_chunk_forward_multiblock` over `kiln_tensor::Tensor` operands.
///
/// Multiblock variant of [`gdn_full_chunk_forward_kt`]. Identical
/// envelope/dtypes; takes an additional `dv_tile` parameter that
/// controls block tiling along the `dv` axis (used when `dv` exceeds
/// the single-block kernel's hardware tile).
///
/// Returns BF16 `[B, H, C, dv]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_full_chunk_forward_multiblock_kt(
    g: &KtTensor,
    v: &KtTensor,
    kkt: &KtTensor,
    qkt: &KtTensor,
    ks_entry: &KtTensor,
    q_s: &KtTensor,
    beta: &KtTensor,
    k_t: &KtTensor,
    state: &KtTensor,
    dv_tile: usize,
) -> Result<KtTensor, GdnError> {
    let g_shape = g.shape();
    if g_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb g must be [B, H, C], got {g_shape:?}"
        )));
    }
    let (b, h, c) = (g_shape[0], g_shape[1], g_shape[2]);

    let v_shape = v.shape();
    if v_shape.len() != 4 || (v_shape[0], v_shape[1], v_shape[2]) != (b, h, c) {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb v {v_shape:?} != [{b}, {h}, {c}, dv]"
        )));
    }
    let dv = v_shape[3];

    if kkt.shape() != [b, h, c, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb kkt {:?} != [{b}, {h}, {c}, {c}]",
            kkt.shape()
        )));
    }
    if qkt.shape() != [b, h, c, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb qkt {:?} != [{b}, {h}, {c}, {c}]",
            qkt.shape()
        )));
    }
    if ks_entry.shape() != [b, h, c, dv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb ks_entry {:?} != [{b}, {h}, {c}, {dv}]",
            ks_entry.shape()
        )));
    }
    if q_s.shape() != [b, h, c, dv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb q_s {:?} != [{b}, {h}, {c}, {dv}]",
            q_s.shape()
        )));
    }
    if beta.shape() != [b, h, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb beta {:?} != [{b}, {h}, {c}]",
            beta.shape()
        )));
    }
    let kt_shape = k_t.shape();
    if kt_shape.len() != 4
        || (kt_shape[0], kt_shape[1], kt_shape[3]) != (b, h, c)
        || kt_shape[2] == 0
        || kt_shape[2] > 128
    {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb k_t {kt_shape:?} != [{b}, {h}, dk in 1..=128, {c}]"
        )));
    }
    let dk = kt_shape[2];
    if state.shape() != [b, h, dk, dv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb state {:?} != [{b}, {h}, {dk}, {dv}]",
            state.shape()
        )));
    }
    if dv_tile == 0 || dv_tile > dv {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-mb dv_tile {dv_tile} must be in 1..={dv}"
        )));
    }

    let g_ptr = kiln_kt_bridge::cuda_input_device_ptr(g, KtDType::BF16, "g")?;
    let (g_st, _) = cuda_storage_and_byte_offset(g, KtDType::BF16, "g")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let kkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(kkt, KtDType::BF16, "kkt")?;
    let (kkt_st, _) = cuda_storage_and_byte_offset(kkt, KtDType::BF16, "kkt")?;
    let qkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(qkt, KtDType::BF16, "qkt")?;
    let (qkt_st, _) = cuda_storage_and_byte_offset(qkt, KtDType::BF16, "qkt")?;
    let ks_ptr = kiln_kt_bridge::cuda_input_device_ptr(ks_entry, KtDType::BF16, "ks_entry")?;
    let (ks_st, _) = cuda_storage_and_byte_offset(ks_entry, KtDType::BF16, "ks_entry")?;
    let qs_ptr = kiln_kt_bridge::cuda_input_device_ptr(q_s, KtDType::BF16, "q_s")?;
    let (qs_st, _) = cuda_storage_and_byte_offset(q_s, KtDType::BF16, "q_s")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let (bt_st, _) = cuda_storage_and_byte_offset(beta, KtDType::BF16, "beta")?;
    let kt_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_t, KtDType::BF16, "k_t")?;
    let (kt_st, _) = cuda_storage_and_byte_offset(k_t, KtDType::BF16, "k_t")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (s_st, _) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, dv])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = g_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_full_chunk_forward_multiblock(
            g_ptr as *const _,
            v_ptr as *const _,
            kkt_ptr as *const _,
            qkt_ptr as *const _,
            ks_ptr as *const _,
            qs_ptr as *const _,
            bt_ptr as *const _,
            kt_ptr as *const _,
            s_ptr as *mut _,
            o_ptr as *mut _,
            (b * h) as i32,
            c as i32,
            dk as i32,
            dv as i32,
            dv_tile as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: full_chunk_forward_multiblock FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `gdn_gated_rms_norm_bf16` over kt operands.
///
/// Fused gated RMS-norm. Inputs:
/// - `x`:      BF16 `[rows, hidden]` — input activations
/// - `z`:      BF16 `[rows, hidden]` — gate
/// - `weight`: BF16 `[hidden]` — learned per-channel scale
///
/// Returns BF16 `[rows, hidden]`. Computes
/// `out = (x / rms(x)) * weight * silu(z)` in one pass.
pub fn gdn_gated_rms_norm_bf16_kt(
    x: &KtTensor,
    z: &KtTensor,
    weight: &KtTensor,
    eps: f32,
) -> Result<KtTensor, GdnError> {
    let x_shape = x.shape();
    if x_shape.len() != 2 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gated-rmsnorm x must be [rows, hidden], got {x_shape:?}"
        )));
    }
    let (rows, hidden) = (x_shape[0], x_shape[1]);
    if z.shape() != x_shape {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gated-rmsnorm z {:?} != x {x_shape:?}",
            z.shape()
        )));
    }
    if weight.shape() != [hidden] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gated-rmsnorm weight {:?} != [{hidden}]",
            weight.shape()
        )));
    }

    let x_ptr = kiln_kt_bridge::cuda_input_device_ptr(x, KtDType::BF16, "x")?;
    let (x_st, _) = cuda_storage_and_byte_offset(x, KtDType::BF16, "x")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let (z_st, _) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::BF16, "weight")?;
    let (w_st, _) = cuda_storage_and_byte_offset(weight, KtDType::BF16, "weight")?;

    let out = alloc_cuda_tensor(x_st, KtDType::BF16, vec![rows, hidden])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);
    

    let raw_stream = x_st.cuda_stream_raw();
    let status = unsafe {        kiln_gdn_gated_rms_norm_bf16(
            x_ptr as *const _,
            z_ptr as *const _,
            w_ptr as *const _,
            o_ptr as *mut _,
            rows as i32,
            hidden as i32,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gated_rms_norm FFI returned {status}"
        )));
    }
    Ok(out)
}

/// Shared shape/dtype check for the gates family. Returns
/// `(rows, nv, out_shape)`.
fn gates_validate_inputs(
    a: &KtTensor,
    b: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    expected_al_dtype: KtDType,
    expected_dt_dtype: KtDType,
) -> Result<(usize, usize, Vec<usize>), GdnError> {
    let a_shape = a.shape();
    if a_shape.is_empty() {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates a must have >= 1 axes, got {a_shape:?}"
        )));
    }
    let nv = *a_shape.last().unwrap();
    if nv == 0 {
        return Err(GdnError::Msg("kt-gdn: gates nv must be >= 1".into()));
    }
    if nv > 256 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates nv {nv} exceeds kernel limit (256)"
        )));
    }
    if b.shape() != a_shape {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates b {:?} != a {a_shape:?}",
            b.shape()
        )));
    }
    if a_log.shape() != [nv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates a_log {:?} != [{nv}]",
            a_log.shape()
        )));
    }
    if dt_bias.shape() != [nv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates dt_bias {:?} != [{nv}]",
            dt_bias.shape()
        )));
    }
    if a.dtype() != KtDType::BF16 || b.dtype() != KtDType::BF16 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates a/b must be BF16, got a={}, b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    if a_log.dtype() != expected_al_dtype {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates a_log must be {}, got {}",
            expected_al_dtype,
            a_log.dtype()
        )));
    }
    if dt_bias.dtype() != expected_dt_dtype {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates dt_bias must be {}, got {}",
            expected_dt_dtype,
            dt_bias.dtype()
        )));
    }
    let rows = a_shape.iter().take(a_shape.len() - 1).product::<usize>();
    Ok((rows, nv, a_shape.to_vec()))
}

/// `kiln_gdn_gates_bf16` over kt operands.
///
/// Fused beta + g gate kernel. Both `A_log` and `dt_bias` are BF16.
///
/// Inputs:
/// - `a`, `b`: BF16, shape `[.., nv]`. Last-axis-contiguous required.
/// - `A_log`, `dt_bias`: BF16, `[nv]`.
///
/// Returns `(beta_out, g_out)`, both BF16 with the same shape as `a`.
pub fn gdn_gates_bf16_kt(
    a: &KtTensor,
    b: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
) -> Result<(KtTensor, KtTensor), GdnError> {
    let (rows, nv, out_shape) =
        gates_validate_inputs(a, b, a_log, dt_bias, KtDType::BF16, KtDType::BF16)?;

    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let b_ptr = kiln_kt_bridge::cuda_input_device_ptr(b, KtDType::BF16, "b")?;
    let (b_st, _) = cuda_storage_and_byte_offset(b, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;

    let beta = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape.clone())?;
    let beta_ptr = kiln_kt_bridge::cuda_output_device_ptr(&beta);
    let g = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape)?;
    let g_ptr = kiln_kt_bridge::cuda_output_device_ptr(&g);
    
    

    let raw_stream = a_st.cuda_stream_raw();

    let status = unsafe {        kiln_gdn_gates_bf16(
            a_ptr as *const _,
            b_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            beta_ptr as *mut _,
            g_ptr as *mut _,
            rows as i32,
            nv as i32,
            nv as i32,
            nv as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates_bf16 FFI returned {status}"
        )));
    }
    Ok((beta, g))
}

/// `kiln_gdn_gates_bf16_f32_params` over kt operands.
///
/// Variant where `A_log` and `dt_bias` are both F32; useful when those
/// parameters are kept in higher precision (e.g., trained as F32 master).
pub fn gdn_gates_bf16_f32_params_kt(
    a: &KtTensor,
    b: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
) -> Result<(KtTensor, KtTensor), GdnError> {
    let (rows, nv, out_shape) =
        gates_validate_inputs(a, b, a_log, dt_bias, KtDType::F32, KtDType::F32)?;

    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let b_ptr = kiln_kt_bridge::cuda_input_device_ptr(b, KtDType::BF16, "b")?;
    let (b_st, _) = cuda_storage_and_byte_offset(b, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::F32, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::F32, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::F32, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::F32, "dt_bias")?;

    let beta = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape.clone())?;
    let beta_ptr = kiln_kt_bridge::cuda_output_device_ptr(&beta);
    let g = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape)?;
    let g_ptr = kiln_kt_bridge::cuda_output_device_ptr(&g);
    
    

    let raw_stream = a_st.cuda_stream_raw();

    let status = unsafe {        kiln_gdn_gates_bf16_f32_params(
            a_ptr as *const _,
            b_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            beta_ptr as *mut _,
            g_ptr as *mut _,
            rows as i32,
            nv as i32,
            nv as i32,
            nv as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates_bf16_f32_params FFI returned {status}"
        )));
    }
    Ok((beta, g))
}

/// `kiln_gdn_gates_bf16_f32_bf16_params` over kt operands.
///
/// Mixed-precision variant: `A_log` is F32, `dt_bias` stays BF16.
pub fn gdn_gates_bf16_f32_bf16_params_kt(
    a: &KtTensor,
    b: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
) -> Result<(KtTensor, KtTensor), GdnError> {
    let (rows, nv, out_shape) =
        gates_validate_inputs(a, b, a_log, dt_bias, KtDType::F32, KtDType::BF16)?;

    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let b_ptr = kiln_kt_bridge::cuda_input_device_ptr(b, KtDType::BF16, "b")?;
    let (b_st, _) = cuda_storage_and_byte_offset(b, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::F32, "a_log")?;
    let (al_st, _) = cuda_storage_and_byte_offset(a_log, KtDType::F32, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (dt_st, _) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;

    let beta = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape.clone())?;
    let beta_ptr = kiln_kt_bridge::cuda_output_device_ptr(&beta);
    let g = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape)?;
    let g_ptr = kiln_kt_bridge::cuda_output_device_ptr(&g);
    
    

    let raw_stream = a_st.cuda_stream_raw();

    let status = unsafe {        kiln_gdn_gates_bf16_f32_bf16_params(
            a_ptr as *const _,
            b_ptr as *const _,
            al_ptr as *const _,
            dt_ptr as *const _,
            beta_ptr as *mut _,
            g_ptr as *mut _,
            rows as i32,
            nv as i32,
            nv as i32,
            nv as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: gates_bf16_f32_bf16_params FFI returned {status}"
        )));
    }
    Ok((beta, g))
}

/// `kiln_gdn_chunk_prep` over kt operands.
///
/// Prepares the auxiliary matrices that `kiln_gdn_chunk_scan` consumes.
///
/// Inputs (all BF16, all CUDA, all contiguous):
/// - `g`:        `[B, H, C]`
/// - `v`:        `[B, H, C, dv]`
/// - `kkt`,`qkt`: `[B, H, C, C]`
/// - `ks_entry`,`q_s`: `[B, H, C, dv]`
///
/// Returns a tuple of six BF16 outputs:
/// `(a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last)`
/// with shapes:
/// - `a_strict`, `b_mask`: `[B, H, C, C]`
/// - `v_prime`, `q_s_scaled`: `[B, H, C, dv]`
/// - `decay_last_col`: `[B, H, C]`
/// - `p_last`: `[B, H]`
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_prep_kt(
    g: &KtTensor,
    v: &KtTensor,
    kkt: &KtTensor,
    qkt: &KtTensor,
    ks_entry: &KtTensor,
    q_s: &KtTensor,
) -> Result<(KtTensor, KtTensor, KtTensor, KtTensor, KtTensor, KtTensor), GdnError> {
    let g_shape = g.shape();
    if g_shape.len() != 3 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-prep g must be [B, H, C], got {g_shape:?}"
        )));
    }
    let (b, h, c) = (g_shape[0], g_shape[1], g_shape[2]);

    let v_shape = v.shape();
    if v_shape.len() != 4 || (v_shape[0], v_shape[1], v_shape[2]) != (b, h, c) {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-prep v {v_shape:?} != [{b}, {h}, {c}, dv]"
        )));
    }
    let dv = v_shape[3];

    if kkt.shape() != [b, h, c, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-prep kkt {:?} != [{b}, {h}, {c}, {c}]",
            kkt.shape()
        )));
    }
    if qkt.shape() != [b, h, c, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-prep qkt {:?} != [{b}, {h}, {c}, {c}]",
            qkt.shape()
        )));
    }
    if ks_entry.shape() != [b, h, c, dv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-prep ks_entry {:?} != [{b}, {h}, {c}, {dv}]",
            ks_entry.shape()
        )));
    }
    if q_s.shape() != [b, h, c, dv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-prep q_s {:?} != [{b}, {h}, {c}, {dv}]",
            q_s.shape()
        )));
    }

    let g_ptr = kiln_kt_bridge::cuda_input_device_ptr(g, KtDType::BF16, "g")?;
    let (g_st, _) = cuda_storage_and_byte_offset(g, KtDType::BF16, "g")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let (v_st, _) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let kkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(kkt, KtDType::BF16, "kkt")?;
    let (kkt_st, _) = cuda_storage_and_byte_offset(kkt, KtDType::BF16, "kkt")?;
    let qkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(qkt, KtDType::BF16, "qkt")?;
    let (qkt_st, _) = cuda_storage_and_byte_offset(qkt, KtDType::BF16, "qkt")?;
    let ks_ptr = kiln_kt_bridge::cuda_input_device_ptr(ks_entry, KtDType::BF16, "ks_entry")?;
    let (ks_st, _) = cuda_storage_and_byte_offset(ks_entry, KtDType::BF16, "ks_entry")?;
    let qs_ptr = kiln_kt_bridge::cuda_input_device_ptr(q_s, KtDType::BF16, "q_s")?;
    let (qs_st, _) = cuda_storage_and_byte_offset(q_s, KtDType::BF16, "q_s")?;

    let a_strict = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, c])?;
    let a_ptr = kiln_kt_bridge::cuda_output_device_ptr(&a_strict);
    let b_mask = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, c])?;
    let bm_ptr = kiln_kt_bridge::cuda_output_device_ptr(&b_mask);
    let v_prime = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, dv])?;
    let vp_ptr = kiln_kt_bridge::cuda_output_device_ptr(&v_prime);
    let q_s_scaled = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, dv])?;
    let qss_ptr = kiln_kt_bridge::cuda_output_device_ptr(&q_s_scaled);
    let decay_last_col = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c])?;
    let dl_ptr = kiln_kt_bridge::cuda_output_device_ptr(&decay_last_col);
    let p_last = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h])?;
    let pl_ptr = kiln_kt_bridge::cuda_output_device_ptr(&p_last);

    
    
    
    
    
    

    let raw_stream = g_st.cuda_stream_raw();





    let status = unsafe {        kiln_gdn_chunk_prep(
            g_ptr as *const _,
            v_ptr as *const _,
            kkt_ptr as *const _,
            qkt_ptr as *const _,
            ks_ptr as *const _,
            qs_ptr as *const _,
            a_ptr as *mut _,
            bm_ptr as *mut _,
            vp_ptr as *mut _,
            qss_ptr as *mut _,
            dl_ptr as *mut _,
            pl_ptr as *mut _,
            (b * h) as i32,
            c as i32,
            dv as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk_prep FFI returned {status}"
        )));
    }
    Ok((a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last))
}

/// `kiln_gdn_chunk_scan` over kt operands.
///
/// The companion to [`gdn_chunk_prep_kt`]. Consumes the prepared
/// matrices plus `beta` and computes the chunk-scan output.
///
/// Inputs (all BF16):
/// - `a_strict`, `b_mask`: `[B, H, C, C]`
/// - `v_prime`, `q_s_scaled`: `[B, H, C, dv]`
/// - `beta`: `[B, H, C]`
/// - `decay_last_col`: `[B, H, C]`
///
/// Returns `(out_chunk, w_weighted)`, both BF16 `[B, H, C, dv]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_scan_kt(
    a_strict: &KtTensor,
    b_mask: &KtTensor,
    v_prime: &KtTensor,
    q_s_scaled: &KtTensor,
    beta: &KtTensor,
    decay_last_col: &KtTensor,
) -> Result<(KtTensor, KtTensor), GdnError> {
    let a_shape = a_strict.shape();
    if a_shape.len() != 4 || a_shape[2] != a_shape[3] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-scan a_strict must be [B, H, C, C], got {a_shape:?}"
        )));
    }
    let (b, h, c) = (a_shape[0], a_shape[1], a_shape[2]);
    if b_mask.shape() != [b, h, c, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-scan b_mask {:?} != [{b}, {h}, {c}, {c}]",
            b_mask.shape()
        )));
    }
    let vp_shape = v_prime.shape();
    if vp_shape.len() != 4 || (vp_shape[0], vp_shape[1], vp_shape[2]) != (b, h, c) {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-scan v_prime {vp_shape:?} != [{b}, {h}, {c}, dv]"
        )));
    }
    let dv = vp_shape[3];
    if q_s_scaled.shape() != [b, h, c, dv] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-scan q_s_scaled {:?} != [{b}, {h}, {c}, {dv}]",
            q_s_scaled.shape()
        )));
    }
    if beta.shape() != [b, h, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-scan beta {:?} != [{b}, {h}, {c}]",
            beta.shape()
        )));
    }
    if decay_last_col.shape() != [b, h, c] {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk-scan decay_last_col {:?} != [{b}, {h}, {c}]",
            decay_last_col.shape()
        )));
    }

    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_strict, KtDType::BF16, "a_strict")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a_strict, KtDType::BF16, "a_strict")?;
    let bm_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_mask, KtDType::BF16, "b_mask")?;
    let (bm_st, _) = cuda_storage_and_byte_offset(b_mask, KtDType::BF16, "b_mask")?;
    let vp_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_prime, KtDType::BF16, "v_prime")?;
    let (vp_st, _) = cuda_storage_and_byte_offset(v_prime, KtDType::BF16, "v_prime")?;
    let qss_ptr = kiln_kt_bridge::cuda_input_device_ptr(q_s_scaled, KtDType::BF16, "q_s_scaled")?;
    let (qss_st, _) = cuda_storage_and_byte_offset(q_s_scaled, KtDType::BF16, "q_s_scaled")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let (bt_st, _) = cuda_storage_and_byte_offset(beta, KtDType::BF16, "beta")?;
    let dl_ptr = kiln_kt_bridge::cuda_input_device_ptr(decay_last_col, KtDType::BF16, "decay_last_col")?;
    let (dl_st, _) = cuda_storage_and_byte_offset(decay_last_col, KtDType::BF16, "decay_last_col")?;

    let out_chunk = alloc_cuda_tensor(a_st, KtDType::BF16, vec![b, h, c, dv])?;
    let out_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out_chunk);
    let w_weighted = alloc_cuda_tensor(a_st, KtDType::BF16, vec![b, h, c, dv])?;
    let ww_ptr = kiln_kt_bridge::cuda_output_device_ptr(&w_weighted);
    
    

    let raw_stream = a_st.cuda_stream_raw();

    let status = unsafe {        kiln_gdn_chunk_scan(
            a_ptr as *const _,
            bm_ptr as *const _,
            vp_ptr as *const _,
            qss_ptr as *const _,
            bt_ptr as *const _,
            dl_ptr as *const _,
            out_ptr as *mut _,
            ww_ptr as *mut _,
            (b * h) as i32,
            c as i32,
            dv as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(GdnError::Msg(format!(
            "kt-gdn: chunk_scan FFI returned {status}"
        )));
    }
    Ok((out_chunk, w_weighted))
}

// ============================================================================
// kt-typed envelope predicates
// ============================================================================
//
// These mirror the candle-typed `gdn_*_supports` predicates in `lib.rs`
// one-for-one: same dtype + device + shape envelope checks. Inputs are
// kt-tensors instead of candle-tensors so callers can dispatch on a
// pre-borrowed kt-Tensor without round-tripping back through candle.
// All four are pure — no CUDA dispatch, no FFI — and therefore safe to
// call from any thread / under graph capture.

fn shape_dim_at(t: &KtTensor, axis: usize) -> Option<usize> {
    let s = t.shape();
    if axis < s.len() {
        Some(s[axis])
    } else {
        None
    }
}

fn is_cuda(t: &KtTensor) -> bool {
    matches!(t.device(), KtDevice::Cuda(_))
}

/// kt-typed mirror of [`crate::gdn_chunk_prep_supports`].
pub fn gdn_chunk_prep_supports_kt(
    g: &KtTensor,
    v: &KtTensor,
    kkt: &KtTensor,
    qkt: &KtTensor,
    ks_entry: &KtTensor,
    q_s: &KtTensor,
) -> bool {
    if !is_cuda(g) {
        return false;
    }
    if g.dtype() != KtDType::BF16
        || v.dtype() != KtDType::BF16
        || kkt.dtype() != KtDType::BF16
        || qkt.dtype() != KtDType::BF16
        || ks_entry.dtype() != KtDType::BF16
        || q_s.dtype() != KtDType::BF16
    {
        return false;
    }
    let g_dims = g.shape();
    let c = match g_dims.last() {
        Some(n) => *n,
        None => return false,
    };
    if c == 0 || c > 128 {
        return false;
    }
    let vd = v.shape();
    if vd.len() < 2 {
        return false;
    }
    let dv = vd[vd.len() - 1];
    if dv == 0 || dv > 1024 {
        return false;
    }
    if vd[vd.len() - 2] != c {
        return false;
    }
    let kdims = kkt.shape();
    let qdims = qkt.shape();
    if kdims.len() < 2 || qdims.len() < 2 {
        return false;
    }
    if kdims[kdims.len() - 1] != c
        || kdims[kdims.len() - 2] != c
        || qdims[qdims.len() - 1] != c
        || qdims[qdims.len() - 2] != c
    {
        return false;
    }
    if ks_entry.shape().last().copied() != Some(dv) || q_s.shape().last().copied() != Some(dv) {
        return false;
    }
    true
}

/// kt-typed mirror of [`crate::gdn_chunk_scan_supports`].
pub fn gdn_chunk_scan_supports_kt(
    a_strict: &KtTensor,
    b_mask: &KtTensor,
    v_prime: &KtTensor,
    q_s_scaled: &KtTensor,
    beta: &KtTensor,
    decay_last_col: &KtTensor,
) -> bool {
    if !is_cuda(a_strict) {
        return false;
    }
    if a_strict.dtype() != KtDType::BF16
        || b_mask.dtype() != KtDType::BF16
        || v_prime.dtype() != KtDType::BF16
        || q_s_scaled.dtype() != KtDType::BF16
        || beta.dtype() != KtDType::BF16
        || decay_last_col.dtype() != KtDType::BF16
    {
        return false;
    }
    let a = a_strict.shape();
    if a.len() != 4 {
        return false;
    }
    let (b, h, c, c2) = (a[0], a[1], a[2], a[3]);
    if c != c2 || c == 0 || c > 64 {
        return false;
    }
    let bm = b_mask.shape();
    if bm.len() != 4 || (bm[0], bm[1], bm[2], bm[3]) != (b, h, c, c) {
        return false;
    }
    let vp = v_prime.shape();
    if vp.len() != 4 {
        return false;
    }
    let dv = vp[3];
    if (vp[0], vp[1], vp[2]) != (b, h, c) || dv == 0 || dv > 128 {
        return false;
    }
    let qs = q_s_scaled.shape();
    if qs.len() != 4 || (qs[0], qs[1], qs[2], qs[3]) != (b, h, c, dv) {
        return false;
    }
    let be = beta.shape();
    if be.len() != 3 || (be[0], be[1], be[2]) != (b, h, c) {
        return false;
    }
    let dl = decay_last_col.shape();
    dl.len() == 3 && (dl[0], dl[1], dl[2]) == (b, h, c)
}

/// kt-typed mirror of [`crate::gdn_full_chunk_forward_supports`].
#[allow(clippy::too_many_arguments)]
pub fn gdn_full_chunk_forward_supports_kt(
    g: &KtTensor,
    v: &KtTensor,
    kkt: &KtTensor,
    qkt: &KtTensor,
    ks_entry: &KtTensor,
    q_s: &KtTensor,
    beta: &KtTensor,
    k_t: &KtTensor,
    state: &KtTensor,
) -> bool {
    if !is_cuda(g) {
        return false;
    }
    if g.dtype() != KtDType::BF16
        || v.dtype() != KtDType::BF16
        || kkt.dtype() != KtDType::BF16
        || qkt.dtype() != KtDType::BF16
        || ks_entry.dtype() != KtDType::BF16
        || q_s.dtype() != KtDType::BF16
        || beta.dtype() != KtDType::BF16
        || k_t.dtype() != KtDType::BF16
        || state.dtype() != KtDType::BF16
    {
        return false;
    }
    let gs = g.shape();
    if gs.len() != 3 {
        return false;
    }
    let (b, h, c) = (gs[0], gs[1], gs[2]);
    if c != 64 {
        return false;
    }
    let vs = v.shape();
    if vs.len() != 4 {
        return false;
    }
    let dv = vs[3];
    if (vs[0], vs[1], vs[2]) != (b, h, c) || dv == 0 || dv > 128 {
        return false;
    }
    let ks = kkt.shape();
    if ks.len() != 4 || (ks[0], ks[1], ks[2], ks[3]) != (b, h, c, c) {
        return false;
    }
    let qs = qkt.shape();
    if qs.len() != 4 || (qs[0], qs[1], qs[2], qs[3]) != (b, h, c, c) {
        return false;
    }
    let kes = ks_entry.shape();
    if kes.len() != 4 || (kes[0], kes[1], kes[2], kes[3]) != (b, h, c, dv) {
        return false;
    }
    let qse = q_s.shape();
    if qse.len() != 4 || (qse[0], qse[1], qse[2], qse[3]) != (b, h, c, dv) {
        return false;
    }
    let be = beta.shape();
    if be.len() != 3 || (be[0], be[1], be[2]) != (b, h, c) {
        return false;
    }
    let kt = k_t.shape();
    if kt.len() != 4 {
        return false;
    }
    let dk = kt[2];
    if (kt[0], kt[1], kt[3]) != (b, h, c) || dk == 0 || dk > 128 {
        return false;
    }
    let st = state.shape();
    st.len() == 4 && (st[0], st[1], st[2], st[3]) == (b, h, dk, dv)
}

/// kt-typed mirror of [`crate::gdn_full_chunk_forward_multiblock_supports`].
#[allow(clippy::too_many_arguments)]
pub fn gdn_full_chunk_forward_multiblock_supports_kt(
    g: &KtTensor,
    v: &KtTensor,
    kkt: &KtTensor,
    qkt: &KtTensor,
    ks_entry: &KtTensor,
    q_s: &KtTensor,
    beta: &KtTensor,
    k_t: &KtTensor,
    state: &KtTensor,
    dv_tile: usize,
) -> bool {
    if dv_tile != crate::GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_DV_TILE {
        return false;
    }
    if !gdn_full_chunk_forward_supports_kt(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state) {
        return false;
    }
    let dv = match shape_dim_at(v, 3) {
        Some(d) => d,
        None => return false,
    };
    dv >= dv_tile && dv % dv_tile == 0
}

/// kt-typed mirror of [`crate::gdn_decode_gates_recurrent_supports`].
///
/// Same dtype + device + shape envelope as the candle predicate
/// (`q/k/a/b/a_log/dt_bias/state` BF16; `v` BF16 or F32; `weight` F32;
/// 4D `[B, 1, q_heads, dk=128]` for q/k; `[B, 1, value_heads, dv=128]`
/// for v/z; 3D `[B, 1, value_heads]` for a/b; `[value_heads]` for
/// a_log/dt_bias; `[dv]` for weight; 4D `[B, value_heads, dk, dv]` for
/// state; state must be contiguous; `value_heads >= q_heads` and
/// `value_heads % q_heads == 0`).
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_gates_recurrent_supports_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    z: &KtTensor,
    weight: &KtTensor,
) -> bool {
    if !is_cuda(q) {
        return false;
    }
    if q.dtype() != KtDType::BF16
        || k.dtype() != KtDType::BF16
        || !matches!(v.dtype(), KtDType::BF16 | KtDType::F32)
        || a.dtype() != KtDType::BF16
        || b.dtype() != KtDType::BF16
        || a_log.dtype() != KtDType::BF16
        || dt_bias.dtype() != KtDType::BF16
        || state.dtype() != KtDType::BF16
    {
        return false;
    }
    let qs = q.shape();
    if qs.len() != 4 {
        return false;
    }
    let (batch, seq_len, q_heads, dk) = (qs[0], qs[1], qs[2], qs[3]);
    let ks = k.shape();
    if ks.len() != 4 || (ks[0], ks[1], ks[2], ks[3]) != (batch, seq_len, q_heads, dk) {
        return false;
    }
    let vs = v.shape();
    if vs.len() != 4 || (vs[0], vs[1]) != (batch, seq_len) {
        return false;
    }
    let (value_heads, dv) = (vs[2], vs[3]);
    let as_ = a.shape();
    if as_.len() != 3 || (as_[0], as_[1], as_[2]) != (batch, seq_len, value_heads) {
        return false;
    }
    let bs = b.shape();
    if bs.len() != 3 || (bs[0], bs[1], bs[2]) != (batch, seq_len, value_heads) {
        return false;
    }
    if z.shape() != [batch, seq_len, value_heads, dv] {
        return false;
    }
    if a_log.shape() != [value_heads] || dt_bias.shape() != [value_heads] {
        return false;
    }
    if weight.shape() != [dv] {
        return false;
    }
    let ss = state.shape();
    if ss.len() != 4 || (ss[0], ss[1], ss[2], ss[3]) != (batch, value_heads, dk, dv) {
        return false;
    }
    batch >= 1
        && seq_len == 1
        && q_heads > 0
        && value_heads >= q_heads
        && value_heads % q_heads == 0
        && dk == 128
        && dv == 128
        && state.is_contiguous()
}

/// kt-typed mirror of [`crate::gdn_decode_qk_norm_gates_recurrent_supports`].
///
/// Same dtype + device + shape envelope as the candle predicate. Unlike
/// [`gdn_decode_gates_recurrent_supports_kt`], q and k may be either
/// BF16 or F32 (matching the four production variants: bf16/bf16,
/// qf32/vbf16, qf32/vf32, vf32/bf16).
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_supports_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
) -> bool {
    if !is_cuda(q) {
        return false;
    }
    if !matches!(q.dtype(), KtDType::BF16 | KtDType::F32)
        || k.dtype() != q.dtype()
        || !matches!(v.dtype(), KtDType::BF16 | KtDType::F32)
        || a.dtype() != KtDType::BF16
        || b.dtype() != KtDType::BF16
        || a_log.dtype() != KtDType::BF16
        || dt_bias.dtype() != KtDType::BF16
        || state.dtype() != KtDType::BF16
    {
        return false;
    }
    let qs = q.shape();
    if qs.len() != 4 {
        return false;
    }
    let (batch, seq_len, q_heads, dk) = (qs[0], qs[1], qs[2], qs[3]);
    let ks = k.shape();
    if ks.len() != 4 || (ks[0], ks[1], ks[2], ks[3]) != (batch, seq_len, q_heads, dk) {
        return false;
    }
    let vs = v.shape();
    if vs.len() != 4 || (vs[0], vs[1]) != (batch, seq_len) {
        return false;
    }
    let (value_heads, dv) = (vs[2], vs[3]);
    let as_ = a.shape();
    if as_.len() != 3 || (as_[0], as_[1], as_[2]) != (batch, seq_len, value_heads) {
        return false;
    }
    let bs = b.shape();
    if bs.len() != 3 || (bs[0], bs[1], bs[2]) != (batch, seq_len, value_heads) {
        return false;
    }
    if a_log.shape() != [value_heads] || dt_bias.shape() != [value_heads] {
        return false;
    }
    let ss = state.shape();
    if ss.len() != 4 || (ss[0], ss[1], ss[2], ss[3]) != (batch, value_heads, dk, dv) {
        return false;
    }
    batch >= 1
        && seq_len == 1
        && q_heads > 0
        && value_heads >= q_heads
        && value_heads % q_heads == 0
        && dk == 128
        && dv == 128
        && state.is_contiguous()
}

/// kt-typed mirror of
/// [`crate::gdn_decode_qk_norm_gates_recurrent_rmsnorm_supports`].
///
/// Composes [`gdn_decode_qk_norm_gates_recurrent_supports_kt`] with the
/// extra `z` (BF16, shape `[B, 1, value_heads, dv]`) and `weight` (F32,
/// shape `[dv]`) checks needed by the rmsnorm-fused variant.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_qk_norm_gates_recurrent_rmsnorm_supports_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    a: &KtTensor,
    b: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
    state: &KtTensor,
    z: &KtTensor,
    weight: &KtTensor,
) -> bool {
    if !gdn_decode_qk_norm_gates_recurrent_supports_kt(q, k, v, a, b, a_log, dt_bias, state) {
        return false;
    }
    if z.dtype() != KtDType::BF16 || weight.dtype() != KtDType::F32 {
        return false;
    }
    let qs = q.shape();
    let (batch, seq_len) = (qs[0], qs[1]);
    let vs = v.shape();
    let (value_heads, dv) = (vs[2], vs[3]);
    z.shape() == [batch, seq_len, value_heads, dv] && weight.shape() == [dv]
}

/// kt-typed mirror of [`crate::gdn_gates_supports`].
///
/// Same dtype + device + shape envelope as
/// [`crate::gdn_gates_decline_reason`]:
///   - all on CUDA
///   - `a, b` BF16
///   - `(a_log, dt_bias)` ∈ {(BF16, BF16), (F32, F32), (F32, BF16)}
///   - `a.shape == b.shape`, last dim `nv ∈ 1..=256`
///   - `a_log.shape == dt_bias.shape == [nv]`
pub fn gdn_gates_supports_kt(
    a: &KtTensor,
    b: &KtTensor,
    a_log: &KtTensor,
    dt_bias: &KtTensor,
) -> bool {
    if !is_cuda(a) || !is_cuda(b) || !is_cuda(a_log) || !is_cuda(dt_bias) {
        return false;
    }
    if a.dtype() != KtDType::BF16 || b.dtype() != KtDType::BF16 {
        return false;
    }
    if !matches!(
        (a_log.dtype(), dt_bias.dtype()),
        (KtDType::BF16, KtDType::BF16)
            | (KtDType::F32, KtDType::F32)
            | (KtDType::F32, KtDType::BF16)
    ) {
        return false;
    }
    if a.shape() != b.shape() {
        return false;
    }
    let last = match a.shape().last() {
        Some(n) => *n,
        None => return false,
    };
    if last == 0 || last > 256 {
        return false;
    }
    if a_log.shape() != [last] || dt_bias.shape() != [last] {
        return false;
    }
    true
}

/// kt-typed mirror of [`crate::gdn_gated_rms_norm_supports`].
///
/// Same dtype + device + shape envelope as the candle predicate:
///   - CUDA + BF16 for x, z, weight
///   - `x.shape == z.shape`
///   - last dim `hidden == 128`
///   - `weight.shape == [hidden]`
pub fn gdn_gated_rms_norm_supports_kt(x: &KtTensor, z: &KtTensor, weight: &KtTensor) -> bool {
    if !is_cuda(x) {
        return false;
    }
    if x.dtype() != KtDType::BF16 || z.dtype() != KtDType::BF16 || weight.dtype() != KtDType::BF16
    {
        return false;
    }
    if x.shape() != z.shape() {
        return false;
    }
    let hidden = match x.shape().last() {
        Some(n) => *n,
        None => return false,
    };
    hidden == 128 && weight.shape() == [hidden]
}

#[cfg(test)]
mod predicate_tests {
    use super::*;

    // These tests confirm the predicates compile + return `false` for the
    // trivial CPU-tensor case (no GPU required). Byte-exact parity vs the
    // candle predicates is implicit: both functions implement the same
    // pure shape/dtype/device check chain on the same inputs.

    #[test]
    fn predicates_decline_on_cpu() {
        // Build CPU kt-tensors and confirm `_supports_kt` declines because
        // of the CUDA-device check (all predicates start with `is_cuda`).
        let g = KtTensor::zeros_cpu(vec![1, 32, 64], KtDType::BF16);
        let v = KtTensor::zeros_cpu(vec![1, 32, 64, 128], KtDType::BF16);
        let kkt = KtTensor::zeros_cpu(vec![1, 32, 64, 64], KtDType::BF16);
        let qkt = KtTensor::zeros_cpu(vec![1, 32, 64, 64], KtDType::BF16);
        let ks = KtTensor::zeros_cpu(vec![1, 32, 64, 128], KtDType::BF16);
        let qs = KtTensor::zeros_cpu(vec![1, 32, 64, 128], KtDType::BF16);
        let beta = KtTensor::zeros_cpu(vec![1, 32, 64], KtDType::BF16);
        let kt = KtTensor::zeros_cpu(vec![1, 32, 128, 64], KtDType::BF16);
        let st = KtTensor::zeros_cpu(vec![1, 32, 128, 128], KtDType::BF16);

        assert!(!gdn_chunk_prep_supports_kt(&g, &v, &kkt, &qkt, &ks, &qs));
        assert!(!gdn_chunk_scan_supports_kt(&kkt, &qkt, &v, &qs, &beta, &g));
        assert!(!gdn_full_chunk_forward_supports_kt(
            &g, &v, &kkt, &qkt, &ks, &qs, &beta, &kt, &st
        ));
        assert!(!gdn_full_chunk_forward_multiblock_supports_kt(
            &g,
            &v,
            &kkt,
            &qkt,
            &ks,
            &qs,
            &beta,
            &kt,
            &st,
            crate::GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_DV_TILE,
        ));
    }

    #[test]
    fn decode_predicates_decline_on_cpu() {
        // CPU tensors should always decline (CUDA-only kernels). This
        // exercises the new decode + gates + gated_rms_norm predicates
        // in the host-only compile path (#1082 phase 7 follow-up).
        let batch = 1usize;
        let seq_len = 1usize;
        let q_heads = 16usize;
        let value_heads = 16usize;
        let dk = 128usize;
        let dv = 128usize;
        let q = KtTensor::zeros_cpu(vec![batch, seq_len, q_heads, dk], KtDType::BF16);
        let k = KtTensor::zeros_cpu(vec![batch, seq_len, q_heads, dk], KtDType::BF16);
        let v = KtTensor::zeros_cpu(vec![batch, seq_len, value_heads, dv], KtDType::BF16);
        let a = KtTensor::zeros_cpu(vec![batch, seq_len, value_heads], KtDType::BF16);
        let b = KtTensor::zeros_cpu(vec![batch, seq_len, value_heads], KtDType::BF16);
        let a_log = KtTensor::zeros_cpu(vec![value_heads], KtDType::BF16);
        let dt_bias = KtTensor::zeros_cpu(vec![value_heads], KtDType::BF16);
        let state = KtTensor::zeros_cpu(vec![batch, value_heads, dk, dv], KtDType::BF16);
        let z = KtTensor::zeros_cpu(vec![batch, seq_len, value_heads, dv], KtDType::BF16);
        let weight_f32 = KtTensor::zeros_cpu(vec![dv], KtDType::F32);
        let weight_bf16 = KtTensor::zeros_cpu(vec![dv], KtDType::BF16);

        assert!(!gdn_decode_gates_recurrent_supports_kt(
            &q,
            &k,
            &v,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &state,
            &z,
            &weight_f32,
        ));
        assert!(!gdn_decode_qk_norm_gates_recurrent_supports_kt(
            &q, &k, &v, &a, &b, &a_log, &dt_bias, &state,
        ));
        assert!(!gdn_decode_qk_norm_gates_recurrent_rmsnorm_supports_kt(
            &q,
            &k,
            &v,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &state,
            &z,
            &weight_f32,
        ));

        // gdn_gates: a,b matching [.., nv], a_log/dt_bias [nv].
        let nv = 32usize;
        let g_a = KtTensor::zeros_cpu(vec![1, 1, nv], KtDType::BF16);
        let g_b = KtTensor::zeros_cpu(vec![1, 1, nv], KtDType::BF16);
        let g_al = KtTensor::zeros_cpu(vec![nv], KtDType::BF16);
        let g_dt = KtTensor::zeros_cpu(vec![nv], KtDType::BF16);
        assert!(!gdn_gates_supports_kt(&g_a, &g_b, &g_al, &g_dt));

        // gdn_gated_rms_norm: x/z BF16 [.., 128], weight BF16 [128].
        let r_x = KtTensor::zeros_cpu(vec![1, 1, value_heads, dv], KtDType::BF16);
        let r_z = KtTensor::zeros_cpu(vec![1, 1, value_heads, dv], KtDType::BF16);
        assert!(!gdn_gated_rms_norm_supports_kt(&r_x, &r_z, &weight_bf16));
    }
}

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

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use kiln_kt_bridge::BridgeError;
use kiln_tensor::{CudaStorage, DType as KtDType, Tensor as KtTensor};

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

    // Owner-agnostic input pointers (Phase 7 v2).
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_strict, KtDType::BF16, "a_strict")?;
    let vp_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_prime, KtDType::BF16, "v_prime")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a_strict, KtDType::BF16, "a_strict")?;

    let w_out = alloc_cuda_tensor(a_st, KtDType::BF16, vec![b, h, c, dv])?;
    let w_ptr = kiln_kt_bridge::cuda_output_device_ptr(&w_out);

    let stream = a_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_gdn_forward_substitution(
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
    }    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let g_ptr = kiln_kt_bridge::cuda_input_device_ptr(g, KtDType::BF16, "g")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, h, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;






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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;










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
    }    // Owner-agnostic input pointers (Phase 7 v2).
    let g_ptr = kiln_kt_bridge::cuda_input_device_ptr(g, KtDType::BF16, "g")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let kkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(kkt, KtDType::BF16, "kkt")?;
    let qkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(qkt, KtDType::BF16, "qkt")?;
    let ks_ptr = kiln_kt_bridge::cuda_input_device_ptr(ks_entry, KtDType::BF16, "ks_entry")?;
    let qs_ptr = kiln_kt_bridge::cuda_input_device_ptr(q_s, KtDType::BF16, "q_s")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let kt_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_t, KtDType::BF16, "k_t")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (g_st, _) = cuda_storage_and_byte_offset(g, KtDType::BF16, "g")?;
    let out = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, dv])?;    let stream = g_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;









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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;










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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;








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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;










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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;








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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::F32, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;








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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::F32, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;








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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::BF16, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;










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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::F32, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::F32, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;










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
    let dv = v_shape[2];    // Owner-agnostic input pointers (Phase 7 v2).
    let q_ptr = kiln_kt_bridge::cuda_input_device_ptr(q, KtDType::F32, "q")?;
    let k_ptr = kiln_kt_bridge::cuda_input_device_ptr(k, KtDType::BF16, "k")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let bp_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_param, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (q_st, _) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;










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
    }    // Owner-agnostic input pointers (Phase 7 v2).
    let g_ptr = kiln_kt_bridge::cuda_input_device_ptr(g, KtDType::BF16, "g")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let kkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(kkt, KtDType::BF16, "kkt")?;
    let qkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(qkt, KtDType::BF16, "qkt")?;
    let ks_ptr = kiln_kt_bridge::cuda_input_device_ptr(ks_entry, KtDType::BF16, "ks_entry")?;
    let qs_ptr = kiln_kt_bridge::cuda_input_device_ptr(q_s, KtDType::BF16, "q_s")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let kt_ptr = kiln_kt_bridge::cuda_input_device_ptr(k_t, KtDType::BF16, "k_t")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(state, KtDType::BF16, "state")?;
    let (g_st, _) = cuda_storage_and_byte_offset(g, KtDType::BF16, "g")?;
    let out = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, dv])?;    let stream = g_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;









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
/// - `weight`: F32  `[hidden]` — learned per-channel scale
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
    }    // Owner-agnostic input pointers (Phase 7 v2).
    let x_ptr = kiln_kt_bridge::cuda_input_device_ptr(x, KtDType::BF16, "x")?;
    let z_ptr = kiln_kt_bridge::cuda_input_device_ptr(z, KtDType::BF16, "z")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(weight, KtDType::F32, "weight")?;
    let (x_st, _) = cuda_storage_and_byte_offset(x, KtDType::BF16, "x")?;
    let out = alloc_cuda_tensor(x_st, KtDType::BF16, vec![rows, hidden])?;    let stream = x_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;



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
        gates_validate_inputs(a, b, a_log, dt_bias, KtDType::BF16, KtDType::BF16)?;    // Owner-agnostic input pointers (Phase 7 v2).
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let b_ptr = kiln_kt_bridge::cuda_input_device_ptr(b, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::BF16, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let beta = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape.clone())?;
    let g = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape)?;    let stream = a_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;





    let status = unsafe {        let beta_ptr = kiln_kt_bridge::cuda_output_device_ptr(&beta);
        let g_ptr = kiln_kt_bridge::cuda_output_device_ptr(&g);
        kiln_gdn_gates_bf16(
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
        gates_validate_inputs(a, b, a_log, dt_bias, KtDType::F32, KtDType::F32)?;    // Owner-agnostic input pointers (Phase 7 v2).
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let b_ptr = kiln_kt_bridge::cuda_input_device_ptr(b, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::F32, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::F32, "dt_bias")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let beta = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape.clone())?;
    let g = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape)?;    let stream = a_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;





    let status = unsafe {        let beta_ptr = kiln_kt_bridge::cuda_output_device_ptr(&beta);
        let g_ptr = kiln_kt_bridge::cuda_output_device_ptr(&g);
        kiln_gdn_gates_bf16_f32_params(
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
        gates_validate_inputs(a, b, a_log, dt_bias, KtDType::F32, KtDType::BF16)?;    // Owner-agnostic input pointers (Phase 7 v2).
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a, KtDType::BF16, "a")?;
    let b_ptr = kiln_kt_bridge::cuda_input_device_ptr(b, KtDType::BF16, "b")?;
    let al_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_log, KtDType::F32, "a_log")?;
    let dt_ptr = kiln_kt_bridge::cuda_input_device_ptr(dt_bias, KtDType::BF16, "dt_bias")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let beta = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape.clone())?;
    let g = alloc_cuda_tensor(a_st, KtDType::BF16, out_shape)?;    let stream = a_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;





    let status = unsafe {        let beta_ptr = kiln_kt_bridge::cuda_output_device_ptr(&beta);
        let g_ptr = kiln_kt_bridge::cuda_output_device_ptr(&g);
        kiln_gdn_gates_bf16_f32_bf16_params(
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
    }    // Owner-agnostic input pointers (Phase 7 v2).
    let g_ptr = kiln_kt_bridge::cuda_input_device_ptr(g, KtDType::BF16, "g")?;
    let v_ptr = kiln_kt_bridge::cuda_input_device_ptr(v, KtDType::BF16, "v")?;
    let kkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(kkt, KtDType::BF16, "kkt")?;
    let qkt_ptr = kiln_kt_bridge::cuda_input_device_ptr(qkt, KtDType::BF16, "qkt")?;
    let ks_ptr = kiln_kt_bridge::cuda_input_device_ptr(ks_entry, KtDType::BF16, "ks_entry")?;
    let qs_ptr = kiln_kt_bridge::cuda_input_device_ptr(q_s, KtDType::BF16, "q_s")?;
    let (g_st, _) = cuda_storage_and_byte_offset(g, KtDType::BF16, "g")?;
    let a_strict = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, c])?;
    let b_mask = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, c])?;
    let v_prime = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, dv])?;
    let q_s_scaled = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, dv])?;
    let decay_last_col = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c])?;
    let p_last = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h])?;    let stream = g_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;











    let status = unsafe {        let a_ptr = kiln_kt_bridge::cuda_output_device_ptr(&a_strict);
        let bm_ptr = kiln_kt_bridge::cuda_output_device_ptr(&b_mask);
        let vp_ptr = kiln_kt_bridge::cuda_output_device_ptr(&v_prime);
        let qss_ptr = kiln_kt_bridge::cuda_output_device_ptr(&q_s_scaled);
        let dl_ptr = kiln_kt_bridge::cuda_output_device_ptr(&decay_last_col);
        let pl_ptr = kiln_kt_bridge::cuda_output_device_ptr(&p_last);
        kiln_gdn_chunk_prep(
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
    }    // Owner-agnostic input pointers (Phase 7 v2).
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_strict, KtDType::BF16, "a_strict")?;
    let bm_ptr = kiln_kt_bridge::cuda_input_device_ptr(b_mask, KtDType::BF16, "b_mask")?;
    let vp_ptr = kiln_kt_bridge::cuda_input_device_ptr(v_prime, KtDType::BF16, "v_prime")?;
    let qss_ptr = kiln_kt_bridge::cuda_input_device_ptr(q_s_scaled, KtDType::BF16, "q_s_scaled")?;
    let bt_ptr = kiln_kt_bridge::cuda_input_device_ptr(beta, KtDType::BF16, "beta")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a_strict, KtDType::BF16, "a_strict")?;
    let (dl_st, dl_off) =
        cuda_storage_and_byte_offset(decay_last_col, KtDType::BF16, "decay_last_col")?;

    let out_chunk = alloc_cuda_tensor(a_st, KtDType::BF16, vec![b, h, c, dv])?;
    let w_weighted = alloc_cuda_tensor(a_st, KtDType::BF16, vec![b, h, c, dv])?;    let stream = a_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;







    let status = unsafe {        let out_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out_chunk);
        let ww_ptr = kiln_kt_bridge::cuda_output_device_ptr(&w_weighted);
        kiln_gdn_chunk_scan(
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

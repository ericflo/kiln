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
    kiln_gdn_decode_qk_norm_gates_recurrent_vf32_bf16, kiln_gdn_forward_substitution,
    kiln_gdn_full_chunk_forward, kiln_gdn_recurrent_forward,
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

    let (a_st, a_off) = cuda_storage_and_byte_offset(a_strict, KtDType::BF16, "a_strict")?;
    let (vp_st, vp_off) = cuda_storage_and_byte_offset(v_prime, KtDType::BF16, "v_prime")?;
    let (bt_st, bt_off) = cuda_storage_and_byte_offset(beta, KtDType::BF16, "beta")?;

    let w_out = alloc_cuda_tensor(a_st, KtDType::BF16, vec![b, h, c, dv])?;
    let w_cuda = w_out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = a_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let a_slice = a_st.slice().slice(a_off..);
    let vp_slice = vp_st.slice().slice(vp_off..);
    let bt_slice = bt_st.slice().slice(bt_off..);
    let w_slice = w_cuda.slice().slice(0..);

    let status = unsafe {
        let (a_ptr, _g1) = a_slice.device_ptr(&stream);
        let (vp_ptr, _g2) = vp_slice.device_ptr(&stream);
        let (bt_ptr, _g3) = bt_slice.device_ptr(&stream);
        let (w_ptr, _g4) = w_slice.device_ptr(&stream);

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
    }

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (bt_st, bt_off) = cuda_storage_and_byte_offset(beta, KtDType::BF16, "beta")?;
    let (g_st, g_off) = cuda_storage_and_byte_offset(g, KtDType::BF16, "g")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![b, h, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let bt_slice = bt_st.slice().slice(bt_off..);
    let g_slice = g_st.slice().slice(g_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (bt_ptr, _g4) = bt_slice.device_ptr(&stream);
        let (g_ptr, _g5) = g_slice.device_ptr(&stream);
        let (s_ptr, _g6) = s_slice.device_ptr(&stream);
        let (o_ptr, _g7) = o_slice.device_ptr(&stream);

        kiln_gdn_recurrent_forward(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let (z_st, z_off) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let z_slice = z_st.slice().slice(z_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (z_ptr, _g9) = z_slice.device_ptr(&stream);
        let (w_ptr, _g10) = w_slice.device_ptr(&stream);
        let (o_ptr, _g11) = o_slice.device_ptr(&stream);

        kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16(
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

    let (g_st, g_off) = cuda_storage_and_byte_offset(g, KtDType::BF16, "g")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (kkt_st, kkt_off) = cuda_storage_and_byte_offset(kkt, KtDType::BF16, "kkt")?;
    let (qkt_st, qkt_off) = cuda_storage_and_byte_offset(qkt, KtDType::BF16, "qkt")?;
    let (ks_st, ks_off) = cuda_storage_and_byte_offset(ks_entry, KtDType::BF16, "ks_entry")?;
    let (qs_st, qs_off) = cuda_storage_and_byte_offset(q_s, KtDType::BF16, "q_s")?;
    let (bt_st, bt_off) = cuda_storage_and_byte_offset(beta, KtDType::BF16, "beta")?;
    let (kt_st, kt_off) = cuda_storage_and_byte_offset(k_t, KtDType::BF16, "k_t")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(g_st, KtDType::BF16, vec![b, h, c, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = g_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let g_slice = g_st.slice().slice(g_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let kkt_slice = kkt_st.slice().slice(kkt_off..);
    let qkt_slice = qkt_st.slice().slice(qkt_off..);
    let ks_slice = ks_st.slice().slice(ks_off..);
    let qs_slice = qs_st.slice().slice(qs_off..);
    let bt_slice = bt_st.slice().slice(bt_off..);
    let kt_slice = kt_st.slice().slice(kt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (g_ptr, _g1) = g_slice.device_ptr(&stream);
        let (v_ptr, _g2) = v_slice.device_ptr(&stream);
        let (kkt_ptr, _g3) = kkt_slice.device_ptr(&stream);
        let (qkt_ptr, _g4) = qkt_slice.device_ptr(&stream);
        let (ks_ptr, _g5) = ks_slice.device_ptr(&stream);
        let (qs_ptr, _g6) = qs_slice.device_ptr(&stream);
        let (bt_ptr, _g7) = bt_slice.device_ptr(&stream);
        let (kt_ptr, _g8) = kt_slice.device_ptr(&stream);
        let (s_ptr, _g9) = s_slice.device_ptr(&stream);
        let (o_ptr, _g10) = o_slice.device_ptr(&stream);

        kiln_gdn_full_chunk_forward(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let (z_st, z_off) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let z_slice = z_st.slice().slice(z_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (z_ptr, _g9) = z_slice.device_ptr(&stream);
        let (w_ptr, _g10) = w_slice.device_ptr(&stream);
        let (o_ptr, _g11) = o_slice.device_ptr(&stream);
        kiln_gdn_decode_gates_recurrent_bf16(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (o_ptr, _g9) = o_slice.device_ptr(&stream);
        kiln_gdn_decode_qk_norm_gates_recurrent_bf16(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let (z_st, z_off) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let z_slice = z_st.slice().slice(z_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (z_ptr, _g9) = z_slice.device_ptr(&stream);
        let (w_ptr, _g10) = w_slice.device_ptr(&stream);
        let (o_ptr, _g11) = o_slice.device_ptr(&stream);
        kiln_gdn_decode_gates_recurrent_vf32_bf16(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (o_ptr, _g9) = o_slice.device_ptr(&stream);
        kiln_gdn_decode_qk_norm_gates_recurrent_vf32_bf16(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (o_ptr, _g9) = o_slice.device_ptr(&stream);
        kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vf32_bf16(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (o_ptr, _g9) = o_slice.device_ptr(&stream);
        kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vbf16_bf16(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let (z_st, z_off) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let z_slice = z_st.slice().slice(z_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (z_ptr, _g9) = z_slice.device_ptr(&stream);
        let (w_ptr, _g10) = w_slice.device_ptr(&stream);
        let (o_ptr, _g11) = o_slice.device_ptr(&stream);
        kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_vf32_bf16(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::F32, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let (z_st, z_off) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let z_slice = z_st.slice().slice(z_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (z_ptr, _g9) = z_slice.device_ptr(&stream);
        let (w_ptr, _g10) = w_slice.device_ptr(&stream);
        let (o_ptr, _g11) = o_slice.device_ptr(&stream);
        kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vf32_bf16(
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

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::F32, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (v_st, v_off) = cuda_storage_and_byte_offset(v, KtDType::BF16, "v")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let (bp_st, bp_off) = cuda_storage_and_byte_offset(b_param, KtDType::BF16, "b")?;
    let (al_st, al_off) = cuda_storage_and_byte_offset(a_log, KtDType::BF16, "a_log")?;
    let (dt_st, dt_off) = cuda_storage_and_byte_offset(dt_bias, KtDType::BF16, "dt_bias")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::BF16, "state")?;
    let (z_st, z_off) = cuda_storage_and_byte_offset(z, KtDType::BF16, "z")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;

    let out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, value_heads, dv])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let v_slice = v_st.slice().slice(v_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let bp_slice = bp_st.slice().slice(bp_off..);
    let al_slice = al_st.slice().slice(al_off..);
    let dt_slice = dt_st.slice().slice(dt_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let z_slice = z_st.slice().slice(z_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (a_ptr, _g4) = a_slice.device_ptr(&stream);
        let (bp_ptr, _g5) = bp_slice.device_ptr(&stream);
        let (al_ptr, _g6) = al_slice.device_ptr(&stream);
        let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
        let (s_ptr, _g8) = s_slice.device_ptr(&stream);
        let (z_ptr, _g9) = z_slice.device_ptr(&stream);
        let (w_ptr, _g10) = w_slice.device_ptr(&stream);
        let (o_ptr, _g11) = o_slice.device_ptr(&stream);
        kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vbf16_bf16(
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

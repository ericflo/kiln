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
use kiln_tensor::{CudaStorage, DType as KtDType, StorageBackend, Tensor as KtTensor};

use crate::kiln_gdn_forward_substitution;

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

fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), GdnError> {
    if t.dtype() != expected {
        return Err(GdnError::Msg(format!(
            "kt-gdn: {name} must be {expected}, got {}",
            t.dtype()
        )));
    }
    if !t.is_contiguous() {
        return Err(GdnError::Msg(format!(
            "kt-gdn: {name} must be contiguous"
        )));
    }
    let st = t
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| GdnError::Msg(format!("kt-gdn: {name} must be CUDA")))?;
    let off = t.layout().start_offset() * expected.size_in_bytes();
    Ok((st, off))
}

fn alloc_cuda_tensor(
    source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, GdnError> {
    let candle_device = source.candle_device().clone();
    let device_index = source.device().index().unwrap_or(0);
    let n: usize = shape.iter().product();
    let storage = kiln_tensor::cuda_zeros(candle_device, device_index, dtype, n)
        .map_err(|e| GdnError::Msg(format!("kt-gdn alloc: {e}")))?;
    KtTensor::from_parts(
        storage,
        kiln_tensor::Layout::contiguous(shape),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| GdnError::Msg(format!("kt-gdn alloc wrap: {e}")))
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
            b as i32,
            h as i32,
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

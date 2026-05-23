//! `kiln_tensor::Tensor`-typed surface for `marlin_w4a16_gemm`.
//!
//! Phase 7 prep — same pattern as #1316–#1319. The kt-API takes
//! F16 inputs (the kernel's native dtype); the BF16→F16 cast at the
//! caller side avoids tying kiln-marlin-gemm to a particular cast
//! op. Callers that have BF16 activations should use
//! `kiln_tensor::ops::to_f16(...)` (or the candle-typed
//! `marlin_w4a16_gemm` which still includes the internal cast) until
//! Phase 7 lands.

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use kiln_kt_bridge::BridgeError;
use kiln_tensor::{CudaStorage, DType as KtDType, Tensor as KtTensor};

use crate::{kiln_marlin_w4a16_gemm, DEFAULT_MAX_PAR, WORKSPACE_TILE_N};

#[derive(Debug)]
pub enum MarlinError {
    Msg(String),
}

impl std::fmt::Display for MarlinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarlinError::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for MarlinError {}

impl From<BridgeError> for MarlinError {
    fn from(e: BridgeError) -> Self {
        MarlinError::Msg(e.message)
    }
}

fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), MarlinError> {
    Ok(kiln_kt_bridge::cuda_storage_and_byte_offset(t, expected, name)?)
}

fn alloc_cuda_tensor(
    source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, MarlinError> {
    Ok(kiln_kt_bridge::alloc_cuda_tensor(source, dtype, shape)?)
}

/// `marlin_w4a16_gemm` over `kiln_tensor::Tensor` operands.
///
/// Inputs:
/// - `a_fp16`: F16 `[m, k]` activations (caller has already cast from BF16
///   if applicable — kiln-tensor's `to_f16` op handles this).
/// - `b_packed`: I64 `[k/16, n*16/8]` Marlin packed weights.
///   (kiln-tensor stores I32 as I64 in the dtype enum — Marlin needs i32
///   stride, but the I64 storage is interpreted as packed i32 via the FFI's
///   `*const c_void` element-agnostic pointer. **Caller must ensure** the
///   layout is correct.)
/// - `scales`: F16 `[k/groupsize, n]` Marlin-permuted scales.
/// - `groupsize`: -1 (per-column) or 128.
///
/// Returns F16 `[m, n]` (caller casts back to BF16 if needed).
pub fn marlin_w4a16_gemm_kt(
    a_fp16: &KtTensor,
    b_packed: &KtTensor,
    scales: &KtTensor,
    groupsize: i32,
) -> Result<KtTensor, MarlinError> {
    let a_shape = a_fp16.shape();
    if a_shape.len() != 2 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: a must be rank-2 [m, k], got {a_shape:?}"
        )));
    }
    let (m, k) = (a_shape[0], a_shape[1]);
    let b_shape = b_packed.shape();
    if b_shape.len() != 2 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: b_packed must be rank-2, got {b_shape:?}"
        )));
    }
    let (b_rows, b_cols) = (b_shape[0], b_shape[1]);
    let s_shape = scales.shape();
    if s_shape.len() != 2 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: scales must be rank-2, got {s_shape:?}"
        )));
    }
    let (s_rows, n) = (s_shape[0], s_shape[1]);

    if k % 128 != 0 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: k must be multiple of 128, got {k}"
        )));
    }
    if n % 256 != 0 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: n must be multiple of 256, got {n}"
        )));
    }
    if b_rows != k / 16 || b_cols != n * 16 / 8 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: b_packed shape [{b_rows}, {b_cols}] != [k/16={}, n*16/8={}]",
            k / 16,
            n * 16 / 8
        )));
    }
    if !(groupsize == -1 || groupsize == 128) {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: groupsize must be -1 or 128, got {groupsize}"
        )));
    }
    let groupsize_for_dims = if groupsize == -1 {
        k
    } else {
        groupsize as usize
    };
    let expected_s_rows = k / groupsize_for_dims;
    if s_rows != expected_s_rows {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: scales rows {s_rows} != {expected_s_rows} (k={k}, groupsize={groupsize})"
        )));
    }

    let (a_st, a_off) = cuda_storage_and_byte_offset(a_fp16, KtDType::F16, "a_fp16")?;
    // b_packed is i32 packed; kiln-tensor doesn't have an I32 dtype today,
    // so the canonical Marlin packed tensor is stored as I64 (8 bytes per
    // packed element pair), or callers can pass U32. Accept either.
    let (b_st, b_off) = if b_packed.dtype() == KtDType::U32 {
        cuda_storage_and_byte_offset(b_packed, KtDType::U32, "b_packed")?
    } else {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: b_packed must be U32 (interpreted as packed i32), got {}",
            b_packed.dtype()
        )));
    };
    let (s_st, s_off) = cuda_storage_and_byte_offset(scales, KtDType::F16, "scales")?;

    let c = alloc_cuda_tensor(a_st, KtDType::F16, vec![m, n])?;
    let workspace_len = (n / WORKSPACE_TILE_N) * DEFAULT_MAX_PAR as usize;
    let workspace = alloc_cuda_tensor(a_st, KtDType::U32, vec![workspace_len])?;

    let c_cuda = c.storage().as_any().downcast_ref::<CudaStorage>().unwrap();
    let w_cuda = workspace
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .unwrap();

    let stream = a_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let dev_ord: i32 = 0;

    let a_slice = a_st.slice().slice(a_off..);
    let b_slice = b_st.slice().slice(b_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let c_slice = c_cuda.slice().slice(0..);
    let w_slice = w_cuda.slice().slice(0..);

    let status = unsafe {
        let (a_ptr, _g1) = a_slice.device_ptr(&stream);
        let (b_ptr, _g2) = b_slice.device_ptr(&stream);
        let (s_ptr, _g3) = s_slice.device_ptr(&stream);
        let (c_ptr, _g4) = c_slice.device_ptr(&stream);
        let (w_ptr, _g5) = w_slice.device_ptr(&stream);

        kiln_marlin_w4a16_gemm(
            a_ptr as *const _,
            b_ptr as *const _,
            c_ptr as *mut _,
            s_ptr as *const _,
            m as i32,
            n as i32,
            k as i32,
            w_ptr as *mut _,
            groupsize,
            dev_ord,
            raw_stream,
            -1,
            -1,
            -1,
            DEFAULT_MAX_PAR,
        )
    };
    if status != 0 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: FFI returned {status}"
        )));
    }
    Ok(c)
}

//! `kiln_tensor::Tensor`-typed surface for `marlin_w4a16_gemm`.
//!
//! Phase 7 prep — same pattern as #1316–#1319. The kt-API takes
//! F16 inputs (the kernel's native dtype); the BF16→F16 cast at the
//! caller side avoids tying kiln-marlin-gemm to a particular cast
//! op. Callers that have BF16 activations should use
//! `kiln_tensor::ops::to_f16(...)` (or the candle-typed
//! `marlin_w4a16_gemm` which still includes the internal cast) until
//! Phase 7 lands.
//!
//! ## Backends
//!
//! - **CUDA** (`feature = "cuda"`): dispatches to the vendored Marlin GEMM
//!   (`csrc/marlin_kernel.cu`), inline `mma.sync` PTX over the packed int4
//!   weights + per-group scales. Byte-identical to the historical path.
//! - **ROCm** (`feature = "rocm"`, Phase R.8): Marlin's PTX can't be hipified,
//!   so this lane is a correctness-first composite — read the packed weights +
//!   scales back to host, dequantize to a dense `[k, n]` F16 weight (the exact
//!   inverse of [`crate::pack::quantize_and_pack`]), upload, and finish with
//!   `kiln_tensor::rocm_matmul`. Native MFMA/WMMA Marlin is the R.10 follow-up.

use kiln_kt_bridge::BridgeError;
#[cfg(feature = "cuda")]
use kiln_tensor::{CudaStorage, DType as KtDType, Tensor as KtTensor};
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

#[cfg(feature = "cuda")]
use crate::{DEFAULT_MAX_PAR, WORKSPACE_TILE_N, kiln_marlin_w4a16_gemm};

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

#[cfg(feature = "rocm")]
impl From<kiln_tensor::Error> for MarlinError {
    fn from(e: kiln_tensor::Error) -> Self {
        MarlinError::Msg(e.to_string())
    }
}

#[cfg(feature = "cuda")]
fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), MarlinError> {
    Ok(kiln_kt_bridge::cuda_storage_and_byte_offset(
        t, expected, name,
    )?)
}

#[cfg(feature = "cuda")]
fn alloc_cuda_tensor(
    source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, MarlinError> {
    Ok(kiln_kt_bridge::alloc_cuda_tensor(source, dtype, shape)?)
}

/// `marlin_w4a16_gemm` over `kiln_tensor::Tensor` operands (CUDA backend).
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
#[cfg(feature = "cuda")]
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

    // Owner-agnostic input pointers (Phase 7 v2).
    let a_ptr = kiln_kt_bridge::cuda_input_device_ptr(a_fp16, KtDType::F16, "a_fp16")?;
    // b_packed is i32 packed; kiln-tensor doesn't have an I32 dtype today,
    // so the canonical Marlin packed tensor is stored as I64 (8 bytes per
    // packed element pair), or callers can pass U32. Accept either.
    let b_ptr = if b_packed.dtype() == KtDType::U32 {
        kiln_kt_bridge::cuda_input_device_ptr(b_packed, KtDType::U32, "b_packed")?
    } else {
        return Err(MarlinError::Msg(format!(
            "kt-marlin: b_packed must be U32 (interpreted as packed i32), got {}",
            b_packed.dtype()
        )));
    };
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(scales, KtDType::F16, "scales")?;
    let (a_st, _) = cuda_storage_and_byte_offset(a_fp16, KtDType::F16, "a_fp16")?;

    let c = alloc_cuda_tensor(a_st, KtDType::F16, vec![m, n])?;
    let workspace_len = (n / WORKSPACE_TILE_N) * DEFAULT_MAX_PAR as usize;
    let workspace = alloc_cuda_tensor(a_st, KtDType::U32, vec![workspace_len])?;
    let c_ptr = kiln_kt_bridge::cuda_output_device_ptr(&c);
    let w_ptr = kiln_kt_bridge::cuda_output_device_ptr(&workspace);

    let raw_stream = a_st.cuda_stream_raw();
    let dev_ord: i32 = 0;

    let status = unsafe {
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

/// `marlin_w4a16_gemm` over `kiln_tensor::Tensor` operands (ROCm backend,
/// Phase R.8).
///
/// Same contract as the CUDA entry — F16 `[m, k]` activations, U32
/// `[k/16, n*16/8]` Marlin packed weights, F16 `[k/groupsize, n]`
/// Marlin-permuted scales, `groupsize ∈ {-1, 128}`, F16 `[m, n]` out — but a
/// different mechanism: Marlin's GEMM kernel is inline `mma.sync` PTX that
/// can't be hipified, so this lane dequantizes the packed int4 weights to a
/// dense F16 `[k, n]` matrix on the host (the exact inverse of the packer,
/// see [`crate::pack::unpack_dequant`]), uploads it, and runs a plain dense
/// `kiln_tensor::rocm_matmul(a, w)`.
///
/// Correctness over speed: the host round-trip + dense GEMM is the canonical
/// reference behaviour; a native MFMA/WMMA Marlin is the R.10 follow-up.
#[cfg(feature = "rocm")]
pub fn marlin_w4a16_gemm_kt(
    a_fp16: &KtTensor,
    b_packed: &KtTensor,
    scales: &KtTensor,
    groupsize: i32,
) -> Result<KtTensor, MarlinError> {
    use kiln_tensor::Device as KtDevice;

    let a_shape = a_fp16.shape();
    if a_shape.len() != 2 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): a must be rank-2 [m, k], got {a_shape:?}"
        )));
    }
    let (m, k) = (a_shape[0], a_shape[1]);
    let b_shape = b_packed.shape();
    if b_shape.len() != 2 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): b_packed must be rank-2, got {b_shape:?}"
        )));
    }
    let (b_rows, b_cols) = (b_shape[0], b_shape[1]);
    let s_shape = scales.shape();
    if s_shape.len() != 2 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): scales must be rank-2, got {s_shape:?}"
        )));
    }
    let (s_rows, n) = (s_shape[0], s_shape[1]);

    if k % 128 != 0 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): k must be multiple of 128, got {k}"
        )));
    }
    if n % 256 != 0 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): n must be multiple of 256, got {n}"
        )));
    }
    if b_rows != k / 16 || b_cols != n * 16 / 8 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): b_packed shape [{b_rows}, {b_cols}] != [k/16={}, n*16/8={}]",
            k / 16,
            n * 16 / 8
        )));
    }
    if !(groupsize == -1 || groupsize == 128) {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): groupsize must be -1 or 128, got {groupsize}"
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
            "kt-marlin(rocm): scales rows {s_rows} != {expected_s_rows} (k={k}, groupsize={groupsize})"
        )));
    }

    // Dtype + device gate. The composite ends in rocm_matmul, so every operand
    // must live on the same ROCm device.
    if a_fp16.dtype() != KtDType::F16 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): a must be F16, got {}",
            a_fp16.dtype()
        )));
    }
    if b_packed.dtype() != KtDType::U32 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): b_packed must be U32 (packed int4), got {}",
            b_packed.dtype()
        )));
    }
    if scales.dtype() != KtDType::F16 {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): scales must be F16, got {}",
            scales.dtype()
        )));
    }
    let device = match a_fp16.device() {
        KtDevice::Rocm(i) => KtDevice::Rocm(i),
        other => {
            return Err(MarlinError::Msg(format!(
                "kt-marlin(rocm): a must be on a ROCm device, got {other}"
            )));
        }
    };
    if b_packed.device() != device || scales.device() != device {
        return Err(MarlinError::Msg(format!(
            "kt-marlin(rocm): all operands must share device {device}; got a={}, b={}, s={}",
            a_fp16.device(),
            b_packed.device(),
            scales.device()
        )));
    }

    // 1) Read the packed weights + scales back to host.
    let b_host: Vec<u32> = b_packed.to_vec::<u32>()?;
    let s_host_f16: Vec<half::f16> = scales.to_vec::<half::f16>()?;
    let s_host_f32: Vec<f32> = s_host_f16.iter().map(|s| s.to_f32()).collect();

    // 2) Dequantize to a dense row-major [k, n] f32 weight (exact inverse of
    //    the packer), then narrow to f16 for the F16 matmul (matching the
    //    kernel's native dtype and the kt-API's F16 output contract).
    let w_f32 = crate::pack::unpack_dequant(&b_host, &s_host_f32, k, n, groupsize as i64);
    let w_f16: Vec<half::f16> = w_f32.iter().map(|&v| half::f16::from_f32(v)).collect();

    // 3) Upload the dense weight [k, n] and run a plain dense GEMM
    //    a[m, k] @ w[k, n] = c[m, n] on the device.
    let w_kt = KtTensor::from_vec_on(device, w_f16, vec![k, n])?;
    let a_contig = if a_fp16.is_contiguous() {
        a_fp16.clone()
    } else {
        a_fp16.contiguous()?
    };
    let c = kiln_tensor::rocm_matmul(&a_contig, &w_kt)?;
    let _ = m; // m is implied by a_contig's shape; kept for symmetry/validation.
    Ok(c)
}

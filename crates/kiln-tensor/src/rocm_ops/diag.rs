//! ROCm wrappers for the diag kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, StorageBackend, Tensor, TensorId};

// Mirrors `kiln_diagonal_extract_async` / `kiln_diag_build_async` from
// `cuda_storage.rs` (same stable C ABI symbols, compiled into
// `libkiln_tensor_rocm_ops.a` by `build.rs::build_rocm()`). diag.cu has no
// cross-lane reductions (one thread per diagonal index), so it is wave-size
// clean — no block-reduce fix needed.
unsafe extern "C" {
    fn kiln_diagonal_extract_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n: i64,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_diag_build_async(
        v: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n: i64,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Map a kt float dtype to the diag kernel's dtype tag (0=F32, 1=BF16, 2=F16).
fn dtype_tag(dtype: DType, op: &str) -> Result<i32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F16 => Ok(2),
        other => Err(Error::Msg(format!("{op}: unsupported dtype {other}"))),
    }
}

/// Extract the main diagonal of a square `[n, n]` ROCm tensor into a fresh
/// rank-1 `[n]` tensor (`out[i] = x[i*n + i]`). ROCm analog of
/// `cuda_diagonal_extract`, routing through the elementwise `diag.cu` kernel
/// (Phase R.5). F32 / BF16 / F16.
pub fn rocm_diagonal_extract(x: &Tensor) -> Result<Tensor> {
    if x.rank() != 2 {
        return Err(Error::Msg(format!(
            "rocm_diagonal_extract: input must be rank-2, got {:?}",
            x.shape()
        )));
    }
    let n = x.shape()[0];
    if x.shape()[1] != n {
        return Err(Error::Msg(format!(
            "rocm_diagonal_extract: input must be square, got {:?}",
            x.shape()
        )));
    }
    let dtype = x.dtype();
    let tag = dtype_tag(dtype, "rocm_diagonal_extract")?;
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "rocm_diagonal_extract: input must be contiguous".to_string(),
        ));
    }

    let bpe = dtype.size_in_bytes();
    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_diagonal_extract: x must be ROCm".to_string()))?;
    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage tensor device is always Rocm"),
    };
    // The kernel writes all `n` output elements, so skip the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe { kiln_diagonal_extract_async(x_ptr, out_ptr, n as i64, tag, raw_stream) };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_diagonal_extract: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(vec![n]), TensorId::next())
        .map_err(|e| Error::Msg(format!("rocm_diagonal_extract: wrap: {e}")))
}

/// Construct a diagonal matrix from a rank-1 vector `v` of length `n`, producing
/// a fresh zero-initialized `[n, n]` tensor with `v` on the main diagonal
/// (`out[i*n + i] = v[i]`). ROCm analog of `cuda_diag_build`, routing through the
/// `diag.cu` build kernel (Phase R.5). F32 / BF16 / F16.
pub fn rocm_diag_build(v: &Tensor) -> Result<Tensor> {
    if v.rank() != 1 {
        return Err(Error::Msg(format!(
            "rocm_diag_build: input must be rank-1, got {:?}",
            v.shape()
        )));
    }
    let dtype = v.dtype();
    let tag = dtype_tag(dtype, "rocm_diag_build")?;
    if !v.is_contiguous() {
        return Err(Error::Msg(
            "rocm_diag_build: input must be contiguous".to_string(),
        ));
    }

    let n = v.element_count();
    let bpe = dtype.size_in_bytes();
    let v_storage = v
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_diag_build: v must be ROCm".to_string()))?;
    let ctx = v_storage.context();
    let device_index = match v.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage tensor device is always Rocm"),
    };
    // The kernel only writes the n diagonal entries — pre-zero the rest.
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, dtype, n * n)?;

    let raw_stream = v_storage.rocm_stream_raw()?;
    let (v_base, _) = v_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let v_off = (v.layout().start_offset() * bpe) as u64;
    let v_ptr = (v_base + v_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe { kiln_diag_build_async(v_ptr, out_ptr, n as i64, tag, raw_stream) };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_diag_build: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(vec![n, n]),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_diag_build: wrap: {e}")))
}

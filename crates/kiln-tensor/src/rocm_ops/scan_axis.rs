//! ROCm wrappers for the scan kernel (`scan_axis.cu`) — cumulative sum /
//! cumulative product over the trailing axis (Phase R.5b).
//!
//! These are the GDN (gated-delta-net) cumsum hot path. Mirrors
//! `cuda_cumsum_axis` / `cuda_cumprod_axis`, routing through the hipify-clean
//! `kiln_scan_last_axis_async` kernel. F32 accumulation regardless of dtype;
//! F32 / BF16 / F16 supported. The scan reduces only through shared memory +
//! `__syncthreads()`, so it is wave32/wave64-correct as-is.

use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// ROCm-side launcher for `scan_axis.cu`, compiled by `build.rs::build_rocm()`
// into `libkiln_tensor_rocm_ops.a` with the same stable C ABI as the CUDA build.
unsafe extern "C" {
    fn kiln_scan_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        kind: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// ROCm cumulative sum along the trailing axis. Routes through
/// `kiln_scan_last_axis_async(..., kind=0)`. F32 / BF16 / F16, F32 accumulation.
pub fn rocm_cumsum_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    rocm_scan_axis_impl(x, axis, 0, "rocm_cumsum_axis")
}

/// ROCm cumulative product along the trailing axis. Routes through
/// `kiln_scan_last_axis_async(..., kind=1)`. F32 / BF16 / F16, F32 accumulation.
pub fn rocm_cumprod_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    rocm_scan_axis_impl(x, axis, 1, "rocm_cumprod_axis")
}

fn rocm_scan_axis_impl(x: &Tensor, axis: usize, kind: i32, label: &str) -> Result<Tensor> {
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!("{label}: unsupported dtype {other}")));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(format!("{label}: input must be contiguous")));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(format!("{label}: input must have rank >= 1")));
    }
    if axis != rank - 1 {
        return Err(Error::Msg(format!(
            "{label}: only last-axis scan supported (axis={axis}, rank={rank})"
        )));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = x.element_count().checked_div(shape[rank - 1]).unwrap_or(0) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{label}: input must be ROCm")))?;
    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage::device is always Rocm"),
    };
    // Scan writes every output element, so an uninitialized buffer is fine.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, x.element_count())?;

    let stream_submission = x_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_scan_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, dtype_tag, kind, raw_stream)
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!("{label}: FFI returned status {status}")));
    }
    stream_submission.complete();

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(shape.to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("{label}: wrap: {e}")))
}

//! ROCm wrappers for the argmax_last_axis kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// The ROCm-side launcher for `argmax_last_axis.cu`, compiled by
// `build.rs::build_rocm()` into `libkiln_tensor_rocm_ops.a` (same stable C ABI
// as the CUDA build). Signature is identical to the `cuda_storage.rs` extern
// `kiln_argmax_last_axis_async`.
unsafe extern "C" {
    fn kiln_argmax_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Per-row argmax over the trailing axis of a contiguous ROCm tensor. ROCm
/// analog of `cuda_argmax_last_axis`, routing through the wave-size-fixed
/// `argmax_last_axis.cu` kernel (Phase R.5).
///
/// Operates on a contiguous `[..., D]` tensor; produces a fresh contiguous
/// tensor of shape `[...]` (trailing axis dropped) with `I64` element type
/// containing per-row argmax indices. F32 / BF16 / F16 inputs supported;
/// comparison happens in F32. Ties break to the lowest index — same convention
/// as `candle_core::Tensor::argmax` and kt's CPU `argmax_last_dim`.
pub fn rocm_argmax_last_axis(x: &Tensor) -> Result<Tensor> {
    let in_dtype = x.dtype();
    let dtype_tag: i32 = match in_dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_argmax_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "rocm_argmax_last_axis: input must be contiguous".to_string(),
        ));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(
            "rocm_argmax_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_argmax_last_axis: input must be ROCm".to_string()))?;
    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_argmax_last_axis: input must be ROCm"),
    };

    // Output: shape = leading axes, dtype = I64. Allocate zeroed (n_rows blocks
    // each write one element; `.max(1)` guards the rank-1 all-reduced case).
    let out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    let out_elem_count: usize = out_shape.iter().product::<usize>().max(1);
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, DType::I64, out_elem_count)?;

    let raw_stream = x_storage.rocm_stream_raw();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * in_dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_argmax_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_argmax_last_axis: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}

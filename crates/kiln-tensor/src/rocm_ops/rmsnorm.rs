//! ROCm wrappers for the rmsnorm kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

unsafe extern "C" {
    fn kiln_rmsnorm_last_axis_async(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        eps: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// RMSNorm over the last axis of a contiguous ROCm tensor. ROCm analog of
/// `cuda_rmsnorm_last_axis`, routing through the wave-size-fixed `rmsnorm.cu`
/// kernel (Phase R.5). F32 / BF16 / F16; F32 accumulation throughout.
///
/// `weight` is a rank-1 `[D]` tensor broadcast over rows; its dtype must match
/// `x.dtype()`. Computes, per row of `n_cols` elements:
///   out[r, c] = x[r, c] * rsqrt(mean(x[r, :]^2) + eps) * weight[c].
pub fn rocm_rmsnorm_last_axis(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_rmsnorm_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "rocm_rmsnorm_last_axis: input must be contiguous".to_string(),
        ));
    }
    if !weight.is_contiguous() {
        return Err(Error::Msg(
            "rocm_rmsnorm_last_axis: weight must be contiguous".to_string(),
        ));
    }
    if weight.dtype() != dtype {
        return Err(Error::Msg(format!(
            "rocm_rmsnorm_last_axis: weight dtype {} != x dtype {}",
            weight.dtype(),
            dtype
        )));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(
            "rocm_rmsnorm_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    if weight.rank() != 1 {
        return Err(Error::Msg(format!(
            "rocm_rmsnorm_last_axis: weight must be rank-1, got rank {}",
            weight.rank()
        )));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    if weight.shape()[0] as i64 != n_cols {
        return Err(Error::Msg(format!(
            "rocm_rmsnorm_last_axis: weight len {} != x trailing axis {}",
            weight.shape()[0],
            n_cols
        )));
    }
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_rmsnorm_last_axis: x must be ROCm".to_string()))?;
    let weight_storage = weight
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_rmsnorm_last_axis: weight must be ROCm".to_string()))?;
    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage tensor is always on a Rocm device"),
    };
    // rmsnorm writes every element of the last-axis output (Pass 2 stores
    // out[row, c] for all rows × cols), so skip the zero-fill via uninit.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, x.element_count())?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (weight_base, _) = weight_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let w_off = (weight.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let weight_ptr = (weight_base + w_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_rmsnorm_last_axis_async(
            x_ptr, weight_ptr, out_ptr, n_rows, n_cols, eps, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_rmsnorm_last_axis: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(shape.to_vec()),
        TensorId::next(),
    )
}

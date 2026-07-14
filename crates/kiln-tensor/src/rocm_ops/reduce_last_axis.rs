//! ROCm wrappers for the reduce_last_axis kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// Same stable C ABI as the CUDA build (`csrc/reduce_last_axis.cu`). Declared
// locally so this module owns its FFI surface, mirroring the exemplar
// `rocm_softmax_last_axis` wrapper. Signatures are copied verbatim from the
// `cuda_storage.rs` extern block.
unsafe extern "C" {
    fn kiln_sum_squared_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_l2norm_apply_async(
        x: *const core::ffi::c_void,
        sum_sq_f32: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        eps: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_sum_last_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        divisor: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

fn dtype_tag(dtype: DType, label: &str) -> Result<i32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F16 => Ok(2),
        other => Err(Error::Msg(format!("{label}: unsupported dtype {other}"))),
    }
}

/// Per-row sum-of-squares over the trailing axis of a contiguous ROCm tensor.
/// ROCm analog of `cuda_sum_squared_last_axis`, routing through the
/// wave-size-fixed `reduce_last_axis.cu` kernel (Phase R.5).
///
/// For a contiguous `[..., D]` input, produces a contiguous F32 output of shape
/// `[...]` (one rank less). The reduction runs in F32 regardless of input dtype;
/// the output is always F32. F32 / BF16 / F16 supported.
pub fn rocm_sum_squared_last_axis(x: &Tensor) -> Result<Tensor> {
    let label = "rocm_sum_squared_last_axis";
    let dtype = x.dtype();
    let tag = dtype_tag(dtype, label)?;
    if !x.is_contiguous() {
        return Err(Error::Msg(format!("{label}: input must be contiguous")));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(format!("{label}: input must have rank >= 1")));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{label}: input must be ROCm")))?;
    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("ROCm tensor device is always Rocm"),
    };
    // Output is always F32; the kernel writes every output row (lane 0 of each
    // per-row block stores unconditionally), so skip the zero-fill.
    let out_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::F32, n_rows as usize)?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_sum_squared_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, tag, raw_stream)
    };
    if status != 0 {
        return Err(Error::Msg(format!("{label}: FFI returned status {status}")));
    }

    let out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}

/// L2 normalization over the trailing axis of a contiguous ROCm tensor. ROCm
/// analog of `cuda_l2norm_last_axis`.
///
/// For a contiguous `[..., D]` input, produces a contiguous output of the same
/// shape and dtype with each row scaled by `1 / sqrt(sum_d x[..., d]^2 + eps)`.
/// Internally: per-row sum-of-squares (F32) via
/// `kiln_sum_squared_last_axis_async`, then per-element scale + cast back via
/// `kiln_l2norm_apply_async`. F32 / BF16 / F16 supported.
pub fn rocm_l2norm_last_axis(x: &Tensor, eps: f32) -> Result<Tensor> {
    let label = "rocm_l2norm_last_axis";
    let dtype = x.dtype();
    let tag = dtype_tag(dtype, label)?;
    if !x.is_contiguous() {
        return Err(Error::Msg(format!("{label}: input must be contiguous")));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(format!("{label}: input must have rank >= 1")));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    // Phase 1: produce per-row sum-of-squares (F32, shape [..rows]).
    let sum_sq = rocm_sum_squared_last_axis(x)?;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{label}: input must be ROCm")))?;
    let sum_sq_storage = sum_sq
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{label}: sum_sq must be ROCm (internal invariant)")))?;
    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("ROCm tensor device is always Rocm"),
    };
    // The apply kernel writes every element of the output, so skip the
    // zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, x.element_count())?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (sum_sq_base, _) = sum_sq_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let sum_sq_off = (sum_sq.layout().start_offset() * DType::F32.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let sum_sq_ptr = (sum_sq_base + sum_sq_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_l2norm_apply_async(
            x_ptr, sum_sq_ptr, out_ptr, n_rows, n_cols, eps, tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!("{label}: FFI returned status {status}")));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(shape.to_vec()),
        TensorId::next(),
    )
}

/// Shared implementation behind [`rocm_sum_last_axis`] / [`rocm_mean_last_axis`].
///
/// Reduces `x` over the trailing axis, producing a tensor of shape
/// `x.shape[..-1]` at the same dtype. F32 accumulation; `divisor` is applied in
/// F32 before the cast back (`1.0` for sum, `1.0 / n_cols` for mean). Routes
/// through the wave-size-fixed `kiln_sum_last_axis_async`.
fn rocm_reduce_last_axis_impl(x: &Tensor, divisor: f32, label: &str) -> Result<Tensor> {
    let dtype = x.dtype();
    let tag = dtype_tag(dtype, label)?;
    if !x.is_contiguous() {
        return Err(Error::Msg(format!("{label}: input must be contiguous")));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(format!("{label}: input must have rank >= 1")));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{label}: input must be ROCm")))?;
    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("ROCm tensor device is always Rocm"),
    };
    // Output: same dtype as input, shape = leading axes. The kernel writes
    // every output row (lane 0 stores unconditionally), but mirror the CUDA
    // path's zeros_ctx for the degenerate empty-leading-axes case.
    let out_shape: Vec<usize> = shape[..rank - 1].to_vec();
    let out_elem_count: usize = out_shape.iter().product::<usize>().max(1);
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, dtype, out_elem_count)?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_sum_last_axis_async(x_ptr, out_ptr, n_rows, n_cols, divisor, tag, raw_stream)
    };
    if status != 0 {
        return Err(Error::Msg(format!("{label}: FFI returned status {status}")));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}

/// `out = sum(x, axis=-1)` on ROCm — produces a tensor of shape
/// `x.shape[..-1]` at the same dtype as `x`. F32 accumulation.
pub fn rocm_sum_last_axis(x: &Tensor) -> Result<Tensor> {
    rocm_reduce_last_axis_impl(x, 1.0, "rocm_sum_last_axis")
}

/// `out = mean(x, axis=-1)` on ROCm — produces a tensor of shape
/// `x.shape[..-1]` at the same dtype as `x`. F32 accumulation; divide-by-N
/// applied in F32 before the cast back.
pub fn rocm_mean_last_axis(x: &Tensor) -> Result<Tensor> {
    let label = "rocm_mean_last_axis";
    if x.rank() == 0 {
        return Err(Error::Msg(format!("{label}: input must have rank >= 1")));
    }
    let n_cols = x.shape()[x.rank() - 1];
    if n_cols == 0 {
        return Err(Error::Msg(format!(
            "{label}: trailing dim is 0; mean is undefined"
        )));
    }
    let inv = 1.0_f32 / (n_cols as f32);
    rocm_reduce_last_axis_impl(x, inv, label)
}

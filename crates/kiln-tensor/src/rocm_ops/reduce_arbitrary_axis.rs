//! ROCm wrappers for the reduce_arbitrary_axis kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

unsafe extern "C" {
    fn kiln_sum_arbitrary_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        outer: i64,
        axis_dim: i64,
        inner: i64,
        divisor: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_minmax_arbitrary_axis_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        outer: i64,
        axis_dim: i64,
        inner: i64,
        kind: i32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_bool_reduce_arbitrary_axis_async(
        mask: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        outer: i64,
        axis_dim: i64,
        inner: i64,
        kind: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Shared implementation behind [`rocm_sum_axis`] / [`rocm_mean_axis`].
///
/// Reduces `x` over a single (non-last) axis via
/// `kiln_sum_arbitrary_axis_async` in `csrc/reduce_arbitrary_axis.cu`
/// (wave-size-fixed block reduction; Phase R.5). One block per
/// (outer, inner) output element. `divisor` is applied in F32 before
/// the cast back: sum ⇒ 1.0, mean ⇒ 1.0 / axis_dim. Issue #1082.
fn rocm_reduce_arbitrary_axis_impl(
    x: &Tensor,
    axis: usize,
    divisor: f32,
    label: &str,
) -> Result<Tensor> {
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
    if axis >= rank {
        return Err(Error::Msg(format!(
            "{label}: axis {axis} out of bounds (rank {rank})"
        )));
    }
    let shape = x.shape();
    let axis_dim = shape[axis] as i64;
    let outer: i64 = shape[..axis].iter().product::<usize>() as i64;
    let inner: i64 = shape[axis + 1..].iter().product::<usize>() as i64;
    let outer = outer.max(1);
    let inner = inner.max(1);

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

    // Output: same dtype, shape = input shape with `axis` removed.
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    let out_elem_count: usize = (outer as usize) * (inner as usize);
    // One block per output element, and that block always writes its element,
    // so every output is fully written — uninit alloc is safe.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, out_elem_count)?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_sum_arbitrary_axis_async(
            x_ptr, out_ptr, outer, axis_dim, inner, divisor, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!("{label}: FFI returned status {status}")));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}

/// `out = sum(x, axis=A)` on ROCm — produces a tensor of shape
/// `x.shape` with axis `A` removed, at the same dtype as `x`.
/// Issue #1082.
pub fn rocm_sum_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    rocm_reduce_arbitrary_axis_impl(x, axis, 1.0, "rocm_sum_axis")
}

/// `out = mean(x, axis=A)` on ROCm — produces a tensor of shape
/// `x.shape` with axis `A` removed, at the same dtype as `x`.
/// Issue #1082.
pub fn rocm_mean_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(
            "rocm_mean_axis: input must have rank >= 1".to_string(),
        ));
    }
    if axis >= rank {
        return Err(Error::Msg(format!(
            "rocm_mean_axis: axis {axis} out of bounds (rank {rank})"
        )));
    }
    let axis_dim = x.shape()[axis];
    if axis_dim == 0 {
        return Err(Error::Msg(
            "rocm_mean_axis: axis dim is 0; mean is undefined".to_string(),
        ));
    }
    let inv = 1.0_f32 / (axis_dim as f32);
    rocm_reduce_arbitrary_axis_impl(x, axis, inv, "rocm_mean_axis")
}

/// Shared implementation behind [`rocm_min_axis`] / [`rocm_max_axis`].
///
/// Reduces `x` over a single axis by min or max via
/// `kiln_minmax_arbitrary_axis_async` in `csrc/reduce_arbitrary_axis.cu`
/// (wave-size-fixed block reduction; Phase R.5). F32 accumulation
/// throughout; cast back to T on the final store. `kind == 0` is MIN,
/// `kind == 1` is MAX. Issue #1082.
fn rocm_minmax_arbitrary_axis_impl(
    x: &Tensor,
    axis: usize,
    kind: i32,
    label: &str,
) -> Result<Tensor> {
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
    if axis >= rank {
        return Err(Error::Msg(format!(
            "{label}: axis {axis} out of bounds (rank {rank})"
        )));
    }
    let shape = x.shape();
    let axis_dim = shape[axis] as i64;
    if axis_dim == 0 {
        return Err(Error::Msg(format!(
            "{label}: axis dim is 0; {label} is undefined"
        )));
    }
    let outer: i64 = shape[..axis].iter().product::<usize>() as i64;
    let inner: i64 = shape[axis + 1..].iter().product::<usize>() as i64;
    let outer = outer.max(1);
    let inner = inner.max(1);

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

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    let out_elem_count: usize = (outer as usize) * (inner as usize);
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, out_elem_count)?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_minmax_arbitrary_axis_async(
            x_ptr, out_ptr, outer, axis_dim, inner, kind, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!("{label}: FFI returned status {status}")));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}

/// `out = min(x, axis=A)` on ROCm — shape `x.shape` with axis `A`
/// removed. Issue #1082.
pub fn rocm_min_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    rocm_minmax_arbitrary_axis_impl(x, axis, 0, "rocm_min_axis")
}

/// `out = max(x, axis=A)` on ROCm — shape `x.shape` with axis `A`
/// removed. Issue #1082.
pub fn rocm_max_axis(x: &Tensor, axis: usize) -> Result<Tensor> {
    rocm_minmax_arbitrary_axis_impl(x, axis, 1, "rocm_max_axis")
}

/// `out = all/any(mask, axis=A)` on ROCm — U8 mask in, U8 result of
/// shape `mask.shape` with axis `A` removed. `kind == 0` is ALL,
/// `kind == 1` is ANY. Routes through
/// `kiln_bool_reduce_arbitrary_axis_async` (wave-size-fixed block
/// reduction; Phase R.5). Issue #1082.
pub fn rocm_bool_reduce_axis(mask: &Tensor, axis: usize, kind: u8) -> Result<Tensor> {
    let label = if kind == 0 {
        "rocm_all_axis"
    } else {
        "rocm_any_axis"
    };
    if mask.dtype() != DType::U8 {
        return Err(Error::Msg(format!(
            "{label}: mask dtype must be U8, got {}",
            mask.dtype()
        )));
    }
    if !mask.is_contiguous() {
        return Err(Error::Msg(format!("{label}: mask must be contiguous")));
    }
    let rank = mask.rank();
    if rank == 0 {
        return Err(Error::Msg(format!("{label}: mask must have rank >= 1")));
    }
    if axis >= rank {
        return Err(Error::Msg(format!(
            "{label}: axis {axis} out of bounds (rank {rank})"
        )));
    }
    let shape = mask.shape();
    let axis_dim = shape[axis] as i64;
    let outer: i64 = shape[..axis].iter().product::<usize>() as i64;
    let inner: i64 = shape[axis + 1..].iter().product::<usize>() as i64;
    let outer = outer.max(1);
    let inner = inner.max(1);

    let x_storage = mask
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg(format!("{label}: input must be ROCm")))?;
    let ctx = x_storage.context();
    let device_index = match mask.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage::device is always Rocm"),
    };

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(axis);
    let out_elem_count: usize = (outer as usize) * (inner as usize);
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::U8, out_elem_count)?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (mask.layout().start_offset() * DType::U8.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_bool_reduce_arbitrary_axis_async(
            x_ptr,
            out_ptr,
            outer,
            axis_dim,
            inner,
            kind as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!("{label}: FFI returned status {status}")));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
}

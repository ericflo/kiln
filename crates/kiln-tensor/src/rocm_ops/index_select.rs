//! ROCm wrappers for the index_select kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

const ROCM_BROADCAST_MAX_RANK: usize = 8;

// FFI surface for `csrc/index_select.cu`. The same hipify-clean kernel object
// services both CUDA and ROCm builds; this module owns the ROCm declarations.
unsafe extern "C" {
    fn kiln_index_select_dim0_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        indices_u32: *const core::ffi::c_void,
        row_bytes: i64,
        n_indices: i64,
        src_n_rows: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_index_select_axis_n_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        indices_u32: *const core::ffi::c_void,
        right_bytes: i64,
        ids_dim: i64,
        src_dim: i64,
        left_size: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_broadcast_to_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        in_shape_host: *const i64,
        target_shape_host: *const i64,
        in_strides_host: *const i64,
        out_strides_host: *const i64,
        rank: i32,
        n_out: i64,
        elem_bytes: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// ROCm-side `index_select` along axis 0 (the dim0 fast path). ROCm analog of
/// `cuda_index_select_dim0`, routing through the hipify-clean `index_select.cu`
/// gather kernel (Phase R.5).
///
/// Output shape is `indices.shape ++ src.shape[1..]`. Out-of-range indices
/// leave their output rows zero (the output is zero-filled before launch).
pub fn rocm_index_select_dim0(src: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if src.dtype().is_packed() {
        return Err(Error::Msg(
            "rocm_index_select_dim0: packed dtype not supported".to_string(),
        ));
    }
    if indices.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "rocm_index_select_dim0: indices dtype must be U32, got {}",
            indices.dtype()
        )));
    }
    if !src.is_contiguous() {
        return Err(Error::Msg(
            "rocm_index_select_dim0: src must be contiguous (call .contiguous()? first)"
                .to_string(),
        ));
    }
    if !indices.is_contiguous() {
        return Err(Error::Msg(
            "rocm_index_select_dim0: indices must be contiguous".to_string(),
        ));
    }

    let src_shape = src.shape();
    if src_shape.is_empty() {
        return Err(Error::Msg(
            "rocm_index_select_dim0: src must have rank >= 1".to_string(),
        ));
    }
    let src_n_rows = src_shape[0];
    let inner: usize = src_shape[1..].iter().product();
    let dtype = src.dtype();
    let bpe = dtype.size_in_bytes();
    let row_bytes = (inner * bpe) as i64;

    let n_indices = indices.element_count();
    let mut out_shape = vec![n_indices];
    out_shape.extend_from_slice(&src_shape[1..]);
    let n_out_elements = n_indices * inner;

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_index_select_dim0: src must be ROCm storage".to_string())
        })?;
    let idx_storage = indices
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_index_select_dim0: indices must be ROCm storage".to_string())
        })?;

    let ctx = src_storage.context();
    let device_index = match src.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_index_select_dim0: src must be on a ROCm device"),
    };
    // Out-of-range indices skip the copy, so unwritten output rows must read
    // back as zero — accumulation/partial-write output, use zeros_ctx.
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, dtype, n_out_elements)?;

    let stream_submission = src_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (idx_base, _) = idx_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let src_off = (src.layout().start_offset() * bpe) as u64;
    let idx_off = (indices.layout().start_offset() * DType::U32.size_in_bytes()) as u64;

    let src_ptr = (src_base + src_off) as *const core::ffi::c_void;
    let idx_ptr = (idx_base + idx_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_index_select_dim0_async(
            src_ptr,
            out_ptr,
            idx_ptr,
            row_bytes,
            n_indices as i64,
            src_n_rows as i64,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_index_select_dim0: FFI returned status {status}"
        )));
    }
    stream_submission.complete();

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
        .map_err(|e| Error::Msg(format!("rocm_index_select_dim0: wrap: {e}")))
}

/// ROCm-side `index_select` along an arbitrary axis. ROCm analog of
/// `cuda_index_select_axis_n`, routing through the hipify-clean
/// `index_select.cu` gather kernel (Phase R.5).
///
/// Output shape is `src.shape[..axis] ++ indices.shape ++ src.shape[axis+1..]`.
/// Out-of-range indices leave their output slices zero (the output is
/// zero-filled before launch).
pub fn rocm_index_select_axis_n(src: &Tensor, axis: usize, indices: &Tensor) -> Result<Tensor> {
    if src.dtype().is_packed() {
        return Err(Error::Msg(
            "rocm_index_select_axis_n: packed dtype not supported".to_string(),
        ));
    }
    if indices.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "rocm_index_select_axis_n: indices dtype must be U32, got {}",
            indices.dtype()
        )));
    }
    if !src.is_contiguous() {
        return Err(Error::Msg(
            "rocm_index_select_axis_n: src must be contiguous (call .contiguous()? first)"
                .to_string(),
        ));
    }
    if !indices.is_contiguous() {
        return Err(Error::Msg(
            "rocm_index_select_axis_n: indices must be contiguous".to_string(),
        ));
    }

    let src_shape = src.shape();
    if src_shape.is_empty() {
        return Err(Error::Msg(
            "rocm_index_select_axis_n: src must have rank >= 1".to_string(),
        ));
    }
    if axis >= src_shape.len() {
        return Err(Error::Msg(format!(
            "rocm_index_select_axis_n: axis {axis} out of bounds (src rank {})",
            src_shape.len()
        )));
    }

    let src_dim = src_shape[axis];
    let left_size: usize = src_shape[..axis].iter().product();
    let right_size: usize = src_shape[axis + 1..].iter().product();
    let dtype = src.dtype();
    let bpe = dtype.size_in_bytes();
    let right_bytes = (right_size * bpe) as i64;
    let ids_dim = indices.element_count();

    let mut out_shape: Vec<usize> = src_shape[..axis].to_vec();
    out_shape.extend_from_slice(indices.shape());
    out_shape.extend_from_slice(&src_shape[axis + 1..]);
    let n_out_elements = left_size * ids_dim * right_size;

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_index_select_axis_n: src must be ROCm storage".to_string())
        })?;
    let idx_storage = indices
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_index_select_axis_n: indices must be ROCm storage".to_string())
        })?;

    let ctx = src_storage.context();
    let device_index = match src.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_index_select_axis_n: src must be on a ROCm device"),
    };
    // Out-of-range indices skip the copy, so unwritten output slices must read
    // back as zero — accumulation/partial-write output, use zeros_ctx.
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, dtype, n_out_elements)?;

    let stream_submission = src_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (idx_base, _) = idx_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let src_off = (src.layout().start_offset() * bpe) as u64;
    let idx_off = (indices.layout().start_offset() * DType::U32.size_in_bytes()) as u64;

    let src_ptr = (src_base + src_off) as *const core::ffi::c_void;
    let idx_ptr = (idx_base + idx_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_index_select_axis_n_async(
            src_ptr,
            out_ptr,
            idx_ptr,
            right_bytes,
            ids_dim as i64,
            src_dim as i64,
            left_size as i64,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_index_select_axis_n: FFI returned status {status}"
        )));
    }
    stream_submission.complete();

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
        .map_err(|e| Error::Msg(format!("rocm_index_select_axis_n: wrap: {e}")))
}

fn dimensions_i64(values: &[usize], name: &str) -> Result<Vec<i64>> {
    values
        .iter()
        .map(|&value| {
            i64::try_from(value).map_err(|_| {
                Error::Msg(format!(
                    "rocm_broadcast_to: {name} dimension {value} exceeds i64"
                ))
            })
        })
        .collect()
}

fn contiguous_strides_i64(shape: &[usize], name: &str) -> Result<Vec<i64>> {
    let mut strides = vec![1_i64; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        let dimension = i64::try_from(shape[axis + 1]).map_err(|_| {
            Error::Msg(format!(
                "rocm_broadcast_to: {name} dimension {} exceeds i64",
                shape[axis + 1]
            ))
        })?;
        strides[axis] = strides[axis + 1].checked_mul(dimension).ok_or_else(|| {
            Error::Msg(format!(
                "rocm_broadcast_to: {name} contiguous stride exceeds i64"
            ))
        })?;
    }
    Ok(strides)
}

/// ROCm-side materialized broadcast. Unlike the generic ROCm broadcast path this
/// does not flatten into a giant index_select; it computes source offsets from
/// an at-most-eight-rank shape/stride descriptor passed by value to the kernel.
/// The descriptor becomes part of a captured kernel node and requires no
/// external metadata allocation or host-to-device copy.
pub fn rocm_broadcast_to(src: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    if src.dtype().is_packed() {
        return Err(Error::Msg(
            "rocm_broadcast_to: packed dtype not supported".to_string(),
        ));
    }
    let in_shape = src.shape();
    if in_shape.len() != target_shape.len() {
        return Err(Error::Msg(format!(
            "rocm_broadcast_to: rank mismatch input {in_shape:?} target {target_shape:?}"
        )));
    }
    for (axis, (&in_d, &out_d)) in in_shape.iter().zip(target_shape).enumerate() {
        if in_d != out_d && in_d != 1 {
            return Err(Error::Msg(format!(
                "rocm_broadcast_to: cannot broadcast axis {axis} from {in_d} to {out_d} \
                 (input {in_shape:?}, target {target_shape:?})"
            )));
        }
    }
    if in_shape == target_shape {
        return src.reshape(src.shape().to_vec());
    }
    if in_shape.len() > ROCM_BROADCAST_MAX_RANK {
        return Err(Error::Msg(format!(
            "rocm_broadcast_to: rank {} exceeds kernel MAX_RANK={ROCM_BROADCAST_MAX_RANK}",
            in_shape.len()
        )));
    }

    let src_c = if src.is_contiguous() {
        src.clone()
    } else {
        src.contiguous()?
    };
    let device = src_c.device();
    let device_index = match device {
        Device::Rocm(i) => i,
        _ => {
            return Err(Error::Msg(
                "rocm_broadcast_to: src must be on ROCm".to_string(),
            ));
        }
    };
    let dtype = src_c.dtype();
    let elem_bytes = dtype.size_in_bytes();
    if elem_bytes == 0 {
        return Err(Error::Msg(format!(
            "rocm_broadcast_to: unsupported zero-width dtype {dtype}"
        )));
    }
    let n_out = target_shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            Error::Msg("rocm_broadcast_to: target element count overflow".to_string())
        })
    })?;
    if n_out == 0 {
        return Err(Error::Msg(
            "rocm_broadcast_to: zero-sized outputs are not supported".to_string(),
        ));
    }

    let in_shape_i64 = dimensions_i64(in_shape, "input")?;
    let target_i64 = dimensions_i64(target_shape, "target")?;
    let in_strides_i64 = contiguous_strides_i64(in_shape, "input")?;
    let out_strides_i64 = contiguous_strides_i64(target_shape, "target")?;
    let n_out_i64 = i64::try_from(n_out)
        .map_err(|_| Error::Msg("rocm_broadcast_to: output size exceeds i64".to_string()))?;

    let src_storage = src_c
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_broadcast_to: src must be ROCm storage".to_string()))?;
    let ctx = src_storage.context();
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n_out)?;

    let stream_submission = src_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (dst_base, _) = out_storage.device_ptr_raw();
    let src_off = (src_c.layout().start_offset() * elem_bytes) as u64;

    let status = unsafe {
        kiln_broadcast_to_async(
            (src_base + src_off) as *const _,
            dst_base as *mut _,
            in_shape_i64.as_ptr(),
            target_i64.as_ptr(),
            in_strides_i64.as_ptr(),
            out_strides_i64.as_ptr(),
            in_shape.len() as i32,
            n_out_i64,
            elem_bytes as i64,
            raw_stream,
        )
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_broadcast_to: FFI returned status {status}"
        )));
    }
    stream_submission.complete();

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(target_shape.to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_broadcast_to: wrap: {e}")))
}

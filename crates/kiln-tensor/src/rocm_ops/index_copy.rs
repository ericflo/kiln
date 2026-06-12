//! ROCm wrapper for the `index_copy_dim0` kernel (`csrc/index_copy.cu`) — an
//! on-device scatter-COPY along axis 0, the inverse of `index_select` (R.9).
//!
//! `dst[indices[i], ..] = src[i, ..]`, written through `dst`'s existing device
//! buffer (no realloc). The destination pointer is therefore stable, and the
//! copy is issued on `dst`'s ACTIVE stream — so inside a
//! `with_active_rocm_stream` scope it lands on (and records into) the HIP-graph
//! capture stream. This is what lets the paged-KV slot write store K/V into
//! `pool[*slot]` with a DEVICE slot index, with no host readback, so the write
//! is recordable into a captured decode graph.

use crate::{DType, Error, Result, RocmStorage, Tensor};

unsafe extern "C" {
    fn kiln_index_copy_dim0_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        indices_u32: *const core::ffi::c_void,
        row_bytes: i64,
        n_indices: i64,
        dst_n_rows: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// In-place ROCm scatter-copy along axis 0: `dst[indices[i], ..] = src[i, ..]`.
///
/// Overwrite semantics (each index must name a distinct `dst` row). `dst`,
/// `src`, and `indices` must be contiguous; `dst`/`src` share the same
/// non-packed dtype and the same inner row size (product of `dims[1..]`);
/// `indices` is U32 with `indices.element_count() == src.dims()[0]`. Bounds:
/// indices `>= dst.dims()[0]` skip their write.
pub fn rocm_index_copy_dim0(dst: &Tensor, indices: &Tensor, src: &Tensor) -> Result<()> {
    if dst.dtype().is_packed() || src.dtype().is_packed() {
        return Err(Error::Msg(
            "rocm_index_copy_dim0: packed dtype not supported".to_string(),
        ));
    }
    if dst.dtype() != src.dtype() {
        return Err(Error::Msg(format!(
            "rocm_index_copy_dim0: dst dtype {} != src dtype {}",
            dst.dtype(),
            src.dtype()
        )));
    }
    if indices.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "rocm_index_copy_dim0: indices dtype must be U32, got {}",
            indices.dtype()
        )));
    }
    if !dst.is_contiguous() || !src.is_contiguous() || !indices.is_contiguous() {
        return Err(Error::Msg(
            "rocm_index_copy_dim0: dst/src/indices must be contiguous".to_string(),
        ));
    }
    let dst_inner: usize = dst.dims().iter().skip(1).product();
    let src_inner: usize = src.dims().iter().skip(1).product();
    if dst_inner != src_inner {
        return Err(Error::Msg(format!(
            "rocm_index_copy_dim0: inner row size mismatch (dst {dst_inner} != src {src_inner})"
        )));
    }
    let n_indices = indices.element_count();
    let src_rows = if src.dims().is_empty() {
        1
    } else {
        src.dims()[0]
    };
    if src_rows != n_indices {
        return Err(Error::Msg(format!(
            "rocm_index_copy_dim0: src rows {src_rows} != indices len {n_indices}"
        )));
    }
    let dst_n_rows = if dst.dims().is_empty() {
        1
    } else {
        dst.dims()[0]
    };
    let bpe = dst.dtype().size_in_bytes();
    let row_bytes = (dst_inner * bpe) as i64;
    if row_bytes == 0 || n_indices == 0 {
        return Ok(());
    }

    let dst_storage = dst
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_index_copy_dim0: dst must be ROCm storage".to_string()))?;
    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_index_copy_dim0: src must be ROCm storage".to_string()))?;
    let idx_storage = indices
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_index_copy_dim0: indices must be ROCm storage".to_string())
        })?;

    let raw_stream = dst_storage.rocm_stream_raw();
    let (dst_base, _) = dst_storage.device_ptr_raw();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (idx_base, _) = idx_storage.device_ptr_raw();
    let dst_ptr = (dst_base + (dst.layout().start_offset() * bpe) as u64) as *mut core::ffi::c_void;
    let src_ptr =
        (src_base + (src.layout().start_offset() * bpe) as u64) as *const core::ffi::c_void;
    let idx_ptr = (idx_base
        + (indices.layout().start_offset() * core::mem::size_of::<u32>()) as u64)
        as *const core::ffi::c_void;

    // SAFETY: all three pointers address contiguous device buffers of the
    // validated extents; `raw_stream` is `dst`'s active ROCm stream (the capture
    // stream under a with_active_rocm_stream scope).
    let status = unsafe {
        kiln_index_copy_dim0_async(
            src_ptr,
            dst_ptr,
            idx_ptr,
            row_bytes,
            n_indices as i64,
            dst_n_rows as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_index_copy_dim0: kiln_index_copy_dim0_async returned status {status}"
        )));
    }
    Ok(())
}

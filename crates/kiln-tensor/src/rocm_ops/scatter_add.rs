//! ROCm wrapper for the scatter_add kernel (`scatter_add.cu`) — the
//! embedding-backward / index_select-backward primitive (Phase R.5b).
//!
//! `out[indices[i], j] += updates[i, j]` over a PRE-ZEROED, contiguous `out`.
//! axis=0 only; 1-D U32 indices. Mirrors `cuda_scatter_add_dim0`.
//!
//! # bf16 atomics
//!
//! HIP has no native `atomicAdd(__hip_bfloat16*)` (ROCm 7.x), so the kernel's
//! bf16 path routes through a CAS-on-dword helper (`kiln_atomic_add_bf16` in
//! `scatter_add.cu`) that is atomic-correct under the index collisions typical
//! of embedding-backward, with F32 accumulation. F32 uses native
//! `atomicAdd(float*)`. F32 + BF16 only (matches the CUDA wrapper); F16 falls
//! through to the host path at the op layer.

use crate::{DType, Device, Error, Result, RocmStorage, Tensor};

unsafe extern "C" {
    fn kiln_scatter_add_dim0_async(
        updates: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        indices_u32: *const core::ffi::c_void,
        n_indices: i64,
        row_inner: i64,
        target_dim: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// In-place ROCm scatter-add into a pre-zeroed `out`:
/// `out[indices[i], j] += updates[i, j]`. F32 native atomicAdd; BF16 via the
/// CAS-on-dword helper. ROCm analog of `cuda_scatter_add_dim0`.
pub fn rocm_scatter_add_dim0(
    out: &Tensor,
    indices: &Tensor,
    updates: &Tensor,
) -> Result<()> {
    // ---- dtype + shape validation (mirrors cuda_scatter_add_dim0) ----
    if out.dtype() != updates.dtype() {
        return Err(Error::Msg(format!(
            "rocm_scatter_add_dim0: out dtype {} != updates dtype {}",
            out.dtype(),
            updates.dtype()
        )));
    }
    let dtype_tag: i32 = match out.dtype() {
        DType::F32 => 0,
        DType::BF16 => 1,
        other => {
            return Err(Error::Msg(format!(
                "rocm_scatter_add_dim0: unsupported dtype {other} (F32/BF16 only)"
            )));
        }
    };
    if indices.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "rocm_scatter_add_dim0: indices dtype must be U32, got {}",
            indices.dtype()
        )));
    }
    if !out.is_contiguous() || !updates.is_contiguous() || !indices.is_contiguous() {
        return Err(Error::Msg(
            "rocm_scatter_add_dim0: out/updates/indices must be contiguous".to_string(),
        ));
    }
    let out_shape = out.shape();
    let upd_shape = updates.shape();
    if out_shape.is_empty() || upd_shape.is_empty() {
        return Err(Error::Msg(
            "rocm_scatter_add_dim0: out and updates must have rank >= 1".to_string(),
        ));
    }
    if out_shape[1..] != upd_shape[1..] {
        return Err(Error::Msg(format!(
            "rocm_scatter_add_dim0: inner shape mismatch out={:?} updates={:?}",
            out_shape, upd_shape
        )));
    }
    let n_indices = indices.element_count();
    if upd_shape[0] != n_indices {
        return Err(Error::Msg(format!(
            "rocm_scatter_add_dim0: updates.shape[0]={} != indices.len()={}",
            upd_shape[0], n_indices
        )));
    }
    let target_dim = out_shape[0];
    let row_inner: usize = out_shape[1..].iter().product::<usize>().max(1);

    // ---- storage downcasts ----
    let out_storage = out
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_scatter_add_dim0: out must be ROCm storage".to_string())
        })?;
    let upd_storage = updates
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_scatter_add_dim0: updates must be ROCm storage".to_string())
        })?;
    let idx_storage = indices
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_scatter_add_dim0: indices must be ROCm storage".to_string())
        })?;

    // Sanity: all three must live on the same ROCm device.
    match (out.device(), updates.device(), indices.device()) {
        (Device::Rocm(_), Device::Rocm(_), Device::Rocm(_)) => {}
        _ => {
            return Err(Error::Msg(
                "rocm_scatter_add_dim0: out/updates/indices must all be ROCm".to_string(),
            ));
        }
    }

    let raw_stream = out_storage.rocm_stream_raw();
    let (upd_base, _) = upd_storage.device_ptr_raw();
    let (idx_base, _) = idx_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let bpe = out.dtype().size_in_bytes();
    let upd_byte_off = (updates.layout().start_offset() * bpe) as u64;
    let idx_byte_off = (indices.layout().start_offset() * DType::U32.size_in_bytes()) as u64;
    let out_byte_off = (out.layout().start_offset() * bpe) as u64;

    let upd_ptr = (upd_base + upd_byte_off) as *const core::ffi::c_void;
    let idx_ptr = (idx_base + idx_byte_off) as *const core::ffi::c_void;
    let out_ptr = (out_base + out_byte_off) as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_scatter_add_dim0_async(
            upd_ptr,
            out_ptr,
            idx_ptr,
            n_indices as i64,
            row_inner as i64,
            target_dim as i64,
            dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_scatter_add_dim0: FFI returned status {status}"
        )));
    }
    Ok(())
}

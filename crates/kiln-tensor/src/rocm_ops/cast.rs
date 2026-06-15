//! ROCm wrappers for the cast kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, StorageBackend, Tensor, TensorId};

// Mirrors `kiln_cast_async` from `cuda_storage.rs` (same stable C ABI symbol,
// compiled into `libkiln_tensor_rocm_ops.a` by `build.rs::build_rocm()`).
unsafe extern "C" {
    fn kiln_cast_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        n_elements: i64,
        cast_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Dtype cast of a contiguous ROCm tensor. ROCm analog of `cuda_cast`, routing
/// through the elementwise `cast.cu` kernel (Phase R.5). Supports the
/// F32 ↔ BF16 ↔ F16 matrix only (integer round-trips stay CPU-side, matching
/// the CUDA path). F32 staging in registers throughout.
pub fn rocm_cast(src: &Tensor, target: DType) -> Result<Tensor> {
    let from = src.dtype();
    if from == target {
        return src
            .contiguous()
            .map_err(|e| Error::Msg(format!("rocm_cast: no-op contiguous: {e}")));
    }
    let cast_tag: i32 = match (from, target) {
        (DType::F32, DType::BF16) => 0,
        (DType::F32, DType::F16) => 1,
        (DType::BF16, DType::F32) => 2,
        (DType::BF16, DType::F16) => 3,
        (DType::F16, DType::F32) => 4,
        (DType::F16, DType::BF16) => 5,
        _ => {
            return Err(Error::Msg(format!(
                "rocm_cast: unsupported pair {from} -> {target} \
                 (ROCm path supports F32↔BF16↔F16 only)"
            )));
        }
    };
    if !src.is_contiguous() {
        return Err(Error::Msg(
            "rocm_cast: contiguous input required".to_string(),
        ));
    }

    let n = src.element_count();
    let from_bpe = from.size_in_bytes();

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_cast: src must be ROCm".to_string()))?;
    let ctx = src_storage.context();
    let device_index = match src.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage tensor device is always Rocm"),
    };
    // The cast kernel writes every one of `n` output elements
    // (`out[i] = cast(src[i])`), so the output is fully overwritten before any
    // read — allocate uninitialized to skip the zero-fill.
    let dst_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, target, n)?;

    let raw_stream = src_storage.rocm_stream_raw();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (dst_base, _) = dst_storage.device_ptr_raw();
    let src_off = (src.layout().start_offset() * from_bpe) as u64;
    let src_ptr = (src_base + src_off) as *const core::ffi::c_void;
    let dst_ptr = dst_base as *mut core::ffi::c_void;

    let status = unsafe { kiln_cast_async(src_ptr, dst_ptr, n as i64, cast_tag, raw_stream) };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_cast: FFI returned status {status}"
        )));
    }
    crate::rocm_synchronize_compute_stream(device_index).map_err(|e| {
        Error::Msg(format!(
            "rocm_cast: synchronize after async kernel launch: {e}"
        ))
    })?;

    let storage_arc: crate::Storage = Arc::new(dst_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(src.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_cast: wrap: {e}")))
}

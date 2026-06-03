//! ROCm wrappers for the lerp kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, StorageBackend, Tensor, TensorId};

// The ROCm-side kernel launcher lives in `csrc/lerp.cu`, compiled by
// `build.rs::build_rocm()` into `libkiln_tensor_rocm_ops.a` (same stable C ABI
// as the CUDA build). Signature copied verbatim from `cuda_storage.rs`.
unsafe extern "C" {
    fn kiln_lerp_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        weight: f32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Element-wise linear interpolation of two contiguous ROCm tensors. ROCm analog
/// of `cuda_lerp`, routing through the `lerp.cu` kernel (Phase R.5).
///
///   out = a + weight * (b - a)
///
/// Both inputs must share shape + dtype (F32/BF16/F16), be contiguous, and live
/// on the same ROCm device. `weight` is a scalar f32 uniform. Returns a fresh
/// contiguous tensor of the same shape/dtype. Mirrors `torch.lerp(a, b, weight)`.
pub fn rocm_lerp(a: &Tensor, b: &Tensor, weight: f32) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!(
            "rocm_lerp: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(Error::Msg(format!(
            "rocm_lerp: dtype mismatch a={} b={}",
            a.dtype(),
            b.dtype()
        )));
    }
    let dtype = a.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!("rocm_lerp: unsupported dtype {other}")));
        }
    };
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "rocm_lerp: contiguous inputs required".to_string(),
        ));
    }

    let n = a.element_count();
    let bpe = dtype.size_in_bytes();

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_lerp: a must be ROCm".to_string()))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_lerp: b must be ROCm".to_string()))?;

    let ctx = a_storage.context();
    let device_index = match a.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_lerp: a must be on a ROCm device"),
    };

    // lerp writes the full output (out[i] = a[i] + w*(b[i]-a[i]) for all n),
    // so allocate uninitialized and skip the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let raw_stream = a_storage.rocm_stream_raw();

    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let a_off = (a.layout().start_offset() * bpe) as u64;
    let b_off = (b.layout().start_offset() * bpe) as u64;

    let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
    let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_lerp_async(
            a_ptr,
            b_ptr,
            out_ptr,
            n as i64,
            weight,
            dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_lerp: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(a.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_lerp: wrap: {e}")))
}

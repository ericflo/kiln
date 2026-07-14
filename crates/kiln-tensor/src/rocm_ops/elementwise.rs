//! ROCm wrapper for the element-wise binary kernel (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, StorageBackend, Tensor, TensorId};

// Stable C ABI launcher compiled by `build.rs::build_rocm()` from
// `csrc/elementwise.cu` (already in the R.2 ROCM_KERNELS baseline). Signature is
// identical to the CUDA-side decl in `cuda_storage.rs` (same symbol, same args)
// — the element-wise binary kernel is one thread per element with no cross-lane
// reductions, so the `.cu` is hipify-clean and needs no wave-size fix.
unsafe extern "C" {
    fn kiln_elementwise_binary_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Element-wise binary op (`out[i] = op(a[i], b[i])`) over two contiguous ROCm
/// tensors. ROCm analog of [`crate::cuda_elementwise_binary`], routing through
/// the `elementwise.cu` kernel (Phase R.5).
///
/// `kind` encodes the op (0=Add, 1=Sub, 2=Mul, 3=Div). Dtype is inferred from
/// `a.dtype()`; must be F32/BF16/F16. Both inputs must share shape, dtype, and
/// ROCm device, and both must be contiguous. Returns a fresh contiguous output
/// of the same shape and dtype.
pub fn rocm_elementwise_binary(a: &Tensor, b: &Tensor, kind: i32) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!(
            "rocm_elementwise_binary: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(Error::Msg(format!(
            "rocm_elementwise_binary: dtype mismatch a={} b={}",
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
            return Err(Error::Msg(format!(
                "rocm_elementwise_binary: unsupported dtype {other}"
            )));
        }
    };
    if !(0..=3).contains(&kind) {
        return Err(Error::Msg(format!(
            "rocm_elementwise_binary: kind {kind} out of range 0..=3"
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "rocm_elementwise_binary: contiguous inputs required".to_string(),
        ));
    }

    let n = a.element_count();
    let bpe = dtype.size_in_bytes();

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_elementwise_binary: a must be ROCm".to_string()))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_elementwise_binary: b must be ROCm".to_string()))?;

    if a_storage.device() != b_storage.device() {
        return Err(Error::Msg(format!(
            "rocm_elementwise_binary: device mismatch a={} b={}",
            a_storage.device(),
            b_storage.device()
        )));
    }

    let ctx = a_storage.context();
    if !Arc::ptr_eq(&ctx, &b_storage.context()) {
        return Err(Error::Msg(
            "rocm_elementwise_binary: both inputs must share one ROCm context".to_string(),
        ));
    }
    let device_index = match a.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_elementwise_binary: a must be on a ROCm device"),
    };

    // Element-wise binary writes the full output (out[i] = op(a[i], b[i]) for
    // all n), so the uninit alloc skips the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let raw_stream = a_storage.rocm_stream_raw()?;

    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let a_off = (a.layout().start_offset() * bpe) as u64;
    let b_off = (b.layout().start_offset() * bpe) as u64;

    let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
    let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_elementwise_binary_async(a_ptr, b_ptr, out_ptr, n as i64, kind, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_elementwise_binary: FFI returned status {status}"
        )));
    }
    crate::rocm_storage::rocm_synchronize_context_same_stream_dependency_with_inputs(
        &ctx,
        &[a_storage, b_storage],
        crate::RocmSyncReason::ElementwiseOutput,
    )
    .map_err(|e| {
        Error::Msg(format!(
            "rocm_elementwise_binary: synchronize after async kernel launch: {e}"
        ))
    })?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(a.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_elementwise_binary: wrap: {e}")))
}

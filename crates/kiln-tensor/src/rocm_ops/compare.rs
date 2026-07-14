//! ROCm wrappers for the compare kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// The ROCm-side kernel launcher lives in `csrc/compare.cu`, compiled by
// `build.rs::build_rocm()` into `libkiln_tensor_rocm_ops.a` (same stable C ABI
// as the CUDA build). Signature copied verbatim from `cuda_storage.rs`.
unsafe extern "C" {
    fn kiln_compare_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Element-wise comparison of two contiguous ROCm tensors. ROCm analog of
/// `cuda_compare`, routing through the `compare.cu` kernel (Phase R.5).
///
/// `kind` encodes the op (0=Eq, 1=Ne, 2=Lt, 3=Le, 4=Gt, 5=Ge). Dtype is
/// inferred from `a.dtype()`; must be F32/BF16/F16. Both inputs must be
/// contiguous and on the same ROCm device. Returns a fresh contiguous U8
/// tensor of the same shape (1 / 0 per element).
pub fn rocm_compare(a: &Tensor, b: &Tensor, kind: i32) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!(
            "rocm_compare: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(Error::Msg(format!(
            "rocm_compare: dtype mismatch a={} b={}",
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
                "rocm_compare: unsupported dtype {other}"
            )));
        }
    };
    if kind < 0 || kind > 5 {
        return Err(Error::Msg(format!("rocm_compare: invalid kind {kind}")));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "rocm_compare: contiguous inputs required".to_string(),
        ));
    }

    let n = a.element_count();
    let bpe = dtype.size_in_bytes();

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_compare: a must be ROCm".to_string()))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_compare: b must be ROCm".to_string()))?;

    let ctx = a_storage.context();
    let device_index = match a.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_compare: a must be on a ROCm device"),
    };

    // Output is U8 (one byte per element). The kernel writes the full output
    // (out[i] = cmp(a[i], b[i]) ? 1 : 0 for all n), so skip the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::U8, n)?;

    let stream_submission = a_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();

    let (a_base, _) = a_storage.device_ptr_raw();
    let (b_base, _) = b_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let a_off = (a.layout().start_offset() * bpe) as u64;
    let b_off = (b.layout().start_offset() * bpe) as u64;

    let a_ptr = (a_base + a_off) as *const core::ffi::c_void;
    let b_ptr = (b_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status =
        unsafe { kiln_compare_async(a_ptr, b_ptr, out_ptr, n as i64, kind, dtype_tag, raw_stream) };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_compare: FFI returned status {status}"
        )));
    }
    stream_submission.complete();

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(a.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_compare: wrap: {e}")))
}

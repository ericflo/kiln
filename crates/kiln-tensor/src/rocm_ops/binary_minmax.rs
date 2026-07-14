//! ROCm wrappers for the binary_minmax kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// Stable C ABI launcher compiled by `build.rs::build_rocm()` from
// `csrc/binary_minmax.cu`. Signature is identical to the CUDA-side decl in
// `cuda_storage.rs` (same symbol, same args) — the binary min/max kernel is
// elementwise (one thread per element, no cross-lane reductions), so the
// `.cu` is hipify-clean and needs no wave-size fix.
unsafe extern "C" {
    fn kiln_binary_minmax_async(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Element-wise binary minimum / maximum of two contiguous ROCm tensors. ROCm
/// analog of `cuda_binary_minmax`, routing through the `binary_minmax.cu`
/// kernel (Phase R.5).
///
/// `kind` encodes the op (0=minimum, 1=maximum). Dtype is inferred from
/// `a.dtype()`; must be F32/BF16/F16. Both inputs must share the same shape,
/// dtype, and ROCm device, and both must be contiguous. BF16/F16 are promoted
/// to F32, compared, then narrowed back (same numerical contract as the CPU
/// reference). Returns a fresh contiguous tensor of the same shape/dtype.
pub fn rocm_binary_minmax(a: &Tensor, b: &Tensor, kind: i32) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!(
            "rocm_binary_minmax: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if a.dtype() != b.dtype() {
        return Err(Error::Msg(format!(
            "rocm_binary_minmax: dtype mismatch a={} b={}",
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
                "rocm_binary_minmax: unsupported dtype {other}"
            )));
        }
    };
    if kind != 0 && kind != 1 {
        return Err(Error::Msg(format!(
            "rocm_binary_minmax: kind must be 0 (min) or 1 (max), got {kind}"
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "rocm_binary_minmax: contiguous inputs required".to_string(),
        ));
    }

    let n = a.element_count();
    let bpe = dtype.size_in_bytes();

    let a_storage = a
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_binary_minmax: a must be ROCm".to_string()))?;
    let b_storage = b
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_binary_minmax: b must be ROCm".to_string()))?;

    let ctx = a_storage.context();
    let device_index = match a.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_binary_minmax: a must be on a ROCm device"),
    };

    // Binary min/max writes the full output (out[i] = min/max(a[i], b[i]) for
    // all n), so the uninit alloc skips the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

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

    let status = unsafe {
        kiln_binary_minmax_async(a_ptr, b_ptr, out_ptr, n as i64, kind, dtype_tag, raw_stream)
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_binary_minmax: FFI returned status {status}"
        )));
    }
    stream_submission.complete();

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(a.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_binary_minmax: wrap: {e}")))
}

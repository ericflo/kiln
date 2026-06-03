//! ROCm wrappers for the scalar_op kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// The ROCm-side launcher for `scalar_op.cu`, compiled by `build.rs::build_rocm()`
// into `libkiln_tensor_rocm_ops.a` with the same stable C ABI as the CUDA build.
// Signature is identical to the `cuda_storage.rs` declaration of the same symbol.
unsafe extern "C" {
    fn kiln_scalar_op_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        c: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Tensor-scalar elementwise op over a contiguous ROCm tensor. ROCm analog of
/// `cuda_scalar_op`, routing through the hipify-clean `scalar_op.cu` kernel
/// (Phase R.5). `kind` selects the op (0..=7, matching `ScalarKind`); the
/// scalar `c` is applied in F32 (BF16/F16 promote to F32, op runs, narrow back).
/// F32 / BF16 / F16 only.
pub fn rocm_scalar_op(x: &Tensor, kind: i32, c: f32) -> Result<Tensor> {
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_scalar_op: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "rocm_scalar_op: contiguous input required".to_string(),
        ));
    }
    if !(0..=7).contains(&kind) {
        return Err(Error::Msg(format!(
            "rocm_scalar_op: kind {kind} out of range 0..=7"
        )));
    }

    let n = x.element_count();
    let bpe = dtype.size_in_bytes();

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_scalar_op: x must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage::device is always Rocm"),
    };
    // Scalar op writes the full output (out[i] = f(x[i], c) for all n), so skip
    // the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let raw_stream = x_storage.rocm_stream_raw();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_scalar_op_async(x_ptr, out_ptr, n as i64, kind, dtype_tag, c, raw_stream)
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_scalar_op: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(x.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_scalar_op: wrap: {e}")))
}

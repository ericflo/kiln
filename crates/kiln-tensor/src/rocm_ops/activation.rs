//! ROCm wrappers for the activation kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// Stable C ABI launcher compiled by `build.rs::build_rocm()` from
// `csrc/activation.cu`. Signature is identical to the CUDA-side decl in
// `cuda_storage.rs` (same symbol, same args) — the activation kernel is
// elementwise (one thread per element, no cross-lane reductions), so the
// `.cu` is hipify-clean and needs no wave-size fix.
unsafe extern "C" {
    fn kiln_activation_unary_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        kind: i32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// ROCm-side unary activation: `out[i] = f_kind(x[i])` over a contiguous
/// tensor. `kind` matches `UnaryKind` (silu/sigmoid/gelu/tanh/relu plus the
/// unary-math family). Mirrors [`crate::cuda_activation_unary`].
pub fn rocm_activation_unary(x: &Tensor, kind: i32) -> Result<Tensor> {
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_activation_unary: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "rocm_activation_unary: contiguous input required".to_string(),
        ));
    }

    let n = x.element_count();

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_activation_unary: x must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage tensor is always Rocm"),
    };

    // Unary activation writes the full output (out[i] = f(x[i]) for all n),
    // so the uninit alloc skips the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let raw_stream = x_storage.rocm_stream_raw();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_activation_unary_async(x_ptr, out_ptr, n as i64, kind, dtype_tag, raw_stream)
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_activation_unary: FFI returned status {status}"
        )));
    }
    crate::rocm_synchronize_compute_stream(device_index).map_err(|e| {
        Error::Msg(format!(
            "rocm_activation_unary: synchronize after async kernel launch: {e}"
        ))
    })?;

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(x.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_activation_unary: wrap: {e}")))
}

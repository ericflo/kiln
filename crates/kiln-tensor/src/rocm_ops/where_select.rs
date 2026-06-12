//! ROCm wrappers for the where_select kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// The ROCm-side launcher lives in `csrc/where_select.cu`, compiled by
// `build.rs::build_rocm()` into `libkiln_tensor_rocm_ops.a` with the same stable
// C ABI as the CUDA build. The signature is identical to the `extern "C"` decl in
// `cuda_storage.rs` (same symbol `kiln_where_select_async`, same args).
unsafe extern "C" {
    fn kiln_where_select_async(
        mask: *const core::ffi::c_void,
        t: *const core::ffi::c_void,
        f: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// ROCm-side ternary mask-based select: `out[i] = mask[i] != 0 ? t[i] : f[i]`.
///
/// `mask` must be U8 on ROCm. `t` / `f` must share dtype (F32/BF16/F16),
/// shape, and ROCm device. All inputs must be contiguous. Output is a fresh
/// contiguous tensor of the same shape + dtype as `t`/`f`. Mirrors
/// `cuda_where_select`. (Phase R.5)
pub fn rocm_where_select(mask: &Tensor, t: &Tensor, f: &Tensor) -> Result<Tensor> {
    if mask.shape() != t.shape() || t.shape() != f.shape() {
        return Err(Error::Msg(format!(
            "rocm_where_select: shape mismatch mask={:?} t={:?} f={:?}",
            mask.shape(),
            t.shape(),
            f.shape()
        )));
    }
    if mask.dtype() != DType::U8 {
        return Err(Error::Msg(format!(
            "rocm_where_select: mask dtype must be U8, got {}",
            mask.dtype()
        )));
    }
    if t.dtype() != f.dtype() {
        return Err(Error::Msg(format!(
            "rocm_where_select: t/f dtype mismatch t={} f={}",
            t.dtype(),
            f.dtype()
        )));
    }
    let dtype = t.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_where_select: unsupported dtype {other}"
            )));
        }
    };
    if !mask.is_contiguous() || !t.is_contiguous() || !f.is_contiguous() {
        return Err(Error::Msg(
            "rocm_where_select: contiguous inputs required".to_string(),
        ));
    }

    let n = t.element_count();
    let bpe = dtype.size_in_bytes();

    let mask_storage = mask
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_where_select: mask must be ROCm".to_string()))?;
    let t_storage = t
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_where_select: t must be ROCm".to_string()))?;
    let f_storage = f
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_where_select: f must be ROCm".to_string()))?;

    let ctx = t_storage.context();
    let device_index = match t.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage::device is always Rocm"),
    };
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, dtype, n)?;

    let raw_stream = t_storage.rocm_stream_raw();
    let (mask_base, _) = mask_storage.device_ptr_raw();
    let (t_base, _) = t_storage.device_ptr_raw();
    let (f_base, _) = f_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    // mask is U8 — one byte per element regardless of dtype.
    let mask_off = mask.layout().start_offset() as u64;
    let t_off = (t.layout().start_offset() * bpe) as u64;
    let f_off = (f.layout().start_offset() * bpe) as u64;

    let mask_ptr = (mask_base + mask_off) as *const core::ffi::c_void;
    let t_ptr = (t_base + t_off) as *const core::ffi::c_void;
    let f_ptr = (f_base + f_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_where_select_async(
            mask_ptr, t_ptr, f_ptr, out_ptr, n as i64, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_where_select: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(t.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_where_select: wrap: {e}")))
}

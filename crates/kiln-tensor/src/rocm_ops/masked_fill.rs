//! ROCm wrappers for the masked_fill kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

unsafe extern "C" {
    fn kiln_masked_fill_u8_async(
        x: *const core::ffi::c_void,
        mask: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        fill_value: f32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_mask_fill_async(
        x: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_elements: i64,
        sq: i64,
        sk: i64,
        fill_value: f32,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// ROCm masked-fill: `out[i] = (mask[i] != 0) ? fill_value : x[i]`.
///
/// `x` and `mask` must be contiguous, same-shape ROCm tensors; `mask` is U8 and
/// `fill_value` arrives as F32 (cast to `x`'s dtype on store). ROCm analog of
/// `cuda_masked_fill`, routing through the hipify-clean `masked_fill.cu` kernel
/// (Phase R.5). F32 / BF16 / F16.
pub fn rocm_masked_fill(x: &Tensor, mask: &Tensor, fill_value: f32) -> Result<Tensor> {
    if x.shape() != mask.shape() {
        return Err(Error::Msg(format!(
            "rocm_masked_fill: shape mismatch x={:?} mask={:?}",
            x.shape(),
            mask.shape()
        )));
    }
    if mask.dtype() != DType::U8 {
        return Err(Error::Msg(format!(
            "rocm_masked_fill: mask dtype must be U8, got {}",
            mask.dtype()
        )));
    }
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_masked_fill: unsupported x dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() || !mask.is_contiguous() {
        return Err(Error::Msg(
            "rocm_masked_fill: contiguous inputs required".to_string(),
        ));
    }

    let n = x.element_count();
    let x_bpe = dtype.size_in_bytes();

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_masked_fill: x must be ROCm".to_string()))?;
    let mask_storage = mask
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_masked_fill: mask must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_masked_fill: x.device() is always Rocm"),
    };
    // masked-fill writes every output element, so skip the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (mask_base, _) = mask_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    // mask dtype is U8 → bpe = 1.
    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let mask_off = mask.layout().start_offset() as u64;

    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let mask_ptr = (mask_base + mask_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_masked_fill_u8_async(
            x_ptr, mask_ptr, out_ptr, n as i64, fill_value, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_masked_fill: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(x.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_masked_fill: wrap: {e}")))
}

/// ROCm causal masked-fill over a contiguous score tensor shaped
/// `[batch_heads, sq, sk]`.
///
/// This is equivalent to building the standard causal mask used by SDPA and
/// then calling [`rocm_masked_fill`], but it avoids allocating or uploading the
/// dense U8 mask. A key column `j` is filled when
/// `j > row + (sk - sq)`, matching the existing causal convention for both
/// full self-attention (`sq == sk`) and tiled prefix attention (`sk > sq`).
pub fn rocm_causal_mask_fill(x: &Tensor, sq: usize, sk: usize, fill_value: f32) -> Result<Tensor> {
    let shape = x.shape();
    if shape.len() != 3 || shape[1] != sq || shape[2] != sk {
        return Err(Error::Msg(format!(
            "rocm_causal_mask_fill: expected [batch_heads, {sq}, {sk}], got {shape:?}"
        )));
    }
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_causal_mask_fill: unsupported x dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "rocm_causal_mask_fill: contiguous input required".to_string(),
        ));
    }
    if sq == 0 || sk == 0 {
        return Err(Error::Msg(
            "rocm_causal_mask_fill: sq and sk must be positive".to_string(),
        ));
    }

    let n = x.element_count();
    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_causal_mask_fill: x must be ROCm".to_string()))?;
    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_causal_mask_fill: x.device() is always Rocm"),
    };
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_causal_mask_fill_async(
            x_ptr, out_ptr, n as i64, sq as i64, sk as i64, fill_value, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_causal_mask_fill: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(shape.to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_causal_mask_fill: wrap: {e}")))
}

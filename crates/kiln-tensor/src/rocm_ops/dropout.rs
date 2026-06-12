//! ROCm wrapper for the dropout kernel (`dropout.cu`) — inverted dropout,
//! training-time (Phase R.5b).
//!
//! `out[i] = (rand_i >= p) ? x[i] / (1 - p) : 0`, with the survival mask emitted
//! to `mask[i] ∈ {0, 1}` (U8). The per-element random value is a counter-based
//! splitmix64 hash of `(seed, i)` — fully deterministic given `seed` and with NO
//! curand / hiprand dependency, so the kernel is self-contained on ROCm. Mirrors
//! `cuda_dropout`; the CPU path keeps its sequential RNG (a different mask byte
//! stream), so parity is at the distribution / scale / shape level, not
//! bit-identical. F32 / BF16 / F16.

use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

unsafe extern "C" {
    fn kiln_dropout_async(
        x: *const core::ffi::c_void,
        y: *mut core::ffi::c_void,
        mask: *mut core::ffi::c_void,
        n_elements: i64,
        p: f32,
        inv_keep: f32,
        seed: u64,
        dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Apply inverted dropout on a ROCm tensor. Returns `(y, mask)` where `mask` is
/// a U8 tensor of the same shape (`0` dropped, `1` survived). ROCm analog of
/// `cuda_dropout`.
pub fn rocm_dropout(x: &Tensor, p: f32, seed: u64) -> Result<(Tensor, Tensor)> {
    if !(0.0..1.0).contains(&p) {
        return Err(Error::Msg(format!(
            "rocm_dropout: p must be in [0, 1), got {p}"
        )));
    }
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_dropout: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "rocm_dropout: input must be contiguous".to_string(),
        ));
    }

    let n = x.element_count();
    let x_bpe = dtype.size_in_bytes();
    let inv_keep = if p == 0.0 { 1.0 } else { 1.0 / (1.0 - p) };

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_dropout: x must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("RocmStorage::device is always Rocm"),
    };

    // y and mask are both written for every element → uninitialized alloc is
    // fine (the kernel writes a 0 or 1 to mask and a value or 0 to y per index).
    let y_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, n)?;
    let mask_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::U8, n)?;

    let raw_stream = x_storage.rocm_stream_raw();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (y_base, _) = y_storage.device_ptr_raw();
    let (mask_base, _) = mask_storage.device_ptr_raw();

    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let y_ptr = y_base as *mut core::ffi::c_void;
    let mask_ptr = mask_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_dropout_async(
            x_ptr, y_ptr, mask_ptr, n as i64, p, inv_keep, seed, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_dropout: FFI returned status {status}"
        )));
    }

    let y_arc: crate::Storage = Arc::new(y_storage);
    let mask_arc: crate::Storage = Arc::new(mask_storage);
    let y = Tensor::from_parts(
        y_arc,
        Layout::contiguous(x.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_dropout: wrap y: {e}")))?;
    let mask = Tensor::from_parts(
        mask_arc,
        Layout::contiguous(x.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_dropout: wrap mask: {e}")))?;
    Ok((y, mask))
}

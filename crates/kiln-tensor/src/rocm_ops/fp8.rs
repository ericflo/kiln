//! ROCm-side FP8 (E4M3FN) quantize / dequantize — the ROCm twin of the
//! CUDA `crate::fp8` module (R.5b). Routes through the same `csrc/fp8.cu`
//! extern-C launchers (now compiled into `libkiln_tensor_rocm_ops.a` by
//! `build.rs::build_rocm()`), wrapping `kt::Tensor`s with `RocmStorage`.
//!
//! Mirrors `cuda_fp8_quantize_with_scale` / `cuda_fp8_quantize_direct` /
//! `cuda_fp8_quantize` / `cuda_fp8_dequantize` / `cuda_fp8_dequantize_direct`
//! 1:1 — same E4M3FN semantics, same `(src / scale)` quantize / `(src * scale)`
//! dequant, same U8 (E4M3 bit pattern) storage dtype. The kernel is pure
//! elementwise bit math (no wave-size hazard).

use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, StorageBackend, Tensor, TensorId};

// Same stable C ABI as the CUDA build's `kiln_fp8_quantize_async` /
// `kiln_fp8_dequantize_async` in `csrc/fp8.cu`.
unsafe extern "C" {
    fn kiln_fp8_quantize_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        n_elements: i64,
        scale: f32,
        src_dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fp8_dequantize_async(
        src: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        n_elements: i64,
        scale: f32,
        dst_dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Maximum representable absolute value in E4M3FN.
pub const ROCM_E4M3_MAX: f32 = 448.0;

fn dtype_tag(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F16 => Ok(2),
        other => Err(Error::Msg(format!("rocm_fp8: unsupported dtype {other}"))),
    }
}

/// ROCm-side FP8 quantize with explicit scale. `src` must be F32/BF16/F16
/// contiguous on ROCm; writes `n` U8 bytes (E4M3FN bit pattern). Values outside
/// `[-448*scale, 448*scale]` are clamped after scaling. Returns a U8 tensor with
/// the same shape as `src`; the caller preserves `scale` for dequant.
pub fn rocm_fp8_quantize_with_scale(src: &Tensor, scale: f32) -> Result<Tensor> {
    if !scale.is_finite() || scale == 0.0 {
        return Err(Error::Msg(format!(
            "rocm_fp8_quantize_with_scale: scale must be finite and non-zero, got {scale}"
        )));
    }
    if !src.is_contiguous() {
        return Err(Error::Msg(
            "rocm_fp8_quantize_with_scale: contiguous input required".to_string(),
        ));
    }
    let src_dtype = src.dtype();
    let src_tag = dtype_tag(src_dtype)?;

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_fp8_quantize_with_scale: src must be ROCm".to_string()))?;
    let ctx = src_storage.context();
    let device_index = match src.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_fp8_quantize_with_scale: src must be ROCm"),
    };
    let n = src.element_count();
    // The kernel writes every output byte, but zeros_ctx is cheap defensiveness
    // and keeps the capture-arena hook (allocates from the arena under capture).
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, DType::U8, n)?;

    let stream_submission = src_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let per = src_dtype.size_in_bytes();
    let src_off = (src.layout().start_offset() * per) as u64;
    let src_ptr = (src_base + src_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status =
        unsafe { kiln_fp8_quantize_async(src_ptr, out_ptr, n as i64, scale, src_tag, raw_stream) };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_fp8_quantize_with_scale: FFI returned status {status}"
        )));
    }
    stream_submission.complete();

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(src.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_fp8_quantize_with_scale: wrap: {e}")))
}

/// ROCm-side FP8 quantize with implicit scale = 1.0 (direct mode). Values
/// outside `[-448, 448]` are clamped. This is the mode the FP8 paged KV cache
/// uses.
pub fn rocm_fp8_quantize_direct(src: &Tensor) -> Result<Tensor> {
    rocm_fp8_quantize_with_scale(src, 1.0)
}

/// ROCm-side FP8 quantize with per-tensor absmax scale. Computes
/// `scale = absmax(src) / 448.0` host-side via a D2H round-trip (offline /
/// calibration use only; the KV-cache hot path uses `rocm_fp8_quantize_direct`).
/// Returns `(u8_tensor, scale)`.
pub fn rocm_fp8_quantize(src: &Tensor) -> Result<(Tensor, f32)> {
    let src_dtype = src.dtype();
    if !matches!(src_dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "rocm_fp8_quantize: unsupported dtype {src_dtype}"
        )));
    }
    if !src.is_contiguous() {
        return Err(Error::Msg(
            "rocm_fp8_quantize: contiguous input required".to_string(),
        ));
    }

    // Pull the tensor host-side (bytes only — interpreted per dtype). Mirrors
    // `kiln_model::fp8::quantize_to_fp8`'s absmax computation.
    let host_tensor = crate::rocm_to_host_copy(src)?;
    let host_cpu = host_tensor
        .storage()
        .as_any()
        .downcast_ref::<crate::CpuStorage>()
        .ok_or_else(|| Error::Msg("rocm_fp8_quantize: host copy must be CpuStorage".to_string()))?;
    let host_bytes = host_cpu.as_bytes();
    let n = src.element_count();
    let abs_max = match src_dtype {
        DType::F32 => {
            let mut m = 0.0f32;
            for i in 0..n {
                let v = f32::from_le_bytes([
                    host_bytes[i * 4],
                    host_bytes[i * 4 + 1],
                    host_bytes[i * 4 + 2],
                    host_bytes[i * 4 + 3],
                ])
                .abs();
                if v > m {
                    m = v;
                }
            }
            m
        }
        DType::BF16 => {
            let mut m = 0.0f32;
            for i in 0..n {
                let v = half::bf16::from_le_bytes([host_bytes[i * 2], host_bytes[i * 2 + 1]])
                    .to_f32()
                    .abs();
                if v > m {
                    m = v;
                }
            }
            m
        }
        DType::F16 => {
            let mut m = 0.0f32;
            for i in 0..n {
                let v = half::f16::from_le_bytes([host_bytes[i * 2], host_bytes[i * 2 + 1]])
                    .to_f32()
                    .abs();
                if v > m {
                    m = v;
                }
            }
            m
        }
        _ => unreachable!(),
    };

    let scale = if abs_max < 1e-12 {
        1.0
    } else {
        abs_max / ROCM_E4M3_MAX
    };
    let quantized = rocm_fp8_quantize_with_scale(src, scale)?;
    Ok((quantized, scale))
}

/// ROCm-side FP8 dequantize. `src` must be a U8 tensor (E4M3FN bit pattern),
/// contiguous, on ROCm. `scale` is multiplied into each dequantized value.
/// `target_dtype` is F32/BF16/F16. Returns a tensor of `target_dtype` with the
/// same shape as `src`.
pub fn rocm_fp8_dequantize(src: &Tensor, scale: f32, target_dtype: DType) -> Result<Tensor> {
    if src.dtype() != DType::U8 {
        return Err(Error::Msg(format!(
            "rocm_fp8_dequantize: src dtype must be U8, got {}",
            src.dtype()
        )));
    }
    if !src.is_contiguous() {
        return Err(Error::Msg(
            "rocm_fp8_dequantize: contiguous input required".to_string(),
        ));
    }
    let dst_tag = dtype_tag(target_dtype)?;

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_fp8_dequantize: src must be ROCm".to_string()))?;
    let ctx = src_storage.context();
    let device_index = match src.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_fp8_dequantize: src must be ROCm"),
    };
    let n = src.element_count();
    let out_storage = RocmStorage::zeros_ctx(&ctx, device_index, target_dtype, n)?;

    let stream_submission = src_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();
    let (src_base, _) = src_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let src_off = src.layout().start_offset() as u64; // U8: 1 byte per element
    let src_ptr = (src_base + src_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_fp8_dequantize_async(src_ptr, out_ptr, n as i64, scale, dst_tag, raw_stream)
    };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_fp8_dequantize: FFI returned status {status}"
        )));
    }
    stream_submission.complete();

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(src.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_fp8_dequantize: wrap: {e}")))
}

/// ROCm-side FP8 dequantize with implicit scale = 1.0 (direct mode).
pub fn rocm_fp8_dequantize_direct(src: &Tensor, target_dtype: DType) -> Result<Tensor> {
    rocm_fp8_dequantize(src, 1.0, target_dtype)
}

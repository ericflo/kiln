//! FP8 (E4M3FN) quantize / dequantize on CUDA — kt-API entry points (#1082).
//!
//! Mirrors `crates/kiln-model/src/fp8.rs` (the candle-typed reference) but
//! operates on `kt::Tensor`s. Used by the Phase 7 `PagedKvCacheKt::new_with_fp8`
//! path to close the FP8 KV-cache write/read loop without bouncing through
//! candle.
//!
//! # Two modes
//!
//! 1. **Scaled** — `cuda_fp8_quantize(src) -> (u8, scale)` computes a
//!    per-tensor absmax scale and quantizes to E4M3FN; the scale must
//!    be preserved alongside the buffer for dequant. Mirrors
//!    `kiln_model::fp8::quantize_to_fp8`.
//!
//! 2. **Direct** (scale = 1.0) — `cuda_fp8_quantize_direct(src)` quantizes
//!    without a per-tensor scale, clamping out-of-range values to ±448.
//!    Used by the FP8 paged KV cache where different writes carry
//!    different value ranges and per-tensor scaling isn't practical.
//!    Mirrors `kiln_model::fp8::quantize_to_fp8_direct`.
//!
//! # FFI surface
//!
//! Two extern calls into `csrc/fp8.cu`:
//!
//! - `kiln_fp8_quantize_async(src, dst, n, scale, src_dtype, stream)`
//! - `kiln_fp8_dequantize_async(src, dst, n, scale, dst_dtype, stream)`
//!
//! The kernel does `(src / scale)` for quantize and `(src * scale)` for
//! dequant. Direct mode passes `scale = 1.0`.

use std::sync::Arc;

use crate::cuda_storage::{CudaStorage, SliceOwner};
use crate::{DType, Result, StorageBackend};

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
pub const E4M3_MAX: f32 = 448.0;

fn dtype_tag(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F16 => Ok(2),
        other => Err(crate::Error::Msg(format!(
            "cuda_fp8: unsupported dtype {other}"
        ))),
    }
}

/// CUDA-side FP8 quantize with explicit scale.
///
/// `src` must be F32/BF16/F16 contiguous on CUDA. Writes `n` U8 bytes
/// (E4M3FN bit pattern) into a fresh output tensor. Values outside
/// `[-448*scale, 448*scale]` are clamped after scaling.
///
/// Returns a U8 tensor with the same shape as `src`. The caller must
/// preserve `scale` alongside the buffer for later dequantization.
pub fn cuda_fp8_quantize_with_scale(
    src: &crate::Tensor,
    scale: f32,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    if !scale.is_finite() || scale == 0.0 {
        return Err(crate::Error::Msg(format!(
            "cuda_fp8_quantize_with_scale: scale must be finite and non-zero, got {scale}"
        )));
    }
    if !src.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_fp8_quantize_with_scale: contiguous input required".to_string(),
        ));
    }
    let src_dtype = src.dtype();
    let src_tag = dtype_tag(src_dtype)?;

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_fp8_quantize_with_scale: src must be CUDA".to_string())
        })?;

    let device_index = match src_storage.device() {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let n = src.element_count();
    // CudaStorage::zeros_ctx (#1082) replaces the old
    // CudaStorage::zeros(candle_device, ...) — the cudarc CudaContext is
    // pulled directly off src_storage, no .candle_device() read.
    let ctx = src_storage.context();
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, DType::U8, n)?;

    // #1082 CUDA-graph fix: route through the thread-local active stream
    // (outside a capture scope this is exactly `ctx.default_stream()`).
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = src_storage.cuda_stream_raw();

    let src_base = match src_storage.slice_owner() {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match out_storage.slice_owner() {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };

    let per = src_dtype.size_in_bytes();
    let src_off = (src.layout().start_offset() * per) as u64;
    let src_ptr = (src_base + src_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_fp8_quantize_async(src_ptr, out_ptr, n as i64, scale, src_tag, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_fp8_quantize_with_scale: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_fp8_quantize_with_scale: wrap: {e}")))
}

/// CUDA-side FP8 quantize with implicit scale = 1.0 (direct mode).
///
/// Values outside `[-448, 448]` are clamped. This is the mode used by
/// the FP8 paged KV cache.
pub fn cuda_fp8_quantize_direct(src: &crate::Tensor) -> Result<crate::Tensor> {
    cuda_fp8_quantize_with_scale(src, 1.0)
}

/// CUDA-side FP8 quantize with per-tensor absmax scale.
///
/// Computes `scale = absmax(src) / 448.0` host-side via a D2H round-trip,
/// then dispatches the scaled quantization. Returns `(u8_tensor, scale)`.
///
/// The D2H round-trip is acceptable because per-tensor scaling is only
/// used in offline paths (model loading, calibration); the FP8 KV-cache
/// hot path uses `cuda_fp8_quantize_direct` instead.
pub fn cuda_fp8_quantize(src: &crate::Tensor) -> Result<(crate::Tensor, f32)> {
    // Compute scale host-side: round-trip the tensor to host bytes and
    // walk it. The reference impl in kiln-model does the same thing
    // (just through candle). We mirror that pattern here using the
    // existing kt CUDA -> host copy path.
    let src_dtype = src.dtype();
    if !matches!(src_dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(crate::Error::Msg(format!(
            "cuda_fp8_quantize: unsupported dtype {src_dtype}"
        )));
    }
    if !src.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_fp8_quantize: contiguous input required".to_string(),
        ));
    }

    // Pull the tensor host-side (bytes only — we interpret per dtype
    // below). This is the same pattern as
    // `kiln_model::fp8::quantize_to_fp8`.
    let host_tensor = crate::cuda_to_host_copy(src)?;
    let host_cpu = host_tensor
        .storage()
        .as_any()
        .downcast_ref::<crate::CpuStorage>()
        .ok_or_else(|| {
            crate::Error::Msg("cuda_fp8_quantize: host copy must be CpuStorage".to_string())
        })?;
    let host_bytes = host_cpu.as_bytes();
    let n = src.element_count();
    let abs_max = match src_dtype {
        DType::F32 => {
            let mut m: f32 = 0.0;
            for i in 0..n {
                let bytes = [
                    host_bytes[i * 4],
                    host_bytes[i * 4 + 1],
                    host_bytes[i * 4 + 2],
                    host_bytes[i * 4 + 3],
                ];
                let v = f32::from_le_bytes(bytes).abs();
                if v > m {
                    m = v;
                }
            }
            m
        }
        DType::BF16 => {
            let mut m: f32 = 0.0;
            for i in 0..n {
                let bytes = [host_bytes[i * 2], host_bytes[i * 2 + 1]];
                let v = half::bf16::from_le_bytes(bytes).to_f32().abs();
                if v > m {
                    m = v;
                }
            }
            m
        }
        DType::F16 => {
            let mut m: f32 = 0.0;
            for i in 0..n {
                let bytes = [host_bytes[i * 2], host_bytes[i * 2 + 1]];
                let v = half::f16::from_le_bytes(bytes).to_f32().abs();
                if v > m {
                    m = v;
                }
            }
            m
        }
        _ => unreachable!(),
    };

    // Same zero-guard as the candle reference.
    let scale = if abs_max < 1e-12 { 1.0 } else { abs_max / E4M3_MAX };

    let quantized = cuda_fp8_quantize_with_scale(src, scale)?;
    Ok((quantized, scale))
}

/// CUDA-side FP8 dequantize.
///
/// `src` must be a U8 tensor (E4M3FN bit pattern), contiguous, on CUDA.
/// `scale` is multiplied into each dequantized value. `target_dtype`
/// must be F32, BF16, or F16. Returns a tensor of `target_dtype` with
/// the same shape as `src`.
pub fn cuda_fp8_dequantize(
    src: &crate::Tensor,
    scale: f32,
    target_dtype: DType,
) -> Result<crate::Tensor> {
    use cudarc::driver::DevicePtr;

    if src.dtype() != DType::U8 {
        return Err(crate::Error::Msg(format!(
            "cuda_fp8_dequantize: src dtype must be U8, got {}",
            src.dtype()
        )));
    }
    if !src.is_contiguous() {
        return Err(crate::Error::Msg(
            "cuda_fp8_dequantize: contiguous input required".to_string(),
        ));
    }
    let dst_tag = dtype_tag(target_dtype)?;

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| crate::Error::Msg("cuda_fp8_dequantize: src must be CUDA".to_string()))?;

    let device_index = match src_storage.device() {
        crate::Device::Cuda(i) => i,
        _ => unreachable!(),
    };
    let n = src.element_count();
    // CudaStorage::zeros_ctx + cuda_stream_raw (#1082) — no
    // .candle_device() read on this hot dequant path.
    let ctx = src_storage.context();
    let out_storage = CudaStorage::zeros_ctx(&ctx, device_index, target_dtype, n)?;

    // #1082 CUDA-graph fix: route through the thread-local active stream
    // (outside a capture scope this is exactly `ctx.default_stream()`).
    let stream = crate::active_cuda_stream(&ctx);
    let raw_stream = src_storage.cuda_stream_raw();

    let src_base = match src_storage.slice_owner() {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { ptr, .. } => *ptr,
    };
    let out_base = match out_storage.slice_owner() {
        SliceOwner::Owned(s) => {
            let (p, _g) = s.device_ptr(&stream);
            p
        }
        SliceOwner::Borrowed { .. } => unreachable!("cuda zeros produces Owned"),
    };

    let src_off = src.layout().start_offset() as u64; // U8: 1 byte per element
    let src_ptr = (src_base + src_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_fp8_dequantize_async(src_ptr, out_ptr, n as i64, scale, dst_tag, raw_stream)
    };
    if status != 0 {
        return Err(crate::Error::Msg(format!(
            "cuda_fp8_dequantize: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        storage_arc,
        crate::Layout::contiguous(src.shape().to_vec()),
        crate::TensorId::next(),
    )
    .map_err(|e| crate::Error::Msg(format!("cuda_fp8_dequantize: wrap: {e}")))
}

/// CUDA-side FP8 dequantize with implicit scale = 1.0 (direct mode).
pub fn cuda_fp8_dequantize_direct(
    src: &crate::Tensor,
    target_dtype: DType,
) -> Result<crate::Tensor> {
    cuda_fp8_dequantize(src, 1.0, target_dtype)
}

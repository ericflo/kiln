//! ROCm W8A16 GEMV wrapper for decode-only projections.

use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

unsafe extern "C" {
    fn kiln_w8a16_gemv_bf16_async(
        x: *const core::ffi::c_void,
        w_q: *const core::ffi::c_void,
        scales: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        m: i64,
        n: i64,
        k: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_w8a8_quantize_bf16_async(
        x: *const core::ffi::c_void,
        x_q: *mut core::ffi::c_void,
        x_scales: *mut core::ffi::c_void,
        m: i64,
        k: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_w8a8_gemv_bf16_async(
        x_q: *const core::ffi::c_void,
        w_q: *const core::ffi::c_void,
        x_scales: *const core::ffi::c_void,
        w_scales: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        m: i64,
        n: i64,
        k: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_w8a16_gemv_argmax_bf16_async(
        x: *const core::ffi::c_void,
        w_q: *const core::ffi::c_void,
        scales: *const core::ffi::c_void,
        scores: *mut core::ffi::c_void,
        out_idx: *mut core::ffi::c_void,
        n: i64,
        k: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Decode-only BF16 activation × row-wise int8 weight GEMV.
///
/// `x` is rank-2 `[m, k]` or rank-3 `[batch, seq, k]` BF16 on ROCm.
/// `w_q` is U8 storage interpreted as signed int8 with shape `[n, k]`.
/// `scales` is F32 `[n]`; dequantized weight row `j` is
/// `i8(w_q[j, :]) * scales[j]`. The result is BF16 with the same leading
/// dimensions as `x` and last dim `n`.
pub fn rocm_w8a16_gemv_bf16(x: &Tensor, w_q: &Tensor, scales: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_bf16: x must be BF16, got {}",
            x.dtype()
        )));
    }
    if w_q.dtype() != DType::U8 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_bf16: w_q must be U8, got {}",
            w_q.dtype()
        )));
    }
    if scales.dtype() != DType::F32 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_bf16: scales must be F32, got {}",
            scales.dtype()
        )));
    }
    if !x.is_contiguous() || !w_q.is_contiguous() || !scales.is_contiguous() {
        return Err(Error::Msg(
            "rocm_w8a16_gemv_bf16: contiguous inputs required".to_string(),
        ));
    }
    let device = match x.device() {
        Device::Rocm(i) => Device::Rocm(i),
        other => {
            return Err(Error::Msg(format!(
                "rocm_w8a16_gemv_bf16: x must be ROCm, got {other}"
            )))
        }
    };
    if w_q.device() != device || scales.device() != device {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_bf16: device mismatch x={} w_q={} scales={}",
            x.device(),
            w_q.device(),
            scales.device()
        )));
    }

    let x_shape = x.shape();
    if x_shape.len() < 2 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_bf16: x rank must be >=2, got {x_shape:?}"
        )));
    }
    let k = *x_shape.last().unwrap();
    let m: usize = x_shape[..x_shape.len() - 1].iter().product();
    let w_shape = w_q.shape();
    if w_shape.len() != 2 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_bf16: w_q must be [n, k], got {w_shape:?}"
        )));
    }
    let (n, wk) = (w_shape[0], w_shape[1]);
    if wk != k {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_bf16: x k={k} but w_q k={wk}"
        )));
    }
    if scales.shape() != [n] {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_bf16: scales shape {:?} != [{n}]",
            scales.shape()
        )));
    }

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_w8a16_gemv_bf16: x must be ROCm".to_string()))?;
    let w_storage = w_q
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_w8a16_gemv_bf16: w_q must be ROCm".to_string()))?;
    let s_storage = scales
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_w8a16_gemv_bf16: scales must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match device {
        Device::Rocm(i) => i,
        _ => unreachable!(),
    };
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::BF16, m * n)?;

    let raw_stream = x_storage.rocm_stream_raw();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (w_base, _) = w_storage.device_ptr_raw();
    let (s_base, _) = s_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_ptr = (x_base + (x.layout().start_offset() * DType::BF16.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let w_ptr = (w_base + (w_q.layout().start_offset() * DType::U8.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let s_ptr = (s_base + (scales.layout().start_offset() * DType::F32.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_w8a16_gemv_bf16_async(
            x_ptr,
            w_ptr,
            s_ptr,
            out_ptr,
            m as i64,
            n as i64,
            k as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_bf16: FFI returned status {status}"
        )));
    }

    let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
    out_shape.push(n);
    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
        .map_err(|e| Error::Msg(format!("rocm_w8a16_gemv_bf16: wrap: {e}")))
}

/// Decode-only BF16 activation quantized to row-wise int8, then int8 × int8
/// GEMV against the existing row-wise int8 weights.
///
/// This is an intentionally more aggressive speed/quality tradeoff than
/// [`rocm_w8a16_gemv_bf16`]: each activation row gets one dynamic F32 scale,
/// so the dot product is `int8(x) @ int8(w)` with `x_scale * w_scale`.
pub fn rocm_w8a8_gemv_bf16(x: &Tensor, w_q: &Tensor, scales: &Tensor) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: x must be BF16, got {}",
            x.dtype()
        )));
    }
    if w_q.dtype() != DType::U8 {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: w_q must be U8, got {}",
            w_q.dtype()
        )));
    }
    if scales.dtype() != DType::F32 {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: scales must be F32, got {}",
            scales.dtype()
        )));
    }
    if !x.is_contiguous() || !w_q.is_contiguous() || !scales.is_contiguous() {
        return Err(Error::Msg(
            "rocm_w8a8_gemv_bf16: contiguous inputs required".to_string(),
        ));
    }
    let device = match x.device() {
        Device::Rocm(i) => Device::Rocm(i),
        other => {
            return Err(Error::Msg(format!(
                "rocm_w8a8_gemv_bf16: x must be ROCm, got {other}"
            )))
        }
    };
    if w_q.device() != device || scales.device() != device {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: device mismatch x={} w_q={} scales={}",
            x.device(),
            w_q.device(),
            scales.device()
        )));
    }

    let x_shape = x.shape();
    if x_shape.len() < 2 {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: x rank must be >=2, got {x_shape:?}"
        )));
    }
    let k = *x_shape.last().unwrap();
    let m: usize = x_shape[..x_shape.len() - 1].iter().product();
    let w_shape = w_q.shape();
    if w_shape.len() != 2 {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: w_q must be [n, k], got {w_shape:?}"
        )));
    }
    let (n, wk) = (w_shape[0], w_shape[1]);
    if wk != k {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: x k={k} but w_q k={wk}"
        )));
    }
    if scales.shape() != [n] {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: scales shape {:?} != [{n}]",
            scales.shape()
        )));
    }

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_w8a8_gemv_bf16: x must be ROCm".to_string()))?;
    let w_storage = w_q
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_w8a8_gemv_bf16: w_q must be ROCm".to_string()))?;
    let s_storage = scales
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_w8a8_gemv_bf16: scales must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match device {
        Device::Rocm(i) => i,
        _ => unreachable!(),
    };
    let x_q_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::U8, m * k)?;
    let x_scales_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::F32, m)?;
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::BF16, m * n)?;

    let raw_stream = x_storage.rocm_stream_raw();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (w_base, _) = w_storage.device_ptr_raw();
    let (s_base, _) = s_storage.device_ptr_raw();
    let (x_q_base, _) = x_q_storage.device_ptr_raw();
    let (x_scales_base, _) = x_scales_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_ptr = (x_base + (x.layout().start_offset() * DType::BF16.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let w_ptr = (w_base + (w_q.layout().start_offset() * DType::U8.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let s_ptr = (s_base + (scales.layout().start_offset() * DType::F32.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let x_q_ptr = x_q_base as *mut core::ffi::c_void;
    let x_scales_ptr = x_scales_base as *mut core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_w8a8_quantize_bf16_async(
            x_ptr,
            x_q_ptr,
            x_scales_ptr,
            m as i64,
            k as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: quantize FFI returned status {status}"
        )));
    }

    let status = unsafe {
        kiln_w8a8_gemv_bf16_async(
            x_q_ptr,
            w_ptr,
            x_scales_ptr as *const core::ffi::c_void,
            s_ptr,
            out_ptr,
            m as i64,
            n as i64,
            k as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_w8a8_gemv_bf16: gemv FFI returned status {status}"
        )));
    }

    let mut out_shape = x_shape[..x_shape.len() - 1].to_vec();
    out_shape.push(n);
    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(out_shape), TensorId::next())
        .map_err(|e| Error::Msg(format!("rocm_w8a8_gemv_bf16: wrap: {e}")))
}

/// Decode-only fused argmax for one BF16 activation row and row-wise W8 weights.
///
/// `x` must flatten to exactly one `[k]` row. The kernel computes
/// `argmax(x @ dequant(w_q)^T)` and returns a rank-1 I64 tensor `[1]`, avoiding
/// materialization of the full vocabulary logits tensor.
pub fn rocm_w8a16_gemv_argmax_bf16(
    x: &Tensor,
    w_q: &Tensor,
    scales: &Tensor,
) -> Result<Tensor> {
    if x.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: x must be BF16, got {}",
            x.dtype()
        )));
    }
    if w_q.dtype() != DType::U8 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: w_q must be U8, got {}",
            w_q.dtype()
        )));
    }
    if scales.dtype() != DType::F32 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: scales must be F32, got {}",
            scales.dtype()
        )));
    }
    if !x.is_contiguous() || !w_q.is_contiguous() || !scales.is_contiguous() {
        return Err(Error::Msg(
            "rocm_w8a16_gemv_argmax_bf16: contiguous inputs required".to_string(),
        ));
    }
    let device = match x.device() {
        Device::Rocm(i) => Device::Rocm(i),
        other => {
            return Err(Error::Msg(format!(
                "rocm_w8a16_gemv_argmax_bf16: x must be ROCm, got {other}"
            )))
        }
    };
    if w_q.device() != device || scales.device() != device {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: device mismatch x={} w_q={} scales={}",
            x.device(),
            w_q.device(),
            scales.device()
        )));
    }

    let x_shape = x.shape();
    if x_shape.len() < 2 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: x rank must be >=2, got {x_shape:?}"
        )));
    }
    let k = *x_shape.last().unwrap();
    let m: usize = x_shape[..x_shape.len() - 1].iter().product();
    if m != 1 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: x must have one leading row, got {m}"
        )));
    }
    let w_shape = w_q.shape();
    if w_shape.len() != 2 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: w_q must be [n, k], got {w_shape:?}"
        )));
    }
    let (n, wk) = (w_shape[0], w_shape[1]);
    if wk != k {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: x k={k} but w_q k={wk}"
        )));
    }
    if scales.shape() != [n] {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: scales shape {:?} != [{n}]",
            scales.shape()
        )));
    }

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_w8a16_gemv_argmax_bf16: x must be ROCm".to_string()))?;
    let w_storage = w_q
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_w8a16_gemv_argmax_bf16: w_q must be ROCm".to_string()))?;
    let s_storage = scales
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_w8a16_gemv_argmax_bf16: scales must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match device {
        Device::Rocm(i) => i,
        _ => unreachable!(),
    };
    let scores_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::F32, n)?;
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, DType::I64, 1)?;

    let raw_stream = x_storage.rocm_stream_raw();
    let (x_base, _) = x_storage.device_ptr_raw();
    let (w_base, _) = w_storage.device_ptr_raw();
    let (s_base, _) = s_storage.device_ptr_raw();
    let (scores_base, _) = scores_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();
    let x_ptr = (x_base + (x.layout().start_offset() * DType::BF16.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let w_ptr = (w_base + (w_q.layout().start_offset() * DType::U8.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let s_ptr = (s_base + (scales.layout().start_offset() * DType::F32.size_in_bytes()) as u64)
        as *const core::ffi::c_void;
    let scores_ptr = scores_base as *mut core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_w8a16_gemv_argmax_bf16_async(
            x_ptr,
            w_ptr,
            s_ptr,
            scores_ptr,
            out_ptr,
            n as i64,
            k as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_w8a16_gemv_argmax_bf16: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(storage_arc, Layout::contiguous(vec![1]), TensorId::next())
        .map_err(|e| Error::Msg(format!("rocm_w8a16_gemv_argmax_bf16: wrap: {e}")))
}

//! ROCm wrappers for the layernorm kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// Same stable C ABI as the CUDA build (`csrc/layernorm.cu`); symbol + args are
// identical to the `unsafe extern "C"` decl in `cuda_storage.rs`.
unsafe extern "C" {
    fn kiln_layernorm_last_axis_async(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        bias: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n_rows: i64,
        n_cols: i64,
        eps: f32,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// LayerNorm over the last axis of a contiguous ROCm tensor. ROCm analog of
/// `cuda_layernorm_last_axis`, routing through the wave-size-fixed `layernorm.cu`
/// kernel (Phase R.5). F32 / BF16 / F16; F32 accumulation throughout.
///
/// `weight.dtype()` and `bias.dtype()` must match `x.dtype()`; both are rank-1
/// `[D]` tensors broadcast over rows. Computes
/// `out[row, c] = (x[row, c] - mean[row]) * inv_std[row] * weight[c] + bias[c]`.
pub fn rocm_layernorm_last_axis(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dtype = x.dtype();
    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "rocm_layernorm_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if !x.is_contiguous() || !weight.is_contiguous() || !bias.is_contiguous() {
        return Err(Error::Msg(
            "rocm_layernorm_last_axis: all inputs must be contiguous".to_string(),
        ));
    }
    if weight.dtype() != dtype || bias.dtype() != dtype {
        return Err(Error::Msg(format!(
            "rocm_layernorm_last_axis: dtype mismatch x={} weight={} bias={}",
            dtype,
            weight.dtype(),
            bias.dtype()
        )));
    }
    let rank = x.rank();
    if rank == 0 {
        return Err(Error::Msg(
            "rocm_layernorm_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    if weight.rank() != 1 || bias.rank() != 1 {
        return Err(Error::Msg(format!(
            "rocm_layernorm_last_axis: weight/bias must be rank-1, got {}/{}",
            weight.rank(),
            bias.rank()
        )));
    }
    let shape = x.shape();
    let n_cols = shape[rank - 1] as i64;
    if weight.shape()[0] as i64 != n_cols || bias.shape()[0] as i64 != n_cols {
        return Err(Error::Msg(format!(
            "rocm_layernorm_last_axis: weight len {} / bias len {} != x trailing axis {}",
            weight.shape()[0],
            bias.shape()[0],
            n_cols
        )));
    }
    let n_rows = (x.element_count() / shape[rank - 1]) as i64;

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_layernorm_last_axis: x must be ROCm".to_string()))?;
    let weight_storage = weight
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_layernorm_last_axis: weight must be ROCm".to_string()))?;
    let bias_storage = bias
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_layernorm_last_axis: bias must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_layernorm_last_axis: x must be a ROCm tensor"),
    };
    // LayerNorm writes every element of the last-axis output (Pass 2 stores
    // out[row, c] for all rows × cols), so skip the zero-fill.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, dtype, x.element_count())?;

    let raw_stream = x_storage.rocm_stream_raw()?;

    let (x_base, _) = x_storage.device_ptr_raw();
    let (weight_base, _) = weight_storage.device_ptr_raw();
    let (bias_base, _) = bias_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let x_off = (x.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let w_off = (weight.layout().start_offset() * dtype.size_in_bytes()) as u64;
    let b_off = (bias.layout().start_offset() * dtype.size_in_bytes()) as u64;

    let x_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let weight_ptr = (weight_base + w_off) as *const core::ffi::c_void;
    let bias_ptr = (bias_base + b_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_layernorm_last_axis_async(
            x_ptr, weight_ptr, bias_ptr, out_ptr, n_rows, n_cols, eps, dtype_tag, raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_layernorm_last_axis: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(shape.to_vec()),
        TensorId::next(),
    )
}

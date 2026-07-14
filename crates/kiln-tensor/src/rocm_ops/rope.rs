//! ROCm wrappers for the rope kernel(s) (Phase R.5).
use std::sync::Arc;

use crate::{DType, Device, Error, Layout, Result, RocmStorage, Tensor, TensorId};

// FFI into the HIP-compiled rope kernel. Signature is byte-identical to the
// CUDA declaration in `cuda_storage.rs` (same symbol `kiln_rope_async`, same
// args) — the kernel source (`csrc/rope.cu`) is hipify-clean (no cross-lane
// reductions), so the CUDA and ROCm paths share one launcher.
unsafe extern "C" {
    fn kiln_rope_async(
        x_in: *const core::ffi::c_void,
        x_out: *mut core::ffi::c_void,
        cos: *const core::ffi::c_void,
        sin: *const core::ffi::c_void,
        leading: i64,
        seq: i64,
        head_dim: i64,
        pair_count: i64,
        x_dtype: i32,
        cs_dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_rope_split_half_4d_async(
        x_in: *const core::ffi::c_void,
        x_out: *mut core::ffi::c_void,
        cos: *const core::ffi::c_void,
        sin: *const core::ffi::c_void,
        batch: i64,
        seq: i64,
        heads: i64,
        head_dim: i64,
        rotary_dim: i64,
        x_dtype: i32,
        cs_dtype: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Rotary position embedding over a contiguous ROCm tensor. ROCm analog of
/// `cuda_rope`, routing through the hipify-clean `rope.cu` kernel (Phase R.5).
///
/// `x` is `[..., seq, head_dim]`; `cos` / `sin` are `[seq, rotary_dim/2]`. The
/// first `rotary_dim` of each row's `head_dim` is rotated; the trailing tail is
/// passed through unchanged (partial-rotary support). F32 / BF16 / F16; all
/// arithmetic promoted to F32. The kernel writes every output element, so the
/// output buffer is allocated uninitialized.
pub fn rocm_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, rotary_dim: usize) -> Result<Tensor> {
    // Shape / dtype validation (mirrors `cuda_rope` / ops/rope.rs::validate).
    if x.rank() < 2 {
        return Err(Error::Msg(format!(
            "rocm_rope: x must have rank >= 2, got shape {:?}",
            x.shape()
        )));
    }
    if cos.rank() != 2 || sin.rank() != 2 {
        return Err(Error::Msg(format!(
            "rocm_rope: cos / sin must be rank-2, got cos={:?} sin={:?}",
            cos.shape(),
            sin.shape()
        )));
    }
    if cos.shape() != sin.shape() {
        return Err(Error::Msg(format!(
            "rocm_rope: cos / sin shape mismatch {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        )));
    }
    if cos.dtype() != sin.dtype() {
        return Err(Error::Msg(format!(
            "rocm_rope: cos / sin dtype mismatch {} vs {}",
            cos.dtype(),
            sin.dtype()
        )));
    }
    let head_dim = x.shape()[x.rank() - 1];
    let seq = x.shape()[x.rank() - 2];
    if cos.shape()[0] != seq {
        return Err(Error::Msg(format!(
            "rocm_rope: cos.shape[0] ({}) != x seq ({seq})",
            cos.shape()[0]
        )));
    }
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "rocm_rope: rotary_dim must be positive and even, got {rotary_dim}"
        )));
    }
    if rotary_dim > head_dim {
        return Err(Error::Msg(format!(
            "rocm_rope: rotary_dim ({rotary_dim}) > head_dim ({head_dim})"
        )));
    }
    if cos.shape()[1] * 2 != rotary_dim {
        return Err(Error::Msg(format!(
            "rocm_rope: cos.shape[1] ({}) * 2 != rotary_dim ({rotary_dim})",
            cos.shape()[1]
        )));
    }
    if !x.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return Err(Error::Msg(
            "rocm_rope: contiguous inputs required".to_string(),
        ));
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "rocm_rope: x dtype must be F32/BF16/F16, got {}",
            x.dtype()
        )));
    }
    if !matches!(cos.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "rocm_rope: cos / sin dtype must be F32/BF16/F16, got {}",
            cos.dtype()
        )));
    }

    let x_dtype = x.dtype();
    let cs_dtype = cos.dtype();
    let x_dtype_tag: i32 = match x_dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        _ => unreachable!(),
    };
    let cs_dtype_tag: i32 = match cs_dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        _ => unreachable!(),
    };

    let pair_count = rotary_dim / 2;
    let leading: usize = x.shape()[..x.rank() - 2].iter().product::<usize>().max(1);
    let n = x.element_count();
    let x_bpe = x_dtype.size_in_bytes();
    let cs_bpe = cs_dtype.size_in_bytes();

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_rope: x must be ROCm".to_string()))?;
    let cos_storage = cos
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_rope: cos must be ROCm".to_string()))?;
    let sin_storage = sin
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_rope: sin must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_rope: x.device() must be Rocm"),
    };

    // The kernel writes the entire output (rotated region + pass-through tail),
    // so an uninitialized buffer is safe — no zero-fill needed.
    let out_storage = RocmStorage::alloc_uninit_ctx(&ctx, device_index, x_dtype, n)?;

    let raw_stream = x_storage.rocm_stream_raw()?;

    let (x_base, _) = x_storage.device_ptr_raw();
    let (cos_base, _) = cos_storage.device_ptr_raw();
    let (sin_base, _) = sin_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let cos_off = (cos.layout().start_offset() * cs_bpe) as u64;
    let sin_off = (sin.layout().start_offset() * cs_bpe) as u64;

    let x_in_ptr = (x_base + x_off) as *const core::ffi::c_void;
    let cos_ptr = (cos_base + cos_off) as *const core::ffi::c_void;
    let sin_ptr = (sin_base + sin_off) as *const core::ffi::c_void;
    let out_ptr = out_base as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_rope_async(
            x_in_ptr,
            out_ptr,
            cos_ptr,
            sin_ptr,
            leading as i64,
            seq as i64,
            head_dim as i64,
            pair_count as i64,
            x_dtype_tag,
            cs_dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_rope: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(x.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_rope: wrap: {e}")))
}

/// ROCm split-half/GPT-NeoX RoPE for `[batch, seq, heads, head_dim]` tensors.
pub fn rocm_rope_split_half(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotary_dim: usize,
) -> Result<Tensor> {
    if x.rank() != 4 {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: x must be rank-4 [batch, seq, heads, head_dim], got {:?}",
            x.shape()
        )));
    }
    if cos.rank() != 2 || sin.rank() != 2 {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: cos / sin must be rank-2, got cos={:?} sin={:?}",
            cos.shape(),
            sin.shape()
        )));
    }
    if cos.shape() != sin.shape() {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: cos / sin shape mismatch {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        )));
    }
    if cos.dtype() != sin.dtype() {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: cos / sin dtype mismatch {} vs {}",
            cos.dtype(),
            sin.dtype()
        )));
    }
    let shape = x.shape();
    let (batch, seq, heads, head_dim) = (shape[0], shape[1], shape[2], shape[3]);
    if cos.shape()[0] != seq {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: cos.shape[0] ({}) != x seq ({seq})",
            cos.shape()[0]
        )));
    }
    if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: rotary_dim must be positive and even, got {rotary_dim}"
        )));
    }
    if rotary_dim > head_dim {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: rotary_dim ({rotary_dim}) > head_dim ({head_dim})"
        )));
    }
    if cos.shape()[1] * 2 != rotary_dim {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: cos.shape[1] ({}) * 2 != rotary_dim ({rotary_dim})",
            cos.shape()[1]
        )));
    }
    if !x.is_contiguous() || !cos.is_contiguous() || !sin.is_contiguous() {
        return Err(Error::Msg(
            "rocm_rope_split_half: contiguous inputs required".to_string(),
        ));
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: x dtype must be F32/BF16/F16, got {}",
            x.dtype()
        )));
    }
    if !matches!(cos.dtype(), DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: cos / sin dtype must be F32/BF16/F16, got {}",
            cos.dtype()
        )));
    }

    let x_dtype = x.dtype();
    let cs_dtype = cos.dtype();
    let x_dtype_tag: i32 = match x_dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        _ => unreachable!(),
    };
    let cs_dtype_tag: i32 = match cs_dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        _ => unreachable!(),
    };

    let x_storage = x
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_rope_split_half: x must be ROCm".to_string()))?;
    let cos_storage = cos
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_rope_split_half: cos must be ROCm".to_string()))?;
    let sin_storage = sin
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| Error::Msg("rocm_rope_split_half: sin must be ROCm".to_string()))?;

    let ctx = x_storage.context();
    let device_index = match x.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_rope_split_half: x.device() must be Rocm"),
    };
    let out_storage =
        RocmStorage::alloc_uninit_ctx(&ctx, device_index, x_dtype, x.element_count())?;

    let raw_stream = x_storage.rocm_stream_raw()?;
    let (x_base, _) = x_storage.device_ptr_raw();
    let (cos_base, _) = cos_storage.device_ptr_raw();
    let (sin_base, _) = sin_storage.device_ptr_raw();
    let (out_base, _) = out_storage.device_ptr_raw();

    let x_bpe = x_dtype.size_in_bytes();
    let cs_bpe = cs_dtype.size_in_bytes();
    let x_off = (x.layout().start_offset() * x_bpe) as u64;
    let cos_off = (cos.layout().start_offset() * cs_bpe) as u64;
    let sin_off = (sin.layout().start_offset() * cs_bpe) as u64;

    let status = unsafe {
        kiln_rope_split_half_4d_async(
            (x_base + x_off) as *const _,
            out_base as *mut _,
            (cos_base + cos_off) as *const _,
            (sin_base + sin_off) as *const _,
            batch as i64,
            seq as i64,
            heads as i64,
            head_dim as i64,
            rotary_dim as i64,
            x_dtype_tag,
            cs_dtype_tag,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Error::Msg(format!(
            "rocm_rope_split_half: FFI returned status {status}"
        )));
    }

    let storage_arc: crate::Storage = Arc::new(out_storage);
    Tensor::from_parts(
        storage_arc,
        Layout::contiguous(x.shape().to_vec()),
        TensorId::next(),
    )
    .map_err(|e| Error::Msg(format!("rocm_rope_split_half: wrap: {e}")))
}

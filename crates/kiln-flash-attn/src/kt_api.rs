//! `kiln_tensor::Tensor`-typed surface alongside the candle-typed
//! API.
//!
//! Phase 7 prep — line 322 of #1082:
//!
//! > **`kiln-flash-attn` Rust API switch.** Current API takes
//! > `candle_core::Tensor`. Switch the Rust shell to
//! > `kiln_tensor::Tensor`; **do not touch the FFI/CUDA-side** —
//! > keep `kiln_flash_attn_fwd`, `_fwd_paged_decode`,
//! > `_fwd_paged_decode_dyn_seqlen`, `_bwd`, and
//! > `kiln_paged_kv_write_token_major_bf16{_slot,}`.
//!
//! Strategy: ship the kiln-tensor surface alongside the existing
//! candle-typed surface. Both call the same FFI symbols. Callers
//! migrate one site at a time. When the migration completes, the
//! candle-typed API is deleted.
//!
//! Today only `flash_attn_fwd_kt` lands here; the remaining four
//! public entry points port in subsequent PRs using the same
//! pattern.

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use kiln_tensor::{CudaStorage, DType as KtDType, StorageBackend, Tensor as KtTensor};

use crate::kiln_flash_attn_fwd;

/// Error type for the kiln-tensor-typed flash-attn surface. Stays
/// independent of candle's error so Phase 7 can delete candle
/// without rewriting this module.
#[derive(Debug)]
pub enum FlashAttnError {
    Msg(String),
}

impl std::fmt::Display for FlashAttnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlashAttnError::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for FlashAttnError {}

/// Borrow the kiln-tensor's [`CudaStorage`], returning a typed
/// reference. Errors if the tensor isn't backed by CUDA, isn't
/// contiguous, or has the wrong dtype.
fn cuda_storage_of<'a>(
    t: &'a KtTensor,
    expected_dtype: KtDType,
    name: &'static str,
) -> Result<&'a CudaStorage, FlashAttnError> {
    if t.dtype() != expected_dtype {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: {name} must be {expected_dtype}, got {}",
            t.dtype()
        )));
    }
    if !t.is_contiguous() {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: {name} must be contiguous (call .contiguous() first)"
        )));
    }
    t.storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            FlashAttnError::Msg(format!("kt-flash-attn: {name} must be a CUDA tensor"))
        })
}

/// `flash_attn_fwd` over `kiln_tensor::Tensor` operands.
///
/// Mirrors [`crate::flash_attn_fwd`] one-for-one: same FFI, same
/// shape contract `[batch, seqlen, num_heads, head_dim]`, same
/// (output, softmax_lse) return tuple. Differences:
/// - Operand type is `kiln_tensor::Tensor` instead of `candle_core::Tensor`.
/// - Output + softmax_lse are allocated through `kiln_tensor`'s
///   `cuda_zeros` rather than `candle_core::Tensor::zeros`.
pub fn flash_attn_fwd_kt(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: q must be rank-4 [batch, seqlen, num_heads, head_dim], got {q_shape:?}"
        )));
    }
    let k_shape = k.shape();
    if k_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: k must be rank-4, got {k_shape:?}"
        )));
    }
    let v_shape = v.shape();
    if v_shape.len() != 4 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: v must be rank-4, got {v_shape:?}"
        )));
    }

    let (b, seqlen_q, num_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let (_b, seqlen_k, num_heads_k, _hd) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);

    if head_dim != 128 && head_dim != 256 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: only head_dim=128,256 supported, got {head_dim}"
        )));
    }

    let q_st = cuda_storage_of(q, KtDType::BF16, "q")?;
    let k_st = cuda_storage_of(k, KtDType::BF16, "k")?;
    let v_st = cuda_storage_of(v, KtDType::BF16, "v")?;

    // All three operands must share the same CUDA device (cudarc
    // pointers from different devices are not interchangeable).
    let candle_device = q_st.candle_device().clone();
    let device_index = q_st.device().index().unwrap_or(0);

    // Output + softmax_lse allocated through kiln-tensor's CUDA
    // helpers. Shapes match the FFI contract exactly.
    let n_out: usize = b * seqlen_q * num_heads * head_dim;
    let n_lse: usize = b * num_heads * seqlen_q;

    let out_storage =
        kiln_tensor::cuda_zeros(candle_device.clone(), device_index, KtDType::BF16, n_out)
            .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: out alloc: {e}")))?;
    let lse_storage =
        kiln_tensor::cuda_zeros(candle_device.clone(), device_index, KtDType::F32, n_lse)
            .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: lse alloc: {e}")))?;

    let out_cuda = out_storage
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("cuda_zeros produced non-CUDA storage");
    let lse_cuda = lse_storage
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("cuda_zeros produced non-CUDA storage");

    // Grab the CUDA stream from the device handle that allocated `q`.
    let stream = candle_device.cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    // Slice each operand by its layout's byte offset, then take the
    // device pointer through the cudarc 0.19 stream-bound API.
    let q_off_bytes = q.layout().start_offset() * KtDType::BF16.size_in_bytes();
    let k_off_bytes = k.layout().start_offset() * KtDType::BF16.size_in_bytes();
    let v_off_bytes = v.layout().start_offset() * KtDType::BF16.size_in_bytes();

    let q_slice = q_st.slice().slice(q_off_bytes..);
    let k_slice = k_st.slice().slice(k_off_bytes..);
    let v_slice = v_st.slice().slice(v_off_bytes..);
    let out_slice = out_cuda.slice().slice(0..);
    let lse_slice = lse_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (v_ptr, _g3) = v_slice.device_ptr(&stream);
        let (out_ptr, _g4) = out_slice.device_ptr(&stream);
        let (lse_ptr, _g5) = lse_slice.device_ptr(&stream);

        kiln_flash_attn_fwd(
            q_ptr as *const _,
            k_ptr as *const _,
            v_ptr as *const _,
            out_ptr as *mut _,
            lse_ptr as *mut _,
            b as i32,
            seqlen_q as i32,
            seqlen_k as i32,
            num_heads as i32,
            num_heads_k as i32,
            head_dim as i32,
            softmax_scale,
            if causal { 1 } else { 0 },
            raw_stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "kt-flash-attn: kiln_flash_attn_fwd returned status {status}"
        )));
    }

    // Wrap each storage into a kiln-tensor with the appropriate shape.
    let out_t = KtTensor::from_parts(
        out_storage,
        kiln_tensor::Layout::contiguous(vec![b, seqlen_q, num_heads, head_dim]),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: out wrap: {e}")))?;
    let lse_t = KtTensor::from_parts(
        lse_storage,
        kiln_tensor::Layout::contiguous(vec![b, num_heads, seqlen_q]),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| FlashAttnError::Msg(format!("kt-flash-attn: lse wrap: {e}")))?;
    Ok((out_t, lse_t))
}

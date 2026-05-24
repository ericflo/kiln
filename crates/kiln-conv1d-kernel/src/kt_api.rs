//! `kiln_tensor::Tensor`-typed surface alongside the candle-typed
//! conv1d API.
//!
//! Phase 7 prep — line 322's spirit applied to kiln-conv1d-kernel.
//! Same C-ABI/FFI; same shape contract; only the Rust shell types
//! switch from `candle_core::Tensor` to `kiln_tensor::Tensor`. The
//! candle-typed `causal_conv1d_update` / `causal_conv1d_prefill`
//! remain in place; Phase 7 deletes them when call sites migrate.

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use kiln_kt_bridge::BridgeError;
use kiln_tensor::{CudaStorage, DType as KtDType, Tensor as KtTensor};

use crate::{kiln_causal_conv1d_prefill_bf16_f32, kiln_causal_conv1d_update_bf16_f32};

#[derive(Debug)]
pub enum Conv1dError {
    Msg(String),
}

impl std::fmt::Display for Conv1dError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Conv1dError::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Conv1dError {}

impl From<BridgeError> for Conv1dError {
    fn from(e: BridgeError) -> Self {
        Conv1dError::Msg(e.message)
    }
}

fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), Conv1dError> {
    Ok(kiln_kt_bridge::cuda_storage_and_byte_offset(t, expected, name)?)
}

fn alloc_cuda_tensor(
    source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, Conv1dError> {
    Ok(kiln_kt_bridge::alloc_cuda_tensor(source, dtype, shape)?)
}

/// `causal_conv1d_update` over `kiln_tensor::Tensor` operands.
///
/// `x`: BF16 `[B, C, 1]`. `weight`: BF16 `[C, K]` (or `[C, 1, K]`).
/// `conv_state`: F32 `[B, C, K-1]` — mutated in place by the kernel.
/// Returns F32 `[B, C, 1]` with SiLU fused inline.
///
/// `conv_state` is borrowed `&KtTensor`; the FFI mutates its
/// underlying CUDA buffer through the raw device pointer. Anti-
/// pattern 16's version-counter bump happens at the call site when
/// kiln-autograd integration lands.
pub fn causal_conv1d_update_kt(
    x: &KtTensor,
    weight: &KtTensor,
    conv_state: &KtTensor,
    kernel_size: usize,
) -> Result<KtTensor, Conv1dError> {
    if kernel_size != 4 {
        return Err(Conv1dError::Msg(format!(
            "kt-conv1d: only kernel_size=4 supported, got {kernel_size}"
        )));
    }
    let x_shape = x.shape();
    if x_shape.len() != 3 || x_shape[2] != 1 {
        return Err(Conv1dError::Msg(format!(
            "kt-conv1d: x must be [B, C, 1], got {x_shape:?}"
        )));
    }
    let (batch, channels) = (x_shape[0], x_shape[1]);

    // Accept either [C, K] or [C, 1, K] for weight.
    let weight_shape = weight.shape();
    let weight_flat = match weight_shape.len() {
        3 => {
            if weight_shape[0] != channels
                || weight_shape[1] != 1
                || weight_shape[2] != kernel_size
            {
                return Err(Conv1dError::Msg(format!(
                    "kt-conv1d: weight shape {weight_shape:?} does not match [{channels}, 1, {kernel_size}]"
                )));
            }
            weight
                .reshape(vec![channels, kernel_size])
                .map_err(|e| Conv1dError::Msg(format!("kt-conv1d weight reshape: {e}")))?
        }
        2 => {
            if weight_shape[0] != channels || weight_shape[1] != kernel_size {
                return Err(Conv1dError::Msg(format!(
                    "kt-conv1d: weight shape {weight_shape:?} does not match [{channels}, {kernel_size}]"
                )));
            }
            weight.clone()
        }
        r => {
            return Err(Conv1dError::Msg(format!(
                "kt-conv1d: weight must be rank 2 or 3, got rank {r}"
            )));
        }
    };

    let cs_shape = conv_state.shape();
    if cs_shape.len() != 3
        || (cs_shape[0], cs_shape[1], cs_shape[2]) != (batch, channels, kernel_size - 1)
    {
        return Err(Conv1dError::Msg(format!(
            "kt-conv1d: conv_state shape {cs_shape:?} does not match [{batch}, {channels}, {}]",
            kernel_size - 1
        )));
    }

    // Owner-agnostic input pointers (Phase 7 v2). conv_state is
    // mutated in place — caller convention: Owned.
    let x_ptr = kiln_kt_bridge::cuda_input_device_ptr(x, KtDType::BF16, "x")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(&weight_flat, KtDType::BF16, "weight")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(conv_state, KtDType::F32, "conv_state")?;
    let (x_st, _) = cuda_storage_and_byte_offset(x, KtDType::BF16, "x")?;
    let out = alloc_cuda_tensor(x_st, KtDType::F32, vec![batch, channels, 1])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);

    let stream = x_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_causal_conv1d_update_bf16_f32(
            x_ptr as *const _,
            w_ptr as *const _,
            s_ptr as *mut _,
            o_ptr as *mut _,
            batch as i32,
            channels as i32,
            kernel_size as i32,
            /* silu */ 1,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Conv1dError::Msg(format!(
            "kt-conv1d: update FFI status {status}"
        )));
    }
    Ok(out)
}

/// `causal_conv1d_prefill` over `kiln_tensor::Tensor` operands.
///
/// `x`: BF16 `[B, C, T]`. `weight`: BF16 `[C, K]`. `conv_state`: F32
/// `[B, C, K-1]` — populated with the last `K-1` samples of each
/// sequence by the kernel. Returns F32 `[B, C, T]` with SiLU fused.
pub fn causal_conv1d_prefill_kt(
    x: &KtTensor,
    weight: &KtTensor,
    conv_state: &KtTensor,
    kernel_size: usize,
) -> Result<KtTensor, Conv1dError> {
    if kernel_size != 4 {
        return Err(Conv1dError::Msg(format!(
            "kt-conv1d: only kernel_size=4 supported, got {kernel_size}"
        )));
    }
    let x_shape = x.shape();
    if x_shape.len() != 3 {
        return Err(Conv1dError::Msg(format!(
            "kt-conv1d: prefill x must be rank-3 [B, C, T], got {x_shape:?}"
        )));
    }
    let (batch, channels, seq_len) = (x_shape[0], x_shape[1], x_shape[2]);
    if seq_len < 1 {
        return Err(Conv1dError::Msg(
            "kt-conv1d: prefill needs seq_len >= 1".to_string(),
        ));
    }

    let weight_shape = weight.shape();
    let weight_flat = match weight_shape.len() {
        3 => {
            if weight_shape[0] != channels
                || weight_shape[1] != 1
                || weight_shape[2] != kernel_size
            {
                return Err(Conv1dError::Msg(format!(
                    "kt-conv1d: weight shape {weight_shape:?} != [{channels}, 1, {kernel_size}]"
                )));
            }
            weight
                .reshape(vec![channels, kernel_size])
                .map_err(|e| Conv1dError::Msg(format!("kt-conv1d weight reshape: {e}")))?
        }
        2 => {
            if weight_shape[0] != channels || weight_shape[1] != kernel_size {
                return Err(Conv1dError::Msg(format!(
                    "kt-conv1d: weight shape {weight_shape:?} != [{channels}, {kernel_size}]"
                )));
            }
            weight.clone()
        }
        r => {
            return Err(Conv1dError::Msg(format!(
                "kt-conv1d: weight rank {r} not 2 or 3"
            )));
        }
    };

    let cs_shape = conv_state.shape();
    if cs_shape.len() != 3
        || (cs_shape[0], cs_shape[1], cs_shape[2]) != (batch, channels, kernel_size - 1)
    {
        return Err(Conv1dError::Msg(format!(
            "kt-conv1d: prefill conv_state {cs_shape:?} != [{batch}, {channels}, {}]",
            kernel_size - 1
        )));
    }

    let x_ptr = kiln_kt_bridge::cuda_input_device_ptr(x, KtDType::BF16, "x")?;
    let w_ptr = kiln_kt_bridge::cuda_input_device_ptr(&weight_flat, KtDType::BF16, "weight")?;
    let s_ptr = kiln_kt_bridge::cuda_input_device_ptr(conv_state, KtDType::F32, "conv_state")?;
    let (x_st, _) = cuda_storage_and_byte_offset(x, KtDType::BF16, "x")?;
    let out = alloc_cuda_tensor(x_st, KtDType::F32, vec![batch, channels, seq_len])?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);

    let stream = x_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_causal_conv1d_prefill_bf16_f32(
            x_ptr as *const _,
            w_ptr as *const _,
            s_ptr as *mut _,
            o_ptr as *mut _,
            batch as i32,
            channels as i32,
            seq_len as i32,
            kernel_size as i32,
            /* silu */ 1,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(Conv1dError::Msg(format!(
            "kt-conv1d: prefill FFI status {status}"
        )));
    }
    Ok(out)
}

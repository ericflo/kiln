//! `kiln_tensor::Tensor`-typed surface for the canonical
//! `fused_rmsnorm` + `fused_rmsnorm_backward` entry points.
//!
//! Phase 7 prep — same pattern as kiln-flash-attn + kiln-conv1d-kernel.
//! Same FFI underneath (`kiln_fused_rmsnorm` + `kiln_fused_rmsnorm_bwd`).
//!
//! Only the two RMSNorm functions are ported in this PR; the rest of
//! the rmsnorm-kernel surface (rotary, MLP fusions, LoRA, SGD/AdamW
//! steps, etc.) follows the same template and lands in subsequent
//! PRs.

use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use kiln_tensor::{CudaStorage, DType as KtDType, StorageBackend, Tensor as KtTensor};

use crate::{kiln_fused_rmsnorm, kiln_fused_rmsnorm_bwd};

#[derive(Debug)]
pub enum RmsNormError {
    Msg(String),
}

impl std::fmt::Display for RmsNormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RmsNormError::Msg(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for RmsNormError {}

fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), RmsNormError> {
    if t.dtype() != expected {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm: {name} must be {expected}, got {}",
            t.dtype()
        )));
    }
    if !t.is_contiguous() {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm: {name} must be contiguous"
        )));
    }
    let st = t
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| RmsNormError::Msg(format!("kt-rmsnorm: {name} must be CUDA")))?;
    let off = t.layout().start_offset() * expected.size_in_bytes();
    Ok((st, off))
}

fn alloc_cuda_tensor(
    source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, RmsNormError> {
    let candle_device = source.candle_device().clone();
    let device_index = source.device().index().unwrap_or(0);
    let n: usize = shape.iter().product();
    let storage = kiln_tensor::cuda_zeros(candle_device, device_index, dtype, n)
        .map_err(|e| RmsNormError::Msg(format!("kt-rmsnorm alloc: {e}")))?;
    KtTensor::from_parts(
        storage,
        kiln_tensor::Layout::contiguous(shape),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| RmsNormError::Msg(format!("kt-rmsnorm alloc wrap: {e}")))
}

/// `fused_rmsnorm` over `kiln_tensor::Tensor` operands.
///
/// `x`: BF16 `[..., hidden]`. `weight`: BF16 `[hidden]`. Returns BF16
/// same shape as `x`. Matches `kiln-model::forward::rms_norm`
/// (Qwen3.5-style, weight centred on 0).
pub fn fused_rmsnorm_kt(
    x: &KtTensor,
    weight: &KtTensor,
    eps: f32,
) -> Result<KtTensor, RmsNormError> {
    let x_shape = x.shape().to_vec();
    let hidden = *x_shape.last().ok_or_else(|| {
        RmsNormError::Msg("kt-rmsnorm: x must have rank >= 1".to_string())
    })?;
    let weight_shape = weight.shape();
    if weight_shape != [hidden] {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm: weight {weight_shape:?} != [{hidden}]"
        )));
    }
    if hidden > 8192 {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm: hidden dim {hidden} > 8192 envelope"
        )));
    }
    let rows: usize = x_shape[..x_shape.len() - 1].iter().product();

    let (x_st, x_off) = cuda_storage_and_byte_offset(x, KtDType::BF16, "x")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::BF16, "weight")?;
    let out = alloc_cuda_tensor(x_st, KtDType::BF16, x_shape.clone())?;
    if rows == 0 {
        return Ok(out);
    }
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = x_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_slice = x_st.slice().slice(x_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (x_ptr, _g1) = x_slice.device_ptr(&stream);
        let (w_ptr, _g2) = w_slice.device_ptr(&stream);
        let (o_ptr, _g3) = o_slice.device_ptr(&stream);
        kiln_fused_rmsnorm(
            x_ptr as *const _,
            w_ptr as *const _,
            o_ptr as *mut _,
            rows as i32,
            hidden as i32,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm: FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `fused_rmsnorm_backward` over `kiln_tensor::Tensor` operands.
///
/// Returns `(grad_x, grad_w_partial_f32)` where `grad_w_partial_f32`
/// is the rows-reduced partial gradient that the caller sums across
/// the rows axis to get the final weight gradient.
///
/// Shapes:
/// - `x`, `weight`, `grad_out`: BF16, matching the forward
/// - `grad_x`: BF16, shape == x
/// - `grad_w_partial_f32`: F32 `[rows / WARP_SIZE_OR_BUCKET, hidden]` —
///   the kernel writes per-row partials; the caller reduces.
pub fn fused_rmsnorm_backward_kt(
    x: &KtTensor,
    weight: &KtTensor,
    grad_out: &KtTensor,
    eps: f32,
) -> Result<(KtTensor, KtTensor), RmsNormError> {
    let x_shape = x.shape().to_vec();
    let hidden = *x_shape.last().ok_or_else(|| {
        RmsNormError::Msg("kt-rmsnorm bwd: x must have rank >= 1".to_string())
    })?;
    if weight.shape() != [hidden] {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm bwd: weight {:?} != [{hidden}]",
            weight.shape()
        )));
    }
    if grad_out.shape() != x.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm bwd: grad_out {:?} != x {x_shape:?}",
            grad_out.shape()
        )));
    }
    let rows: usize = x_shape[..x_shape.len() - 1].iter().product();
    if rows == 0 {
        return Err(RmsNormError::Msg(
            "kt-rmsnorm bwd: rows=0 not supported".to_string(),
        ));
    }

    let (x_st, x_off) = cuda_storage_and_byte_offset(x, KtDType::BF16, "x")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::BF16, "weight")?;
    let (g_st, g_off) = cuda_storage_and_byte_offset(grad_out, KtDType::BF16, "grad_out")?;

    let grad_x = alloc_cuda_tensor(x_st, KtDType::BF16, x_shape.clone())?;
    // grad_w_partial: the kernel writes one row of partials per warp
    // of rows; the caller sums. For the kt-API we mirror the candle
    // shape: [rows, hidden] (the full per-row form). The kernel
    // expects a contiguous F32 buffer of `rows * hidden`.
    let grad_w_partial = alloc_cuda_tensor(x_st, KtDType::F32, vec![rows, hidden])?;

    let gx_cuda = grad_x
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");
    let gw_cuda = grad_w_partial
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = x_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let x_slice = x_st.slice().slice(x_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let g_slice = g_st.slice().slice(g_off..);
    let gx_slice = gx_cuda.slice().slice(0..);
    let gw_slice = gw_cuda.slice().slice(0..);

    let status = unsafe {
        let (x_ptr, _g1) = x_slice.device_ptr(&stream);
        let (w_ptr, _g2) = w_slice.device_ptr(&stream);
        let (g_ptr, _g3) = g_slice.device_ptr(&stream);
        let (gx_ptr, _g4) = gx_slice.device_ptr(&stream);
        let (gw_ptr, _g5) = gw_slice.device_ptr(&stream);

        kiln_fused_rmsnorm_bwd(
            x_ptr as *const _,
            w_ptr as *const _,
            g_ptr as *const _,
            gx_ptr as *mut _,
            gw_ptr as *mut f32,
            rows as i32,
            hidden as i32,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-rmsnorm bwd: FFI returned {status}"
        )));
    }
    Ok((grad_x, grad_w_partial))
}

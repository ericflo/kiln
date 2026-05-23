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
use kiln_kt_bridge::BridgeError;
use kiln_tensor::{CudaStorage, DType as KtDType, Tensor as KtTensor};

use crate::{
    kiln_adamw_step_f32, kiln_fused_mlp_silu_mul_bf16, kiln_fused_rmsnorm,
    kiln_fused_rmsnorm_bwd, kiln_fused_rotary_qk, kiln_sgd_step_f32,
};

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

impl From<BridgeError> for RmsNormError {
    fn from(e: BridgeError) -> Self {
        RmsNormError::Msg(e.message)
    }
}

fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), RmsNormError> {
    Ok(kiln_kt_bridge::cuda_storage_and_byte_offset(t, expected, name)?)
}

fn alloc_cuda_tensor(
    source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, RmsNormError> {
    Ok(kiln_kt_bridge::alloc_cuda_tensor(source, dtype, shape)?)
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

/// `fused_rotary_qk` over `kiln_tensor::Tensor` operands.
///
/// In-place rotary application to Q and K projections. Inputs:
/// - `q`: BF16 `[batch, seq_len, q_heads, head_dim]`
/// - `k`: BF16 `[batch, seq_len, k_heads, head_dim]`
/// - `cos`, `sin`: F32 `[seq_len, rotary_dim]` precomputed tables
/// - `rotary_dim`: applied head dim slice; must be ≤ head_dim.
///
/// Returns `(q_out, k_out)` BF16 tensors of the same shapes as the
/// inputs.
#[allow(clippy::too_many_arguments)]
pub fn fused_rotary_qk_kt(
    q: &KtTensor,
    k: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    rotary_dim: usize,
) -> Result<(KtTensor, KtTensor), RmsNormError> {
    let q_shape = q.shape();
    if q_shape.len() != 4 {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: q must be [B, S, q_heads, head_dim], got {q_shape:?}"
        )));
    }
    let (batch, seq_len, q_heads, head_dim) = (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
    let k_shape = k.shape();
    if k_shape.len() != 4
        || (k_shape[0], k_shape[1], k_shape[3]) != (batch, seq_len, head_dim)
    {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: k {k_shape:?} != [{batch}, {seq_len}, k_heads, {head_dim}]"
        )));
    }
    let k_heads = k_shape[2];
    if rotary_dim > head_dim {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: rotary_dim {rotary_dim} > head_dim {head_dim}"
        )));
    }
    if cos.shape() != [seq_len, rotary_dim] {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: cos {:?} != [{seq_len}, {rotary_dim}]",
            cos.shape()
        )));
    }
    if sin.shape() != [seq_len, rotary_dim] {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: sin {:?} != [{seq_len}, {rotary_dim}]",
            sin.shape()
        )));
    }

    let (q_st, q_off) = cuda_storage_and_byte_offset(q, KtDType::BF16, "q")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k, KtDType::BF16, "k")?;
    let (cos_st, cos_off) = cuda_storage_and_byte_offset(cos, KtDType::F32, "cos")?;
    let (sin_st, sin_off) = cuda_storage_and_byte_offset(sin, KtDType::F32, "sin")?;

    let q_out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, seq_len, q_heads, head_dim])?;
    let k_out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![batch, seq_len, k_heads, head_dim])?;
    let qo_cuda = q_out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");
    let ko_cuda = k_out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = q_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let q_slice = q_st.slice().slice(q_off..);
    let k_slice = k_st.slice().slice(k_off..);
    let cos_slice = cos_st.slice().slice(cos_off..);
    let sin_slice = sin_st.slice().slice(sin_off..);
    let qo_slice = qo_cuda.slice().slice(0..);
    let ko_slice = ko_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (cos_ptr, _g3) = cos_slice.device_ptr(&stream);
        let (sin_ptr, _g4) = sin_slice.device_ptr(&stream);
        let (qo_ptr, _g5) = qo_slice.device_ptr(&stream);
        let (ko_ptr, _g6) = ko_slice.device_ptr(&stream);

        kiln_fused_rotary_qk(
            q_ptr as *const _,
            k_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            qo_ptr as *mut _,
            ko_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            q_heads as i32,
            k_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary: FFI returned {status}"
        )));
    }
    Ok((q_out, k_out))
}

/// `fused_mlp_silu_mul` over `kiln_tensor::Tensor` operands.
///
/// Element-wise: `out = silu(gate) * up`. Both inputs and output
/// are BF16 of equal element count. Used by the MLP gate||up||silu
/// fused path.
pub fn fused_mlp_silu_mul_kt(
    gate: &KtTensor,
    up: &KtTensor,
) -> Result<KtTensor, RmsNormError> {
    if gate.shape() != up.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-mlp-silu-mul: gate {:?} != up {:?}",
            gate.shape(),
            up.shape()
        )));
    }
    let elems = gate.element_count();
    let shape = gate.shape().to_vec();

    let (g_st, g_off) = cuda_storage_and_byte_offset(gate, KtDType::BF16, "gate")?;
    let (u_st, u_off) = cuda_storage_and_byte_offset(up, KtDType::BF16, "up")?;
    let out = alloc_cuda_tensor(g_st, KtDType::BF16, shape)?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = g_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let g_slice = g_st.slice().slice(g_off..);
    let u_slice = u_st.slice().slice(u_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (g_ptr, _g1) = g_slice.device_ptr(&stream);
        let (u_ptr, _g2) = u_slice.device_ptr(&stream);
        let (o_ptr, _g3) = o_slice.device_ptr(&stream);
        kiln_fused_mlp_silu_mul_bf16(
            g_ptr as *const _,
            u_ptr as *const _,
            o_ptr as *mut _,
            elems as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-mlp-silu-mul: FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `sgd_step_f32` over `kiln_tensor::Tensor` operands.
///
/// In-place SGD update: `param -= lr * grad`. F32 only. `param` is
/// mutated in place through the raw device pointer; the caller must
/// hold a unique reference (kt-Tensor borrow-check is at the
/// version-counter layer, anti-pattern 16).
pub fn sgd_step_f32_kt(
    param: &KtTensor,
    grad: &KtTensor,
    lr: f32,
) -> Result<(), RmsNormError> {
    if param.shape() != grad.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-sgd-step: param {:?} != grad {:?}",
            param.shape(),
            grad.shape()
        )));
    }
    let n = param.element_count() as i64;

    let (p_st, p_off) = cuda_storage_and_byte_offset(param, KtDType::F32, "param")?;
    let (g_st, g_off) = cuda_storage_and_byte_offset(grad, KtDType::F32, "grad")?;

    let stream = p_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let p_slice = p_st.slice().slice(p_off..);
    let g_slice = g_st.slice().slice(g_off..);

    let status = unsafe {
        let (p_ptr, _g1) = p_slice.device_ptr(&stream);
        let (g_ptr, _g2) = g_slice.device_ptr(&stream);
        kiln_sgd_step_f32(p_ptr as *mut f32, g_ptr as *const f32, lr, n, raw_stream)
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-sgd-step: FFI returned {status}"
        )));
    }
    Ok(())
}

/// `adamw_step_f32` over `kiln_tensor::Tensor` operands.
///
/// In-place AdamW update with running moments. F32 only. `param`,
/// `first_moment`, `second_moment` are all mutated in place. The
/// caller passes pre-computed bias-correction terms (matches the
/// candle path's contract).
#[allow(clippy::too_many_arguments)]
pub fn adamw_step_f32_kt(
    param: &KtTensor,
    grad: &KtTensor,
    first_moment: &KtTensor,
    second_moment: &KtTensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
) -> Result<(), RmsNormError> {
    if param.shape() != grad.shape()
        || param.shape() != first_moment.shape()
        || param.shape() != second_moment.shape()
    {
        return Err(RmsNormError::Msg(format!(
            "kt-adamw-step: shape mismatch — param {:?}, grad {:?}, m1 {:?}, m2 {:?}",
            param.shape(),
            grad.shape(),
            first_moment.shape(),
            second_moment.shape()
        )));
    }
    let n = param.element_count() as i64;

    let (p_st, p_off) = cuda_storage_and_byte_offset(param, KtDType::F32, "param")?;
    let (g_st, g_off) = cuda_storage_and_byte_offset(grad, KtDType::F32, "grad")?;
    let (m1_st, m1_off) =
        cuda_storage_and_byte_offset(first_moment, KtDType::F32, "first_moment")?;
    let (m2_st, m2_off) =
        cuda_storage_and_byte_offset(second_moment, KtDType::F32, "second_moment")?;

    let stream = p_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let p_slice = p_st.slice().slice(p_off..);
    let g_slice = g_st.slice().slice(g_off..);
    let m1_slice = m1_st.slice().slice(m1_off..);
    let m2_slice = m2_st.slice().slice(m2_off..);

    let status = unsafe {
        let (p_ptr, _g1) = p_slice.device_ptr(&stream);
        let (g_ptr, _g2) = g_slice.device_ptr(&stream);
        let (m1_ptr, _g3) = m1_slice.device_ptr(&stream);
        let (m2_ptr, _g4) = m2_slice.device_ptr(&stream);
        kiln_adamw_step_f32(
            p_ptr as *mut f32,
            g_ptr as *const f32,
            m1_ptr as *mut f32,
            m2_ptr as *mut f32,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
            n,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-adamw-step: FFI returned {status}"
        )));
    }
    Ok(())
}

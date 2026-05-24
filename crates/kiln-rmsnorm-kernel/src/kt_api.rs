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
    kiln_adamw_step_bf16, kiln_adamw_step_f32, kiln_attn_decode_qkv_split_qk_norm_rope_bf16,
    kiln_causal_depthwise_conv1d_bwd_input_f32, kiln_causal_depthwise_conv1d_bwd_state_f32,
    kiln_causal_depthwise_conv1d_bwd_weight_f32, kiln_causal_depthwise_conv1d_f32,
    kiln_causal_depthwise_conv1d_inplace_f32, kiln_f32_to_bf16, kiln_fused_l2_qk_norm,
    kiln_fused_l2_qk_norm_gqa, kiln_fused_mlp_silu_mul_bf16, kiln_fused_mlp_silu_mul_packed_bf16,
    kiln_fused_rmsnorm, kiln_fused_rmsnorm_bwd, kiln_fused_rotary_one, kiln_fused_rotary_one_bwd,
    kiln_fused_rotary_qk, kiln_fused_sigmoid_mul_bf16, kiln_lora_add_inplace_f32,
    kiln_lora_decode_add_bf16, kiln_lora_decode_hidden_bf16, kiln_sgd_step_bf16, kiln_sgd_step_f32,
    kiln_silu_inplace_save_sigmoid_f32,
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

/// `lora_decode_hidden_bf16` over `kiln_tensor::Tensor` operands.
///
/// Computes the LoRA-A projection at decode time: `hidden = x @ A`,
/// where:
/// - `x`: BF16 `[batch, in_dim]`
/// - `a`: BF16 `[in_dim, rank]` (the LoRA-A matrix)
///
/// Returns F32 `[batch, rank]` (the LoRA hidden state, in F32 for
/// downstream numerical accuracy). Used by the multi-LoRA decode
/// path (line 307 of #1082).
pub fn lora_decode_hidden_kt(
    x: &KtTensor,
    a: &KtTensor,
) -> Result<KtTensor, RmsNormError> {
    let x_shape = x.shape();
    if x_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-hidden: x must be [batch, in_dim], got {x_shape:?}"
        )));
    }
    let (batch, in_dim) = (x_shape[0], x_shape[1]);
    let a_shape = a.shape();
    if a_shape.len() != 2 || a_shape[0] != in_dim {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-hidden: a {a_shape:?} != [{in_dim}, rank]"
        )));
    }
    let rank = a_shape[1];

    let (x_st, x_off) = cuda_storage_and_byte_offset(x, KtDType::BF16, "x")?;
    let (a_st, a_off) = cuda_storage_and_byte_offset(a, KtDType::BF16, "a")?;
    let hidden = alloc_cuda_tensor(x_st, KtDType::F32, vec![batch, rank])?;
    let h_cuda = hidden
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = x_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let x_slice = x_st.slice().slice(x_off..);
    let a_slice = a_st.slice().slice(a_off..);
    let h_slice = h_cuda.slice().slice(0..);

    let status = unsafe {
        let (x_ptr, _g1) = x_slice.device_ptr(&stream);
        let (a_ptr, _g2) = a_slice.device_ptr(&stream);
        let (h_ptr, _g3) = h_slice.device_ptr(&stream);
        kiln_lora_decode_hidden_bf16(
            x_ptr as *const _,
            a_ptr as *const _,
            h_ptr as *mut f32,
            batch as i32,
            in_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-hidden: FFI returned {status}"
        )));
    }
    Ok(hidden)
}

/// `lora_decode_add_bf16` over `kiln_tensor::Tensor` operands.
///
/// Adds the LoRA-B contribution to the base projection:
/// `out = base + scale * (hidden @ B)`, where:
/// - `base`: BF16 `[batch, out_dim]` (the base linear projection)
/// - `hidden`: F32 `[batch, rank]` (output of [`lora_decode_hidden_kt`])
/// - `b`: BF16 `[rank, out_dim]` (the LoRA-B matrix)
/// - `scale`: f32 LoRA alpha scale
///
/// Returns BF16 `[batch, out_dim]`.
pub fn lora_decode_add_kt(
    base: &KtTensor,
    hidden: &KtTensor,
    b: &KtTensor,
    scale: f32,
) -> Result<KtTensor, RmsNormError> {
    let base_shape = base.shape();
    if base_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: base must be [batch, out_dim], got {base_shape:?}"
        )));
    }
    let (batch, out_dim) = (base_shape[0], base_shape[1]);
    let h_shape = hidden.shape();
    if h_shape.len() != 2 || h_shape[0] != batch {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: hidden {h_shape:?} != [{batch}, rank]"
        )));
    }
    let rank = h_shape[1];
    let b_shape = b.shape();
    if b_shape != [rank, out_dim] {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: b {b_shape:?} != [{rank}, {out_dim}]"
        )));
    }

    let (base_st, base_off) = cuda_storage_and_byte_offset(base, KtDType::BF16, "base")?;
    let (h_st, h_off) = cuda_storage_and_byte_offset(hidden, KtDType::F32, "hidden")?;
    let (b_st, b_off) = cuda_storage_and_byte_offset(b, KtDType::BF16, "b")?;
    let out = alloc_cuda_tensor(base_st, KtDType::BF16, vec![batch, out_dim])?;
    let out_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = base_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let base_slice = base_st.slice().slice(base_off..);
    let h_slice = h_st.slice().slice(h_off..);
    let b_slice = b_st.slice().slice(b_off..);
    let o_slice = out_cuda.slice().slice(0..);

    let status = unsafe {
        let (base_ptr, _g1) = base_slice.device_ptr(&stream);
        let (h_ptr, _g2) = h_slice.device_ptr(&stream);
        let (b_ptr, _g3) = b_slice.device_ptr(&stream);
        let (o_ptr, _g4) = o_slice.device_ptr(&stream);
        kiln_lora_decode_add_bf16(
            base_ptr as *const _,
            h_ptr as *const f32,
            b_ptr as *const _,
            o_ptr as *mut _,
            scale,
            batch as i32,
            out_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `fused_l2_qk_norm` over `kiln_tensor::Tensor` operands.
///
/// L2-normalize each row of `q_in` and `k_in` (shape `[rows, hidden]`,
/// BF16) and scale `q` by `q_scale`. Used by the QK-norm pass in
/// attention pre-projection. Returns `(q_out, k_out)`.
pub fn fused_l2_qk_norm_kt(
    q_in: &KtTensor,
    k_in: &KtTensor,
    q_scale: f32,
    eps: f32,
) -> Result<(KtTensor, KtTensor), RmsNormError> {
    if q_in.shape() != k_in.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm: q {:?} != k {:?}",
            q_in.shape(),
            k_in.shape()
        )));
    }
    let q_shape = q_in.shape().to_vec();
    if q_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm: q must be [rows, hidden], got {q_shape:?}"
        )));
    }
    let (rows, hidden) = (q_shape[0], q_shape[1]);

    let (q_st, q_off) = cuda_storage_and_byte_offset(q_in, KtDType::BF16, "q_in")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k_in, KtDType::BF16, "k_in")?;
    let q_out = alloc_cuda_tensor(q_st, KtDType::BF16, q_shape.clone())?;
    let k_out = alloc_cuda_tensor(q_st, KtDType::BF16, q_shape)?;
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
    let qo_slice = qo_cuda.slice().slice(0..);
    let ko_slice = ko_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (qo_ptr, _g3) = qo_slice.device_ptr(&stream);
        let (ko_ptr, _g4) = ko_slice.device_ptr(&stream);
        kiln_fused_l2_qk_norm(
            q_ptr as *const _,
            k_ptr as *const _,
            qo_ptr as *mut _,
            ko_ptr as *mut _,
            rows as i32,
            hidden as i32,
            q_scale,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm: FFI returned {status}"
        )));
    }
    Ok((q_out, k_out))
}

/// GQA variant of `fused_l2_qk_norm`. K has `nk` distinct heads per
/// `ratio` (group size) Q heads. Shapes:
/// - `q_in`, `q_out`: BF16 `[rows, hidden_q]` where `hidden_q = nk*ratio*head_dim`
/// - `k_in`, `k_out`: BF16 `[rows, hidden_k]` where `hidden_k = nk*head_dim`
pub fn fused_l2_qk_norm_gqa_kt(
    q_in: &KtTensor,
    k_in: &KtTensor,
    nk: usize,
    ratio: usize,
    head_dim: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(KtTensor, KtTensor), RmsNormError> {
    let q_shape = q_in.shape();
    if q_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm-gqa: q must be [rows, hidden_q], got {q_shape:?}"
        )));
    }
    let rows = q_shape[0];
    let hidden_q = q_shape[1];
    if hidden_q != nk * ratio * head_dim {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm-gqa: q hidden {hidden_q} != nk({nk}) * ratio({ratio}) * head_dim({head_dim})"
        )));
    }
    let k_shape = k_in.shape();
    if k_shape != [rows, nk * head_dim] {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm-gqa: k {k_shape:?} != [{rows}, {}]",
            nk * head_dim
        )));
    }

    let (q_st, q_off) = cuda_storage_and_byte_offset(q_in, KtDType::BF16, "q_in")?;
    let (k_st, k_off) = cuda_storage_and_byte_offset(k_in, KtDType::BF16, "k_in")?;
    let q_out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![rows, hidden_q])?;
    let k_out = alloc_cuda_tensor(q_st, KtDType::BF16, vec![rows, nk * head_dim])?;
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
    let qo_slice = qo_cuda.slice().slice(0..);
    let ko_slice = ko_cuda.slice().slice(0..);

    let status = unsafe {
        let (q_ptr, _g1) = q_slice.device_ptr(&stream);
        let (k_ptr, _g2) = k_slice.device_ptr(&stream);
        let (qo_ptr, _g3) = qo_slice.device_ptr(&stream);
        let (ko_ptr, _g4) = ko_slice.device_ptr(&stream);
        kiln_fused_l2_qk_norm_gqa(
            q_ptr as *const _,
            k_ptr as *const _,
            qo_ptr as *mut _,
            ko_ptr as *mut _,
            rows as i32,
            nk as i32,
            ratio as i32,
            head_dim as i32,
            q_scale,
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-l2-qk-norm-gqa: FFI returned {status}"
        )));
    }
    Ok((q_out, k_out))
}

/// `fused_rotary_one` over `kiln_tensor::Tensor` operands.
///
/// Single-tensor rotary application (used for Q or K, but not both
/// in one launch — see [`fused_rotary_qk_kt`] for the fused pair).
/// Shape: `[batch, seq_len, heads, head_dim]` BF16; rotary applied
/// to the first `rotary_dim` of head_dim. Returns the rotated tensor.
pub fn fused_rotary_one_kt(
    x: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    rotary_dim: usize,
) -> Result<KtTensor, RmsNormError> {
    let x_shape = x.shape();
    if x_shape.len() != 4 {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one: x must be [B, S, H, D], got {x_shape:?}"
        )));
    }
    let (batch, seq_len, heads, head_dim) = (x_shape[0], x_shape[1], x_shape[2], x_shape[3]);
    if rotary_dim > head_dim {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one: rotary_dim {rotary_dim} > head_dim {head_dim}"
        )));
    }
    if cos.shape() != [seq_len, rotary_dim] || sin.shape() != [seq_len, rotary_dim] {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one: cos/sin must be [{seq_len}, {rotary_dim}]"
        )));
    }

    let (x_st, x_off) = cuda_storage_and_byte_offset(x, KtDType::BF16, "x")?;
    let (cos_st, cos_off) = cuda_storage_and_byte_offset(cos, KtDType::F32, "cos")?;
    let (sin_st, sin_off) = cuda_storage_and_byte_offset(sin, KtDType::F32, "sin")?;

    let out = alloc_cuda_tensor(x_st, KtDType::BF16, vec![batch, seq_len, heads, head_dim])?;
    let o_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = x_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let x_slice = x_st.slice().slice(x_off..);
    let cos_slice = cos_st.slice().slice(cos_off..);
    let sin_slice = sin_st.slice().slice(sin_off..);
    let o_slice = o_cuda.slice().slice(0..);

    let status = unsafe {
        let (x_ptr, _g1) = x_slice.device_ptr(&stream);
        let (cos_ptr, _g2) = cos_slice.device_ptr(&stream);
        let (sin_ptr, _g3) = sin_slice.device_ptr(&stream);
        let (o_ptr, _g4) = o_slice.device_ptr(&stream);
        kiln_fused_rotary_one(
            x_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            o_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one: FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `fused_sigmoid_mul_bf16` over `kiln_tensor::Tensor` operands.
///
/// Element-wise: `out = sigmoid(gate) * x`. Both BF16, same shape.
/// Used by gated activation paths. Like `fused_mlp_silu_mul` but
/// with sigmoid instead of silu.
pub fn fused_sigmoid_mul_kt(
    x: &KtTensor,
    gate: &KtTensor,
) -> Result<KtTensor, RmsNormError> {
    if x.shape() != gate.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-sigmoid-mul: x {:?} != gate {:?}",
            x.shape(),
            gate.shape()
        )));
    }
    let elems = x.element_count();
    let shape = x.shape().to_vec();

    // Owner-agnostic input device pointers: works for both Owned and
    // Borrowed kt storage (Phase 7 v2 — accepts kt-Tensors built via
    // `kt_tensor_from_candle_cuda_borrow`). This is the migration
    // template for the rest of the kt-API surface.
    let x_ptr = kiln_kt_bridge::cuda_input_device_ptr(x, KtDType::BF16, "x")?;
    let g_ptr = kiln_kt_bridge::cuda_input_device_ptr(gate, KtDType::BF16, "gate")?;

    // Output is always Owned (alloc_cuda_tensor produces owned storage),
    // so we can reach for the raw pointer the same way.
    let (x_st, _) = cuda_storage_and_byte_offset(x, KtDType::BF16, "x")?;
    let out = alloc_cuda_tensor(x_st, KtDType::BF16, shape)?;
    let o_ptr = kiln_kt_bridge::cuda_output_device_ptr(&out);

    let stream = x_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = unsafe {
        kiln_fused_sigmoid_mul_bf16(
            x_ptr as *const _,
            g_ptr as *const _,
            o_ptr as *mut _,
            elems as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-sigmoid-mul: FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `attn_decode_qkv_split_qk_norm_rope_bf16` over `kiln_tensor::Tensor`
/// operands.
///
/// The decode-time mega-fused attention pre-projection kernel.
/// Splits the QKV-packed Q/gate inputs, applies QK-norm (with eps),
/// then applies RoPE — all in one launch. Optionally writes the
/// gate output if `has_gate` is true.
///
/// Shapes (production decode):
/// - `q_raw`: BF16 `[batch, 1, q_heads * head_dim + (q_heads * head_dim if has_gate)]`
/// - `k_raw`: BF16 `[batch, 1, k_heads * head_dim]`
/// - `q_weight`, `k_weight`: BF16 `[head_dim]`
/// - `cos`, `sin`: F32 `[1, rotary_dim / 2]`
///
/// Returns `(q_out, k_out, gate_out)` where:
/// - `q_out`: BF16 `[batch, 1, q_heads, head_dim]`
/// - `k_out`: BF16 `[batch, 1, k_heads, head_dim]`
/// - `gate_out`: Some(BF16 `[batch, 1, q_heads * head_dim]`) if has_gate, else None
#[allow(clippy::too_many_arguments)]
pub fn attn_decode_qkv_split_qk_norm_rope_kt(
    q_raw: &KtTensor,
    k_raw: &KtTensor,
    q_weight: &KtTensor,
    k_weight: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    has_gate: bool,
    eps: f32,
) -> Result<(KtTensor, KtTensor, Option<KtTensor>), RmsNormError> {
    let qr_shape = q_raw.shape();
    if qr_shape.len() != 3 || qr_shape[1] != 1 {
        return Err(RmsNormError::Msg(format!(
            "kt-attn-decode-prep: q_raw must be [B, 1, hidden_q], got {qr_shape:?}"
        )));
    }
    let batch = qr_shape[0];

    let (qr_st, qr_off) = cuda_storage_and_byte_offset(q_raw, KtDType::BF16, "q_raw")?;
    let (kr_st, kr_off) = cuda_storage_and_byte_offset(k_raw, KtDType::BF16, "k_raw")?;
    let (qw_st, qw_off) = cuda_storage_and_byte_offset(q_weight, KtDType::BF16, "q_weight")?;
    let (kw_st, kw_off) = cuda_storage_and_byte_offset(k_weight, KtDType::BF16, "k_weight")?;
    let (cos_st, cos_off) = cuda_storage_and_byte_offset(cos, KtDType::F32, "cos")?;
    let (sin_st, sin_off) = cuda_storage_and_byte_offset(sin, KtDType::F32, "sin")?;

    let q_out = alloc_cuda_tensor(qr_st, KtDType::BF16, vec![batch, 1, q_heads, head_dim])?;
    let k_out = alloc_cuda_tensor(qr_st, KtDType::BF16, vec![batch, 1, k_heads, head_dim])?;
    let gate_out = if has_gate {
        Some(alloc_cuda_tensor(
            qr_st,
            KtDType::BF16,
            vec![batch, 1, q_heads * head_dim],
        )?)
    } else {
        None
    };
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

    let stream = qr_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let qr_slice = qr_st.slice().slice(qr_off..);
    let kr_slice = kr_st.slice().slice(kr_off..);
    let qw_slice = qw_st.slice().slice(qw_off..);
    let kw_slice = kw_st.slice().slice(kw_off..);
    let cos_slice = cos_st.slice().slice(cos_off..);
    let sin_slice = sin_st.slice().slice(sin_off..);
    let qo_slice = qo_cuda.slice().slice(0..);
    let ko_slice = ko_cuda.slice().slice(0..);

    let status = unsafe {
        let (qr_ptr, _g1) = qr_slice.device_ptr(&stream);
        let (kr_ptr, _g2) = kr_slice.device_ptr(&stream);
        let (qw_ptr, _g3) = qw_slice.device_ptr(&stream);
        let (kw_ptr, _g4) = kw_slice.device_ptr(&stream);
        let (cos_ptr, _g5) = cos_slice.device_ptr(&stream);
        let (sin_ptr, _g6) = sin_slice.device_ptr(&stream);
        let (qo_ptr, _g7) = qo_slice.device_ptr(&stream);
        let (ko_ptr, _g8) = ko_slice.device_ptr(&stream);

        let go_ptr = if let Some(go) = gate_out.as_ref() {
            let go_cuda = go
                .storage()
                .as_any()
                .downcast_ref::<CudaStorage>()
                .expect("alloc CUDA");
            let go_slice = go_cuda.slice().slice(0..);
            let (p, _g9) = go_slice.device_ptr(&stream);
            p as *mut _
        } else {
            core::ptr::null_mut()
        };

        kiln_attn_decode_qkv_split_qk_norm_rope_bf16(
            qr_ptr as *const _,
            kr_ptr as *const _,
            qw_ptr as *const _,
            kw_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            qo_ptr as *mut _,
            ko_ptr as *mut _,
            go_ptr,
            batch as i32,
            q_heads as i32,
            k_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            if has_gate { 1 } else { 0 },
            eps,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-attn-decode-prep: FFI returned {status}"
        )));
    }
    Ok((q_out, k_out, gate_out))
}

// ============================================================================
// Causal depthwise conv1d family (F32; used by training paths)
// ============================================================================

/// `causal_depthwise_conv1d_f32` over kt operands.
///
/// `input`: F32 `[rows, channels]` (one row per time step).
/// `weight`: F32 `[channels, kernel]`. `state`: F32 `[channels, kernel-1]`.
/// Returns F32 `[rows, channels]`.
pub fn causal_depthwise_conv1d_kt(
    input: &KtTensor,
    weight: &KtTensor,
    state: &KtTensor,
    kernel: usize,
) -> Result<KtTensor, RmsNormError> {
    let input_shape = input.shape();
    if input_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d: input must be [rows, channels], got {input_shape:?}"
        )));
    }
    let (rows, channels) = (input_shape[0], input_shape[1]);
    if weight.shape() != [channels, kernel] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d: weight {:?} != [{channels}, {kernel}]",
            weight.shape()
        )));
    }
    if state.shape() != [channels, kernel - 1] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d: state {:?} != [{channels}, {}]",
            state.shape(),
            kernel - 1
        )));
    }
    let (i_st, i_off) = cuda_storage_and_byte_offset(input, KtDType::F32, "input")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::F32, "state")?;
    let out = alloc_cuda_tensor(i_st, KtDType::F32, vec![rows, channels])?;
    let o_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = i_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let i_slice = i_st.slice().slice(i_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let o_slice = o_cuda.slice().slice(0..);
    let status = unsafe {
        let (i_ptr, _g1) = i_slice.device_ptr(&stream);
        let (w_ptr, _g2) = w_slice.device_ptr(&stream);
        let (s_ptr, _g3) = s_slice.device_ptr(&stream);
        let (o_ptr, _g4) = o_slice.device_ptr(&stream);
        kiln_causal_depthwise_conv1d_f32(
            i_ptr as *const f32,
            w_ptr as *const f32,
            s_ptr as *const f32,
            o_ptr as *mut f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d: FFI returned {status}"
        )));
    }
    Ok(out)
}

/// In-place variant of [`causal_depthwise_conv1d_kt`]. `input_out`
/// is mutated in place; no allocation, no copy. Returns `()`.
pub fn causal_depthwise_conv1d_inplace_kt(
    input_out: &KtTensor,
    weight: &KtTensor,
    state: &KtTensor,
    kernel: usize,
) -> Result<(), RmsNormError> {
    let shape = input_out.shape();
    if shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-inplace: input_out must be [rows, channels], got {shape:?}"
        )));
    }
    let (rows, channels) = (shape[0], shape[1]);
    if weight.shape() != [channels, kernel] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-inplace: weight {:?} != [{channels}, {kernel}]",
            weight.shape()
        )));
    }
    if state.shape() != [channels, kernel - 1] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-inplace: state {:?} != [{channels}, {}]",
            state.shape(),
            kernel - 1
        )));
    }
    let (i_st, i_off) = cuda_storage_and_byte_offset(input_out, KtDType::F32, "input_out")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::F32, "state")?;
    let stream = i_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let i_slice = i_st.slice().slice(i_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let status = unsafe {
        let (i_ptr, _g1) = i_slice.device_ptr(&stream);
        let (w_ptr, _g2) = w_slice.device_ptr(&stream);
        let (s_ptr, _g3) = s_slice.device_ptr(&stream);
        kiln_causal_depthwise_conv1d_inplace_f32(
            i_ptr as *mut f32,
            w_ptr as *const f32,
            s_ptr as *const f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-inplace: FFI returned {status}"
        )));
    }
    Ok(())
}

/// Backward through `causal_depthwise_conv1d` w.r.t. the input.
pub fn causal_depthwise_conv1d_bwd_input_kt(
    grad_out: &KtTensor,
    weight: &KtTensor,
    kernel: usize,
) -> Result<KtTensor, RmsNormError> {
    let go_shape = grad_out.shape();
    if go_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-input: grad_out must be [rows, channels], got {go_shape:?}"
        )));
    }
    let (rows, channels) = (go_shape[0], go_shape[1]);
    if weight.shape() != [channels, kernel] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-input: weight {:?} != [{channels}, {kernel}]",
            weight.shape()
        )));
    }
    let (g_st, g_off) = cuda_storage_and_byte_offset(grad_out, KtDType::F32, "grad_out")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;
    let gi = alloc_cuda_tensor(g_st, KtDType::F32, vec![rows, channels])?;
    let gi_cuda = gi
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");
    let stream = g_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let g_slice = g_st.slice().slice(g_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let gi_slice = gi_cuda.slice().slice(0..);
    let status = unsafe {
        let (g_ptr, _g1) = g_slice.device_ptr(&stream);
        let (w_ptr, _g2) = w_slice.device_ptr(&stream);
        let (gi_ptr, _g3) = gi_slice.device_ptr(&stream);
        kiln_causal_depthwise_conv1d_bwd_input_f32(
            g_ptr as *const f32,
            w_ptr as *const f32,
            gi_ptr as *mut f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-input: FFI returned {status}"
        )));
    }
    Ok(gi)
}

/// Backward w.r.t. the weight. Output shape: `[channels, kernel]`.
pub fn causal_depthwise_conv1d_bwd_weight_kt(
    grad_out: &KtTensor,
    input: &KtTensor,
    state: &KtTensor,
    kernel: usize,
) -> Result<KtTensor, RmsNormError> {
    let go_shape = grad_out.shape();
    if go_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-weight: grad_out must be [rows, channels], got {go_shape:?}"
        )));
    }
    let (rows, channels) = (go_shape[0], go_shape[1]);
    if input.shape() != [rows, channels] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-weight: input {:?} != [{rows}, {channels}]",
            input.shape()
        )));
    }
    if state.shape() != [channels, kernel - 1] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-weight: state {:?} != [{channels}, {}]",
            state.shape(),
            kernel - 1
        )));
    }
    let (g_st, g_off) = cuda_storage_and_byte_offset(grad_out, KtDType::F32, "grad_out")?;
    let (i_st, i_off) = cuda_storage_and_byte_offset(input, KtDType::F32, "input")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(state, KtDType::F32, "state")?;
    let gw = alloc_cuda_tensor(g_st, KtDType::F32, vec![channels, kernel])?;
    let gw_cuda = gw
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");
    let stream = g_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let g_slice = g_st.slice().slice(g_off..);
    let i_slice = i_st.slice().slice(i_off..);
    let s_slice = s_st.slice().slice(s_off..);
    let gw_slice = gw_cuda.slice().slice(0..);
    let status = unsafe {
        let (g_ptr, _g1) = g_slice.device_ptr(&stream);
        let (i_ptr, _g2) = i_slice.device_ptr(&stream);
        let (s_ptr, _g3) = s_slice.device_ptr(&stream);
        let (gw_ptr, _g4) = gw_slice.device_ptr(&stream);
        kiln_causal_depthwise_conv1d_bwd_weight_f32(
            g_ptr as *const f32,
            i_ptr as *const f32,
            s_ptr as *const f32,
            gw_ptr as *mut f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-weight: FFI returned {status}"
        )));
    }
    Ok(gw)
}

/// Backward w.r.t. the conv state. Output shape: `[channels, kernel-1]`.
pub fn causal_depthwise_conv1d_bwd_state_kt(
    grad_out: &KtTensor,
    weight: &KtTensor,
    kernel: usize,
) -> Result<KtTensor, RmsNormError> {
    let go_shape = grad_out.shape();
    if go_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-state: grad_out must be [rows, channels], got {go_shape:?}"
        )));
    }
    let (rows, channels) = (go_shape[0], go_shape[1]);
    if weight.shape() != [channels, kernel] {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-state: weight {:?} != [{channels}, {kernel}]",
            weight.shape()
        )));
    }
    let (g_st, g_off) = cuda_storage_and_byte_offset(grad_out, KtDType::F32, "grad_out")?;
    let (w_st, w_off) = cuda_storage_and_byte_offset(weight, KtDType::F32, "weight")?;
    let gs = alloc_cuda_tensor(g_st, KtDType::F32, vec![channels, kernel - 1])?;
    let gs_cuda = gs
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");
    let stream = g_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let g_slice = g_st.slice().slice(g_off..);
    let w_slice = w_st.slice().slice(w_off..);
    let gs_slice = gs_cuda.slice().slice(0..);
    let status = unsafe {
        let (g_ptr, _g1) = g_slice.device_ptr(&stream);
        let (w_ptr, _g2) = w_slice.device_ptr(&stream);
        let (gs_ptr, _g3) = gs_slice.device_ptr(&stream);
        kiln_causal_depthwise_conv1d_bwd_state_f32(
            g_ptr as *const f32,
            w_ptr as *const f32,
            gs_ptr as *mut f32,
            rows as i32,
            channels as i32,
            kernel as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-depth-conv1d-bwd-state: FFI returned {status}"
        )));
    }
    Ok(gs)
}

/// Suppresses unused-import warnings for the bwd-state extern import
/// in builds that don't reference it directly through the kt-API.
#[allow(dead_code)]
fn _unused_imports_keep() {
    let _ = kiln_causal_depthwise_conv1d_bwd_input_f32;
    let _ = kiln_causal_depthwise_conv1d_bwd_weight_f32;
    let _ = kiln_causal_depthwise_conv1d_bwd_state_f32;
}

/// `kiln_f32_to_bf16` over kt operands.
///
/// Casts an F32 tensor of arbitrary shape to BF16 element-wise.
/// Returns a freshly allocated BF16 tensor with the same shape.
pub fn f32_to_bf16_kt(src: &KtTensor) -> Result<KtTensor, RmsNormError> {
    let shape = src.shape().to_vec();
    let n = src.element_count();
    let (s_st, s_off) = cuda_storage_and_byte_offset(src, KtDType::F32, "src")?;
    let out = alloc_cuda_tensor(s_st, KtDType::BF16, shape)?;
    let o_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = s_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let s_slice = s_st.slice().slice(s_off..);
    let o_slice = o_cuda.slice().slice(0..);

    let status = unsafe {
        let (s_ptr, _g1) = s_slice.device_ptr(&stream);
        let (o_ptr, _g2) = o_slice.device_ptr(&stream);
        kiln_f32_to_bf16(s_ptr as *const f32, o_ptr as *mut _, n as i32, raw_stream)
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-f32-to-bf16: FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `sgd_step_bf16` over `kiln_tensor::Tensor` operands.
///
/// BF16-master SGD step. `param` and `grad` both BF16; updated in
/// place. See [`sgd_step_f32_kt`] for the F32-master variant.
pub fn sgd_step_bf16_kt(
    param: &KtTensor,
    grad: &KtTensor,
    lr: f32,
) -> Result<(), RmsNormError> {
    if param.shape() != grad.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-sgd-step-bf16: param {:?} != grad {:?}",
            param.shape(),
            grad.shape()
        )));
    }
    let n = param.element_count() as i64;
    let (p_st, p_off) = cuda_storage_and_byte_offset(param, KtDType::BF16, "param")?;
    let (g_st, g_off) = cuda_storage_and_byte_offset(grad, KtDType::BF16, "grad")?;

    let stream = p_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let p_slice = p_st.slice().slice(p_off..);
    let g_slice = g_st.slice().slice(g_off..);

    let status = unsafe {
        let (p_ptr, _g1) = p_slice.device_ptr(&stream);
        let (g_ptr, _g2) = g_slice.device_ptr(&stream);
        kiln_sgd_step_bf16(p_ptr as *mut _, g_ptr as *const _, lr, n, raw_stream)
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-sgd-step-bf16: FFI returned {status}"
        )));
    }
    Ok(())
}

/// `adamw_step_bf16` over `kiln_tensor::Tensor` operands.
///
/// BF16-master AdamW step. All four buffers (param, first_moment,
/// second_moment, grad) are BF16. Mutates `param`, `first_moment`,
/// `second_moment` in place.
#[allow(clippy::too_many_arguments)]
pub fn adamw_step_bf16_kt(
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
            "kt-adamw-step-bf16: shape mismatch — param {:?}, grad {:?}, m1 {:?}, m2 {:?}",
            param.shape(),
            grad.shape(),
            first_moment.shape(),
            second_moment.shape()
        )));
    }
    let n = param.element_count() as i64;

    let (p_st, p_off) = cuda_storage_and_byte_offset(param, KtDType::BF16, "param")?;
    let (g_st, g_off) = cuda_storage_and_byte_offset(grad, KtDType::BF16, "grad")?;
    let (m1_st, m1_off) =
        cuda_storage_and_byte_offset(first_moment, KtDType::BF16, "first_moment")?;
    let (m2_st, m2_off) =
        cuda_storage_and_byte_offset(second_moment, KtDType::BF16, "second_moment")?;

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
        kiln_adamw_step_bf16(
            p_ptr as *mut _,
            g_ptr as *const _,
            m1_ptr as *mut _,
            m2_ptr as *mut _,
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
            "kt-adamw-step-bf16: FFI returned {status}"
        )));
    }
    Ok(())
}

/// `fused_rotary_one_bwd` over `kiln_tensor::Tensor` operands.
///
/// Backward of [`fused_rotary_one_kt`]. Takes `grad_y` BF16
/// `[B, S, H, D]`, the original `cos`/`sin` F32 tables `[S, R]`, and
/// returns `grad_x` BF16 `[B, S, H, D]` (the gradient w.r.t. the input).
pub fn fused_rotary_one_bwd_kt(
    grad_y: &KtTensor,
    cos: &KtTensor,
    sin: &KtTensor,
    rotary_dim: usize,
) -> Result<KtTensor, RmsNormError> {
    let y_shape = grad_y.shape();
    if y_shape.len() != 4 {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one-bwd: grad_y must be [B, S, H, D], got {y_shape:?}"
        )));
    }
    let (batch, seq_len, heads, head_dim) = (y_shape[0], y_shape[1], y_shape[2], y_shape[3]);
    if rotary_dim > head_dim {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one-bwd: rotary_dim {rotary_dim} > head_dim {head_dim}"
        )));
    }
    if cos.shape() != [seq_len, rotary_dim] || sin.shape() != [seq_len, rotary_dim] {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one-bwd: cos/sin must be [{seq_len}, {rotary_dim}]"
        )));
    }

    let (y_st, y_off) = cuda_storage_and_byte_offset(grad_y, KtDType::BF16, "grad_y")?;
    let (cos_st, cos_off) = cuda_storage_and_byte_offset(cos, KtDType::F32, "cos")?;
    let (sin_st, sin_off) = cuda_storage_and_byte_offset(sin, KtDType::F32, "sin")?;

    let out = alloc_cuda_tensor(y_st, KtDType::BF16, vec![batch, seq_len, heads, head_dim])?;
    let o_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = y_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let y_slice = y_st.slice().slice(y_off..);
    let cos_slice = cos_st.slice().slice(cos_off..);
    let sin_slice = sin_st.slice().slice(sin_off..);
    let o_slice = o_cuda.slice().slice(0..);

    let status = unsafe {
        let (y_ptr, _g1) = y_slice.device_ptr(&stream);
        let (cos_ptr, _g2) = cos_slice.device_ptr(&stream);
        let (sin_ptr, _g3) = sin_slice.device_ptr(&stream);
        let (o_ptr, _g4) = o_slice.device_ptr(&stream);
        kiln_fused_rotary_one_bwd(
            y_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            o_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-rotary-one-bwd: FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `fused_mlp_silu_mul_packed_bf16` over kt operands.
///
/// Packed MLP variant: input is `[.., 2*cols]` where the first half is
/// the gate and the second half is the up projection (already
/// matmul-fused). Computes `silu(gate) * up` and returns `[.., cols]`.
pub fn fused_mlp_silu_mul_packed_kt(
    gate_up_packed: &KtTensor,
    cols: usize,
) -> Result<KtTensor, RmsNormError> {
    let dims = gate_up_packed.shape().to_vec();
    if dims.is_empty() || dims[dims.len() - 1] != 2 * cols {
        return Err(RmsNormError::Msg(format!(
            "kt-mlp-packed: gate_up last dim {:?} != 2*cols (cols={cols})",
            dims.last(),
        )));
    }
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_dims.push(cols);

    let (gu_st, gu_off) =
        cuda_storage_and_byte_offset(gate_up_packed, KtDType::BF16, "gate_up_packed")?;
    let out = alloc_cuda_tensor(gu_st, KtDType::BF16, out_dims)?;
    if rows == 0 {
        return Ok(out);
    }
    let o_cuda = out
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("alloc CUDA");

    let stream = gu_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let gu_slice = gu_st.slice().slice(gu_off..);
    let o_slice = o_cuda.slice().slice(0..);

    let status = unsafe {
        let (gu_ptr, _g1) = gu_slice.device_ptr(&stream);
        let (o_ptr, _g2) = o_slice.device_ptr(&stream);
        kiln_fused_mlp_silu_mul_packed_bf16(
            gu_ptr as *const _,
            o_ptr as *mut _,
            rows as i64,
            cols as i64,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-mlp-packed: FFI returned {status}"
        )));
    }
    Ok(out)
}

/// `lora_add_inplace_f32` over kt operands.
///
/// In-place LoRA-B fused add. Computes
/// `base += scale * (hidden @ B)` in F32, where:
/// - `base`:   F32 `[rows, out_dim]` — mutated in place
/// - `hidden`: F32 `[rows, rank]`
/// - `b`:      F32 `[rank, out_dim]`
///
/// `scale` is the LoRA alpha/rank scaling factor.
#[allow(clippy::too_many_arguments)]
pub fn lora_add_inplace_f32_kt(
    base: &KtTensor,
    hidden: &KtTensor,
    b: &KtTensor,
    scale: f32,
) -> Result<(), RmsNormError> {
    let base_shape = base.shape();
    if base_shape.len() != 2 {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: base must be [rows, out_dim], got {base_shape:?}"
        )));
    }
    let (rows, out_dim) = (base_shape[0], base_shape[1]);
    let h_shape = hidden.shape();
    if h_shape.len() != 2 || h_shape[0] != rows {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: hidden {h_shape:?} != [{rows}, rank]"
        )));
    }
    let rank = h_shape[1];
    if b.shape() != [rank, out_dim] {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: b {:?} != [{rank}, {out_dim}]",
            b.shape()
        )));
    }

    let (base_st, base_off) = cuda_storage_and_byte_offset(base, KtDType::F32, "base")?;
    let (h_st, h_off) = cuda_storage_and_byte_offset(hidden, KtDType::F32, "hidden")?;
    let (b_st, b_off) = cuda_storage_and_byte_offset(b, KtDType::F32, "b")?;

    let stream = base_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let base_slice = base_st.slice().slice(base_off..);
    let h_slice = h_st.slice().slice(h_off..);
    let b_slice = b_st.slice().slice(b_off..);

    let status = unsafe {
        let (base_ptr, _g1) = base_slice.device_ptr(&stream);
        let (h_ptr, _g2) = h_slice.device_ptr(&stream);
        let (b_ptr, _g3) = b_slice.device_ptr(&stream);
        kiln_lora_add_inplace_f32(
            base_ptr as *mut f32,
            h_ptr as *const f32,
            b_ptr as *const f32,
            scale,
            rows as i32,
            out_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-lora-add: FFI returned {status}"
        )));
    }
    Ok(())
}

/// `silu_inplace_save_sigmoid_f32` over kt operands.
///
/// Apply SiLU in place to `input_out`, simultaneously writing
/// `sigmoid_out = sigmoid(input)` into a separate buffer (saved for
/// the backward pass). Both buffers are F32 with the same element
/// count.
pub fn silu_inplace_save_sigmoid_f32_kt(
    input_out: &KtTensor,
    sigmoid_out: &KtTensor,
) -> Result<(), RmsNormError> {
    if input_out.shape() != sigmoid_out.shape() {
        return Err(RmsNormError::Msg(format!(
            "kt-silu-save: input {:?} != sigmoid {:?}",
            input_out.shape(),
            sigmoid_out.shape(),
        )));
    }
    let elems = input_out.element_count() as i64;
    let (i_st, i_off) = cuda_storage_and_byte_offset(input_out, KtDType::F32, "input_out")?;
    let (s_st, s_off) = cuda_storage_and_byte_offset(sigmoid_out, KtDType::F32, "sigmoid_out")?;

    let stream = i_st.candle_device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let i_slice = i_st.slice().slice(i_off..);
    let s_slice = s_st.slice().slice(s_off..);

    let status = unsafe {
        let (i_ptr, _g1) = i_slice.device_ptr(&stream);
        let (s_ptr, _g2) = s_slice.device_ptr(&stream);
        kiln_silu_inplace_save_sigmoid_f32(
            i_ptr as *mut f32,
            s_ptr as *mut f32,
            elems,
            raw_stream,
        )
    };
    if status != 0 {
        return Err(RmsNormError::Msg(format!(
            "kt-silu-save: FFI returned {status}"
        )));
    }
    Ok(())
}


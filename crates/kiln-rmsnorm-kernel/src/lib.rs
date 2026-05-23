//! Vendored fused norm CUDA kernels (Liger-style).
//!
//! This crate hosts decode-critical Liger-style fused norm kernels for kiln:
//!
//! 1. [`fused_rmsnorm`] — Qwen3.5-style RMSNorm `(1 + w) * x * rsqrt(mean(x^2) + eps)`.
//!    Replaces the ~11 candle ops behind `kiln-model::forward::rms_norm`.
//!    Used by `kiln/norm/pre_attn` and `kiln/norm/pre_mlp`.
//! 2. [`fused_rmsnorm_with_autograd`] — Phase 10 long-context training path:
//!    same forward semantics as [`fused_rmsnorm`], plus a manual CUDA
//!    backward kernel ([`fused_rmsnorm_backward`]) wired through
//!    [`candle_core::CustomOp2`] so the autograd engine saves only `x` and
//!    `weight` (not the F32 intermediates that the candle-op chain
//!    materializes). For Qwen3.5-4B at T=8192 this avoids ~32 × 2 saved
//!    F32 RMSNorm intermediates per training segment.
//! 3. [`fused_l2_qk_norm`] — fused L2-norm(Q) + scale(Q) + L2-norm(K) used by
//!    GDN linear attention. Replaces the ~11 candle ops behind the
//!    `kiln/gdn/qk_norm` block in `forward.rs`.
//! 4. [`fused_l2_qk_norm_gqa`] — CUDA GDN GQA fast path that normalizes
//!    unexpanded `[B, T, nk, dk]` Q/K and emits expanded `[B, T, nv, dk]`
//!    outputs in one launch.
//! 5. [`fused_rotary_qk`] — decode/paged-attention RoPE(Q,K) for contiguous
//!    bf16 Q/K tensors using precomputed f32 cos/sin tables.
//! 6. [`fused_mlp_silu_mul`] — fused bf16 `silu(gate) * up` for Qwen3.5
//!    SwiGLU MLPs.
//! 7. [`fused_sigmoid_mul`] — fused bf16 `x * sigmoid(gate)` for attention
//!    output gates.
//!
//! # Why
//!
//! Both norm chains expand into ~11 CUDA kernel launches per call when
//! expressed as candle ops. At decode time each launch has ~10 µs of
//! per-launch overhead, and the intermediate F32 tensors round-trip through
//! HBM on every step. Per PROFILING.md, the two RMSNorm NVTX ranges combined
//! for ~22% of decode wallclock pre-fusion (PR #130 era), and `kiln/gdn/qk_norm`
//! is 14.9% of decode wallclock post-PR #166 — the largest *unfused* GDN
//! region. Fusing each chain into a single kernel collapses launch overhead
//! and HBM traffic into one launch + one round-trip per call.
//!
//! # Provenance
//!
//! Algorithm modelled after LinkedIn's Liger-Kernel
//! (<https://github.com/linkedin/Liger-Kernel>, `src/liger_kernel/ops/rms_norm.py`),
//! reimplemented in raw CUDA C so kiln doesn't add a Triton runtime
//! dependency. Matches kiln's Qwen3.5 convention of `(1 + w) * x * rms_inv`
//! (weights centred on 0, not on 1) for RMSNorm; matches the
//! `kiln-model::forward::l2_normalize` contract `x / sqrt(sum(x^2) + eps)`
//! for the QK fused norm.
//!
//! # APIs
//!
//! - [`fused_rmsnorm`] — candle-compatible wrapper around the RMSNorm kernel.
//! - [`fused_rmsnorm_with_autograd`] — autograd-aware RMSNorm forward (uses
//!   the manual CUDA backward when grads are propagated).
//! - [`fused_rmsnorm_backward`] — direct CUDA backward kernel call.
//! - [`rmsnorm_backward_fallback`] — closed-form backward via candle ops,
//!   the correctness oracle for [`fused_rmsnorm_backward`].
//! - [`supports`] — `(x, weight)` capability check for the RMSNorm kernel.
//! - [`fused_l2_qk_norm`] — candle-compatible wrapper around the GDN QK
//!   fused-norm kernel. Returns `(q_out, k_out)`.
//! - [`supports_l2_qk_norm`] — capability check for the QK kernel.
//! - [`fused_l2_qk_norm_gqa`] / [`supports_l2_qk_norm_gqa`] — GDN GQA
//!   head-expand + QK norm CUDA path.
//!
//! # Envelope
//!
//! - bf16 activations, bf16 weights, bf16 outputs.
//! - Contiguous CUDA tensors only.
//! - Last dim (`hidden`) must be <= 8192 for expanded QK norm; exactly 128
//!   for the GQA head-expand fast path.
//! - `eps` is F32 — kiln uses 1e-6 for both kernels.
//!
//! Out of scope: fused GEMM prologue, non-bf16 dtypes, non-contiguous input.
//! Backward currently only supported for the RMSNorm kernel
//! ([`fused_rmsnorm_backward`]); the QK-norm kernels remain forward-only.

use candle_core::{
    CpuStorage, CudaStorage, DType, Device, Layout, Result, Shape, Tensor, backend::BackendStorage,
    cuda_backend::cudarc::driver::DevicePtr,
};
use half::bf16;
use std::sync::OnceLock;

/// kiln-tensor-typed surface alongside candle-typed. Same FFI.
/// Phase 7 deletes the candle path.
mod kt_api;
pub use kt_api::{
    adamw_step_f32_kt, fused_l2_qk_norm_gqa_kt, fused_l2_qk_norm_kt, fused_mlp_silu_mul_kt,
    fused_rmsnorm_backward_kt, fused_rmsnorm_kt, fused_rotary_one_kt, fused_rotary_qk_kt,
    fused_sigmoid_mul_kt, lora_decode_add_kt, lora_decode_hidden_kt, sgd_step_f32_kt,
    RmsNormError,
};

unsafe extern "C" {
    fn kiln_fused_rmsnorm(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        rows: i32,
        hidden: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_rmsnorm_bwd(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        grad_out: *const core::ffi::c_void,
        grad_x: *mut core::ffi::c_void,
        grad_w_partial_f32: *mut f32,
        rows: i32,
        hidden: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_f32_to_bf16(
        src: *const f32,
        dst: *mut core::ffi::c_void,
        n: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_l2_qk_norm(
        q_in: *const core::ffi::c_void,
        k_in: *const core::ffi::c_void,
        q_out: *mut core::ffi::c_void,
        k_out: *mut core::ffi::c_void,
        rows: i32,
        hidden: i32,
        q_scale: f32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_l2_qk_norm_gqa(
        q_in: *const core::ffi::c_void,
        k_in: *const core::ffi::c_void,
        q_out: *mut core::ffi::c_void,
        k_out: *mut core::ffi::c_void,
        rows: i32,
        nk: i32,
        ratio: i32,
        hidden: i32,
        q_scale: f32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_rotary_qk(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        cos: *const f32,
        sin: *const f32,
        q_out: *mut core::ffi::c_void,
        k_out: *mut core::ffi::c_void,
        batch: i32,
        seq_len: i32,
        q_heads: i32,
        k_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_rotary_one(
        x: *const core::ffi::c_void,
        cos: *const f32,
        sin: *const f32,
        out: *mut core::ffi::c_void,
        batch: i32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_rotary_one_bwd(
        grad_y: *const core::ffi::c_void,
        cos: *const f32,
        sin: *const f32,
        grad_x: *mut core::ffi::c_void,
        batch: i32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_attn_decode_qkv_split_qk_norm_rope_bf16(
        q_raw: *const core::ffi::c_void,
        k_raw: *const core::ffi::c_void,
        q_weight: *const core::ffi::c_void,
        k_weight: *const core::ffi::c_void,
        cos: *const f32,
        sin: *const f32,
        q_out: *mut core::ffi::c_void,
        k_out: *mut core::ffi::c_void,
        gate_out: *mut core::ffi::c_void,
        batch: i32,
        q_heads: i32,
        k_heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        has_gate: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_mlp_silu_mul_bf16(
        gate: *const core::ffi::c_void,
        up: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        elems: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_mlp_silu_mul_packed_bf16(
        gate_up_packed: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        rows: i64,
        cols: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_sigmoid_mul_bf16(
        x: *const core::ffi::c_void,
        gate: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        elems: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_lora_decode_hidden_bf16(
        x: *const core::ffi::c_void,
        a: *const core::ffi::c_void,
        hidden: *mut f32,
        batch: i32,
        in_dim: i32,
        rank: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_lora_decode_add_bf16(
        base: *const core::ffi::c_void,
        hidden: *const f32,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        scale: f32,
        batch: i32,
        out_dim: i32,
        rank: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_lora_add_inplace_f32(
        base: *mut f32,
        hidden: *const f32,
        b: *const f32,
        scale: f32,
        rows: i32,
        out_dim: i32,
        rank: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_f32(
        input: *const f32,
        weight: *const f32,
        state: *const f32,
        out: *mut f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_inplace_f32(
        input_out: *mut f32,
        weight: *const f32,
        state: *const f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_bwd_input_f32(
        grad_out: *const f32,
        weight: *const f32,
        grad_input: *mut f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_bwd_weight_f32(
        grad_out: *const f32,
        input: *const f32,
        state: *const f32,
        grad_weight: *mut f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_bwd_state_f32(
        grad_out: *const f32,
        weight: *const f32,
        grad_state: *mut f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_silu_inplace_save_sigmoid_f32(
        input_out: *mut f32,
        sigmoid_out: *mut f32,
        elems: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_sgd_step_f32(
        param: *mut f32,
        grad: *const f32,
        lr: f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_sgd_step_bf16(
        param: *mut core::ffi::c_void,
        grad: *const core::ffi::c_void,
        lr: f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_adamw_step_f32(
        param: *mut f32,
        grad: *const f32,
        first_moment: *mut f32,
        second_moment: *mut f32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_adamw_step_bf16(
        param: *mut core::ffi::c_void,
        grad: *const core::ffi::c_void,
        first_moment: *mut core::ffi::c_void,
        second_moment: *mut core::ffi::c_void,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        bias_correction1: f32,
        bias_correction2: f32,
        n: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

fn cuda_empty_kernel_outputs_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_CUDA_EMPTY_KERNEL_OUTPUTS").is_err())
}

/// Whether the fused RMSNorm kernel is available on the given tensor.
///
/// The kernel only supports CUDA + bf16 + contiguous + hidden <= 8192.
pub fn supports(x: &Tensor, weight: &Tensor) -> bool {
    matches!(x.device(), Device::Cuda(_))
        && x.dtype() == DType::BF16
        && weight.dtype() == DType::BF16
        && x.is_contiguous()
        && weight.is_contiguous()
        && x.rank() >= 1
        && x.dims().last().copied().unwrap_or(0) <= 8192
        && weight.dims() == &[x.dims().last().copied().unwrap_or(0)]
}

/// Run the fused RMSNorm kernel.
///
/// Inputs:
///   - `x`: bf16, CUDA, contiguous, any rank; last dim is the normalised axis.
///   - `weight`: bf16, CUDA, contiguous, shape `[hidden]` matching `x.last_dim()`.
///   - `eps`: epsilon inside the rsqrt. Qwen3.5 uses 1e-6.
///
/// Returns a freshly allocated bf16 tensor with the same shape as `x`.
///
/// Semantics: `out = (1 + weight) * x * rsqrt(mean(x^2, dim=-1) + eps)` cast
/// back to bf16. Matches `kiln-model::forward::rms_norm` (Qwen3.5-style,
/// weight centred on 0).
pub fn fused_rmsnorm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let device = x.device();

    if x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: both x and weight must be bf16 (got {:?}, {:?})",
            x.dtype(),
            weight.dtype()
        );
    }

    let x_dims = x.dims().to_vec();
    let hidden = *x_dims.last().ok_or_else(|| {
        candle_core::Error::Msg("kiln-rmsnorm-kernel: x must have rank >= 1".to_string())
    })?;

    let weight_dims = weight.dims();
    if weight_dims.len() != 1 || weight_dims[0] != hidden {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: weight shape {:?} does not match x last dim {hidden}",
            weight_dims
        );
    }

    if hidden > 8192 {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: hidden dim {hidden} exceeds kernel envelope (<= 8192)"
        );
    }

    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    if rows == 0 {
        // Empty leading axis — nothing to do. Return a zeros tensor with the
        // same shape so callers don't have to special-case.
        return Tensor::zeros(x_dims.as_slice(), DType::BF16, device);
    }

    let x = x.contiguous()?;
    let weight = weight.contiguous()?;

    let out = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty(x_dims.as_slice(), DType::BF16, device)? }
    } else {
        Tensor::zeros(x_dims.as_slice(), DType::BF16, device)?
    };

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: x must be on CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: weight must be on CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: out must be on CUDA"),
        };

        let stream = x_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let x_slice = x_cuda
            .as_cuda_slice::<bf16>()?
            .slice(x_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<bf16>()?
            .slice(w_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<bf16>()?
            .slice(o_layout.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (w_ptr, _g2) = w_slice.device_ptr(&stream);
            let (o_ptr, _g3) = o_slice.device_ptr(&stream);

            let status = kiln_fused_rmsnorm(
                x_ptr as *const _,
                w_ptr as *const _,
                o_ptr as *mut _,
                rows as i32,
                hidden as i32,
                eps,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_fused_rmsnorm failed with status {status}");
            }
        }
    }

    Ok(out)
}

/// Closed-form RMSNorm backward via candle ops. Correctness oracle for
/// [`fused_rmsnorm_backward`] and the CPU implementation behind
/// [`fused_rmsnorm_with_autograd`].
///
/// For `out = (1 + w) * x * rms_inv` with `rms_inv = rsqrt(mean(x^2) + eps)`
/// and `H = hidden`, the analytical gradients are:
///
/// ```text
/// c     = (1/H) * rms_inv^2 * sum_j ((1 + w_j) * x_j * grad_out_j)
/// dx_j  = rms_inv * ((1 + w_j) * grad_out_j - x_j * c)
/// dw_j  = sum_i (x_ij * rms_inv_i * grad_out_ij)
/// ```
///
/// All intermediate reductions stay in F32 for numerical stability; the
/// outputs are cast back to the input dtype at the end. Works on any device.
pub fn rmsnorm_backward_fallback(
    x: &Tensor,
    weight: &Tensor,
    grad_out: &Tensor,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let dtype = x.dtype();
    let last_dim = candle_core::D::Minus1;

    let x_dims = x.dims().to_vec();
    let hidden = *x_dims.last().ok_or_else(|| {
        candle_core::Error::Msg("rmsnorm_backward_fallback: x must have rank >= 1".to_string())
    })?;

    let x_f32 = x.to_dtype(DType::F32)?;
    let g_f32 = grad_out.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let w_plus_one = (w_f32.ones_like()? + &w_f32)?; // [hidden]

    // Per-row rms_inv = rsqrt(mean(x^2) + eps)  →  shape [..., 1]
    let mean_sq = x_f32.sqr()?.mean_keepdim(last_dim)?;
    let rms_inv = (mean_sq + eps)?.sqrt()?.recip()?;

    // c = (1/H) * rms_inv^2 * sum_j ((1+w_j) * x_j * g_j)   →  [..., 1]
    let inv_h = 1.0f64 / (hidden as f64);
    let weighted = x_f32.broadcast_mul(&w_plus_one)?.mul(&g_f32)?;
    let sum_xgw = weighted.sum_keepdim(last_dim)?;
    let c = (sum_xgw * inv_h)?.mul(&rms_inv)?.mul(&rms_inv)?;

    // grad_x = rms_inv * ((1+w) * grad_out - x * c)
    let term_a = g_f32.broadcast_mul(&w_plus_one)?;
    let term_b = x_f32.broadcast_mul(&c)?;
    let grad_x_f32 = (term_a - term_b)?.broadcast_mul(&rms_inv)?;
    let grad_x = grad_x_f32.to_dtype(dtype)?;

    // grad_w[j] = sum over leading dims of (x[..., j] * rms_inv[..., 0] * grad_out[..., j])
    let per_elem = x_f32.broadcast_mul(&rms_inv)?.mul(&g_f32)?;
    // Reduce all leading axes; `Tensor::sum(0)` removes axis 0 each call.
    let mut grad_w_f32 = per_elem;
    while grad_w_f32.rank() > 1 {
        grad_w_f32 = grad_w_f32.sum(0)?;
    }
    let grad_w = grad_w_f32.to_dtype(weight.dtype())?;

    Ok((grad_x, grad_w))
}

/// Run the fused CUDA RMSNorm backward kernel.
///
/// Inputs:
///   - `x`: bf16 CUDA contiguous, same shape as the forward input.
///   - `weight`: bf16 CUDA contiguous, shape `[hidden]`.
///   - `grad_out`: bf16 CUDA contiguous, same shape as `x`.
///   - `eps`: epsilon used in the forward (kiln uses 1e-6).
///
/// Returns `(grad_x, grad_w)` with the same dtype + shape as `x` and `weight`
/// respectively. Raises if the inputs are out-of-envelope (CPU, non-bf16,
/// non-contiguous, hidden > 8192).
///
/// `grad_w` is accumulated in F32 inside the kernel and cast to bf16 at the
/// end via [`kiln_f32_to_bf16`]. F32 accumulation is required because the
/// per-element contributions are O(2^-8) at typical scales and bf16
/// accumulation would lose precision over 8K rows.
pub fn fused_rmsnorm_backward(
    x: &Tensor,
    weight: &Tensor,
    grad_out: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    if !matches!(x.device(), Device::Cuda(_)) {
        candle_core::bail!("fused_rmsnorm_backward requires CUDA tensors");
    }
    if x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 || grad_out.dtype() != DType::BF16
    {
        candle_core::bail!(
            "fused_rmsnorm_backward requires bf16 (got x={:?}, w={:?}, g={:?})",
            x.dtype(),
            weight.dtype(),
            grad_out.dtype()
        );
    }
    if x.dims() != grad_out.dims() {
        candle_core::bail!(
            "fused_rmsnorm_backward shape mismatch: x={:?} grad_out={:?}",
            x.dims(),
            grad_out.dims()
        );
    }
    let x_dims = x.dims().to_vec();
    let hidden = *x_dims.last().ok_or_else(|| {
        candle_core::Error::Msg("fused_rmsnorm_backward: x must have rank >= 1".to_string())
    })?;
    if weight.dims().len() != 1 || weight.dims()[0] != hidden {
        candle_core::bail!(
            "fused_rmsnorm_backward: weight shape {:?} != [{hidden}]",
            weight.dims()
        );
    }
    if hidden > 8192 {
        candle_core::bail!(
            "fused_rmsnorm_backward: hidden dim {hidden} exceeds envelope (<= 8192)"
        );
    }

    let device = x.device();
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();

    let grad_x = Tensor::zeros(x_dims.as_slice(), DType::BF16, device)?;
    let grad_w = Tensor::zeros((hidden,), DType::BF16, device)?;

    if rows == 0 {
        return Ok((grad_x, grad_w));
    }

    let x = x.contiguous()?;
    let weight_c = weight.contiguous()?;
    let grad_out = grad_out.contiguous()?;

    // F32 partial accumulator for the cross-row grad_w reduction. Must be
    // zero-initialized: the kernel uses `atomicAdd` to accumulate.
    let grad_w_partial = Tensor::zeros((hidden,), DType::F32, device)?;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (w_storage, w_layout) = weight_c.storage_and_layout();
        let (g_storage, g_layout) = grad_out.storage_and_layout();
        let (dx_storage, dx_layout) = grad_x.storage_and_layout();
        let (dw_storage, dw_layout) = grad_w_partial.storage_and_layout();

        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("fused_rmsnorm_backward: x must be CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("fused_rmsnorm_backward: weight must be CUDA"),
        };
        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("fused_rmsnorm_backward: grad_out must be CUDA"),
        };
        let dx_cuda = match &*dx_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("fused_rmsnorm_backward: grad_x must be CUDA"),
        };
        let dw_cuda = match &*dw_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("fused_rmsnorm_backward: grad_w_partial must be CUDA"),
        };

        let stream = x_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let x_slice = x_cuda
            .as_cuda_slice::<bf16>()?
            .slice(x_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<bf16>()?
            .slice(w_layout.start_offset()..);
        let g_slice = g_cuda
            .as_cuda_slice::<bf16>()?
            .slice(g_layout.start_offset()..);
        let dx_slice = dx_cuda
            .as_cuda_slice::<bf16>()?
            .slice(dx_layout.start_offset()..);
        let dw_slice = dw_cuda
            .as_cuda_slice::<f32>()?
            .slice(dw_layout.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (w_ptr, _g2) = w_slice.device_ptr(&stream);
            let (g_ptr, _g3) = g_slice.device_ptr(&stream);
            let (dx_ptr, _g4) = dx_slice.device_ptr(&stream);
            let (dw_ptr, _g5) = dw_slice.device_ptr(&stream);

            let status = kiln_fused_rmsnorm_bwd(
                x_ptr as *const _,
                w_ptr as *const _,
                g_ptr as *const _,
                dx_ptr as *mut _,
                dw_ptr as *mut f32,
                rows as i32,
                hidden as i32,
                eps,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_fused_rmsnorm_bwd failed with status {status}");
            }
        }
    }

    // Cast partial F32 accumulator to bf16 grad_w.
    {
        let (dwp_storage, dwp_layout) = grad_w_partial.storage_and_layout();
        let (dw_storage, dw_layout) = grad_w.storage_and_layout();

        let dwp_cuda = match &*dwp_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("fused_rmsnorm_backward: grad_w_partial must be CUDA"),
        };
        let dw_cuda = match &*dw_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("fused_rmsnorm_backward: grad_w must be CUDA"),
        };

        let stream = dwp_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let dwp_slice = dwp_cuda
            .as_cuda_slice::<f32>()?
            .slice(dwp_layout.start_offset()..);
        let dw_slice = dw_cuda
            .as_cuda_slice::<bf16>()?
            .slice(dw_layout.start_offset()..);

        unsafe {
            let (src_ptr, _g1) = dwp_slice.device_ptr(&stream);
            let (dst_ptr, _g2) = dw_slice.device_ptr(&stream);

            let status = kiln_f32_to_bf16(
                src_ptr as *const f32,
                dst_ptr as *mut _,
                hidden as i32,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!("kiln_f32_to_bf16 failed with status {status}");
            }
        }
    }

    Ok((grad_x, grad_w))
}

/// `CustomOp2` wrapping the fused RMSNorm forward + manual backward.
///
/// Forward dispatches to:
///   - `cuda_fwd` → [`fused_rmsnorm`] (single-launch CUDA kernel).
///   - `cpu_fwd` → an explicit f32 row-wise loop matching
///     [`crate::rmsnorm_backward_fallback`]'s convention. Handles f32 and bf16
///     inputs; other dtypes error out.
///
/// Backward dispatches to:
///   - On CUDA (bf16 envelope): [`fused_rmsnorm_backward`] (single-launch
///     kernel that recomputes `rms_inv` from `x` rather than saving it).
///   - On CPU or out-of-envelope inputs: [`rmsnorm_backward_fallback`].
///
/// The training-time saved-tensor reduction comes from the CUDA path: the
/// candle-op chain materializes 4 F32 intermediates per RMSNorm call (x_f32,
/// rms_inv, w_plus_one, normed); the custom op saves only `x` and `weight`,
/// recomputing the rest in the fused backward kernel.
pub struct RmsNormCustomOp {
    pub eps: f32,
}

fn rmsnorm_cpu_forward_f32(x: &[f32], weight: &[f32], hidden: usize, eps: f32) -> Vec<f32> {
    let rows = x.len() / hidden;
    let mut out: Vec<f32> = Vec::with_capacity(x.len());
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mut sum_sq = 0.0f64;
        for &xj in row.iter() {
            sum_sq += (xj as f64) * (xj as f64);
        }
        let mean_sq = sum_sq / (hidden as f64);
        let rms_inv = ((mean_sq + eps as f64).sqrt() as f32).recip();
        for j in 0..hidden {
            let xj = row[j];
            let wj = weight[j];
            out.push((1.0f32 + wj) * xj * rms_inv);
        }
    }
    out
}

fn rmsnorm_cpu_forward_bf16(x: &[bf16], weight: &[bf16], hidden: usize, eps: f32) -> Vec<bf16> {
    let rows = x.len() / hidden;
    let mut out: Vec<bf16> = Vec::with_capacity(x.len());
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mut sum_sq = 0.0f64;
        for &xj in row.iter() {
            let v = xj.to_f32();
            sum_sq += (v as f64) * (v as f64);
        }
        let mean_sq = sum_sq / (hidden as f64);
        let rms_inv = ((mean_sq + eps as f64).sqrt() as f32).recip();
        for j in 0..hidden {
            let xj = row[j].to_f32();
            let wj = weight[j].to_f32();
            out.push(bf16::from_f32((1.0f32 + wj) * xj * rms_inv));
        }
    }
    out
}

impl candle_core::CustomOp2 for RmsNormCustomOp {
    fn name(&self) -> &'static str {
        "kiln-fused-rmsnorm"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_w: &CpuStorage,
        l_w: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dims = l_x.shape().dims().to_vec();
        let hidden = *dims.last().ok_or_else(|| {
            candle_core::Error::Msg("RmsNormCustomOp::cpu_fwd: x must have rank >= 1".to_string())
        })?;
        if l_w.shape().dims() != [hidden] {
            candle_core::bail!(
                "RmsNormCustomOp::cpu_fwd: weight shape {:?} != [{hidden}]",
                l_w.shape().dims()
            );
        }
        if !l_x.is_contiguous() || !l_w.is_contiguous() {
            candle_core::bail!("RmsNormCustomOp::cpu_fwd: requires contiguous inputs");
        }

        let shape = Shape::from(dims.as_slice());

        let x_n = l_x.shape().elem_count();
        let w_n = l_w.shape().elem_count();
        let result = match (s_x, s_w) {
            (CpuStorage::F32(x), CpuStorage::F32(w)) => {
                let x_slice = &x[l_x.start_offset()..l_x.start_offset() + x_n];
                let w_slice = &w[l_w.start_offset()..l_w.start_offset() + w_n];
                let out = rmsnorm_cpu_forward_f32(x_slice, w_slice, hidden, self.eps);
                CpuStorage::F32(out)
            }
            (CpuStorage::BF16(x), CpuStorage::BF16(w)) => {
                let x_slice = &x[l_x.start_offset()..l_x.start_offset() + x_n];
                let w_slice = &w[l_w.start_offset()..l_w.start_offset() + w_n];
                let out = rmsnorm_cpu_forward_bf16(x_slice, w_slice, hidden, self.eps);
                CpuStorage::BF16(out)
            }
            _ => candle_core::bail!(
                "RmsNormCustomOp::cpu_fwd: dtype combination not supported (x={:?}, w={:?})",
                s_x.dtype(),
                s_w.dtype()
            ),
        };

        Ok((result, shape))
    }

    fn cuda_fwd(
        &self,
        s_x: &CudaStorage,
        l_x: &Layout,
        s_w: &CudaStorage,
        l_w: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dims = l_x.shape().dims().to_vec();
        let hidden = *dims.last().ok_or_else(|| {
            candle_core::Error::Msg("RmsNormCustomOp::cuda_fwd: x must have rank >= 1".to_string())
        })?;
        if l_w.shape().dims() != [hidden] {
            candle_core::bail!(
                "RmsNormCustomOp::cuda_fwd: weight shape {:?} != [{hidden}]",
                l_w.shape().dims()
            );
        }
        if !l_x.is_contiguous() || !l_w.is_contiguous() {
            candle_core::bail!("RmsNormCustomOp::cuda_fwd: requires contiguous inputs");
        }
        if hidden > 8192 {
            candle_core::bail!(
                "RmsNormCustomOp::cuda_fwd: hidden {hidden} exceeds envelope (<= 8192)"
            );
        }

        let rows: usize = dims[..dims.len() - 1].iter().product();
        let device = s_x.device();
        let stream = device.cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let elem_count: usize = dims.iter().product();
        // alloc_zeros handles both the zero-row no-op case and uninitialised
        // tail when `rows * hidden < elem_count` for higher-rank inputs.
        let out_slice = device.alloc_zeros::<bf16>(elem_count)?;
        let shape = Shape::from(dims.as_slice());

        if rows == 0 {
            return Ok((
                CudaStorage::wrap_cuda_slice(out_slice, device.clone()),
                shape,
            ));
        }

        let x_slice = s_x.as_cuda_slice::<bf16>()?.slice(l_x.start_offset()..);
        let w_slice = s_w.as_cuda_slice::<bf16>()?.slice(l_w.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (w_ptr, _g2) = w_slice.device_ptr(&stream);
            let (o_ptr, _g3) = out_slice.device_ptr(&stream);

            let status = kiln_fused_rmsnorm(
                x_ptr as *const _,
                w_ptr as *const _,
                o_ptr as *mut _,
                rows as i32,
                hidden as i32,
                self.eps,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!(
                    "RmsNormCustomOp::cuda_fwd: kiln_fused_rmsnorm failed (status {status})"
                );
            }
        }

        Ok((
            CudaStorage::wrap_cuda_slice(out_slice, device.clone()),
            shape,
        ))
    }

    fn bwd(
        &self,
        x: &Tensor,
        weight: &Tensor,
        _res: &Tensor,
        grad_out: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let cuda_eligible = matches!(x.device(), Device::Cuda(_))
            && x.dtype() == DType::BF16
            && weight.dtype() == DType::BF16
            && grad_out.dtype() == DType::BF16
            && x.is_contiguous()
            && weight.is_contiguous()
            && x.dims().last().copied().unwrap_or(0) <= 8192;

        if cuda_eligible {
            let grad_out_c = grad_out.contiguous()?;
            let (gx, gw) = fused_rmsnorm_backward(x, weight, &grad_out_c, self.eps)?;
            return Ok((Some(gx), Some(gw)));
        }

        // CPU / out-of-envelope path: closed-form backward via candle ops.
        let (gx, gw) = rmsnorm_backward_fallback(x, weight, grad_out, self.eps as f64)?;
        Ok((Some(gx), Some(gw)))
    }
}

/// Apply the fused RMSNorm op with manual-backward autograd support.
///
/// Equivalent in math to [`fused_rmsnorm`] (and `kiln-model::forward::rms_norm`)
/// but routes the forward through [`candle_core::CustomOp2`] so the gradient
/// graph saves only `x` and `weight`, with the backward kernel recomputing
/// `rms_inv` on the fly. This is the Phase 10 long-context training path —
/// the saved-tensor reduction compounds 32× across Qwen3.5-4B layers.
pub fn fused_rmsnorm_with_autograd(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let op = RmsNormCustomOp { eps };
    x.apply_op2(weight, op)
}

/// Whether [`fused_rmsnorm_with_autograd`] should be used for the given
/// inputs. CPU is always eligible (cpu_fwd handles f32/bf16 directly); CUDA
/// is eligible when the existing forward kernel envelope holds.
pub fn supports_autograd(x: &Tensor, weight: &Tensor) -> bool {
    if x.dtype() != weight.dtype() {
        return false;
    }
    if !matches!(x.dtype(), DType::F32 | DType::BF16) {
        return false;
    }
    if !x.is_contiguous() || !weight.is_contiguous() {
        return false;
    }
    if x.rank() < 1 {
        return false;
    }
    let hidden = x.dims().last().copied().unwrap_or(0);
    if weight.dims() != [hidden] {
        return false;
    }
    match x.device() {
        Device::Cpu => true,
        Device::Cuda(_) => {
            // CUDA fwd kernel only handles bf16 + hidden <= 8192.
            x.dtype() == DType::BF16 && hidden <= 8192
        }
        _ => false,
    }
}

/// Whether the fused L2 QK-norm kernel is available for the given Q, K tensors.
///
/// Both must be CUDA + bf16 + contiguous, with identical shape, last dim <= 8192.
/// Decode shape on Qwen3.5-4B is `[batch, seq, nv, dk]` with `dk = 128`.
pub fn supports_l2_qk_norm(q: &Tensor, k: &Tensor) -> bool {
    matches!(q.device(), Device::Cuda(_))
        && matches!(k.device(), Device::Cuda(_))
        && q.dtype() == DType::BF16
        && k.dtype() == DType::BF16
        && q.is_contiguous()
        && k.is_contiguous()
        && q.dims() == k.dims()
        && q.rank() >= 1
        && q.dims().last().copied().unwrap_or(0) <= 8192
}

/// Run the fused L2 QK-norm kernel.
///
/// Inputs:
///   - `q`, `k`: bf16, CUDA, contiguous, identical shape; last dim is the
///     normalised axis (`dk`). Decode shape: `[batch=1, seq=1, nv, dk]`.
///   - `q_scale`: scalar applied to Q after L2-normalisation. kiln uses
///     `1 / sqrt(dk)`.
///   - `eps`: epsilon inside the rsqrt. kiln uses 1e-6.
///
/// Returns `(q_out, k_out)`, freshly allocated bf16 tensors with the same
/// shape as the inputs.
///
/// Semantics — matches `kiln-model::forward::l2_normalize` + the
/// `kiln/gdn/qk_norm` block in forward.rs exactly:
///
///   q_out = (q / sqrt(sum(q^2, dim=-1) + eps)) * q_scale, cast to bf16
///   k_out =  k / sqrt(sum(k^2, dim=-1) + eps),            cast to bf16
pub fn fused_l2_qk_norm(
    q: &Tensor,
    k: &Tensor,
    q_scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let device = q.device();

    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: l2_qk_norm requires bf16 inputs (got {:?}, {:?})",
            q.dtype(),
            k.dtype()
        );
    }
    if q.dims() != k.dims() {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: l2_qk_norm requires q.dims == k.dims (got {:?}, {:?})",
            q.dims(),
            k.dims()
        );
    }

    let dims = q.dims().to_vec();
    let hidden = *dims.last().ok_or_else(|| {
        candle_core::Error::Msg("kiln-rmsnorm-kernel: q must have rank >= 1".to_string())
    })?;

    if hidden > 8192 {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: l2_qk_norm hidden dim {hidden} exceeds envelope (<= 8192)"
        );
    }

    let rows: usize = dims[..dims.len() - 1].iter().product();

    let q_out = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty(dims.as_slice(), DType::BF16, device)? }
    } else {
        Tensor::zeros(dims.as_slice(), DType::BF16, device)?
    };
    let k_out = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty(dims.as_slice(), DType::BF16, device)? }
    } else {
        Tensor::zeros(dims.as_slice(), DType::BF16, device)?
    };

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let q = q.contiguous()?;
    let k = k.contiguous()?;

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();

        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: q must be on CUDA"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: k must be on CUDA"),
        };
        let qo_cuda = match &*qo_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: q_out must be on CUDA"),
        };
        let ko_cuda = match &*ko_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: k_out must be on CUDA"),
        };

        let stream = q_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let q_slice = q_cuda
            .as_cuda_slice::<bf16>()?
            .slice(q_layout.start_offset()..);
        let k_slice = k_cuda
            .as_cuda_slice::<bf16>()?
            .slice(k_layout.start_offset()..);
        let qo_slice = qo_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qo_layout.start_offset()..);
        let ko_slice = ko_cuda
            .as_cuda_slice::<bf16>()?
            .slice(ko_layout.start_offset()..);

        unsafe {
            let (q_ptr, _g1) = q_slice.device_ptr(&stream);
            let (k_ptr, _g2) = k_slice.device_ptr(&stream);
            let (qo_ptr, _g3) = qo_slice.device_ptr(&stream);
            let (ko_ptr, _g4) = ko_slice.device_ptr(&stream);

            let status = kiln_fused_l2_qk_norm(
                q_ptr as *const _,
                k_ptr as *const _,
                qo_ptr as *mut _,
                ko_ptr as *mut _,
                rows as i32,
                hidden as i32,
                q_scale,
                eps,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_fused_l2_qk_norm failed with status {status}");
            }
        }
    }

    Ok((q_out, k_out))
}

/// Whether the fused GQA head-expand + L2 QK-norm kernel is available.
///
/// Inputs must be CUDA + bf16 with shape `[batch, seq, nk, dk]`; the wrapper
/// materializes contiguous inputs before launch when needed.
/// `nv` must be a positive multiple of `nk`; `dk` is intentionally limited to
/// Qwen3.5 GDN's `128` so this remains a narrow forward-only CUDA path.
pub fn supports_l2_qk_norm_gqa(q: &Tensor, k: &Tensor, nv: usize) -> bool {
    if !matches!(q.device(), Device::Cuda(_))
        || !matches!(k.device(), Device::Cuda(_))
        || q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || q.dims() != k.dims()
        || q.rank() != 4
    {
        return false;
    }

    let dims = q.dims();
    let nk = dims[2];
    let dk = dims[3];
    nk > 0 && dk == 128 && nv >= nk && nv % nk == 0
}

/// Run fused GQA head-expand + L2 QK-norm.
///
/// Inputs are unexpanded GDN Q/K tensors `[batch, seq, nk, dk]`; outputs are
/// freshly allocated bf16 tensors `[batch, seq, nv, dk]`, with each normalized
/// input head repeated `nv / nk` times. Semantics match explicit Candle
/// `expand(...).contiguous().reshape(...)` followed by [`fused_l2_qk_norm`].
pub fn fused_l2_qk_norm_gqa(
    q: &Tensor,
    k: &Tensor,
    nv: usize,
    q_scale: f32,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: l2_qk_norm_gqa requires bf16 inputs (got {:?}, {:?})",
            q.dtype(),
            k.dtype()
        );
    }
    if q.dims() != k.dims() {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: l2_qk_norm_gqa requires q.dims == k.dims (got {:?}, {:?})",
            q.dims(),
            k.dims()
        );
    }
    if q.rank() != 4 {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: l2_qk_norm_gqa requires rank-4 [B,T,nk,dk] input (got {:?})",
            q.dims()
        );
    }

    let dims = q.dims();
    let batch = dims[0];
    let seq = dims[1];
    let nk = dims[2];
    let dk = dims[3];

    if nk == 0 || nv < nk || nv % nk != 0 {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: l2_qk_norm_gqa requires nv to be a positive multiple of nk (nk={nk}, nv={nv})"
        );
    }
    if dk != 128 {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: l2_qk_norm_gqa dk {dk} outside envelope (expected 128)"
        );
    }

    let ratio = nv / nk;
    let rows = batch * seq * nk;
    let device = q.device();
    let out_dims = [batch, seq, nv, dk];

    let q_out = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty(&out_dims, DType::BF16, device)? }
    } else {
        Tensor::zeros(&out_dims, DType::BF16, device)?
    };
    let k_out = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty(&out_dims, DType::BF16, device)? }
    } else {
        Tensor::zeros(&out_dims, DType::BF16, device)?
    };

    if rows == 0 {
        return Ok((q_out, k_out));
    }

    let q = q.contiguous()?;
    let k = k.contiguous()?;

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();

        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: q must be on CUDA"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: k must be on CUDA"),
        };
        let qo_cuda = match &*qo_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: q_out must be on CUDA"),
        };
        let ko_cuda = match &*ko_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: k_out must be on CUDA"),
        };

        let stream = q_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let q_slice = q_cuda
            .as_cuda_slice::<bf16>()?
            .slice(q_layout.start_offset()..);
        let k_slice = k_cuda
            .as_cuda_slice::<bf16>()?
            .slice(k_layout.start_offset()..);
        let qo_slice = qo_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qo_layout.start_offset()..);
        let ko_slice = ko_cuda
            .as_cuda_slice::<bf16>()?
            .slice(ko_layout.start_offset()..);

        unsafe {
            let (q_ptr, _g1) = q_slice.device_ptr(&stream);
            let (k_ptr, _g2) = k_slice.device_ptr(&stream);
            let (qo_ptr, _g3) = qo_slice.device_ptr(&stream);
            let (ko_ptr, _g4) = ko_slice.device_ptr(&stream);

            let status = kiln_fused_l2_qk_norm_gqa(
                q_ptr as *const _,
                k_ptr as *const _,
                qo_ptr as *mut _,
                ko_ptr as *mut _,
                rows as i32,
                nk as i32,
                ratio as i32,
                dk as i32,
                q_scale,
                eps,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_fused_l2_qk_norm_gqa failed with status {status}");
            }
        }
    }

    Ok((q_out, k_out))
}

/// Whether the fused rotary Q/K kernel is available.
///
/// Supports CUDA bf16 contiguous Q/K tensors shaped `[batch, seq, heads,
/// head_dim]` and f32 contiguous cos/sin tables shaped
/// `[seq, rotary_dim / 2]`. This matches the table shape produced once per
/// eager paged forward in `kiln-model::forward`.
pub fn supports_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> bool {
    if !matches!(q.device(), Device::Cuda(_))
        || !matches!(k.device(), Device::Cuda(_))
        || !matches!(cos.device(), Device::Cuda(_))
        || !matches!(sin.device(), Device::Cuda(_))
        || q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || q.rank() != 4
        || k.rank() != 4
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
    {
        return false;
    }
    let qd = q.dims();
    let kd = k.dims();
    let batch = qd[0];
    let seq_len = qd[1];
    qd[3] == head_dim
        && kd[0] == batch
        && kd[1] == seq_len
        && kd[3] == head_dim
        && cos.dims() == [seq_len, rotary_dim / 2]
        && sin.dims() == [seq_len, rotary_dim / 2]
        && batch <= i32::MAX as usize
        && seq_len <= i32::MAX as usize
        && qd[2] <= i32::MAX as usize
        && kd[2] <= i32::MAX as usize
        && head_dim <= i32::MAX as usize
        && rotary_dim <= i32::MAX as usize
}

/// Run fused RoPE over Q and K.
///
/// Semantics match `kiln-model::forward::apply_rope` for the contiguous-half
/// layout used by Qwen3.5: first half and second half of `rotary_dim` are
/// rotated together, remaining head dimensions pass through unchanged.
pub fn fused_rotary_qk(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<(Tensor, Tensor)> {
    if !supports_rotary_qk(q, k, cos, sin, head_dim, rotary_dim) {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: rotary_qk unsupported shapes q={:?} k={:?} cos={:?} sin={:?} dtypes=({:?},{:?},{:?},{:?}) head_dim={head_dim} rotary_dim={rotary_dim}",
            q.shape(),
            k.shape(),
            cos.shape(),
            sin.shape(),
            q.dtype(),
            k.dtype(),
            cos.dtype(),
            sin.dtype()
        );
    }

    let q_dims = q.dims();
    let k_dims = k.dims();
    let batch = q_dims[0];
    let seq_len = q_dims[1];
    let q_heads = q_dims[2];
    let k_heads = k_dims[2];
    let device = q.device();
    let q_out = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty(q_dims, DType::BF16, device)? }
    } else {
        Tensor::zeros(q_dims, DType::BF16, device)?
    };
    let k_out = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty(k_dims, DType::BF16, device)? }
    } else {
        Tensor::zeros(k_dims, DType::BF16, device)?
    };

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (cos_storage, cos_layout) = cos.storage_and_layout();
        let (sin_storage, sin_layout) = sin.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();

        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary q must be on CUDA"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary k must be on CUDA"),
        };
        let cos_cuda = match &*cos_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary cos must be on CUDA"),
        };
        let sin_cuda = match &*sin_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary sin must be on CUDA"),
        };
        let qo_cuda = match &*qo_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary q_out must be on CUDA"),
        };
        let ko_cuda = match &*ko_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary k_out must be on CUDA"),
        };

        let stream = q_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let q_slice = q_cuda
            .as_cuda_slice::<bf16>()?
            .slice(q_layout.start_offset()..);
        let k_slice = k_cuda
            .as_cuda_slice::<bf16>()?
            .slice(k_layout.start_offset()..);
        let cos_slice = cos_cuda
            .as_cuda_slice::<f32>()?
            .slice(cos_layout.start_offset()..);
        let sin_slice = sin_cuda
            .as_cuda_slice::<f32>()?
            .slice(sin_layout.start_offset()..);
        let qo_slice = qo_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qo_layout.start_offset()..);
        let ko_slice = ko_cuda
            .as_cuda_slice::<bf16>()?
            .slice(ko_layout.start_offset()..);

        unsafe {
            let (q_ptr, _g1) = q_slice.device_ptr(&stream);
            let (k_ptr, _g2) = k_slice.device_ptr(&stream);
            let (cos_ptr, _g3) = cos_slice.device_ptr(&stream);
            let (sin_ptr, _g4) = sin_slice.device_ptr(&stream);
            let (qo_ptr, _g5) = qo_slice.device_ptr(&stream);
            let (ko_ptr, _g6) = ko_slice.device_ptr(&stream);
            let status = kiln_fused_rotary_qk(
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
            );
            if status != 0 {
                candle_core::bail!("kiln_fused_rotary_qk failed with status {status}");
            }
        }
    }

    Ok((q_out, k_out))
}

/// Storage-level single-tensor BF16 RoPE for CUDA custom ops.
pub fn rotary_one_bf16_storage(
    out_cuda: &CudaStorage,
    out_layout: &Layout,
    x_cuda: &CudaStorage,
    x_layout: &Layout,
    cos_cuda: &CudaStorage,
    cos_layout: &Layout,
    sin_cuda: &CudaStorage,
    sin_layout: &Layout,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<()> {
    let x_dims = x_layout.dims();
    let out_dims = out_layout.dims();
    if x_dims.len() != 4 || out_dims != x_dims {
        candle_core::bail!(
            "rotary_one_bf16_storage: x/out must be matching rank-4 tensors, got x={x_dims:?} out={out_dims:?}"
        );
    }
    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        candle_core::bail!(
            "rotary_one_bf16_storage: invalid head_dim={head_dim} rotary_dim={rotary_dim}"
        );
    }
    let (batch, seq_len, heads, x_head_dim) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    if x_head_dim != head_dim {
        candle_core::bail!(
            "rotary_one_bf16_storage: x head dim {x_head_dim} != head_dim {head_dim}"
        );
    }
    if cos_layout.dims() != [seq_len, rotary_dim / 2]
        || sin_layout.dims() != [seq_len, rotary_dim / 2]
    {
        candle_core::bail!(
            "rotary_one_bf16_storage: table shape mismatch cos={:?} sin={:?} expected=[{seq_len}, {}]",
            cos_layout.dims(),
            sin_layout.dims(),
            rotary_dim / 2
        );
    }
    if out_cuda.dtype() != DType::BF16
        || x_cuda.dtype() != DType::BF16
        || cos_cuda.dtype() != DType::F32
        || sin_cuda.dtype() != DType::F32
    {
        candle_core::bail!(
            "rotary_one_bf16_storage: expected out/x BF16 and cos/sin F32, got out={:?} x={:?} cos={:?} sin={:?}",
            out_cuda.dtype(),
            x_cuda.dtype(),
            cos_cuda.dtype(),
            sin_cuda.dtype()
        );
    }
    if !out_layout.is_contiguous()
        || !x_layout.is_contiguous()
        || !cos_layout.is_contiguous()
        || !sin_layout.is_contiguous()
    {
        candle_core::bail!("rotary_one_bf16_storage: tensors must be contiguous");
    }
    if batch > i32::MAX as usize
        || seq_len > i32::MAX as usize
        || heads > i32::MAX as usize
        || head_dim > i32::MAX as usize
        || rotary_dim > i32::MAX as usize
    {
        candle_core::bail!("rotary_one_bf16_storage: dimensions exceed i32 kernel envelope");
    }

    let stream = x_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let x_slice = x_cuda
        .as_cuda_slice::<bf16>()?
        .slice(x_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_cuda_slice::<f32>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_cuda_slice::<f32>()?
        .slice(sin_layout.start_offset()..);
    let out_slice = out_cuda
        .as_cuda_slice::<bf16>()?
        .slice(out_layout.start_offset()..);

    let status = unsafe {
        let (x_ptr, _x_guard) = x_slice.device_ptr(&stream);
        let (cos_ptr, _cos_guard) = cos_slice.device_ptr(&stream);
        let (sin_ptr, _sin_guard) = sin_slice.device_ptr(&stream);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        kiln_fused_rotary_one(
            x_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            out_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_fused_rotary_one failed with status {status}");
    }
    Ok(())
}

/// Whether the fused single-tensor BF16 RoPE backward kernel is available.
pub fn supports_rotary_one_bwd_bf16(
    grad_y: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> bool {
    if !matches!(grad_y.device(), Device::Cuda(_))
        || !matches!(cos.device(), Device::Cuda(_))
        || !matches!(sin.device(), Device::Cuda(_))
        || grad_y.dtype() != DType::BF16
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || !grad_y.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || grad_y.rank() != 4
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
    {
        return false;
    }
    let dims = grad_y.dims();
    let batch = dims[0];
    let seq_len = dims[1];
    let heads = dims[2];
    dims[3] == head_dim
        && cos.dims() == [seq_len, rotary_dim / 2]
        && sin.dims() == [seq_len, rotary_dim / 2]
        && batch <= i32::MAX as usize
        && seq_len <= i32::MAX as usize
        && heads <= i32::MAX as usize
        && head_dim <= i32::MAX as usize
        && rotary_dim <= i32::MAX as usize
}

/// Run fused single-tensor BF16 RoPE backward.
pub fn rotary_one_bwd_bf16(
    grad_y: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Tensor> {
    if !supports_rotary_one_bwd_bf16(grad_y, cos, sin, head_dim, rotary_dim) {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: rotary_one_bwd unsupported shapes grad_y={:?} cos={:?} sin={:?} dtypes=({:?},{:?},{:?}) head_dim={head_dim} rotary_dim={rotary_dim}",
            grad_y.shape(),
            cos.shape(),
            sin.shape(),
            grad_y.dtype(),
            cos.dtype(),
            sin.dtype()
        );
    }

    let grad_x = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty(grad_y.dims(), DType::BF16, grad_y.device())? }
    } else {
        Tensor::zeros(grad_y.dims(), DType::BF16, grad_y.device())?
    };

    {
        let (grad_y_storage, grad_y_layout) = grad_y.storage_and_layout();
        let (cos_storage, cos_layout) = cos.storage_and_layout();
        let (sin_storage, sin_layout) = sin.storage_and_layout();
        let (grad_x_storage, grad_x_layout) = grad_x.storage_and_layout();

        let grad_y_cuda = match &*grad_y_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary_one_bwd grad_y must be on CUDA"),
        };
        let cos_cuda = match &*cos_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary_one_bwd cos must be on CUDA"),
        };
        let sin_cuda = match &*sin_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary_one_bwd sin must be on CUDA"),
        };
        let grad_x_cuda = match &*grad_x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary_one_bwd grad_x must be on CUDA"),
        };

        rotary_one_bwd_bf16_storage(
            grad_x_cuda,
            &grad_x_layout,
            grad_y_cuda,
            &grad_y_layout,
            cos_cuda,
            &cos_layout,
            sin_cuda,
            &sin_layout,
            head_dim,
            rotary_dim,
        )?;
    }
    Ok(grad_x)
}

/// Storage-level single-tensor BF16 RoPE backward for CUDA custom ops.
pub fn rotary_one_bwd_bf16_storage(
    grad_x_cuda: &CudaStorage,
    grad_x_layout: &Layout,
    grad_y_cuda: &CudaStorage,
    grad_y_layout: &Layout,
    cos_cuda: &CudaStorage,
    cos_layout: &Layout,
    sin_cuda: &CudaStorage,
    sin_layout: &Layout,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<()> {
    let grad_y_dims = grad_y_layout.dims();
    let grad_x_dims = grad_x_layout.dims();
    if grad_y_dims.len() != 4 || grad_x_dims != grad_y_dims {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: grad_y/grad_x must be matching rank-4 tensors, got grad_y={grad_y_dims:?} grad_x={grad_x_dims:?}"
        );
    }
    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: invalid head_dim={head_dim} rotary_dim={rotary_dim}"
        );
    }
    let (batch, seq_len, heads, grad_head_dim) = (
        grad_y_dims[0],
        grad_y_dims[1],
        grad_y_dims[2],
        grad_y_dims[3],
    );
    if grad_head_dim != head_dim {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: grad_y head dim {grad_head_dim} != head_dim {head_dim}"
        );
    }
    if cos_layout.dims() != [seq_len, rotary_dim / 2]
        || sin_layout.dims() != [seq_len, rotary_dim / 2]
    {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: table shape mismatch cos={:?} sin={:?} expected=[{seq_len}, {}]",
            cos_layout.dims(),
            sin_layout.dims(),
            rotary_dim / 2
        );
    }
    if grad_x_cuda.dtype() != DType::BF16
        || grad_y_cuda.dtype() != DType::BF16
        || cos_cuda.dtype() != DType::F32
        || sin_cuda.dtype() != DType::F32
    {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: expected grad_x/grad_y BF16 and cos/sin F32, got grad_x={:?} grad_y={:?} cos={:?} sin={:?}",
            grad_x_cuda.dtype(),
            grad_y_cuda.dtype(),
            cos_cuda.dtype(),
            sin_cuda.dtype()
        );
    }
    if !grad_x_layout.is_contiguous()
        || !grad_y_layout.is_contiguous()
        || !cos_layout.is_contiguous()
        || !sin_layout.is_contiguous()
    {
        candle_core::bail!("rotary_one_bwd_bf16_storage: tensors must be contiguous");
    }
    if batch > i32::MAX as usize
        || seq_len > i32::MAX as usize
        || heads > i32::MAX as usize
        || head_dim > i32::MAX as usize
        || rotary_dim > i32::MAX as usize
    {
        candle_core::bail!("rotary_one_bwd_bf16_storage: dimensions exceed i32 kernel envelope");
    }

    let stream = grad_y_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let grad_y_slice = grad_y_cuda
        .as_cuda_slice::<bf16>()?
        .slice(grad_y_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_cuda_slice::<f32>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_cuda_slice::<f32>()?
        .slice(sin_layout.start_offset()..);
    let grad_x_slice = grad_x_cuda
        .as_cuda_slice::<bf16>()?
        .slice(grad_x_layout.start_offset()..);

    let status = unsafe {
        let (grad_y_ptr, _grad_y_guard) = grad_y_slice.device_ptr(&stream);
        let (cos_ptr, _cos_guard) = cos_slice.device_ptr(&stream);
        let (sin_ptr, _sin_guard) = sin_slice.device_ptr(&stream);
        let (grad_x_ptr, _grad_x_guard) = grad_x_slice.device_ptr(&stream);
        kiln_fused_rotary_one_bwd(
            grad_y_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            grad_x_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_fused_rotary_one_bwd failed with status {status}");
    }
    Ok(())
}

/// Whether the fused single-token decode QKV-prep kernel is available.
///
/// Supports post-projection CUDA bf16 tensors for a decode step:
/// `q_raw=[batch, 1, q_heads * head_dim * (has_gate ? 2 : 1)]`,
/// `k_raw=[batch, 1, k_heads * head_dim]`, bf16 norm weights
/// `[head_dim]`, and f32 RoPE tables `[1, rotary_dim / 2]`.
pub fn supports_attn_decode_qkv_prep(
    q_raw: &Tensor,
    k_raw: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    has_gate: bool,
) -> bool {
    if !matches!(q_raw.device(), Device::Cuda(_))
        || !matches!(k_raw.device(), Device::Cuda(_))
        || !matches!(q_weight.device(), Device::Cuda(_))
        || !matches!(k_weight.device(), Device::Cuda(_))
        || !matches!(cos.device(), Device::Cuda(_))
        || !matches!(sin.device(), Device::Cuda(_))
        || q_raw.dtype() != DType::BF16
        || k_raw.dtype() != DType::BF16
        || q_weight.dtype() != DType::BF16
        || k_weight.dtype() != DType::BF16
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || !q_raw.is_contiguous()
        || !k_raw.is_contiguous()
        || !q_weight.is_contiguous()
        || !k_weight.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || q_raw.rank() != 3
        || k_raw.rank() != 3
        || q_heads == 0
        || k_heads == 0
        || head_dim == 0
        || head_dim > 8192
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
    {
        return false;
    }

    let qd = q_raw.dims();
    let kd = k_raw.dims();
    let batch = qd[0];
    let Some(q_base) = q_heads.checked_mul(head_dim) else {
        return false;
    };
    let Some(q_inner) = (if has_gate {
        q_base.checked_mul(2)
    } else {
        Some(q_base)
    }) else {
        return false;
    };
    let Some(k_inner) = k_heads.checked_mul(head_dim) else {
        return false;
    };
    let Some(total_heads) = q_heads.checked_add(k_heads) else {
        return false;
    };
    let Some(total_rows) = batch.checked_mul(total_heads) else {
        return false;
    };
    qd[1] == 1
        && kd[0] == batch
        && kd[1] == 1
        && qd[2] == q_inner
        && kd[2] == k_inner
        && q_weight.dims() == [head_dim]
        && k_weight.dims() == [head_dim]
        && cos.dims() == [1, rotary_dim / 2]
        && sin.dims() == [1, rotary_dim / 2]
        && batch <= i32::MAX as usize
        && total_rows <= i32::MAX as usize
        && q_heads <= i32::MAX as usize
        && k_heads <= i32::MAX as usize
        && head_dim <= i32::MAX as usize
        && rotary_dim <= i32::MAX as usize
}

/// Fuse decode-time Q/gate split, Q/K RMSNorm, and Q/K RoPE.
///
/// This is forward-only and intended for inference decode. It preserves the
/// existing numerical sequence by bf16-rounding the RMSNorm output before the
/// RoPE arithmetic.
pub fn fused_attn_decode_qkv_prep(
    q_raw: &Tensor,
    k_raw: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q_heads: usize,
    k_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    has_gate: bool,
    eps: f32,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    if !supports_attn_decode_qkv_prep(
        q_raw, k_raw, q_weight, k_weight, cos, sin, q_heads, k_heads, head_dim, rotary_dim,
        has_gate,
    ) {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: attn_decode_qkv_prep unsupported shapes q_raw={:?} k_raw={:?} q_weight={:?} k_weight={:?} cos={:?} sin={:?} dtypes=({:?},{:?},{:?},{:?},{:?},{:?}) heads=({q_heads},{k_heads}) head_dim={head_dim} rotary_dim={rotary_dim} has_gate={has_gate}",
            q_raw.shape(),
            k_raw.shape(),
            q_weight.shape(),
            k_weight.shape(),
            cos.shape(),
            sin.shape(),
            q_raw.dtype(),
            k_raw.dtype(),
            q_weight.dtype(),
            k_weight.dtype(),
            cos.dtype(),
            sin.dtype()
        );
    }

    let batch = q_raw.dims()[0];
    let device = q_raw.device();
    let q_out = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty((batch, 1, q_heads, head_dim), DType::BF16, device)? }
    } else {
        Tensor::zeros((batch, 1, q_heads, head_dim), DType::BF16, device)?
    };
    let k_out = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty((batch, 1, k_heads, head_dim), DType::BF16, device)? }
    } else {
        Tensor::zeros((batch, 1, k_heads, head_dim), DType::BF16, device)?
    };
    let gate_out = if has_gate {
        Some(if cuda_empty_kernel_outputs_enabled() {
            unsafe { Tensor::empty((batch, 1, q_heads * head_dim), DType::BF16, device)? }
        } else {
            Tensor::zeros((batch, 1, q_heads * head_dim), DType::BF16, device)?
        })
    } else {
        None
    };

    let q_raw = q_raw.contiguous()?;
    let k_raw = k_raw.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;

    {
        let (qr_storage, qr_layout) = q_raw.storage_and_layout();
        let (kr_storage, kr_layout) = k_raw.storage_and_layout();
        let (qw_storage, qw_layout) = q_weight.storage_and_layout();
        let (kw_storage, kw_layout) = k_weight.storage_and_layout();
        let (cos_storage, cos_layout) = cos.storage_and_layout();
        let (sin_storage, sin_layout) = sin.storage_and_layout();
        let (qo_storage, qo_layout) = q_out.storage_and_layout();
        let (ko_storage, ko_layout) = k_out.storage_and_layout();

        let qr_cuda = match &*qr_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: q_raw must be on CUDA"),
        };
        let kr_cuda = match &*kr_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: k_raw must be on CUDA"),
        };
        let qw_cuda = match &*qw_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: q_weight must be on CUDA"),
        };
        let kw_cuda = match &*kw_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: k_weight must be on CUDA"),
        };
        let cos_cuda = match &*cos_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: cos must be on CUDA"),
        };
        let sin_cuda = match &*sin_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: sin must be on CUDA"),
        };
        let qo_cuda = match &*qo_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: q_out must be on CUDA"),
        };
        let ko_cuda = match &*ko_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: k_out must be on CUDA"),
        };

        let stream = qr_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let qr_slice = qr_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qr_layout.start_offset()..);
        let kr_slice = kr_cuda
            .as_cuda_slice::<bf16>()?
            .slice(kr_layout.start_offset()..);
        let qw_slice = qw_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qw_layout.start_offset()..);
        let kw_slice = kw_cuda
            .as_cuda_slice::<bf16>()?
            .slice(kw_layout.start_offset()..);
        let cos_slice = cos_cuda
            .as_cuda_slice::<f32>()?
            .slice(cos_layout.start_offset()..);
        let sin_slice = sin_cuda
            .as_cuda_slice::<f32>()?
            .slice(sin_layout.start_offset()..);
        let qo_slice = qo_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qo_layout.start_offset()..);
        let ko_slice = ko_cuda
            .as_cuda_slice::<bf16>()?
            .slice(ko_layout.start_offset()..);

        unsafe {
            let (qr_ptr, _g1) = qr_slice.device_ptr(&stream);
            let (kr_ptr, _g2) = kr_slice.device_ptr(&stream);
            let (qw_ptr, _g3) = qw_slice.device_ptr(&stream);
            let (kw_ptr, _g4) = kw_slice.device_ptr(&stream);
            let (cos_ptr, _g5) = cos_slice.device_ptr(&stream);
            let (sin_ptr, _g6) = sin_slice.device_ptr(&stream);
            let (qo_ptr, _g7) = qo_slice.device_ptr(&stream);
            let (ko_ptr, _g8) = ko_slice.device_ptr(&stream);

            let status = if let Some(gate_out) = gate_out.as_ref() {
                let (go_storage, go_layout) = gate_out.storage_and_layout();
                let go_cuda = match &*go_storage {
                    candle_core::Storage::Cuda(c) => c,
                    _ => candle_core::bail!("kiln-rmsnorm-kernel: gate_out must be on CUDA"),
                };
                let go_slice = go_cuda
                    .as_cuda_slice::<bf16>()?
                    .slice(go_layout.start_offset()..);
                let (go_ptr, _g9) = go_slice.device_ptr(&stream);
                kiln_attn_decode_qkv_split_qk_norm_rope_bf16(
                    qr_ptr as *const _,
                    kr_ptr as *const _,
                    qw_ptr as *const _,
                    kw_ptr as *const _,
                    cos_ptr as *const f32,
                    sin_ptr as *const f32,
                    qo_ptr as *mut _,
                    ko_ptr as *mut _,
                    go_ptr as *mut _,
                    batch as i32,
                    q_heads as i32,
                    k_heads as i32,
                    head_dim as i32,
                    rotary_dim as i32,
                    1,
                    eps,
                    raw_stream,
                )
            } else {
                kiln_attn_decode_qkv_split_qk_norm_rope_bf16(
                    qr_ptr as *const _,
                    kr_ptr as *const _,
                    qw_ptr as *const _,
                    kw_ptr as *const _,
                    cos_ptr as *const f32,
                    sin_ptr as *const f32,
                    qo_ptr as *mut _,
                    ko_ptr as *mut _,
                    core::ptr::null_mut(),
                    batch as i32,
                    q_heads as i32,
                    k_heads as i32,
                    head_dim as i32,
                    rotary_dim as i32,
                    0,
                    eps,
                    raw_stream,
                )
            };
            if status != 0 {
                candle_core::bail!(
                    "kiln_attn_decode_qkv_split_qk_norm_rope_bf16 failed with status {status}"
                );
            }
        }
    }

    Ok((q_out, k_out, gate_out))
}

/// Whether the fused MLP `silu(gate) * up` kernel is available.
///
/// Supports matching CUDA bf16 contiguous tensors. The operation is
/// forward-only; callers that need autograd should keep using the Candle path.
pub fn supports_mlp_silu_mul(gate: &Tensor, up: &Tensor) -> bool {
    matches!(gate.device(), Device::Cuda(_))
        && matches!(up.device(), Device::Cuda(_))
        && gate.dtype() == DType::BF16
        && up.dtype() == DType::BF16
        && gate.is_contiguous()
        && up.is_contiguous()
        && gate.dims() == up.dims()
        && gate.elem_count() <= i64::MAX as usize
}

/// Run fused bf16 `silu(gate) * up`.
///
/// This matches the CUDA-safe SiLU used in `kiln-model::forward`:
/// `gate / (1 + exp(-gate))`, multiplied by `up`, and cast to bf16.
pub fn fused_mlp_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    if !supports_mlp_silu_mul(gate, up) {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: mlp_silu_mul unsupported shapes gate={:?} up={:?} dtypes=({:?},{:?})",
            gate.shape(),
            up.shape(),
            gate.dtype(),
            up.dtype()
        );
    }

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let out = unsafe { Tensor::empty(gate.dims(), DType::BF16, gate.device())? };
    let elems = gate.elem_count();
    if elems == 0 {
        return Ok(out);
    }

    {
        let (gate_storage, gate_layout) = gate.storage_and_layout();
        let (up_storage, up_layout) = up.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let gate_cuda = match &*gate_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: mlp_silu_mul gate must be on CUDA"),
        };
        let up_cuda = match &*up_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: mlp_silu_mul up must be on CUDA"),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: mlp_silu_mul out must be on CUDA"),
        };

        let stream = gate_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let gate_slice = gate_cuda
            .as_cuda_slice::<bf16>()?
            .slice(gate_layout.start_offset()..);
        let up_slice = up_cuda
            .as_cuda_slice::<bf16>()?
            .slice(up_layout.start_offset()..);
        let out_slice = out_cuda
            .as_cuda_slice::<bf16>()?
            .slice(out_layout.start_offset()..);

        unsafe {
            let (gate_ptr, _g1) = gate_slice.device_ptr(&stream);
            let (up_ptr, _g2) = up_slice.device_ptr(&stream);
            let (out_ptr, _g3) = out_slice.device_ptr(&stream);
            let status = kiln_fused_mlp_silu_mul_bf16(
                gate_ptr as *const _,
                up_ptr as *const _,
                out_ptr as *mut _,
                elems as i64,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!("kiln_fused_mlp_silu_mul_bf16 failed with status {status}");
            }
        }
    }

    Ok(out)
}

/// Whether the packed fused MLP `silu(gate) * up` kernel can handle this
/// `gate_up_packed` tensor. The kernel expects a contiguous BF16 tensor
/// whose last dim is `2 * cols` — it will read gate from the first `cols`
/// of each row and up from the next `cols`. `cols` is supplied separately
/// (the model knows the intermediate width via the FFN weight shape) so
/// the caller can produce a `[B, T, 2*intermediate]` packed tensor and
/// project it into a `[B, T, intermediate]` output in one launch.
pub fn supports_mlp_silu_mul_packed(gate_up_packed: &Tensor, cols: usize) -> bool {
    let dims = gate_up_packed.dims();
    matches!(gate_up_packed.device(), Device::Cuda(_))
        && gate_up_packed.dtype() == DType::BF16
        && gate_up_packed.is_contiguous()
        && !dims.is_empty()
        && dims[dims.len() - 1] == cols * 2
        && cols > 0
        && gate_up_packed.elem_count() <= i64::MAX as usize
}

/// Run fused bf16 `silu(gate_packed[..., :cols]) * gate_packed[..., cols:2*cols]`.
///
/// Output is BF16 with the same leading shape as `gate_up_packed` and a
/// trailing dim of `cols`. The kernel reads each output element's gate and
/// up operands from adjacent halves of the same packed row, avoiding the
/// explicit `.contiguous()` copy required when splitting the packed matmul
/// output via `Tensor::narrow` first.
pub fn fused_mlp_silu_mul_packed(gate_up_packed: &Tensor, cols: usize) -> Result<Tensor> {
    if !supports_mlp_silu_mul_packed(gate_up_packed, cols) {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: mlp_silu_mul_packed unsupported gate_up shape={:?} dtype={:?} cols={cols}",
            gate_up_packed.shape(),
            gate_up_packed.dtype(),
        );
    }
    let dims = gate_up_packed.dims();
    let rows: usize = dims[..dims.len() - 1].iter().product();
    let mut out_dims: Vec<usize> = dims[..dims.len() - 1].to_vec();
    out_dims.push(cols);
    let out = unsafe { Tensor::empty(out_dims.as_slice(), DType::BF16, gate_up_packed.device())? };
    if rows == 0 {
        return Ok(out);
    }

    {
        let (gate_up_storage, gate_up_layout) = gate_up_packed.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();
        let gate_up_cuda = match &*gate_up_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!(
                "kiln-rmsnorm-kernel: mlp_silu_mul_packed gate_up must be CUDA"
            ),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!(
                "kiln-rmsnorm-kernel: mlp_silu_mul_packed out must be CUDA"
            ),
        };

        let stream = gate_up_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let gate_up_slice = gate_up_cuda
            .as_cuda_slice::<bf16>()?
            .slice(gate_up_layout.start_offset()..);
        let out_slice = out_cuda
            .as_cuda_slice::<bf16>()?
            .slice(out_layout.start_offset()..);

        unsafe {
            let (gate_up_ptr, _g1) = gate_up_slice.device_ptr(&stream);
            let (out_ptr, _g2) = out_slice.device_ptr(&stream);
            let status = kiln_fused_mlp_silu_mul_packed_bf16(
                gate_up_ptr as *const _,
                out_ptr as *mut _,
                rows as i64,
                cols as i64,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!(
                    "kiln_fused_mlp_silu_mul_packed_bf16 failed with status {status}"
                );
            }
        }
    }

    Ok(out)
}

pub fn supports_sigmoid_mul(x: &Tensor, gate: &Tensor) -> bool {
    matches!(x.device(), Device::Cuda(_))
        && matches!(gate.device(), Device::Cuda(_))
        && x.dtype() == DType::BF16
        && gate.dtype() == DType::BF16
        && x.is_contiguous()
        && gate.is_contiguous()
        && x.dims() == gate.dims()
        && x.elem_count() <= i64::MAX as usize
}

/// Run fused bf16 `x * sigmoid(gate)`.
pub fn fused_sigmoid_mul(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    if !supports_sigmoid_mul(x, gate) {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: sigmoid_mul unsupported shapes x={:?} gate={:?} dtypes=({:?},{:?})",
            x.shape(),
            gate.shape(),
            x.dtype(),
            gate.dtype()
        );
    }

    let x = x.contiguous()?;
    let gate = gate.contiguous()?;
    let out = unsafe { Tensor::empty(x.dims(), DType::BF16, x.device())? };
    let elems = x.elem_count();
    if elems == 0 {
        return Ok(out);
    }

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (gate_storage, gate_layout) = gate.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: sigmoid_mul x must be on CUDA"),
        };
        let gate_cuda = match &*gate_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: sigmoid_mul gate must be on CUDA"),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: sigmoid_mul out must be on CUDA"),
        };

        let stream = x_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let x_slice = x_cuda
            .as_cuda_slice::<bf16>()?
            .slice(x_layout.start_offset()..);
        let gate_slice = gate_cuda
            .as_cuda_slice::<bf16>()?
            .slice(gate_layout.start_offset()..);
        let out_slice = out_cuda
            .as_cuda_slice::<bf16>()?
            .slice(out_layout.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (gate_ptr, _g2) = gate_slice.device_ptr(&stream);
            let (out_ptr, _g3) = out_slice.device_ptr(&stream);
            let status = kiln_fused_sigmoid_mul_bf16(
                x_ptr as *const _,
                gate_ptr as *const _,
                out_ptr as *mut _,
                elems as i64,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!("kiln_fused_sigmoid_mul_bf16 failed with status {status}");
            }
        }
    }

    Ok(out)
}

/// Storage-level variant of [`fused_sigmoid_mul`] for CUDA custom ops.
///
/// Computes `out = x * sigmoid(gate)` for matching contiguous BF16 CUDA
/// tensors. `out` may alias neither input.
pub fn fused_sigmoid_mul_storage(
    out_cuda: &CudaStorage,
    out_layout: &Layout,
    x_cuda: &CudaStorage,
    x_layout: &Layout,
    gate_cuda: &CudaStorage,
    gate_layout: &Layout,
) -> Result<()> {
    let out_dims = out_layout.dims();
    let x_dims = x_layout.dims();
    let gate_dims = gate_layout.dims();
    if out_dims != x_dims || out_dims != gate_dims {
        candle_core::bail!(
            "fused_sigmoid_mul_storage: shape mismatch out={out_dims:?} x={x_dims:?} gate={gate_dims:?}"
        );
    }
    if out_cuda.dtype() != DType::BF16
        || x_cuda.dtype() != DType::BF16
        || gate_cuda.dtype() != DType::BF16
    {
        candle_core::bail!(
            "fused_sigmoid_mul_storage: expected BF16 tensors, got out={:?} x={:?} gate={:?}",
            out_cuda.dtype(),
            x_cuda.dtype(),
            gate_cuda.dtype()
        );
    }
    if !out_layout.is_contiguous() || !x_layout.is_contiguous() || !gate_layout.is_contiguous() {
        candle_core::bail!("fused_sigmoid_mul_storage: tensors must be contiguous");
    }
    let elems: usize = out_dims.iter().product();
    if elems > i64::MAX as usize {
        candle_core::bail!("fused_sigmoid_mul_storage: element count exceeds i64");
    }
    if elems == 0 {
        return Ok(());
    }

    let stream = x_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let x_slice = x_cuda
        .as_cuda_slice::<bf16>()?
        .slice(x_layout.start_offset()..);
    let gate_slice = gate_cuda
        .as_cuda_slice::<bf16>()?
        .slice(gate_layout.start_offset()..);
    let out_slice = out_cuda
        .as_cuda_slice::<bf16>()?
        .slice(out_layout.start_offset()..);
    let status = unsafe {
        let (x_ptr, _x_guard) = x_slice.device_ptr(&stream);
        let (gate_ptr, _gate_guard) = gate_slice.device_ptr(&stream);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        kiln_fused_sigmoid_mul_bf16(
            x_ptr as *const _,
            gate_ptr as *const _,
            out_ptr as *mut _,
            elems as i64,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_fused_sigmoid_mul_bf16 failed with status {status}");
    }
    Ok(())
}

/// Whether the fused forward-only LoRA delta/add decode kernels support this shape.
///
/// Supports BF16 CUDA tensors for single-token decode rows:
/// `base=[batch,1,out]`, `x=[batch,1,in]`, `a=[rank,in]`, `b=[out,rank]`.
/// Callers that need autograd must decline before invoking this helper.
pub fn supports_lora_decode_add(base: &Tensor, x: &Tensor, a: &Tensor, b: &Tensor) -> bool {
    let Ok((batch, one, out_dim)) = base.dims3() else {
        return false;
    };
    let Ok((x_batch, x_one, in_dim)) = x.dims3() else {
        return false;
    };
    let Ok((rank, a_in_dim)) = a.dims2() else {
        return false;
    };
    let Ok((b_out_dim, b_rank)) = b.dims2() else {
        return false;
    };
    matches!(base.device(), Device::Cuda(_))
        && matches!(x.device(), Device::Cuda(_))
        && matches!(a.device(), Device::Cuda(_))
        && matches!(b.device(), Device::Cuda(_))
        && base.dtype() == DType::BF16
        && x.dtype() == DType::BF16
        && a.dtype() == DType::BF16
        && b.dtype() == DType::BF16
        && base.is_contiguous()
        && x.is_contiguous()
        && a.is_contiguous()
        && b.is_contiguous()
        && batch == x_batch
        && one == 1
        && x_one == 1
        && rank == b_rank
        && in_dim == a_in_dim
        && out_dim == b_out_dim
        && batch > 0
        && in_dim > 0
        && out_dim > 0
        && rank > 0
        && rank <= 64
        && batch <= i32::MAX as usize
        && in_dim <= i32::MAX as usize
        && out_dim <= i32::MAX as usize
        && rank <= i32::MAX as usize
}

fn matmul_f32_bf16w_dims(lhs: &Tensor, weight: &Tensor) -> Result<(usize, usize, usize)> {
    let Ok((m, k)) = lhs.dims2() else {
        candle_core::bail!(
            "matmul_f32_bf16w: lhs must be rank-2 [rows,in], got {:?}",
            lhs.shape()
        );
    };
    let Ok((w_k, n)) = weight.dims2() else {
        candle_core::bail!(
            "matmul_f32_bf16w: weight must be rank-2 [in,out], got {:?}",
            weight.shape()
        );
    };
    if k != w_k {
        candle_core::bail!(
            "matmul_f32_bf16w: inner dim mismatch lhs={:?} weight={:?}",
            lhs.shape(),
            weight.shape()
        );
    }
    if m > i32::MAX as usize || k > i32::MAX as usize || n > i32::MAX as usize {
        candle_core::bail!("matmul_f32_bf16w: dimensions exceed i32 kernel envelope");
    }
    Ok((m, k, n))
}

pub fn supports_matmul_f32_bf16w(lhs: &Tensor, weight: &Tensor) -> bool {
    matches!(lhs.device(), Device::Cuda(_))
        && matches!(weight.device(), Device::Cuda(_))
        && lhs.dtype() == DType::F32
        && weight.dtype() == DType::BF16
        && matmul_f32_bf16w_dims(lhs, weight).is_ok()
}

pub fn matmul_f32_bf16w(lhs: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if !supports_matmul_f32_bf16w(lhs, weight) {
        candle_core::bail!(
            "matmul_f32_bf16w unsupported lhs={:?} weight={:?} dtypes=({:?},{:?})",
            lhs.shape(),
            weight.shape(),
            lhs.dtype(),
            weight.dtype()
        );
    }
    let (m, _k, n) = matmul_f32_bf16w_dims(lhs, weight)?;
    let lhs = lhs.contiguous()?;
    let weight = weight.contiguous()?;
    if m == 0 {
        return Tensor::zeros((m, n), DType::F32, lhs.device());
    }
    let weight_f32 = weight.to_dtype(DType::F32)?;
    lhs.matmul(&weight_f32)
}

pub fn matmul_f32_bf16w_bwd_lhs(grad_out: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (m, _n, k) = {
        let Ok((m, out_dim)) = grad_out.dims2() else {
            candle_core::bail!(
                "matmul_f32_bf16w_bwd_lhs: grad must be rank-2 [rows,out], got {:?}",
                grad_out.shape()
            );
        };
        let Ok((in_dim, weight_out)) = weight.dims2() else {
            candle_core::bail!(
                "matmul_f32_bf16w_bwd_lhs: weight must be rank-2 [in,out], got {:?}",
                weight.shape()
            );
        };
        if out_dim != weight_out {
            candle_core::bail!(
                "matmul_f32_bf16w_bwd_lhs: out dim mismatch grad={:?} weight={:?}",
                grad_out.shape(),
                weight.shape()
            );
        }
        if m > i32::MAX as usize || in_dim > i32::MAX as usize || out_dim > i32::MAX as usize {
            candle_core::bail!("matmul_f32_bf16w_bwd_lhs: dimensions exceed i32 kernel envelope");
        }
        (m, out_dim, in_dim)
    };
    if !matches!(grad_out.device(), Device::Cuda(_)) || !matches!(weight.device(), Device::Cuda(_))
    {
        candle_core::bail!("matmul_f32_bf16w_bwd_lhs requires CUDA tensors");
    }
    if grad_out.dtype() != DType::F32 || weight.dtype() != DType::BF16 {
        candle_core::bail!(
            "matmul_f32_bf16w_bwd_lhs expects F32 grad and BF16 weight, got {:?}/{:?}",
            grad_out.dtype(),
            weight.dtype()
        );
    }

    let grad_out = grad_out.contiguous()?;
    let weight = weight.contiguous()?;
    if m == 0 {
        return Tensor::zeros((m, k), DType::F32, grad_out.device());
    }
    let weight_t = weight.to_dtype(DType::F32)?.t()?.contiguous()?;
    grad_out.matmul(&weight_t)
}

fn optimizer_tensors_supported(tensors: &[&Tensor]) -> bool {
    let Some(first) = tensors.first() else {
        return false;
    };
    matches!(first.device(), Device::Cuda(_))
        && matches!(first.dtype(), DType::F32 | DType::BF16)
        && first.is_contiguous()
        && tensors.iter().all(|tensor| {
            matches!(tensor.device(), Device::Cuda(_))
                && tensor.dtype() == first.dtype()
                && tensor.shape().elem_count() == first.shape().elem_count()
                && tensor.is_contiguous()
        })
}

pub fn supports_optimizer_step(tensors: &[&Tensor]) -> bool {
    optimizer_tensors_supported(tensors)
}

pub fn sgd_step_inplace(param: &Tensor, grad: &Tensor, lr: f32) -> Result<()> {
    if !optimizer_tensors_supported(&[param, grad]) {
        candle_core::bail!(
            "sgd_step_inplace unsupported param={:?} grad={:?} dtypes=({:?},{:?})",
            param.shape(),
            grad.shape(),
            param.dtype(),
            grad.dtype()
        );
    }
    let n = param.shape().elem_count() as i64;
    if n == 0 {
        return Ok(());
    }

    let (param_storage, param_layout) = param.storage_and_layout();
    let (grad_storage, grad_layout) = grad.storage_and_layout();
    let param_cuda = match &*param_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("sgd_step_inplace: param must be CUDA"),
    };
    let grad_cuda = match &*grad_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("sgd_step_inplace: grad must be CUDA"),
    };
    let stream = param_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = match param.dtype() {
        DType::F32 => {
            let p = param_cuda
                .as_cuda_slice::<f32>()?
                .slice(param_layout.start_offset()..);
            let g = grad_cuda
                .as_cuda_slice::<f32>()?
                .slice(grad_layout.start_offset()..);
            unsafe {
                let (p_ptr, _p_guard) = p.device_ptr(&stream);
                let (g_ptr, _g_guard) = g.device_ptr(&stream);
                kiln_sgd_step_f32(p_ptr as *mut f32, g_ptr as *const f32, lr, n, raw_stream)
            }
        }
        DType::BF16 => {
            let p = param_cuda
                .as_cuda_slice::<bf16>()?
                .slice(param_layout.start_offset()..);
            let g = grad_cuda
                .as_cuda_slice::<bf16>()?
                .slice(grad_layout.start_offset()..);
            unsafe {
                let (p_ptr, _p_guard) = p.device_ptr(&stream);
                let (g_ptr, _g_guard) = g.device_ptr(&stream);
                kiln_sgd_step_bf16(
                    p_ptr as *mut core::ffi::c_void,
                    g_ptr as *const core::ffi::c_void,
                    lr,
                    n,
                    raw_stream,
                )
            }
        }
        dtype => candle_core::bail!("sgd_step_inplace unsupported dtype {dtype:?}"),
    };
    if status != 0 {
        candle_core::bail!("kiln_sgd_step failed with status {status}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn adamw_step_inplace(
    param: &Tensor,
    grad: &Tensor,
    first_moment: &Tensor,
    second_moment: &Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<()> {
    if step == 0 {
        candle_core::bail!("adamw_step_inplace: step must be >= 1");
    }
    if !optimizer_tensors_supported(&[param, grad, first_moment, second_moment]) {
        candle_core::bail!(
            "adamw_step_inplace unsupported param={:?} grad={:?} m={:?} v={:?} dtypes=({:?},{:?},{:?},{:?})",
            param.shape(),
            grad.shape(),
            first_moment.shape(),
            second_moment.shape(),
            param.dtype(),
            grad.dtype(),
            first_moment.dtype(),
            second_moment.dtype()
        );
    }
    let n = param.shape().elem_count() as i64;
    if n == 0 {
        return Ok(());
    }
    let bias_correction1 = (1.0f32 - beta1.powi(step as i32)).max(1e-20);
    let bias_correction2 = (1.0f32 - beta2.powi(step as i32)).max(1e-20);

    let (p_storage, p_layout) = param.storage_and_layout();
    let (g_storage, g_layout) = grad.storage_and_layout();
    let (m_storage, m_layout) = first_moment.storage_and_layout();
    let (v_storage, v_layout) = second_moment.storage_and_layout();
    let p_cuda = match &*p_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("adamw_step_inplace: param must be CUDA"),
    };
    let g_cuda = match &*g_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("adamw_step_inplace: grad must be CUDA"),
    };
    let m_cuda = match &*m_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("adamw_step_inplace: first_moment must be CUDA"),
    };
    let v_cuda = match &*v_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("adamw_step_inplace: second_moment must be CUDA"),
    };
    let stream = p_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

    let status = match param.dtype() {
        DType::F32 => {
            let p = p_cuda
                .as_cuda_slice::<f32>()?
                .slice(p_layout.start_offset()..);
            let g = g_cuda
                .as_cuda_slice::<f32>()?
                .slice(g_layout.start_offset()..);
            let m = m_cuda
                .as_cuda_slice::<f32>()?
                .slice(m_layout.start_offset()..);
            let v = v_cuda
                .as_cuda_slice::<f32>()?
                .slice(v_layout.start_offset()..);
            unsafe {
                let (p_ptr, _p_guard) = p.device_ptr(&stream);
                let (g_ptr, _g_guard) = g.device_ptr(&stream);
                let (m_ptr, _m_guard) = m.device_ptr(&stream);
                let (v_ptr, _v_guard) = v.device_ptr(&stream);
                kiln_adamw_step_f32(
                    p_ptr as *mut f32,
                    g_ptr as *const f32,
                    m_ptr as *mut f32,
                    v_ptr as *mut f32,
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
            }
        }
        DType::BF16 => {
            let p = p_cuda
                .as_cuda_slice::<bf16>()?
                .slice(p_layout.start_offset()..);
            let g = g_cuda
                .as_cuda_slice::<bf16>()?
                .slice(g_layout.start_offset()..);
            let m = m_cuda
                .as_cuda_slice::<bf16>()?
                .slice(m_layout.start_offset()..);
            let v = v_cuda
                .as_cuda_slice::<bf16>()?
                .slice(v_layout.start_offset()..);
            unsafe {
                let (p_ptr, _p_guard) = p.device_ptr(&stream);
                let (g_ptr, _g_guard) = g.device_ptr(&stream);
                let (m_ptr, _m_guard) = m.device_ptr(&stream);
                let (v_ptr, _v_guard) = v.device_ptr(&stream);
                kiln_adamw_step_bf16(
                    p_ptr as *mut core::ffi::c_void,
                    g_ptr as *const core::ffi::c_void,
                    m_ptr as *mut core::ffi::c_void,
                    v_ptr as *mut core::ffi::c_void,
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
            }
        }
        dtype => candle_core::bail!("adamw_step_inplace unsupported dtype {dtype:?}"),
    };
    if status != 0 {
        candle_core::bail!("kiln_adamw_step failed with status {status}");
    }
    Ok(())
}

/// Run fused forward-only LoRA delta/add for decode.
pub fn lora_decode_add(
    base: &Tensor,
    x: &Tensor,
    a: &Tensor,
    b: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    if !supports_lora_decode_add(base, x, a, b) {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: lora_decode_add unsupported base={:?} x={:?} a={:?} b={:?} dtypes=({:?},{:?},{:?},{:?})",
            base.shape(),
            x.shape(),
            a.shape(),
            b.shape(),
            base.dtype(),
            x.dtype(),
            a.dtype(),
            b.dtype()
        );
    }

    let (batch, _, out_dim) = base.dims3()?;
    let (_, _, in_dim) = x.dims3()?;
    let (rank, _) = a.dims2()?;
    let base = base.contiguous()?;
    let x = x.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let hidden = unsafe { Tensor::empty((batch, rank), DType::F32, base.device())? };
    let out = unsafe { Tensor::empty(base.dims(), DType::BF16, base.device())? };

    {
        let (base_storage, base_layout) = base.storage_and_layout();
        let (x_storage, x_layout) = x.storage_and_layout();
        let (a_storage, a_layout) = a.storage_and_layout();
        let (b_storage, b_layout) = b.storage_and_layout();
        let (hidden_storage, hidden_layout) = hidden.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let base_cuda = match &*base_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: lora base must be on CUDA"),
        };
        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: lora x must be on CUDA"),
        };
        let a_cuda = match &*a_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: lora A must be on CUDA"),
        };
        let b_cuda = match &*b_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: lora B must be on CUDA"),
        };
        let hidden_cuda = match &*hidden_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: lora hidden must be on CUDA"),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: lora out must be on CUDA"),
        };

        let stream = base_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let base_slice = base_cuda
            .as_cuda_slice::<bf16>()?
            .slice(base_layout.start_offset()..);
        let x_slice = x_cuda
            .as_cuda_slice::<bf16>()?
            .slice(x_layout.start_offset()..);
        let a_slice = a_cuda
            .as_cuda_slice::<bf16>()?
            .slice(a_layout.start_offset()..);
        let b_slice = b_cuda
            .as_cuda_slice::<bf16>()?
            .slice(b_layout.start_offset()..);
        let hidden_slice = hidden_cuda
            .as_cuda_slice::<f32>()?
            .slice(hidden_layout.start_offset()..);
        let out_slice = out_cuda
            .as_cuda_slice::<bf16>()?
            .slice(out_layout.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (a_ptr, _g2) = a_slice.device_ptr(&stream);
            let (hidden_ptr, _g3) = hidden_slice.device_ptr(&stream);
            let status = kiln_lora_decode_hidden_bf16(
                x_ptr as *const _,
                a_ptr as *const _,
                hidden_ptr as *mut f32,
                batch as i32,
                in_dim as i32,
                rank as i32,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!("kiln_lora_decode_hidden_bf16 failed with status {status}");
            }

            let (base_ptr, _g4) = base_slice.device_ptr(&stream);
            let (b_ptr, _g5) = b_slice.device_ptr(&stream);
            let (out_ptr, _g6) = out_slice.device_ptr(&stream);
            let status = kiln_lora_decode_add_bf16(
                base_ptr as *const _,
                hidden_ptr as *const f32,
                b_ptr as *const _,
                out_ptr as *mut _,
                scale,
                batch as i32,
                out_dim as i32,
                rank as i32,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!("kiln_lora_decode_add_bf16 failed with status {status}");
            }
        }
    }

    Ok(out)
}

/// Add a F32 LoRA delta into an already-computed projection output.
///
/// Shapes are row-major: `base=[rows,out]`, `hidden=[rows,rank]`,
/// `b=[out,rank]`. `base` is mutated in place and returned by the caller.
pub fn lora_add_inplace_f32(base: &Tensor, hidden: &Tensor, b: &Tensor, scale: f32) -> Result<()> {
    let Ok((rows, out_dim)) = base.dims2() else {
        candle_core::bail!(
            "lora_add_inplace_f32: base must be rank-2, got {:?}",
            base.shape()
        );
    };
    let Ok((hidden_rows, rank)) = hidden.dims2() else {
        candle_core::bail!(
            "lora_add_inplace_f32: hidden must be rank-2, got {:?}",
            hidden.shape()
        );
    };
    let Ok((b_out, b_rank)) = b.dims2() else {
        candle_core::bail!(
            "lora_add_inplace_f32: B must be rank-2, got {:?}",
            b.shape()
        );
    };
    if rows != hidden_rows || out_dim != b_out || rank != b_rank {
        candle_core::bail!(
            "lora_add_inplace_f32: shape mismatch base={:?} hidden={:?} b={:?}",
            base.shape(),
            hidden.shape(),
            b.shape()
        );
    }
    if base.dtype() != DType::F32 || hidden.dtype() != DType::F32 || b.dtype() != DType::F32 {
        candle_core::bail!(
            "lora_add_inplace_f32: expected F32 tensors, got base={:?} hidden={:?} b={:?}",
            base.dtype(),
            hidden.dtype(),
            b.dtype()
        );
    }
    if !base.is_contiguous() || !hidden.is_contiguous() || !b.is_contiguous() {
        candle_core::bail!("lora_add_inplace_f32: tensors must be contiguous");
    }
    if rows > i32::MAX as usize || out_dim > i32::MAX as usize || rank > i32::MAX as usize {
        candle_core::bail!("lora_add_inplace_f32: dimensions exceed i32 kernel envelope");
    }
    if rows == 0 || out_dim == 0 || rank == 0 {
        return Ok(());
    }

    let (base_storage, base_layout) = base.storage_and_layout();
    let (hidden_storage, hidden_layout) = hidden.storage_and_layout();
    let (b_storage, b_layout) = b.storage_and_layout();
    let base_cuda = match &*base_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("lora_add_inplace_f32: base must be CUDA"),
    };
    let hidden_cuda = match &*hidden_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("lora_add_inplace_f32: hidden must be CUDA"),
    };
    let b_cuda = match &*b_storage {
        candle_core::Storage::Cuda(c) => c,
        _ => candle_core::bail!("lora_add_inplace_f32: B must be CUDA"),
    };
    lora_add_inplace_f32_storage(
        base_cuda,
        base_layout,
        hidden_cuda,
        hidden_layout,
        b_cuda,
        b_layout,
        scale,
    )
}

/// Storage-level variant of [`lora_add_inplace_f32`] for CUDA custom ops.
///
/// `base` is mutated in-place. Layouts must describe contiguous rank-2 F32
/// tensors with shapes `base=[rows,out]`, `hidden=[rows,rank]`,
/// `b=[out,rank]`.
pub fn lora_add_inplace_f32_storage(
    base_cuda: &CudaStorage,
    base_layout: &Layout,
    hidden_cuda: &CudaStorage,
    hidden_layout: &Layout,
    b_cuda: &CudaStorage,
    b_layout: &Layout,
    scale: f32,
) -> Result<()> {
    let base_dims = base_layout.dims();
    let hidden_dims = hidden_layout.dims();
    let b_dims = b_layout.dims();
    if base_dims.len() != 2 {
        candle_core::bail!("lora_add_inplace_f32_storage: base must be rank-2, got {base_dims:?}");
    }
    if hidden_dims.len() != 2 {
        candle_core::bail!(
            "lora_add_inplace_f32_storage: hidden must be rank-2, got {hidden_dims:?}"
        );
    }
    if b_dims.len() != 2 {
        candle_core::bail!("lora_add_inplace_f32_storage: B must be rank-2, got {b_dims:?}");
    }
    let (rows, out_dim) = (base_dims[0], base_dims[1]);
    let (hidden_rows, rank) = (hidden_dims[0], hidden_dims[1]);
    let (b_out, b_rank) = (b_dims[0], b_dims[1]);
    if rows != hidden_rows || out_dim != b_out || rank != b_rank {
        candle_core::bail!(
            "lora_add_inplace_f32_storage: shape mismatch base={base_dims:?} hidden={hidden_dims:?} b={b_dims:?}"
        );
    }
    if base_cuda.dtype() != DType::F32
        || hidden_cuda.dtype() != DType::F32
        || b_cuda.dtype() != DType::F32
    {
        candle_core::bail!(
            "lora_add_inplace_f32_storage: expected F32 tensors, got base={:?} hidden={:?} b={:?}",
            base_cuda.dtype(),
            hidden_cuda.dtype(),
            b_cuda.dtype()
        );
    }
    if !base_layout.is_contiguous() || !hidden_layout.is_contiguous() || !b_layout.is_contiguous() {
        candle_core::bail!("lora_add_inplace_f32_storage: tensors must be contiguous");
    }
    if rows > i32::MAX as usize || out_dim > i32::MAX as usize || rank > i32::MAX as usize {
        candle_core::bail!("lora_add_inplace_f32_storage: dimensions exceed i32 kernel envelope");
    }
    if rows == 0 || out_dim == 0 || rank == 0 {
        return Ok(());
    }

    let stream = base_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let base_slice = base_cuda
        .as_cuda_slice::<f32>()?
        .slice(base_layout.start_offset()..);
    let hidden_slice = hidden_cuda
        .as_cuda_slice::<f32>()?
        .slice(hidden_layout.start_offset()..);
    let b_slice = b_cuda
        .as_cuda_slice::<f32>()?
        .slice(b_layout.start_offset()..);

    let status = unsafe {
        let (base_ptr, _base_guard) = base_slice.device_ptr(&stream);
        let (hidden_ptr, _hidden_guard) = hidden_slice.device_ptr(&stream);
        let (b_ptr, _b_guard) = b_slice.device_ptr(&stream);
        kiln_lora_add_inplace_f32(
            base_ptr as *mut f32,
            hidden_ptr as *const f32,
            b_ptr as *const f32,
            scale,
            rows as i32,
            out_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_lora_add_inplace_f32 failed with status {status}");
    }
    Ok(())
}

/// Storage-level BF16 LoRA add for CUDA custom ops.
///
/// Computes `out = base + scale * hidden @ b.T`, where
/// `base=[rows,out]` and `b=[out,rank]` are BF16, `hidden=[rows,rank]`
/// is F32, and `out=[rows,out]` is BF16.
pub fn lora_add_bf16_storage(
    out_cuda: &CudaStorage,
    out_layout: &Layout,
    base_cuda: &CudaStorage,
    base_layout: &Layout,
    hidden_cuda: &CudaStorage,
    hidden_layout: &Layout,
    b_cuda: &CudaStorage,
    b_layout: &Layout,
    scale: f32,
) -> Result<()> {
    let out_dims = out_layout.dims();
    let base_dims = base_layout.dims();
    let hidden_dims = hidden_layout.dims();
    let b_dims = b_layout.dims();
    if out_dims.len() != 2 || base_dims.len() != 2 {
        candle_core::bail!(
            "lora_add_bf16_storage: out/base must be rank-2, got out={out_dims:?} base={base_dims:?}"
        );
    }
    if hidden_dims.len() != 2 || b_dims.len() != 2 {
        candle_core::bail!(
            "lora_add_bf16_storage: hidden/B must be rank-2, got hidden={hidden_dims:?} b={b_dims:?}"
        );
    }
    let (rows, out_dim) = (base_dims[0], base_dims[1]);
    let (hidden_rows, rank) = (hidden_dims[0], hidden_dims[1]);
    let (b_out, b_rank) = (b_dims[0], b_dims[1]);
    if out_dims != base_dims || rows != hidden_rows || out_dim != b_out || rank != b_rank {
        candle_core::bail!(
            "lora_add_bf16_storage: shape mismatch out={out_dims:?} base={base_dims:?} hidden={hidden_dims:?} b={b_dims:?}"
        );
    }
    if out_cuda.dtype() != DType::BF16
        || base_cuda.dtype() != DType::BF16
        || hidden_cuda.dtype() != DType::F32
        || b_cuda.dtype() != DType::BF16
    {
        candle_core::bail!(
            "lora_add_bf16_storage: expected out/base BF16, hidden F32, B BF16; got out={:?} base={:?} hidden={:?} b={:?}",
            out_cuda.dtype(),
            base_cuda.dtype(),
            hidden_cuda.dtype(),
            b_cuda.dtype()
        );
    }
    if !out_layout.is_contiguous()
        || !base_layout.is_contiguous()
        || !hidden_layout.is_contiguous()
        || !b_layout.is_contiguous()
    {
        candle_core::bail!("lora_add_bf16_storage: tensors must be contiguous");
    }
    if rows > i32::MAX as usize || out_dim > i32::MAX as usize || rank > i32::MAX as usize {
        candle_core::bail!("lora_add_bf16_storage: dimensions exceed i32 kernel envelope");
    }
    if rows == 0 || out_dim == 0 || rank == 0 {
        return Ok(());
    }

    let stream = base_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let out_slice = out_cuda
        .as_cuda_slice::<bf16>()?
        .slice(out_layout.start_offset()..);
    let base_slice = base_cuda
        .as_cuda_slice::<bf16>()?
        .slice(base_layout.start_offset()..);
    let hidden_slice = hidden_cuda
        .as_cuda_slice::<f32>()?
        .slice(hidden_layout.start_offset()..);
    let b_slice = b_cuda
        .as_cuda_slice::<bf16>()?
        .slice(b_layout.start_offset()..);

    let status = unsafe {
        let (base_ptr, _base_guard) = base_slice.device_ptr(&stream);
        let (hidden_ptr, _hidden_guard) = hidden_slice.device_ptr(&stream);
        let (b_ptr, _b_guard) = b_slice.device_ptr(&stream);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        kiln_lora_decode_add_bf16(
            base_ptr as *const _,
            hidden_ptr as *const f32,
            b_ptr as *const _,
            out_ptr as *mut _,
            scale,
            rows as i32,
            out_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_lora_decode_add_bf16 failed with status {status}");
    }
    Ok(())
}

pub fn causal_depthwise_conv1d_f32(
    input: &Tensor,
    weight: &Tensor,
    state: &Tensor,
) -> Result<Tensor> {
    let (rows, channels) = input.dims2()?;
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32: empty weight".into())
    })?;
    if kernel <= 1 {
        candle_core::bail!("causal_depthwise_conv1d_f32: kernel must be > 1");
    }
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!("causal_depthwise_conv1d_f32: weight rank {rank} unsupported"),
    };
    if state.dims() != [kernel - 1, channels] {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32: state shape {:?} != [{},{}]",
            state.shape(),
            kernel - 1,
            channels
        );
    }
    if input.dtype() != DType::F32
        || weight_flat.dtype() != DType::F32
        || state.dtype() != DType::F32
    {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32: expected F32 input/weight/state, got {:?}/{:?}/{:?}",
            input.dtype(),
            weight_flat.dtype(),
            state.dtype()
        );
    }
    if rows > i32::MAX as usize || channels > i32::MAX as usize || kernel > i32::MAX as usize {
        candle_core::bail!("causal_depthwise_conv1d_f32: dimensions exceed i32 kernel envelope");
    }

    let input = input.contiguous()?;
    let weight_flat = weight_flat.contiguous()?;
    let state = state.contiguous()?;
    let out = unsafe { Tensor::empty((rows, channels), DType::F32, input.device())? };

    {
        let (i_storage, i_layout) = input.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (s_storage, s_layout) = state.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();
        let i_cuda = match &*i_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32: input must be CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32: weight must be CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32: state must be CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32: output must be CUDA"),
        };
        let stream = i_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let i_slice = i_cuda
            .as_cuda_slice::<f32>()?
            .slice(i_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<f32>()?
            .slice(w_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<f32>()?
            .slice(s_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);

        let status = unsafe {
            let (i_ptr, _i_guard) = i_slice.device_ptr(&stream);
            let (w_ptr, _w_guard) = w_slice.device_ptr(&stream);
            let (s_ptr, _s_guard) = s_slice.device_ptr(&stream);
            let (o_ptr, _o_guard) = o_slice.device_ptr(&stream);
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
            candle_core::bail!("kiln_causal_depthwise_conv1d_f32 failed with status {status}");
        }
    }
    Ok(out)
}

pub fn causal_depthwise_conv1d_f32_inplace(
    input_out: &Tensor,
    weight: &Tensor,
    state: &Tensor,
) -> Result<Tensor> {
    let (rows, channels) = input_out.dims2()?;
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32_inplace: empty weight".into())
    })?;
    if kernel <= 1 {
        candle_core::bail!("causal_depthwise_conv1d_f32_inplace: kernel must be > 1");
    }
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_inplace: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_inplace: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!(
            "causal_depthwise_conv1d_f32_inplace: weight rank {rank} unsupported"
        ),
    };
    if state.dims() != [kernel - 1, channels] {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_inplace: state shape {:?} != [{},{}]",
            state.shape(),
            kernel - 1,
            channels
        );
    }
    if input_out.dtype() != DType::F32
        || weight_flat.dtype() != DType::F32
        || state.dtype() != DType::F32
    {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_inplace: expected F32 input/weight/state, got {:?}/{:?}/{:?}",
            input_out.dtype(),
            weight_flat.dtype(),
            state.dtype()
        );
    }
    if !input_out.is_contiguous() {
        candle_core::bail!("causal_depthwise_conv1d_f32_inplace: input must be contiguous");
    }
    if rows > i32::MAX as usize || channels > i32::MAX as usize || kernel > i32::MAX as usize {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_inplace: dimensions exceed i32 kernel envelope"
        );
    }
    if rows == 0 || channels == 0 {
        return Ok(input_out.clone());
    }

    let weight_flat = weight_flat.contiguous()?;
    let state = state.contiguous()?;
    {
        let (io_storage, io_layout) = input_out.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (s_storage, s_layout) = state.storage_and_layout();
        let io_cuda = match &*io_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_inplace: input must be CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_inplace: weight must be CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_inplace: state must be CUDA"),
        };
        let stream = io_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let io_slice = io_cuda
            .as_cuda_slice::<f32>()?
            .slice(io_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<f32>()?
            .slice(w_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<f32>()?
            .slice(s_layout.start_offset()..);

        let status = unsafe {
            let (io_ptr, _io_guard) = io_slice.device_ptr(&stream);
            let (w_ptr, _w_guard) = w_slice.device_ptr(&stream);
            let (s_ptr, _s_guard) = s_slice.device_ptr(&stream);
            kiln_causal_depthwise_conv1d_inplace_f32(
                io_ptr as *mut f32,
                w_ptr as *const f32,
                s_ptr as *const f32,
                rows as i32,
                channels as i32,
                kernel as i32,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!(
                "kiln_causal_depthwise_conv1d_inplace_f32 failed with status {status}"
            );
        }
    }
    Ok(input_out.clone())
}

pub fn silu_inplace_save_sigmoid_f32(input_out: &Tensor) -> Result<(Tensor, Tensor)> {
    if input_out.dtype() != DType::F32 {
        candle_core::bail!(
            "silu_inplace_save_sigmoid_f32: expected F32 input, got {:?}",
            input_out.dtype()
        );
    }
    if !input_out.is_contiguous() {
        candle_core::bail!("silu_inplace_save_sigmoid_f32: input must be contiguous");
    }
    let elems = input_out.elem_count();
    if elems > i64::MAX as usize {
        candle_core::bail!(
            "silu_inplace_save_sigmoid_f32: element count exceeds i64 kernel envelope"
        );
    }
    if elems == 0 {
        let sigmoid =
            unsafe { Tensor::empty(input_out.shape().clone(), DType::F32, input_out.device())? };
        return Ok((input_out.clone(), sigmoid));
    }

    let sigmoid =
        unsafe { Tensor::empty(input_out.shape().clone(), DType::F32, input_out.device())? };
    {
        let (storage, layout) = input_out.storage_and_layout();
        let cuda = match &*storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("silu_inplace_save_sigmoid_f32: input must be CUDA"),
        };
        let (sigmoid_storage, sigmoid_layout) = sigmoid.storage_and_layout();
        let sigmoid_cuda = match &*sigmoid_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("silu_inplace_save_sigmoid_f32: sigmoid output must be CUDA"),
        };
        let stream = cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let slice = cuda.as_cuda_slice::<f32>()?.slice(layout.start_offset()..);
        let sigmoid_slice = sigmoid_cuda
            .as_cuda_slice::<f32>()?
            .slice(sigmoid_layout.start_offset()..);

        let status = unsafe {
            let (ptr, _guard) = slice.device_ptr(&stream);
            let (sigmoid_ptr, _sigmoid_guard) = sigmoid_slice.device_ptr(&stream);
            kiln_silu_inplace_save_sigmoid_f32(
                ptr as *mut f32,
                sigmoid_ptr as *mut f32,
                elems as i64,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!("kiln_silu_inplace_save_sigmoid_f32 failed with status {status}");
        }
    }
    Ok((input_out.clone(), sigmoid))
}

pub fn causal_depthwise_conv1d_f32_bwd_input(grad_out: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, channels) = grad_out.dims2()?;
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32_bwd_input: empty weight".into())
    })?;
    if kernel <= 1 {
        candle_core::bail!("causal_depthwise_conv1d_f32_bwd_input: kernel must be > 1");
    }
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_input: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_input: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_input: weight rank {rank} unsupported"
        ),
    };
    if grad_out.dtype() != DType::F32 || weight_flat.dtype() != DType::F32 {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_input: expected F32 grad/weight, got {:?}/{:?}",
            grad_out.dtype(),
            weight_flat.dtype()
        );
    }
    if rows > i32::MAX as usize || channels > i32::MAX as usize || kernel > i32::MAX as usize {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_input: dimensions exceed i32 kernel envelope"
        );
    }

    let grad_out = grad_out.contiguous()?;
    let weight_flat = weight_flat.contiguous()?;
    let grad_input = unsafe { Tensor::empty((rows, channels), DType::F32, grad_out.device())? };

    {
        let (g_storage, g_layout) = grad_out.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (o_storage, o_layout) = grad_input.storage_and_layout();
        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_input: grad must be CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_input: weight must be CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_input: output must be CUDA"),
        };
        let stream = g_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let g_slice = g_cuda
            .as_cuda_slice::<f32>()?
            .slice(g_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<f32>()?
            .slice(w_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);

        let status = unsafe {
            let (g_ptr, _g_guard) = g_slice.device_ptr(&stream);
            let (w_ptr, _w_guard) = w_slice.device_ptr(&stream);
            let (o_ptr, _o_guard) = o_slice.device_ptr(&stream);
            kiln_causal_depthwise_conv1d_bwd_input_f32(
                g_ptr as *const f32,
                w_ptr as *const f32,
                o_ptr as *mut f32,
                rows as i32,
                channels as i32,
                kernel as i32,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!(
                "kiln_causal_depthwise_conv1d_bwd_input_f32 failed with status {status}"
            );
        }
    }
    Ok(grad_input)
}

pub fn causal_depthwise_conv1d_f32_bwd_weight(
    grad_out: &Tensor,
    input: &Tensor,
    state: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    let (rows, channels) = grad_out.dims2()?;
    if input.dims() != [rows, channels] {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_weight: input shape {:?} != grad {:?}",
            input.shape(),
            grad_out.shape()
        );
    }
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32_bwd_weight: empty weight".into())
    })?;
    if state.dims() != [kernel - 1, channels] {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_weight: state shape {:?} != [{},{}]",
            state.shape(),
            kernel - 1,
            channels
        );
    }
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_weight: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_weight: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_weight: weight rank {rank} unsupported"
        ),
    };
    if grad_out.dtype() != DType::F32
        || input.dtype() != DType::F32
        || state.dtype() != DType::F32
        || weight_flat.dtype() != DType::F32
    {
        candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: expected F32 tensors");
    }

    let grad_out = grad_out.contiguous()?;
    let input = input.contiguous()?;
    let state = state.contiguous()?;
    let grad_weight = unsafe { Tensor::empty((channels, kernel), DType::F32, grad_out.device())? };
    {
        let (g_storage, g_layout) = grad_out.storage_and_layout();
        let (i_storage, i_layout) = input.storage_and_layout();
        let (s_storage, s_layout) = state.storage_and_layout();
        let (o_storage, o_layout) = grad_weight.storage_and_layout();
        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: grad must be CUDA"),
        };
        let i_cuda = match &*i_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: input must be CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: state must be CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: output must be CUDA"),
        };
        let stream = g_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let g_slice = g_cuda
            .as_cuda_slice::<f32>()?
            .slice(g_layout.start_offset()..);
        let i_slice = i_cuda
            .as_cuda_slice::<f32>()?
            .slice(i_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<f32>()?
            .slice(s_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);
        let status = unsafe {
            let (g_ptr, _g_guard) = g_slice.device_ptr(&stream);
            let (i_ptr, _i_guard) = i_slice.device_ptr(&stream);
            let (s_ptr, _s_guard) = s_slice.device_ptr(&stream);
            let (o_ptr, _o_guard) = o_slice.device_ptr(&stream);
            kiln_causal_depthwise_conv1d_bwd_weight_f32(
                g_ptr as *const f32,
                i_ptr as *const f32,
                s_ptr as *const f32,
                o_ptr as *mut f32,
                rows as i32,
                channels as i32,
                kernel as i32,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!(
                "kiln_causal_depthwise_conv1d_bwd_weight_f32 failed with status {status}"
            );
        }
    }
    if weight.rank() == 3 {
        grad_weight.reshape(weight.dims())
    } else {
        Ok(grad_weight)
    }
}

pub fn causal_depthwise_conv1d_f32_bwd_state(grad_out: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, channels) = grad_out.dims2()?;
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32_bwd_state: empty weight".into())
    })?;
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_state: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_state: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_state: weight rank {rank} unsupported"
        ),
    };
    if grad_out.dtype() != DType::F32 || weight_flat.dtype() != DType::F32 {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_state: expected F32 grad/weight, got {:?}/{:?}",
            grad_out.dtype(),
            weight_flat.dtype()
        );
    }
    let grad_out = grad_out.contiguous()?;
    let weight_flat = weight_flat.contiguous()?;
    let grad_state =
        unsafe { Tensor::empty((kernel - 1, channels), DType::F32, grad_out.device())? };
    {
        let (g_storage, g_layout) = grad_out.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (o_storage, o_layout) = grad_state.storage_and_layout();
        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_state: grad must be CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_state: weight must be CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_state: output must be CUDA"),
        };
        let stream = g_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let g_slice = g_cuda
            .as_cuda_slice::<f32>()?
            .slice(g_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<f32>()?
            .slice(w_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);
        let status = unsafe {
            let (g_ptr, _g_guard) = g_slice.device_ptr(&stream);
            let (w_ptr, _w_guard) = w_slice.device_ptr(&stream);
            let (o_ptr, _o_guard) = o_slice.device_ptr(&stream);
            kiln_causal_depthwise_conv1d_bwd_state_f32(
                g_ptr as *const f32,
                w_ptr as *const f32,
                o_ptr as *mut f32,
                rows as i32,
                channels as i32,
                kernel as i32,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!(
                "kiln_causal_depthwise_conv1d_bwd_state_f32 failed with status {status}"
            );
        }
    }
    Ok(grad_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    // Reference implementation — mirrors `kiln-model::forward::rms_norm`
    // exactly (including the F32 cast + Qwen3.5 `(1 + w)` convention).
    // Used as the correctness oracle for `fused_rmsnorm`.
    fn reference_rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms_inv = (variance + eps)?.sqrt()?.recip()?;
        let normed = x_f32.broadcast_mul(&rms_inv)?;
        let w_f32 = weight.to_dtype(DType::F32)?;
        let w_plus_one = (w_f32.ones_like()? + w_f32)?;
        let out = normed.broadcast_mul(&w_plus_one)?;
        out.to_dtype(x.dtype())
    }

    fn try_cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    fn reference_rope(
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        head_dim: usize,
        rotary_dim: usize,
    ) -> Result<Tensor> {
        let half = rotary_dim / 2;
        let x_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let x_rot = x.narrow(candle_core::D::Minus1, 0, rotary_dim)?;
        let x_pass = if rotary_dim < head_dim {
            Some(x.narrow(candle_core::D::Minus1, rotary_dim, head_dim - rotary_dim)?)
        } else {
            None
        };
        let x1 = x_rot.narrow(candle_core::D::Minus1, 0, half)?;
        let x2 = x_rot.narrow(candle_core::D::Minus1, half, half)?;
        let cos = cos.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        let out = match x_pass {
            Some(pass) => Tensor::cat(&[&r1, &r2, &pass], candle_core::D::Minus1)?,
            None => Tensor::cat(&[&r1, &r2], candle_core::D::Minus1)?,
        };
        out.to_dtype(x_dtype)
    }

    fn reference_mlp_silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let dtype = gate.dtype();
        let gate = gate.to_dtype(DType::F32)?;
        let up = up.to_dtype(DType::F32)?;
        let denom = (gate.neg()?.exp()? + 1.0)?;
        let silu = (&gate * denom.recip()?)?;
        (silu * up)?.to_dtype(dtype)
    }

    fn reference_sigmoid_mul(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let gate = gate.to_dtype(DType::F32)?;
        let sigmoid = (gate.neg()?.exp()? + 1.0)?.recip()?;
        (x * sigmoid)?.to_dtype(dtype)
    }

    #[test]
    fn parity_decode_row() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Qwen3.5-4B decode shape: [batch=1, seq=1, hidden=2560].
        let hidden = 2560usize;
        let rows = 1usize;
        let eps = 1e-6;

        // Deterministic pseudo-random input (seeded), so the parity test is
        // reproducible without adding a dev-dep on `rand`.
        let mut raw_x = Vec::with_capacity(rows * hidden);
        let mut raw_w = Vec::with_capacity(hidden);
        let mut state: u32 = 0x1234_5678;
        for _ in 0..rows * hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            raw_x.push(f * 0.5);
        }
        for _ in 0..hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            raw_w.push(f * 0.1);
        }

        let x_f32 = Tensor::from_vec(raw_x, (rows, hidden), &device).unwrap();
        let w_f32 = Tensor::from_vec(raw_w, (hidden,), &device).unwrap();
        let x = x_f32.to_dtype(DType::BF16).unwrap();
        let w = w_f32.to_dtype(DType::BF16).unwrap();

        let y_ref = reference_rms_norm(&x, &w, eps).unwrap();
        let y_fused = fused_rmsnorm(&x, &w, eps as f32).unwrap();

        let diff = (&y_ref - &y_fused)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            diff < 1e-2,
            "parity failed: max_abs_diff={diff} exceeds 1e-2 tolerance"
        );
    }

    #[test]
    fn rotary_qk_parity_qwen_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let batch = 1usize;
        let seq_len = 2usize;
        let q_heads = 16usize;
        let k_heads = 4usize;
        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let half = rotary_dim / 2;

        let q_data: Vec<f32> = (0..batch * seq_len * q_heads * head_dim)
            .map(|i| ((i as f32 * 0.013).sin() * 0.5) + ((i as f32 * 0.017).cos() * 0.25))
            .collect();
        let k_data: Vec<f32> = (0..batch * seq_len * k_heads * head_dim)
            .map(|i| ((i as f32 * 0.019).sin() * 0.4) + ((i as f32 * 0.011).cos() * 0.2))
            .collect();
        let cos_data: Vec<f32> = (0..seq_len * half)
            .map(|i| (i as f32 * 0.007).cos())
            .collect();
        let sin_data: Vec<f32> = (0..seq_len * half)
            .map(|i| (i as f32 * 0.007).sin())
            .collect();

        let q = Tensor::from_vec(q_data, (batch, seq_len, q_heads, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::from_vec(k_data, (batch, seq_len, k_heads, head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let cos = Tensor::from_vec(cos_data, (seq_len, half), &device).unwrap();
        let sin = Tensor::from_vec(sin_data, (seq_len, half), &device).unwrap();

        assert!(supports_rotary_qk(&q, &k, &cos, &sin, head_dim, rotary_dim));
        let (q_fused, k_fused) =
            fused_rotary_qk(&q, &k, &cos, &sin, head_dim, rotary_dim).expect("fused rotary qk");
        let q_ref = reference_rope(&q, &cos, &sin, head_dim, rotary_dim).unwrap();
        let k_ref = reference_rope(&k, &cos, &sin, head_dim, rotary_dim).unwrap();

        let q_diff = (&q_ref - &q_fused)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let k_diff = (&k_ref - &k_fused)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert_eq!(q_diff, 0.0, "rotary q max_abs_diff={q_diff:e}");
        assert_eq!(k_diff, 0.0, "rotary k max_abs_diff={k_diff:e}");
    }

    #[test]
    fn attn_decode_qkv_prep_parity_qwen_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let batch = 1usize;
        let seq_len = 1usize;
        let q_heads = 16usize;
        let k_heads = 4usize;
        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let half = rotary_dim / 2;
        let eps = 1e-6f64;

        let mut q_raw_data = Vec::with_capacity(batch * seq_len * q_heads * head_dim * 2);
        let mut k_raw_data = Vec::with_capacity(batch * seq_len * k_heads * head_dim);
        let mut q_weight_data = Vec::with_capacity(head_dim);
        let mut k_weight_data = Vec::with_capacity(head_dim);
        fill_pseudo_random(
            &mut q_raw_data,
            batch * seq_len * q_heads * head_dim * 2,
            0x1020_3040,
            0.75,
        );
        fill_pseudo_random(
            &mut k_raw_data,
            batch * seq_len * k_heads * head_dim,
            0x5060_7080,
            0.65,
        );
        fill_pseudo_random(&mut q_weight_data, head_dim, 0x90ab_cdef, 0.1);
        fill_pseudo_random(&mut k_weight_data, head_dim, 0x1357_2468, 0.1);

        let cos_data: Vec<f32> = (0..seq_len * half)
            .map(|i| (i as f32 * 0.007).cos())
            .collect();
        let sin_data: Vec<f32> = (0..seq_len * half)
            .map(|i| (i as f32 * 0.007).sin())
            .collect();

        let q_raw = Tensor::from_vec(
            q_raw_data,
            (batch, seq_len, q_heads * head_dim * 2),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let k_raw = Tensor::from_vec(k_raw_data, (batch, seq_len, k_heads * head_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let q_weight = Tensor::from_vec(q_weight_data, (head_dim,), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k_weight = Tensor::from_vec(k_weight_data, (head_dim,), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let cos = Tensor::from_vec(cos_data, (seq_len, half), &device).unwrap();
        let sin = Tensor::from_vec(sin_data, (seq_len, half), &device).unwrap();

        let q_raw4 = q_raw
            .reshape((batch, seq_len, q_heads, head_dim * 2))
            .unwrap();
        let q_pre = q_raw4.narrow(3, 0, head_dim).unwrap().contiguous().unwrap();
        let gate_ref = q_raw4
            .narrow(3, head_dim, head_dim)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((batch, seq_len, q_heads * head_dim))
            .unwrap();
        let k_pre = k_raw.reshape((batch, seq_len, k_heads, head_dim)).unwrap();
        let q_norm = reference_rms_norm(&q_pre, &q_weight, eps).unwrap();
        let k_norm = reference_rms_norm(&k_pre, &k_weight, eps).unwrap();
        let q_ref = reference_rope(&q_norm, &cos, &sin, head_dim, rotary_dim).unwrap();
        let k_ref = reference_rope(&k_norm, &cos, &sin, head_dim, rotary_dim).unwrap();

        assert!(supports_attn_decode_qkv_prep(
            &q_raw, &k_raw, &q_weight, &k_weight, &cos, &sin, q_heads, k_heads, head_dim,
            rotary_dim, true
        ));
        let (q_fused, k_fused, gate_fused) = fused_attn_decode_qkv_prep(
            &q_raw, &k_raw, &q_weight, &k_weight, &cos, &sin, q_heads, k_heads, head_dim,
            rotary_dim, true, eps as f32,
        )
        .expect("fused attn decode qkv prep");

        assert_eq!(q_fused.dims(), &[batch, seq_len, q_heads, head_dim]);
        assert_eq!(k_fused.dims(), &[batch, seq_len, k_heads, head_dim]);
        let gate_fused = gate_fused.expect("gate output");
        assert_eq!(gate_fused.dims(), &[batch, seq_len, q_heads * head_dim]);

        let q_diff = max_abs_diff(&q_ref, &q_fused);
        let k_diff = max_abs_diff(&k_ref, &k_fused);
        let gate_diff = max_abs_diff(&gate_ref, &gate_fused);
        assert!(
            q_diff < 1e-2,
            "attn q prep parity failed: max_abs_diff={q_diff:e}"
        );
        assert!(
            k_diff < 1e-2,
            "attn k prep parity failed: max_abs_diff={k_diff:e}"
        );
        assert_eq!(gate_diff, 0.0, "gate copy max_abs_diff={gate_diff:e}");
    }

    #[test]
    fn mlp_silu_mul_parity_qwen_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let batch = 1usize;
        let seq_len = 2usize;
        let intermediate = 9728usize;
        let total = batch * seq_len * intermediate;
        let mut gate_raw = Vec::with_capacity(total);
        let mut up_raw = Vec::with_capacity(total);
        fill_pseudo_random(&mut gate_raw, total, 0x3141_5926, 3.0);
        fill_pseudo_random(&mut up_raw, total, 0x2718_2818, 2.0);

        let gate = Tensor::from_vec(gate_raw, (batch, seq_len, intermediate), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let up = Tensor::from_vec(up_raw, (batch, seq_len, intermediate), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        assert!(supports_mlp_silu_mul(&gate, &up));
        let reference = reference_mlp_silu_mul(&gate, &up).unwrap();
        let fused = fused_mlp_silu_mul(&gate, &up).expect("fused mlp silu mul");
        assert_eq!(fused.dims(), &[batch, seq_len, intermediate]);

        let diff = max_abs_diff(&reference, &fused);
        assert!(
            diff < 1e-2,
            "MLP silu*mul parity failed: max_abs_diff={diff} exceeds 1e-2 tolerance"
        );
    }

    #[test]
    fn sigmoid_mul_parity_qwen_attn_gate_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let batch = 1usize;
        let seq_len = 2usize;
        let hidden = 4096usize;
        let total = batch * seq_len * hidden;
        let mut x_raw = Vec::with_capacity(total);
        let mut gate_raw = Vec::with_capacity(total);
        fill_pseudo_random(&mut x_raw, total, 0x5151_0101, 1.5);
        fill_pseudo_random(&mut gate_raw, total, 0x9191_0202, 3.0);

        let x = Tensor::from_vec(x_raw, (batch, seq_len, hidden), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let gate = Tensor::from_vec(gate_raw, (batch, seq_len, hidden), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        assert!(supports_sigmoid_mul(&x, &gate));
        let reference = reference_sigmoid_mul(&x, &gate).unwrap();
        let fused = fused_sigmoid_mul(&x, &gate).expect("fused sigmoid mul");
        assert_eq!(fused.dims(), &[batch, seq_len, hidden]);

        let diff = max_abs_diff(&reference, &fused);
        assert!(
            diff < 1e-2,
            "sigmoid*mul parity failed: max_abs_diff={diff} exceeds 1e-2 tolerance"
        );
    }

    #[test]
    fn parity_multi_row() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Prefill-like shape: [batch=1, seq=512, hidden=2560].
        let hidden = 2560usize;
        let rows = 512usize;
        let eps = 1e-6;

        let mut raw_x = Vec::with_capacity(rows * hidden);
        let mut raw_w = Vec::with_capacity(hidden);
        let mut state: u32 = 0xcafe_babe;
        for _ in 0..rows * hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            raw_x.push(f * 0.7);
        }
        for _ in 0..hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            raw_w.push(f * 0.1);
        }

        let x_f32 = Tensor::from_vec(raw_x, (rows, hidden), &device).unwrap();
        let w_f32 = Tensor::from_vec(raw_w, (hidden,), &device).unwrap();
        let x = x_f32.to_dtype(DType::BF16).unwrap();
        let w = w_f32.to_dtype(DType::BF16).unwrap();

        let y_ref = reference_rms_norm(&x, &w, eps).unwrap();
        let y_fused = fused_rmsnorm(&x, &w, eps as f32).unwrap();

        let diff = (&y_ref - &y_fused)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            diff < 1e-2,
            "parity failed: max_abs_diff={diff} exceeds 1e-2 tolerance"
        );
    }

    #[test]
    fn parity_with_batch_dim() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // [batch=2, seq=3, hidden=2560] — exercises 3D reshape path.
        let b = 2usize;
        let s = 3usize;
        let hidden = 2560usize;
        let eps = 1e-6;

        let mut raw_x = Vec::with_capacity(b * s * hidden);
        let mut raw_w = Vec::with_capacity(hidden);
        let mut state: u32 = 0xdead_beef;
        for _ in 0..b * s * hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            raw_x.push(f * 0.3);
        }
        for _ in 0..hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            raw_w.push(f * 0.1);
        }

        let x_f32 = Tensor::from_vec(raw_x, (b, s, hidden), &device).unwrap();
        let w_f32 = Tensor::from_vec(raw_w, (hidden,), &device).unwrap();
        let x = x_f32.to_dtype(DType::BF16).unwrap();
        let w = w_f32.to_dtype(DType::BF16).unwrap();

        let y_ref = reference_rms_norm(&x, &w, eps).unwrap();
        let y_fused = fused_rmsnorm(&x, &w, eps as f32).unwrap();

        assert_eq!(y_fused.dims(), &[b, s, hidden]);

        let diff = (&y_ref - &y_fused)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(
            diff < 1e-2,
            "parity failed: max_abs_diff={diff} exceeds 1e-2 tolerance"
        );
    }

    // ----- L2 QK-norm parity (matches `kiln-model::forward::l2_normalize`
    // followed by the `q * scale` + `to_dtype(bf16)` epilogue used in the
    // `kiln/gdn/qk_norm` block). -----

    fn reference_l2_qk_norm(
        q: &Tensor,
        k: &Tensor,
        q_scale: f64,
        eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        let dtype = q.dtype();

        let q_f32 = q.to_dtype(DType::F32)?;
        let q_sq = q_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
        let q_norm = (q_sq + eps)?.sqrt()?;
        let q_normed = q_f32.broadcast_div(&q_norm)?;
        let q_out = (q_normed * q_scale)?.to_dtype(dtype)?;

        let k_f32 = k.to_dtype(DType::F32)?;
        let k_sq = k_f32.sqr()?.sum_keepdim(candle_core::D::Minus1)?;
        let k_norm = (k_sq + eps)?.sqrt()?;
        let k_normed = k_f32.broadcast_div(&k_norm)?;
        let k_out = k_normed.to_dtype(dtype)?;

        Ok((q_out, k_out))
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn fill_pseudo_random(buf: &mut Vec<f32>, n: usize, seed: u32, scale: f32) {
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            buf.push(f * scale);
        }
    }

    #[test]
    fn parity_l2_qk_norm_decode_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Qwen3.5-4B GDN decode shape: [batch=1, seq=1, nv=16, dk=128]
        let batch = 1usize;
        let seq = 1usize;
        let nv = 16usize;
        let dk = 128usize;
        let total = batch * seq * nv * dk;
        let q_scale = 1.0 / (dk as f64).sqrt();
        let eps = 1e-6;

        let mut q_raw = Vec::with_capacity(total);
        let mut k_raw = Vec::with_capacity(total);
        fill_pseudo_random(&mut q_raw, total, 0x1357_9bdf, 0.5);
        fill_pseudo_random(&mut k_raw, total, 0x2468_ace0, 0.5);

        let q_f32 = Tensor::from_vec(q_raw, (batch, seq, nv, dk), &device).unwrap();
        let k_f32 = Tensor::from_vec(k_raw, (batch, seq, nv, dk), &device).unwrap();
        let q = q_f32.to_dtype(DType::BF16).unwrap();
        let k = k_f32.to_dtype(DType::BF16).unwrap();

        let (q_ref, k_ref) = reference_l2_qk_norm(&q, &k, q_scale, eps).unwrap();
        let (q_fused, k_fused) = fused_l2_qk_norm(&q, &k, q_scale as f32, eps as f32).unwrap();

        assert_eq!(q_fused.dims(), &[batch, seq, nv, dk]);
        assert_eq!(k_fused.dims(), &[batch, seq, nv, dk]);

        let q_diff = max_abs_diff(&q_ref, &q_fused);
        let k_diff = max_abs_diff(&k_ref, &k_fused);

        // bf16 tolerance: 1e-2 matches the existing rmsnorm tests. Both
        // outputs are bf16, both reductions are F32, so this is the right
        // bar.
        assert!(
            q_diff < 1e-2,
            "Q parity failed: max_abs_diff={q_diff} exceeds 1e-2 tolerance"
        );
        assert!(
            k_diff < 1e-2,
            "K parity failed: max_abs_diff={k_diff} exceeds 1e-2 tolerance"
        );
    }

    #[test]
    fn parity_l2_qk_norm_prefill_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Prefill-like shape with Qwen3.5-4B GDN dims:
        // [batch=1, seq=512, nv=16, dk=128] — 8192 rows.
        let batch = 1usize;
        let seq = 512usize;
        let nv = 16usize;
        let dk = 128usize;
        let total = batch * seq * nv * dk;
        let q_scale = 1.0 / (dk as f64).sqrt();
        let eps = 1e-6;

        let mut q_raw = Vec::with_capacity(total);
        let mut k_raw = Vec::with_capacity(total);
        fill_pseudo_random(&mut q_raw, total, 0xdead_beef, 0.7);
        fill_pseudo_random(&mut k_raw, total, 0xfeed_face, 0.7);

        let q_f32 = Tensor::from_vec(q_raw, (batch, seq, nv, dk), &device).unwrap();
        let k_f32 = Tensor::from_vec(k_raw, (batch, seq, nv, dk), &device).unwrap();
        let q = q_f32.to_dtype(DType::BF16).unwrap();
        let k = k_f32.to_dtype(DType::BF16).unwrap();

        let (q_ref, k_ref) = reference_l2_qk_norm(&q, &k, q_scale, eps).unwrap();
        let (q_fused, k_fused) = fused_l2_qk_norm(&q, &k, q_scale as f32, eps as f32).unwrap();

        assert_eq!(q_fused.dims(), &[batch, seq, nv, dk]);
        assert_eq!(k_fused.dims(), &[batch, seq, nv, dk]);

        let q_diff = max_abs_diff(&q_ref, &q_fused);
        let k_diff = max_abs_diff(&k_ref, &k_fused);

        assert!(
            q_diff < 1e-2,
            "Q parity failed: max_abs_diff={q_diff} exceeds 1e-2 tolerance"
        );
        assert!(
            k_diff < 1e-2,
            "K parity failed: max_abs_diff={k_diff} exceeds 1e-2 tolerance"
        );
    }

    #[test]
    fn parity_l2_qk_norm_gqa_decode_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let batch = 1usize;
        let seq = 1usize;
        let nk = 8usize;
        let nv = 16usize;
        let dk = 128usize;
        let total = batch * seq * nk * dk;
        let q_scale = 1.0 / (dk as f64).sqrt();
        let eps = 1e-6;

        let mut q_raw = Vec::with_capacity(total);
        let mut k_raw = Vec::with_capacity(total);
        fill_pseudo_random(&mut q_raw, total, 0x3141_5926, 0.5);
        fill_pseudo_random(&mut k_raw, total, 0x2718_2818, 0.5);

        let q = Tensor::from_vec(q_raw, (batch, seq, nk, dk), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::from_vec(k_raw, (batch, seq, nk, dk), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        assert!(supports_l2_qk_norm_gqa(&q, &k, nv));

        let ratio = nv / nk;
        let q_expanded = q
            .unsqueeze(3)
            .unwrap()
            .expand(&[batch, seq, nk, ratio, dk])
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((batch, seq, nv, dk))
            .unwrap();
        let k_expanded = k
            .unsqueeze(3)
            .unwrap()
            .expand(&[batch, seq, nk, ratio, dk])
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((batch, seq, nv, dk))
            .unwrap();

        let (q_ref, k_ref) = reference_l2_qk_norm(&q_expanded, &k_expanded, q_scale, eps).unwrap();
        let (q_fused, k_fused) =
            fused_l2_qk_norm_gqa(&q, &k, nv, q_scale as f32, eps as f32).unwrap();

        assert_eq!(q_fused.dims(), &[batch, seq, nv, dk]);
        assert_eq!(k_fused.dims(), &[batch, seq, nv, dk]);

        let q_diff = max_abs_diff(&q_ref, &q_fused);
        let k_diff = max_abs_diff(&k_ref, &k_fused);

        assert!(
            q_diff < 1e-2,
            "GQA Q parity failed: max_abs_diff={q_diff} exceeds 1e-2 tolerance"
        );
        assert!(
            k_diff < 1e-2,
            "GQA K parity failed: max_abs_diff={k_diff} exceeds 1e-2 tolerance"
        );
    }

    #[test]
    fn parity_l2_qk_norm_gqa_prefill_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let batch = 2usize;
        let seq = 17usize;
        let nk = 8usize;
        let nv = 16usize;
        let dk = 128usize;
        let total = batch * seq * nk * dk;
        let q_scale = 1.0 / (dk as f64).sqrt();
        let eps = 1e-6;

        let mut q_raw = Vec::with_capacity(total);
        let mut k_raw = Vec::with_capacity(total);
        fill_pseudo_random(&mut q_raw, total, 0x0bad_f00d, 0.7);
        fill_pseudo_random(&mut k_raw, total, 0x00c0_ffee, 0.7);

        let q = Tensor::from_vec(q_raw, (batch, seq, nk, dk), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let k = Tensor::from_vec(k_raw, (batch, seq, nk, dk), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let ratio = nv / nk;
        let q_expanded = q
            .unsqueeze(3)
            .unwrap()
            .expand(&[batch, seq, nk, ratio, dk])
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((batch, seq, nv, dk))
            .unwrap();
        let k_expanded = k
            .unsqueeze(3)
            .unwrap()
            .expand(&[batch, seq, nk, ratio, dk])
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((batch, seq, nv, dk))
            .unwrap();

        let (q_ref, k_ref) = reference_l2_qk_norm(&q_expanded, &k_expanded, q_scale, eps).unwrap();
        let (q_fused, k_fused) =
            fused_l2_qk_norm_gqa(&q, &k, nv, q_scale as f32, eps as f32).unwrap();

        let q_diff = max_abs_diff(&q_ref, &q_fused);
        let k_diff = max_abs_diff(&k_ref, &k_fused);

        assert!(
            q_diff < 1e-2,
            "GQA prefill Q parity failed: max_abs_diff={q_diff} exceeds 1e-2 tolerance"
        );
        assert!(
            k_diff < 1e-2,
            "GQA prefill K parity failed: max_abs_diff={k_diff} exceeds 1e-2 tolerance"
        );
    }

    #[test]
    fn parity_l2_qk_norm_batch_two() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // [batch=2, seq=1, nv=16, dk=128] — exercises the multi-batch path.
        let batch = 2usize;
        let seq = 1usize;
        let nv = 16usize;
        let dk = 128usize;
        let total = batch * seq * nv * dk;
        let q_scale = 1.0 / (dk as f64).sqrt();
        let eps = 1e-6;

        let mut q_raw = Vec::with_capacity(total);
        let mut k_raw = Vec::with_capacity(total);
        fill_pseudo_random(&mut q_raw, total, 0x1234_5678, 0.4);
        fill_pseudo_random(&mut k_raw, total, 0xaaaa_5555, 0.4);

        let q_f32 = Tensor::from_vec(q_raw, (batch, seq, nv, dk), &device).unwrap();
        let k_f32 = Tensor::from_vec(k_raw, (batch, seq, nv, dk), &device).unwrap();
        let q = q_f32.to_dtype(DType::BF16).unwrap();
        let k = k_f32.to_dtype(DType::BF16).unwrap();

        let (q_ref, k_ref) = reference_l2_qk_norm(&q, &k, q_scale, eps).unwrap();
        let (q_fused, k_fused) = fused_l2_qk_norm(&q, &k, q_scale as f32, eps as f32).unwrap();

        let q_diff = max_abs_diff(&q_ref, &q_fused);
        let k_diff = max_abs_diff(&k_ref, &k_fused);

        assert!(
            q_diff < 1e-2,
            "Q parity failed: max_abs_diff={q_diff} exceeds 1e-2 tolerance"
        );
        assert!(
            k_diff < 1e-2,
            "K parity failed: max_abs_diff={k_diff} exceeds 1e-2 tolerance"
        );
    }

    // ----- Backward (Phase 10) parity tests -----
    //
    // The analytical formula in `rmsnorm_backward_fallback` is the
    // correctness oracle. The CPU test confirms the closed-form math
    // matches what candle autograd computes by differentiating
    // `rms_norm_fallback`. The CUDA test confirms the fused backward
    // kernel matches the analytical formula on hardware.

    fn reference_autograd_backward(
        x: &Tensor,
        weight: &Tensor,
        grad_out: &Tensor,
        eps: f64,
    ) -> Result<(Tensor, Tensor)> {
        // Build the same forward graph as rms_norm_fallback but using a
        // candle Var so we can request gradients through `backward()`.
        // Using x_var.broadcast_mul(grad_out_const).sum() doesn't quite
        // give us the same VJP shape, so instead we form the loss
        // `L = sum(grad_out * out)` and read x.grad() / weight.grad();
        // those are precisely d(L)/dx and d(L)/dw, i.e. the analytical
        // backward outputs we want.
        use candle_core::Var;

        let x_var = Var::from_tensor(&x.detach())?;
        let w_var = Var::from_tensor(&weight.detach())?;
        let g_const = grad_out.detach();

        let x_t = x_var.as_tensor();
        let w_t = w_var.as_tensor();

        let x_f32 = x_t.to_dtype(DType::F32)?;
        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms_inv = (variance + eps)?.sqrt()?.recip()?;
        let normed = x_f32.broadcast_mul(&rms_inv)?;
        let w_f32 = w_t.to_dtype(DType::F32)?;
        let w_plus_one = (w_f32.ones_like()? + w_f32)?;
        let out = normed.broadcast_mul(&w_plus_one)?;
        let out_dtype = out.to_dtype(x.dtype())?;

        // Loss = sum(grad_out * out). VJP gives d(L)/dx == bwd grad_x.
        let loss = (out_dtype.broadcast_mul(&g_const))?.sum_all()?;
        let grads = loss.backward()?;
        let gx = grads
            .get(x_t)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing x grad".to_string()))?;
        let gw = grads
            .get(w_t)
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("missing w grad".to_string()))?;
        Ok((gx, gw))
    }

    #[test]
    fn test_fused_rmsnorm_backward_matches_fallback_cpu_f32() {
        let device = Device::Cpu;

        let rows = 8usize;
        let hidden = 64usize;
        let eps = 1e-6f64;

        let mut x_raw: Vec<f32> = Vec::with_capacity(rows * hidden);
        let mut w_raw: Vec<f32> = Vec::with_capacity(hidden);
        let mut g_raw: Vec<f32> = Vec::with_capacity(rows * hidden);
        let mut state: u32 = 0xa1b2_c3d4;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..rows * hidden {
            x_raw.push(next() * 0.5);
        }
        for _ in 0..hidden {
            w_raw.push(next() * 0.1);
        }
        for _ in 0..rows * hidden {
            g_raw.push(next() * 0.4);
        }

        let x = Tensor::from_vec(x_raw, (rows, hidden), &device).unwrap();
        let w = Tensor::from_vec(w_raw, (hidden,), &device).unwrap();
        let g = Tensor::from_vec(g_raw, (rows, hidden), &device).unwrap();

        let (gx_a, gw_a) = rmsnorm_backward_fallback(&x, &w, &g, eps).unwrap();
        let (gx_b, gw_b) = reference_autograd_backward(&x, &w, &g, eps).unwrap();

        let gx_diff = max_abs_diff(&gx_a, &gx_b);
        let gw_diff = max_abs_diff(&gw_a, &gw_b);

        // F32 reduction-order drift only — analytical and autograd compute
        // the same math through different reduction orders.
        assert!(
            gx_diff < 1e-4,
            "grad_x parity failed: max_abs_diff={gx_diff} (tol=1e-4)"
        );
        assert!(
            gw_diff < 1e-4,
            "grad_w parity failed: max_abs_diff={gw_diff} (tol=1e-4)"
        );
    }

    #[test]
    fn test_fused_rmsnorm_custom_op_forward_cpu_f32() {
        // The CustomOp2 cpu_fwd should match rms_norm_fallback (modulo F32
        // reduction order) for any rank-2+ input. Exercise it here directly.
        let device = Device::Cpu;
        let rows = 4usize;
        let hidden = 32usize;
        let eps = 1e-6f64;

        let mut x_raw: Vec<f32> = Vec::with_capacity(rows * hidden);
        let mut w_raw: Vec<f32> = Vec::with_capacity(hidden);
        let mut state: u32 = 0xfeed_face;
        for _ in 0..rows * hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            x_raw.push(((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0);
        }
        for _ in 0..hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            w_raw.push(((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.2);
        }

        let x = Tensor::from_vec(x_raw, (rows, hidden), &device).unwrap();
        let w = Tensor::from_vec(w_raw, (hidden,), &device).unwrap();

        let y_fused = fused_rmsnorm_with_autograd(&x, &w, eps as f32).unwrap();
        let y_ref = reference_rms_norm(&x, &w, eps).unwrap();
        let diff = max_abs_diff(&y_fused, &y_ref);
        assert!(
            diff < 1e-5,
            "custom-op CPU forward parity failed: max_abs_diff={diff} (tol=1e-5)"
        );
    }

    #[test]
    fn test_fused_rmsnorm_custom_op_backward_cpu_f32() {
        // End-to-end CPU autograd parity: rms_norm via the custom op should
        // produce the same VJP as rms_norm_fallback (within F32 reduction
        // tolerance). This exercises BOTH the CustomOp2 forward (cpu_fwd)
        // and its bwd hook on CPU.
        use candle_core::Var;

        let device = Device::Cpu;
        let rows = 4usize;
        let hidden = 32usize;
        let eps = 1e-6f64;

        let mut x_raw: Vec<f32> = Vec::with_capacity(rows * hidden);
        let mut w_raw: Vec<f32> = Vec::with_capacity(hidden);
        let mut g_raw: Vec<f32> = Vec::with_capacity(rows * hidden);
        let mut state: u32 = 0x1234_5678;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        for _ in 0..rows * hidden {
            x_raw.push(next() * 0.4);
        }
        for _ in 0..hidden {
            w_raw.push(next() * 0.1);
        }
        for _ in 0..rows * hidden {
            g_raw.push(next() * 0.3);
        }

        let x_t = Tensor::from_vec(x_raw, (rows, hidden), &device).unwrap();
        let w_t = Tensor::from_vec(w_raw, (hidden,), &device).unwrap();
        let g_const = Tensor::from_vec(g_raw, (rows, hidden), &device).unwrap();

        // Path A: custom op (cpu_fwd + bwd via fallback formula).
        let x_var_a = Var::from_tensor(&x_t).unwrap();
        let w_var_a = Var::from_tensor(&w_t).unwrap();
        let y_a = fused_rmsnorm_with_autograd(x_var_a.as_tensor(), w_var_a.as_tensor(), eps as f32)
            .unwrap();
        let loss_a = (y_a.broadcast_mul(&g_const)).unwrap().sum_all().unwrap();
        let grads_a = loss_a.backward().unwrap();
        let gx_a = grads_a.get(x_var_a.as_tensor()).cloned().unwrap();
        let gw_a = grads_a.get(w_var_a.as_tensor()).cloned().unwrap();

        // Path B: candle autograd through rms_norm_fallback.
        let (gx_b, gw_b) = reference_autograd_backward(&x_t, &w_t, &g_const, eps).unwrap();

        let gx_diff = max_abs_diff(&gx_a, &gx_b);
        let gw_diff = max_abs_diff(&gw_a, &gw_b);

        assert!(
            gx_diff < 1e-4,
            "custom-op grad_x parity failed: max_abs_diff={gx_diff} (tol=1e-4)"
        );
        assert!(
            gw_diff < 1e-4,
            "custom-op grad_w parity failed: max_abs_diff={gw_diff} (tol=1e-4)"
        );
    }

    #[test]
    fn parity_backward_decode_row_cuda() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Qwen3.5-4B decode shape: [batch=1, seq=1, hidden=2560].
        let hidden = 2560usize;
        let rows = 1usize;
        let eps = 1e-6;

        let mut raw_x = Vec::with_capacity(rows * hidden);
        let mut raw_w = Vec::with_capacity(hidden);
        let mut raw_g = Vec::with_capacity(rows * hidden);
        let mut state: u32 = 0xbeef_cafe;
        for _ in 0..rows * hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            raw_x.push(((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0);
        }
        for _ in 0..hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            raw_w.push((((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * 0.1);
        }
        for _ in 0..rows * hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            raw_g.push((((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * 0.3);
        }

        let x = Tensor::from_vec(raw_x, (rows, hidden), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::from_vec(raw_w, (hidden,), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let g = Tensor::from_vec(raw_g, (rows, hidden), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let (gx_ref, gw_ref) = rmsnorm_backward_fallback(&x, &w, &g, eps).unwrap();
        let (gx_fused, gw_fused) = fused_rmsnorm_backward(&x, &w, &g, eps as f32).unwrap();

        let gx_diff = max_abs_diff(&gx_ref, &gx_fused);
        let gw_diff = max_abs_diff(&gw_ref, &gw_fused);

        // bf16 round-trip + atomicAdd ordering — match the forward kernel
        // tolerance.
        assert!(
            gx_diff < 1e-2,
            "CUDA grad_x parity failed: max_abs_diff={gx_diff} (tol=1e-2)"
        );
        assert!(
            gw_diff < 1e-2,
            "CUDA grad_w parity failed: max_abs_diff={gw_diff} (tol=1e-2)"
        );
    }

    #[test]
    fn lora_decode_add_parity_cuda() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        let batch = 2usize;
        let in_dim = 96usize;
        let out_dim = 128usize;
        let rank = 4usize;
        let scale = 0.375f32;
        let mut state: u32 = 0x51f1_0a0a;
        let mut next = |mul: f32| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * mul
        };
        let base_raw: Vec<f32> = (0..batch * out_dim).map(|_| next(0.2)).collect();
        let x_raw: Vec<f32> = (0..batch * in_dim).map(|_| next(0.2)).collect();
        let a_raw: Vec<f32> = (0..rank * in_dim).map(|_| next(0.05)).collect();
        let b_raw: Vec<f32> = (0..out_dim * rank).map(|_| next(0.05)).collect();

        let base = Tensor::from_vec(base_raw, (batch, 1, out_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let x = Tensor::from_vec(x_raw, (batch, 1, in_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let a = Tensor::from_vec(a_raw, (rank, in_dim), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let b = Tensor::from_vec(b_raw, (out_dim, rank), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        assert!(supports_lora_decode_add(&base, &x, &a, &b));
        let hidden = x.broadcast_matmul(&a.t().unwrap()).unwrap();
        let delta = hidden.broadcast_matmul(&b.t().unwrap()).unwrap();
        let delta = (delta * scale as f64)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let reference = (base.clone() + delta).unwrap();
        let fused = lora_decode_add(&base, &x, &a, &b, scale).unwrap();
        let diff = max_abs_diff(&reference, &fused);
        assert!(
            diff < 2e-2,
            "CUDA lora_decode_add parity failed: max_abs_diff={diff} (tol=2e-2)"
        );
    }

    #[test]
    fn parity_backward_multi_row_cuda() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Prefill-like shape: [batch=1, seq=512, hidden=2560].
        let hidden = 2560usize;
        let rows = 512usize;
        let eps = 1e-6;

        let mut raw_x = Vec::with_capacity(rows * hidden);
        let mut raw_w = Vec::with_capacity(hidden);
        let mut raw_g = Vec::with_capacity(rows * hidden);
        let mut state: u32 = 0x0bad_d00d;
        for _ in 0..rows * hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            raw_x.push(((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0);
        }
        for _ in 0..hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            raw_w.push((((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * 0.1);
        }
        for _ in 0..rows * hidden {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            raw_g.push((((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * 0.3);
        }

        let x = Tensor::from_vec(raw_x, (rows, hidden), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::from_vec(raw_w, (hidden,), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let g = Tensor::from_vec(raw_g, (rows, hidden), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let (gx_ref, gw_ref) = rmsnorm_backward_fallback(&x, &w, &g, eps).unwrap();
        let (gx_fused, gw_fused) = fused_rmsnorm_backward(&x, &w, &g, eps as f32).unwrap();

        let gx_diff = max_abs_diff(&gx_ref, &gx_fused);
        let gw_diff = max_abs_diff(&gw_ref, &gw_fused);

        // grad_w accumulates 512 rows of bf16-cast values via atomicAdd in
        // F32, then a final bf16 cast. atomicAdd is order-nondeterministic
        // and the candle reference does the cross-row reduction in a fixed
        // tree order, so the bf16 outputs can differ by ~1 ULP near
        // boundaries where F32 results straddle a bf16 quantum (e.g.
        // 0.015625 = 2^-6 at typical magnitudes). Tolerance reflects that.
        assert!(
            gx_diff < 1e-2,
            "CUDA grad_x parity (multi-row) failed: max_abs_diff={gx_diff} (tol=1e-2)"
        );
        assert!(
            gw_diff < 2e-2,
            "CUDA grad_w parity (multi-row) failed: max_abs_diff={gw_diff} (tol=2e-2)"
        );
    }
}

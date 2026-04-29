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
    CpuStorage, CudaStorage, DType, Device, Layout, Result, Shape, Tensor,
    backend::BackendStorage,
    cuda_backend::cudarc::driver::DevicePtr,
};
use half::bf16;

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

    let out = Tensor::zeros(x_dims.as_slice(), DType::BF16, device)?;

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
///     c     = (1/H) * rms_inv^2 * sum_j ((1 + w_j) * x_j * grad_out_j)
///     dx_j  = rms_inv * ((1 + w_j) * grad_out_j - x_j * c)
///     dw_j  = sum_i (x_ij * rms_inv_i * grad_out_ij)
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
        candle_core::Error::Msg(
            "rmsnorm_backward_fallback: x must have rank >= 1".to_string(),
        )
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
    let weighted = x_f32
        .broadcast_mul(&w_plus_one)?
        .mul(&g_f32)?;
    let sum_xgw = weighted.sum_keepdim(last_dim)?;
    let c = (sum_xgw * inv_h)?
        .mul(&rms_inv)?
        .mul(&rms_inv)?;

    // grad_x = rms_inv * ((1+w) * grad_out - x * c)
    let term_a = g_f32.broadcast_mul(&w_plus_one)?;
    let term_b = x_f32.broadcast_mul(&c)?;
    let grad_x_f32 = (term_a - term_b)?.broadcast_mul(&rms_inv)?;
    let grad_x = grad_x_f32.to_dtype(dtype)?;

    // grad_w[j] = sum over leading dims of (x[..., j] * rms_inv[..., 0] * grad_out[..., j])
    let per_elem = x_f32
        .broadcast_mul(&rms_inv)?
        .mul(&g_f32)?;
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
    if x.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || grad_out.dtype() != DType::BF16
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
            return Ok((CudaStorage::wrap_cuda_slice(out_slice, device.clone()), shape));
        }

        let x_slice = s_x
            .as_cuda_slice::<bf16>()?
            .slice(l_x.start_offset()..);
        let w_slice = s_w
            .as_cuda_slice::<bf16>()?
            .slice(l_w.start_offset()..);

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

        Ok((CudaStorage::wrap_cuda_slice(out_slice, device.clone()), shape))
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

    let q_out = Tensor::zeros(dims.as_slice(), DType::BF16, device)?;
    let k_out = Tensor::zeros(dims.as_slice(), DType::BF16, device)?;

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

    let q_out = Tensor::zeros(&out_dims, DType::BF16, device)?;
    let k_out = Tensor::zeros(&out_dims, DType::BF16, device)?;

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
        let loss_a = (y_a.broadcast_mul(&g_const))
            .unwrap()
            .sum_all()
            .unwrap();
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

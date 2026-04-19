//! Vendored fused norm CUDA kernels (Liger-style).
//!
//! This crate hosts decode-critical Liger-style fused norm kernels for kiln:
//!
//! 1. [`fused_rmsnorm`] — Qwen3.5-style RMSNorm `(1 + w) * x * rsqrt(mean(x^2) + eps)`.
//!    Replaces the ~11 candle ops behind `kiln-model::forward::rms_norm`.
//!    Used by `kiln/norm/pre_attn` and `kiln/norm/pre_mlp`.
//! 2. [`fused_l2_qk_norm`] — fused L2-norm(Q) + scale(Q) + L2-norm(K) used by
//!    GDN linear attention. Replaces the ~11 candle ops behind the
//!    `kiln/gdn/qk_norm` block in `forward.rs`.
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
//! - [`supports`] — `(x, weight)` capability check for the RMSNorm kernel.
//! - [`fused_l2_qk_norm`] — candle-compatible wrapper around the GDN QK
//!   fused-norm kernel. Returns `(q_out, k_out)`.
//! - [`supports_l2_qk_norm`] — capability check for the QK kernel.
//!
//! # Envelope
//!
//! - bf16 activations, bf16 weights, bf16 outputs.
//! - Contiguous CUDA tensors only.
//! - Last dim (`hidden`) must be <= 8192.
//! - `eps` is F32 — kiln uses 1e-6 for both kernels.
//!
//! Out of scope: backward pass, fused GEMM prologue, non-bf16 dtypes,
//! non-contiguous input.

use candle_core::{
    backend::BackendStorage,
    cuda_backend::cudarc::driver::DevicePtr,
    DType, Device, Result, Tensor,
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
        let (q_fused, k_fused) =
            fused_l2_qk_norm(&q, &k, q_scale as f32, eps as f32).unwrap();

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
        let (q_fused, k_fused) =
            fused_l2_qk_norm(&q, &k, q_scale as f32, eps as f32).unwrap();

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
        let (q_fused, k_fused) =
            fused_l2_qk_norm(&q, &k, q_scale as f32, eps as f32).unwrap();

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
}

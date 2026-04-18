//! Vendored fused RMSNorm CUDA kernel (Liger-style).
//!
//! # Why
//!
//! `kiln-model::forward::rms_norm` currently expresses Qwen3.5-style
//! RMSNorm as a chain of candle ops:
//!
//! ```text
//! to_dtype(F32) → sqr → mean_keepdim → + eps → sqrt → recip →
//! broadcast_mul → to_dtype(F32 weight) → ones_like + w →
//! broadcast_mul → to_dtype(bf16)
//! ```
//!
//! That is ~11 CUDA kernel launches per RMSNorm. At decode time each
//! launch has ~10 µs of per-launch overhead, and the intermediate F32
//! tensors round-trip through HBM on every step. Per PROFILING.md
//! (post-PR #130), the two RMSNorm NVTX ranges `kiln/norm/pre_attn`
//! (11.1 %) and `kiln/norm/pre_mlp` (11.0 %) combine for ~22 % of
//! decode wallclock and are the #1 hotspot of the current profile.
//!
//! This crate collapses those ~11 launches into a single fused kernel
//! (`fused_rmsnorm_kernel` in `csrc/fused_rmsnorm.cu`): one block per
//! row, 256 threads per block, F32 reduction + F32 epilogue accumulator,
//! bf16 load/store.
//!
//! # Provenance
//!
//! Algorithm modelled after LinkedIn's Liger-Kernel `fused_rmsnorm`
//! (<https://github.com/linkedin/Liger-Kernel>, `src/liger_kernel/ops/rms_norm.py`),
//! reimplemented in raw CUDA C so kiln doesn't add a Triton runtime
//! dependency. Matches kiln's Qwen3.5 convention of `(1 + w) * x * rms_inv`
//! (weights are centred on 0, not on 1).
//!
//! # API
//!
//! - [`fused_rmsnorm`] — candle-compatible wrapper. Accepts any bf16 input
//!   whose last dim is the normalised axis; flattens leading dims, runs
//!   the kernel, and reshapes back. Only forward pass.
//!
//! # Envelope
//!
//! - bf16 activations, bf16 weights.
//! - Contiguous CUDA tensors only.
//! - Last dim (`hidden`) must be <= 8192. Qwen3.5-4B uses 2560.
//! - `eps` is F32 — kiln uses 1e-6 for Qwen3.5.
//!
//! Out of scope: backward pass, fused GEMM prologue, non-bf16 dtypes,
//! non-contiguous input, unpaired cast (`weight` of a different dtype).

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
}

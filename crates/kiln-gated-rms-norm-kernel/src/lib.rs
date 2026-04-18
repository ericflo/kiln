//! Vendored fused gated RMSNorm CUDA kernel (FLA `fused_norm_gate`-style).
//!
//! # Why
//!
//! `kiln-model::forward::gated_rms_norm` (called from `gated_deltanet_forward`
//! under NVTX range `kiln/gdn/gated_norm`) expresses the GDN gated-norm
//! epilogue as a chain of candle ops:
//!
//! ```text
//! to_dtype(x→F32) → to_dtype(z→F32) → to_dtype(w→F32) →
//! sqr → mean_keepdim → + eps → sqrt → recip →
//! broadcast_mul(x, rms_inv) → broadcast_mul(*, w) →
//! silu(z) → mul(normed, gate)
//! ```
//!
//! That is ~11 CUDA kernel launches per GDN layer. At decode time each
//! launch has ~5 µs of per-launch overhead, and the intermediate F32
//! tensors round-trip through HBM on every step. Per PROFILING.md
//! (post-PR #135), the `kiln/gdn/gated_norm` NVTX range is **13.66 %**
//! of paged decode wallclock — the largest remaining single hotspot.
//!
//! This crate collapses those launches into a single fused kernel
//! (`fused_gated_rmsnorm_kernel` in `csrc/fused_gated_rmsnorm.cu`): one
//! block per row, warp-aligned threads per block, F32 reduction + F32
//! epilogue accumulator, bf16 load/store.
//!
//! # Provenance
//!
//! Algorithm modelled after fla-org/flash-linear-attention's
//! `fused_norm_gate` (<https://github.com/fla-org/flash-linear-attention>,
//! `fla/modules/fused_norm_gate.py`), reimplemented in raw CUDA C so
//! kiln doesn't add a Triton runtime dependency. Note that unlike the
//! transformer-block RMSNorm used in kiln-rmsnorm-kernel (Qwen3.5
//! `(1 + w)` convention), the GDN gated RMSNorm uses the standard
//! `w * x * rms_inv` form with weights centred on 1.
//!
//! # API
//!
//! - [`fused_gated_rms_norm`] — candle-compatible wrapper. Accepts any
//!   bf16 input whose last dim is the normalised axis; flattens leading
//!   dims, runs the kernel, and reshapes back. Returns bf16 directly so
//!   the caller avoids the F32→bf16 epilogue cast.
//!
//! # Envelope
//!
//! - bf16 activations, bf16 weights.
//! - Contiguous CUDA tensors only.
//! - Last dim (`hidden`) must be <= 8192. Qwen3.5-4B uses 128 here
//!   (`linear_value_head_dim`).
//! - `eps` is F32 — kiln uses 1e-6 for Qwen3.5.
//!
//! Out of scope: backward pass, non-bf16 dtypes, non-contiguous input,
//! fused prologue (the fused single-pass epilogue is already the minimum
//! viable ship to collapse the 10-op chain).

use candle_core::{
    backend::BackendStorage,
    cuda_backend::cudarc::driver::DevicePtr,
    DType, Device, Result, Tensor,
};
use half::bf16;

unsafe extern "C" {
    fn kiln_fused_gated_rmsnorm(
        x: *const core::ffi::c_void,
        z: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        rows: i32,
        hidden: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Whether the fused gated RMSNorm kernel is available on the given tensors.
///
/// Requires CUDA + bf16 + `hidden <= 8192`. `x` and `z` must share the same
/// shape and `weight` must be `[hidden]`. Non-contiguous inputs are accepted —
/// the wrapper materialises contiguous copies internally before launching the
/// kernel. This is load-bearing at the GDN callsite: `attn_out` enters
/// `gated_rms_norm` with strided layout (post-`transpose(1, 2)`), and requiring
/// contiguity here would force a fallback to the candle-op path.
pub fn supports(x: &Tensor, z: &Tensor, weight: &Tensor) -> bool {
    if !matches!(x.device(), Device::Cuda(_)) {
        return false;
    }
    if x.dtype() != DType::BF16 || z.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return false;
    }
    if x.dims() != z.dims() || x.rank() < 1 {
        return false;
    }
    let hidden = x.dims().last().copied().unwrap_or(0);
    if hidden == 0 || hidden > 8192 {
        return false;
    }
    weight.dims() == &[hidden]
}

/// Run the fused gated RMSNorm kernel.
///
/// Inputs:
///   - `x`: bf16, CUDA, contiguous, any rank; last dim is the normalised axis.
///   - `z`: bf16, CUDA, contiguous, same shape as `x` (output gate input).
///   - `weight`: bf16, CUDA, contiguous, shape `[hidden]` matching `x.last_dim()`.
///   - `eps`: epsilon inside the rsqrt. Qwen3.5 uses 1e-6.
///
/// Returns a freshly allocated bf16 tensor with the same shape as `x`.
///
/// Semantics: `out = weight * x * rsqrt(mean(x^2, dim=-1) + eps) * silu(z)`
/// cast back to bf16. Matches `kiln-model::forward::gated_rms_norm` (standard
/// weight convention centred on 1, not the `(1 + w)` form used by the main
/// transformer RMSNorm).
pub fn fused_gated_rms_norm(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let device = x.device();

    if x.dtype() != DType::BF16 || z.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        candle_core::bail!(
            "kiln-gated-rms-norm-kernel: x/z/weight must all be bf16 (got {:?}, {:?}, {:?})",
            x.dtype(),
            z.dtype(),
            weight.dtype()
        );
    }

    if x.dims() != z.dims() {
        candle_core::bail!(
            "kiln-gated-rms-norm-kernel: x and z shapes must match (got {:?} vs {:?})",
            x.dims(),
            z.dims()
        );
    }

    let x_dims = x.dims().to_vec();
    let hidden = *x_dims.last().ok_or_else(|| {
        candle_core::Error::Msg(
            "kiln-gated-rms-norm-kernel: x must have rank >= 1".to_string(),
        )
    })?;

    let weight_dims = weight.dims();
    if weight_dims.len() != 1 || weight_dims[0] != hidden {
        candle_core::bail!(
            "kiln-gated-rms-norm-kernel: weight shape {:?} does not match x last dim {hidden}",
            weight_dims
        );
    }

    if hidden > 8192 {
        candle_core::bail!(
            "kiln-gated-rms-norm-kernel: hidden dim {hidden} exceeds kernel envelope (<= 8192)"
        );
    }

    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    if rows == 0 {
        // Empty leading axis — nothing to do.
        return Tensor::zeros(x_dims.as_slice(), DType::BF16, device);
    }

    let x = x.contiguous()?;
    let z = z.contiguous()?;
    let weight = weight.contiguous()?;

    let out = Tensor::zeros(x_dims.as_slice(), DType::BF16, device)?;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (z_storage, z_layout) = z.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();

        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gated-rms-norm-kernel: x must be on CUDA"),
        };
        let z_cuda = match &*z_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gated-rms-norm-kernel: z must be on CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gated-rms-norm-kernel: weight must be on CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gated-rms-norm-kernel: out must be on CUDA"),
        };

        let stream = x_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let x_slice = x_cuda
            .as_cuda_slice::<bf16>()?
            .slice(x_layout.start_offset()..);
        let z_slice = z_cuda
            .as_cuda_slice::<bf16>()?
            .slice(z_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<bf16>()?
            .slice(w_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<bf16>()?
            .slice(o_layout.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (z_ptr, _g2) = z_slice.device_ptr(&stream);
            let (w_ptr, _g3) = w_slice.device_ptr(&stream);
            let (o_ptr, _g4) = o_slice.device_ptr(&stream);

            let status = kiln_fused_gated_rmsnorm(
                x_ptr as *const _,
                z_ptr as *const _,
                w_ptr as *const _,
                o_ptr as *mut _,
                rows as i32,
                hidden as i32,
                eps,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_fused_gated_rmsnorm failed with status {status}");
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    // Reference implementation — mirrors `kiln-model::forward::gated_rms_norm`
    // exactly (F32 compute, standard RMSNorm weight convention centred on 1,
    // silu(z) output gate). Used as the correctness oracle for
    // `fused_gated_rms_norm`. Returns bf16 so it can be compared directly.
    fn reference_gated_rms_norm(
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f64,
    ) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let z_f32 = z.to_dtype(DType::F32)?;
        let w_f32 = weight.to_dtype(DType::F32)?;

        let variance = x_f32.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let rms_inv = (variance + eps)?.sqrt()?.recip()?;
        let normed = x_f32.broadcast_mul(&rms_inv)?;
        let normed = normed.broadcast_mul(&w_f32)?;

        // silu(z) = z * sigmoid(z)
        let neg_z = z_f32.neg()?;
        let sigmoid = (neg_z.exp()? + 1.0)?.recip()?;
        let gate = (&z_f32 * &sigmoid)?;
        let out = (normed * gate)?;
        out.to_dtype(x.dtype())
    }

    fn try_cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    fn rand_tensor(shape: &[usize], seed: u32, scale: f32, device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut raw = Vec::with_capacity(n);
        let mut state: u32 = seed;
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let f = ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0;
            raw.push(f * scale);
        }
        Tensor::from_vec(raw, shape, device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
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

    #[test]
    fn parity_decode_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Qwen3.5-4B decode GDN shape: [batch=1, seq=1, nv=32, dv=128].
        // Flattens to rows=32, hidden=128.
        let shape = &[1usize, 1, 32, 128];
        let hidden = 128usize;
        let eps = 1e-6;

        let x = rand_tensor(shape, 0x1234_5678, 0.5, &device);
        let z = rand_tensor(shape, 0xfeed_face, 0.5, &device);
        let w = rand_tensor(&[hidden], 0xbeef_cafe, 0.1, &device);

        let y_ref = reference_gated_rms_norm(&x, &z, &w, eps).unwrap();
        let y_fused = fused_gated_rms_norm(&x, &z, &w, eps as f32).unwrap();

        let diff = max_abs_diff(&y_ref, &y_fused);
        assert!(
            diff < 1e-2,
            "decode parity failed: max_abs_diff={diff} exceeds 1e-2"
        );
    }

    #[test]
    fn parity_prefill_shape() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Prefill GDN shape: [batch=1, seq=512, nv=32, dv=128].
        let shape = &[1usize, 512, 32, 128];
        let hidden = 128usize;
        let eps = 1e-6;

        let x = rand_tensor(shape, 0xcafe_babe, 0.7, &device);
        let z = rand_tensor(shape, 0xdead_beef, 0.7, &device);
        let w = rand_tensor(&[hidden], 0x1337_cafe, 0.1, &device);

        let y_ref = reference_gated_rms_norm(&x, &z, &w, eps).unwrap();
        let y_fused = fused_gated_rms_norm(&x, &z, &w, eps as f32).unwrap();

        let diff = max_abs_diff(&y_ref, &y_fused);
        assert!(
            diff < 1e-2,
            "prefill parity failed: max_abs_diff={diff} exceeds 1e-2"
        );
    }

    #[test]
    fn parity_batch_dim() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Batched shape: [batch=2, seq=3, nv=32, dv=128] — exercises the 4D
        // reshape path.
        let shape = &[2usize, 3, 32, 128];
        let hidden = 128usize;
        let eps = 1e-6;

        let x = rand_tensor(shape, 0x0f0f_f0f0, 0.3, &device);
        let z = rand_tensor(shape, 0xa5a5_5a5a, 0.3, &device);
        let w = rand_tensor(&[hidden], 0x9876_5432, 0.1, &device);

        let y_ref = reference_gated_rms_norm(&x, &z, &w, eps).unwrap();
        let y_fused = fused_gated_rms_norm(&x, &z, &w, eps as f32).unwrap();

        assert_eq!(y_fused.dims(), shape);

        let diff = max_abs_diff(&y_ref, &y_fused);
        assert!(
            diff < 1e-2,
            "batch parity failed: max_abs_diff={diff} exceeds 1e-2"
        );
    }

    #[test]
    fn parity_wider_hidden() {
        let Some(device) = try_cuda_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };

        // Exercise hidden > kThreadsPerBlock (256) to verify the strided loop.
        let shape = &[8usize, 1024];
        let hidden = 1024usize;
        let eps = 1e-6;

        let x = rand_tensor(shape, 0x1111_2222, 0.5, &device);
        let z = rand_tensor(shape, 0x3333_4444, 0.5, &device);
        let w = rand_tensor(&[hidden], 0x5555_6666, 0.1, &device);

        let y_ref = reference_gated_rms_norm(&x, &z, &w, eps).unwrap();
        let y_fused = fused_gated_rms_norm(&x, &z, &w, eps as f32).unwrap();

        let diff = max_abs_diff(&y_ref, &y_fused);
        assert!(
            diff < 1e-2,
            "wider-hidden parity failed: max_abs_diff={diff} exceeds 1e-2"
        );
    }
}

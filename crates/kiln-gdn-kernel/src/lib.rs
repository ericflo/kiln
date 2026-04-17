//! Vendored Gated DeltaNet (GDN) chunk forward-substitution CUDA kernel.
//!
//! This crate provides [`gdn_forward_substitution`], a thin candle wrapper
//! around a single fused CUDA kernel that replaces the per-token forward
//! substitution loop in kiln's chunkwise analytical GDN recurrence:
//!
//! ```text
//! W[t, :] = beta[t] * ( V_prime[t, :]
//!                      - sum_{i<t} A_strict[t, i] * W[i, :] )
//! ```
//!
//! The kernel is intentionally narrow:
//!   - bf16 activations only.
//!   - `dv` <= 1024 (kiln uses 128).
//!   - `chunk_size` <= 128 (kiln uses 64).
//!   - One block per (batch, head); no tensor-core path.
//!
//! Anything outside that envelope should fall back to the Rust+candle
//! implementation in `kiln-model::forward`.

use candle_core::{
    backend::BackendStorage,
    cuda_backend::cudarc::driver::DevicePtr,
    DType, Result, Tensor,
};
use half::bf16;

unsafe extern "C" {
    fn kiln_gdn_forward_substitution(
        a_strict: *const core::ffi::c_void,
        v_prime: *const core::ffi::c_void,
        beta: *const core::ffi::c_void,
        w_out: *mut core::ffi::c_void,
        batch_heads: i32,
        chunk_size: i32,
        dv: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Run the fused GDN forward-substitution kernel.
///
/// Inputs (all bf16, all CUDA, all contiguous):
///   - `a_strict`: `[B, H, C, C]`
///   - `v_prime`:  `[B, H, C, dv]`
///   - `beta`:     `[B, H, C]`
///
/// Returns a freshly allocated bf16 tensor `W` with shape `[B, H, C, dv]`.
pub fn gdn_forward_substitution(
    a_strict: &Tensor,
    v_prime: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let device = a_strict.device();

    let (b, h, c, c2) = a_strict.dims4()?;
    if c != c2 {
        candle_core::bail!(
            "kiln-gdn-kernel: a_strict must be square in the last two dims, got [{b}, {h}, {c}, {c2}]"
        );
    }

    let (b_v, h_v, c_v, dv) = v_prime.dims4()?;
    if (b_v, h_v, c_v) != (b, h, c) {
        candle_core::bail!(
            "kiln-gdn-kernel: v_prime shape [{b_v}, {h_v}, {c_v}, {dv}] mismatch with a_strict"
        );
    }

    let (b_b, h_b, c_b) = beta.dims3()?;
    if (b_b, h_b, c_b) != (b, h, c) {
        candle_core::bail!(
            "kiln-gdn-kernel: beta shape [{b_b}, {h_b}, {c_b}] mismatch with a_strict"
        );
    }

    if a_strict.dtype() != DType::BF16
        || v_prime.dtype() != DType::BF16
        || beta.dtype() != DType::BF16
    {
        candle_core::bail!(
            "kiln-gdn-kernel: all inputs must be bf16 (got {:?}, {:?}, {:?})",
            a_strict.dtype(),
            v_prime.dtype(),
            beta.dtype()
        );
    }

    if c > 128 {
        candle_core::bail!(
            "kiln-gdn-kernel: chunk_size must be <= 128 (got {c})"
        );
    }
    if dv > 1024 {
        candle_core::bail!(
            "kiln-gdn-kernel: dv must be <= 1024 (got {dv})"
        );
    }

    let a_strict = a_strict.contiguous()?;
    let v_prime = v_prime.contiguous()?;
    let beta = beta.contiguous()?;

    let w_out = Tensor::zeros((b, h, c, dv), DType::BF16, device)?;

    {
        let (a_storage, a_layout) = a_strict.storage_and_layout();
        let (vp_storage, vp_layout) = v_prime.storage_and_layout();
        let (beta_storage, beta_layout) = beta.storage_and_layout();
        let (w_storage, w_layout) = w_out.storage_and_layout();

        let a_cuda = match &*a_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: a_strict must be on CUDA"),
        };
        let vp_cuda = match &*vp_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: v_prime must be on CUDA"),
        };
        let beta_cuda = match &*beta_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: beta must be on CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: w_out must be on CUDA"),
        };

        let stream = a_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let a_slice = a_cuda.as_cuda_slice::<bf16>()?;
        let vp_slice = vp_cuda.as_cuda_slice::<bf16>()?;
        let beta_slice = beta_cuda.as_cuda_slice::<bf16>()?;
        let w_slice = w_cuda.as_cuda_slice::<bf16>()?;

        let a_slice = a_slice.slice(a_layout.start_offset()..);
        let vp_slice = vp_slice.slice(vp_layout.start_offset()..);
        let beta_slice = beta_slice.slice(beta_layout.start_offset()..);
        let w_slice = w_slice.slice(w_layout.start_offset()..);

        unsafe {
            let (a_ptr, _g1) = a_slice.device_ptr(&stream);
            let (vp_ptr, _g2) = vp_slice.device_ptr(&stream);
            let (beta_ptr, _g3) = beta_slice.device_ptr(&stream);
            let (w_ptr, _g4) = w_slice.device_ptr(&stream);

            let status = kiln_gdn_forward_substitution(
                a_ptr as *const _,
                vp_ptr as *const _,
                beta_ptr as *const _,
                w_ptr as *mut _,
                (b * h) as i32,
                c as i32,
                dv as i32,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!(
                    "kiln_gdn_forward_substitution failed with status {status}"
                );
            }
        }
    }

    Ok(w_out)
}

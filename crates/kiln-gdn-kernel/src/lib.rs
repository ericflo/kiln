//! Vendored Gated DeltaNet (GDN) chunk forward-substitution CUDA kernel.
//!
//! # Provenance: fla-org `chunk_gla_fwd`
//!
//! This crate is the vendored port of
//! [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)'s
//! `chunk_gla_fwd` Triton kernel (source: `fla/ops/gla/chunk.py`) into
//! raw CUDA C. It landed as PR #80 (commit `0c9c519`) and fulfills the
//! "Phase 6 — Vendor fla-org chunk_gla_fwd (minimal)" item from the
//! project's performance-optimization queue in the project description.
//!
//! Any follow-up planning task titled "vendor chunk_gla_fwd" should
//! re-verify current `PROFILING.md` before opening a new PR — the core
//! vendor is here, and the per-token Rust forward-sub loop it replaced
//! is gone from the CUDA path. The remaining candle ops in
//! `kiln-model::forward::gdn_chunkwise_recurrence` (cumsum + exp decay
//! matrix, KKT/QKT matmuls, intra-chunk `B_mask @ W`, final state
//! update) are *not* inside this vendor's scope; they are distinct
//! operations the scheduler launches per chunk.
//!
//! # API
//!
//! - [`gdn_forward_substitution`] — chunkwise prefill forward-sub step.
//!   Thin candle wrapper around a single fused CUDA kernel that
//!   replaces the per-token forward-substitution loop in kiln's
//!   chunkwise analytical GDN recurrence:
//!
//!   ```text
//!   W[t, :] = beta[t] * ( V_prime[t, :]
//!                        - sum_{i<t} A_strict[t, i] * W[i, :] )
//!   ```
//!
//! - [`gdn_recurrent_forward`] — seq_len==1 decode fast path. Collapses
//!   the single-token GDN recurrence (decay, delta, state-update,
//!   output projection) into one block per `(batch, head)`.
//!
//! # Envelope
//!
//! The kernels are intentionally narrow (per the project's
//! "minimal-scope vendoring" policy):
//!
//!   - bf16 activations, F32 accumulators inside the kernel.
//!   - Causal / forward-pass only.
//!   - `dv` <= 1024 (kiln uses 128).
//!   - `chunk_size` <= 128 (kiln uses 64) for forward-sub.
//!   - `dk` <= 256 (kiln uses 128) for recurrent.
//!   - One CUDA block per `(batch, head)`; no tensor-core path.
//!
//! Anything outside that envelope falls back to the Rust+candle
//! reference in `kiln-model::forward::compute_w_chunk_fallback`, which
//! also serves as the correctness oracle for
//! `test_gdn_kernel_matches_fallback`.
//!
//! # Not yet vendored
//!
//! Per `PROFILING.md` (post-PR #130, Phase 6), the next GDN-side
//! targets are the GDN body ranges (`gated_norm`, `gates`, `conv`,
//! `qk_norm`) and the two RMSNorm stages — these are upstream of the
//! chunkwise recurrence and are *not* covered by this crate.

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

    fn kiln_gdn_recurrent_forward(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        beta: *const core::ffi::c_void,
        g: *const core::ffi::c_void,
        state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch_heads: i32,
        dk: i32,
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

/// Run the fused single-token GDN recurrent kernel.
///
/// Inputs (all bf16, all CUDA, all contiguous):
///   - `q`     : `[B, H, dk]`
///   - `k`     : `[B, H, dk]`
///   - `v`     : `[B, H, dv]`
///   - `beta`  : `[B, H]`
///   - `g`     : `[B, H]`
///   - `state` : `[B, H, dk, dv]` — read-modify-write in place
///
/// Returns a freshly allocated bf16 tensor `out` with shape `[B, H, dv]`.
///
/// This is the seq_len == 1 / chunk_size == 1 decode path. The kernel
/// folds the per-token GDN recurrence (decay, delta, state-update, output
/// projection) into a single block-per-(batch, head) launch so we don't
/// pay the chunkwise candle-op overhead at decode time.
pub fn gdn_recurrent_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let device = q.device();

    let (b, h, dk) = q.dims3()?;
    let (b_k, h_k, dk_k) = k.dims3()?;
    if (b_k, h_k, dk_k) != (b, h, dk) {
        candle_core::bail!(
            "kiln-gdn-kernel: k shape [{b_k}, {h_k}, {dk_k}] mismatch with q [{b}, {h}, {dk}]"
        );
    }

    let (b_v, h_v, dv) = v.dims3()?;
    if (b_v, h_v) != (b, h) {
        candle_core::bail!(
            "kiln-gdn-kernel: v shape [{b_v}, {h_v}, {dv}] mismatch with q [{b}, {h}, *]"
        );
    }

    let (b_b, h_b) = beta.dims2()?;
    if (b_b, h_b) != (b, h) {
        candle_core::bail!(
            "kiln-gdn-kernel: beta shape [{b_b}, {h_b}] mismatch with q [{b}, {h}, *]"
        );
    }

    let (b_g, h_g) = g.dims2()?;
    if (b_g, h_g) != (b, h) {
        candle_core::bail!(
            "kiln-gdn-kernel: g shape [{b_g}, {h_g}] mismatch with q [{b}, {h}, *]"
        );
    }

    let (b_s, h_s, dk_s, dv_s) = state.dims4()?;
    if (b_s, h_s, dk_s, dv_s) != (b, h, dk, dv) {
        candle_core::bail!(
            "kiln-gdn-kernel: state shape [{b_s}, {h_s}, {dk_s}, {dv_s}] mismatch with [{b}, {h}, {dk}, {dv}]"
        );
    }

    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || beta.dtype() != DType::BF16
        || g.dtype() != DType::BF16
        || state.dtype() != DType::BF16
    {
        candle_core::bail!(
            "kiln-gdn-kernel: all inputs must be bf16 (got q={:?}, k={:?}, v={:?}, beta={:?}, g={:?}, state={:?})",
            q.dtype(), k.dtype(), v.dtype(), beta.dtype(), g.dtype(), state.dtype()
        );
    }

    if dk > 256 {
        candle_core::bail!("kiln-gdn-kernel: dk must be <= 256 (got {dk})");
    }
    if dv > 1024 {
        candle_core::bail!("kiln-gdn-kernel: dv must be <= 1024 (got {dv})");
    }

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    // `state` is mutated in place by the kernel. If the caller passed a
    // non-contiguous view, materialize a contiguous copy and rebind so the
    // kernel writes land in storage the caller still holds a handle to.
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }

    let out = Tensor::zeros((b, h, dv), DType::BF16, device)?;

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (beta_storage, beta_layout) = beta.storage_and_layout();
        let (g_storage, g_layout) = g.storage_and_layout();
        let (s_storage, s_layout) = state.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: q must be on CUDA"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: k must be on CUDA"),
        };
        let v_cuda = match &*v_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: v must be on CUDA"),
        };
        let beta_cuda = match &*beta_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: beta must be on CUDA"),
        };
        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: g must be on CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: state must be on CUDA"),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: out must be on CUDA"),
        };

        let stream = q_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let q_slice = q_cuda.as_cuda_slice::<bf16>()?.slice(q_layout.start_offset()..);
        let k_slice = k_cuda.as_cuda_slice::<bf16>()?.slice(k_layout.start_offset()..);
        let v_slice = v_cuda.as_cuda_slice::<bf16>()?.slice(v_layout.start_offset()..);
        let beta_slice = beta_cuda.as_cuda_slice::<bf16>()?.slice(beta_layout.start_offset()..);
        let g_slice = g_cuda.as_cuda_slice::<bf16>()?.slice(g_layout.start_offset()..);
        let s_slice = s_cuda.as_cuda_slice::<bf16>()?.slice(s_layout.start_offset()..);
        let out_slice = out_cuda.as_cuda_slice::<bf16>()?.slice(out_layout.start_offset()..);

        unsafe {
            let (q_ptr, _g1) = q_slice.device_ptr(&stream);
            let (k_ptr, _g2) = k_slice.device_ptr(&stream);
            let (v_ptr, _g3) = v_slice.device_ptr(&stream);
            let (beta_ptr, _g4) = beta_slice.device_ptr(&stream);
            let (g_ptr, _g5) = g_slice.device_ptr(&stream);
            let (s_ptr, _g6) = s_slice.device_ptr(&stream);
            let (out_ptr, _g7) = out_slice.device_ptr(&stream);

            let status = kiln_gdn_recurrent_forward(
                q_ptr as *const _,
                k_ptr as *const _,
                v_ptr as *const _,
                beta_ptr as *const _,
                g_ptr as *const _,
                s_ptr as *mut _,
                out_ptr as *mut _,
                (b * h) as i32,
                dk as i32,
                dv as i32,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!(
                    "kiln_gdn_recurrent_forward failed with status {status}"
                );
            }
        }
    }

    Ok(out)
}

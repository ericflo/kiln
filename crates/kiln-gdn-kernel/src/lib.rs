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

    #[allow(clippy::too_many_arguments)]
    fn kiln_gdn_recurrent_forward_fused_norm(
        q_raw: *const core::ffi::c_void,
        k_raw: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        beta: *const core::ffi::c_void,
        g: *const core::ffi::c_void,
        z: *const core::ffi::c_void,
        gamma: *const core::ffi::c_void,
        state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch_heads: i32,
        dk: i32,
        dv: i32,
        q_scale: f32,
        l2_eps: f32,
        rms_eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_chunk_prep(
        g: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        kkt: *const core::ffi::c_void,
        qkt: *const core::ffi::c_void,
        ks_entry: *const core::ffi::c_void,
        q_s: *const core::ffi::c_void,
        a_strict: *mut core::ffi::c_void,
        b_mask: *mut core::ffi::c_void,
        v_prime: *mut core::ffi::c_void,
        q_s_scaled: *mut core::ffi::c_void,
        decay_last_col: *mut core::ffi::c_void,
        p_last: *mut core::ffi::c_void,
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

/// Cheap shape / dtype check for [`gdn_recurrent_forward_fused_norm`].
///
/// Returns `true` when the caller-provided shapes/dtypes are inside the
/// fused kernel's envelope. The fused kernel inherits the non-fused
/// kernel's envelope (bf16, `dk <= 256`, `dv <= 1024`) plus:
///
///   - `z.shape() == v.shape()` — one gate vector per (batch, head)
///   - `gamma.shape() == [dv]` — learnable RMSNorm scale, shared
pub fn gdn_recurrent_forward_fused_norm_supports(
    q_raw: &Tensor,
    k_raw: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    z: &Tensor,
    gamma: &Tensor,
    state: &Tensor,
) -> bool {
    if q_raw.dtype() != DType::BF16
        || k_raw.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || beta.dtype() != DType::BF16
        || g.dtype() != DType::BF16
        || z.dtype() != DType::BF16
        || gamma.dtype() != DType::BF16
        || state.dtype() != DType::BF16
    {
        return false;
    }
    let Ok((b, h, dk)) = q_raw.dims3() else { return false };
    let Ok((b_k, h_k, dk_k)) = k_raw.dims3() else { return false };
    if (b_k, h_k, dk_k) != (b, h, dk) {
        return false;
    }
    let Ok((b_v, h_v, dv)) = v.dims3() else { return false };
    if (b_v, h_v) != (b, h) {
        return false;
    }
    let Ok((b_z, h_z, dv_z)) = z.dims3() else { return false };
    if (b_z, h_z, dv_z) != (b, h, dv) {
        return false;
    }
    let Ok((dv_g,)) = gamma.dims1().map(|d| (d,)) else { return false };
    if dv_g != dv {
        return false;
    }
    let Ok((b_s, h_s, dk_s, dv_s)) = state.dims4() else { return false };
    if (b_s, h_s, dk_s, dv_s) != (b, h, dk, dv) {
        return false;
    }
    let Ok((b_b, h_b)) = beta.dims2() else { return false };
    if (b_b, h_b) != (b, h) {
        return false;
    }
    let Ok((b_g, h_g)) = g.dims2() else { return false };
    if (b_g, h_g) != (b, h) {
        return false;
    }
    dk <= 256 && dv <= 1024
}

/// Run the fused GDN recurrent-step-with-norms kernel.
///
/// Inputs (all bf16, all CUDA; not necessarily contiguous — contiguous
/// copies are materialised internally):
///   - `q_raw` : `[B, H, dk]` — pre-L2-norm Q (post conv1d + silu)
///   - `k_raw` : `[B, H, dk]` — pre-L2-norm K
///   - `v`     : `[B, H, dv]`
///   - `beta`  : `[B, H]`
///   - `g`     : `[B, H]` — gate (kernel applies `exp`)
///   - `z`     : `[B, H, dv]` — output gate (pre-silu)
///   - `gamma` : `[dv]` — RMSNorm learnable scale (shared across B, H)
///   - `state` : `[B, H, dk, dv]` — read-modify-write in place
///
/// Scalars:
///   - `q_scale` — multiplier on Q after L2 norm (typically `1/sqrt(dk)`)
///   - `l2_eps`  — eps inside sqrt for L2 norm (typically `1e-6`)
///   - `rms_eps` — eps inside sqrt for RMSNorm (typically `1e-6`)
///
/// Returns a freshly allocated bf16 `out` tensor with shape `[B, H, dv]`
/// already multiplied by `gamma * silu(z)`. Semantics match
/// `kiln-model::forward::gated_deltanet_forward` when the input to qk_norm,
/// the recurrent step, and gated_norm is this exact (q_raw, k_raw, v,
/// beta, g, z, gamma) at `seq_len == 1`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_forward_fused_norm(
    q_raw: &Tensor,
    k_raw: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    z: &Tensor,
    gamma: &Tensor,
    state: &mut Tensor,
    q_scale: f32,
    l2_eps: f32,
    rms_eps: f32,
) -> Result<Tensor> {
    let device = q_raw.device();

    let (b, h, dk) = q_raw.dims3()?;
    let (b_k, h_k, dk_k) = k_raw.dims3()?;
    if (b_k, h_k, dk_k) != (b, h, dk) {
        candle_core::bail!(
            "kiln-gdn-kernel: k_raw shape [{b_k}, {h_k}, {dk_k}] mismatch with q_raw [{b}, {h}, {dk}]"
        );
    }

    let (b_v, h_v, dv) = v.dims3()?;
    if (b_v, h_v) != (b, h) {
        candle_core::bail!(
            "kiln-gdn-kernel: v shape [{b_v}, {h_v}, {dv}] mismatch with q_raw [{b}, {h}, *]"
        );
    }

    let (b_b, h_b) = beta.dims2()?;
    if (b_b, h_b) != (b, h) {
        candle_core::bail!(
            "kiln-gdn-kernel: beta shape [{b_b}, {h_b}] mismatch with q_raw [{b}, {h}, *]"
        );
    }

    let (b_g, h_g) = g.dims2()?;
    if (b_g, h_g) != (b, h) {
        candle_core::bail!(
            "kiln-gdn-kernel: g shape [{b_g}, {h_g}] mismatch with q_raw [{b}, {h}, *]"
        );
    }

    let (b_z, h_z, dv_z) = z.dims3()?;
    if (b_z, h_z, dv_z) != (b, h, dv) {
        candle_core::bail!(
            "kiln-gdn-kernel: z shape [{b_z}, {h_z}, {dv_z}] mismatch with v [{b}, {h}, {dv}]"
        );
    }

    let gamma_dims = gamma.dims();
    if gamma_dims.len() != 1 || gamma_dims[0] != dv {
        candle_core::bail!(
            "kiln-gdn-kernel: gamma must have shape [{dv}], got {:?}",
            gamma_dims
        );
    }

    let (b_s, h_s, dk_s, dv_s) = state.dims4()?;
    if (b_s, h_s, dk_s, dv_s) != (b, h, dk, dv) {
        candle_core::bail!(
            "kiln-gdn-kernel: state shape [{b_s}, {h_s}, {dk_s}, {dv_s}] mismatch with [{b}, {h}, {dk}, {dv}]"
        );
    }

    if q_raw.dtype() != DType::BF16
        || k_raw.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || beta.dtype() != DType::BF16
        || g.dtype() != DType::BF16
        || z.dtype() != DType::BF16
        || gamma.dtype() != DType::BF16
        || state.dtype() != DType::BF16
    {
        candle_core::bail!(
            "kiln-gdn-kernel: all inputs to fused_norm must be bf16 (got q_raw={:?}, k_raw={:?}, v={:?}, beta={:?}, g={:?}, z={:?}, gamma={:?}, state={:?})",
            q_raw.dtype(), k_raw.dtype(), v.dtype(), beta.dtype(),
            g.dtype(), z.dtype(), gamma.dtype(), state.dtype()
        );
    }

    if dk > 256 {
        candle_core::bail!("kiln-gdn-kernel: dk must be <= 256 (got {dk})");
    }
    if dv > 1024 {
        candle_core::bail!("kiln-gdn-kernel: dv must be <= 1024 (got {dv})");
    }

    let q_raw = q_raw.contiguous()?;
    let k_raw = k_raw.contiguous()?;
    let v = v.contiguous()?;
    let beta = beta.contiguous()?;
    let g = g.contiguous()?;
    let z = z.contiguous()?;
    let gamma = gamma.contiguous()?;
    if !state.is_contiguous() {
        *state = state.contiguous()?;
    }

    let out = Tensor::zeros((b, h, dv), DType::BF16, device)?;

    {
        let (q_storage, q_layout) = q_raw.storage_and_layout();
        let (k_storage, k_layout) = k_raw.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (beta_storage, beta_layout) = beta.storage_and_layout();
        let (g_storage, g_layout) = g.storage_and_layout();
        let (z_storage, z_layout) = z.storage_and_layout();
        let (gamma_storage, gamma_layout) = gamma.storage_and_layout();
        let (s_storage, s_layout) = state.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let q_cuda = match &*q_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: q_raw must be on CUDA"),
        };
        let k_cuda = match &*k_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: k_raw must be on CUDA"),
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
        let z_cuda = match &*z_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: z must be on CUDA"),
        };
        let gamma_cuda = match &*gamma_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: gamma must be on CUDA"),
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
        let z_slice = z_cuda.as_cuda_slice::<bf16>()?.slice(z_layout.start_offset()..);
        let gamma_slice = gamma_cuda.as_cuda_slice::<bf16>()?.slice(gamma_layout.start_offset()..);
        let s_slice = s_cuda.as_cuda_slice::<bf16>()?.slice(s_layout.start_offset()..);
        let out_slice = out_cuda.as_cuda_slice::<bf16>()?.slice(out_layout.start_offset()..);

        unsafe {
            let (q_ptr, _g1) = q_slice.device_ptr(&stream);
            let (k_ptr, _g2) = k_slice.device_ptr(&stream);
            let (v_ptr, _g3) = v_slice.device_ptr(&stream);
            let (beta_ptr, _g4) = beta_slice.device_ptr(&stream);
            let (g_ptr, _g5) = g_slice.device_ptr(&stream);
            let (z_ptr, _g6) = z_slice.device_ptr(&stream);
            let (gamma_ptr, _g7) = gamma_slice.device_ptr(&stream);
            let (s_ptr, _g8) = s_slice.device_ptr(&stream);
            let (out_ptr, _g9) = out_slice.device_ptr(&stream);

            let status = kiln_gdn_recurrent_forward_fused_norm(
                q_ptr as *const _,
                k_ptr as *const _,
                v_ptr as *const _,
                beta_ptr as *const _,
                g_ptr as *const _,
                z_ptr as *const _,
                gamma_ptr as *const _,
                s_ptr as *mut _,
                out_ptr as *mut _,
                (b * h) as i32,
                dk as i32,
                dv as i32,
                q_scale,
                l2_eps,
                rms_eps,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!(
                    "kiln_gdn_recurrent_forward_fused_norm failed with status {status}"
                );
            }
        }
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Fused GDN gates kernel (PR #TBD — follow-up to PR #156's profile).
//
// Collapses the `kiln/gdn/gates` NVTX region — ~18.2% of A6000 BF16+Marlin
// decode wallclock per PR #156 — into one CUDA launch. Replaces the 8-op
// candle chain in `kiln-model::forward::gated_deltanet_forward` Step 6:
//
//   beta = sigmoid(b)                                  // bf16
//   g    = -exp(A_log) * softplus(a + dt_bias)         // bf16
//
// Algorithm mirrors FLA's `naive_gdn_gate` reference in
// fla/ops/gated_delta_rule/gate.py (sigmoid + softplus + exp + mul).
// The FLA Triton kernel there additionally does a chunkwise cumsum; we
// only need the elementwise portion because the cumsum happens later
// inside `gdn_chunkwise_recurrence`.
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn kiln_gdn_gates_bf16(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        A_log: *const core::ffi::c_void,
        dt_bias: *const core::ffi::c_void,
        beta_out: *mut core::ffi::c_void,
        g_out: *mut core::ffi::c_void,
        rows: i32,
        nv: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Cheap shape / dtype check so callers can skip allocating the outputs
/// (and materialising contiguous copies) when they know they'd fall back.
///
/// Requires:
///   - `a`, `b` of shape `[.., nv]`, bf16, CUDA
///   - `A_log`, `dt_bias` of shape `[nv]`, bf16, CUDA
///   - `nv <= 256`
pub fn gdn_gates_supports(
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> bool {
    if !matches!(a.device(), candle_core::Device::Cuda(_)) {
        return false;
    }
    if a.dtype() != DType::BF16
        || b.dtype() != DType::BF16
        || a_log.dtype() != DType::BF16
        || dt_bias.dtype() != DType::BF16
    {
        return false;
    }
    if a.shape() != b.shape() {
        return false;
    }
    let last = match a.dims().last() {
        Some(n) => *n,
        None => return false,
    };
    if last == 0 || last > 256 {
        return false;
    }
    if a_log.dims() != [last] || dt_bias.dims() != [last] {
        return false;
    }
    true
}

/// Run the fused GDN gates kernel.
///
/// Inputs (all bf16, all CUDA):
///   - `a`       : `[.., nv]` (last-dim is heads; other dims are collapsed
///                  to `rows`)
///   - `b`       : `[.., nv]` (same shape as `a`)
///   - `a_log`   : `[nv]`
///   - `dt_bias` : `[nv]`
///
/// Returns `(beta, g)`, both bf16 with the same shape as `a`. The
/// non-head dims are collapsed internally for the kernel launch and
/// restored on the outputs.
///
/// If `supports()` returns false, callers should fall back to the
/// candle-op reference path.
pub fn gdn_gates(
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    if !gdn_gates_supports(a, b, a_log, dt_bias) {
        candle_core::bail!(
            "kiln-gdn-kernel: gdn_gates: envelope violation \
             (a={:?}, b={:?}, a_log={:?}, dt_bias={:?}, a.dtype={:?})",
            a.shape(), b.shape(), a_log.shape(), dt_bias.shape(), a.dtype()
        );
    }

    let device = a.device();
    let shape = a.dims().to_vec();
    let nv = *shape.last().unwrap();
    let rows: usize = shape.iter().take(shape.len() - 1).product();

    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;

    let beta = Tensor::zeros(shape.clone(), DType::BF16, device)?;
    let g = Tensor::zeros(shape, DType::BF16, device)?;

    {
        let (a_storage, a_layout) = a.storage_and_layout();
        let (b_storage, b_layout) = b.storage_and_layout();
        let (al_storage, al_layout) = a_log.storage_and_layout();
        let (dt_storage, dt_layout) = dt_bias.storage_and_layout();
        let (beta_storage, beta_layout) = beta.storage_and_layout();
        let (g_storage, g_layout) = g.storage_and_layout();

        let a_cuda = match &*a_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: a must be on CUDA"),
        };
        let b_cuda = match &*b_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: b must be on CUDA"),
        };
        let al_cuda = match &*al_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: a_log must be on CUDA"),
        };
        let dt_cuda = match &*dt_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: dt_bias must be on CUDA"),
        };
        let beta_cuda = match &*beta_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: beta must be on CUDA"),
        };
        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: g must be on CUDA"),
        };

        let stream = a_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let a_slice = a_cuda.as_cuda_slice::<bf16>()?.slice(a_layout.start_offset()..);
        let b_slice = b_cuda.as_cuda_slice::<bf16>()?.slice(b_layout.start_offset()..);
        let al_slice = al_cuda.as_cuda_slice::<bf16>()?.slice(al_layout.start_offset()..);
        let dt_slice = dt_cuda.as_cuda_slice::<bf16>()?.slice(dt_layout.start_offset()..);
        let beta_slice = beta_cuda.as_cuda_slice::<bf16>()?.slice(beta_layout.start_offset()..);
        let g_slice = g_cuda.as_cuda_slice::<bf16>()?.slice(g_layout.start_offset()..);

        unsafe {
            let (a_ptr, _g1) = a_slice.device_ptr(&stream);
            let (b_ptr, _g2) = b_slice.device_ptr(&stream);
            let (al_ptr, _g3) = al_slice.device_ptr(&stream);
            let (dt_ptr, _g4) = dt_slice.device_ptr(&stream);
            let (beta_ptr, _g5) = beta_slice.device_ptr(&stream);
            let (g_ptr, _g6) = g_slice.device_ptr(&stream);

            let status = kiln_gdn_gates_bf16(
                a_ptr as *const _,
                b_ptr as *const _,
                al_ptr as *const _,
                dt_ptr as *const _,
                beta_ptr as *mut _,
                g_ptr as *mut _,
                rows as i32,
                nv as i32,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_gdn_gates_bf16 failed with status {status}");
            }
        }
    }

    Ok((beta, g))
}

// ---------------------------------------------------------------------------
// Fused GDN chunk-prep kernel.
//
// Collapses the 7+ candle op launches in the non-matmul part of
// `gdn_chunkwise_recurrence` into a single CUDA launch per chunk. See
// `csrc/gdn_chunk_prep.h` for the algebraic spec.
//
// The matmuls themselves (KKT, QKT, ks_entry, q_s) remain on cuBLAS via
// candle — they are well-optimised GEMMs and are not the bottleneck. The
// cost we reclaim is launch overhead and the elementwise-zoo pass count
// that PROFILING.md (post-PR #158 / #160) called out as the reason the
// surrounding GDN body still dominates wallclock.
// ---------------------------------------------------------------------------

/// Cheap envelope check callers use to decide whether to allocate outputs
/// and pack inputs contiguously.
pub fn gdn_chunk_prep_supports(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
) -> bool {
    if !matches!(g.device(), candle_core::Device::Cuda(_)) {
        return false;
    }
    if g.dtype() != DType::BF16
        || v.dtype() != DType::BF16
        || kkt.dtype() != DType::BF16
        || qkt.dtype() != DType::BF16
        || ks_entry.dtype() != DType::BF16
        || q_s.dtype() != DType::BF16
    {
        return false;
    }
    // g: [B, H, C] or [B*H, C].
    let g_dims = g.dims();
    let c = match g_dims.last() {
        Some(n) => *n,
        None => return false,
    };
    if c == 0 || c > 128 {
        return false;
    }
    // v / ks_entry / q_s: [..., C, dv]
    let vd = v.dims();
    if vd.len() < 2 {
        return false;
    }
    let dv = vd[vd.len() - 1];
    if dv == 0 || dv > 1024 {
        return false;
    }
    if vd[vd.len() - 2] != c {
        return false;
    }
    // kkt / qkt: [..., C, C]
    let kdims = kkt.dims();
    let qdims = qkt.dims();
    if kdims.len() < 2 || qdims.len() < 2 {
        return false;
    }
    if kdims[kdims.len() - 1] != c
        || kdims[kdims.len() - 2] != c
        || qdims[qdims.len() - 1] != c
        || qdims[qdims.len() - 2] != c
    {
        return false;
    }
    if ks_entry.dims().last().copied() != Some(dv)
        || q_s.dims().last().copied() != Some(dv)
    {
        return false;
    }
    true
}

/// Run the fused GDN chunk-prep kernel.
///
/// Inputs (all bf16, all CUDA):
///   - `g`         : `[B, H, C]`
///   - `v`         : `[B, H, C, dv]`
///   - `kkt`       : `[B, H, C, C]`
///   - `qkt`       : `[B, H, C, C]`
///   - `ks_entry`  : `[B, H, C, dv]`
///   - `q_s`       : `[B, H, C, dv]`
///
/// Returns `(a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last)`
/// where:
///   - `a_strict`       : `[B, H, C, C]`
///   - `b_mask`         : `[B, H, C, C]`
///   - `v_prime`        : `[B, H, C, dv]`
///   - `q_s_scaled`     : `[B, H, C, dv]`
///   - `decay_last_col` : `[B, H, C]`   (the `decay[C-1, :]` row)
///   - `p_last`         : `[B, H]`      (scalar `exp(big_g[C-1])` per (B,H))
pub fn gdn_chunk_prep(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    if !gdn_chunk_prep_supports(g, v, kkt, qkt, ks_entry, q_s) {
        candle_core::bail!(
            "kiln-gdn-kernel: gdn_chunk_prep: envelope violation \
             (g={:?}, v={:?}, kkt={:?}, qkt={:?}, ks_entry={:?}, q_s={:?})",
            g.shape(), v.shape(), kkt.shape(), qkt.shape(),
            ks_entry.shape(), q_s.shape()
        );
    }

    let (b, h, c) = g.dims3()?;
    let dv = v.dim(v.rank() - 1)?;
    let device = g.device();

    let g_c = g.contiguous()?;
    let v_c = v.contiguous()?;
    let kkt_c = kkt.contiguous()?;
    let qkt_c = qkt.contiguous()?;
    let ks_c = ks_entry.contiguous()?;
    let qs_c = q_s.contiguous()?;

    let a_strict = Tensor::zeros((b, h, c, c), DType::BF16, device)?;
    let b_mask = Tensor::zeros((b, h, c, c), DType::BF16, device)?;
    let v_prime = Tensor::zeros((b, h, c, dv), DType::BF16, device)?;
    let q_s_scaled = Tensor::zeros((b, h, c, dv), DType::BF16, device)?;
    let decay_last_col = Tensor::zeros((b, h, c), DType::BF16, device)?;
    let p_last = Tensor::zeros((b, h), DType::BF16, device)?;

    {
        let (g_storage, g_layout) = g_c.storage_and_layout();
        let (v_storage, v_layout) = v_c.storage_and_layout();
        let (kkt_storage, kkt_layout) = kkt_c.storage_and_layout();
        let (qkt_storage, qkt_layout) = qkt_c.storage_and_layout();
        let (ks_storage, ks_layout) = ks_c.storage_and_layout();
        let (qs_storage, qs_layout) = qs_c.storage_and_layout();
        let (a_storage, a_layout) = a_strict.storage_and_layout();
        let (b_storage, b_layout) = b_mask.storage_and_layout();
        let (vp_storage, vp_layout) = v_prime.storage_and_layout();
        let (qss_storage, qss_layout) = q_s_scaled.storage_and_layout();
        let (dl_storage, dl_layout) = decay_last_col.storage_and_layout();
        let (pl_storage, pl_layout) = p_last.storage_and_layout();

        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: g must be on CUDA"),
        };
        let v_cuda = match &*v_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: v must be on CUDA"),
        };
        let kkt_cuda = match &*kkt_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: kkt must be on CUDA"),
        };
        let qkt_cuda = match &*qkt_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: qkt must be on CUDA"),
        };
        let ks_cuda = match &*ks_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: ks_entry must be on CUDA"),
        };
        let qs_cuda = match &*qs_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: q_s must be on CUDA"),
        };
        let a_cuda = match &*a_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: a_strict must be on CUDA"),
        };
        let b_cuda = match &*b_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: b_mask must be on CUDA"),
        };
        let vp_cuda = match &*vp_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: v_prime must be on CUDA"),
        };
        let qss_cuda = match &*qss_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: q_s_scaled must be on CUDA"),
        };
        let dl_cuda = match &*dl_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: decay_last_col must be on CUDA"),
        };
        let pl_cuda = match &*pl_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: p_last must be on CUDA"),
        };

        let stream = g_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let g_slice = g_cuda.as_cuda_slice::<bf16>()?.slice(g_layout.start_offset()..);
        let v_slice = v_cuda.as_cuda_slice::<bf16>()?.slice(v_layout.start_offset()..);
        let kkt_slice = kkt_cuda.as_cuda_slice::<bf16>()?.slice(kkt_layout.start_offset()..);
        let qkt_slice = qkt_cuda.as_cuda_slice::<bf16>()?.slice(qkt_layout.start_offset()..);
        let ks_slice = ks_cuda.as_cuda_slice::<bf16>()?.slice(ks_layout.start_offset()..);
        let qs_slice = qs_cuda.as_cuda_slice::<bf16>()?.slice(qs_layout.start_offset()..);
        let a_slice = a_cuda.as_cuda_slice::<bf16>()?.slice(a_layout.start_offset()..);
        let b_slice = b_cuda.as_cuda_slice::<bf16>()?.slice(b_layout.start_offset()..);
        let vp_slice = vp_cuda.as_cuda_slice::<bf16>()?.slice(vp_layout.start_offset()..);
        let qss_slice = qss_cuda.as_cuda_slice::<bf16>()?.slice(qss_layout.start_offset()..);
        let dl_slice = dl_cuda.as_cuda_slice::<bf16>()?.slice(dl_layout.start_offset()..);
        let pl_slice = pl_cuda.as_cuda_slice::<bf16>()?.slice(pl_layout.start_offset()..);

        unsafe {
            let (g_ptr, _g1) = g_slice.device_ptr(&stream);
            let (v_ptr, _g2) = v_slice.device_ptr(&stream);
            let (kkt_ptr, _g3) = kkt_slice.device_ptr(&stream);
            let (qkt_ptr, _g4) = qkt_slice.device_ptr(&stream);
            let (ks_ptr, _g5) = ks_slice.device_ptr(&stream);
            let (qs_ptr, _g6) = qs_slice.device_ptr(&stream);
            let (a_ptr, _g7) = a_slice.device_ptr(&stream);
            let (b_ptr, _g8) = b_slice.device_ptr(&stream);
            let (vp_ptr, _g9) = vp_slice.device_ptr(&stream);
            let (qss_ptr, _g10) = qss_slice.device_ptr(&stream);
            let (dl_ptr, _g11) = dl_slice.device_ptr(&stream);
            let (pl_ptr, _g12) = pl_slice.device_ptr(&stream);

            let status = kiln_gdn_chunk_prep(
                g_ptr as *const _,
                v_ptr as *const _,
                kkt_ptr as *const _,
                qkt_ptr as *const _,
                ks_ptr as *const _,
                qs_ptr as *const _,
                a_ptr as *mut _,
                b_ptr as *mut _,
                vp_ptr as *mut _,
                qss_ptr as *mut _,
                dl_ptr as *mut _,
                pl_ptr as *mut _,
                (b * h) as i32,
                c as i32,
                dv as i32,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_gdn_chunk_prep failed with status {status}");
            }
        }
    }

    Ok((a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last))
}

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
    CustomOp1, CustomOp3, DType, Layout, Result, Shape, Tensor, backend::BackendStorage,
    cuda_backend::{CudaStorage, cudarc::driver::DevicePtr},
};
use half::bf16;
use std::cell::RefCell;

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

    fn kiln_gdn_decode_gates_recurrent_vf32_bf16(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        a_log: *const core::ffi::c_void,
        dt_bias: *const core::ffi::c_void,
        state: *mut core::ffi::c_void,
        z: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        q_heads: i32,
        value_heads: i32,
        dk: i32,
        dv: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_decode_gates_recurrent_bf16(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        a_log: *const core::ffi::c_void,
        dt_bias: *const core::ffi::c_void,
        state: *mut core::ffi::c_void,
        z: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        q_heads: i32,
        value_heads: i32,
        dk: i32,
        dv: i32,
        eps: f32,
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

    fn kiln_gdn_chunk_scan(
        a_strict: *const core::ffi::c_void,
        b_mask: *const core::ffi::c_void,
        v_prime: *const core::ffi::c_void,
        q_s_scaled: *const core::ffi::c_void,
        beta: *const core::ffi::c_void,
        decay_last_col: *const core::ffi::c_void,
        out_chunk: *mut core::ffi::c_void,
        w_weighted: *mut core::ffi::c_void,
        batch_heads: i32,
        chunk_size: i32,
        dv: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_full_chunk_forward(
        g: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        kkt: *const core::ffi::c_void,
        qkt: *const core::ffi::c_void,
        ks_entry: *const core::ffi::c_void,
        q_s: *const core::ffi::c_void,
        beta: *const core::ffi::c_void,
        k_t: *const core::ffi::c_void,
        state: *mut core::ffi::c_void,
        out_chunk: *mut core::ffi::c_void,
        batch_heads: i32,
        chunk_size: i32,
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
        candle_core::bail!("kiln-gdn-kernel: chunk_size must be <= 128 (got {c})");
    }
    if dv > 1024 {
        candle_core::bail!("kiln-gdn-kernel: dv must be <= 1024 (got {dv})");
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
                candle_core::bail!("kiln_gdn_forward_substitution failed with status {status}");
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
        candle_core::bail!("kiln-gdn-kernel: g shape [{b_g}, {h_g}] mismatch with q [{b}, {h}, *]");
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
            q.dtype(),
            k.dtype(),
            v.dtype(),
            beta.dtype(),
            g.dtype(),
            state.dtype()
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

        let q_slice = q_cuda
            .as_cuda_slice::<bf16>()?
            .slice(q_layout.start_offset()..);
        let k_slice = k_cuda
            .as_cuda_slice::<bf16>()?
            .slice(k_layout.start_offset()..);
        let v_slice = v_cuda
            .as_cuda_slice::<bf16>()?
            .slice(v_layout.start_offset()..);
        let beta_slice = beta_cuda
            .as_cuda_slice::<bf16>()?
            .slice(beta_layout.start_offset()..);
        let g_slice = g_cuda
            .as_cuda_slice::<bf16>()?
            .slice(g_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<bf16>()?
            .slice(s_layout.start_offset()..);
        let out_slice = out_cuda
            .as_cuda_slice::<bf16>()?
            .slice(out_layout.start_offset()..);

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
                candle_core::bail!("kiln_gdn_recurrent_forward failed with status {status}");
            }
        }
    }

    Ok(out)
}

pub fn gdn_decode_gates_recurrent_supports(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    weight: &Tensor,
) -> bool {
    if !matches!(q.device(), candle_core::Device::Cuda(_)) {
        return false;
    }
    if q.dtype() != DType::BF16
        || k.dtype() != DType::BF16
        || !matches!(v.dtype(), DType::BF16 | DType::F32)
        || a.dtype() != DType::BF16
        || b.dtype() != DType::BF16
        || a_log.dtype() != DType::BF16
        || dt_bias.dtype() != DType::BF16
        || state.dtype() != DType::BF16
    {
        return false;
    }

    let Ok((batch, seq_len, q_heads, dk)) = q.dims4() else {
        return false;
    };
    let Ok((b_k, t_k, h_k, dk_k)) = k.dims4() else {
        return false;
    };
    let Ok((b_v, t_v, value_heads, dv)) = v.dims4() else {
        return false;
    };
    let Ok((b_a, t_a, h_a)) = a.dims3() else {
        return false;
    };
    let Ok((b_b, t_b, h_b)) = b.dims3() else {
        return false;
    };
    let Ok((b_s, h_s, dk_s, dv_s)) = state.dims4() else {
        return false;
    };

    batch >= 1
        && seq_len == 1
        && (b_k, t_k, h_k, dk_k) == (batch, seq_len, q_heads, dk)
        && (b_v, t_v) == (batch, seq_len)
        && (b_a, t_a, h_a) == (batch, seq_len, value_heads)
        && (b_b, t_b, h_b) == (batch, seq_len, value_heads)
        && z.dims() == [batch, seq_len, value_heads, dv]
        && a_log.dims() == [value_heads]
        && dt_bias.dims() == [value_heads]
        && weight.dims() == [dv]
        && (b_s, h_s, dk_s, dv_s) == (batch, value_heads, dk, dv)
        && q_heads > 0
        && value_heads >= q_heads
        && value_heads % q_heads == 0
        && dk == 128
        && dv == 128
        && state.is_contiguous()
}

struct DecodeGatesRecurrentOutputs {
    outputs: Vec<Tensor>,
    next: usize,
}

thread_local! {
    static DECODE_GATES_RECURRENT_OUTPUTS: RefCell<Option<DecodeGatesRecurrentOutputs>> = const { RefCell::new(None) };
}

pub fn with_decode_gates_recurrent_outputs<R>(
    outputs: Vec<Tensor>,
    f: impl FnOnce() -> R,
) -> R {
    let previous = DECODE_GATES_RECURRENT_OUTPUTS.replace(Some(DecodeGatesRecurrentOutputs {
        outputs,
        next: 0,
    }));
    let result = f();
    DECODE_GATES_RECURRENT_OUTPUTS.replace(previous);
    result
}

fn next_decode_gates_recurrent_output(
    shape: (usize, usize, usize, usize),
    device: &candle_core::Device,
) -> Result<Option<Tensor>> {
    DECODE_GATES_RECURRENT_OUTPUTS.with(|cell| {
        let mut borrowed = cell.borrow_mut();
        let Some(scratch) = borrowed.as_mut() else {
            return Ok(None);
        };
        let Some(out) = scratch.outputs.get(scratch.next).cloned() else {
            candle_core::bail!(
                "kiln-gdn-kernel: missing graph GDN decode output {}",
                scratch.next
            );
        };
        scratch.next += 1;
        if !matches!(out.device(), candle_core::Device::Cuda(_))
            || !matches!(device, candle_core::Device::Cuda(_))
        {
            candle_core::bail!("kiln-gdn-kernel: graph GDN decode output device mismatch");
        }
        if out.dtype() != DType::BF16 {
            candle_core::bail!("kiln-gdn-kernel: graph GDN decode output must be BF16");
        }
        if out.dims() != [shape.0, shape.1, shape.2, shape.3] {
            candle_core::bail!(
                "kiln-gdn-kernel: graph GDN decode output shape {:?} != {:?}",
                out.dims(),
                [shape.0, shape.1, shape.2, shape.3]
            );
        }
        Ok(Some(out))
    })
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_gates_recurrent(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &mut Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if !gdn_decode_gates_recurrent_supports(q, k, v, a, b, a_log, dt_bias, state, z, weight) {
        candle_core::bail!("kiln-gdn-kernel: gdn_decode_gates_recurrent envelope violation");
    }

    let device = q.device();
    let (batch, _, q_heads, dk) = q.dims4()?;
    let (_, _, value_heads, dv) = v.dims4()?;

    let q = gdn_gates_ctx(q.contiguous(), "gdn_decode_gates_recurrent q contiguous")?;
    let k = gdn_gates_ctx(k.contiguous(), "gdn_decode_gates_recurrent k contiguous")?;
    let v = gdn_gates_ctx(v.contiguous(), "gdn_decode_gates_recurrent v contiguous")?;
    let a = gdn_gates_ctx(a.contiguous(), "gdn_decode_gates_recurrent a contiguous")?;
    let b = gdn_gates_ctx(b.contiguous(), "gdn_decode_gates_recurrent b contiguous")?;
    let a_log = gdn_gates_ctx(a_log.contiguous(), "gdn_decode_gates_recurrent a_log contiguous")?;
    let dt_bias = gdn_gates_ctx(dt_bias.contiguous(), "gdn_decode_gates_recurrent dt_bias contiguous")?;
    let out = match next_decode_gates_recurrent_output((batch, 1, value_heads, dv), device)? {
        Some(out) => out,
        None => gdn_gates_ctx(
            Tensor::zeros((batch, 1, value_heads, dv), DType::BF16, device),
            "gdn_decode_gates_recurrent out zeros",
        )?,
    };

    {
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();
        let (a_storage, a_layout) = a.storage_and_layout();
        let (b_storage, b_layout) = b.storage_and_layout();
        let (al_storage, al_layout) = a_log.storage_and_layout();
        let (dt_storage, dt_layout) = dt_bias.storage_and_layout();
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

        let q_slice = q_cuda
            .as_cuda_slice::<bf16>()?
            .slice(q_layout.start_offset()..);
        let k_slice = k_cuda
            .as_cuda_slice::<bf16>()?
            .slice(k_layout.start_offset()..);
        let a_slice = a_cuda
            .as_cuda_slice::<bf16>()?
            .slice(a_layout.start_offset()..);
        let b_slice = b_cuda
            .as_cuda_slice::<bf16>()?
            .slice(b_layout.start_offset()..);
        let al_slice = al_cuda
            .as_cuda_slice::<bf16>()?
            .slice(al_layout.start_offset()..);
        let dt_slice = dt_cuda
            .as_cuda_slice::<bf16>()?
            .slice(dt_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<bf16>()?
            .slice(s_layout.start_offset()..);
        let out_slice = out_cuda
            .as_cuda_slice::<bf16>()?
            .slice(out_layout.start_offset()..);

        unsafe {
            let (q_ptr, _g1) = q_slice.device_ptr(&stream);
            let (k_ptr, _g2) = k_slice.device_ptr(&stream);
            let (a_ptr, _g4) = a_slice.device_ptr(&stream);
            let (b_ptr, _g5) = b_slice.device_ptr(&stream);
            let (al_ptr, _g6) = al_slice.device_ptr(&stream);
            let (dt_ptr, _g7) = dt_slice.device_ptr(&stream);
            let (s_ptr, _g8) = s_slice.device_ptr(&stream);
            let (out_ptr, _g11) = out_slice.device_ptr(&stream);

            let status = match v.dtype() {
                DType::BF16 => {
                    let v_slice = v_cuda
                        .as_cuda_slice::<bf16>()?
                        .slice(v_layout.start_offset()..);
                    let (v_ptr, _g3) = v_slice.device_ptr(&stream);
                    kiln_gdn_decode_gates_recurrent_bf16(
                        q_ptr as *const _,
                        k_ptr as *const _,
                        v_ptr as *const _,
                        a_ptr as *const _,
                        b_ptr as *const _,
                        al_ptr as *const _,
                        dt_ptr as *const _,
                        s_ptr as *mut _,
                        core::ptr::null(),
                        core::ptr::null(),
                        out_ptr as *mut _,
                        batch as i32,
                        q_heads as i32,
                        value_heads as i32,
                        dk as i32,
                        dv as i32,
                        eps,
                        raw_stream,
                    )
                }
                DType::F32 => {
                    let v_slice = v_cuda
                        .as_cuda_slice::<f32>()?
                        .slice(v_layout.start_offset()..);
                    let (v_ptr, _g3) = v_slice.device_ptr(&stream);
                    kiln_gdn_decode_gates_recurrent_vf32_bf16(
                        q_ptr as *const _,
                        k_ptr as *const _,
                        v_ptr as *const _,
                        a_ptr as *const _,
                        b_ptr as *const _,
                        al_ptr as *const _,
                        dt_ptr as *const _,
                        s_ptr as *mut _,
                        core::ptr::null(),
                        core::ptr::null(),
                        out_ptr as *mut _,
                        batch as i32,
                        q_heads as i32,
                        value_heads as i32,
                        dk as i32,
                        dv as i32,
                        eps,
                        raw_stream,
                    )
                }
                other => candle_core::bail!("kiln-gdn-kernel: unsupported fused decode v dtype {other:?}"),
            };
            if status != 0 {
                candle_core::bail!(
                    "kiln_gdn_decode_gates_recurrent_bf16 failed with status {status}"
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

    fn kiln_gdn_gates_bf16_f32_params(
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

    fn kiln_gdn_gates_bf16_f32_bf16_params(
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

    fn kiln_gdn_gate_beta_bf16(
        b: *const core::ffi::c_void,
        beta_out: *mut core::ffi::c_void,
        rows: i32,
        nv: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_gate_g_bf16(
        a: *const core::ffi::c_void,
        A_log: *const core::ffi::c_void,
        dt_bias: *const core::ffi::c_void,
        g_out: *mut core::ffi::c_void,
        rows: i32,
        nv: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

unsafe extern "C" {
    fn kiln_gdn_gated_rms_norm_bf16(
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

/// Cheap shape / dtype check so callers can skip allocating the outputs
/// (and materialising contiguous copies) when they know they'd fall back.
///
/// Requires:
///   - `a`, `b` of shape `[.., nv]`, bf16, CUDA
///   - `A_log`, `dt_bias` of shape `[nv]`, bf16 or f32, CUDA
///   - `nv <= 256`

struct GdnGateBetaOp;

impl CustomOp1 for GdnGateBetaOp {
    fn name(&self) -> &'static str { "gdn-gate-beta" }

    fn cpu_fwd(&self, _s: &candle_core::CpuStorage, _l: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("GdnGateBetaOp CPU fallback is not implemented")
    }

    fn cuda_fwd(&self, s_b: &CudaStorage, l_b: &Layout) -> Result<(CudaStorage, Shape)> {
        let dims = l_b.shape().dims().to_vec();
        let nv = *dims.last().ok_or_else(|| candle_core::Error::Msg("GdnGateBetaOp: b must have rank >= 1".to_string()))?;
        if nv == 0 || nv > 256 { candle_core::bail!("GdnGateBetaOp: nv {nv} outside 1..=256"); }
        if !l_b.is_contiguous() { candle_core::bail!("GdnGateBetaOp requires contiguous b"); }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let device = s_b.device();
        let stream = device.cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let elem_count: usize = dims.iter().product();
        let out_slice = device.alloc_zeros::<bf16>(elem_count)?;
        let b_slice = s_b.as_cuda_slice::<bf16>()?.slice(l_b.start_offset()..);
        unsafe {
            let (b_ptr, _g1) = b_slice.device_ptr(&stream);
            let (out_ptr, _g2) = out_slice.device_ptr(&stream);
            let status = kiln_gdn_gate_beta_bf16(b_ptr as *const _, out_ptr as *mut _, rows as i32, nv as i32, raw_stream);
            if status != 0 { candle_core::bail!("GdnGateBetaOp: kernel failed with status {status}"); }
        }
        Ok((CudaStorage::wrap_cuda_slice(out_slice, device.clone()), Shape::from(dims.as_slice())))
    }
}

struct GdnGateGOp;

impl CustomOp3 for GdnGateGOp {
    fn name(&self) -> &'static str { "gdn-gate-g" }

    fn cpu_fwd(&self, _s1: &candle_core::CpuStorage, _l1: &Layout, _s2: &candle_core::CpuStorage, _l2: &Layout, _s3: &candle_core::CpuStorage, _l3: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("GdnGateGOp CPU fallback is not implemented")
    }

    fn cuda_fwd(&self, s_a: &CudaStorage, l_a: &Layout, s_al: &CudaStorage, l_al: &Layout, s_dt: &CudaStorage, l_dt: &Layout) -> Result<(CudaStorage, Shape)> {
        let dims = l_a.shape().dims().to_vec();
        let nv = *dims.last().ok_or_else(|| candle_core::Error::Msg("GdnGateGOp: a must have rank >= 1".to_string()))?;
        if nv == 0 || nv > 256 { candle_core::bail!("GdnGateGOp: nv {nv} outside 1..=256"); }
        if l_al.shape().dims() != [nv] || l_dt.shape().dims() != [nv] { candle_core::bail!("GdnGateGOp: params must have shape [nv]"); }
        if !l_a.is_contiguous() || !l_al.is_contiguous() || !l_dt.is_contiguous() { candle_core::bail!("GdnGateGOp requires contiguous inputs"); }
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let device = s_a.device();
        let stream = device.cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let elem_count: usize = dims.iter().product();
        let out_slice = device.alloc_zeros::<bf16>(elem_count)?;
        let a_slice = s_a.as_cuda_slice::<bf16>()?.slice(l_a.start_offset()..);
        let al_slice = s_al.as_cuda_slice::<bf16>()?.slice(l_al.start_offset()..);
        let dt_slice = s_dt.as_cuda_slice::<bf16>()?.slice(l_dt.start_offset()..);
        unsafe {
            let (a_ptr, _g1) = a_slice.device_ptr(&stream);
            let (al_ptr, _g2) = al_slice.device_ptr(&stream);
            let (dt_ptr, _g3) = dt_slice.device_ptr(&stream);
            let (out_ptr, _g4) = out_slice.device_ptr(&stream);
            let status = kiln_gdn_gate_g_bf16(a_ptr as *const _, al_ptr as *const _, dt_ptr as *const _, out_ptr as *mut _, rows as i32, nv as i32, raw_stream);
            if status != 0 { candle_core::bail!("GdnGateGOp: kernel failed with status {status}"); }
        }
        Ok((CudaStorage::wrap_cuda_slice(out_slice, device.clone()), Shape::from(dims.as_slice())))
    }
}

fn gdn_gates_ctx<T>(res: Result<T>, label: &str) -> Result<T> {
    res.map_err(|err| candle_core::Error::Msg(format!("{label}: {err}")))
}

pub fn gdn_gates_decline_reason(
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Option<&'static str> {
    if !matches!(a.device(), candle_core::Device::Cuda(_))
        || !matches!(b.device(), candle_core::Device::Cuda(_))
        || !matches!(a_log.device(), candle_core::Device::Cuda(_))
        || !matches!(dt_bias.device(), candle_core::Device::Cuda(_))
    {
        return Some("all inputs must be CUDA tensors");
    }
    if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
        return Some("a and b must be bf16");
    }
    if !matches!(
        (a_log.dtype(), dt_bias.dtype()),
        (DType::BF16, DType::BF16) | (DType::F32, DType::F32) | (DType::F32, DType::BF16)
    ) {
        return Some("a_log/dt_bias must be bf16/bf16, f32/f32, or f32/bf16");
    }
    if a.shape() != b.shape() {
        return Some("a and b shapes must match");
    }
    let last = match a.dims().last() {
        Some(n) => *n,
        None => return Some("a must have rank >= 1"),
    };
    if last == 0 || last > 256 {
        return Some("last dimension nv must be in 1..=256");
    }
    if a_log.dims() != [last] || dt_bias.dims() != [last] {
        return Some("a_log and dt_bias must have shape [nv]");
    }
    None
}

pub fn gdn_gates_supports(a: &Tensor, b: &Tensor, a_log: &Tensor, dt_bias: &Tensor) -> bool {
    gdn_gates_decline_reason(a, b, a_log, dt_bias).is_none()
}

/// Run the fused GDN gates kernel.
///
/// Inputs (CUDA):
///   - `a`       : `[.., nv]` (last-dim is heads; other dims are collapsed
///                  to `rows`)
///   - `b`       : `[.., nv]` (same shape as `a`)
///   - `a_log`   : `[nv]` bf16 or f32
///   - `dt_bias` : `[nv]` bf16 or f32
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
            "kiln-gdn-kernel: gdn_gates: envelope violation ({}) \
             (a={:?} {:?}, b={:?} {:?}, a_log={:?} {:?}, dt_bias={:?} {:?})",
            gdn_gates_decline_reason(a, b, a_log, dt_bias).unwrap_or("unknown"),
            a.shape(),
            a.dtype(),
            b.shape(),
            b.dtype(),
            a_log.shape(),
            a_log.dtype(),
            dt_bias.shape(),
            dt_bias.dtype()
        );
    }

    if a_log.dtype() == DType::BF16 && dt_bias.dtype() == DType::BF16 {
        let beta = b.apply_op1_no_bwd(&GdnGateBetaOp)?;
        let g = a.apply_op3_no_bwd(a_log, dt_bias, &GdnGateGOp)?;
        return Ok((beta, g));
    }

    let device = a.device();
    let shape = a.dims().to_vec();
    let nv = *shape.last().unwrap();
    let rows: usize = shape.iter().take(shape.len() - 1).product();

    let a = gdn_gates_ctx(a.contiguous(), "gdn_gates a contiguous")?;
    let b = gdn_gates_ctx(b.contiguous(), "gdn_gates b contiguous")?;
    let a_log = gdn_gates_ctx(a_log.contiguous(), "gdn_gates a_log contiguous")?;
    let dt_bias = gdn_gates_ctx(dt_bias.contiguous(), "gdn_gates dt_bias contiguous")?;

    let beta = gdn_gates_ctx(Tensor::zeros(shape.clone(), DType::BF16, device), "gdn_gates beta zeros")?;
    let g = gdn_gates_ctx(Tensor::zeros(shape, DType::BF16, device), "gdn_gates g zeros")?;

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

        let a_slice = a_cuda
            .as_cuda_slice::<bf16>()
            .map_err(|err| candle_core::Error::Msg(format!("gdn_gates bf16 as_cuda_slice: {err}")))?
            .slice(a_layout.start_offset()..);
        let b_slice = b_cuda
            .as_cuda_slice::<bf16>()
            .map_err(|err| candle_core::Error::Msg(format!("gdn_gates bf16 as_cuda_slice: {err}")))?
            .slice(b_layout.start_offset()..);
        let beta_slice = beta_cuda
            .as_cuda_slice::<bf16>()
            .map_err(|err| candle_core::Error::Msg(format!("gdn_gates bf16 as_cuda_slice: {err}")))?
            .slice(beta_layout.start_offset()..);
        let g_slice = g_cuda
            .as_cuda_slice::<bf16>()
            .map_err(|err| candle_core::Error::Msg(format!("gdn_gates bf16 as_cuda_slice: {err}")))?
            .slice(g_layout.start_offset()..);

        unsafe {
            let (a_ptr, _g1) = a_slice.device_ptr(&stream);
            let (b_ptr, _g2) = b_slice.device_ptr(&stream);
            let (beta_ptr, _g5) = beta_slice.device_ptr(&stream);
            let (g_ptr, _g6) = g_slice.device_ptr(&stream);

            let status = match (a_log.dtype(), dt_bias.dtype()) {
                (DType::BF16, DType::BF16) => {
                    let al_slice = al_cuda
                        .as_cuda_slice::<bf16>()
            .map_err(|err| candle_core::Error::Msg(format!("gdn_gates bf16 as_cuda_slice: {err}")))?
                        .slice(al_layout.start_offset()..);
                    let dt_slice = dt_cuda
                        .as_cuda_slice::<bf16>()
            .map_err(|err| candle_core::Error::Msg(format!("gdn_gates bf16 as_cuda_slice: {err}")))?
                        .slice(dt_layout.start_offset()..);
                    let (al_ptr, _g3) = al_slice.device_ptr(&stream);
                    let (dt_ptr, _g4) = dt_slice.device_ptr(&stream);
                    kiln_gdn_gates_bf16(
                        a_ptr as *const _,
                        b_ptr as *const _,
                        al_ptr as *const _,
                        dt_ptr as *const _,
                        beta_ptr as *mut _,
                        g_ptr as *mut _,
                        rows as i32,
                        nv as i32,
                        raw_stream,
                    )
                }
                (DType::F32, DType::F32) => {
                    let al_slice = al_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|err| candle_core::Error::Msg(format!("gdn_gates f32 as_cuda_slice: {err}")))?
                        .slice(al_layout.start_offset()..);
                    let dt_slice = dt_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|err| candle_core::Error::Msg(format!("gdn_gates f32 as_cuda_slice: {err}")))?
                        .slice(dt_layout.start_offset()..);
                    let (al_ptr, _g3) = al_slice.device_ptr(&stream);
                    let (dt_ptr, _g4) = dt_slice.device_ptr(&stream);
                    kiln_gdn_gates_bf16_f32_params(
                        a_ptr as *const _,
                        b_ptr as *const _,
                        al_ptr as *const _,
                        dt_ptr as *const _,
                        beta_ptr as *mut _,
                        g_ptr as *mut _,
                        rows as i32,
                        nv as i32,
                        raw_stream,
                    )
                }
                (DType::F32, DType::BF16) => {
                    let al_slice = al_cuda
                        .as_cuda_slice::<f32>()
                        .map_err(|err| candle_core::Error::Msg(format!("gdn_gates f32 as_cuda_slice: {err}")))?
                        .slice(al_layout.start_offset()..);
                    let dt_slice = dt_cuda
                        .as_cuda_slice::<bf16>()
            .map_err(|err| candle_core::Error::Msg(format!("gdn_gates bf16 as_cuda_slice: {err}")))?
                        .slice(dt_layout.start_offset()..);
                    let (al_ptr, _g3) = al_slice.device_ptr(&stream);
                    let (dt_ptr, _g4) = dt_slice.device_ptr(&stream);
                    kiln_gdn_gates_bf16_f32_bf16_params(
                        a_ptr as *const _,
                        b_ptr as *const _,
                        al_ptr as *const _,
                        dt_ptr as *const _,
                        beta_ptr as *mut _,
                        g_ptr as *mut _,
                        rows as i32,
                        nv as i32,
                        raw_stream,
                    )
                }
                _ => candle_core::bail!(
                    "kiln-gdn-kernel: unsupported gate parameter dtypes {:?} / {:?}",
                    a_log.dtype(),
                    dt_bias.dtype()
                ),
            };

            if status != 0 {
                candle_core::bail!("kiln_gdn_gates_bf16 failed with status {status}");
            }
        }
    }

    Ok((beta, g))
}

// ---------------------------------------------------------------------------
// Fused GDN gated RMSNorm kernel.
//
// Collapses the `kiln/gdn/gated_norm` body for Qwen3.5 GDN from the
// candle F32 `rms_norm(x, weight, eps) * silu(z)` op chain into one CUDA
// launch. The envelope is intentionally narrow: CUDA bf16 tensors,
// matching contiguous-last-dim shapes, and hidden=128.
// ---------------------------------------------------------------------------

pub fn gdn_gated_rms_norm_supports(x: &Tensor, z: &Tensor, weight: &Tensor) -> bool {
    if !matches!(x.device(), candle_core::Device::Cuda(_)) {
        return false;
    }
    if x.dtype() != DType::BF16 || z.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return false;
    }
    if x.shape() != z.shape() {
        return false;
    }
    let Some(hidden) = x.dims().last().copied() else {
        return false;
    };
    hidden == 128 && weight.dims() == &[hidden]
}

pub fn gdn_gated_rms_norm(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if !gdn_gated_rms_norm_supports(x, z, weight) {
        candle_core::bail!(
            "kiln-gdn-kernel: gdn_gated_rms_norm: envelope violation \
             (x={:?}, z={:?}, weight={:?}, x.dtype={:?})",
            x.shape(),
            z.shape(),
            weight.shape(),
            x.dtype()
        );
    }

    let device = x.device();
    let shape = x.dims().to_vec();
    let hidden = *shape.last().unwrap();
    let rows: usize = shape.iter().take(shape.len() - 1).product();
    if rows > i32::MAX as usize {
        candle_core::bail!("kiln-gdn-kernel: gdn_gated_rms_norm rows exceed i32");
    }

    let x = x.contiguous()?;
    let z = z.contiguous()?;
    let weight = weight.contiguous()?;
    let out = Tensor::zeros(shape, DType::BF16, device)?;

    {
        let (x_storage, x_layout) = x.storage_and_layout();
        let (z_storage, z_layout) = z.storage_and_layout();
        let (w_storage, w_layout) = weight.storage_and_layout();
        let (out_storage, out_layout) = out.storage_and_layout();

        let x_cuda = match &*x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: x must be on CUDA"),
        };
        let z_cuda = match &*z_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: z must be on CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: weight must be on CUDA"),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: out must be on CUDA"),
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
        let out_slice = out_cuda
            .as_cuda_slice::<bf16>()?
            .slice(out_layout.start_offset()..);

        unsafe {
            let (x_ptr, _g1) = x_slice.device_ptr(&stream);
            let (z_ptr, _g2) = z_slice.device_ptr(&stream);
            let (w_ptr, _g3) = w_slice.device_ptr(&stream);
            let (out_ptr, _g4) = out_slice.device_ptr(&stream);

            let status = kiln_gdn_gated_rms_norm_bf16(
                x_ptr as *const _,
                z_ptr as *const _,
                w_ptr as *const _,
                out_ptr as *mut _,
                rows as i32,
                hidden as i32,
                eps,
                raw_stream,
            );

            if status != 0 {
                candle_core::bail!("kiln_gdn_gated_rms_norm_bf16 failed with status {status}");
            }
        }
    }

    Ok(out)
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
    if ks_entry.dims().last().copied() != Some(dv) || q_s.dims().last().copied() != Some(dv) {
        return false;
    }
    true
}

/// Run the fused GDN chunk-prep kernel.
///
/// Inputs (CUDA):
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
            g.shape(),
            v.shape(),
            kkt.shape(),
            qkt.shape(),
            ks_entry.shape(),
            q_s.shape()
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

        let g_slice = g_cuda
            .as_cuda_slice::<bf16>()?
            .slice(g_layout.start_offset()..);
        let v_slice = v_cuda
            .as_cuda_slice::<bf16>()?
            .slice(v_layout.start_offset()..);
        let kkt_slice = kkt_cuda
            .as_cuda_slice::<bf16>()?
            .slice(kkt_layout.start_offset()..);
        let qkt_slice = qkt_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qkt_layout.start_offset()..);
        let ks_slice = ks_cuda
            .as_cuda_slice::<bf16>()?
            .slice(ks_layout.start_offset()..);
        let qs_slice = qs_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qs_layout.start_offset()..);
        let a_slice = a_cuda
            .as_cuda_slice::<bf16>()?
            .slice(a_layout.start_offset()..);
        let b_slice = b_cuda
            .as_cuda_slice::<bf16>()?
            .slice(b_layout.start_offset()..);
        let vp_slice = vp_cuda
            .as_cuda_slice::<bf16>()?
            .slice(vp_layout.start_offset()..);
        let qss_slice = qss_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qss_layout.start_offset()..);
        let dl_slice = dl_cuda
            .as_cuda_slice::<bf16>()?
            .slice(dl_layout.start_offset()..);
        let pl_slice = pl_cuda
            .as_cuda_slice::<bf16>()?
            .slice(pl_layout.start_offset()..);

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

    Ok((
        a_strict,
        b_mask,
        v_prime,
        q_s_scaled,
        decay_last_col,
        p_last,
    ))
}

pub fn gdn_chunk_scan_supports(
    a_strict: &Tensor,
    b_mask: &Tensor,
    v_prime: &Tensor,
    q_s_scaled: &Tensor,
    beta: &Tensor,
    decay_last_col: &Tensor,
) -> bool {
    if !matches!(a_strict.device(), candle_core::Device::Cuda(_)) {
        return false;
    }
    if a_strict.dtype() != DType::BF16
        || b_mask.dtype() != DType::BF16
        || v_prime.dtype() != DType::BF16
        || q_s_scaled.dtype() != DType::BF16
        || beta.dtype() != DType::BF16
        || decay_last_col.dtype() != DType::BF16
    {
        return false;
    }
    let Ok((b, h, c, c2)) = a_strict.dims4() else {
        return false;
    };
    if c != c2 || c == 0 || c > 64 {
        return false;
    }
    let Ok((b2, h2, c3, c4)) = b_mask.dims4() else {
        return false;
    };
    if (b2, h2, c3, c4) != (b, h, c, c) {
        return false;
    }
    let Ok((b3, h3, c5, dv)) = v_prime.dims4() else {
        return false;
    };
    if (b3, h3, c5) != (b, h, c) || dv == 0 || dv > 128 {
        return false;
    }
    let Ok((b4, h4, c6, dv2)) = q_s_scaled.dims4() else {
        return false;
    };
    if (b4, h4, c6, dv2) != (b, h, c, dv) {
        return false;
    }
    let Ok((b5, h5, c7)) = beta.dims3() else {
        return false;
    };
    if (b5, h5, c7) != (b, h, c) {
        return false;
    }
    let Ok((b6, h6, c8)) = decay_last_col.dims3() else {
        return false;
    };
    (b6, h6, c8) == (b, h, c)
}

pub fn gdn_chunk_scan(
    a_strict: &Tensor,
    b_mask: &Tensor,
    v_prime: &Tensor,
    q_s_scaled: &Tensor,
    beta: &Tensor,
    decay_last_col: &Tensor,
) -> Result<(Tensor, Tensor)> {
    if !gdn_chunk_scan_supports(a_strict, b_mask, v_prime, q_s_scaled, beta, decay_last_col) {
        candle_core::bail!(
            "kiln-gdn-kernel: gdn_chunk_scan: envelope violation \
             (a={:?}, b={:?}, v_prime={:?}, q_s_scaled={:?}, beta={:?}, decay_last_col={:?})",
            a_strict.shape(),
            b_mask.shape(),
            v_prime.shape(),
            q_s_scaled.shape(),
            beta.shape(),
            decay_last_col.shape()
        );
    }

    let (b, h, c, _) = a_strict.dims4()?;
    let dv = v_prime.dim(3)?;
    let device = a_strict.device();

    let a_c = a_strict.contiguous()?;
    let b_c = b_mask.contiguous()?;
    let vp_c = v_prime.contiguous()?;
    let qss_c = q_s_scaled.contiguous()?;
    let beta_c = beta.contiguous()?;
    let dlast_c = decay_last_col.contiguous()?;

    let out_chunk = Tensor::zeros((b, h, c, dv), DType::BF16, device)?;
    let w_weighted = Tensor::zeros((b, h, c, dv), DType::BF16, device)?;

    {
        let (a_storage, a_layout) = a_c.storage_and_layout();
        let (b_storage, b_layout) = b_c.storage_and_layout();
        let (vp_storage, vp_layout) = vp_c.storage_and_layout();
        let (qss_storage, qss_layout) = qss_c.storage_and_layout();
        let (beta_storage, beta_layout) = beta_c.storage_and_layout();
        let (dlast_storage, dlast_layout) = dlast_c.storage_and_layout();
        let (out_storage, out_layout) = out_chunk.storage_and_layout();
        let (ww_storage, ww_layout) = w_weighted.storage_and_layout();

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
        let beta_cuda = match &*beta_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: beta must be on CUDA"),
        };
        let dlast_cuda = match &*dlast_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: decay_last_col must be on CUDA"),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: out_chunk must be on CUDA"),
        };
        let ww_cuda = match &*ww_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: w_weighted must be on CUDA"),
        };

        let stream = a_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let a_slice = a_cuda
            .as_cuda_slice::<bf16>()?
            .slice(a_layout.start_offset()..);
        let b_slice = b_cuda
            .as_cuda_slice::<bf16>()?
            .slice(b_layout.start_offset()..);
        let vp_slice = vp_cuda
            .as_cuda_slice::<bf16>()?
            .slice(vp_layout.start_offset()..);
        let qss_slice = qss_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qss_layout.start_offset()..);
        let beta_slice = beta_cuda
            .as_cuda_slice::<bf16>()?
            .slice(beta_layout.start_offset()..);
        let dlast_slice = dlast_cuda
            .as_cuda_slice::<bf16>()?
            .slice(dlast_layout.start_offset()..);
        let out_slice = out_cuda
            .as_cuda_slice::<bf16>()?
            .slice(out_layout.start_offset()..);
        let ww_slice = ww_cuda
            .as_cuda_slice::<bf16>()?
            .slice(ww_layout.start_offset()..);

        unsafe {
            let (a_ptr, _g1) = a_slice.device_ptr(&stream);
            let (b_ptr, _g2) = b_slice.device_ptr(&stream);
            let (vp_ptr, _g3) = vp_slice.device_ptr(&stream);
            let (qss_ptr, _g4) = qss_slice.device_ptr(&stream);
            let (beta_ptr, _g5) = beta_slice.device_ptr(&stream);
            let (dlast_ptr, _g6) = dlast_slice.device_ptr(&stream);
            let (out_ptr, _g7) = out_slice.device_ptr(&stream);
            let (ww_ptr, _g8) = ww_slice.device_ptr(&stream);

            let status = kiln_gdn_chunk_scan(
                a_ptr as *const _,
                b_ptr as *const _,
                vp_ptr as *const _,
                qss_ptr as *const _,
                beta_ptr as *const _,
                dlast_ptr as *const _,
                out_ptr as *mut _,
                ww_ptr as *mut _,
                (b * h) as i32,
                c as i32,
                dv as i32,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!("kiln_gdn_chunk_scan failed with status {status}");
            }
        }
    }

    Ok((out_chunk, w_weighted))
}

pub fn gdn_full_chunk_forward_supports(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
    beta: &Tensor,
    k_t: &Tensor,
    state: &Tensor,
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
        || beta.dtype() != DType::BF16
        || k_t.dtype() != DType::BF16
        || state.dtype() != DType::BF16
    {
        return false;
    }

    let Ok((b, h, c)) = g.dims3() else {
        return false;
    };
    if c != 64 {
        return false;
    }
    let Ok((b_v, h_v, c_v, dv)) = v.dims4() else {
        return false;
    };
    if (b_v, h_v, c_v) != (b, h, c) || dv == 0 || dv > 128 {
        return false;
    }
    let Ok((b_kkt, h_kkt, c1, c2)) = kkt.dims4() else {
        return false;
    };
    if (b_kkt, h_kkt, c1, c2) != (b, h, c, c) {
        return false;
    }
    let Ok((b_qkt, h_qkt, cq1, cq2)) = qkt.dims4() else {
        return false;
    };
    if (b_qkt, h_qkt, cq1, cq2) != (b, h, c, c) {
        return false;
    }
    let Ok((b_ks, h_ks, c_ks, dv_ks)) = ks_entry.dims4() else {
        return false;
    };
    if (b_ks, h_ks, c_ks, dv_ks) != (b, h, c, dv) {
        return false;
    }
    let Ok((b_qs, h_qs, c_qs, dv_qs)) = q_s.dims4() else {
        return false;
    };
    if (b_qs, h_qs, c_qs, dv_qs) != (b, h, c, dv) {
        return false;
    }
    let Ok((b_beta, h_beta, c_beta)) = beta.dims3() else {
        return false;
    };
    if (b_beta, h_beta, c_beta) != (b, h, c) {
        return false;
    }
    let Ok((b_kt, h_kt, dk, c_kt)) = k_t.dims4() else {
        return false;
    };
    if (b_kt, h_kt, c_kt) != (b, h, c) || dk == 0 || dk > 128 {
        return false;
    }
    let Ok((b_state, h_state, dk_state, dv_state)) = state.dims4() else {
        return false;
    };
    (b_state, h_state, dk_state, dv_state) == (b, h, dk, dv)
}

pub fn gdn_full_chunk_forward(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
    beta: &Tensor,
    k_t: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    if !gdn_full_chunk_forward_supports(g, v, kkt, qkt, ks_entry, q_s, beta, k_t, state) {
        candle_core::bail!(
            "kiln-gdn-kernel: gdn_full_chunk_forward: envelope violation \
             (g={:?}, v={:?}, kkt={:?}, qkt={:?}, ks_entry={:?}, q_s={:?}, beta={:?}, k_t={:?}, state={:?})",
            g.shape(),
            v.shape(),
            kkt.shape(),
            qkt.shape(),
            ks_entry.shape(),
            q_s.shape(),
            beta.shape(),
            k_t.shape(),
            state.shape()
        );
    }

    let (b, h, c) = g.dims3()?;
    let dk = k_t.dim(2)?;
    let dv = v.dim(3)?;
    let device = g.device();

    let g_c = g.contiguous()?;
    let v_c = v.contiguous()?;
    let kkt_c = kkt.contiguous()?;
    let qkt_c = qkt.contiguous()?;
    let ks_c = ks_entry.contiguous()?;
    let qs_c = q_s.contiguous()?;
    let beta_c = beta.contiguous()?;
    let kt_c = k_t.contiguous()?;
    let state_c = state.contiguous()?;

    let out_chunk = Tensor::zeros((b, h, c, dv), DType::BF16, device)?;

    {
        let (g_storage, g_layout) = g_c.storage_and_layout();
        let (v_storage, v_layout) = v_c.storage_and_layout();
        let (kkt_storage, kkt_layout) = kkt_c.storage_and_layout();
        let (qkt_storage, qkt_layout) = qkt_c.storage_and_layout();
        let (ks_storage, ks_layout) = ks_c.storage_and_layout();
        let (qs_storage, qs_layout) = qs_c.storage_and_layout();
        let (beta_storage, beta_layout) = beta_c.storage_and_layout();
        let (kt_storage, kt_layout) = kt_c.storage_and_layout();
        let (state_storage, state_layout) = state_c.storage_and_layout();
        let (out_storage, out_layout) = out_chunk.storage_and_layout();

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
        let beta_cuda = match &*beta_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: beta must be on CUDA"),
        };
        let kt_cuda = match &*kt_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: k_t must be on CUDA"),
        };
        let state_cuda = match &*state_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: state must be on CUDA"),
        };
        let out_cuda = match &*out_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-gdn-kernel: out_chunk must be on CUDA"),
        };

        let stream = g_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;

        let g_slice = g_cuda
            .as_cuda_slice::<bf16>()?
            .slice(g_layout.start_offset()..);
        let v_slice = v_cuda
            .as_cuda_slice::<bf16>()?
            .slice(v_layout.start_offset()..);
        let kkt_slice = kkt_cuda
            .as_cuda_slice::<bf16>()?
            .slice(kkt_layout.start_offset()..);
        let qkt_slice = qkt_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qkt_layout.start_offset()..);
        let ks_slice = ks_cuda
            .as_cuda_slice::<bf16>()?
            .slice(ks_layout.start_offset()..);
        let qs_slice = qs_cuda
            .as_cuda_slice::<bf16>()?
            .slice(qs_layout.start_offset()..);
        let beta_slice = beta_cuda
            .as_cuda_slice::<bf16>()?
            .slice(beta_layout.start_offset()..);
        let kt_slice = kt_cuda
            .as_cuda_slice::<bf16>()?
            .slice(kt_layout.start_offset()..);
        let state_slice = state_cuda
            .as_cuda_slice::<bf16>()?
            .slice(state_layout.start_offset()..);
        let out_slice = out_cuda
            .as_cuda_slice::<bf16>()?
            .slice(out_layout.start_offset()..);

        unsafe {
            let (g_ptr, _g1) = g_slice.device_ptr(&stream);
            let (v_ptr, _g2) = v_slice.device_ptr(&stream);
            let (kkt_ptr, _g3) = kkt_slice.device_ptr(&stream);
            let (qkt_ptr, _g4) = qkt_slice.device_ptr(&stream);
            let (ks_ptr, _g5) = ks_slice.device_ptr(&stream);
            let (qs_ptr, _g6) = qs_slice.device_ptr(&stream);
            let (beta_ptr, _g7) = beta_slice.device_ptr(&stream);
            let (kt_ptr, _g8) = kt_slice.device_ptr(&stream);
            let (state_ptr, _g9) = state_slice.device_ptr(&stream);
            let (out_ptr, _g10) = out_slice.device_ptr(&stream);

            let status = kiln_gdn_full_chunk_forward(
                g_ptr as *const _,
                v_ptr as *const _,
                kkt_ptr as *const _,
                qkt_ptr as *const _,
                ks_ptr as *const _,
                qs_ptr as *const _,
                beta_ptr as *const _,
                kt_ptr as *const _,
                state_ptr as *mut _,
                out_ptr as *mut _,
                (b * h) as i32,
                c as i32,
                dk as i32,
                dv as i32,
                raw_stream,
            );
            if status != 0 {
                candle_core::bail!("kiln_gdn_full_chunk_forward failed with status {status}");
            }
        }
    }

    *state = state_c;
    Ok(out_chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn patterned_data(len: usize, scale: f32, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = i as f32;
                ((x * 0.013 + phase).sin() * 0.7 + (x * 0.007 + phase * 0.5).cos() * 0.3)
                    * scale
            })
            .collect()
    }

    fn max_mean_abs_diff(lhs: &Tensor, rhs: &Tensor) -> Result<(f32, f32)> {
        let diff = (lhs.to_dtype(DType::F32)? - rhs.to_dtype(DType::F32)?)?.abs()?;
        let flat = diff.flatten_all()?;
        Ok((
            flat.max(0)?.to_scalar::<f32>()?,
            flat.mean(0)?.to_scalar::<f32>()?,
        ))
    }

    #[test]
    fn test_cuda_decode_gates_recurrent_matches_split_path() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping decode gates+recurrent parity test: {err}");
                return Ok(());
            }
        };

        let batch = 1usize;
        let seq_len = 1usize;
        let heads = 32usize;
        let dk = 128usize;
        let dv = 128usize;

        let q = Tensor::from_slice(
            &patterned_data(batch * seq_len * heads * dk, 0.35, 0.1),
            (batch, seq_len, heads, dk),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let k = Tensor::from_slice(
            &patterned_data(batch * seq_len * heads * dk, 0.25, 0.7),
            (batch, seq_len, heads, dk),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let v = Tensor::from_slice(
            &patterned_data(batch * seq_len * heads * dv, 0.5, 1.3),
            (batch, seq_len, heads, dv),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let a = Tensor::from_slice(
            &patterned_data(batch * seq_len * heads, 0.4, 2.1),
            (batch, seq_len, heads),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let b = Tensor::from_slice(
            &patterned_data(batch * seq_len * heads, 0.6, 2.9),
            (batch, seq_len, heads),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let a_log = Tensor::from_slice(&patterned_data(heads, 0.15, 3.7), (heads,), &device)?
            .to_dtype(DType::BF16)?;
        let dt_bias = Tensor::from_slice(&patterned_data(heads, 0.2, 4.3), (heads,), &device)?
            .to_dtype(DType::BF16)?;
        let z = Tensor::from_slice(
            &patterned_data(batch * seq_len * heads * dv, 0.45, 4.9),
            (batch, seq_len, heads, dv),
            &device,
        )?
        .to_dtype(DType::BF16)?;
        let weight = Tensor::from_slice(&patterned_data(dv, 0.3, 5.5), (dv,), &device)?
            .to_dtype(DType::BF16)?;
        let state = Tensor::from_slice(
            &patterned_data(batch * heads * dk * dv, 0.08, 6.1),
            (batch, heads, dk, dv),
            &device,
        )?
        .to_dtype(DType::BF16)?;

        let (beta, g) = gdn_gates(&a, &b, &a_log, &dt_bias)?;
        let q_split = q.squeeze(1)?.contiguous()?;
        let k_split = k.squeeze(1)?.contiguous()?;
        let v_split = v.squeeze(1)?.contiguous()?;
        let beta_split = beta.squeeze(1)?.contiguous()?;
        let g_split = g.squeeze(1)?.contiguous()?;
        let mut state_split = state.copy()?;
        let out_split = gdn_recurrent_forward(
            &q_split,
            &k_split,
            &v_split,
            &beta_split,
            &g_split,
            &mut state_split,
        )?
        .unsqueeze(1)?;

        let mut state_fused = state.copy()?;
        let out_fused = gdn_decode_gates_recurrent(
            &q,
            &k,
            &v,
            &a,
            &b,
            &a_log,
            &dt_bias,
            &mut state_fused,
            &z,
            &weight,
            1e-6,
        )?;

        let (out_max, out_mean) = max_mean_abs_diff(&out_fused, &out_split)?;
        let (state_max, state_mean) = max_mean_abs_diff(&state_fused, &state_split)?;
        eprintln!(
            "cuda decode gates+recurrent vs split: out max={out_max:e} mean={out_mean:e}, state max={state_max:e} mean={state_mean:e}"
        );

        assert!(
            out_max == 0.0,
            "fused decode recurrent output max_abs_diff={out_max:e}"
        );
        assert!(
            out_mean == 0.0,
            "fused decode recurrent output mean_abs_diff={out_mean:e}"
        );
        assert!(
            state_max == 0.0,
            "fused decode recurrent state max_abs_diff={state_max:e}"
        );
        assert!(
            state_mean == 0.0,
            "fused decode recurrent state mean_abs_diff={state_mean:e}"
        );

        Ok(())
    }
}

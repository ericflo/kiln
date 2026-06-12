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
//! Phase 7 closeout (#1082): the candle-typed surface has been removed.
//! All entry points are now `kiln-tensor`-typed `*_kt` functions exported
//! from [`kt_api`]:
//!
//! - [`gdn_forward_substitution_kt`] — chunkwise prefill forward-sub step.
//!   Thin kt wrapper around a single fused CUDA kernel that
//!   replaces the per-token forward-substitution loop in kiln's
//!   chunkwise analytical GDN recurrence:
//!
//!   ```text
//!   W[t, :] = beta[t] * ( V_prime[t, :]
//!                        - sum_{i<t} A_strict[t, i] * W[i, :] )
//!   ```
//!
//! - [`gdn_recurrent_forward_kt`] — seq_len==1 decode fast path. Collapses
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
//! Anything outside that envelope falls back to the Rust reference in
//! `kiln-model::forward::compute_w_chunk_fallback`.
//!
//! # Not yet vendored
//!
//! Per `PROFILING.md` (post-PR #130, Phase 6), the next GDN-side
//! targets are the GDN body ranges (`gated_norm`, `gates`, `conv`,
//! `qk_norm`) and the two RMSNorm stages — these are upstream of the
//! chunkwise recurrence and are *not* covered by this crate.

/// kiln-tensor-typed surface. Same FFI used by the kernels.
mod kt_api;

// Pure shape/dtype predicates + the error type carry no FFI and compile on any
// configuration (including with neither GPU backend enabled).
pub use kt_api::{
    GdnError, gdn_chunk_prep_supports_kt, gdn_chunk_scan_supports_kt,
    gdn_decode_gates_recurrent_supports_kt, gdn_decode_qk_norm_gates_recurrent_rmsnorm_supports_kt,
    gdn_decode_qk_norm_gates_recurrent_supports_kt, gdn_full_chunk_forward_multiblock_supports_kt,
    gdn_full_chunk_forward_supports_kt, gdn_gated_rms_norm_f32_weight_supports_kt,
    gdn_gated_rms_norm_supports_kt, gdn_gates_supports_kt,
};

// The device-launching `_kt` entry points bottom out in the FFI symbols and the
// backend-neutral `kiln_kt_bridge::device_*` seam, so they need a GPU backend
// (cuda or rocm). Phase R.7: the ROCm path reuses these byte-for-byte via the
// neutral seam (the seam dispatches `Device::Rocm` tensors to the hipcc-built
// kernels).
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use kt_api::{
    GdnGatedRmsNormBwdKt, gdn_chunk_prep_kt, gdn_chunk_scan_kt, gdn_decode_gates_recurrent_bf16_kt,
    gdn_decode_gates_recurrent_vf32_bf16_kt, gdn_decode_qk_norm_gates_recurrent_bf16_kt,
    gdn_decode_qk_norm_gates_recurrent_qf32_vbf16_bf16_kt,
    gdn_decode_qk_norm_gates_recurrent_qf32_vf32_bf16_kt,
    gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16_kt,
    gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vbf16_bf16_kt,
    gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vf32_bf16_kt,
    gdn_decode_qk_norm_gates_recurrent_rmsnorm_vf32_bf16_kt,
    gdn_decode_qk_norm_gates_recurrent_vf32_bf16_kt, gdn_forward_substitution_kt,
    gdn_full_chunk_forward_kt, gdn_full_chunk_forward_multiblock_kt,
    gdn_gated_rms_norm_bf16_f32_weight_kt, gdn_gated_rms_norm_bf16_kt,
    gdn_gated_rms_norm_bwd_bf16_f32_weight_kt, gdn_gated_rms_norm_bwd_bf16_kt,
    gdn_gated_rms_norm_bwd_supports_kt, gdn_gates_bf16_f32_bf16_params_kt,
    gdn_gates_bf16_f32_params_kt, gdn_gates_bf16_kt, gdn_l2_norm_scale_bwd_bf16_kt,
    gdn_l2_norm_scale_bwd_supports_kt, gdn_recurrent_forward_kt,
};

// The device-launching FFI symbols are provided by build.rs (nvcc under
// `--features cuda`, hipcc under `--features rocm`). Gate the declarations so the
// crate still type-checks with neither GPU backend enabled (only the pure
// shape/dtype predicates remain in that configuration).
#[cfg(any(feature = "cuda", feature = "rocm"))]
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

    fn kiln_gdn_decode_qk_norm_gates_recurrent_vf32_bf16(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        a_log: *const core::ffi::c_void,
        dt_bias: *const core::ffi::c_void,
        state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        q_heads: i32,
        value_heads: i32,
        dk: i32,
        dv: i32,
        q_scale: f32,
        qk_eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_decode_qk_norm_gates_recurrent_bf16(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        a_log: *const core::ffi::c_void,
        dt_bias: *const core::ffi::c_void,
        state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        q_heads: i32,
        value_heads: i32,
        dk: i32,
        dv: i32,
        q_scale: f32,
        qk_eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vf32_bf16(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        a_log: *const core::ffi::c_void,
        dt_bias: *const core::ffi::c_void,
        state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        q_heads: i32,
        value_heads: i32,
        dk: i32,
        dv: i32,
        q_scale: f32,
        qk_eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_decode_qk_norm_gates_recurrent_qf32_vbf16_bf16(
        q: *const core::ffi::c_void,
        k: *const core::ffi::c_void,
        v: *const core::ffi::c_void,
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        a_log: *const core::ffi::c_void,
        dt_bias: *const core::ffi::c_void,
        state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        q_heads: i32,
        value_heads: i32,
        dk: i32,
        dv: i32,
        q_scale: f32,
        qk_eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_vf32_bf16(
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
        q_scale: f32,
        qk_eps: f32,
        rms_eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_bf16(
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
        q_scale: f32,
        qk_eps: f32,
        rms_eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vf32_bf16(
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
        q_scale: f32,
        qk_eps: f32,
        rms_eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_decode_qk_norm_gates_recurrent_rmsnorm_qf32_vbf16_bf16(
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
        q_scale: f32,
        qk_eps: f32,
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

    fn kiln_gdn_full_chunk_forward_multiblock(
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
        dv_tile: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

// Phase 7 closeout (#1082): the candle-typed GDN decode entries
// (`gdn_decode_gates_recurrent[_supports]`,
// `gdn_decode_qk_norm_gates_recurrent[_supports]`,
// `gdn_decode_qk_norm_gates_recurrent_rmsnorm[_supports]`) and the
// `with_decode_gates_recurrent_outputs` thread-local wrapper have
// been removed. The production path is now the kt-typed surface
// (`gdn_decode_*_recurrent_*_kt` in `kt_api.rs`); cuda_graph.rs no
// longer needs to install pre-allocated outputs because the kt
// entries own their own allocations end-to-end. Same closeout
// pattern as conv1d (commit 2ebcfb08) and marlin (0841c266).

#[cfg(any(feature = "cuda", feature = "rocm"))]
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
        a_row_stride: i32,
        b_row_stride: i32,
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
        a_row_stride: i32,
        b_row_stride: i32,
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
        a_row_stride: i32,
        b_row_stride: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

}

#[cfg(any(feature = "cuda", feature = "rocm"))]
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

    fn kiln_gdn_gated_rms_norm_wf32_bf16(
        x: *const core::ffi::c_void,
        z: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        rows: i32,
        hidden: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_gated_rms_norm_bwd_bf16(
        grad_out: *const core::ffi::c_void,
        x: *const core::ffi::c_void,
        z: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        d_x: *mut core::ffi::c_void,
        d_z: *mut core::ffi::c_void,
        d_weight: *mut core::ffi::c_void,
        rows: i32,
        hidden: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_gated_rms_norm_bwd_wf32_bf16(
        grad_out: *const core::ffi::c_void,
        x: *const core::ffi::c_void,
        z: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        d_x: *mut core::ffi::c_void,
        d_z: *mut core::ffi::c_void,
        d_weight: *mut core::ffi::c_void,
        rows: i32,
        hidden: i32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_gdn_l2_norm_scale_bwd_bf16(
        grad_out: *const core::ffi::c_void,
        x: *const core::ffi::c_void,
        d_x: *mut core::ffi::c_void,
        rows: i32,
        hidden: i32,
        scale: f32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Default dv tile size for the multi-block path. 32 columns per block gives
/// 4 blocks per (B,H) at the Qwen3.5-4B dv=128 envelope, taking the launch
/// from 32 blocks (~42% occupancy of 76 SMs on RTX 4090 Laptop) to 128 blocks
/// (~1.7x oversubscription).
pub const GDN_FULL_CHUNK_FORWARD_MULTIBLOCK_DV_TILE: usize = 32;

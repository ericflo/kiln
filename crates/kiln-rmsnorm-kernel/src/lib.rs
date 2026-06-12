//! Vendored fused norm CUDA kernels (Liger-style).
//!
//! This crate hosts decode-critical Liger-style fused norm kernels for kiln:
//!
//! 1. [`fused_rmsnorm_kt`] / [`fused_rmsnorm_backward_kt`] — Phase 10
//!    long-context training path: Qwen3.5-style RMSNorm
//!    `(1 + w) * x * rsqrt(mean(x^2) + eps)` plus a manual CUDA backward
//!    kernel. The candle-autograd shim that wires these through
//!    `KtForwardOp2` (`fused_rmsnorm_via_kt_forward_op`) moved to
//!    `kiln-model::rmsnorm_candle_shim` in (#1082) so this crate stays
//!    candle-free; it saves only `x` and `weight` (not the F32
//!    intermediates the candle-op chain materializes), replacing the ~11
//!    candle ops behind `kiln-model::forward::rms_norm`. Used by
//!    `kiln/norm/pre_attn` and `kiln/norm/pre_mlp`. For Qwen3.5-4B at
//!    T=8192 this avoids ~32 × 2 saved F32 RMSNorm intermediates per
//!    training segment.
//! 2. [`fused_l2_qk_norm_kt`] — fused L2-norm(Q) + scale(Q) + L2-norm(K) used
//!    by GDN linear attention. Replaces the ~11 candle ops behind the
//!    `kiln/gdn/qk_norm` block in `forward.rs`.
//! 3. [`fused_l2_qk_norm_gqa_kt`] — CUDA GDN GQA fast path that normalizes
//!    unexpanded `[B, T, nk, dk]` Q/K and emits expanded `[B, T, nv, dk]`
//!    outputs in one launch.
//! 4. [`fused_rotary_qk_kt`] — decode/paged-attention RoPE(Q,K) for contiguous
//!    bf16 Q/K tensors using precomputed f32 cos/sin tables. (kt-typed only;
//!    the candle-typed wrappers were removed in (#1082).)
//! 5. [`fused_mlp_silu_mul_kt`] — fused bf16 `silu(gate) * up` for Qwen3.5
//!    SwiGLU MLPs. (kt-typed only; the candle-typed wrappers were removed
//!    in (#1082).)
//! 6. [`fused_sigmoid_mul_kt`] — fused bf16 `x * sigmoid(gate)` for attention
//!    output gates. The candle-typed `fused_sigmoid_mul` entry was removed
//!    in (#1082); the storage-level `fused_sigmoid_mul_storage` candle
//!    CustomOp2 backing for `CudaSigmoidMulTrainingBf16` moved to
//!    `kiln-model::rmsnorm_candle_shim`.
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
//! - [`fused_rmsnorm_kt`] / [`fused_rmsnorm_backward_kt`] — kt-typed
//!   RMSNorm forward + manual CUDA backward. The autograd-aware shim
//!   (`fused_rmsnorm_via_kt_forward_op`, uses `KtForwardOp2` when grads
//!   are propagated) and its `(x, weight)` capability check (`supports`)
//!   moved to `kiln-model::rmsnorm_candle_shim` in (#1082).
//! - [`supports_rmsnorm_kt`] — kt-typed `(x, weight)` capability check for
//!   the RMSNorm kernel.
//! - [`fused_l2_qk_norm_kt`] — kt-typed wrapper around the GDN QK fused-norm
//!   kernel. Returns `(q_out, k_out)`.
//! - [`supports_l2_qk_norm_kt`] — capability check for the QK kernel.
//! - [`fused_l2_qk_norm_gqa_kt`] / [`supports_l2_qk_norm_gqa_kt`] — GDN GQA
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
//! Backward currently only supported for the RMSNorm kernel (via
//! [`fused_rmsnorm_backward_kt`], wired into autograd by the
//! `kiln-model::rmsnorm_candle_shim` `KtForwardOp2` shim); the QK-norm
//! kernels remain forward-only.

// (#1082) candle-drop: this crate is now candle-free. The candle-typed
// glue (the `*_storage` / `matmul_f32_bf16w` / `causal_depthwise_conv1d_f32*`
// wrappers + the `KtForwardOp2` autograd shim, formerly `kt_forward_op.rs`)
// was the last user of `candle-core` here; it moved UP into the consumer
// crate `kiln-model::rmsnorm_candle_shim`, which keeps candle. The raw
// `extern "C"` FFI declarations below + the pure-`kiln_tensor` kt surface
// (`kt_api`, `kt_tape`) stay. The moved wrappers re-declare the subset of
// these FFI symbols they need; both declarations resolve to the same
// single linker symbol in `libkiln_rmsnorm_kernel.a` (no duplicate-def).

/// kiln-tensor-typed surface. Same FFI symbols as the (now-relocated)
/// candle-typed surface.
///
/// Gated on a GPU backend (Phase R.7): the `_kt` wrappers call the fused FFI
/// kernels, which only exist when a backend is compiled in. With `cuda` they
/// drive the nvcc-built lib; with `rocm` the hipcc-built lib; the wrapper
/// bodies are backend-neutral (they route through `kiln_kt_bridge::device_*`).
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod kt_api;

/// Phase 6a/CP-4 (#1082): parallel kt-tape entry that drops the candle
/// CustomOp2 wrapper in favour of recording onto a `kiln_autograd::Tape`
/// directly. Same FFI symbols, same envelope. See `kt_tape.rs` for the
/// pilot port rationale.
#[cfg(any(feature = "cuda", feature = "rocm"))]
mod kt_tape;
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use kt_api::{
    RmsNormError, adamw_step_bf16_kt, adamw_step_f32_kt, attn_decode_qkv_split_qk_norm_rope_kt,
    causal_depthwise_conv1d_bwd_input_kt, causal_depthwise_conv1d_bwd_state_kt,
    causal_depthwise_conv1d_bwd_weight_kt, causal_depthwise_conv1d_inplace_kt,
    causal_depthwise_conv1d_kt, f32_to_bf16_kt, fused_l2_qk_norm_gqa_kt, fused_l2_qk_norm_kt,
    fused_mlp_silu_mul_kt, fused_mlp_silu_mul_packed_kt, fused_rmsnorm_backward_kt,
    fused_rmsnorm_kt, fused_rotary_one_bwd_kt, fused_rotary_one_kt, fused_rotary_qk_kt,
    fused_sigmoid_mul_kt, lora_add_inplace_f32_kt, lora_decode_add_full_kt, lora_decode_add_kt,
    lora_decode_hidden_kt, muon_step_bf16_kt, muon_step_f32_kt, sgd_step_bf16_kt, sgd_step_f32_kt,
    silu_inplace_save_sigmoid_f32_kt, supports_attn_decode_qkv_prep_kt, supports_l2_qk_norm_gqa_kt,
    supports_l2_qk_norm_kt, supports_lora_decode_add_kt, supports_mlp_silu_mul_kt,
    supports_mlp_silu_mul_packed_kt, supports_optimizer_step_kt, supports_rmsnorm_kt,
    supports_rotary_qk_kt, supports_sigmoid_mul_kt,
};
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use kt_tape::{CudaFusedRmsNormBackward, fused_rmsnorm_via_kt_tape};

// The fused-kernel FFI symbols are provided by the backend lib built in
// build.rs: nvcc-built under `cuda`, hipcc-built under `rocm`. Gate the
// declarations on a backend so a no-backend build of the crate doesn't
// reference symbols that won't be linked.
#[cfg(any(feature = "cuda", feature = "rocm"))]
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

    fn kiln_muon_step_f32(
        param: *mut f32,
        grad: *const f32,
        momentum: *mut f32,
        lr: f32,
        mom: f32,
        nesterov: i32,
        ns_iters: i32,
        weight_decay: f32,
        rows: i32,
        cols: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_muon_step_bf16(
        param: *mut core::ffi::c_void,
        grad: *const core::ffi::c_void,
        momentum: *mut core::ffi::c_void,
        lr: f32,
        mom: f32,
        nesterov: i32,
        ns_iters: i32,
        weight_decay: f32,
        rows: i32,
        cols: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

// All in-crate candle-typed parity tests were deleted in (#1082) alongside
// the candle-typed entries they exercised. The kt-typed surface is covered
// by the kt_api unit tests under cfg(test) and the integration test in
// `tests/kt_v2_smoke.rs`.

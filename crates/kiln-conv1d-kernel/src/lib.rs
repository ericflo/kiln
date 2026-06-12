//! Vendored mamba-ssm `causal_conv1d_update` CUDA kernel, narrowed to kiln's
//! Qwen3.5-4B GDN decode envelope.
//!
//! # Why
//!
//! `kiln-model::forward::causal_conv1d_decode` used to express the
//! single-step depthwise conv1d + state update as a chain of candle
//! ops — ~6 CUDA launches per GDN layer × 24 layers ≈ 144 launches
//! per decode step. Per PROFILING.md (post-PR #158) the
//! `kiln/gdn/conv` NVTX region was **12.2 %** of decode wall-clock.
//!
//! This crate collapses the whole chain into a single launch:
//! `kiln_causal_conv1d_update_bf16_f32` — one thread per (batch, channel),
//! K=4 registers for the window, F32 accumulator, F32 state, F32 silu
//! epilogue.
//!
//! # Provenance
//!
//! Algorithm vendored from
//! [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)
//! (mamba-ssm) `csrc/causal_conv1d_update.cu`, kIsCircularBuffer=false,
//! kNBytes=2, kWidth=4 specialisation. License: Apache 2.0 upstream; this
//! crate retains the same licence for the vendored portion.
//!
//! # Scope
//!
//! - Decode single-step only (`seq_len == 1`).
//! - bf16 activations, bf16 weights, F32 state, F32 output.
//! - `kernel_size == 4` (Qwen3.5 GDN). Other widths return `Ok(None)`.
//! - No bias (kiln doesn't load conv1d bias — see forward.rs line ~1478).
//! - No per-batch conv_state_indices / circular buffer.
//! - SiLU fused inline (matches the `cuda_silu` call immediately after the
//!   conv in `gated_deltanet_forward`).
//!
//! # API
//!
//! Phase 7 (#1082) — the crate now exposes only the kt-typed surface
//! (`kt_api::*_kt`). The previous candle-typed `supports*` /
//! `causal_conv1d_*` functions had zero production callers after
//! 2ebcfb08 (cuda.rs migration) and have been removed alongside their
//! in-lib parity scaffolds. The kt smoke tests at
//! `tests/kt_v2_smoke.rs` exercise the kt API on real CUDA hardware
//! via the candle-free `Tensor::cuda_from_slice` substrate helper.

/// kiln-tensor-typed kt-API surface. All callers route through this
/// module; the crate no longer exposes a candle-typed parallel API.
mod kt_api;
// The device-launching entry points need a GPU backend (cuda or rocm) for the
// FFI symbols below. The pure shape/dtype predicates have no FFI and compile on
// any configuration.
pub use kt_api::{Conv1dError, supports_kt, supports_prefill_kt, supports_update_kt};
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub use kt_api::{causal_conv1d_prefill_kt, causal_conv1d_update_kt};

#[cfg(any(feature = "cuda", feature = "rocm"))]
unsafe extern "C" {
    pub(crate) fn kiln_causal_conv1d_update_bf16_f32(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        conv_state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        channels: i32,
        kernel_width: i32,
        silu: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_causal_conv1d_prefill_bf16_f32(
        x: *const core::ffi::c_void,
        weight: *const core::ffi::c_void,
        conv_state: *mut core::ffi::c_void,
        out: *mut core::ffi::c_void,
        batch: i32,
        channels: i32,
        seq_len: i32,
        kernel_width: i32,
        silu: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

//! OPD top-K reverse-KL — fused CUDA backward FFI declarations + envelope.
//!
//! Originally this module hosted the candle `CustomOp1` wrapper
//! `OpdLossCustomOp` along with the analytic chunked backward, the
//! fused-CUDA fast paths, and the `compute_per_position_metrics`
//! diagnostics. After Wave-9 (`0c1be227`) wired the candle `CustomOp::bwd`
//! through the kt bridge, then Wave-12 (`#1082`) flipped the production
//! caller in `kiln-train::opd::opd_step_loss` onto the kt-shim
//! `kiln_train::opd_candle_shim::opd_top_k_reverse_kl_per_position_via_kt_forward_op`
//! (and Wave-13 added the tape-gated short-circuit
//! `kiln_train::opd_candle_shim::try_tape_opd_per_position_cuda`), the
//! candle CustomOp1 became dead in production — used only by the smoke
//! `kiln-train::opd::opd_train_synthetic_validation` and the
//! `kiln-train/tests/vk_cuda_opd_parity.rs` parity gate. Both were
//! migrated to the production shim on 2026-05-28 (commit `e495554c`),
//! at which point the candle CustomOp1, its analytic backward, the
//! fused-FWD CUDA kernel symbols, and the metrics path could all go
//! ((#1082)).
//!
//! What survives in this module:
//!
//! * `extern "C"` declarations for the fused backward CUDA kernels
//!   `kiln_opd_topk_kl_bwd_{bf16,f32}` — still called by the
//!   kt-typed backward
//!   [`crate::kt_api::opd_top_k_reverse_kl_phase_b_bwd_kt`], which
//!   powers both the kt-tape pilot
//!   ([`crate::opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`])
//!   and the candle kt-forward-op shim
//!   (`kiln_train::opd_candle_shim::opd_top_k_reverse_kl_per_position_via_kt_forward_op`).
//! * [`cuda_kernel_supports`] — the `(K ∈ {16, 32}, dtype ∈ {F32, BF16})`
//!   envelope used by both call sites above.
//!
//! What was removed:
//!
//! * `OpdLossCustomOp` (the candle `CustomOp1` impl) + `apply_op` helper.
//! * `opd_top_k_reverse_kl_phase_b` and `_per_position` (the candle entry
//!   points that wrapped `OpdLossCustomOp` via `apply_op1`).
//! * `OpdLossOutput` enum (scalar-mean vs per-position output mode).
//! * `opd_loss_phase_b_backward_via_kt_bridge` +
//!   `opd_bwd_kt_bridge_disabled` (the on-by-default bridge into the
//!   kt-typed backward — now reached only through the kt-shim).
//! * Fused-FWD FFI declarations
//!   `kiln_opd_topk_kl_fwd_{bf16,f32}` — only called by the deleted
//!   `OpdLossCustomOp::cuda_kernel_forward`. The production path runs
//!   the kt composite (`per_position_forward_kt` in `kt_api`) instead.
//! * `PerPositionMetrics` + `compute_per_position_metrics` + the
//!   `kiln_opd_topk_metrics_{bf16,f32}` FFI declarations (no external
//!   callers; the kt-typed sibling `compute_per_position_metrics_kt`
//!   covers the same diagnostic, and neither has a live caller as of
//!   the audit in `docs/opd-loss-kernel-candle-removal-stop-2026-05-28.md`).
//!
//! The matching `extern "C"` definitions for the deleted kernels in
//! `csrc/opd_topk_kl.cu` are removed in the same commit so the static
//! library doesn't carry dead code.

use kiln_tensor::DType;

// FFI declarations for the fused CUDA backward kernel (§9.2 of the
// grand plan). Linked in only when the `cuda` feature is active — the
// `build.rs` compiles `csrc/opd_topk_kl.cu` and produces
// `libkiln_opd_loss_kernel.a` which Cargo links into the binary.
//
// The forward symbols `kiln_opd_topk_kl_fwd_{bf16,f32}` were retired
// in (#1082) — production runs the kt composite
// (`crate::kt_api::per_position_forward_kt`) on CUDA storage instead.
// The metrics symbols `kiln_opd_topk_metrics_{bf16,f32}` were retired
// at the same time (no external callers).
#[cfg(feature = "cuda")]
unsafe extern "C" {
    // crate-visible so `kt_api::opd_top_k_reverse_kl_phase_b_bwd_kt`
    // can call the same FFI symbols the (removed) candle path used.
    // Bit-exact by construction across the kt-shim and kt-tape paths
    // since they share these symbols.
    pub(crate) fn kiln_opd_topk_kl_bwd_bf16(
        hidden: *const core::ffi::c_void,
        head_t: *const core::ffi::c_void,
        topk_indices: *const core::ffi::c_void,
        topk_lp_q: *const core::ffi::c_void,
        grad_loss: *const core::ffi::c_void,
        scale_factor: f32,
        d_hidden: *mut core::ffi::c_void,
        t_active: i32,
        hidden_size: i32,
        vocab_size: i32,
        top_k: i32,
        output_mode: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    pub(crate) fn kiln_opd_topk_kl_bwd_f32(
        hidden: *const core::ffi::c_void,
        head_t: *const core::ffi::c_void,
        topk_indices: *const core::ffi::c_void,
        topk_lp_q: *const core::ffi::c_void,
        grad_loss: *const core::ffi::c_void,
        scale_factor: f32,
        d_hidden: *mut core::ffi::c_void,
        t_active: i32,
        hidden_size: i32,
        vocab_size: i32,
        top_k: i32,
        output_mode: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Returns `true` when the fused CUDA backward kernel supports the
/// requested `(top_k, dtype)` combination. K ∈ {16, 32} is the
/// milestone-5 fast path (§6 default is 32; the kernel hits that
/// with 1024 threads per block, the Ampere max). Both kt-shim and
/// kt-tape call sites pre-check via this helper before borrowing
/// into the kt bridge and dispatching the fused backward.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_kernel_supports(top_k: usize, dtype: DType) -> bool {
    let dtype_ok = matches!(dtype, DType::F32 | DType::BF16);
    dtype_ok && (top_k == 16 || top_k == 32)
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub(crate) fn cuda_kernel_supports(_top_k: usize, _dtype: DType) -> bool {
    false
}

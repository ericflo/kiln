//! Production-caller migration to `KtForwardOp` for the fused RMSNorm
//! forward+backward op ((#1082) — see `docs/CANDLE_REMOVAL_PLAN.md`).
//!
//! # What this module wires together
//!
//! The kiln-model production caller
//! (`crates/kiln-model/src/forward.rs::rms_norm`, the autograd-tracked
//! path) previously dispatched to [`crate::fused_rmsnorm_with_autograd`]
//! — a thin wrapper over [`crate::RmsNormCustomOp`] (a candle
//! [`candle_core::CustomOp2`]). That CustomOp's `cuda_fwd` calls the
//! candle-typed [`crate::fused_rmsnorm`], and its `bwd` body (since
//! commit `341da876`) routes through the kt bridge via
//! [`crate::fused_rmsnorm_backward_via_kt_bridge`].
//!
//! [`fused_rmsnorm_via_kt_forward_op`] replaces that CustomOp wrapper
//! with a **single** generic candle `CustomOp2` —
//! [`kiln_kt_bridge::forward_op::KtForwardOp2`] (commit `095f1c74`) —
//! whose forward closure calls the kt-typed
//! [`crate::kt_api::fused_rmsnorm_kt`] and whose backward closure
//! calls [`crate::kt_api::fused_rmsnorm_backward_kt`] on the same
//! `kiln_fused_rmsnorm{,_bwd}` FFI symbols. Both halves of the autograd
//! roundtrip now go through kt, removing one production caller from
//! the candle-typed RmsNormCustomOp surface so it can eventually be
//! deleted alongside the rest of the candle-typed kernel entries
//! (see `docs/CANDLE_REMOVAL_PLAN.md`).
//!
//! # Mirrors the OPD migration template
//!
//! Same shape as
//! `kiln-opd-loss-kernel/src/kt_forward_op.rs` (commit `f214f168`),
//! adapted from `KtForwardOp1` (unary OPD shape) to `KtForwardOp2`
//! (binary `(x, weight)` shape). The differences:
//!
//! - **Forward closure**: calls the kt-typed
//!   [`crate::kt_api::fused_rmsnorm_kt`] directly (the kt forward has
//!   no axis-N gather dependency — it's just `(x, weight) -> y` over
//!   the existing `kiln_fused_rmsnorm` FFI symbol, identical to what
//!   the candle path already calls).
//! - **Backward closure**: same `(grad_x, grad_w_partial_f32)` kt
//!   path the `RmsNormCustomOp::bwd` migration uses
//!   ([`crate::fused_rmsnorm_backward_kt_via_kt_forward_op_bwd`]
//!   helper, defined below), with the F32→BF16 cast of `grad_w` done
//!   in the kt domain via the same direct `kiln_f32_to_bf16` call
//!   the existing `fused_rmsnorm_backward_via_kt_bridge` uses.
//!
//! # Envelope and fallback
//!
//! The kt-typed forward + backward are gated to:
//! `dtype == BF16` for both `x` and `weight`, CUDA storage, contiguous,
//! `hidden <= 8192`. These match [`crate::supports`] /
//! [`crate::supports_autograd`] CUDA-side exactly (which is what the
//! kiln-model caller already checks before calling
//! `fused_rmsnorm_with_autograd`). When ANY of the following hold the
//! shim falls back to [`crate::fused_rmsnorm_with_autograd`] — the
//! candle-typed CustomOp2 wrapper — so the production caller stays
//! correct for the full input envelope it supports today:
//!
//! - `KILN_DISABLE_RMSNORM_KT_FORWARD_OP=1` (kill switch — same
//!   convention as `KILN_DISABLE_OPD_KT_FORWARD_OP` (commit
//!   `f214f168`), `KILN_DISABLE_RMSNORM_BWD_KT_BRIDGE`,
//!   `KILN_DISABLE_RMSNORM_KERNEL`, `KILN_DISABLE_FUSED_CONV1D`, etc.).
//! - `x` is not on CUDA, or `weight` is not on CUDA.
//! - `x.dtype()` is not BF16 (the CUDA kernel envelope is bf16-only).
//! - `weight.dtype()` is not BF16.
//! - `x` or `weight` is not contiguous.
//! - `hidden > 8192` (outside the kernel envelope).
//! - rank < 1 / weight shape mismatch (cheap correctness check).
//!
//! Falling back preserves the autograd chain — the candle
//! CustomOp2 wrapper attaches the same gradient parents and produces
//! gradients of the same shape/dtype.
//!
//! # Why this lives in `kiln-rmsnorm-kernel` and not `kiln-model`
//!
//! Three reasons (mirror of the OPD justification):
//!
//! 1. The kt-typed forward + backward entries
//!    ([`crate::kt_api::fused_rmsnorm_kt`] and
//!    [`crate::kt_api::fused_rmsnorm_backward_kt`]) plus the
//!    `supports` envelope check are crate-internal here.
//! 2. The kill switch + envelope check are kernel-policy concerns;
//!    pushing them into `kiln-model::forward::rms_norm` would
//!    duplicate them per call site.
//! 3. `kiln-model` already depends on `kiln-rmsnorm-kernel`, but not
//!    directly on `kiln-kt-bridge::forward_op`. Keeping the shim
//!    plumbing here means `kiln-model` doesn't need to learn about
//!    the bridge's `KtForwardOp2` shape.
//!
//! The call site change in `kiln-model::forward::rms_norm` is then a
//! one-line swap:
//! `kiln_rmsnorm_kernel::fused_rmsnorm_with_autograd(...)` →
//! `kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_forward_op(...)`.
//!
//! # Numerical contract
//!
//! Forward: bit-exact equality with [`crate::fused_rmsnorm`] (the
//! candle-typed entry) — both call the same `kiln_fused_rmsnorm`
//! FFI symbol on the same input bytes. The forward result is BF16,
//! same shape as `x`.
//!
//! Backward: `grad_x` bit-exact with
//! [`crate::fused_rmsnorm_backward`] (same FFI symbol
//! `kiln_fused_rmsnorm_bwd`, no cross-row reduction). `grad_w` is
//! within one BF16 ULP of the candle path because the kernel's
//! `atomicAdd` cross-row reduction is order-non-deterministic across
//! separate launches — same caveat the existing
//! `parity_backward_multi_row_cuda` test absorbs with `tol=2e-2`.

// `kiln-rmsnorm-kernel` is always built with CUDA (the crate's
// `Cargo.toml` pins `candle-core` / `kiln-tensor` to the `cuda`
// feature and there is no non-CUDA build of this crate). No
// `feature = "cuda"` gate is needed here; the OPD migration's
// equivalent gate is there because `kiln-opd-loss-kernel` does
// expose an opt-in `cuda` feature on its own Cargo.toml.
use candle_core::{Context, DType, Device, Result, Tensor};

/// Read the `KILN_DISABLE_RMSNORM_KT_FORWARD_OP` kill switch. When set
/// (`1` / `true` / `yes` / `TRUE`), the production caller falls back
/// to the candle CustomOp2 path
/// ([`crate::fused_rmsnorm_with_autograd`]). Same convention as
/// `KILN_DISABLE_OPD_KT_FORWARD_OP` (commit `f214f168`),
/// `KILN_DISABLE_RMSNORM_BWD_KT_BRIDGE` (commit `341da876`),
/// `KILN_DISABLE_RMSNORM_KERNEL`, `KILN_DISABLE_FUSED_CONV1D`, etc.
pub fn kt_forward_op_disabled() -> bool {
    std::env::var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// Returns `true` when `(x, weight)` is in the kt-typed forward+backward
/// envelope: CUDA, both BF16, contiguous, rank >= 1, weight shape matches
/// `x`'s last dim, last dim <= 8192. This is the same envelope as
/// [`crate::supports`] for the CUDA fused forward kernel.
fn shim_envelope_ok(x: &Tensor, weight: &Tensor) -> bool {
    if !matches!(x.device(), Device::Cuda(_)) {
        return false;
    }
    if !matches!(weight.device(), Device::Cuda(_)) {
        return false;
    }
    if x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return false;
    }
    if !x.is_contiguous() || !weight.is_contiguous() {
        return false;
    }
    if x.rank() < 1 {
        return false;
    }
    let hidden = x.dims().last().copied().unwrap_or(0);
    if hidden == 0 || hidden > 8192 {
        return false;
    }
    if weight.dims() != [hidden] {
        return false;
    }
    true
}

/// kt-shim fused RMSNorm forward+backward with candle-autograd integration.
///
/// Behavioral envelope:
/// - CUDA + BF16 + contiguous + `hidden <= 8192` → routes through
///   [`KtForwardOp2`] over the kt-typed fused forward+backward.
/// - Anything outside the envelope (CPU, non-bf16, etc.) → falls
///   through to [`crate::fused_rmsnorm_with_autograd`] (the
///   candle-typed CustomOp2 wrapper). The autograd chain through
///   `.backward()` is preserved in either case.
///
/// The signature mirrors [`crate::fused_rmsnorm_with_autograd`] so
/// the kiln-model call site is a one-line swap.
///
/// [`KtForwardOp2`]: kiln_kt_bridge::forward_op::KtForwardOp2
pub fn fused_rmsnorm_via_kt_forward_op(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    // Kill switch + out-of-envelope fallback: identical to what the
    // kiln-model caller used before this migration.
    if kt_forward_op_disabled() || !shim_envelope_ok(x, weight) {
        return crate::fused_rmsnorm_with_autograd(x, weight, eps);
    }

    cuda_via_kt_forward_op(x, weight, eps)
}

// ---------------------------------------------------------------------------
// CUDA fast path: KtForwardOp2 over kt-typed forward + backward.
// ---------------------------------------------------------------------------

fn cuda_via_kt_forward_op(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use crate::kt_api::{fused_rmsnorm_backward_kt, fused_rmsnorm_kt};
    use kiln_kt_bridge::forward_op::KtForwardOp2;
    use kiln_kt_bridge::{
        kt_tensor_from_candle_cuda_borrow, kt_tensor_to_candle_cuda_copy,
    };
    use std::sync::Arc;

    // ----- Force-contiguous inputs ---------------------------------------
    //
    // `apply_op2` passes the input layouts through to the CustomOp's
    // `cuda_fwd` hook; the kt-bridge borrow path requires contiguous
    // storage. The kernel itself also requires contiguous (the FFI
    // assumes row-major linear layout). The OPD shim and the existing
    // `RmsNormCustomOp::cuda_fwd` both apply the same defensive
    // contiguous().
    let x_contig = x
        .contiguous()
        .context("force-contiguous x for rmsnorm kt-shim")?;
    let w_contig = weight
        .contiguous()
        .context("force-contiguous weight for rmsnorm kt-shim")?;

    // ----- Forward closure -----------------------------------------------
    //
    // Calls the kt-typed `fused_rmsnorm_kt` directly. Unlike the OPD
    // forward closure (which has an axis-N gather substrate gap and
    // re-runs the candle composite as a leaf op), the rmsnorm
    // kt-typed forward has no substrate gap: it bottoms out in the
    // same `kiln_fused_rmsnorm` FFI symbol that the candle-typed
    // `fused_rmsnorm` calls, just routed through kt borrows instead
    // of candle's storage_and_layout / as_cuda_slice path. Bit-exact
    // with the candle entry by construction.
    let forward = move |x_in: &Tensor, w_in: &Tensor| -> Result<Tensor> {
        let x_kt = kt_tensor_from_candle_cuda_borrow(x_in)
            .map_err(|e| candle_core::Error::Msg(format!("rmsnorm kt-shim fwd: borrow x: {e}")))?;
        let w_kt = kt_tensor_from_candle_cuda_borrow(w_in)
            .map_err(|e| candle_core::Error::Msg(format!("rmsnorm kt-shim fwd: borrow w: {e}")))?;
        let y_kt = fused_rmsnorm_kt(&x_kt, &w_kt, eps)
            .map_err(|e| candle_core::Error::Msg(format!("rmsnorm kt-shim fwd: kt call: {e}")))?;
        kt_tensor_to_candle_cuda_copy(&y_kt).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim fwd: copy-back y: {e}"))
        })
    };

    // ----- Backward closure ----------------------------------------------
    //
    // Mirrors `fused_rmsnorm_backward_via_kt_bridge` (in this crate's
    // `lib.rs`), which the `RmsNormCustomOp::bwd` body has used since
    // commit `341da876`. The kt path returns
    // `(grad_x: BF16, grad_w_partial: F32 [rows, hidden])`; the
    // kernel writes via atomicAdd into the first `hidden` F32 slots
    // only, so we cast exactly those `hidden` F32 slots to BF16 via
    // the same `kiln_f32_to_bf16` call the existing migration uses.
    //
    // The shim passes us (`arg1=x, arg2=weight, res=y, grad_res=grad_y`);
    // we ignore `res` since the backward doesn't depend on the
    // forward output value (the kernel recomputes `rms_inv` from `x`).
    let backward = move |arg_x: &Tensor,
                         arg_w: &Tensor,
                         _res: &Tensor,
                         grad_res: &Tensor|
          -> Result<(Option<Tensor>, Option<Tensor>)> {
        let x_c = arg_x.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: contiguous x: {e}"))
        })?;
        let w_c = arg_w.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: contiguous weight: {e}"))
        })?;
        let g_c = grad_res.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: contiguous grad_out: {e}"))
        })?;

        let x_dims = x_c.dims();
        let hidden = *x_dims.last().ok_or_else(|| {
            candle_core::Error::Msg(
                "rmsnorm kt-shim bwd: x must have rank >= 1".to_string(),
            )
        })?;

        let x_kt = kt_tensor_from_candle_cuda_borrow(&x_c).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: borrow x: {e}"))
        })?;
        let w_kt = kt_tensor_from_candle_cuda_borrow(&w_c).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: borrow weight: {e}"))
        })?;
        let g_kt = kt_tensor_from_candle_cuda_borrow(&g_c).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: borrow grad_out: {e}"))
        })?;

        let (grad_x_kt, grad_w_partial_kt) =
            fused_rmsnorm_backward_kt(&x_kt, &w_kt, &g_kt, eps).map_err(|e| {
                candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: kt call: {e}"))
            })?;

        // Cast the populated `hidden` prefix of grad_w_partial (F32) to
        // BF16. Identical to the `fused_rmsnorm_backward_via_kt_bridge`
        // step in `lib.rs` — see its docstring for the
        // `csrc/fused_rmsnorm_bwd.cu` lines 12-19/122-123 reference
        // explaining why only the first `hidden` slots are populated.
        let grad_w_kt = {
            use kiln_tensor::{DType as KtDType, Tensor as KtTensor};
            let partial_ptr = kiln_kt_bridge::cuda_output_device_ptr(&grad_w_partial_kt);
            let partial_st = kiln_kt_bridge::cuda_storage_of_output(&grad_w_partial_kt);
            let raw_stream = partial_st.cuda_stream_raw();
            let dst_kt: KtTensor =
                kiln_kt_bridge::alloc_cuda_tensor(partial_st, KtDType::BF16, vec![hidden])
                    .map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "rmsnorm kt-shim bwd: alloc grad_w BF16: {e}"
                        ))
                    })?;
            let dst_ptr = kiln_kt_bridge::cuda_output_device_ptr(&dst_kt);
            // SAFETY: `partial_ptr` points to a F32 buffer of at least
            // `rows*hidden` elements (kt allocation); we read only the
            // first `hidden`. `dst_ptr` points to a BF16 buffer of
            // exactly `hidden` elements (just allocated above).
            let status = unsafe {
                crate::kiln_f32_to_bf16(
                    partial_ptr as *const f32,
                    dst_ptr as *mut _,
                    hidden as i32,
                    raw_stream,
                )
            };
            if status != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "rmsnorm kt-shim bwd: kiln_f32_to_bf16 failed (status {status})"
                )));
            }
            dst_kt
        };

        let gx = kt_tensor_to_candle_cuda_copy(&grad_x_kt).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: copy-back grad_x: {e}"))
        })?;
        let gw = kt_tensor_to_candle_cuda_copy(&grad_w_kt).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: copy-back grad_w: {e}"))
        })?;

        Ok((Some(gx), Some(gw)))
    };

    // ----- Apply ----------------------------------------------------------
    let op = KtForwardOp2::new("kiln-rmsnorm-kt-forward-op", forward, backward);
    x_contig
        .apply_op2_arc(&w_contig, Arc::new(Box::new(op)))
        .context("apply rmsnorm kt-forward-op to (x, weight)")
}

// ---------------------------------------------------------------------------
// Unit tests (cfg-test only). End-to-end CUDA parity lives in the main
// `tests` module in `lib.rs`, in the `cuda_kt_forward_op_parity` sub-
// module added alongside this commit.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kill_switch_default_off() {
        let prior = std::env::var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP").ok();
        // SAFETY: env modification is intra-test; this test runs in
        // its own process under `cargo test`. We restore the prior
        // value at the end. Other tests in this binary don't read
        // the same var.
        unsafe {
            std::env::remove_var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP", "0");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP", "false");
        }
        assert!(!kt_forward_op_disabled());

        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP"),
            }
        }
    }

    #[test]
    fn kill_switch_on() {
        let prior = std::env::var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP").ok();
        for v in ["1", "true", "yes", "TRUE", "Yes"] {
            unsafe {
                std::env::set_var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP", v);
            }
            assert!(
                kt_forward_op_disabled(),
                "expected disabled for env={v}"
            );
        }
        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_RMSNORM_KT_FORWARD_OP"),
            }
        }
    }
}

//! Production-caller migration to `KtForwardOp` for FLCE phase-B
//! ((#1082) — see `docs/CANDLE_REMOVAL_PLAN.md`).
//!
//! # What this module wires together
//!
//! The kiln-train trainer (`crates/kiln-train/src/trainer.rs` —
//! multiple call sites via [`crate::fused_linear_cross_entropy_dispatch`]
//! / [`crate::fused_linear_cross_entropy_dispatch_with_provider`])
//! previously routed all FLCE calls through either
//! [`crate::fused_linear_cross_entropy`] (Phase A — pure-candle
//! reference) or [`crate::fused_linear_cross_entropy_phase_b`] (Phase
//! B — candle `CustomOp1`). Phase B's `bwd()` already routes through
//! the kt bridge for CUDA (commit `ab2da23f`), so the backward is
//! kt-typed today.
//!
//! [`fused_linear_cross_entropy_phase_b_via_kt_forward_op`] replaces
//! the Phase-A and Phase-B candle composites with a **single** candle
//! `CustomOp1` — [`kiln_kt_bridge::forward_op::KtForwardOp1`] (commit
//! `095f1c74`) — whose forward closure runs the same chunked Phase B
//! forward as a leaf operation and whose backward closure calls the
//! kt-typed CUDA backward
//! ([`crate::kt_api::fused_linear_cross_entropy_phase_b_backward_kt`]).
//! The graph collapses from however many candle ops the caller stacked
//! on top of FLCE down to **one** custom-op node per FLCE call.
//!
//! Why this is the same template OPD used in commit `f214f168`:
//!
//!   1. Forward closure → leaf-only candle composite (no upstream
//!      autograd parent inside the closure).
//!   2. Backward closure → kt-typed kernel (the actual fused CUDA
//!      win).
//!   3. Envelope check + kill switch + fallback to the existing
//!      Phase-B `CustomOp1` path on out-of-envelope inputs.
//!
//! # Why the forward closure uses the candle Phase B body, not kt ops
//!
//! The kt-typed forward
//! ([`crate::kt_api::fused_linear_cross_entropy_phase_b_kt`]) was
//! validated CPU-only as of the Phase B substrate work (commits
//! `15d6c91a`, `bcdce7d1`, `f639f268`, `92256f71`, `ab2da23f`). It
//! relies on `kiln-tensor` ops (`index_select`, `narrow`, chunked
//! `matmul` + `log_sum_exp`) — several of which still have CUDA gaps
//! that surface only at the production trainer's full
//! `[1, T=8192, H=2560]` × `[H=2560, V=151936]` shape. Until those
//! kt-CUDA substrate gaps close (separate substrate PRs), the
//! forward closure here runs the candle Phase B reference
//! [`crate::fused_linear_cross_entropy_phase_b`] body on the leaf
//! candle tensor handed to it by [`KtForwardOp1::cuda_fwd`]. This
//! still buys the candle autograd-graph collapse (1 node per FLCE
//! call instead of the upstream caller's tree of ops feeding into
//! FLCE) — and once the kt substrate matures, swapping the forward
//! closure to the kt entry is a localised change here.
//!
//! Mirrors the same "forward via candle, backward via kt" choice the
//! OPD `kt_forward_op` migration made in commit `f214f168`.
//!
//! # Envelope and fallback
//!
//! The kt-typed backward
//! ([`crate::kt_api::fused_linear_cross_entropy_phase_b_backward_kt`])
//! supports `hidden.dtype() in {F32, BF16}` and the same
//! `[1, seq_len, hidden_size]` / `[hidden_size, vocab_size]` shape
//! contract as the candle forward. When ANY of the following hold we
//! fall back to the candle Phase-B `CustomOp1` path so the production
//! caller stays correct for the full input envelope it supports
//! today:
//!
//! - `KILN_DISABLE_FLCE_KT_FORWARD_OP=1` (kill switch — same
//!   convention as `KILN_DISABLE_FLCE_BWD_KT_BRIDGE` (commit
//!   `ab2da23f`), `KILN_DISABLE_OPD_KT_FORWARD_OP` (commit
//!   `f214f168`), `KILN_DISABLE_RMSNORM_KERNEL`, etc.).
//! - `hidden` is not on CUDA (the shim is CUDA-only by construction).
//! - `hidden.dtype()` is not F32 or BF16.
//! - `hidden.dtype() != head_t.dtype()`.
//! - `active_count == 0` (no rows to compute; matches the zero-scalar
//!   short-circuit in Phase A / Phase B).
//! - `seq_len < 2` (no targets to predict).
//! - The caller bound an [`crate::FlceProvider`] (the shim has no
//!   provider hook; the candle path's provider escape is the parity
//!   oracle for provider-bound chunk matmuls. Same rationale as
//!   Phase B's kt-bridge bwd dispatch in `phase_b.rs`).
//!
//! Falling back preserves the upstream-trainer autograd chain (the
//! candle Phase-B `CustomOp1` is also an autograd-leaf composite so
//! the trainer sees the same single-node graph either way — the
//! shim's win is the kt-typed forward enablement once the substrate
//! is ready, plus a marginally cleaner kt-bridge backward path).
//!
//! # Why this lives in `kiln-flce-kernel` and not `kiln-train`
//!
//! Three reasons (identical to the OPD migration's rationale in
//! commit `f214f168`):
//!
//! 1. The kt-typed forward + backward entries
//!    ([`crate::kt_api::fused_linear_cross_entropy_phase_b_kt`],
//!    [`crate::kt_api::fused_linear_cross_entropy_phase_b_backward_kt`])
//!    are crate-internal here.
//! 2. The kill switch + envelope check are kernel-policy concerns;
//!    pushing them into the trainer would duplicate them per call
//!    site (currently four — see [`fused_linear_cross_entropy_phase_b_via_kt_forward_op`]'s
//!    swap pattern in `trainer.rs`).
//! 3. The trainer crate (`kiln-train`) doesn't depend on
//!    `kiln-kt-bridge`. Keeping the shim plumbing here means the
//!    trainer's `Cargo.toml` doesn't need to learn about the bridge.
//!
//! The call-site change in `kiln-train` is then a one-line swap:
//! `fused_linear_cross_entropy_dispatch(...)` →
//! `fused_linear_cross_entropy_phase_b_via_kt_forward_op(...)`.
//!
//! # Numerical contract
//!
//! Up to floating-point associativity in the chunked sum-exp
//! reduction, the forward output matches the candle Phase B path to
//! ≤ 1e-5 in F32 and ≤ 1e-2 in BF16 (the bf16 bound absorbs rounding
//! in the chunked matmul). The backward dhidden matches the candle
//! `backward_dhidden` to the same tolerances — it's the same kt FFI
//! path the existing `cuda_kt_bridge_bwd_parity_{f32,bf16}` tests in
//! `phase_b.rs` use.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::{FlceProvider, fused_linear_cross_entropy_phase_b_with_provider};

/// Read the `KILN_DISABLE_FLCE_KT_FORWARD_OP` kill switch. When set
/// (`1` / `true` / `yes` / `TRUE`), the production caller falls back
/// to the candle Phase-B `CustomOp1` path. Same convention as
/// `KILN_DISABLE_FLCE_BWD_KT_BRIDGE` (commit `ab2da23f`),
/// `KILN_DISABLE_OPD_KT_FORWARD_OP` (commit `f214f168`),
/// `KILN_DISABLE_RMSNORM_KERNEL`, `KILN_DISABLE_FUSED_CONV1D`, etc.
pub fn kt_forward_op_disabled() -> bool {
    std::env::var("KILN_DISABLE_FLCE_KT_FORWARD_OP")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// Returns `true` when the `(dtype, head_t.dtype())` triple is in
/// the fused kt-bwd envelope AND `hidden` is on CUDA. The envelope
/// matches the constraints
/// [`crate::kt_api::fused_linear_cross_entropy_phase_b_backward_kt`]
/// enforces internally and the Phase B kt-bridge bwd dispatch in
/// `phase_b.rs`.
fn shim_envelope_ok(hidden: &Tensor, head_t: &Tensor) -> bool {
    if !matches!(hidden.device(), Device::Cuda(_)) {
        return false;
    }
    let h_dt = hidden.dtype();
    let dtype_ok = matches!(h_dt, DType::F32 | DType::BF16);
    if !dtype_ok {
        return false;
    }
    if h_dt != head_t.dtype() {
        return false;
    }
    true
}

/// kt-shim FLCE phase-B with candle-autograd integration.
///
/// Behavioral envelope:
/// - CUDA + `dtype in {F32, BF16}` + matching `head_t` dtype + no
///   `FlceProvider` bound + non-empty active rows → routes through
///   [`KtForwardOp1`] over the kt-typed forward+backward (the
///   forward closure currently uses the candle Phase-B body — see
///   module docs).
/// - Anything outside the envelope → falls through to
///   [`crate::fused_linear_cross_entropy_phase_b_with_provider`] (the
///   candle `CustomOp1` reference path). The autograd chain through
///   `loss.backward()` is preserved in either case (both paths return
///   a `CustomOp1` candle autograd node parented on `hidden`).
///
/// The signature mirrors
/// [`crate::fused_linear_cross_entropy_phase_b_with_provider`] so the
/// kiln-train call site is a one-line swap.
///
/// [`KtForwardOp1`]: kiln_kt_bridge::forward_op::KtForwardOp1
pub fn fused_linear_cross_entropy_phase_b_via_kt_forward_op(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
    provider: Option<FlceProvider>,
) -> Result<Tensor> {
    // The provider hook is explicit Phase-B state; the shim has no
    // provider plumbing today (the kt-typed backward isn't provider-
    // aware either). When a provider is bound, defer to the Phase-B
    // candle path so the trainer's Vulkan FLCE escape stays intact.
    if provider.is_some() {
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, provider,
        );
    }

    // Kill switch + non-CUDA + out-of-envelope fallback path:
    // identical to the candle Phase-B entry the trainer used before
    // this migration.
    if kt_forward_op_disabled() || !shim_envelope_ok(hidden, head_t) {
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        );
    }

    // Short-circuits the candle Phase-B body already handles
    // internally — but going through `apply_op1` on a hidden that
    // contributes nothing to the loss attaches a dead branch to the
    // autograd graph. Cheap and safer to defer to Phase B's zero-
    // scalar short-circuits explicitly here.
    let seq_len = input_ids.len();
    if seq_len < 2 {
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        );
    }
    if label_mask.len() != seq_len {
        // Defer error reporting to the candle path so we have a
        // single source of truth for the error message format.
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        );
    }
    let active_count = label_mask[1..].iter().filter(|&&m| m).count();
    if active_count == 0 {
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        );
    }

    #[cfg(feature = "cuda")]
    {
        return cuda_via_kt_forward_op(
            hidden,
            head_t,
            input_ids,
            label_mask,
            device,
            chunk_size,
        );
    }

    // Non-cuda builds: shim_envelope_ok() returned true above only
    // if hidden.device() is CUDA — but candle without the `cuda`
    // feature can't have a CUDA device, so we'll never reach here.
    // Leave a belt-and-suspenders fallback so the function is still
    // well-typed.
    #[cfg(not(feature = "cuda"))]
    {
        fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        )
    }
}

// ---------------------------------------------------------------------------
// CUDA fast path: KtForwardOp1 over candle Phase-B forward + kt-typed
// backward. The forward currently uses the candle Phase-B body (see
// module docs for the substrate gap); the backward uses the kt FFI.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn cuda_via_kt_forward_op(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
) -> Result<Tensor> {
    use crate::kt_api::fused_linear_cross_entropy_phase_b_backward_kt;
    use anyhow::Context;
    use kiln_kt_bridge::forward_op::KtForwardOp1;
    use kiln_kt_bridge::{
        kt_tensor_from_candle_cuda_borrow, kt_tensor_to_candle_cuda_copy,
    };
    use std::sync::Arc;

    // Force-contiguous `hidden` before the bridge borrow. Matches
    // `phase_b::fused_linear_cross_entropy_phase_b_with_provider`'s
    // defensive contiguous().
    let hidden_contig = hidden
        .contiguous()
        .context("force-contiguous hidden for FLCE kt-shim")?;

    // Captured shim state. `head_t` is the transposed lm_head (frozen
    // during LoRA training, captured by value). `input_ids` /
    // `label_mask` / `chunk_size` are by-value clones so the shim
    // outlives the call frame — `KtForwardOp1` is held by
    // `Arc<Box<dyn CustomOp1>>` inside candle's autograd graph until
    // `loss.backward()` runs (the parents drop the op when the graph
    // is collected).
    let head_t_owned_fwd = head_t.clone();
    let head_t_owned_bwd = head_t.clone();
    let input_ids_fwd = input_ids.to_vec();
    let input_ids_bwd = input_ids.to_vec();
    let label_mask_fwd = label_mask.to_vec();
    let label_mask_bwd = label_mask.to_vec();
    let device_fwd = device.clone();

    // ----- Forward closure ------------------------------------------------
    //
    // Computes the chunked FLCE forward via the candle Phase-B body
    // [`crate::fused_linear_cross_entropy_phase_b_with_provider`] on
    // the leaf candle tensor handed in by
    // [`KtForwardOp1::cuda_fwd`]. The kt-typed forward
    // (`crate::kt_api::fused_linear_cross_entropy_phase_b_kt`) is
    // the semantic target here, but the kt CUDA substrate has gaps
    // at production-trainer shapes (see module docs); routing the
    // candle composite through the closure still buys the autograd-
    // graph collapse, and the backward below runs through the fused
    // kt CUDA path, which is the actual perf-and-memory win for the
    // FLCE trainer. Once the kt substrate matures at trainer scale,
    // this closure can swap to the kt entry.
    let forward = move |hidden_in: &Tensor| -> candle_core::Result<Tensor> {
        let loss = fused_linear_cross_entropy_phase_b_with_provider(
            hidden_in,
            &head_t_owned_fwd,
            &input_ids_fwd,
            &label_mask_fwd,
            &device_fwd,
            chunk_size,
            None,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim fwd: phase-B candle composite: {e}"
            ))
        })?;
        // Phase B returns a scalar F32 on `device`; force-contiguous
        // before the shim unwraps storage (the candle scalar should
        // be contiguous by construction, but be defensive).
        loss.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim fwd: contiguous loss scalar: {e}"
            ))
        })
    };

    // ----- Backward closure ----------------------------------------------
    //
    // The kt bwd takes `grad_loss` shape `[]` (scalar F32 contiguous
    // on CUDA) and returns `dhidden` shape `[1, seq_len, hidden_size]`
    // in the input dtype. The kt entry handles the chunked recompute
    // of `softmax * head_t.T` and the scatter back into a full-sized
    // `dhidden` zero buffer.
    //
    // The shim passes us (`arg=hidden, res=loss_scalar,
    // grad_res=grad_loss`); we ignore `res` since the backward
    // doesn't depend on the forward output value.
    let backward = move |arg: &Tensor,
                         _res: &Tensor,
                         grad_res: &Tensor|
          -> candle_core::Result<Option<Tensor>> {
        let hidden_c = arg.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: contiguous hidden: {e}"
            ))
        })?;
        let head_t_c = head_t_owned_bwd.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: contiguous head_t: {e}"
            ))
        })?;
        // `grad_res` is the upstream scalar gradient on the loss
        // output (typically 1.0). The kt entry casts and reshapes
        // internally but requires F32 + contiguous on CUDA storage.
        let grad_res_f32 = grad_res.to_dtype(DType::F32).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: cast grad_loss to F32: {e}"
            ))
        })?;
        let grad_res_c = grad_res_f32.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: contiguous grad_loss: {e}"
            ))
        })?;

        let hidden_kt = kt_tensor_from_candle_cuda_borrow(&hidden_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: borrow hidden: {e}"
            ))
        })?;
        let head_t_kt = kt_tensor_from_candle_cuda_borrow(&head_t_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: borrow head_t: {e}"
            ))
        })?;
        let grad_res_kt = kt_tensor_from_candle_cuda_borrow(&grad_res_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: borrow grad_loss: {e}"
            ))
        })?;

        let d_hidden_kt = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden_kt,
            &head_t_kt,
            &input_ids_bwd,
            &label_mask_bwd,
            chunk_size,
            &grad_res_kt,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: kt bwd call: {e}"
            ))
        })?;

        // The kt `scatter_add` produces a contiguous `[1, seq_len,
        // hidden_size]` output but be defensive — the bridge copy-
        // back requires contiguous storage on the source side.
        let d_hidden_kt_c = if d_hidden_kt.is_contiguous() {
            d_hidden_kt
        } else {
            d_hidden_kt.contiguous().map_err(|e| {
                candle_core::Error::Msg(format!(
                    "flce kt-shim bwd: contiguous d_hidden: {e}"
                ))
            })?
        };

        let d_hidden = kt_tensor_to_candle_cuda_copy(&d_hidden_kt_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: copy-back d_hidden: {e}"
            ))
        })?;

        Ok(Some(d_hidden))
    };

    // ----- Apply ----------------------------------------------------------
    let op = KtForwardOp1::new("kiln-flce-kt-forward-op", forward, backward);
    let _ = device; // unused; the device is implicit in `hidden_contig`.
    hidden_contig
        .apply_op1_arc(Arc::new(Box::new(op)))
        .context("apply FLCE kt-forward-op to hidden")
}

// ---------------------------------------------------------------------------
// Unit tests for the kill switch. End-to-end CUDA parity lives in
// `kiln-flce-kernel/src/tests.rs` next to the existing flce tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate `KILN_DISABLE_FLCE_KT_FORWARD_OP`
    // so they don't race against each other. Cargo runs tests in
    // threads within a single process; without a lock,
    // `kill_switch_on` can set the env to "1" in the middle of
    // `kill_switch_default_off`'s `set_var("0")` +
    // `assert!(!kt_forward_op_disabled())` sequence, causing a flaky
    // failure. The same pattern fix was applied to
    // `kiln-opd-loss-kernel` in commit `73109cbe`; this commit
    // applies the same shape to `kiln-flce-kernel` after observing
    // the flake on CI run 26574406306, commit `e82c3017` (#1082).
    //
    // Poison recovery is built in via `clear_poison()` since the
    // only shared state is the env itself, which each test re-
    // establishes from a known starting condition before asserting.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|poisoned| {
            // A previous test panicked while holding the lock.
            // Clear the poison and proceed — each test below re-
            // establishes its own starting env state before
            // asserting.
            ENV_LOCK.clear_poison();
            poisoned.into_inner()
        })
    }

    #[test]
    fn kill_switch_default_off() {
        let _guard = env_lock();
        let prior = std::env::var("KILN_DISABLE_FLCE_KT_FORWARD_OP").ok();
        // SAFETY: `kt_forward_op_disabled()` reads the env on each
        // call (no caching), so the toggle is reversible per-test.
        // The ENV_LOCK guard above ensures no other test in this
        // binary is concurrently mutating the same var.
        unsafe {
            std::env::remove_var("KILN_DISABLE_FLCE_KT_FORWARD_OP");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", "0");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", "false");
        }
        assert!(!kt_forward_op_disabled());

        // Restore the prior value.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_FLCE_KT_FORWARD_OP"),
            }
        }
    }

    #[test]
    fn kill_switch_on() {
        let _guard = env_lock();
        let prior = std::env::var("KILN_DISABLE_FLCE_KT_FORWARD_OP").ok();
        for v in ["1", "true", "yes", "TRUE", "Yes"] {
            unsafe {
                std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", v);
            }
            assert!(
                kt_forward_op_disabled(),
                "expected disabled for env={v}"
            );
        }
        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_FLCE_KT_FORWARD_OP"),
            }
        }
    }
}

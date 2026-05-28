//! Experimental tape-forward path — CP-4 production-caller scaffolding (#1082).
//!
//! This module wires a `KILN_USE_TAPE_FORWARD`-gated branch into
//! `forward.rs` that routes a subset of forward-pass sites through the
//! `kiln_autograd::Tape` substrate (recording onto a thread-local
//! `Tape` instead of building a candle `BackpropOp` graph).
//!
//! # Why this exists
//!
//! The audit in
//! [`docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`]
//! documented why a per-call-site flip of `rms_norm` to
//! `fused_rmsnorm_via_kt_tape` cannot land at HEAD: the production
//! caller signature
//! `rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor>`
//! has no `&mut Tape` in scope and no caller transitively up to the
//! training step root has one either.
//!
//! The CP-4 substrate work in `kiln-train` (wave 11, commits
//! `51643ab` + `de647b8`) added the kt-tape primitives the production
//! forward will eventually need:
//!
//! * `tape_step::rms_norm_via_tape`
//! * `tape_step::matmul_via_tape`
//! * `tape_step::silu_via_tape`
//! * `tape_step::cross_entropy_via_tape`
//! * `tape_step::transformer_block_step_via_tape`
//!
//! …all parameterised on `&mut Tape`. To exercise the substrate end-to-end
//! from `kiln-model`'s forward path, this module provides a
//! **thread-local Tape** so existing forward functions can route through
//! tape-aware primitives without rewriting every signature up to the
//! training-step root.
//!
//! # Design
//!
//! 1. [`tape_forward_enabled`] reads `KILN_USE_TAPE_FORWARD` once and
//!    caches the result. Anything other than `0` / `false` / `no` /
//!    empty enables the path.
//! 2. [`with_thread_local_tape`] runs a closure with a fresh `Tape` as
//!    a thread-local. Tape-aware primitives can fetch the active tape
//!    via [`with_active_tape`] and record onto it.
//! 3. [`try_tape_rms_norm_cuda`] is the per-call-site adapter: it
//!    checks the env flag, borrows kt-tensors via `kiln_kt_bridge`,
//!    runs `fused_rmsnorm_via_kt_tape` with the active tape, copies
//!    the output back to a candle Tensor, and returns. When the gate
//!    is off, the env-tristate returns `None` and the caller falls
//!    through to the existing kt-forward-op shim.
//!
//! # Production-safety
//!
//! This is opt-in only. With `KILN_USE_TAPE_FORWARD` unset (the
//! default), `tape_forward_enabled()` returns `false` and every
//! `try_tape_*` helper short-circuits to `Ok(None)`. The existing
//! `fused_rmsnorm_via_kt_forward_op` path is untouched; the production
//! decode and training loops route through it exactly as before.
//!
//! # What this proves
//!
//! With `KILN_USE_TAPE_FORWARD=1`:
//!
//! * The forward path *can* be made to drive a `Tape`. The tape-
//!   recorded `RmsNormBackward` node is visible to a subsequent
//!   `Tape::backward` walk.
//! * The output tensor is bit-exact with the kt-forward-op shim
//!   (same `kiln_fused_rmsnorm` FFI call underneath; only the
//!   backward-graph machinery differs).
//! * No candle autograd lineage is attached. Callers downstream of
//!   the tape-routed `rms_norm` cannot drive backward through
//!   `loss.backward()` for that op — they must use `Tape::backward`.
//!   This is exactly the trade-off CP-4 is meant to expose: the two
//!   autograd worlds are mutually-incompatible until CP-4's substrate
//!   ports the *entire* training step onto the tape.
//!
//! # Out of scope
//!
//! * Wiring tape-routing for matmul, silu, embedding, etc. Each is a
//!   separate per-site flip following this same pattern. The next
//!   PR extends `try_tape_*` to one more primitive at a time.
//! * Plumbing tape gradients back into the optimiser. That's the
//!   `kiln-optim` integration concern; the substrate proof here is
//!   "forward routes through Tape::record without crashing or
//!   changing numerics."

#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::Tensor;
use kiln_autograd::Tape;

// Phase 6a/CP-4 (#1082): the thread-local-tape scope machinery
// (`with_thread_local_tape`, `with_active_tape`, `tape_forward_enabled`)
// originally lived here. Wave-13 (#1082) promoted it into
// `kiln-autograd::tape_scope` so the OPD and FLCE kernel crates (and
// their `kiln-train` callers) can share the same thread-local handle
// without taking a `kiln-model` dependency. We re-export from here for
// back-compat — every existing call site (the parity test, the
// `forward.rs:7178` adapter call) keeps compiling unchanged.
pub use kiln_autograd::{tape_forward_enabled, with_active_tape, with_thread_local_tape};

/// Attempt to run RMSNorm through the kt-tape pilot
/// (`fused_rmsnorm_via_kt_tape`) instead of the kt-forward-op shim.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned
///   `Tensor` is a copy of the kt-tape output into a candle CUDA
///   tensor; the backward node was recorded on the active
///   thread-local tape.
/// * `Ok(None)` — the gate was off, no thread-local tape is active,
///   the kt envelope rejected the inputs, or the kt-borrow failed.
///   The caller must fall through to the existing dispatch.
/// * `Err(...)` — a kt-tape forward error (e.g. envelope-OK but FFI
///   call failed). Propagated so callers see the failure cleanly
///   instead of silently masking it.
///
/// The candle-typed `(x, weight)` inputs are borrowed (zero-copy via
/// `kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow`), routed
/// through the kt-tape entry, and the kt output is copied back into a
/// candle CUDA tensor (no autograd link — see module docs). For the
/// experimental gating this is the correct trade-off: the recipient
/// of the returned tensor knows the gradient lives on the tape, not
/// on candle's `BackpropOp`.
pub fn try_tape_rms_norm_cuda(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // kt borrow: zero-copy view of the candle CUDA tensors as kt
    // tensors. Returns `Err` (which we treat as "skip") on layout /
    // dtype / device mismatch.
    let x_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(x) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let w_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weight) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    if !kiln_rmsnorm_kernel::supports_rmsnorm_kt(&x_kt, &w_kt) {
        return Ok(None);
    }

    // Borrow the active tape. If no scope is open, we have nowhere to
    // record — fall through.
    let out_kt = match with_active_tape(|tape: &mut Tape| {
        kiln_rmsnorm_kernel::fused_rmsnorm_via_kt_tape(&x_kt, &w_kt, eps, tape)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt
        .map_err(|e| anyhow::anyhow!("fused_rmsnorm_via_kt_tape: {e}"))
        .context("tape_forward::try_tape_rms_norm_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_rms_norm_cuda: kt -> candle copy failed")?;

    Ok(Some(out))
}

// Tape-scope tests live in `kiln-autograd::tape_scope::tests` after
// wave-13 (#1082) promoted the thread-local-tape machinery there. The
// kt-tape adapter test (`try_tape_rms_norm_cuda` round-trip) lives in
// the `kiln-model/tests/tape_forward_parity.rs` integration test
// because it requires the `kiln_kt_bridge` + `kiln_rmsnorm_kernel`
// cuda surface.

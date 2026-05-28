//! Wave-13 (#1082) — OPD kt-tape production-caller adapter.
//!
//! Mirrors the rmsnorm sibling
//! `kiln-model::tape_forward::try_tape_rms_norm_cuda`. The adapter
//! takes the same candle-typed inputs the production caller in
//! `kiln-train::opd::opd_step_loss` already passes to
//! [`crate::opd_top_k_reverse_kl_per_position_via_kt_forward_op`],
//! checks the `KILN_USE_TAPE_FORWARD` env tristate + an active
//! thread-local [`kiln_autograd::Tape`] scope, and on both gates open
//! routes the forward through
//! [`crate::opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`]
//! (recording the backward node onto the active tape) and copies the
//! kt output back into a candle CUDA tensor.
//!
//! # Why this lives next to the existing shim (not in `kiln-model`)
//!
//! The OPD production caller is in `kiln-train`, not `kiln-model`. Both
//! crates already depend on `kiln-opd-loss-kernel`. Putting the adapter
//! here keeps the call-site change to one line at
//! `kiln-train/src/opd.rs:1247` and reuses the kt-tape entry that
//! already lives in this crate (`kt_tape.rs`, commit `5478e64f`).
//!
//! # Production-safety
//!
//! Off by default. When `KILN_USE_TAPE_FORWARD` is unset or no
//! thread-local `Tape` scope is active, [`try_tape_opd_per_position_cuda`]
//! returns `Ok(None)` and the caller falls through to the existing
//! [`crate::opd_top_k_reverse_kl_per_position_via_kt_forward_op`] shim.
//! The shim's autograd chain through candle's `loss.backward()` is
//! preserved exactly as before.
//!
//! # What this does NOT do
//!
//! It does not delete `kt_forward_op.rs`. The audit doc
//! [`docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`]
//! documents the architectural reason: the candle-autograd-based
//! training loop (`loss.backward()` over
//! `candle_core::backprop::GradStore`) cannot consume nodes recorded
//! on a `kiln_autograd::Tape`. Until CP-4 substrate work ports
//! `kiln-train`'s training step onto `Tape::backward`, the
//! kt-forward-op shim remains the production path for callers driving
//! gradients via candle.
//!
//! The wave-13 work shipped here lays the substrate hook so the
//! eventual CP-4 flip becomes a one-line change at the production
//! caller (set `KILN_USE_TAPE_FORWARD=1`, install the scope around the
//! step, observe the tape-routed node land on the tape ready for
//! `Tape::backward`).

#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::Tensor;
use kiln_autograd::{tape_forward_enabled, with_active_tape, Tape};

use crate::kt_tape::opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape;

/// Attempt to run OPD per-position top-K reverse-KL through the
/// kt-tape pilot ([`opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`])
/// instead of the candle-typed [`crate::opd_top_k_reverse_kl_per_position_via_kt_forward_op`]
/// shim.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned
///   `Tensor` is a copy of the kt-tape output into a candle CUDA
///   tensor; the backward node was recorded on the active
///   thread-local tape.
/// * `Ok(None)` — the env gate was off, no thread-local tape scope is
///   active, the kt envelope rejected the inputs, or the kt borrow
///   failed. The caller must fall through to the existing dispatch.
/// * `Err(...)` — a kt-tape forward error (e.g. envelope OK but FFI
///   call failed). Propagated so callers see the failure cleanly
///   instead of silently masking it.
///
/// # Envelope
///
/// Same as
/// [`opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`]: CUDA
/// + matching F32/BF16 `(hidden, head_t)` dtype + `top_k ∈ {16, 32}`.
/// Out-of-envelope inputs return `Ok(None)` rather than `Err` so the
/// adapter is callable from a forward path that wants to short-circuit
/// only on the happy path and fall through to the existing kt-shim on
/// everything else (matching the rmsnorm adapter's contract exactly).
pub fn try_tape_opd_per_position_cuda(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // Active-count short-circuit — match the kt-shim's behaviour. An
    // empty active set has no contribution to the loss, and recording
    // a tape node for a no-op forward is a footgun (the backward apply
    // would short-circuit on an empty `grad_loss` anyway, but the
    // shape mismatch between `[0]` and the saved `(hidden, head_t)`
    // tensor IDs would make later `Tape::backward` walks confusing).
    let active_count = label_mask.iter().filter(|&&m| m).count();
    if active_count == 0 {
        return Ok(None);
    }

    // kt borrow: zero-copy view of the candle CUDA tensors as kt
    // tensors. Returns `Err` (which we treat as "skip") on layout /
    // dtype / device mismatch.
    let hidden_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(hidden) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let head_t_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(head_t) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // Borrow the active tape. If no scope is open, we have nowhere to
    // record — fall through to the existing kt-forward-op shim.
    let out_kt = match with_active_tape(|tape: &mut Tape| {
        opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape(
            &hidden_kt,
            &head_t_kt,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            tape,
        )
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt
        .map_err(|e: kiln_tensor::Error| anyhow::anyhow!("opd_top_k kt-tape: {e}"))
        .context("try_tape_opd_per_position_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("try_tape_opd_per_position_cuda: kt -> candle copy failed")?;

    Ok(Some(out))
}

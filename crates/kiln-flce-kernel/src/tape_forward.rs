//! Wave-13 (#1082) — FLCE kt-tape production-caller adapter.
//!
//! Mirrors the rmsnorm sibling
//! `kiln-model::tape_forward::try_tape_rms_norm_cuda` and the OPD
//! sibling `kiln-opd-loss-kernel::try_tape_opd_per_position_cuda`.
//!
//! The adapter takes the same candle-typed inputs the production
//! caller in [`crate::fused_linear_cross_entropy_dispatch_with_provider`]
//! passes to
//! [`crate::fused_linear_cross_entropy_phase_b_via_kt_forward_op`],
//! checks the `KILN_USE_TAPE_FORWARD` env tristate + an active
//! thread-local [`kiln_autograd::Tape`] scope, and on both gates open
//! routes the forward through
//! [`crate::fused_linear_cross_entropy_phase_b_via_kt_tape`]
//! (recording the backward node onto the active tape) and copies the
//! kt output back into a candle CUDA tensor.
//!
//! # Why this lives next to the existing shim (not in `kiln-model`)
//!
//! Same reason as the OPD analogue: `kiln-model` does not depend on
//! `kiln-flce-kernel`, so the adapter has to live next to the shim it
//! parallels. The call-site change at
//! [`crate::fused_linear_cross_entropy_dispatch_with_provider`] is
//! then a one-line short-circuit before the shim call.
//!
//! # Production-safety
//!
//! Off by default. When `KILN_USE_TAPE_FORWARD` is unset or no
//! thread-local `Tape` scope is active, [`try_tape_flce_phase_b_cuda`]
//! returns `Ok(None)` and the caller falls through to the existing
//! [`crate::fused_linear_cross_entropy_phase_b_via_kt_forward_op`]
//! shim. The shim's autograd chain through candle's `loss.backward()`
//! is preserved exactly as before.
//!
//! # Provider plumbing
//!
//! The tape adapter does NOT support an `FlceMatmulProvider`. The
//! candle shim accepts an optional `provider` (the Vulkan FLCE escape
//! hatch from kiln-train); when a provider is bound, the dispatch
//! caller MUST route to the candle shim (or the Phase-B reference)
//! and NOT through this adapter. The dispatch caller already gates
//! on `provider.is_none()` before invoking us, but the adapter takes
//! no provider parameter as a belt-and-suspenders that this contract
//! is impossible to violate at the call site.
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

use crate::kt_tape::fused_linear_cross_entropy_phase_b_via_kt_tape;

/// Attempt to run FLCE Phase-B through the kt-tape pilot
/// ([`fused_linear_cross_entropy_phase_b_via_kt_tape`]) instead of the
/// candle-typed [`crate::fused_linear_cross_entropy_phase_b_via_kt_forward_op`]
/// shim.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned
///   `Tensor` is a copy of the kt-tape output (the scalar loss) into a
///   candle CUDA tensor; the backward node was recorded on the active
///   thread-local tape.
/// * `Ok(None)` — the env gate was off, no thread-local tape scope is
///   active, the kt envelope rejected the inputs, the active count is
///   0, or the kt borrow failed. The caller must fall through to the
///   existing dispatch.
/// * `Err(...)` — a kt-tape forward error (e.g. envelope OK but FFI
///   call failed). Propagated so callers see the failure cleanly
///   instead of silently masking it.
///
/// # Envelope
///
/// Same as [`fused_linear_cross_entropy_phase_b_via_kt_tape`]: CUDA +
/// dtype ∈ {F32, BF16} + matching `(hidden, head_t)` dtype + 3-D
/// `hidden` `[1, seq, hidden]` + 2-D `head_t` `[hidden, vocab]` +
/// active-count > 0 + seq >= 2. Out-of-envelope inputs return
/// `Ok(None)` rather than `Err` so the adapter is callable from a
/// dispatch path that wants to short-circuit only on the happy path
/// and fall through to the existing kt-shim on everything else.
pub fn try_tape_flce_phase_b_cuda(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // Active-count + seq_len short-circuit — match the kt-shim envelope.
    // FLCE's `[1, seq, H]` shape means `seq < 2` is non-sensical (no
    // next-token shift available), and an empty active set has no loss
    // contribution. The kt-tape entry's `envelope_ok` rejects these too,
    // but checking up front avoids a (small) wasted kt-borrow.
    if label_mask.len() < 2 || label_mask.iter().filter(|&&m| m).count() == 0 {
        return Ok(None);
    }
    if input_ids.len() != label_mask.len() {
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
        fused_linear_cross_entropy_phase_b_via_kt_tape(
            &hidden_kt,
            &head_t_kt,
            input_ids,
            label_mask,
            chunk_size,
            tape,
        )
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt
        .map_err(|e: kiln_tensor::Error| anyhow::anyhow!("flce kt-tape: {e}"))
        .context("try_tape_flce_phase_b_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("try_tape_flce_phase_b_cuda: kt -> candle copy failed")?;

    Ok(Some(out))
}

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

use crate::kt_tape::{
    opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape,
    opd_top_k_reverse_kl_phase_b_via_kt_tape,
};

/// Borrow a candle input as a kt tensor for the tape path, REUSING an
/// upstream adapter's kt output when this candle tensor was produced by one
/// in the active bridge scope (so the recorded kt `Tape` stays connected —
/// the consumer's input id becomes the producer's output id). Falls back to
/// a fresh zero-copy borrow otherwise (and outside any bridge scope, which is
/// the common case + every per-op parity test).
///
/// Mirrors `kiln_model::tape_forward::tape_kt_input` exactly — duplicated here
/// (rather than shared) so this kernel crate does not take a `kiln-model`
/// dependency. (#1082 CP-4 endgame, Step A.)
fn tape_kt_input(x: &Tensor) -> Option<kiln_tensor::Tensor> {
    if let Some(t) = kiln_kt_bridge::tape_bridge::kt_input_for_candle(x.id()) {
        return Some(t);
    }
    kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(x).ok()
}

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
    // tensors. `tape_kt_input` REUSES the upstream adapter's retained kt
    // output for `hidden` when it was produced by a tape adapter in the
    // active bridge scope (e.g. the final-RMSNorm `try_tape_rms_norm_cuda`
    // that produced `student_hidden`) — keeping the recorded tape CONNECTED
    // from this OPD node back through the model's LoRA chain. Falls back to a
    // fresh borrow outside a bridge scope (the common case). Returns `Err`
    // (which we treat as "skip") on layout / dtype / device mismatch.
    let hidden_kt = match tape_kt_input(hidden) {
        Some(t) => t,
        None => return Ok(None),
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

    // CP-4 (#1082) tape_bridge: register the (kt_id ↔ candle_id) IO mappings
    // + retain the output so a surrounding tape-authoritative scope can (a)
    // walk this op's grad back into the candle `hidden` input id, and (b)
    // thread this per-position output into a downstream adapter's input.
    // No-ops cleanly outside a bridge scope. `head_t` is non-differentiable
    // in this op (the kernel emits `d_hidden` only) but we still register its
    // mapping for completeness.
    kiln_kt_bridge::tape_bridge::register_input_mapping(hidden_kt.id(), hidden.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(head_t_kt.id(), head_t.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Attempt to run OPD top-K reverse-KL **reduced to a scalar mean** through
/// the kt-tape pilot ([`opd_top_k_reverse_kl_phase_b_via_kt_tape`]) and record
/// the scalar loss as a tape node ready to be the ROOT of a tape-authoritative
/// backward walk.
///
/// This is the OPD analogue of
/// `kiln_model::tape_forward::try_tape_cross_entropy_from_logits_cuda`: it
/// produces a scalar loss whose kt tensor is retained in the bridge scope so a
/// surrounding [`kiln_kt_bridge::tape_bridge::with_tape_authoritative_scope`]
/// can find it (`candle_output_kt.get(&loss.id())`), seed `dL/dL = 1`, and walk
/// the connected tape — no candle `loss.backward()`. The recorded backward is
/// `ScalarMean` mode: it expects a scalar/1-element `grad_loss` (the seed) and
/// emits `d_hidden` only (`head_t` is non-differentiable in this op).
///
/// Returns:
/// * `Ok(Some(out))` — the scalar tape path ran. The returned candle scalar
///   `Tensor` is a value-identical copy of the kt-tape loss; the backward node
///   was recorded on the active thread-local tape and the IO mappings +
///   retained output were registered for the bridge.
/// * `Ok(None)` — the env gate (`KILN_USE_TAPE_FORWARD`) was off, no
///   thread-local tape scope is active, the active set was empty, the kt
///   envelope rejected the inputs, or the kt borrow failed. The caller must
///   fall through to the existing scalar dispatch (`opd_step_loss` →
///   `mean_kl` via the kt-forward-op shim).
/// * `Err(...)` — a kt-tape forward error (envelope OK but the FFI call
///   failed). Propagated so callers see the failure cleanly.
///
/// # Envelope
///
/// Same as [`try_tape_opd_per_position_cuda`]: CUDA + matching F32/BF16
/// `(hidden, head_t)` dtype + `top_k ∈ {16, 32}`.
pub fn try_tape_opd_scalar_mean_cuda(
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

    // Active-count short-circuit — match the per-position adapter. An empty
    // active set has no loss contribution and recording a tape node for a
    // no-op forward is a footgun.
    let active_count = label_mask.iter().filter(|&&m| m).count();
    if active_count == 0 {
        return Ok(None);
    }

    // Thread `hidden` from the upstream model-chain adapter (final RMSNorm)
    // so the recorded tape is CONNECTED from this loss root back through the
    // LoRA chain. Fall through on borrow failure.
    let hidden_kt = match tape_kt_input(hidden) {
        Some(t) => t,
        None => return Ok(None),
    };
    let head_t_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(head_t) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // Record the SCALAR-mean OPD loss onto the active tape. If no scope is
    // open, fall through to the existing dispatch.
    let loss_kt = match with_active_tape(|tape: &mut Tape| {
        opd_top_k_reverse_kl_phase_b_via_kt_tape(
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

    let loss_kt = loss_kt
        .map_err(|e: kiln_tensor::Error| anyhow::anyhow!("opd_top_k scalar kt-tape: {e}"))
        .context("try_tape_opd_scalar_mean_cuda: kt-tape forward failed")?;

    // Value-identical candle copy of the scalar loss for the caller (loss_val
    // + metrics). Like the cross-entropy adapter, the returned candle tensor
    // carries NO candle autograd lineage — the gradient lives on the tape.
    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&loss_kt)
        .context("try_tape_opd_scalar_mean_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: register the (kt_id ↔ candle_id) IO mappings
    // and retain the loss output keyed on the RETURNED copy's id so
    // `with_tape_authoritative_scope` resolves `loss.id()` → `loss_kt` to seed
    // the tape root. No-ops cleanly outside a bridge scope.
    kiln_kt_bridge::tape_bridge::register_input_mapping(hidden_kt.id(), hidden.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(head_t_kt.id(), head_t.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(loss_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&loss_kt, out.id());

    Ok(Some(out))
}

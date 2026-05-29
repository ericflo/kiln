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
//!   recorded backward node is visible to a subsequent `Tape::backward`
//!   walk.
//! * The output tensor is bit-exact with the kt-forward-op shim
//!   (same kernel FFI call underneath; only the backward-graph
//!   machinery differs).
//! * Inside a `kiln_kt_bridge::tape_bridge::with_tape_scope_emit_to_grad_store`
//!   scope (see `kiln-train::trainer::standard_forward_backward_via_tape_bridge`,
//!   landed `675e0dea`), every adapter registers `(kt_id ↔ candle_id)`
//!   IO mappings via `register_input_mapping` / `register_output_mapping`.
//!   The bridge runs `loss.backward()` candle-side as usual, then walks
//!   the recorded tape with seeds derived from candle's `GradStore`, and
//!   merges the per-kt-input grads back into the same store keyed on the
//!   matched candle TensorIds. Result: callers downstream of the tape-
//!   routed primitives DO see correct grads in candle's `GradStore` even
//!   though the candle walker alone wouldn't traverse the tape op.
//!
//! # Current adapter coverage
//!
//! `try_tape_*_cuda` adapters land for every primitive whose
//! corresponding `kiln_autograd::backwards::*Backward` exists and whose
//! kt-side fused kernel ships a `*_via_kt_tape` entry:
//!
//! * `try_tape_rms_norm_cuda` — `RmsNormBackward` (CP-4 baseline)
//! * `try_tape_matmul_cuda` — `MatmulBackward`
//! * `try_tape_silu_cuda` — `SiluBackward`
//! * `try_tape_embedding_cuda` — `EmbeddingBackward`
//! * `try_tape_swiglu_cuda` — `MulSigmoidGateBackward` (MLP gate path,
//!   ~18% of decode per Phase 6 NVTX profiling)
//!
//! All 5 register IO mappings into the bridge when a scope is active
//! (commits `57f7b678` for rms_norm/matmul/silu, `cf138c9c` for swiglu).
//!
//! # Out of scope (still)
//!
//! * Tape-routing for softmax / log_softmax / cross-entropy (the loss
//!   primitive — would close the kt-tape coverage end-to-end into the
//!   loss). Substrate primitives in `kiln_autograd::backwards::cross_entropy`
//!   exist; the adapter is straightforward but distinct from this PR.
//! * Tape-routing for rotary / layernorm / fused-attn — non-trivial
//!   substrate decisions about which kernels carry their own backward.

#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::Tensor;
use kiln_autograd::{
    AddBackward, BackwardOp, CrossEntropyKtBackward, EmbeddingBackward, LoraDeltaAddBackward,
    MatmulBackward, MulSigmoidGateBackward, ReshapeBackward, RopeSplitHalfBackward, SiluBackward,
    Tape,
};

use crate::backend::BackendRuntime;
use crate::forward::{
    gdn_recurrent_backward_no_grad, gdn_recurrent_forward_from_parts, GDN_CHUNK_SIZE,
};
use crate::lora_loader::LoraProjectionWeights;

// Phase 6a/CP-4 (#1082): the thread-local-tape scope machinery
// (`with_thread_local_tape`, `with_active_tape`, `tape_forward_enabled`)
// originally lived here. Wave-13 (#1082) promoted it into
// `kiln-autograd::tape_scope` so the OPD and FLCE kernel crates (and
// their `kiln-train` callers) can share the same thread-local handle
// without taking a `kiln-model` dependency. We re-export from here for
// back-compat — every existing call site (the parity test, the
// `forward.rs:7178` adapter call) keeps compiling unchanged.
pub use kiln_autograd::{tape_forward_enabled, with_active_tape, with_thread_local_tape};

/// Borrow a candle input as a kt tensor for the tape path, REUSING an
/// upstream adapter's kt output when this candle tensor was produced by one
/// in the active bridge scope (so the recorded kt `Tape` stays connected —
/// the consumer's input id becomes the producer's output id). Falls back to
/// a fresh zero-copy borrow otherwise (and outside any bridge scope, which
/// is the common case + every per-op parity test). (#1082 CP-4 endgame.)
fn tape_kt_input(x: &Tensor) -> Option<kiln_tensor::Tensor> {
    if let Some(t) = kiln_kt_bridge::tape_bridge::kt_input_for_candle(x.id()) {
        return Some(t);
    }
    kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(x).ok()
}

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
    let x_kt = match tape_kt_input(x) {
        Some(t) => t,
        None => return Ok(None),
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

    // CP-4 (#1082) tape_bridge: register the (kt_id ↔ candle_id) IO
    // mappings so a surrounding `with_tape_scope_emit_to_grad_store`
    // can transmute the tape-recorded backward into candle-typed
    // gradients in the candle GradStore. No-ops cleanly when no
    // bridge scope is active.
    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(w_kt.id(), weight.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Attempt to run matmul through the kt-typed op registry
/// (`kiln_tensor::ops::matmul`) and record a `MatmulBackward` node on
/// the active thread-local tape.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned
///   `Tensor` is a copy of the kt-typed output into a candle CUDA
///   tensor; a `MatmulBackward { a, b }` node was recorded on the
///   active thread-local tape.
/// * `Ok(None)` — the gate was off, no thread-local tape is active,
///   the kt-bridge borrow failed (layout / dtype / device mismatch),
///   or the kt op-registry rejected the inputs. The caller must fall
///   through to the existing dispatch.
/// * `Err(...)` — an unexpected forward failure or a kt -> candle
///   copy-back failure. Propagated so callers see the failure cleanly
///   instead of silently masking it.
///
/// Follows the same envelope-tristate-then-record pattern as
/// [`try_tape_rms_norm_cuda`]: borrow zero-copy via
/// `kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow`, run the
/// kt-native forward, record the backward node onto the active tape,
/// and copy the kt output back into a candle CUDA tensor. The returned
/// tensor has no candle `BackpropOp` lineage — backward must be driven
/// via `Tape::backward`.
///
/// # CP-4 (#1082) context
///
/// This is the matmul half of the "copy-paste adapter for each
/// primitive" plan documented in `deed13a8`. The forward is
/// `kiln_tensor::ops::matmul` (kt op-registry dispatch — CUDA path
/// today via `kiln_tensor::cuda_matmul`, future Vulkan/Metal/CPU
/// paths as the op registry grows); the backward is
/// `kiln_autograd::backwards::MatmulBackward` which produces
/// `da = grad_y @ b^T` and `db = a^T @ grad_y`. Saving
/// `a.clone()` + `b.clone()` is an `Arc` bump on the kt-tensor's
/// storage handle (no allocation), so the lifetime of the saved
/// tensors extends past the local borrow at zero compute cost.
pub fn try_tape_matmul_cuda(a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // kt borrow: zero-copy view of the candle CUDA tensors as kt
    // tensors. Returns `Err` (which we treat as "skip") on layout /
    // dtype / device mismatch.
    let a_kt = match tape_kt_input(a) {
        Some(t) => t,
        None => return Ok(None),
    };
    let b_kt = match tape_kt_input(b) {
        Some(t) => t,
        None => return Ok(None),
    };

    // Record only when a tape scope is active. Outside a scope,
    // `with_active_tape` returns `None` and we fall through to the
    // existing dispatch — matching the rms_norm adapter's contract.
    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::matmul(&a_kt, &b_kt)
            .map_err(|e| anyhow::anyhow!("kt matmul: {e}"))?;
        tape.record(
            &y,
            &[&a_kt, &b_kt],
            Box::new(MatmulBackward {
                a: a_kt.clone(),
                b: b_kt.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt.context("tape_forward::try_tape_matmul_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_matmul_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: register the (kt_id ↔ candle_id) IO
    // mappings so a surrounding `with_tape_scope_emit_to_grad_store`
    // can transmute the tape-recorded backward into candle-typed
    // gradients in the candle GradStore. No-ops cleanly when no
    // bridge scope is active.
    kiln_kt_bridge::tape_bridge::register_input_mapping(a_kt.id(), a.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(b_kt.id(), b.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Attempt to run SiLU through the kt-typed op registry
/// (`kiln_tensor::ops::silu`) and record a `SiluBackward` node on the
/// active thread-local tape.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned
///   `Tensor` is a copy of the kt-typed output into a candle CUDA
///   tensor; a `SiluBackward { x }` node was recorded on the active
///   thread-local tape.
/// * `Ok(None)` — the gate was off, no thread-local tape is active,
///   the kt-bridge borrow failed, or the kt op-registry declined.
///   The caller must fall through to the existing dispatch.
/// * `Err(...)` — an unexpected forward failure or a kt -> candle
///   copy-back failure. Propagated so callers see the failure cleanly
///   instead of silently masking it.
///
/// Mirrors the [`try_tape_matmul_cuda`] adapter: zero-copy borrow,
/// kt-native forward via the op registry (CUDA SiLU kernel today via
/// `kiln_tensor::cuda_activation_unary`, kind tag 0), tape record,
/// kt -> candle copy-back. The returned tensor has no candle
/// `BackpropOp` lineage — backward is on the tape only.
///
/// # CP-4 (#1082) context
///
/// SiLU completes the matmul/silu/embedding adapter triplet sketched
/// in `deed13a8`'s "Out of scope" section. The backward is
/// `dx = grad_y * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))`
/// (see `kiln_autograd::backwards::activation::SiluBackward`). Saving
/// `x.clone()` is an `Arc` bump on the kt-tensor storage; the tape
/// owns it through the backward call.
pub fn try_tape_silu_cuda(x: &Tensor) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    let x_kt = match tape_kt_input(x) {
        Some(t) => t,
        None => return Ok(None),
    };

    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::silu(&x_kt)
            .map_err(|e| anyhow::anyhow!("kt silu: {e}"))?;
        tape.record(
            &y,
            &[&x_kt],
            Box::new(SiluBackward { x: x_kt.clone() }),
        );
        Ok(y)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt.context("tape_forward::try_tape_silu_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_silu_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: register the (kt_id ↔ candle_id) IO
    // mappings so a surrounding `with_tape_scope_emit_to_grad_store`
    // can transmute the tape-recorded backward into candle-typed
    // gradients in the candle GradStore. No-ops cleanly when no
    // bridge scope is active.
    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Attempt to run an embedding lookup through the kt-typed op registry
/// (`kiln_tensor::ops::embedding`) and record an `EmbeddingBackward`
/// node on the active thread-local tape.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned
///   `Tensor` is a copy of the kt-typed output into a candle CUDA
///   tensor; an `EmbeddingBackward { vocab_size, hidden, token_ids }`
///   node was recorded on the active thread-local tape.
/// * `Ok(None)` — the gate was off, no thread-local tape is active,
///   the kt-bridge borrow failed (layout / dtype / device mismatch),
///   or the kt op-registry rejected the inputs (e.g. I64 indices, the
///   substrate cast kernel isn't extended for those yet — see
///   `kiln_tensor::ops::embedding::EmbeddingOp::cuda_fwd`). The
///   caller must fall through to the existing dispatch.
/// * `Err(...)` — an unexpected forward failure or a kt -> candle
///   copy-back failure. Propagated so callers see the failure cleanly
///   instead of silently masking it.
///
/// Mirrors the [`try_tape_matmul_cuda`] adapter: zero-copy borrow,
/// kt-native forward via the op registry (the CUDA path dispatches to
/// `kiln_tensor::cuda_index_select_dim0` underneath, with a
/// flatten -> gather -> reshape step for multi-dim `token_ids`),
/// tape record, kt -> candle copy-back. The returned tensor has no
/// candle `BackpropOp` lineage — backward is on the tape only.
///
/// # CP-4 (#1082) context
///
/// Embedding completes the matmul / silu / embedding adapter triplet
/// sketched in `deed13a8`'s "Out of scope" section. The backward is
/// `kiln_autograd::backwards::EmbeddingBackward` which produces
/// `d_weights = scatter_add(grad_output, axis=0, indices=token_ids,
/// target_dim=vocab_size)` and `d_token_ids = None` (indices are
/// non-differentiable). Saving `token_ids.clone()` on the
/// `EmbeddingBackward` struct is an `Arc` bump on the kt-tensor
/// storage handle (no allocation), so the lifetime of the saved
/// indices extends past the local borrow at zero compute cost.
///
/// # Envelope
///
/// Same as `kiln_tensor::ops::embedding`'s CUDA fast-path:
/// * `weights`: rank-2 `[vocab_size, hidden]`, contiguous, F32 / BF16
///   / F16 (packed dtypes return `Ok(None)`).
/// * `token_ids`: rank ≥ 1, contiguous, U32 (I64 returns `Ok(None)`
///   until the substrate cast kernel grows that path).
///
/// Inputs outside the envelope return `Ok(None)` so the caller falls
/// through to the existing candle `index_select` path.
pub fn try_tape_embedding_cuda(
    weights: &Tensor,
    token_ids: &Tensor,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // kt borrow: zero-copy view of the candle CUDA tensors as kt
    // tensors. Returns `Err` (which we treat as "skip") on layout /
    // dtype / device mismatch.
    let w_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weights) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let ids_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(token_ids) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // Forward envelope check up front so we can read vocab_size /
    // hidden from `weights` before any tape recording. If the rank
    // check fails we fall through to the existing dispatch (which
    // will surface a clearer error if applicable).
    if w_kt.shape().len() != 2 || ids_kt.shape().is_empty() {
        return Ok(None);
    }
    let vocab_size = w_kt.shape()[0];
    let hidden = w_kt.shape()[1];

    // Record only when a tape scope is active. Outside a scope,
    // `with_active_tape` returns `None` and we fall through — matching
    // the matmul / silu / rmsnorm adapters' contract.
    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::embedding(&w_kt, &ids_kt)
            .map_err(|e| anyhow::anyhow!("kt embedding: {e}"))?;
        tape.record(
            &y,
            &[&w_kt, &ids_kt],
            Box::new(EmbeddingBackward {
                vocab_size,
                hidden,
                token_ids: ids_kt.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt =
        out_kt.context("tape_forward::try_tape_embedding_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_embedding_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: register the (kt_id ↔ candle_id) IO
    // mappings so a surrounding `with_tape_scope_emit_to_grad_store` can
    // transmute the tape-recorded `EmbeddingBackward` weight gradient into
    // a candle-typed gradient in the candle GradStore. Without these, the
    // embedding-table gradient (which is tied to `lm_head` in Qwen3.5 and
    // therefore trainable) never reaches the optimizer — the bug the DoD
    // "tied-weight grad accumulation parity" item guards against. No-ops
    // cleanly when no bridge scope is active.
    //
    // `token_ids` is intentionally NOT mapped: integer gather indices carry
    // no gradient, and `EmbeddingBackward` returns `None` for that input.
    kiln_kt_bridge::tape_bridge::register_input_mapping(w_kt.id(), weights.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Attempt to run the SwiGLU MLP gate-fuse (`silu(gate) * up`) through
/// the kt-typed op registry (`kiln_tensor::ops::mul_sigmoid_gate`) and
/// record a `MulSigmoidGateBackward` node on the active thread-local
/// tape.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned
///   `Tensor` is a copy of the kt-typed output into a candle CUDA
///   tensor; a `MulSigmoidGateBackward { gate, up }` node was recorded
///   on the active thread-local tape.
/// * `Ok(None)` — the gate was off, no thread-local tape is active,
///   the kt-bridge borrow failed (layout / dtype / device mismatch),
///   or the kt op-registry rejected the inputs (shape / dtype /
///   contiguity envelope). The caller must fall through to the
///   existing dispatch.
/// * `Err(...)` — an unexpected forward failure or a kt -> candle
///   copy-back failure. Propagated so callers see the failure cleanly
///   instead of silently masking it.
///
/// Mirrors the [`try_tape_silu_cuda`] adapter: zero-copy borrow of
/// both inputs, kt-native forward via the op registry (the CUDA path
/// composes `cuda_activation_unary(kind=0)` + `cuda_elementwise_binary
/// (kind=2)` underneath — same kernels the production
/// `fused_mlp_silu_mul_kt` shim drives), tape record, kt -> candle
/// copy-back. The returned tensor has no candle `BackpropOp` lineage —
/// backward is on the tape only.
///
/// # CP-4 (#1082) context
///
/// SwiGLU's `silu(gate) * up` is the MLP gate path (`:kiln/gdn/gates`
/// in Phase 6 profiling — ~18% of decode time). The backward is
/// `kiln_autograd::backwards::swiglu::MulSigmoidGateBackward`:
///
/// ```text
/// d_gate = dy * up * (sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate)))
/// d_up   = dy * gate * sigmoid(gate)
/// ```
///
/// Saving `gate.clone()` + `up.clone()` is an `Arc` bump on the
/// kt-tensor's storage handle (no allocation), so the lifetime of the
/// saved tensors extends past the local borrow at zero compute cost.
///
/// # Envelope
///
/// Same as `kiln_tensor::ops::mul_sigmoid_gate`'s CUDA fast-path:
/// * `gate`: contiguous, F32 / BF16 / F16, shape == `up`.
/// * `up`: contiguous, F32 / BF16 / F16, shape == `gate`.
///
/// Inputs outside the envelope return `Ok(None)` so the caller falls
/// through to the existing `fused_mlp_silu_mul_kt` /
/// `cuda_silu(gate) * up` paths.
pub fn try_tape_swiglu_cuda(gate: &Tensor, up: &Tensor) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // kt borrow: zero-copy view of the candle CUDA tensors as kt
    // tensors. Returns `Err` (which we treat as "skip") on layout /
    // dtype / device mismatch.
    let gate_kt = match tape_kt_input(gate) {
        Some(t) => t,
        None => return Ok(None),
    };
    let up_kt = match tape_kt_input(up) {
        Some(t) => t,
        None => return Ok(None),
    };

    // Record only when a tape scope is active. Outside a scope,
    // `with_active_tape` returns `None` and we fall through to the
    // existing dispatch — matching the matmul / silu / rmsnorm /
    // embedding adapters' contract.
    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::mul_sigmoid_gate(&gate_kt, &up_kt)
            .map_err(|e| anyhow::anyhow!("kt mul_sigmoid_gate: {e}"))?;
        tape.record(
            &y,
            &[&gate_kt, &up_kt],
            Box::new(MulSigmoidGateBackward {
                gate: gate_kt.clone(),
                up: up_kt.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt.context("tape_forward::try_tape_swiglu_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_swiglu_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: register the (kt_id ↔ candle_id) IO
    // mappings so a surrounding `with_tape_scope_emit_to_grad_store`
    // can transmute the tape-recorded `MulSigmoidGateBackward` into
    // candle-typed gradients in the candle GradStore. No-ops cleanly
    // when no bridge scope is active.
    kiln_kt_bridge::tape_bridge::register_input_mapping(gate_kt.id(), gate.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(up_kt.id(), up.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Attempt to run split-half (GPT-NeoX-style) RoPE — kiln's Qwen3.5-4B
/// rotary convention — through the kt-typed op registry
/// (`kiln_tensor::ops::rope_split_half`) and record a
/// `RopeSplitHalfBackward` node on the active thread-local tape.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned `Tensor`
///   is a copy of the kt-typed output into a candle CUDA tensor; a
///   `RopeSplitHalfBackward { rotary_dim, cos, sin }` node was recorded
///   on the active thread-local tape.
/// * `Ok(None)` — gate off / `x` is not rank-4 / no thread-local tape /
///   kt-bridge borrow rejected the inputs. The caller falls through to
///   the existing dispatch.
/// * `Err(...)` — an unexpected forward or kt -> candle copy-back failure.
///
/// # Why split-half (not `kiln_tensor::ops::rope`)
///
/// `kiln_tensor::ops::rope` uses the *interleaved* (GPT-J) convention;
/// kiln's production `apply_rope` uses *split-half* (GPT-NeoX). The two
/// disagree for `rotary_dim >= 4`, so the adapter must route through
/// `rope_split_half` to stay bit-faithful to the model. The backward is
/// the same op with `sin` negated (a rotation's adjoint), computed on the
/// grad's own device with no host round-trip.
///
/// # CP-4 (#1082) context
///
/// `x` is `[batch, seq, num_heads, head_dim]`; `cos`/`sin` are
/// `[seq, rotary_dim/2]` schedules — non-differentiable, so only `x`
/// receives an IO mapping (cf. the embedding adapter's `token_ids`).
/// `cos`/`sin` are saved on the tape node on their native (CUDA) device.
pub fn try_tape_rope_cuda(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // `rope_split_half`'s contract is rank-4 [batch, seq, num_heads,
    // head_dim]; bail to the existing dispatch for anything else rather
    // than recording a node that would fail at backward.
    if x.rank() != 4 || rotary_dim == 0 || rotary_dim > head_dim {
        return Ok(None);
    }

    let x_kt = match tape_kt_input(x) {
        Some(t) => t,
        None => return Ok(None),
    };
    let cos_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(cos) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let sin_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(sin) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // Record only when a tape scope is active. Outside a scope,
    // `with_active_tape` returns `None` and we fall through to the
    // existing dispatch — matching the other adapters' contract.
    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::rope_split_half(&x_kt, &cos_kt, &sin_kt, rotary_dim)
            .map_err(|e| anyhow::anyhow!("kt rope_split_half: {e}"))?;
        tape.record(
            &y,
            &[&x_kt],
            Box::new(RopeSplitHalfBackward {
                rotary_dim,
                cos: cos_kt.clone(),
                sin: sin_kt.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt.context("tape_forward::try_tape_rope_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_rope_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: only `x` is differentiable; `cos`/`sin`
    // are non-differentiable schedules (cf. embedding's `token_ids`), so
    // they carry no input mapping.
    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Attempt to run an elementwise residual add through the kt-typed op
/// registry (`kiln_tensor::ops::add`) and record an `AddBackward` node
/// on the active thread-local tape.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned `Tensor`
///   is a copy of the kt-typed output into a candle CUDA tensor; an
///   `AddBackward` node was recorded on the active thread-local tape.
/// * `Ok(None)` — gate off / shape mismatch (caller may broadcast) / no
///   thread-local tape / kt-bridge borrow rejected the inputs. The caller
///   falls through to the existing candle add.
/// * `Err(...)` — an unexpected forward or kt -> candle copy-back failure.
///
/// # CP-4 (#1082) context
///
/// `add` is the residual-connection primitive (`kiln/residual`):
/// `c = a + b`, so `da = dc` and `db = dc` — `AddBackward` is field-less
/// and routes the upstream grad to both inputs. Both `a` and `b` are
/// differentiable and receive IO mappings. `kiln_tensor::ops::add` is a
/// same-shape op (no broadcast), so the adapter short-circuits to
/// `Ok(None)` on a shape mismatch and lets the caller's candle add (which
/// may broadcast) handle it.
pub fn try_tape_add_cuda(a: &Tensor, b: &Tensor) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // `kiln_tensor::ops::add` requires identical shapes; defer any
    // broadcasting add to the caller's candle path.
    if a.dims() != b.dims() {
        return Ok(None);
    }

    let a_kt = match tape_kt_input(a) {
        Some(t) => t,
        None => return Ok(None),
    };
    let b_kt = match tape_kt_input(b) {
        Some(t) => t,
        None => return Ok(None),
    };

    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::add(&a_kt, &b_kt)
            .map_err(|e| anyhow::anyhow!("kt add: {e}"))?;
        tape.record(&y, &[&a_kt, &b_kt], Box::new(AddBackward));
        Ok(y)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt.context("tape_forward::try_tape_add_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_add_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: both inputs are differentiable.
    kiln_kt_bridge::tape_bridge::register_input_mapping(a_kt.id(), a.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(b_kt.id(), b.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Attempt to run the softmax + NLL cross-entropy LOSS through the
/// kt-typed op registry (`kiln_tensor::ops::cross_entropy`) and record a
/// device-agnostic `CrossEntropyKtBackward` node on the active tape.
///
/// Returns:
/// * `Ok(Some(loss))` — the tape-forward path ran. The returned scalar
///   `Tensor` is a copy of the kt-typed loss into a candle CUDA tensor;
///   a `CrossEntropyKtBackward { logits, targets }` node was recorded.
/// * `Ok(None)` — gate off / non-canonical shapes / no thread-local tape
///   / kt-bridge borrow rejected the inputs. The caller falls through to
///   the existing candle loss composite.
/// * `Err(...)` — an unexpected forward or kt -> candle copy-back failure.
///
/// # CP-4 (#1082) context — the loss closes forward coverage
///
/// With the prior seven adapters (`rms_norm`, `matmul`, `silu`,
/// `embedding`, `swiglu`, `rope`, `add`), the full training forward path
/// — embedding → blocks → final norm → lm_head matmul — can record onto
/// the kt Tape. `cross_entropy` is the terminal op: once it records, the
/// whole forward→loss is tape-covered and `Tape::backward` can be seeded
/// from the scalar loss.
///
/// The recorded backward is `CrossEntropyKtBackward` (device-agnostic:
/// `d_logits = (softmax(logits) - one_hot(targets)) * g / batch`, with the
/// `[batch, vocab]` activations never leaving the device — only the scalar
/// grad multiplier touches the host). `targets` are class indices
/// (non-differentiable), so only `logits` is IO-mapped.
///
/// # Not wired into the production loss (yet)
///
/// Unlike the other adapters, this is NOT called from the trainer's
/// `cross_entropy_loss`. cross_entropy is the backward ROOT, and the
/// current `tape_bridge` keeps candle's `loss.backward()` authoritative
/// (the tape walk merges into candle's `GradStore`). Tape-routing the
/// loss would detach it from candle's graph and stop candle backward
/// from starting. Production wiring waits for the tape-authoritative
/// bridge (the CP-4 endgame). This adapter + its parity tests prove the
/// substrate handles the loss op so that flip is unblocked.
pub fn try_tape_cross_entropy_cuda(
    logits: &Tensor,
    targets: &Tensor,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // kt cross_entropy's contract: logits [batch, vocab], targets
    // [batch]. Defer any other shape to the caller's candle composite.
    if logits.rank() != 2 || targets.rank() != 1 {
        return Ok(None);
    }

    let logits_kt = match tape_kt_input(logits) {
        Some(t) => t,
        None => return Ok(None),
    };
    let targets_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(targets) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_tensor::ops::cross_entropy(&logits_kt, &targets_kt)
            .map_err(|e| anyhow::anyhow!("kt cross_entropy: {e}"))?;
        tape.record(
            &y,
            &[&logits_kt, &targets_kt],
            Box::new(CrossEntropyKtBackward {
                logits: logits_kt.clone(),
                targets: targets_kt.clone(),
            }),
        );
        Ok(y)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt =
        out_kt.context("tape_forward::try_tape_cross_entropy_cuda: kt-tape forward failed")?;

    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_cross_entropy_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: only `logits` is differentiable;
    // `targets` are non-differentiable class indices.
    kiln_kt_bridge::tape_bridge::register_input_mapping(logits_kt.id(), logits.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Attempt to run the LoRA delta-and-add through the kt-typed op surface
/// (`kiln_tensor::ops::{matmul, mul_scalar, add}`) and record a fused
/// `LoraDeltaAddBackward` node on the active thread-local tape.
///
/// The forward computes:
/// ```text
/// out = base + scale * (x @ A^T @ B^T)
/// ```
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran. The returned candle
///   `Tensor` is a copy of the kt-typed output (reshaped back to
///   `base.shape()`); a `LoraDeltaAddBackward { x, a, b, scale }` node was
///   recorded on the active thread-local tape with inputs
///   `[base, x, A, B]` in that order.
/// * `Ok(None)` — gate off (no thread-local tape, `KILN_USE_TAPE_FORWARD`
///   off, `KILN_USE_TAPE_LORA_ADD` off, or device / dtype / shape /
///   contiguity preconditions fail). The caller must fall through to the
///   existing CUDA / Metal / Vulkan / candle dispatch.
/// * `Err(...)` — an unexpected kt forward, kt → candle copy-back, or
///   reshape failure. Propagated so callers see the failure cleanly
///   instead of silently masking it.
///
/// # CP-4 (#1082) context — closes LoRA Var grad coverage
///
/// Without this adapter, the production LoRA delta-add dispatch in
/// `add_lora_delta_to_base` lands in either `cuda_lora_add_training_f32`,
/// `cuda_lora_add_training_bf16`, `backend.lora_decode_add`, or the
/// Phase-4.1 `CustomOp3` path — none of which the kt `Tape` walker sees.
/// Under `KILN_USE_TAPE_AUTHORITATIVE`, the resulting candle `GradStore`
/// has no entries for the LoRA `Var`s (`proj.a`, `proj.b`), so the
/// optimiser step is a no-op for the adapter parameters. With this
/// adapter on, the fused backward emits grads for `proj.a` and `proj.b`
/// in their original `[rank, in_features]` / `[out_features, rank]`
/// shapes, and the IO mapping pairs each kt input id with the Var's
/// candle id so the parity gate sees nonzero matched LoRA grads.
///
/// # Why fused (not 4 chained tape nodes)
///
/// `kiln_tensor::ops::matmul` requires `[..., M, K] @ [..., K, N]`
/// contiguous. The LoRA delta needs `A^T` and `B^T`, which would change
/// the kt `TensorId` and the gradient shape on the way out. Mapping
/// `kt_input_id → candle_id` keyed on a transposed view would deposit
/// `grad_A^T` (shape `[in_features, rank]`) under the candle id for
/// `proj.a` (shape `[rank, in_features]`) — a silent shape disagreement
/// that the bridge surfaces as a `kt -> candle grad copy` failure.
/// Fusing the four ops into a single `LoraDeltaAddBackward` keeps the
/// per-input grads in the original Var layouts so the IO mapping is
/// direct: `(a_kt.id(), proj.a.id())`, `(b_kt.id(), proj.b.id())`.
/// See `kiln_autograd::backwards::lora_delta_add` for the math
/// derivation.
#[allow(clippy::too_many_lines)]
pub fn try_tape_lora_add_cuda(
    base: &Tensor,
    x: &Tensor,
    proj: &LoraProjectionWeights,
    lora_scale: f32,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() || !tape_lora_add_enabled() {
        return Ok(None);
    }

    // Device + dtype gate: kt matmul + kt mul_scalar are CUDA-only here
    // (they have CPU paths too, but the bridge's `kt_tensor_from_candle_cuda_*`
    // helpers are CUDA-only). Match the existing tape adapters.
    if !matches!(base.device(), candle_core::Device::Cuda(_))
        || !matches!(x.device(), candle_core::Device::Cuda(_))
        || !matches!(proj.a.device(), candle_core::Device::Cuda(_))
        || !matches!(proj.b.device(), candle_core::Device::Cuda(_))
    {
        return Ok(None);
    }
    if base.dtype() != x.dtype() {
        // kt matmul requires matching dtypes throughout the composed
        // forward; defer mixed-dtype cases to the existing CUDA path
        // which already handles cross-dtype promotion via CustomOp3.
        return Ok(None);
    }
    // Only BF16 / F32 today (matches the kt matmul envelope).
    if !matches!(
        base.dtype(),
        candle_core::DType::BF16 | candle_core::DType::F32
    ) {
        return Ok(None);
    }
    if proj.a.dtype() != base.dtype() || proj.b.dtype() != base.dtype() {
        // For the first cut we require A/B already pre-cast to the
        // forward dtype. The existing CUDA path handles BF16/F16 → F32
        // upcasts for the adapter weights via CustomOp3; that's
        // deliberately out-of-scope here so the tape adapter stays a
        // single-dtype primitive. The dispatch gate falls through and
        // the existing path takes over when dtypes mismatch.
        return Ok(None);
    }

    // Shape gate.
    let base_dims = base.dims().to_vec();
    let x_dims = x.dims().to_vec();
    if base_dims.len() < 2 || x_dims.len() != base_dims.len() {
        return Ok(None);
    }
    if base_dims[..base_dims.len() - 1] != x_dims[..x_dims.len() - 1] {
        return Ok(None);
    }
    let out_features = *base_dims.last().unwrap();
    let in_features = *x_dims.last().unwrap();
    let rows: usize = base_dims[..base_dims.len() - 1].iter().product();
    if rows == 0 {
        return Ok(None);
    }

    let Ok((rank, a_in)) = proj.a.dims2() else {
        return Ok(None);
    };
    let Ok((b_out, b_rank)) = proj.b.dims2() else {
        return Ok(None);
    };
    if a_in != in_features || b_out != out_features || b_rank != rank {
        return Ok(None);
    }

    // Reshape base + x to 2-D for the kt matmul envelope. The candle
    // reshape is a layout view; .contiguous() materialises if needed.
    let base_2d = base.reshape((rows, out_features))?.contiguous()?;
    let x_2d = x.reshape((rows, in_features))?.contiguous()?;

    // kt borrows: zero-copy views of the candle CUDA tensors. A/B are
    // already 2-D and presumed contiguous (Vars allocated by the
    // optimiser are always so); short-circuit on a non-contig borrow.
    let base_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&base_2d) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    // For x, prefer to thread the kt id from an upstream tape adapter's
    // output so the tape stays connected. The `tape_kt_input` helper
    // looks up `x.id()` (the candle id BEFORE the reshape) — if an
    // upstream adapter produced `x`, we'd see its kt output here. The
    // reshape changed the candle id, so we fall through to a fresh
    // borrow of `x_2d` for the common case (a fully-detached LoRA call).
    let x_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&x_2d) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let a_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&proj.a) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let b_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&proj.b) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    // kt envelope: matmul requires contiguous 2-D inputs. base + x are
    // already so by construction above; A and B come straight from the
    // LoRA loader and are contiguous Vars. If for any reason a Var is
    // non-contig (in-place layout munging upstream), fall through.
    if !a_kt.is_contiguous() || !b_kt.is_contiguous() {
        return Ok(None);
    }

    // Record on the active tape (if any). Outside a scope, fall through.
    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        // h = x @ A^T
        let a_t_kt = a_kt
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("kt a.transpose: {e}"))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt a_t.contiguous: {e}"))?;
        let h_kt = kiln_tensor::ops::matmul(&x_kt, &a_t_kt)
            .map_err(|e| anyhow::anyhow!("kt matmul x@a_t: {e}"))?;
        // d = h @ B^T
        let b_t_kt = b_kt
            .transpose(0, 1)
            .map_err(|e| anyhow::anyhow!("kt b.transpose: {e}"))?
            .contiguous()
            .map_err(|e| anyhow::anyhow!("kt b_t.contiguous: {e}"))?;
        let d_kt = kiln_tensor::ops::matmul(&h_kt, &b_t_kt)
            .map_err(|e| anyhow::anyhow!("kt matmul h@b_t: {e}"))?;
        // delta = d * scale  (kt elementwise)
        let delta_kt = kiln_tensor::ops::mul_scalar(&d_kt, lora_scale)
            .map_err(|e| anyhow::anyhow!("kt mul_scalar(scale): {e}"))?;
        // out = base + delta  (kt elementwise add)
        let out_kt = kiln_tensor::ops::add(&base_kt, &delta_kt)
            .map_err(|e| anyhow::anyhow!("kt add(base, delta): {e}"))?;

        // ONE fused tape node — see module docs on `LoraDeltaAddBackward`
        // for the rationale (transpose handling, IO-mapping shape match).
        tape.record(
            &out_kt,
            &[&base_kt, &x_kt, &a_kt, &b_kt],
            Box::new(LoraDeltaAddBackward {
                x: x_kt.clone(),
                a: a_kt.clone(),
                b: b_kt.clone(),
                scale: lora_scale,
            }),
        );
        Ok(out_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt.context("tape_forward::try_tape_lora_add_cuda: kt-tape forward failed")?;
    let out_2d = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_lora_add_cuda: kt -> candle copy failed")?;
    let out = out_2d.reshape(base_dims.clone())?;

    // CP-4 (#1082) tape_bridge: register IO mappings keyed on the
    // candle ids the optimiser cares about — proj.a and proj.b are the
    // LoRA `Var`s; base / x are intermediate but registered for chaining
    // completeness so a Vars-only consumer of the bridged GradStore can
    // be added later without re-touching this adapter.
    kiln_kt_bridge::tape_bridge::register_input_mapping(base_kt.id(), base_2d.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x_2d.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(a_kt.id(), proj.a.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(b_kt.id(), proj.b.id());
    // The downstream candle consumer holds `out` (the reshape-back of
    // the kt-side 2-D output). Register the user-facing candle id only;
    // the bridge panics on a duplicate kt output id, so we pick the one
    // the consumer actually carries into the loss graph and chain on
    // the same id for upstream re-use via `kt_input_for_candle(out.id())`.
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// True iff `KILN_USE_TAPE_FLASH_ATTN` is set to an enable value.
///
/// Separate gate from `KILN_USE_TAPE_FORWARD` so the flash-attention tape
/// adapter rolls out independently of the rest of the tape-forward fleet:
/// flipping it changes the attention-block gradient path from candle's
/// `CudaFlashAttentionTrainingBf16` CustomOp3 to a kt `Tape` node, so the
/// opt-in is intentionally narrow. Cached after first read, matching
/// [`tape_lora_add_enabled`].
pub fn tape_flash_attn_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_FLASH_ATTN")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(false)
    })
}

/// Fused tape backward for the vendored FlashAttention-2 forward
/// (`kiln_flash_attn::flash_attn_fwd_kt`).
///
/// # Why this `BackwardOp` lives in `kiln-model`, not `kiln-autograd`
///
/// Every other `BackwardOp` is a device-agnostic composite of
/// `kiln_tensor::ops` and lives in `kiln-autograd`. FlashAttention is the
/// exception: its forward and backward are a single fused CUDA kernel
/// (`kiln_flash_attn::flash_attn_{fwd,bwd}_kt`) with no device-agnostic
/// composite matching the kernel's numerics or memory profile. Since
/// `kiln-autograd` deliberately carries no `kiln-flash-attn` dependency
/// (layering — it stays buildable on every backend), the op that
/// dispatches the kernel lives here in `kiln-model` (which already depends
/// on `kiln-flash-attn`) and implements the `kiln_autograd::BackwardOp`
/// trait. The CPU/composite reference for parity lives in the test.
///
/// # Saved tensors
///
/// `q`, `k`, `v` (the GROUPED GQA inputs as handed to `flash_attn_fwd_kt`),
/// the forward output `out`, and `softmax_lse` — all kt clones (Arc bumps;
/// no allocation). `scale`/`causal` are the forward params;
/// `heads_q`/`heads_kv` drive the GQA gradient collapse.
///
/// # Backward
///
/// `flash_attn_bwd_kt(dout, q, k, v, out, lse, scale, causal)` returns
/// `(dq, dk, dv)` where `dk`/`dv` come back EXPANDED to `heads_q` (the
/// kernel internally broadcasts grouped K/V). When `heads_kv != heads_q`
/// we collapse them to `heads_kv` by reshaping to `[b, sk, heads_kv,
/// groups, hd]` and summing the group axis — mirroring the
/// `CudaFlashAttentionTrainingBf16::bwd` candle path exactly. The collapse
/// runs in F32 (cast → sum → cast back to BF16) so the group reduction
/// doesn't lose precision in BF16.
#[derive(Debug)]
pub(crate) struct FlashAttnBackward {
    q: kiln_tensor::Tensor,
    k: kiln_tensor::Tensor,
    v: kiln_tensor::Tensor,
    out: kiln_tensor::Tensor,
    softmax_lse: kiln_tensor::Tensor,
    scale: f32,
    causal: bool,
    heads_q: usize,
    heads_kv: usize,
}

impl BackwardOp for FlashAttnBackward {
    fn name(&self) -> &'static str {
        "flash_attn_backward"
    }
    fn input_count(&self) -> usize {
        // q, k, v.
        3
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        use kiln_tensor::{bail, DType};

        // FA bwd needs a BF16, compact-contiguous dout shaped like `out`.
        let dout = if grad_output.dtype() == DType::BF16 {
            grad_output.clone()
        } else {
            kiln_tensor::ops::cast(grad_output, DType::BF16)?
        };
        let dout = if dout.is_contiguous() {
            dout
        } else {
            dout.contiguous()?
        };

        let (dq, dk_exp, dv_exp) = kiln_flash_attn::flash_attn_bwd_kt(
            &dout,
            &self.q,
            &self.k,
            &self.v,
            &self.out,
            &self.softmax_lse,
            self.scale,
            self.causal,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!("FlashAttnBackward: flash_attn_bwd_kt: {e:?}"))
        })?;

        // GQA collapse: dk/dv come back expanded to heads_q.
        let (dk, dv) = if self.heads_kv != self.heads_q {
            if self.heads_kv == 0 || self.heads_q % self.heads_kv != 0 {
                bail!(
                    "FlashAttnBackward: invalid GQA heads_q={} heads_kv={}",
                    self.heads_q,
                    self.heads_kv
                );
            }
            let groups = self.heads_q / self.heads_kv;
            let collapse =
                |dexp: &kiln_tensor::Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
                    let s = dexp.shape();
                    if s.len() != 4 {
                        bail!(
                            "FlashAttnBackward: expanded grad must be rank-4 \
                             [b,sk,heads_q,hd], got {s:?}"
                        );
                    }
                    let (b, sk, hq, hd) = (s[0], s[1], s[2], s[3]);
                    if hq != self.heads_q {
                        bail!(
                            "FlashAttnBackward: expanded grad heads {hq} != heads_q {}",
                            self.heads_q
                        );
                    }
                    // Reduce groups in F32 (BF16 group-sum loses precision).
                    let f32g = kiln_tensor::ops::cast(dexp, DType::F32)?;
                    let grouped = f32g.reshape(vec![b, sk, self.heads_kv, groups, hd])?;
                    let grouped = if grouped.is_contiguous() {
                        grouped
                    } else {
                        grouped.contiguous()?
                    };
                    let summed = kiln_tensor::ops::sum_axis(&grouped, 3)?; // [b,sk,heads_kv,hd]
                    kiln_tensor::ops::cast(&summed, DType::BF16)
                };
            (collapse(&dk_exp)?, collapse(&dv_exp)?)
        } else {
            (dk_exp, dv_exp)
        };

        Ok(vec![Some(dq), Some(dk), Some(dv)])
    }
}

/// Attempt to route the FlashAttention-2 forward through the kt `Tape`
/// instead of candle's `CudaFlashAttentionTrainingBf16` CustomOp3.
///
/// `q` is `[b, sq, heads_q, hd]`; `k`/`v` are the GROUPED `[b, sk,
/// heads_kv, hd]` GQA tensors (the CUDA FA2 wrapper consumes grouped K/V
/// directly). Returns the attention output `[b, sq, heads_q, hd]` (the
/// caller reshapes to `[b, sq, heads_q*hd]` for o_proj). The recorded
/// [`FlashAttnBackward`] node emits GQA-collapsed `dq/dk/dv` so a
/// tape-authoritative backward seeded at the loss reaches the q/k/v
/// projections (and therefore their LoRA `Var`s) — this is the
/// attention-block link the CP-4 tape-authoritative SFT path was missing
/// (flash-attn previously recorded only onto candle's `BackpropOp` graph,
/// leaving the LoRA tape nodes a disconnected island).
///
/// `Ok(None)` (caller falls through to the existing CustomOp3 / fast path)
/// when: the gate is off, no tape scope is active, the inputs leave the
/// BF16/CUDA/contiguous/`head_dim∈{128,256}`/valid-GQA envelope, or a kt
/// borrow fails.
pub fn try_tape_flash_attn_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() || !tape_flash_attn_enabled() {
        return Ok(None);
    }

    // Device + dtype + layout envelope — mirror
    // `cuda_flash_attention_training_bf16`'s gate so we never record a
    // node the kernel would reject.
    if q.dtype() != candle_core::DType::BF16
        || k.dtype() != candle_core::DType::BF16
        || v.dtype() != candle_core::DType::BF16
        || !matches!(q.device(), candle_core::Device::Cuda(_))
        || !matches!(k.device(), candle_core::Device::Cuda(_))
        || !matches!(v.device(), candle_core::Device::Cuda(_))
        || !q.is_contiguous()
        || !k.is_contiguous()
        || !v.is_contiguous()
        || !matches!(head_dim, 128 | 256)
        || num_kv_heads == 0
        || num_heads % num_kv_heads != 0
    {
        return Ok(None);
    }
    let Ok((bq, _sq, hq, dq_)) = q.dims4() else {
        return Ok(None);
    };
    let Ok((bk, sk, hk, dk_)) = k.dims4() else {
        return Ok(None);
    };
    let Ok((bv, sv, hv, dv_)) = v.dims4() else {
        return Ok(None);
    };
    if bq != bk
        || bq != bv
        || sk != sv
        || hq != num_heads
        || hk != num_kv_heads
        || hv != num_kv_heads
        || dq_ != head_dim
        || dk_ != head_dim
        || dv_ != head_dim
    {
        return Ok(None);
    }

    let softmax_scale = 1.0 / (head_dim as f32).sqrt();
    let causal = true;

    // kt inputs — thread upstream adapter outputs (RoPE / q_norm produced
    // q; v straight from v_proj+lora) so the tape stays connected.
    let q_kt = match tape_kt_input(q) {
        Some(t) => t,
        None => return Ok(None),
    };
    let k_kt = match tape_kt_input(k) {
        Some(t) => t,
        None => return Ok(None),
    };
    let v_kt = match tape_kt_input(v) {
        Some(t) => t,
        None => return Ok(None),
    };

    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let (out_kt, lse_kt) =
            kiln_flash_attn::flash_attn_fwd_kt(&q_kt, &k_kt, &v_kt, softmax_scale, causal)
                .map_err(|e| anyhow::anyhow!("kt flash_attn_fwd_kt: {e:?}"))?;

        tape.record(
            &out_kt,
            &[&q_kt, &k_kt, &v_kt],
            Box::new(FlashAttnBackward {
                q: q_kt.clone(),
                k: k_kt.clone(),
                v: v_kt.clone(),
                out: out_kt.clone(),
                softmax_lse: lse_kt,
                scale: softmax_scale,
                causal,
                heads_q: num_heads,
                heads_kv: num_kv_heads,
            }),
        );
        Ok(out_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt =
        out_kt.context("tape_forward::try_tape_flash_attn_cuda: kt-tape forward failed")?;
    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_flash_attn_cuda: kt -> candle copy failed")?;

    kiln_kt_bridge::tape_bridge::register_input_mapping(q_kt.id(), q.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(k_kt.id(), k.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(v_kt.id(), v.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Route a `reshape` through the kt `Tape` so a tape-authoritative
/// backward stays connected ACROSS the reshape.
///
/// Why this matters for CP-4: the GQA attention fast path produces a
/// `[b, seq, heads, head_dim]` output from [`try_tape_flash_attn_cuda`]
/// and then reshapes it to `[b, seq, heads*head_dim]` before the o_proj
/// matmul. A plain candle reshape mints a fresh tensor id, so the
/// downstream o_proj adapter (`tape_kt_input`) could not chain back to the
/// flash node — the tape would fragment at the reshape and the q/k/v
/// (LoRA) grads would never flow. Recording a `ReshapeBackward` node
/// (whose adjoint just reshapes the grad back to the input shape) keeps
/// the chain intact: flash → reshape → o_proj → … → loss.
///
/// Gated on `KILN_USE_TAPE_FORWARD` + an active tape scope only (reshape
/// is a pure layout op with a trivial, always-safe adjoint, so it needs
/// no dedicated kill switch); it is only ever called from the sites that
/// opt in. Returns `Ok(None)` when the gate is off, no tape is active,
/// the input isn't CUDA, the element counts don't match, or a kt borrow
/// fails — the caller then falls through to a plain candle reshape.
pub fn try_tape_reshape_cuda(x: &Tensor, new_shape: Vec<usize>) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if !matches!(x.device(), candle_core::Device::Cuda(_)) {
        return Ok(None);
    }

    // Reuse an upstream adapter's kt output (e.g. the flash-attn node) so
    // the tape stays connected; else a fresh zero-copy borrow.
    let x_kt = match tape_kt_input(x) {
        Some(t) => t,
        None => return Ok(None),
    };

    let input_shape = x_kt.shape().to_vec();
    let in_elems: usize = input_shape.iter().product();
    let out_elems: usize = new_shape.iter().product();
    if in_elems != out_elems {
        // Not a pure reshape (would need a copy/broadcast); defer to the
        // caller's candle path.
        return Ok(None);
    }

    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        // kt reshape requires a contiguous source; flash output already is,
        // but materialise defensively for the general case.
        let x_c = if x_kt.is_contiguous() {
            x_kt.clone()
        } else {
            x_kt
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt reshape: x.contiguous: {e}"))?
        };
        let out_kt = x_c
            .reshape(new_shape.clone())
            .map_err(|e| anyhow::anyhow!("kt reshape: {e}"))?;
        // Single-input node; the adjoint reshapes the upstream grad back to
        // `input_shape` (the original kt input's shape).
        tape.record(
            &out_kt,
            &[&x_kt],
            Box::new(ReshapeBackward {
                input_shape: input_shape.clone(),
            }),
        );
        Ok(out_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt = out_kt.context("tape_forward::try_tape_reshape_cuda: kt-tape forward failed")?;
    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_reshape_cuda: kt -> candle copy failed")?;

    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// True iff `KILN_USE_TAPE_GDN` is set to an enable value. Narrow opt-in
/// for the GDN (linear-attention) recurrence tape adapter, separate from
/// the rest of the fleet. Cached after first read.
pub fn tape_gdn_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_GDN")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(false)
    })
}

/// Tape backward for the GDN (Gated DeltaNet linear-attention) recurrence.
///
/// # Why a candle-composite wrap (not a kt BackwardOp in kiln-autograd)
///
/// The GDN recurrence backward is a stateful chunk-wise reverse-time
/// algorithm already implemented + CPU-parity-tested as the device-
/// agnostic candle composite [`gdn_recurrent_backward_no_grad`]
/// (`test_gdn_recurrent_backward_no_grad_matches_autograd_cpu`). Rather
/// than re-derive it as kt ops, this `BackwardOp` wraps that proven
/// function: saves the candle forward inputs and, on backward, bridges
/// the upstream grad to candle, runs the existing chunk-wise backward, and
/// bridges the per-input grads back to kt. Lives in kiln-model (not
/// kiln-autograd) because it calls `crate::forward` + `crate::backend`,
/// mirroring [`FlashAttnBackward`].
///
/// # Saved tensors / inputs
///
/// `q`/`k`/`v`/`beta`/`g` + `entry_state` (the recurrent state BEFORE the
/// forward mutated it) as candle clones; `device` reconstructs the
/// backend; `chunk_size` is [`GDN_CHUNK_SIZE`]. 5 differentiable inputs
/// `[q, k, v, beta, g]` in the order the adapter records them;
/// `entry_state` is the initial (zero) state at the SFT layer boundary, so
/// the backward's `grad_exit_state` is `None`.
#[derive(Debug)]
pub(crate) struct GdnRecurrentBackward {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    g: Tensor,
    entry_state: Tensor,
    device: candle_core::Device,
    chunk_size: usize,
}

impl BackwardOp for GdnRecurrentBackward {
    fn name(&self) -> &'static str {
        "gdn_recurrent_backward"
    }
    fn input_count(&self) -> usize {
        // q, k, v, beta, g.
        5
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // Bridge the upstream grad (kt) to candle for the candle composite.
        let grad_c = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(grad_output).map_err(|e| {
            kiln_tensor::Error::Msg(format!("GdnRecurrentBackward: grad kt->candle: {e}"))
        })?;
        let backend = crate::backend::for_device(&self.device);
        let grads = gdn_recurrent_backward_no_grad(
            &*backend,
            &self.q,
            &self.k,
            &self.v,
            &self.beta,
            &self.g,
            &self.entry_state,
            &grad_c,
            None,
            self.chunk_size,
        )
        .map_err(|e| kiln_tensor::Error::Msg(format!("GdnRecurrentBackward: gdn bwd: {e}")))?;
        let to_kt = |t: &Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
            kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(t).map_err(|e| {
                kiln_tensor::Error::Msg(format!("GdnRecurrentBackward: grad candle->kt: {e}"))
            })
        };
        Ok(vec![
            Some(to_kt(&grads.dq)?),
            Some(to_kt(&grads.dk)?),
            Some(to_kt(&grads.dv)?),
            Some(to_kt(&grads.dbeta)?),
            Some(to_kt(&grads.dg)?),
        ])
    }
}

/// Route the GDN recurrence forward through the kt `Tape` so a
/// tape-authoritative backward reaches the GDN-block q/k/v/beta/g
/// projections (and their LoRA `Var`s) — the linear-attention analogue of
/// [`try_tape_flash_attn_cuda`], covering Qwen3.5-4B's 24 GDN layers.
///
/// Runs [`gdn_recurrent_forward_from_parts`] (mutating `recurrent_state`)
/// and records a [`GdnRecurrentBackward`] whose `entry_state` is the
/// snapshot of `recurrent_state` BEFORE the forward. Drop-in for the
/// production recurrence call: `Ok(Some(out))` (with a tape node if a scope
/// is active), or `Ok(None)` (caller runs the recurrence itself) when the
/// gate is off, the inputs aren't CUDA, or a kt borrow fails.
pub fn try_tape_gdn_recurrent_cuda(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    recurrent_state: &mut Tensor,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() || !tape_gdn_enabled() {
        return Ok(None);
    }
    if !matches!(q.device(), candle_core::Device::Cuda(_)) {
        return Ok(None);
    }

    // Snapshot the entry state BEFORE the forward mutates it (the backward
    // needs it), and the device for backend reconstruction in `apply`.
    let entry_state = recurrent_state.clone();
    let device = q.device().clone();

    // Production recurrence forward (mutates recurrent_state in place).
    let out = gdn_recurrent_forward_from_parts(backend, q, k, v, beta, g, recurrent_state)?;

    // kt input ids (chained from upstream adapters where present).
    let q_kt = match tape_kt_input(q) {
        Some(t) => t,
        None => return Ok(Some(out)),
    };
    let k_kt = match tape_kt_input(k) {
        Some(t) => t,
        None => return Ok(Some(out)),
    };
    let v_kt = match tape_kt_input(v) {
        Some(t) => t,
        None => return Ok(Some(out)),
    };
    let beta_kt = match tape_kt_input(beta) {
        Some(t) => t,
        None => return Ok(Some(out)),
    };
    let g_kt = match tape_kt_input(g) {
        Some(t) => t,
        None => return Ok(Some(out)),
    };
    let out_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&out) {
        Ok(t) => t,
        Err(_) => return Ok(Some(out)),
    };

    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            &out_kt,
            &[&q_kt, &k_kt, &v_kt, &beta_kt, &g_kt],
            Box::new(GdnRecurrentBackward {
                q: q.clone(),
                k: k.clone(),
                v: v.clone(),
                beta: beta.clone(),
                g: g.clone(),
                entry_state: entry_state.clone(),
                device: device.clone(),
                chunk_size: GDN_CHUNK_SIZE,
            }),
        );
    });
    if recorded.is_none() {
        // No active tape scope: the forward output is still valid; just no
        // node was recorded.
        return Ok(Some(out));
    }

    kiln_kt_bridge::tape_bridge::register_input_mapping(q_kt.id(), q.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(k_kt.id(), k.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(v_kt.id(), v.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(beta_kt.id(), beta.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(g_kt.id(), g.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// True iff `KILN_USE_TAPE_LORA_ADD` is set to an enable value.
///
/// Separate gate from `KILN_USE_TAPE_FORWARD` so the LoRA add adapter can
/// be rolled out independently of the rest of the tape-forward fleet. The
/// production `add_lora_delta_to_base` dispatches through Marlin-fused
/// CustomOp3 paths today; flipping LoRA to tape recording changes the
/// gradient path (analytic kt backward vs. CustomOp3 backward) so this
/// opt-in is intentionally narrow.
///
/// Cached after first read, matching `tape_forward_enabled()`.
pub fn tape_lora_add_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_LORA_ADD")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(false)
    })
}

// Tape-scope tests live in `kiln-autograd::tape_scope::tests` after
// wave-13 (#1082) promoted the thread-local-tape machinery there. The
// kt-tape adapter tests (`try_tape_{rms_norm,matmul,silu,embedding,swiglu,
// lora_add}_cuda` round-trips) live in the
// `kiln-model/tests/tape_forward_parity.rs` integration test because they
// require the `kiln_kt_bridge` + `kiln_rmsnorm_kernel` cuda surface.

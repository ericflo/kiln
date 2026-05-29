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
    MatmulBackward, MulSigmoidGateBackward, ReshapeBackward, RopeSplitHalfBackward,
    SiluBackward, Tape, TransposeBackward,
};

use crate::backend::BackendRuntime;
use crate::forward::{
    gdn_gated_rms_norm_backward_no_grad, gdn_l2_norm_scale_backward_no_grad,
    gdn_recurrent_backward_no_grad, gdn_recurrent_forward_from_parts,
    sdpa_fallback_backward_no_grad, GDN_CHUNK_SIZE,
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

/// kt-tape backward for the fused "cross-entropy from full logits" loss node
/// ([`try_tape_cross_entropy_from_logits_cuda`]).
///
/// # Why a candle composite (not a kt-native BackwardOp)
///
/// The forward gather inside `cross_entropy_loss` runs through kt ops whose
/// BACKWARD has no CUDA path: kt `index_select`'s backward is a scatter, and kt
/// `scatter_add` / `cast` are CPU-ONLY (`kiln-tensor::ops::scatter_add` downcasts
/// to `CpuStorage`; the byte-conversion `cast` is CPU). Wiring this op via a
/// kt-native `kiln_autograd::BackwardOp` would therefore break on CUDA. Like
/// [`FlashAttnBackward`] / [`GdnRecurrentBackward`], we instead bridge the kt
/// upstream grad to candle, compute `dL/d(full logits)` with the CUDA-safe candle
/// composite [`crate::forward::cross_entropy_from_logits_grad_candle`], and bridge
/// the result back to kt.
///
/// # Saved tensors
///
/// `logits` is the FULL forward logits `[1, T, V]` (a candle clone — an `Arc`
/// bump on the candle storage), plus the host-side `input_ids` / `label_mask` and
/// the `device`. The candle composite recomputes the forward gather + softmax
/// from these (no extra device tensors saved).
///
/// # Gradient
///
/// See [`crate::forward::cross_entropy_from_logits_grad_candle`]: mean reduction
/// (`1/num_active`), the `p - one_hot` per-active-row term scaled by the incoming
/// scalar seed, scattered back to the active shifted rows with a trailing zero row
/// for the dropped `lg[T-1]`. Returned as a single `[1, T, V]` kt grad (input
/// count 1).
#[derive(Debug)]
pub(crate) struct CrossEntropyFromLogitsBackward {
    logits: candle_core::Tensor,
    input_ids: Vec<u32>,
    label_mask: Vec<bool>,
    device: candle_core::Device,
}

impl BackwardOp for CrossEntropyFromLogitsBackward {
    fn name(&self) -> &'static str {
        "cross_entropy_from_logits_backward"
    }
    fn input_count(&self) -> usize {
        // The full logits [1, T, V].
        1
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // The composite recomputes the forward gather from the SAVED `logits`;
        // the tape walker need not re-materialise the input activation.
        false
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // The upstream grad is the scalar dL/dloss seed (typically 1.0). Bridge
        // it to candle and read the scalar so the composite can fold it into the
        // mean-reduction's per-row gradient.
        let grad_c = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(grad_output).map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "CrossEntropyFromLogitsBackward: grad kt->candle: {e}"
            ))
        })?;
        let grad_scalar = grad_c
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| {
                kiln_tensor::Error::Msg(format!(
                    "CrossEntropyFromLogitsBackward: grad scalar read: {e}"
                ))
            })?
            .first()
            .copied()
            .ok_or_else(|| {
                kiln_tensor::Error::Msg(
                    "CrossEntropyFromLogitsBackward: empty grad_output".to_string(),
                )
            })? as f64;

        let grad_logits = crate::forward::cross_entropy_from_logits_grad_candle(
            &self.logits,
            &self.input_ids,
            &self.label_mask,
            grad_scalar,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "CrossEntropyFromLogitsBackward: candle composite: {e}"
            ))
        })?;

        // The composite output is freshly built (cat/unsqueeze → possibly
        // non-contiguous) and owns no kt lifetime — materialise contiguous and
        // COPY into an owned kt tensor (not borrow).
        let grad_logits = grad_logits.contiguous().map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "CrossEntropyFromLogitsBackward: grad contiguous: {e}"
            ))
        })?;
        let grad_logits_kt =
            kiln_kt_bridge::kt_tensor_from_candle_cuda_copy(&grad_logits).map_err(|e| {
                kiln_tensor::Error::Msg(format!(
                    "CrossEntropyFromLogitsBackward: grad candle->kt: {e}"
                ))
            })?;

        Ok(vec![Some(grad_logits_kt)])
    }
}

/// Attempt to route the WHOLE next-token cross-entropy loss — taking the FULL
/// model logits `[1, T, V]` and producing the scalar loss — through the kt
/// `Tape` as a SINGLE fused node, instead of the four un-taped candle ops
/// (`squeeze(0)` → `narrow(0, 0, T-1)` → `index_select` → `to_dtype(F32)`) that
/// precede `try_tape_cross_entropy_cuda` in `cross_entropy_loss`.
///
/// # Why this exists (#1082 CP-4 Increment 1)
///
/// In the tape-authoritative SFT path the loss is the backward ROOT. The prior
/// loss adapter ([`try_tape_cross_entropy_cuda`]) takes `active_logits_f32` —
/// produced by those four un-taped ops — so the root's input was a fresh-borrow
/// island and the chain died one op below the loss (`tape_has_grad=0/50`). This
/// node takes the full logits directly, so `dL/d(logits)` reaches the lm_head
/// output (once that op is tape-wired) and the chain stays connected.
///
/// The four gather ops sit on kt-CUDA gaps (kt `index_select` backward is a
/// scatter; kt `scatter_add` / `cast` are CPU-only), so they CANNOT be wired as
/// kt-native ops on CUDA. The fused node sidesteps every gap: its backward is the
/// CUDA-safe candle composite [`CrossEntropyFromLogitsBackward`].
///
/// # Returns
///
/// * `Ok(Some(loss))` — the tape-forward path ran: a DETACHED, lineage-free
///   candle scalar loss (a fresh kt -> candle CUDA copy, numerically identical to
///   `cross_entropy_loss`'s candle baseline) with a
///   [`CrossEntropyFromLogitsBackward`] node recorded on the active tape, IO-mapped
///   into the bridge. The loss is detached unconditionally so the
///   tape-authoritative caller's `loss.backward()` is always `{loss: ones}` and the
///   recorded node is the sole backward root.
/// * `Ok(None)` — the gate is off, `logits` isn't a CUDA rank-3 `[1, T, V]`, no
///   tape scope is active, or a kt borrow failed. The caller falls through to the
///   existing candle loss composite (which still calls the non-authoritative
///   path).
/// * `Err(...)` — an unexpected forward or kt -> candle copy-back failure.
pub fn try_tape_cross_entropy_from_logits_cuda(
    logits: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &candle_core::Device,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // Full model logits only: [1, T, V] on CUDA. Defer any other shape/device to
    // the caller's candle composite.
    if logits.rank() != 3 || !matches!(logits.device(), candle_core::Device::Cuda(_)) {
        return Ok(None);
    }
    let Ok((b, t, _v)) = logits.dims3() else {
        return Ok(None);
    };
    if b != 1 || t != input_ids.len() || label_mask.len() != input_ids.len() {
        return Ok(None);
    }

    // kt input — thread the lm_head adapter's output so the tape stays connected
    // (today, before lm_head is wired, this falls to a fresh borrow — fine).
    let logits_kt = match tape_kt_input(logits) {
        Some(t) => t,
        None => return Ok(None),
    };

    // Compute the scalar loss FORWARD via candle, replicating
    // `cross_entropy_loss`'s math (trainer.rs:5929-5995) EXACTLY so the value is
    // identical to the candle baseline.
    let seq_len = input_ids.len();
    let lg = logits.squeeze(0)?; // [T, V]
    let shift_logits = lg.narrow(0, 0, seq_len - 1)?; // [T-1, V]
    let shift_labels: Vec<u32> = input_ids[1..].to_vec();
    let shift_mask: Vec<bool> = label_mask[1..].to_vec();
    let active_positions: Vec<usize> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();
    if active_positions.is_empty() {
        // No supervised positions — let the caller's composite raise the same
        // error it would have raised, rather than bailing here.
        return Ok(None);
    }
    let indices = Tensor::new(
        active_positions
            .iter()
            .map(|&i| i as u32)
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    )?;
    let active_logits = shift_logits.index_select(&indices, 0)?; // [A, V]
    let active_labels: Vec<u32> = active_positions.iter().map(|&i| shift_labels[i]).collect();
    let labels_tensor =
        Tensor::new(active_labels.as_slice(), device)?.to_dtype(candle_core::DType::U32)?;
    let active_logits_f32 = active_logits.to_dtype(candle_core::DType::F32)?;
    let log_sum_exp = active_logits_f32.log_sum_exp(candle_core::D::Minus1)?; // [A]
    let labels_2d = labels_tensor.unsqueeze(1)?; // [A, 1]
    let correct_logits = active_logits_f32
        .gather(&labels_2d.to_dtype(candle_core::DType::U32)?, 1)?
        .squeeze(1)?; // [A]
    let per_token_loss = (log_sum_exp - correct_logits)?;
    let loss = per_token_loss.mean_all()?; // scalar

    // Record the node: the OUTPUT must be an OWNED kt copy of the loss. A borrow
    // (`kt_tensor_from_candle_cuda_borrow`) yields a raw device pointer into the
    // candle `loss`'s storage WITHOUT keeping that storage alive, but `loss` is a
    // local that drops when this adapter returns — while the tape is walked much
    // later in `standard_forward_backward_tape_authoritative`. That left the
    // recorded output kt tensor dangling (use-after-free). Use an OWNED device
    // copy instead so the node's output has independent storage regardless of when
    // the candle `loss` drops. The copy is a scalar (negligible). Record
    // `CrossEntropyFromLogitsBackward` with the full logits + host-side gather
    // metadata as the saved state.
    let loss_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let loss_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_copy(&loss)
            .map_err(|e| anyhow::anyhow!("loss kt copy: {e}"))?;
        tape.record(
            &loss_kt,
            &[&logits_kt],
            Box::new(CrossEntropyFromLogitsBackward {
                logits: logits.clone(),
                input_ids: input_ids.to_vec(),
                label_mask: label_mask.to_vec(),
                device: device.clone(),
            }),
        );
        Ok(loss_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };
    let loss_kt = loss_kt
        .context("tape_forward::try_tape_cross_entropy_from_logits_cuda: kt-tape forward failed")?;

    // CP-4 (#1082) Increment 1: return a DETACHED, lineage-free loss by
    // construction — exactly like `try_tape_cross_entropy_cuda` (above). The
    // candle `loss` computed for the forward value still carries candle
    // autograd lineage; if returned, `standard_forward_backward_tape_authoritative`'s
    // `loss.backward()` could let candle's autograd silently fill in LoRA-Var
    // grads, a false positive that defeats the `[CP4-COVERAGE] tape_has_grad`
    // measurement. Copy `loss_kt` back to a FRESH detached CUDA leaf so
    // `loss.backward()` is unconditionally `{loss: ones}` (independent of the
    // forward's lineage state) and the recorded `CrossEntropyFromLogitsBackward`
    // node is the sole tape root that seeds dL/dloss=1. The kt copy-back is
    // value-identical, so the caller's `loss_val` is unchanged.
    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&loss_kt).context(
        "tape_forward::try_tape_cross_entropy_from_logits_cuda: kt -> candle copy failed",
    )?;

    // CP-4 (#1082) tape_bridge: only `logits` is differentiable. Map the
    // output/retain onto the RETURNED detached copy's id (`out.id()`), mirroring
    // `try_tape_cross_entropy_cuda`'s `out_kt.id() -> out.id()` mapping.
    kiln_kt_bridge::tape_bridge::register_input_mapping(logits_kt.id(), logits.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(loss_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&loss_kt, out.id());

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

/// Attempt to run the FULL base projection (and optional fused LoRA delta)
/// through the kt-typed op surface as ONE chained group of `Tape` nodes:
/// `reshape → matmul → [lora_delta_add] → reshape`. Records the backward so
/// `dL/dx` flows through BOTH the frozen base weight AND the LoRA path, and
/// `proj.a` / `proj.b` receive grads.
///
/// The forward computes (matching the existing CUDA dispatch bit-for-bit):
/// ```text
/// out = base + scale * (x @ A^T @ B^T)        // when `lora` is Some
/// out = base = x @ W^T                          // when `lora` is None (lm_head)
/// ```
///
/// # Why a SINGLE fused adapter (base + LoRA together)
///
/// CP-4 (#1082) Increment 2 — the keystone. The tape root
/// ([`try_tape_cross_entropy_from_logits_cuda`], Increment 1) connects to the
/// lm_head output, but the lm_head matmul and every q/k/v/o/gate/up/down/GDN
/// projection were unwired, so nothing below the loss got grads
/// (`tape_has_grad=0/50`). In the authoritative path every intermediate is a
/// DETACHED kt-copy (`track_op()==false`), so neither
/// `lm_head_forward_backend_decode_if` nor
/// `linear_with_lora_t_backend_decode_if` reliably hits `linear_prefill_apply`
/// (both gate the autograd-safe branch on `x.track_op()`). This adapter is
/// therefore wired at the TOP of those two functions, before the backend
/// dispatch.
///
/// Folding base + LoRA into one chained group (instead of routing the base
/// matmul and the LoRA delta-add through two separate adapters across a
/// reshape boundary) keeps `x2d` and `base2d` as SHARED kt ids: `base2d` is
/// the matmul node's output (so `dL/dbase2d` flows into the matmul backward),
/// and `x2d` is shared between the matmul node and the LoRA node (so `dL/dx2d`
/// accumulates BOTH the base-weight and LoRA contributions — the correct full
/// `dL/dx`). Splitting them would mint a fresh kt borrow at the reshape and
/// fragment the chain.
///
/// # Node-recording sequence (inside ONE `with_active_tape`)
///
/// 1. `ReshapeBackward { input_shape: x_kt.shape() }` — output `x2d [rows, k]`,
///    input `x_kt`.
/// 2. `MatmulBackward { a: x2d, b: w_kt }` — output `base2d [rows, n]`, inputs
///    `[x2d, w_kt]`.
/// 3. (lora only) `LoraDeltaAddBackward { x: x2d, a, b, scale }` — output
///    `out2d`, inputs `[base2d, x2d, a_kt, b_kt]` (same order as
///    [`try_tape_lora_add_cuda`]).
/// 4. `ReshapeBackward { input_shape: [rows, n] }` — output `out_kt`, input
///    `out2d` (or `base2d` when `lora` is None).
///
/// # Returns
///
/// * `Ok(Some(out))` — the tape-forward path ran: a candle copy of the kt
///   output, reshaped to `x.dims[..-1] ++ [n]`, with the chained group recorded
///   on the active tape and IO-mapped into the bridge (`x` → `x_kt`, and the
///   LoRA Vars `proj.a`/`proj.b` → `a_kt`/`b_kt` when present).
/// * `Ok(None)` — gate off (no tape, `KILN_USE_TAPE_FORWARD` off,
///   `KILN_USE_TAPE_LORA_ADD` off), or device / dtype / shape / contiguity
///   preconditions fail. The caller falls through to the existing dispatch.
/// * `Err(...)` — an unexpected kt forward or kt → candle copy-back failure.
#[allow(clippy::too_many_lines)]
pub fn try_tape_lora_linear_cuda(
    x: &Tensor,
    weight_t: &Tensor,
    lora: Option<&LoraProjectionWeights>,
    lora_scale: f32,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() || !tape_lora_add_enabled() {
        return Ok(None);
    }

    // Device gate: CUDA-only (the bridge's `kt_tensor_from_candle_cuda_*`
    // helpers are CUDA-only). Match the existing tape adapters.
    if !matches!(x.device(), candle_core::Device::Cuda(_))
        || !matches!(weight_t.device(), candle_core::Device::Cuda(_))
    {
        return Ok(None);
    }
    // Dtype gate: only BF16 / F32 today, and all matching (kt matmul requires
    // matching dtypes throughout the composed forward).
    if !matches!(
        x.dtype(),
        candle_core::DType::BF16 | candle_core::DType::F32
    ) {
        return Ok(None);
    }
    if weight_t.dtype() != x.dtype() {
        return Ok(None);
    }

    // Shape gate: weight_t must be rank-2 `[K, N]` with x's last dim == K.
    let Ok((wk, n)) = weight_t.dims2() else {
        return Ok(None);
    };
    let x_dims = x.dims().to_vec();
    if x_dims.len() < 2 {
        return Ok(None);
    }
    let k = *x_dims.last().unwrap();
    if k != wk {
        return Ok(None);
    }
    let rows: usize = x_dims[..x_dims.len() - 1].iter().product();
    if rows == 0 {
        return Ok(None);
    }

    // LoRA gate (when present): A/B on CUDA, dtype-matching, shape-consistent,
    // contiguous. Any mismatch falls through to the existing dispatch.
    if let Some(proj) = lora {
        if !matches!(proj.a.device(), candle_core::Device::Cuda(_))
            || !matches!(proj.b.device(), candle_core::Device::Cuda(_))
        {
            return Ok(None);
        }
        if proj.a.dtype() != x.dtype() || proj.b.dtype() != x.dtype() {
            return Ok(None);
        }
        let Ok((rank, a_in)) = proj.a.dims2() else {
            return Ok(None);
        };
        let Ok((b_out, b_rank)) = proj.b.dims2() else {
            return Ok(None);
        };
        if a_in != k || b_out != n || b_rank != rank {
            return Ok(None);
        }
        if !proj.a.is_contiguous() || !proj.b.is_contiguous() {
            return Ok(None);
        }
    }

    // kt input — thread the kt id from an upstream tape adapter's output so
    // the tape stays connected (e.g. lm_head's `x` came from the final norm
    // adapter). Falls back to a fresh borrow otherwise.
    let x_kt = match tape_kt_input(x) {
        Some(t) => t,
        None => return Ok(None),
    };
    // Frozen base weight: a fresh zero-copy borrow is correct (no chaining).
    let w_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weight_t) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    // LoRA Var borrows (when present) — fresh zero-copy views, IO-mapped onto
    // the candle Var ids below so the optimiser sees their grads.
    let (a_kt, b_kt) = match lora {
        Some(proj) => {
            let a = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&proj.a) {
                Ok(t) => t,
                Err(_) => return Ok(None),
            };
            let b = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&proj.b) {
                Ok(t) => t,
                Err(_) => return Ok(None),
            };
            (Some(a), Some(b))
        }
        None => (None, None),
    };

    // Record on the active tape (if any). Outside a scope, fall through.
    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        // 1) Flatten x's lead dims to 2-D: `x_kt [..., k] -> x2d [rows, k]`.
        //    kt reshape requires a contiguous source; materialise defensively.
        let x_kt_c = if x_kt.is_contiguous() {
            x_kt.clone()
        } else {
            x_kt
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt x.contiguous: {e}"))?
        };
        let x2d = x_kt_c
            .reshape(vec![rows, k])
            .map_err(|e| anyhow::anyhow!("kt x reshape -> 2d: {e}"))?;
        tape.record(
            &x2d,
            &[&x_kt],
            Box::new(ReshapeBackward {
                input_shape: x_kt.shape().to_vec(),
            }),
        );

        // 2) base2d = x2d @ W^T-less base: `x2d [rows, k] @ w_kt [k, n]`.
        let base2d = kiln_tensor::ops::matmul(&x2d, &w_kt)
            .map_err(|e| anyhow::anyhow!("kt matmul x2d@w: {e}"))?;
        tape.record(
            &base2d,
            &[&x2d, &w_kt],
            Box::new(MatmulBackward {
                a: x2d.clone(),
                b: w_kt.clone(),
            }),
        );

        // 3) Fuse the LoRA delta + add (mirrors `try_tape_lora_add_cuda`):
        //    h = x2d @ A^T ; d = h @ B^T ; delta = d * scale ; out2d = base2d + delta.
        //    ONE `LoraDeltaAddBackward` node with `base2d` as input 0 so its
        //    grad flows into the matmul backward, and `x2d` shared so dL/dx2d
        //    accumulates base + LoRA.
        let out2d = match (lora, a_kt.as_ref(), b_kt.as_ref()) {
            (Some(_proj), Some(a_kt), Some(b_kt)) => {
                let a_t_kt = a_kt
                    .transpose(0, 1)
                    .map_err(|e| anyhow::anyhow!("kt a.transpose: {e}"))?
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("kt a_t.contiguous: {e}"))?;
                let h_kt = kiln_tensor::ops::matmul(&x2d, &a_t_kt)
                    .map_err(|e| anyhow::anyhow!("kt matmul x@a_t: {e}"))?;
                let b_t_kt = b_kt
                    .transpose(0, 1)
                    .map_err(|e| anyhow::anyhow!("kt b.transpose: {e}"))?
                    .contiguous()
                    .map_err(|e| anyhow::anyhow!("kt b_t.contiguous: {e}"))?;
                let d_kt = kiln_tensor::ops::matmul(&h_kt, &b_t_kt)
                    .map_err(|e| anyhow::anyhow!("kt matmul h@b_t: {e}"))?;
                let delta_kt = kiln_tensor::ops::mul_scalar(&d_kt, lora_scale)
                    .map_err(|e| anyhow::anyhow!("kt mul_scalar(scale): {e}"))?;
                let out2d = kiln_tensor::ops::add(&base2d, &delta_kt)
                    .map_err(|e| anyhow::anyhow!("kt add(base, delta): {e}"))?;
                tape.record(
                    &out2d,
                    &[&base2d, &x2d, a_kt, b_kt],
                    Box::new(LoraDeltaAddBackward {
                        x: x2d.clone(),
                        a: a_kt.clone(),
                        b: b_kt.clone(),
                        scale: lora_scale,
                    }),
                );
                out2d
            }
            // No LoRA (e.g. lm_head): the base matmul output IS the projection.
            _ => base2d,
        };

        // 4) Reshape back to `x.dims[..-1] ++ [n]`.
        let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
        out_shape.push(n);
        let out2d_c = if out2d.is_contiguous() {
            out2d.clone()
        } else {
            out2d
                .contiguous()
                .map_err(|e| anyhow::anyhow!("kt out2d.contiguous: {e}"))?
        };
        let out_kt = out2d_c
            .reshape(out_shape)
            .map_err(|e| anyhow::anyhow!("kt out reshape -> nd: {e}"))?;
        tape.record(
            &out_kt,
            &[&out2d],
            Box::new(ReshapeBackward {
                input_shape: vec![rows, n],
            }),
        );

        Ok(out_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt =
        out_kt.context("tape_forward::try_tape_lora_linear_cuda: kt-tape forward failed")?;
    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_lora_linear_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: register IO mappings. `x` chains upstream;
    // `proj.a`/`proj.b` are the differentiable LoRA Vars the optimiser cares
    // about (exactly as `try_tape_lora_add_cuda`). The frozen base weight is
    // not registered (no grad consumer for it).
    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x.id());
    if let (Some(proj), Some(a_kt), Some(b_kt)) = (lora, a_kt.as_ref(), b_kt.as_ref()) {
        kiln_kt_bridge::tape_bridge::register_input_mapping(a_kt.id(), proj.a.id());
        kiln_kt_bridge::tape_bridge::register_input_mapping(b_kt.id(), proj.b.id());
    }
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
///
/// # Output layout (`head_last_output`)
///
/// The production GDN recurrence dispatch returns the attention output in
/// **head-LAST `[B, T, nv, dv]`** layout on the CUDA prefill /
/// full-chunk paths (`gdn_recurrent_prefill_head_last` /
/// `gdn_chunkwise_recurrence_head_last_full_chunks`) and **head-FIRST
/// `[B, nv, T, dv]`** only on the chunkwise fallback
/// (`gdn_chunkwise_recurrence`). [`gdn_recurrent_backward_no_grad`] always
/// expects a **head-FIRST** grad (its internal `grad_out.narrow(2, …)`
/// indexes the seq axis at dim 2). `head_last_output` records which layout
/// the recorded forward output used so `apply` can transpose a head-LAST
/// upstream grad back to head-FIRST before invoking the backward. The
/// saved `q`/`k`/`v`/`beta`/`g`/`entry_state` are ALWAYS head-first (they
/// are the post-`recur_prep`-transpose recurrence inputs), so the returned
/// `dq`/…/`dg` are head-first and match the head-first input `Var`s
/// regardless of `head_last_output`.
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
    /// `true` when the recorded forward output was head-LAST
    /// `[B, T, nv, dv]`; `apply` then transposes the upstream grad to the
    /// head-FIRST `[B, nv, T, dv]` layout the backward requires.
    head_last_output: bool,
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
        // `gdn_recurrent_backward_no_grad` indexes the seq axis at dim 2
        // (head-FIRST). When the recorded forward output was head-LAST
        // `[B, T, nv, dv]`, the upstream grad arrives head-last too, so
        // transpose it back to head-FIRST `[B, nv, T, dv]` before the
        // backward. The saved q/k/v/beta/g/entry_state are already
        // head-first, so the returned grads stay head-first (no transpose
        // on the way out).
        let grad_c = if self.head_last_output {
            grad_c
                .transpose(1, 2)
                .map_err(|e| {
                    kiln_tensor::Error::Msg(format!(
                        "GdnRecurrentBackward: head-last grad transpose: {e}"
                    ))
                })?
                .contiguous()
                .map_err(|e| {
                    kiln_tensor::Error::Msg(format!(
                        "GdnRecurrentBackward: head-last grad contiguous: {e}"
                    ))
                })?
        } else {
            grad_c
        };
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
            // gdn_recurrent_backward_no_grad can return non-contiguous grads
            // (internal transposes/narrows). Materialise contiguous, then
            // COPY into an owned kt tensor (no keep-alive lifetime tie to a
            // local candle temporary).
            let tc = t.contiguous().map_err(|e| {
                kiln_tensor::Error::Msg(format!("GdnRecurrentBackward: grad contiguous: {e}"))
            })?;
            kiln_kt_bridge::kt_tensor_from_candle_cuda_copy(&tc).map_err(|e| {
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
    // `gdn_recurrent_forward_from_parts` returns the recurrence output in
    // head-FIRST `[B, nv, T, dv]` layout (the short-seq chunkwise path),
    // hence `head_last = false` below.
    let out = gdn_recurrent_forward_from_parts(backend, q, k, v, beta, g, recurrent_state)?;

    // Record the node (no-op unless a tape scope is active). The forward
    // already ran above, so this only records the backward and registers
    // the IO mappings.
    tape_record_gdn_recurrent(&out, false, q, k, v, beta, g, &entry_state, &device)?;

    Ok(Some(out))
}

/// Record a [`GdnRecurrentBackward`] node for a GDN recurrence output that
/// the PRODUCTION forward has ALREADY computed (via its chunk-scan
/// kernels), WITHOUT re-running the recurrence.
///
/// This is the wiring entry point used by
/// `forward::gated_deltanet_forward_decode_if`: the prefill/training path
/// computes the recurrence output itself (through
/// `gdn_recurrent_prefill_head_last` /
/// `gdn_chunkwise_recurrence_head_last_full_chunks` /
/// `gdn_chunkwise_recurrence`) and then calls this to attach the tape node.
/// Unlike [`try_tape_gdn_recurrent_cuda`] (which re-runs the recurrence via
/// [`gdn_recurrent_forward_from_parts`] for the per-op parity tests), this
/// adapter takes the already-computed `out` and only records.
///
/// # Arguments
///
/// * `out` — the recurrence output the production forward produced. Its
///   layout is described by `head_last`.
/// * `head_last` — `true` when `out` is head-LAST `[B, T, nv, dv]` (CUDA
///   prefill / full-chunk paths), `false` when head-FIRST `[B, nv, T, dv]`
///   (chunkwise fallback). Stored on the recorded
///   [`GdnRecurrentBackward`] so its `apply` can transpose a head-last grad
///   back to head-first before the head-first-only backward.
/// * `q`/`k`/`v`/`beta`/`g` — the head-FIRST recurrence inputs (post
///   `recur_prep` transpose). They feed the head-first backward, so the
///   returned grads are head-first regardless of `head_last`.
/// * `entry_state` — the recurrent state BEFORE the forward mutated it.
/// * `device` — for backend reconstruction in `apply`.
///
/// Gated on `tape_forward_enabled() && tape_gdn_enabled()` and a CUDA `q`.
/// A no-op (returns `Ok(())`) when the gate is off, the inputs aren't CUDA,
/// no tape scope is active, or any kt borrow fails — the production forward
/// output is unaffected either way.
#[allow(clippy::too_many_arguments)]
pub fn tape_record_gdn_recurrent(
    out: &Tensor,
    head_last: bool,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    entry_state: &Tensor,
    device: &candle_core::Device,
) -> Result<()> {
    if !tape_forward_enabled() || !tape_gdn_enabled() {
        return Ok(());
    }
    if !matches!(q.device(), candle_core::Device::Cuda(_)) {
        return Ok(());
    }

    // kt input ids (chained from upstream adapters where present). A kt
    // borrow failure means we cannot connect this op to the tape — skip
    // recording cleanly (the production output is still valid).
    let q_kt = match tape_kt_input(q) {
        Some(t) => t,
        None => return Ok(()),
    };
    let k_kt = match tape_kt_input(k) {
        Some(t) => t,
        None => return Ok(()),
    };
    let v_kt = match tape_kt_input(v) {
        Some(t) => t,
        None => return Ok(()),
    };
    let beta_kt = match tape_kt_input(beta) {
        Some(t) => t,
        None => return Ok(()),
    };
    let g_kt = match tape_kt_input(g) {
        Some(t) => t,
        None => return Ok(()),
    };
    let out_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(out) {
        Ok(t) => t,
        Err(_) => return Ok(()),
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
                head_last_output: head_last,
            }),
        );
    });
    if recorded.is_none() {
        // No active tape scope: nothing to record. The production output is
        // unaffected.
        return Ok(());
    }

    kiln_kt_bridge::tape_bridge::register_input_mapping(q_kt.id(), q.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(k_kt.id(), k.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(v_kt.id(), v.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(beta_kt.id(), beta.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(g_kt.id(), g.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(())
}

// ===========================================================================
// CP-4 (#1082): GDN surrounding ops — conv1d / L2-qk-norm / gated-RMSNorm.
//
// The GDN recurrence is tape-wired above. For a tape-authoritative backward to
// reach the GDN-block in_proj / out_proj LoRA Vars, EVERY op between the
// projection matmuls and the recurrence must ALSO record onto the kt Tape, or
// the chain fragments. These three adapters cover:
//
//   * `try_tape_causal_conv1d_cuda`     — Step 2 depthwise conv on mixed-QKV.
//   * `try_tape_gdn_l2_norm_scale_cuda` — Step 4/5 L2 qk-norm (q scaled, k not).
//   * `try_tape_gdn_gated_rms_norm_cuda`— Step 8 gated RMSNorm before out_proj.
//
// Plus `try_tape_transpose_cuda` (the head-FIRST→head-LAST chaining-gap fix at
// `forward.rs:gated_deltanet_forward_decode_if`'s `attn_out.transpose(1,2)`).
//
// Each has a narrow `KILN_USE_TAPE_GDN_*` gate (mirroring `tape_gdn_enabled`),
// is a no-op by default, and falls through cleanly so the production forward is
// untouched with the gate off.
// ===========================================================================

/// True iff `KILN_USE_TAPE_GDN_CONV` is set to an enable value. Narrow opt-in
/// for the GDN causal-depthwise-conv1d tape adapter. Cached after first read.
pub fn tape_gdn_conv_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_GDN_CONV")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(false)
    })
}

/// True iff `KILN_USE_TAPE_GDN_QK_NORM` is set to an enable value. Narrow
/// opt-in for the GDN L2-qk-norm tape adapter. Cached after first read.
pub fn tape_gdn_qk_norm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_GDN_QK_NORM")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(false)
    })
}

/// True iff `KILN_USE_TAPE_GDN_GATED_NORM` is set to an enable value. Narrow
/// opt-in for the GDN gated-RMSNorm tape adapter. Cached after first read.
pub fn tape_gdn_gated_norm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_GDN_GATED_NORM")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(false)
    })
}

/// CUDA-native tape backward for the GDN causal depthwise conv1d w.r.t. its
/// INPUT only (the conv `weight` is a frozen base tensor in LoRA training —
/// `weights.conv1d` is NOT a `Var` — so only the input gradient is needed to
/// keep the chain connected back through the in_proj LoRA Vars).
///
/// # Why a CUDA kt `BackwardOp` (not a candle composite)
///
/// `kiln_rmsnorm_kernel::causal_depthwise_conv1d_bwd_input_kt` is a fused CUDA
/// kernel with no device-agnostic composite. Like [`FlashAttnBackward`], the op
/// dispatching it lives in `kiln-model` (which depends on the kernel crate) and
/// implements `kiln_autograd::BackwardOp` over kt tensors directly. The kernel
/// runs in F32 on `[rows, channels]` operands.
///
/// # Saved tensors / inputs
///
/// `weight` (`[channels, kernel]`, F32) is the only saved state; one
/// differentiable input (`input`). The backward needs neither the input nor the
/// conv state (input-grad depends only on `grad_out` and `weight` —
/// `kiln_causal_depthwise_conv1d_bwd_input_f32`).
#[derive(Debug)]
pub(crate) struct CausalConv1dInputBackward {
    /// Saved F32 CUDA conv weight `[channels, kernel]`.
    weight: kiln_tensor::Tensor,
    kernel: usize,
}

impl BackwardOp for CausalConv1dInputBackward {
    fn name(&self) -> &'static str {
        "gdn_causal_conv1d_input_backward"
    }
    fn input_count(&self) -> usize {
        // input only — weight is frozen, state is non-differentiable.
        1
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        use kiln_tensor::DType as KtDType;

        // The kernel wants a contiguous F32 `[rows, channels]` grad.
        let go = if grad_output.dtype() == KtDType::F32 {
            grad_output.clone()
        } else {
            kiln_tensor::ops::cast(grad_output, KtDType::F32)?
        };
        let go = if go.is_contiguous() { go } else { go.contiguous()? };

        let gi =
            kiln_rmsnorm_kernel::causal_depthwise_conv1d_bwd_input_kt(&go, &self.weight, self.kernel)
                .map_err(|e| {
                    kiln_tensor::Error::Msg(format!(
                        "CausalConv1dInputBackward: bwd_input_kt: {e}"
                    ))
                })?;
        Ok(vec![Some(gi)])
    }
}

/// Route the GDN causal depthwise conv1d forward through the kt `Tape`.
///
/// Operates on the kernel's native `[rows, channels]` F32 layout (`rows =
/// batch*time`): `input` `[rows, channels]`, `weight` `[channels, kernel]`,
/// `state` `[channels, kernel-1]`. Returns the conv output `[rows, channels]`
/// (the caller applies SiLU + the head split downstream — those record via
/// `try_tape_silu_cuda` / reshape adapters). Records a
/// [`CausalConv1dInputBackward`] so a tape-authoritative backward flows the conv
/// output grad back to its input (and thus the in_proj LoRA Vars).
///
/// `Ok(None)` (caller runs the production conv itself) when the gate is off,
/// no tape scope is active, the inputs aren't CUDA/F32/contiguous/rank-2, or a
/// kt borrow fails.
pub fn try_tape_causal_conv1d_cuda(
    input: &Tensor,
    weight: &Tensor,
    state: &Tensor,
    kernel: usize,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() || !tape_gdn_conv_enabled() {
        return Ok(None);
    }
    // kernel + layout envelope: the kt conv kernel is F32, rank-2
    // `[rows, channels]`, with `weight == [channels, kernel]` and
    // `state == [channels, kernel-1]`.
    if !matches!(input.device(), candle_core::Device::Cuda(_))
        || input.dtype() != candle_core::DType::F32
        || weight.dtype() != candle_core::DType::F32
        || state.dtype() != candle_core::DType::F32
        || input.rank() != 2
        || weight.rank() != 2
        || state.rank() != 2
        || kernel < 2
        || !input.is_contiguous()
        || !weight.is_contiguous()
        || !state.is_contiguous()
    {
        return Ok(None);
    }
    let (_rows, channels) = match input.dims2() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    // Rank already checked == 2 above; compare dims as slices (candle's Error
    // is not PartialEq, so avoid comparing the Result directly).
    if weight.dims() != [channels, kernel].as_slice()
        || state.dims() != [channels, kernel - 1].as_slice()
    {
        return Ok(None);
    }

    // kt inputs — thread the upstream adapter output (the in_proj LoRA / matmul
    // node) so the tape stays connected.
    let input_kt = match tape_kt_input(input) {
        Some(t) => t,
        None => return Ok(None),
    };
    let weight_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weight) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let state_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(state) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let y = kiln_rmsnorm_kernel::causal_depthwise_conv1d_kt(
            &input_kt,
            &weight_kt,
            &state_kt,
            kernel,
        )
        .map_err(|e| anyhow::anyhow!("kt causal_depthwise_conv1d: {e}"))?;
        tape.record(
            &y,
            &[&input_kt],
            Box::new(CausalConv1dInputBackward {
                weight: weight_kt.clone(),
                kernel,
            }),
        );
        Ok(y)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt =
        out_kt.context("tape_forward::try_tape_causal_conv1d_cuda: kt-tape forward failed")?;
    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_causal_conv1d_cuda: kt -> candle copy failed")?;

    // Only `input` is differentiable (weight frozen, state non-differentiable).
    kiln_kt_bridge::tape_bridge::register_input_mapping(input_kt.id(), input.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

/// Candle-composite tape backward for the GDN L2-qk-norm `y =
/// l2_normalize(x) * scale`. Wraps [`gdn_l2_norm_scale_backward_no_grad`]
/// (analytic adjoint in candle F32), mirroring how [`GdnRecurrentBackward`]
/// wraps `gdn_recurrent_backward_no_grad`.
///
/// One differentiable input (`x`); `scale` is a non-differentiable constant
/// folded into the adjoint (Q uses `1/sqrt(dk)`, K uses `1.0`). `eps` matches
/// `l2_normalize`'s hard-coded `1e-6`.
#[derive(Debug)]
pub(crate) struct GdnL2NormScaleBackward {
    x: Tensor,
    scale: f64,
    eps: f64,
}

impl BackwardOp for GdnL2NormScaleBackward {
    fn name(&self) -> &'static str {
        "gdn_l2_norm_scale_backward"
    }
    fn input_count(&self) -> usize {
        1
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let grad_c = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(grad_output).map_err(|e| {
            kiln_tensor::Error::Msg(format!("GdnL2NormScaleBackward: grad kt->candle: {e}"))
        })?;
        let dx = gdn_l2_norm_scale_backward_no_grad(&self.x, self.scale, self.eps, &grad_c)
            .map_err(|e| {
                kiln_tensor::Error::Msg(format!("GdnL2NormScaleBackward: bwd: {e}"))
            })?;
        // Adjoint can be non-contiguous (broadcast_mul views) — contiguify then
        // COPY to an owned kt tensor (cf. the GdnRecurrentBackward non-contig fix).
        let dx = dx
            .contiguous()
            .map_err(|e| kiln_tensor::Error::Msg(format!("GdnL2NormScaleBackward: contig: {e}")))?;
        let dx_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_copy(&dx).map_err(|e| {
            kiln_tensor::Error::Msg(format!("GdnL2NormScaleBackward: dx candle->kt: {e}"))
        })?;
        Ok(vec![Some(dx_kt)])
    }
}

/// Route a GDN L2-qk-norm `y = l2_normalize(x) * scale` through the kt `Tape`.
///
/// Used for BOTH halves of `gdn_qk_norm`: Q (`scale = 1/sqrt(dk)`) and K
/// (`scale = 1.0`). The forward runs candle `l2_normalize`-then-scale via the
/// existing production helpers; the recorded [`GdnL2NormScaleBackward`] emits
/// the per-input grad so a tape-authoritative backward reaches the conv / split
/// (and thence the in_proj LoRA Vars).
///
/// `Ok(None)` (caller falls through to the existing `gdn_qk_norm`) when the gate
/// is off, no tape scope is active, the input isn't CUDA, or a kt borrow fails.
pub fn try_tape_gdn_l2_norm_scale_cuda(
    x: &Tensor,
    scale: f64,
    out: &Tensor,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() || !tape_gdn_qk_norm_enabled() {
        return Ok(None);
    }
    if !matches!(x.device(), candle_core::Device::Cuda(_)) {
        return Ok(None);
    }
    if x.dims() != out.dims() {
        // The L2-norm-scale forward is shape-preserving; defer anything else.
        return Ok(None);
    }

    let x_kt = match tape_kt_input(x) {
        Some(t) => t,
        None => return Ok(None),
    };
    // The production forward already computed `out` (q_out / k_out); borrow it
    // as the recorded node's output so we record-only (no re-run), like
    // `tape_record_gdn_recurrent`.
    let out_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(out) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            &out_kt,
            &[&x_kt],
            Box::new(GdnL2NormScaleBackward {
                x: x.clone(),
                scale,
                eps: 1e-6,
            }),
        );
    });
    if recorded.is_none() {
        return Ok(None);
    }

    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out.clone()))
}

/// Candle-composite tape backward for the GDN gated RMSNorm `out =
/// rms_norm(x, weight) * silu(z)`. Wraps
/// [`gdn_gated_rms_norm_backward_no_grad`] (analytic adjoint in candle F32),
/// mirroring how [`GdnRecurrentBackward`] wraps `gdn_recurrent_backward_no_grad`.
///
/// Three differentiable inputs `[x, z, weight]` in that order. `x` is the
/// recurrence output (head-LAST `[B, T, nv, dv]`), `z` is the output gate,
/// `weight` is the GDN `norm` `Var`.
#[derive(Debug)]
pub(crate) struct GdnGatedRmsNormBackward {
    x: Tensor,
    z: Tensor,
    weight: Tensor,
    eps: f64,
}

impl BackwardOp for GdnGatedRmsNormBackward {
    fn name(&self) -> &'static str {
        "gdn_gated_rms_norm_backward"
    }
    fn input_count(&self) -> usize {
        // x, z, weight.
        3
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let grad_c = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(grad_output).map_err(|e| {
            kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward: grad kt->candle: {e}"))
        })?;
        let grads = gdn_gated_rms_norm_backward_no_grad(
            &self.x,
            &self.z,
            &self.weight,
            self.eps,
            &grad_c,
        )
        .map_err(|e| kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward: bwd: {e}")))?;
        // Adjoints can be non-contiguous; contiguify then COPY to owned kt
        // (cf. the GdnRecurrentBackward non-contig-grad fix).
        let to_kt = |t: &Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
            let tc = t.contiguous().map_err(|e| {
                kiln_tensor::Error::Msg(format!("GdnGatedRmsNormBackward: grad contiguous: {e}"))
            })?;
            kiln_kt_bridge::kt_tensor_from_candle_cuda_copy(&tc).map_err(|e| {
                kiln_tensor::Error::Msg(format!(
                    "GdnGatedRmsNormBackward: grad candle->kt: {e}"
                ))
            })
        };
        Ok(vec![
            Some(to_kt(&grads.dx)?),
            Some(to_kt(&grads.dz)?),
            Some(to_kt(&grads.dw)?),
        ])
    }
}

/// Route the GDN gated RMSNorm `out = rms_norm(x, weight) * silu(z)` through
/// the kt `Tape`.
///
/// The production forward (`gated_rms_norm`) already computed `out`; this
/// records-only (no re-run), borrowing `out` as the node output — like
/// `tape_record_gdn_recurrent`. The recorded [`GdnGatedRmsNormBackward`] emits
/// `dx`/`dz`/`dw` so a tape-authoritative backward reaches the recurrence
/// output (`x`), the gate (`z`), and the `norm` `Var` (`weight`).
///
/// `Ok(None)` (caller falls through) when the gate is off, no tape scope is
/// active, the inputs aren't CUDA, shapes disagree, or a kt borrow fails.
pub fn try_tape_gdn_gated_rms_norm_cuda(
    x: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f64,
    out: &Tensor,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() || !tape_gdn_gated_norm_enabled() {
        return Ok(None);
    }
    if !matches!(x.device(), candle_core::Device::Cuda(_)) {
        return Ok(None);
    }
    if x.dims() != z.dims() || x.dims() != out.dims() {
        return Ok(None);
    }
    if weight.rank() != 1 || *x.dims().last().unwrap() != weight.dims()[0] {
        return Ok(None);
    }

    let x_kt = match tape_kt_input(x) {
        Some(t) => t,
        None => return Ok(None),
    };
    let z_kt = match tape_kt_input(z) {
        Some(t) => t,
        None => return Ok(None),
    };
    let weight_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(weight) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let out_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(out) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            &out_kt,
            &[&x_kt, &z_kt, &weight_kt],
            Box::new(GdnGatedRmsNormBackward {
                x: x.clone(),
                z: z.clone(),
                weight: weight.clone(),
                eps,
            }),
        );
    });
    if recorded.is_none() {
        return Ok(None);
    }

    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(z_kt.id(), z.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(weight_kt.id(), weight.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out.clone()))
}

/// Route a `transpose(axis_a, axis_b)` through the kt `Tape` so a
/// tape-authoritative backward stays connected ACROSS the transpose.
///
/// CP-4 chaining-gap fix: the production GDN recurrence can return its output
/// head-FIRST `[B, nv, T, dv]` (the chunkwise fallback), and the forward then
/// runs `attn_out.transpose(1, 2)` (`forward.rs:gated_deltanet_forward_decode_if`,
/// ~line 16137) to reach head-LAST `[B, T, nv, dv]` before the gated RMSNorm.
/// A plain candle transpose mints a fresh tensor id, so the gated-rms-norm
/// adapter's `tape_kt_input` couldn't chain back to the recurrence node — the
/// tape would fragment at the transpose and the GDN q/k/v/beta/g (LoRA) grads
/// would never flow. Recording a [`TransposeBackward`] node (whose adjoint
/// re-applies the same transpose — it's an involution) keeps the chain intact.
///
/// Gated on `KILN_USE_TAPE_FORWARD` + an active tape scope only (transpose is a
/// pure layout op with a trivial, always-safe adjoint, so it needs no dedicated
/// kill switch — same contract as [`try_tape_reshape_cuda`]). `Ok(None)` when
/// the gate is off, no tape is active, the input isn't CUDA, the axes are out of
/// bounds, or a kt borrow fails.
pub fn try_tape_transpose_cuda(
    x: &Tensor,
    axis_a: usize,
    axis_b: usize,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    if !matches!(x.device(), candle_core::Device::Cuda(_)) {
        return Ok(None);
    }

    let x_kt = match tape_kt_input(x) {
        Some(t) => t,
        None => return Ok(None),
    };
    let rank = x_kt.rank();
    if axis_a >= rank || axis_b >= rank {
        return Ok(None);
    }

    let out_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        // `transpose` yields a non-contiguous view; `kt_tensor_to_candle_cuda_copy`
        // (below) requires contiguity, so materialise. The recorded output id is
        // this contiguous tensor — the `TransposeBackward` adjoint transposes the
        // upstream grad regardless of the forward output's layout, so this is
        // value-faithful (the production `attn_out.transpose(1,2)` view and this
        // contiguous copy carry identical elements).
        let y = x_kt
            .transpose(axis_a, axis_b)
            .map_err(|e| anyhow::anyhow!("kt transpose: {e}"))?;
        let y = if y.is_contiguous() {
            y
        } else {
            y.contiguous()
                .map_err(|e| anyhow::anyhow!("kt transpose: contiguous: {e}"))?
        };
        tape.record(&y, &[&x_kt], Box::new(TransposeBackward { axis_a, axis_b }));
        Ok(y)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };

    let out_kt =
        out_kt.context("tape_forward::try_tape_transpose_cuda: kt-tape forward failed")?;
    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .context("tape_forward::try_tape_transpose_cuda: kt -> candle copy failed")?;

    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out))
}

// ===========================================================================
// CP-4 (#1082): naive SDPA-fallback attention tape coverage.
//
// `try_tape_flash_attn_cuda` only fires when flash-attention is available
// (`head_dim ∈ {128, 256}`). The GQA full-attention block's NON-flash
// fallback (`forward::gqa_attention_core_prefill`'s naive scaled-dot-product
// path) is the path that runs at every other head_dim — notably the tiny
// synthetic test model's `head_dim = 16`. For a tape-authoritative backward
// to reach the GQA-block q/k/v projection LoRA `Var`s on that path, the
// fallback must ALSO record onto the kt Tape, exactly as the flash path does.
//
// Mirrors `try_tape_gdn_recurrent_cuda` / `GdnRecurrentBackward`: a
// candle-composite `BackwardOp` (`SdpaBackward`) wrapping the analytic
// `forward::sdpa_fallback_backward_no_grad`, recorded on the fallback's
// attention output with `[q, k, v]` as inputs. SDPA is stateless, so there is
// no entry-state to snapshot.
// ===========================================================================

/// True iff `KILN_USE_TAPE_SDPA` is set to an enable value. Narrow opt-in for
/// the naive SDPA-fallback attention tape adapter, separate from
/// `KILN_USE_TAPE_FLASH_ATTN` (which covers the flash path) and the rest of
/// the fleet. Cached after first read, mirroring [`tape_gdn_enabled`].
pub fn tape_sdpa_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_SDPA")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(false)
    })
}

/// Candle-composite tape backward for the naive SDPA fallback
/// (`forward::gqa_attention_core_prefill`'s non-flash path). Wraps
/// [`sdpa_fallback_backward_no_grad`] (analytic adjoint in candle F32),
/// mirroring how [`GdnRecurrentBackward`] wraps `gdn_recurrent_backward_no_grad`.
///
/// # Why a candle-composite wrap (not a kt `BackwardOp` in kiln-autograd)
///
/// The SDPA backward is a composite of broadcast / 4D-batched matmuls, a
/// softmax adjoint, a causal mask, and a GQA head collapse. Those aren't
/// cleanly expressible over `kiln_tensor::ops` (no batched `broadcast_matmul`
/// / softmax-adjoint primitive there), and `kiln-autograd` carries no candle
/// dep — so the analytic backward lives as a candle composite in `kiln-model`
/// and this `BackwardOp` bridges grads through it, exactly like the GDN ops.
///
/// # Saved tensors / inputs
///
/// `q`/`k`/`v` are the **pre-attention, head-FIRST** tensors the fallback
/// consumes (`q = [B, nq, T, hd]`, `k`/`v = [B, nkv, T, hd]`, BEFORE the GQA
/// expand) as candle clones; `scale = 1/sqrt(head_dim)`; `causal` selects the
/// strict-upper-triangular mask. 3 differentiable inputs `[q, k, v]` in the
/// order the adapter records them. The returned `dq` keeps `nq` heads;
/// `dk`/`dv` are GQA-collapsed to `nkv` (matching the `k`/`v` `Var` layouts).
#[derive(Debug)]
pub(crate) struct SdpaBackward {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: f64,
    causal: bool,
}

impl BackwardOp for SdpaBackward {
    fn name(&self) -> &'static str {
        "sdpa_fallback_backward"
    }
    fn input_count(&self) -> usize {
        // q, k, v.
        3
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // Bridge the upstream grad (kt) to candle for the candle composite.
        let grad_c = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(grad_output).map_err(|e| {
            kiln_tensor::Error::Msg(format!("SdpaBackward: grad kt->candle: {e}"))
        })?;
        let grads = sdpa_fallback_backward_no_grad(
            &self.q,
            &self.k,
            &self.v,
            self.scale,
            self.causal,
            &grad_c,
        )
        .map_err(|e| kiln_tensor::Error::Msg(format!("SdpaBackward: sdpa bwd: {e}")))?;
        // Adjoints can be non-contiguous (broadcast/transpose views); contiguify
        // then COPY to owned kt (cf. the GdnRecurrentBackward non-contig fix).
        let to_kt = |t: &Tensor| -> kiln_tensor::Result<kiln_tensor::Tensor> {
            let tc = t.contiguous().map_err(|e| {
                kiln_tensor::Error::Msg(format!("SdpaBackward: grad contiguous: {e}"))
            })?;
            kiln_kt_bridge::kt_tensor_from_candle_cuda_copy(&tc).map_err(|e| {
                kiln_tensor::Error::Msg(format!("SdpaBackward: grad candle->kt: {e}"))
            })
        };
        Ok(vec![
            Some(to_kt(&grads.dq)?),
            Some(to_kt(&grads.dk)?),
            Some(to_kt(&grads.dv)?),
        ])
    }
}

/// Record an [`SdpaBackward`] node for the naive SDPA-fallback attention
/// output that the PRODUCTION forward has ALREADY computed.
///
/// The forward (`gqa_attention_core_prefill`'s non-flash path) computes the
/// attention output itself (q@kᵀ scaled, causal-masked softmax, @v) and then
/// reshapes it back to `[B, T, hidden]`; this adapter takes that already-
/// computed `out` (head-FIRST `[B, nq, T, hd]`, BEFORE the reshape-back) and
/// records-only (no re-run), borrowing `out` as the recorded node's output —
/// like `tape_record_gdn_recurrent`. The recorded [`SdpaBackward`] emits
/// GQA-collapsed `dq`/`dk`/`dv` so a tape-authoritative backward reaches the
/// q/k/v projections (and their LoRA `Var`s) on the non-flash path — the
/// attention-block link the flash path covers via [`try_tape_flash_attn_cuda`].
///
/// # Arguments
///
/// * `q`/`k`/`v` — the **pre-attention head-FIRST** tensors the fallback
///   consumes: `q = [B, nq, T, hd]`, `k`/`v = [B, nkv, T, hd]` (the
///   `prepared.{q,k,v}.transpose(1,2)` layout, BEFORE the GQA expand). They
///   must carry their LoRA lineage from the upstream q/k/v_proj adapters via
///   `tape_kt_input` chaining.
/// * `head_dim` — `scale = 1/sqrt(head_dim)`, matching the forward's score
///   divisor.
/// * `out` — the attention output the forward produced, head-FIRST
///   `[B, nq, T, hd]` (the `attn_weights_softmax.broadcast_matmul(&v)` result
///   BEFORE its `transpose(1,2).reshape(...)`).
///
/// `Ok(None)` (caller's production output unchanged) when the gate is off, no
/// tape scope is active, the inputs aren't CUDA, shapes disagree, or a kt
/// borrow fails.
pub fn try_tape_sdpa_fallback_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    out: &Tensor,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() || !tape_sdpa_enabled() {
        return Ok(None);
    }
    if !matches!(q.device(), candle_core::Device::Cuda(_))
        || !matches!(k.device(), candle_core::Device::Cuda(_))
        || !matches!(v.device(), candle_core::Device::Cuda(_))
        || !matches!(out.device(), candle_core::Device::Cuda(_))
    {
        return Ok(None);
    }

    // Shape envelope: q = [B, nq, T, hd]; k/v = [B, nkv, T, hd]; out matches q.
    let (bq, nq, tq, dq_) = match q.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (bk, nkv, tk, dk_) = match k.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (bv, nvh, tv, dv_) = match v.dims4() {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    if bq != bk
        || bq != bv
        || nkv == 0
        || nq % nkv != 0
        || nvh != nkv
        || tq != tk
        || tq != tv
        || dq_ != head_dim
        || dk_ != head_dim
        || dv_ != head_dim
        || out.dims() != [bq, nq, tq, head_dim].as_slice()
    {
        return Ok(None);
    }

    let scale = 1.0f64 / (head_dim as f64).sqrt();
    let causal = true;

    // kt inputs — thread the upstream q/k/v_proj (+ RoPE / norm) adapter outputs
    // so the tape stays connected back to the LoRA Vars.
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
    // The production forward already computed `out`; borrow it as the recorded
    // node's output so we record-only (no re-run), like `tape_record_gdn_recurrent`.
    let out_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(out) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            &out_kt,
            &[&q_kt, &k_kt, &v_kt],
            Box::new(SdpaBackward {
                q: q.clone(),
                k: k.clone(),
                v: v.clone(),
                scale,
                causal,
            }),
        );
    });
    if recorded.is_none() {
        return Ok(None);
    }

    kiln_kt_bridge::tape_bridge::register_input_mapping(q_kt.id(), q.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(k_kt.id(), k.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(v_kt.id(), v.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&out_kt, out.id());

    Ok(Some(out.clone()))
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

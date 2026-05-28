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
    EmbeddingBackward, MatmulBackward, MulSigmoidGateBackward, SiluBackward, Tape,
};

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

    // CP-4 (#1082) tape_bridge: register the (kt_id ↔ candle_id) IO
    // mappings so a surrounding `with_tape_scope_emit_to_grad_store`
    // can transmute the tape-recorded backward into candle-typed
    // gradients in the candle GradStore. No-ops cleanly when no
    // bridge scope is active.
    kiln_kt_bridge::tape_bridge::register_input_mapping(x_kt.id(), x.id());
    kiln_kt_bridge::tape_bridge::register_input_mapping(w_kt.id(), weight.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(out_kt.id(), out.id());

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
    let a_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(a) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let b_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(b) {
        Ok(t) => t,
        Err(_) => return Ok(None),
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

    let x_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(x) {
        Ok(t) => t,
        Err(_) => return Ok(None),
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
    let gate_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(gate) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let up_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(up) {
        Ok(t) => t,
        Err(_) => return Ok(None),
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

    Ok(Some(out))
}

// Tape-scope tests live in `kiln-autograd::tape_scope::tests` after
// wave-13 (#1082) promoted the thread-local-tape machinery there. The
// kt-tape adapter tests (`try_tape_{rms_norm,matmul,silu,embedding,swiglu}_cuda`
// round-trips) live in the `kiln-model/tests/tape_forward_parity.rs`
// integration test because they require the `kiln_kt_bridge` +
// `kiln_rmsnorm_kernel` cuda surface.

//! GRPO policy-gradient scalar-loss tape root for the tape-authoritative
//! GRPO step (#1082 CP-4).
//!
//! # Why this module exists
//!
//! In the tape-authoritative training path (`KILN_USE_TAPE_AUTHORITATIVE`,
//! the CP-4 production default) the loss is the backward ROOT: gradients are
//! driven by walking the kt `kiln_autograd::Tape` from the scalar loss, NOT by
//! candle's `loss.backward()`. The SFT path roots the tape at
//! `kiln_model::tape_forward::try_tape_cross_entropy_from_logits_cuda` (a fused
//! "scalar CE from full logits" node); OPD roots it at
//! `crate::opd_candle_shim::try_tape_opd_scalar_mean_cuda` (the OPD reverse-KL
//! scalar node). GRPO needs the same: a single tape node that takes the FULL
//! `[1, T, V]` policy logits (the tape-connected lm_head output) and produces
//! the scalar GRPO policy-gradient (+ optional KL) loss, so one `Tape::backward`
//! routes `dL/d(logits)` back through the model chain into every LoRA `Var`.
//!
//! # Why a single fused node (not a chain of primitive tape adapters)
//!
//! The GRPO loss is built from `token_log_probs` (`squeeze` → `narrow` →
//! `index_select` → `log_sum_exp` → `gather` → `sub`) then `grpo_loss`
//! (`exp` / `clamp` / `minimum` / scalar-mul / `sum` / `affine`). Several of
//! those ops sit on kt-CUDA backward gaps (the `index_select` adjoint is a
//! scatter; `gather` / `cast` kt backward are CPU-only) and would each need a
//! dedicated `try_tape_*_cuda` adapter that does not exist. The fused node
//! sidesteps every gap exactly like `CrossEntropyFromLogitsBackward` does: its
//! single differentiable input is the full logits, and its backward is a
//! CUDA-safe candle composite. We recompute the candle GRPO forward with
//! autograd ON against a fresh leaf clone of the saved logits and run candle's
//! own `loss.backward()` to obtain `dL/d(logits)` — guaranteeing bit-for-bit
//! agreement with the candle GRPO loss across ALL IS levels (Token / Sequence /
//! Cispo), KL estimators (None / K1 / K3), and the REINFORCE short-circuit, with
//! zero risk of an analytic-derivation drift bug. The composite is pure candle
//! and runs entirely on the device (no host round-trip of the `[1, T, V]`
//! activation — only the scalar seed touches the host).
//!
//! # Carve-outs (NOT covered by this tape root)
//!
//! * **ECHO env-CE.** The ECHO term (`λ · mean_envCE`) is added to the policy
//!   loss only on agentic off-policy data. It is composed against the SAME
//!   `policy_logits`, so it COULD be folded in here, but matching the OPD
//!   carve-out we keep any step with an active ECHO contribution on the candle
//!   gradient-checkpointing path (the dispatch in
//!   `train_tokenized_grpo_group_with_grad_norms` gates the tape path off when
//!   ECHO fires). Non-ECHO GRPO — the common verifier-based and verifier-free
//!   on-policy case — is the tape-authoritative target.
//! * **`no_policy_loss` with no ECHO.** That config is a constant-zero loss
//!   (already rejected by the candle path); the dispatch keeps it on candle.
//! * **`KlEstimator` / `IsLevel`** are all SUPPORTED — the candle recompute
//!   handles every variant uniformly because it just re-runs `grpo_loss`.

#[cfg(feature = "cuda")]
use crate::cd_types::CdDevice;
#[cfg(feature = "cuda")]
use crate::trainer::{grpo_loss, token_log_probs, GrpoLossParams};
#[cfg(feature = "cuda")]
use anyhow::{Context, Result};
#[cfg(feature = "cuda")]
use candle_core::Tensor;
#[cfg(feature = "cuda")]
use kiln_autograd::{tape_forward_enabled, with_active_tape, BackwardOp, Tape};

/// Fused backward for the GRPO scalar PG (+ KL) loss taken from the full
/// `[1, T, V]` policy logits. Saves the candle `logits` (an `Arc` bump on the
/// candle storage), the host-side `input_ids` / `action_mask`, the
/// (detached, constant) `ref_log_probs`, and the `GrpoLossParams`. The backward
/// recomputes the candle GRPO forward with autograd ON and runs candle's
/// `loss.backward()` to produce `dL/d(logits)` as a single `[1, T, V]` kt grad
/// (input count 1).
///
/// `requires_input` returns `false`: the composite recomputes the forward
/// gather from the SAVED `logits`, so the tape walker need not re-materialise
/// the input activation (mirrors `CrossEntropyFromLogitsBackward`).
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct GrpoPgLossFromLogitsBackward {
    /// FULL forward logits `[1, T, V]` (candle clone — an `Arc` bump).
    logits: Tensor,
    /// Tokenized completion (prompt + completion), the loss positions are
    /// `{ i : action_mask[i] }` under the next-token shift.
    input_ids: Vec<u32>,
    /// Action-token mask (true at supervised completion positions).
    action_mask: Vec<bool>,
    /// Detached, constant reference log-probs `[num_active]` (the IS-ratio
    /// denominator). Never differentiated — saved as a plain candle tensor.
    ref_log_probs: Tensor,
    /// GRPO surrogate / KL parameters (advantage, clip bounds, KL estimator,
    /// loss normalizer, IS level, reinforce flag, entropy-aware quantile).
    loss_params: GrpoLossParams,
    /// Candle device (for the leaf clone + recompute).
    device: candle_core::Device,
}

#[cfg(feature = "cuda")]
impl BackwardOp for GrpoPgLossFromLogitsBackward {
    fn name(&self) -> &'static str {
        "grpo_pg_loss_from_logits_backward"
    }
    fn input_count(&self) -> usize {
        // The full logits `[1, T, V]`.
        1
    }
    fn requires_input(&self, _idx: usize) -> bool {
        // The composite recomputes the forward from the SAVED `logits`; the tape
        // walker need not re-materialise the input activation.
        false
    }
    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        // The upstream grad is the scalar dL/dloss seed (typically 1.0). Bridge
        // it to candle and read the scalar so the composite can scale the
        // recomputed `dL/d(logits)` (linearity of backward in the seed).
        let grad_c = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(grad_output).map_err(|e| {
            kiln_tensor::Error::Msg(format!("GrpoPgLossFromLogitsBackward: grad kt->candle: {e}"))
        })?;
        let grad_scalar = grad_c
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| {
                kiln_tensor::Error::Msg(format!(
                    "GrpoPgLossFromLogitsBackward: grad scalar read: {e}"
                ))
            })?
            .first()
            .copied()
            .ok_or_else(|| {
                kiln_tensor::Error::Msg(
                    "GrpoPgLossFromLogitsBackward: empty grad_output".to_string(),
                )
            })? as f64;

        let grad_logits = grpo_pg_loss_from_logits_grad_candle(
            &self.logits,
            &self.input_ids,
            &self.action_mask,
            &self.ref_log_probs,
            self.loss_params,
            grad_scalar,
            &self.device,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "GrpoPgLossFromLogitsBackward: candle composite: {e}"
            ))
        })?;

        // The composite output may be non-contiguous (cat / unsqueeze / autograd
        // graph) and owns no kt lifetime — materialise contiguous and COPY into
        // an owned kt tensor (not borrow), matching `CrossEntropyFromLogitsBackward`.
        let grad_logits = grad_logits.contiguous().map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "GrpoPgLossFromLogitsBackward: grad contiguous: {e}"
            ))
        })?;
        let grad_logits_kt =
            kiln_kt_bridge::kt_tensor_from_candle_cuda_copy(&grad_logits).map_err(|e| {
                kiln_tensor::Error::Msg(format!(
                    "GrpoPgLossFromLogitsBackward: grad candle->kt: {e}"
                ))
            })?;

        Ok(vec![Some(grad_logits_kt)])
    }
}

/// Device-agnostic candle composite for `dL/d(logits)` of the GRPO scalar loss.
///
/// Recomputes the EXACT candle GRPO forward (`token_log_probs` → `grpo_loss`,
/// the same functions the candle-authoritative path calls) against a fresh
/// autograd-tracked leaf clone of `logits`, runs candle's own `loss.backward()`,
/// and scales the resulting `dL/d(logits)` by the incoming scalar seed
/// `grad_scalar` (backward is linear in the seed). This is pure candle, so it is
/// the autograd oracle and runs (and can be parity-tested) on CPU as well as
/// CUDA — no kt round-trip. Returns `[1, T, V]` cast back to `logits.dtype()`.
///
/// Because `ref_log_probs` is a detached constant and the GRPO loss depends on
/// the model only through `policy_log_probs = token_log_probs(logits, ...)`, the
/// candle graph from the leaf logits to the scalar loss is complete and
/// candle's backward yields the full `dL/d(logits)` for every IS level / KL
/// estimator without any per-variant hand derivation.
#[cfg(feature = "cuda")]
pub(crate) fn grpo_pg_loss_from_logits_grad_candle(
    logits: &Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    ref_log_probs: &Tensor,
    loss_params: GrpoLossParams,
    grad_scalar: f64,
    device: &CdDevice,
) -> Result<Tensor> {
    let logits_dtype = logits.dtype();
    let dims = logits.dims();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] == input_ids.len(),
        "grpo_pg_loss_from_logits_grad_candle: logits must be [1, seq_len, vocab], got {dims:?} \
         for seq_len {}",
        input_ids.len()
    );
    anyhow::ensure!(
        action_mask.len() == input_ids.len(),
        "grpo_pg_loss_from_logits_grad_candle: action_mask len {} != input_ids len {}",
        action_mask.len(),
        input_ids.len()
    );

    // Fresh autograd-tracked leaf clone of the saved (detached) logits. The
    // recomputed forward builds candle lineage from THIS leaf, so `loss.backward`
    // produces a grad keyed on `leaf`'s id. Detach first so we never re-enter any
    // stale lineage carried by the saved tensor, then re-promote to a Var.
    let leaf = candle_core::Var::from_tensor(&logits.detach())
        .context("grpo_pg_loss_from_logits_grad_candle: leaf Var from logits")?;
    let leaf_t = leaf.as_tensor();

    // EXACT candle GRPO forward — the same `token_log_probs` + `grpo_loss` the
    // candle-authoritative GRPO step runs (single source of truth for the math).
    let policy_log_probs = token_log_probs(leaf_t, input_ids, action_mask, device)
        .context("grpo_pg_loss_from_logits_grad_candle: token_log_probs")?;
    let loss = grpo_loss(&policy_log_probs, ref_log_probs, loss_params, device)
        .context("grpo_pg_loss_from_logits_grad_candle: grpo_loss")?;

    let grads = loss
        .backward()
        .context("grpo_pg_loss_from_logits_grad_candle: candle loss.backward()")?;
    let grad_leaf = grads.get(leaf_t).cloned().with_context(|| {
        "grpo_pg_loss_from_logits_grad_candle: candle backward produced no grad for the logits \
         leaf (the loss did not depend on logits — likely an all-false action_mask)"
            .to_string()
    })?;

    // Scale by the incoming scalar seed (backward is linear in dL/dloss) and cast
    // back to the saved logits dtype so the kt grad matches the lm_head output
    // layout the upstream tape node expects.
    let grad_logits = if (grad_scalar - 1.0).abs() > f64::EPSILON {
        grad_leaf.affine(grad_scalar, 0.0)?
    } else {
        grad_leaf
    };
    Ok(grad_logits.to_dtype(logits_dtype)?)
}

/// Attempt to root the GRPO scalar PG (+ KL) loss at a SINGLE fused kt `Tape`
/// node taking the FULL `[1, T, V]` policy logits.
///
/// Mirrors `kiln_model::tape_forward::try_tape_cross_entropy_from_logits_cuda`
/// (SFT) and `crate::opd_candle_shim::try_tape_opd_scalar_mean_cuda` (OPD): the
/// returned candle scalar is a DETACHED, lineage-free value-copy of the loss
/// (so the tape-authoritative caller's `loss.backward()` is unconditionally
/// `{loss: ones}` and the recorded node is the sole backward root); the
/// gradient lives on the tape via the recorded `GrpoPgLossFromLogitsBackward`.
///
/// Returns:
/// * `Ok(Some(loss))` — the tape-forward path ran; a `GrpoPgLossFromLogitsBackward`
///   node was recorded and IO-mapped into the bridge.
/// * `Ok(None)` — gate off (`KILN_USE_TAPE_FORWARD` unset), `logits` is not a
///   CUDA rank-3 `[1, T, V]`, no tape scope is active, an empty active set, or a
///   kt borrow failed. The caller must NOT have selected the tape-authoritative
///   path if the envelope is unmet (the dispatch device-/ECHO-gates it); this
///   surfaces a clean `None` so a misdispatch is caught.
/// * `Err(...)` — an unexpected forward or kt -> candle copy-back failure.
#[cfg(feature = "cuda")]
pub(crate) fn try_tape_grpo_pg_loss_from_logits_cuda(
    logits: &Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    ref_log_probs: &Tensor,
    loss_params: GrpoLossParams,
    device: &CdDevice,
) -> Result<Option<Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // Full model logits only: [1, T, V] on CUDA. Defer any other shape/device to
    // the caller (the dispatch keeps non-CUDA on the candle path anyway).
    if logits.rank() != 3 || !matches!(logits.device(), candle_core::Device::Cuda(_)) {
        return Ok(None);
    }
    let Ok((b, t, _v)) = logits.dims3() else {
        return Ok(None);
    };
    if b != 1 || t != input_ids.len() || action_mask.len() != input_ids.len() {
        return Ok(None);
    }

    // Active-token short-circuit: an empty supervised set has no PG signal, and
    // recording a tape node for a no-op forward is a footgun (matches the OPD
    // scalar adapter). `token_log_probs` uses the next-token shift, so the
    // active set is `{ i in 1..T : action_mask[i] }`.
    let active_count = action_mask
        .get(1..)
        .map_or(0, |m| m.iter().filter(|&&v| v).count());
    if active_count == 0 {
        return Ok(None);
    }

    // kt input — thread the lm_head adapter's output so the tape stays CONNECTED
    // (consumer input id == producer output id). Fall through on borrow failure.
    let logits_kt = match kiln_kt_bridge::tape_bridge::kt_input_for_candle(logits.id()) {
        Some(t) => t,
        None => match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(logits) {
            Ok(t) => t,
            Err(_) => return Ok(None),
        },
    };

    // FORWARD value via candle — the same `token_log_probs` + `grpo_loss` math as
    // the candle-authoritative path, so the returned scalar is numerically
    // identical. `ref_log_probs` is the detached constant denominator.
    let policy_log_probs = token_log_probs(logits, input_ids, action_mask, device)
        .context("try_tape_grpo_pg_loss_from_logits_cuda: token_log_probs")?;
    let loss_candle = grpo_loss(&policy_log_probs, ref_log_probs, loss_params, device)
        .context("try_tape_grpo_pg_loss_from_logits_cuda: grpo_loss")?;

    // Record the fused node: the OUTPUT must be an OWNED kt copy of the loss
    // (a borrow would dangle once the local `loss_candle` drops — the tape is
    // walked much later). The copy is a scalar (negligible). Saved state: the
    // FULL logits + host-side gather metadata + detached ref_log_probs + params.
    let loss_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let loss_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_copy(&loss_candle)
            .map_err(|e| anyhow::anyhow!("loss kt copy: {e}"))?;
        tape.record(
            &loss_kt,
            &[&logits_kt],
            Box::new(GrpoPgLossFromLogitsBackward {
                logits: logits.clone(),
                input_ids: input_ids.to_vec(),
                action_mask: action_mask.to_vec(),
                ref_log_probs: ref_log_probs.detach(),
                loss_params,
                device: device.clone(),
            }) as Box<dyn BackwardOp>,
        );
        Ok(loss_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };
    let loss_kt = loss_kt
        .context("try_tape_grpo_pg_loss_from_logits_cuda: kt-tape forward failed")?;

    // Return a DETACHED, lineage-free loss by construction (a fresh kt -> candle
    // CUDA copy, numerically identical to the candle baseline). The candle
    // `loss_candle` still carries candle autograd lineage; if returned, the
    // tape-authoritative caller's `loss.backward()` could let candle's autograd
    // silently fill in LoRA-Var grads (a false positive defeating the
    // tape-coverage measurement). The fresh copy makes `loss.backward()`
    // unconditionally `{loss: ones}` and the recorded node the sole tape root.
    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&loss_kt)
        .context("try_tape_grpo_pg_loss_from_logits_cuda: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: only `logits` is differentiable. Map the output /
    // retain onto the RETURNED detached copy's id so `with_tape_authoritative_scope`
    // resolves `loss.id()` → `loss_kt` to seed the tape root.
    kiln_kt_bridge::tape_bridge::register_input_mapping(logits_kt.id(), logits.id());
    kiln_kt_bridge::tape_bridge::register_output_mapping(loss_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&loss_kt, out.id());

    Ok(Some(out))
}

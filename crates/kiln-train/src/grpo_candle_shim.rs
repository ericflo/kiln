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
//! single differentiable input is the full logits, and its backward derives
//! `dL/d(logits)` analytically, in kt, on-device.
//!
//! # How the backward computes `dL/d(logits)` (#1082 C2, kt-native)
//!
//! The GRPO loss depends on `logits` ONLY through
//! `policy_log_probs = token_log_probs(logits, …)` (an F32 `[num_active]`
//! vector). By the chain rule, for each active position `p` with label `y`:
//!
//! ```text
//! dL/d(logits[p, j]) = coeff_a * ( onehot(y)[j] - softmax(logits[p, :])[j] )
//! ```
//!
//! where `coeff_a = dL/d(policy_log_prob_a)`. The `(onehot - softmax)` factor is
//! the log-softmax Jacobian-vector product — a pure FORWARD computation (softmax
//! + onehot scatter), which sidesteps the kt-CUDA `index_select` / `gather`
//! backward gaps. `coeff` is obtained by NUMERICALLY differentiating the cheap
//! `[num_active]`-vector `grpo_loss` scalar w.r.t. `policy_log_probs` — calling
//! the SAME kt `grpo_loss`, so it is automatically correct for the value-
//! differentiable IS levels (Token / Sequence) and KL estimators (None / K1 /
//! K3), with no per-variant analytic-derivation drift. The two straight-through
//! variants (a `.detach()` on a value-flowing path makes the loss VALUE diverge
//! from the autograd surface, so a value-FD is wrong) are handled analytically:
//! REINFORCE (`exp(plp - plp.detach())` ≡ 1 by value → value-FD returns 0;
//! `coeff = -advantage · loss_normalizer`) and CISPO (detached `weight`
//! multiplying `plp` → `coeff = -weight + KL grad`). See
//! [`grpo_pg_loss_from_logits_grad_kt`]. (Validated by the candle-free
//! finite-difference / closed-form test
//! `grpo_logit_grad_matches_finite_difference_f32` at the foot of this file,
//! which covers Token+None, Token+K1, and reinforce.)
//!
//! Pre-C2 this node ran a candle recompute: it built a candle `Var` leaf, ran
//! the (kt) forward, bridged the loss back to candle, and called candle's
//! `loss.backward()`. Once the GRPO math flipped to kt that carried no candle
//! lineage from the leaf to the loss, so `grads.get(leaf)` was always `None` and
//! GRPO training was fully broken — the analytic kt derivation above replaces it.
//!
//! The candle `logits` / `ref_log_probs` are bridged into kt ONCE per backward
//! and the final `[1, T, V]` grad bridged back to candle for the return type
//! (eliminating those I/O copies is a separate later #1082 task); the gradient
//! math is otherwise entirely kt on-device.
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
//! * **`KlEstimator` / `IsLevel`** are all SUPPORTED — the numeric-`coeff`
//!   derivation handles every variant uniformly because it just re-runs
//!   `grpo_loss`.

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
/// (#1082 C2) derives `dL/d(logits)` ANALYTICALLY in kt on-device — the
/// log-softmax JVP `coeff_a · (onehot − softmax)` with a numeric `coeff` from
/// `grpo_loss` — producing a single `[1, T, V]` kt grad (input count 1). No
/// candle `Var` / `loss.backward()`; only the candle I/O bridges remain.
///
/// `requires_input` returns `false`: the backward recomputes the forward gather
/// from the SAVED `logits`, so the tape walker need not re-materialise the input
/// activation (mirrors `CrossEntropyFromLogitsBackward`).
#[cfg(feature = "cuda")]
#[derive(Debug)]
struct GrpoPgLossFromLogitsBackward {
    /// FULL forward logits `[1, T, V]` as a kt tensor (an `Arc` bump on the kt
    /// storage — no device copy). (#1082 DoD-100 step 8: was a candle clone.)
    logits: kiln_tensor::Tensor,
    /// Tokenized completion (prompt + completion), the loss positions are
    /// `{ i : action_mask[i] }` under the next-token shift.
    input_ids: Vec<u32>,
    /// Action-token mask (true at supervised completion positions).
    action_mask: Vec<bool>,
    /// Detached, constant reference log-probs `[num_active]` (the IS-ratio
    /// denominator). Never differentiated — saved as a kt tensor (#1082 step 8).
    ref_log_probs: kiln_tensor::Tensor,
    /// GRPO surrogate / KL parameters (advantage, clip bounds, KL estimator,
    /// loss normalizer, IS level, reinforce flag, entropy-aware quantile).
    loss_params: GrpoLossParams,
    /// kt device for the analytic backward (#1082 step 8: `CdDevice` is the kt
    /// `Device` alias; was a candle device bridged per-call).
    device: CdDevice,
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
        // The upstream grad is the scalar dL/dloss seed (typically 1.0). Read the
        // scalar directly off the kt grad (no candle round-trip) so the analytic
        // backward can scale the `dL/d(logits)` it derives (linearity in the seed).
        // Mirrors `CrossEntropyFromLogitsKtBackward`.
        let grad_scalar = grad_output
            .to_dtype(kiln_tensor::DType::F32)
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

        // kt-native analytic dL/d(logits) — no candle. The kt logits / ref are
        // saved directly on the struct (#1082 step 8);
        // `grpo_pg_loss_from_logits_grad_kt` returns a `[1, T, V]` kt grad
        // (input count 1).
        let grad_logits_kt = grpo_pg_loss_from_logits_grad_kt(
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
                "GrpoPgLossFromLogitsBackward: analytic logit-grad: {e}"
            ))
        })?;

        Ok(vec![Some(grad_logits_kt)])
    }
}

/// Per-active-token entropy-aware KL gate (1.0 / 0.0) mirroring the detached
/// mask `crate::trainer::grpo_loss` applies to the KL penalty. Returns an
/// all-ones mask when no quantile is configured (the common case). Used only by
/// the CISPO analytic coeff (the KL grad it adds must respect the same gate).
///
/// ⚠ DRIFT COUPLING with `grpo_loss`'s entropy-quantile block — keep in lockstep.
#[cfg(feature = "cuda")]
fn cispo_entropy_kl_mask(plp_host: &[f32], loss_params: &GrpoLossParams) -> Vec<f64> {
    let n = plp_host.len();
    match loss_params.entropy_aware_kl_quantile {
        Some(q) if q.is_finite() && (0.0..1.0).contains(&q) => {
            // Quantile of {-plp} (proxy entropy), threshold-gate: 1 iff
            // -plp_a >= thr. Matches `grpo_loss` exactly.
            let mut neg: Vec<f64> = plp_host.iter().map(|p| -(*p as f64)).collect();
            neg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((q as f64) * (n.saturating_sub(1)) as f64).round() as usize;
            let thr = neg[idx.min(n.saturating_sub(1))];
            plp_host
                .iter()
                .map(|p| if -(*p as f64) >= thr { 1.0 } else { 0.0 })
                .collect()
        }
        _ => vec![1.0; n],
    }
}

/// kt-native analytic `dL/d(logits)` for the GRPO scalar loss. Pure kt ops on
/// `device`; no candle. See the module-level docs for the derivation. Returns the `[1, T, V]` grad as F32 kt (the caller casts to the
/// saved logits dtype).
///
/// Inputs:
/// * `logits_kt` — `[1, T, V]` policy logits (any float dtype; cast to F32
///   internally to match `token_log_probs`).
/// * `input_ids` / `action_mask` — host-side gather metadata (`len == T`).
/// * `ref_log_probs_kt` — `[num_active]` detached reference log-probs.
/// * `loss_params` — the GRPO surrogate / KL params.
/// * `grad_scalar` — the upstream scalar seed `dL/dloss` (backward is linear).
#[cfg(feature = "cuda")]
fn grpo_pg_loss_from_logits_grad_kt(
    logits_kt: &kiln_tensor::Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    grad_scalar: f64,
    device: &CdDevice,
) -> Result<kiln_tensor::Tensor> {
    use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

    let seq_len = input_ids.len();
    let dims = logits_kt.dims().to_vec();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] == seq_len,
        "grpo_pg_loss_from_logits_grad_kt: logits must be [1, seq_len, vocab], got {dims:?} for \
         seq_len {seq_len}"
    );
    let vocab = dims[2];

    // Active next-token positions in the SHIFTED frame (== seq positions in
    // `logits[0 .. T-1]`): { i : action_mask[i+1] }. This is exactly the set
    // `token_log_probs` builds from `shift_mask = mask[1..]`, in the same order.
    let active_positions: Vec<usize> = action_mask
        .get(1..)
        .map(|shift_mask| {
            shift_mask
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| if m { Some(i) } else { None })
                .collect()
        })
        .unwrap_or_default();
    let num_active = active_positions.len();

    // Defensive empty-active short-circuit (the forward already short-circuits;
    // `try_tape_*` returns Ok(None) for active_count==0 so `apply` is never
    // reached, but keep this self-contained): the loss does not depend on
    // logits — return an all-zero `[1, T, V]` F32 grad.
    if num_active == 0 {
        return KtTensor::zeros(vec![1, seq_len, vocab], KtDType::F32, *device)
            .map_err(Into::into);
    }

    // --- 1) Policy log-probs (kt oracle forward, single source of truth). ---
    let policy_log_probs = token_log_probs(logits_kt, input_ids, action_mask, device)
        .context("grpo_pg_loss_from_logits_grad_kt: token_log_probs")?;
    anyhow::ensure!(
        policy_log_probs.elem_count() == num_active,
        "grpo_pg_loss_from_logits_grad_kt: token_log_probs len {} != num_active {num_active}",
        policy_log_probs.elem_count()
    );

    // --- 2) coeff_a = dL/d(policy_log_prob_a). ---
    //
    // For the value-differentiable variants (`IsLevel::{Token, Sequence}` with
    // any `KlEstimator`), `grpo_loss` depends on `policy_log_probs` ENTIRELY
    // through its loss VALUE — every plp-dependent subexpression
    // (`exp(plp - ref)`, the GSPO mean, the K1/K3 KL) is differentiable with no
    // stop-gradient on the value path. A central finite difference of the cheap
    // `[num_active]`-vector `grpo_loss` therefore recovers the exact coeff,
    // candle-free and with no per-variant hand derivation. `num_active` is tiny
    // (completion length) and `grpo_loss` is ~10 elementwise kt ops, so the
    // 2·num_active extra evaluations are microseconds/step.
    //
    // The straight-through (`.detach()`-on-a-value-path) variants are the
    // exception: their loss VALUE does not match the autograd-intended surface,
    // so a value-FD is WRONG. There are exactly two, both handled analytically:
    //
    //   * REINFORCE: `ratio = exp(plp - plp.detach())` is ≡ 1 by value, so the
    //     loss VALUE is the constant `loss_normalizer·(-advantage)·num_active`
    //     and a value-FD returns 0 — zeroing out the policy gradient. The true
    //     derivative is `d(ratio)/d(plp) = 1`, so `coeff_a = -advantage ·
    //     loss_normalizer` for every active a.
    //
    //   * CISPO: the surrogate is `-weight·plp` with `weight =
    //     clip(exp(plp-ref))·advantage` DETACHED, so autograd treats `weight` as
    //     constant and `d(neg_surrogate)/d(plp) = -weight`. A value-FD would
    //     additionally (wrongly) differentiate through `weight(plp)`. We freeze
    //     `weight` and add the (value-differentiable) KL grad analytically.
    //
    // ⚠ DRIFT COUPLING: the REINFORCE / CISPO analytic coeffs below mirror the
    // exact formulas in `crate::trainer::grpo_loss` (the surrogate and the
    // K1/K3 KL terms). If those formulas change, update these in lockstep — the
    // FD validation test only covers the FD-correct Token variants, not these
    // two straight-through variants.
    let coeff: Vec<f32> = if loss_params.reinforce {
        // REINFORCE straight-through: constant per-token policy gradient.
        let c = (-loss_params.advantage * loss_params.loss_normalizer) as f32;
        vec![c; num_active]
    } else if matches!(loss_params.is_level, crate::IsLevel::Cispo) {
        // CISPO: frozen detached `weight`, plus the value-differentiable KL grad.
        // Read plp / ref to host (tiny `[num_active]`) and compute per-token.
        let plp_host: Vec<f32> = policy_log_probs
            .to_dtype(KtDType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: cispo plp host: {e}"))?;
        let ref_host: Vec<f32> = ref_log_probs_kt
            .to_dtype(KtDType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: cispo ref host: {e}"))?;
        anyhow::ensure!(
            plp_host.len() == num_active && ref_host.len() == num_active,
            "grpo_pg_loss_from_logits_grad_kt: cispo plp/ref len mismatch ({} / {} vs {num_active})",
            plp_host.len(),
            ref_host.len()
        );
        // Optional detached entropy-aware KL gate (1/0 mask over active tokens),
        // matching `grpo_loss`'s CPU quantile (only when KL is on).
        let kl_mask = cispo_entropy_kl_mask(&plp_host, &loss_params);
        let lo = 1.0 - loss_params.clip_low;
        let hi = 1.0 + loss_params.clip_high;
        (0..num_active)
            .map(|a| {
                let log_ratio = (plp_host[a] - ref_host[a]) as f64;
                let ratio = log_ratio.exp();
                let clipped = ratio.clamp(lo, hi);
                let weight = clipped * loss_params.advantage; // detached in grpo_loss
                let kl_grad = match loss_params.kl_estimator {
                    crate::KlEstimator::None => 0.0,
                    crate::KlEstimator::K1 => loss_params.kl_coeff, // d/d(plp)[k*log_ratio]
                    crate::KlEstimator::K3 => {
                        // d/d(plp)[k*(exp(-log_ratio) - 1 + log_ratio)]
                        loss_params.kl_coeff * (1.0 - (-log_ratio).exp())
                    }
                };
                let per_token = -weight + kl_mask[a] * kl_grad;
                (loss_params.loss_normalizer * per_token) as f32
            })
            .collect()
    } else {
        // Token / Sequence: value-differentiable → central finite difference of
        // the cheap `grpo_loss` scalar (the single source of truth for the math).
        let plp_f32 = policy_log_probs
            .to_dtype(KtDType::F32)
            .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: plp to_f32: {e}"))?;
        let eps: f64 = 1e-3;
        let read_scalar = |t: &KtTensor, ctx: &str| -> Result<f64> {
            let v = t
                .to_dtype(KtDType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: {ctx}: {e}"))?;
            v.first().copied().map(|x| x as f64).ok_or_else(|| {
                anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: {ctx}: empty scalar")
            })
        };
        let mut coeff: Vec<f32> = Vec::with_capacity(num_active);
        for a in 0..num_active {
            // e_a one-hot on-device.
            let mut e_host = vec![0f32; num_active];
            e_host[a] = 1.0;
            let e_a = KtTensor::from_vec_on(*device, e_host, vec![num_active])
                .map_err(|err| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: e_a: {err}"))?;
            let plp_plus = plp_f32
                .add(&e_a.affine(eps, 0.0)?)
                .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: plp+: {e}"))?;
            let plp_minus = plp_f32
                .add(&e_a.affine(-eps, 0.0)?)
                .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: plp-: {e}"))?;
            let l_plus = grpo_loss(&plp_plus, ref_log_probs_kt, loss_params, device)
                .context("grpo_pg_loss_from_logits_grad_kt: grpo_loss(+)")?;
            let l_minus = grpo_loss(&plp_minus, ref_log_probs_kt, loss_params, device)
                .context("grpo_pg_loss_from_logits_grad_kt: grpo_loss(-)")?;
            let lp = read_scalar(&l_plus, "loss(+) read")?;
            let lm = read_scalar(&l_minus, "loss(-) read")?;
            coeff.push(((lp - lm) / (2.0 * eps)) as f32);
        }
        coeff
    };

    // --- 3) Per-active-position log-softmax JVP rows: coeff_a * (onehot - softmax). ---
    //
    // Mirror `token_log_probs`'s gather: shift_logits = logits[0 .. T-1],
    // index_select the active positions, cast F32, softmax over vocab. Then
    // subtract the onehot (built on host from the labels) and scale by coeff.
    let shift_logits = logits_kt
        .squeeze(0) // [T, V]
        .and_then(|t| t.narrow(0, 0, seq_len - 1)) // [T-1, V]
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: shift_logits: {e}"))?;
    let active_idx_u32: Vec<u32> = active_positions.iter().map(|&i| i as u32).collect();
    let active_idx = KtTensor::from_vec_on(*device, active_idx_u32.clone(), vec![num_active])
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: active_idx: {e}"))?;
    let active_logits = shift_logits
        .index_select(&active_idx, 0)
        .and_then(|t| t.to_dtype(KtDType::F32)) // [num_active, V] F32
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: active_logits: {e}"))?;
    let softmax = active_logits
        .softmax_last_dim()
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: softmax: {e}"))?;

    // onehot(y_a) over vocab, y_a = input_ids[active_positions[a] + 1]. Built on
    // host then uploaded (num_active * vocab is small).
    let mut onehot_host = vec![0f32; num_active * vocab];
    for (row, &p) in active_positions.iter().enumerate() {
        let label = input_ids[p + 1] as usize;
        anyhow::ensure!(
            label < vocab,
            "grpo_pg_loss_from_logits_grad_kt: label {label} (pos {p}) >= vocab {vocab}"
        );
        onehot_host[row * vocab + label] = 1.0;
    }
    let onehot = KtTensor::from_vec_on(*device, onehot_host, vec![num_active, vocab])
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: onehot: {e}"))?;

    // jac = onehot - softmax  (the log-softmax JVP factor), [num_active, V].
    let jac = onehot
        .sub(&softmax)
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: jac: {e}"))?;

    // rows = coeff[:, None] * jac, scaled by the upstream seed grad_scalar
    // (backward is linear in the seed). Fold grad_scalar into coeff up front.
    let coeff_scaled: Vec<f32> = coeff.iter().map(|c| (*c as f64 * grad_scalar) as f32).collect();
    let coeff_col = KtTensor::from_vec_on(*device, coeff_scaled, vec![num_active, 1])
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: coeff_col: {e}"))?;
    let rows = jac
        .broadcast_mul(&coeff_col) // [num_active, V]
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: rows: {e}"))?;

    // --- 4) Scatter rows into a [T, V] zeros at the active seq positions. ---
    let grad_2d = KtTensor::zeros(vec![seq_len, vocab], KtDType::F32, *device)
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: grad zeros: {e}"))?;
    let grad_2d = grad_2d
        .index_add(&active_idx, &rows, 0)
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: scatter: {e}"))?;
    let grad_logits = grad_2d
        .unsqueeze(0) // [1, T, V]
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: unsqueeze: {e}"))?;
    Ok(grad_logits)
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
pub(crate) fn try_tape_grpo_pg_loss_from_logits_kt(
    logits_kt: &kiln_tensor::Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    device: &CdDevice,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // Full model logits only: [1, T, V] on CUDA. Defer any other shape/device to
    // the caller (the dispatch keeps non-CUDA on the candle path anyway).
    let dims = logits_kt.dims().to_vec();
    if dims.len() != 3
        || dims[0] != 1
        || dims[1] != input_ids.len()
        || action_mask.len() != input_ids.len()
        || !matches!(logits_kt.device(), kiln_tensor::Device::Cuda(_))
    {
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

    // FORWARD value — kt `token_log_probs` + `grpo_loss`. `logits_kt` is the
    // lm_head tape output passed DIRECTLY by the trainer (#1082 step 8: no more
    // [1,T,V] kt->candle copy), so recording against it keeps the tape CONNECTED
    // back through the LoRA forward (consumer input id == producer output id).
    // `ref_log_probs_kt` is the detached constant denominator.
    let policy_log_probs = token_log_probs(logits_kt, input_ids, action_mask, device)
        .context("try_tape_grpo_pg_loss_from_logits_kt: token_log_probs")?;
    let loss_kt_forward = grpo_loss(&policy_log_probs, ref_log_probs_kt, loss_params, device)
        .context("try_tape_grpo_pg_loss_from_logits_kt: grpo_loss")?;

    // Record the fused node: OUTPUT is the OWNED kt loss; the single input is the
    // CONNECTED kt logits. Saved state: kt logits (Arc bump) + host gather
    // metadata + kt ref_log_probs + params (the kt-native backward recomputes).
    let loss_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let loss_kt = loss_kt_forward;
        tape.record(
            &loss_kt,
            &[logits_kt],
            Box::new(GrpoPgLossFromLogitsBackward {
                logits: logits_kt.clone(),
                input_ids: input_ids.to_vec(),
                action_mask: action_mask.to_vec(),
                ref_log_probs: ref_log_probs_kt.clone(),
                loss_params,
                device: *device,
            }) as Box<dyn BackwardOp>,
        );
        Ok(loss_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };
    let loss_kt = loss_kt
        .context("try_tape_grpo_pg_loss_from_logits_kt: kt-tape forward failed")?;

    // (#1082 keystone) Return the kt scalar loss DIRECTLY. The caller seeds it as
    // the tape root via `with_tape_authoritative_scope_kt` (ones_like at
    // `loss_kt.id()`) — no kt->candle copy, no `register_output_mapping` candle
    // round-trip. `logits_kt` is already the recorded tape input.
    Ok(Some(loss_kt))
}

#[cfg(test)]
mod tests {
    // (#1082 C2) Finite-difference ground-truth gate for the kt-native GRPO
    // logit-grad. The whole module is CUDA-gated because the function under test
    // (`grpo_pg_loss_from_logits_grad_kt`) is `#[cfg(feature = "cuda")]`.

    #[cfg(feature = "cuda")]
    use super::grpo_pg_loss_from_logits_grad_kt;
    #[cfg(feature = "cuda")]
    use crate::trainer::{grpo_loss, token_log_probs, GrpoLossParams};
    #[cfg(feature = "cuda")]
    use crate::{IsLevel, KlEstimator};
    #[cfg(feature = "cuda")]
    use kiln_tensor::{DType as KtDType, Tensor as KtTensor};
    #[cfg(feature = "cuda")]
    use rand::rngs::StdRng;
    #[cfg(feature = "cuda")]
    use rand::{RngExt, SeedableRng};

    /// Validate the analytic kt-native GRPO logit-grad
    /// (`grpo_pg_loss_from_logits_grad_kt`, with `grad_scalar = 1.0`) against
    /// an INDEPENDENT central finite-difference of the FULL GRPO loss composite
    /// `grpo_loss(token_log_probs(logits, ...), ...)` taken directly w.r.t.
    /// selected `logits[p, j]` entries.
    ///
    /// This is NOT the same finite difference the function uses internally to
    /// derive `coeff` (that one differentiates `grpo_loss` w.r.t.
    /// `policy_log_probs`). Here we differentiate the entire composite w.r.t. the
    /// raw logits, so the analytic chain `coeff_a * (onehot - softmax)` AND the
    /// `coeff` numeric-diff AND the position/label bookkeeping are all exercised
    /// end-to-end against a logits-space ground truth.
    ///
    /// Discipline mirrors the SFT FD gate `tape_grad_matches_finite_difference_bf16`
    /// in `trainer.rs`: an eps sweep (1e-2 / 3e-2), and only entries whose FD is
    /// genuinely above the F32 noise floor AND eps-consistent feed the hard
    /// relative-tolerance assert. A blatant-disagreement tripwire fires on any
    /// above-floor, eps-consistent entry that is off by more than its own FD
    /// magnitude, so a real bug cannot hide behind the stability gate.
    ///
    /// Variants covered: (Token IS + KlEstimator::None), (Token IS + K1), and
    /// reinforce. F32 (preferred for FD precision). CUDA-only; skips gracefully
    /// when no CUDA device is present, like the other cuda tests.
    #[cfg(feature = "cuda")]
    #[test]
    fn grpo_logit_grad_matches_finite_difference_f32() {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!("[GRPO-FD] no CUDA device — skipping");
            return;
        }
        let device_kt = kiln_tensor::Device::Cuda(0);

        // --- Synthetic case: a few-token sequence, small vocab. ---
        // T tokens, V vocab. action_mask is true at a couple of completion
        // positions; the next-token shift makes the active set
        // { i in 0..T-1 : action_mask[i+1] }.
        let seq_len = 6usize;
        let vocab = 16usize;
        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2];
        // active (shifted) positions are those i where mask[i+1] is true:
        // mask = [F,F,T,T,F,T] -> shift_mask = [F,T,T,F,T] -> active shift idx
        // = {1, 2, 4} (labels input_ids[2], input_ids[3], input_ids[5]).
        let action_mask = vec![false, false, true, true, false, true];
        let active_positions: Vec<usize> = action_mask[1..]
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();
        let num_active = active_positions.len();
        assert_eq!(num_active, 3, "fixture sanity");

        // Random F32 logits [1, T, V] on CUDA (deterministic seed).
        let mut rng = StdRng::seed_from_u64(0xC0FFEE_1082_u64);
        let logits_host: Vec<f32> = (0..seq_len * vocab)
            .map(|_| rng.random_range(-2.0f32..2.0f32))
            .collect();
        let logits_kt =
            KtTensor::from_vec_on(device_kt, logits_host.clone(), vec![1, seq_len, vocab])
                .expect("logits kt");

        // Reference log-probs: a detached constant [num_active] vector built from
        // the fixture's OWN policy log-probs so the IS ratio r = exp(plp - ref) =
        // exp(0.1) ≈ 1.105 lands INSIDE the clip range [1-clip_low, 1+clip_high] =
        // [0.8, 1.2]. With random logits and an arbitrary fixed ref the ratio is
        // ~exp(-1.8) ≈ 0.17 (far below the 0.8 floor), which fully clips the PPO
        // surrogate into a flat region → ZERO gradient everywhere → a vacuous
        // test. Keeping r in-range leaves the surrogate UNCLIPPED (nonzero grad)
        // and gives log_ratio=0.1 to exercise the K1 KL term.
        let plp_fixture: Vec<f32> =
            token_log_probs(&logits_kt, &input_ids, &action_mask, &device_kt)
                .expect("fixture token_log_probs")
                .to_dtype(KtDType::F32)
                .expect("plp f32")
                .flatten_all()
                .expect("plp flat")
                .to_vec1::<f32>()
                .expect("plp host");
        let ref_host: Vec<f32> = plp_fixture.iter().map(|&p| p - 0.1).collect();
        let ref_kt =
            KtTensor::from_vec_on(device_kt, ref_host, vec![num_active]).expect("ref kt");

        // Param variants to test.
        let mk_params = |is_level: IsLevel, kl: KlEstimator, reinforce: bool| GrpoLossParams {
            advantage: -0.7,
            clip_low: 0.2,
            clip_high: 0.2,
            kl_coeff: 0.1,
            kl_estimator: kl,
            loss_normalizer: 1.0,
            is_level,
            reinforce,
            entropy_aware_kl_quantile: None,
        };
        let variants: Vec<(&str, GrpoLossParams)> = vec![
            (
                "Token+None",
                mk_params(IsLevel::Token, KlEstimator::None, false),
            ),
            ("Token+K1", mk_params(IsLevel::Token, KlEstimator::K1, false)),
            // reinforce forces ratio=1 and KL off (matches from_config()).
            (
                "reinforce",
                mk_params(IsLevel::Token, KlEstimator::None, true),
            ),
        ];

        // Scalar GRPO loss for a host logits buffer (FD reference: the FULL
        // composite, candle-free). Builds the kt logits on-device from `host`,
        // runs the kt oracle `token_log_probs` + `grpo_loss`, reads the scalar.
        let read_scalar = |t: &KtTensor| -> f64 {
            t.to_dtype(KtDType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0] as f64
        };
        let loss_of = |host: &[f32], params: &GrpoLossParams| -> f64 {
            let lk = KtTensor::from_vec_on(device_kt, host.to_vec(), vec![1, seq_len, vocab])
                .expect("fd logits kt");
            let plp = token_log_probs(&lk, &input_ids, &action_mask, &device_kt)
                .expect("fd token_log_probs");
            let loss = grpo_loss(&plp, &ref_kt, *params, &device_kt).expect("fd grpo_loss");
            read_scalar(&loss)
        };

        // FD-target entries: for each active position p (in logits index space ==
        // shift index), probe the label column y and one random other column.
        // Label columns carry the onehot term and have large, stable gradients;
        // an off-label column exercises the -softmax term.
        let mut targets: Vec<(usize, usize)> = Vec::new();
        for &p in &active_positions {
            let y = input_ids[p + 1] as usize;
            targets.push((p, y));
            let other = (y + 5) % vocab; // deterministic non-label column
            targets.push((p, if other == y { (y + 1) % vocab } else { other }));
        }

        const EPS_LIST: [f64; 2] = [1e-2, 3e-2];
        const FD_OBSERVE_MIN: f64 = 0.02; // noise floor for "informative"
        const FD_OBSERVE_SWING: f64 = 0.4; // eps-consistency for "informative"
        const FD_HARD_MIN: f64 = 0.05; // hard-assert noise floor
        const FD_HARD_SWING: f64 = 0.25; // hard-assert eps-consistency
        const FD_REL_TOL: f64 = 0.15; // pass tolerance on hard rows (two FD
                                      // schemes — composite vs grpo_loss+JVP —
                                      // differ by O(eps^2) curvature)
        const FD_REL_BLATANT: f64 = 1.0; // real-bug tripwire on observe rows

        // Per-active-row softmax over vocab from `logits_host` — the
        // `(onehot - softmax)` factor's softmax term, computed independently of
        // the function under test. Indexed by active-row order; row `a` is at
        // seq position `active_positions[a]`.
        let row_softmax = |p: usize| -> Vec<f64> {
            let base = p * vocab;
            let row: Vec<f64> = (0..vocab).map(|j| logits_host[base + j] as f64).collect();
            let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = exps.iter().sum();
            exps.iter().map(|e| e / z).collect()
        };

        for (vname, params) in &variants {
            // Analytic grad [1, T, V] (the function under test, seed 1.0).
            // (#1082 step 8) Call the kt-native grad directly — the kt inputs are
            // already built above (no candle bridge).
            let grad_kt = grpo_pg_loss_from_logits_grad_kt(
                &logits_kt,
                &input_ids,
                &action_mask,
                &ref_kt,
                *params,
                1.0,
                &device_kt,
            )
            .unwrap_or_else(|e| panic!("[GRPO-FD] {vname}: analytic grad failed: {e:?}"));
            let grad_host: Vec<f32> = grad_kt
                .to_dtype(KtDType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(
                grad_host.len(),
                seq_len * vocab,
                "[GRPO-FD] {vname}: grad shape"
            );

            if params.reinforce {
                // REINFORCE: the loss VALUE is constant in `policy_log_probs`
                // (straight-through `exp(plp - plp.detach())`), so a value-FD of
                // the composite is ~0 and useless as ground truth. Instead use
                // the CLOSED-FORM REINFORCE policy gradient — derived
                // independently of the function: for each active row a at seq
                // position p with label y,
                //   dL/d(logits[p, j]) = (-advantage * loss_normalizer)
                //                        * (onehot(y)[j] - softmax(logits[p, :])[j]).
                let coeff = -params.advantage * params.loss_normalizer;
                let mut checked = 0usize;
                for &p in &active_positions {
                    let y = input_ids[p + 1] as usize;
                    let sm = row_softmax(p);
                    for &(tp, tj) in &targets {
                        if tp != p {
                            continue;
                        }
                        let onehot = if tj == y { 1.0 } else { 0.0 };
                        let want = coeff * (onehot - sm[tj]);
                        let got = grad_host[p * vocab + tj] as f64;
                        let denom = want.abs().max(1e-6);
                        let rel = (want - got).abs() / denom;
                        eprintln!(
                            "[GRPO-FD] {vname} logits[{p},{tj}] closed-form={want:+.6} \
                             analytic={got:+.6} rel={rel:.4}"
                        );
                        // Closed form is exact (no FD noise): tight tolerance.
                        assert!(
                            rel < 1e-2 || (want - got).abs() < 1e-3,
                            "[GRPO-FD] {vname} logits[{p},{tj}]: analytic grad {got:+.6} != \
                             closed-form REINFORCE {want:+.6} (rel {rel:.4})"
                        );
                        checked += 1;
                    }
                }
                assert!(checked > 0, "[GRPO-FD] {vname}: no entry checked");
                eprintln!("[GRPO-FD] {vname}: {checked} closed-form entr(ies) — PASS");
                continue;
            }

            // Value-differentiable variants: INDEPENDENT central finite difference
            // of the FULL composite loss w.r.t. selected logits entries.
            let mut hard_rows: Vec<(usize, usize, f64, f64)> = Vec::new();
            let mut observe_rows: Vec<(usize, usize, f64, f64)> = Vec::new();
            let mut blatant: Vec<(usize, usize, f64, f64)> = Vec::new();

            for &(p, j) in &targets {
                let analytic = grad_host[p * vocab + j] as f64;
                let mut fd_by_eps: Vec<(f64, f64)> = Vec::new(); // (eps, fd)
                for &eps in &EPS_LIST {
                    let mut lp = logits_host.clone();
                    let mut lm = logits_host.clone();
                    lp[p * vocab + j] += eps as f32;
                    lm[p * vocab + j] -= eps as f32;
                    let l_plus = loss_of(&lp, params);
                    let l_minus = loss_of(&lm, params);
                    let fd = (l_plus - l_minus) / (2.0 * eps);
                    assert!(
                        fd.is_finite(),
                        "[GRPO-FD] {vname} ({p},{j}) eps={eps:.0e}: fd not finite"
                    );
                    fd_by_eps.push((eps, fd));
                }
                let fd1 = fd_by_eps[0].1; // eps 1e-2
                let fd3 = fd_by_eps[1].1; // eps 3e-2
                let denom = fd1.abs().max(1e-9);
                let rel = (fd1 - analytic).abs() / denom;
                let eps_swing = (fd1 - fd3).abs() / fd1.abs().max(fd3.abs()).max(1e-9);
                eprintln!(
                    "[GRPO-FD] {vname} logits[{p},{j}] analytic={analytic:+.6} \
                     fd_1e-2={fd1:+.6} fd_3e-2={fd3:+.6} rel={rel:.4} eps_swing={eps_swing:.4}"
                );

                if fd1.abs() <= FD_OBSERVE_MIN || eps_swing >= FD_OBSERVE_SWING {
                    // Below noise floor or eps-inconsistent: not ground truth.
                    continue;
                }
                if fd1.abs() > FD_HARD_MIN && eps_swing < FD_HARD_SWING {
                    hard_rows.push((p, j, fd1, rel));
                } else {
                    observe_rows.push((p, j, fd1, rel));
                    if rel >= FD_REL_BLATANT {
                        blatant.push((p, j, fd1, rel));
                    }
                }
            }

            if let Some((p, j, fd, rel)) = blatant.first() {
                panic!(
                    "[GRPO-FD] {vname} logits[{p},{j}]: analytic grad rel {rel:.4} >= \
                     {FD_REL_BLATANT} vs finite-diff (fd={fd:+.6}) on an above-noise, \
                     eps-consistent entry — the kt logit-grad is WRONG."
                );
            }
            assert!(
                !hard_rows.is_empty() || !observe_rows.is_empty(),
                "[GRPO-FD] {vname}: no informative finite-diff entry (|fd|>{FD_OBSERVE_MIN} AND \
                 eps_swing<{FD_OBSERVE_SWING}) — gate would be vacuous; widen targets"
            );
            for (p, j, fd, rel) in &hard_rows {
                assert!(
                    *rel < FD_REL_TOL,
                    "[GRPO-FD] {vname} logits[{p},{j}]: analytic grad rel {rel:.4} >= \
                     {FD_REL_TOL} vs finite-diff (fd={fd:+.6}) — kt logit-grad disagrees with \
                     ground truth"
                );
            }
            eprintln!(
                "[GRPO-FD] {vname}: {} hard-gated row(s), {} observe-only row(s) — PASS",
                hard_rows.len(),
                observe_rows.len()
            );
        }
    }
}

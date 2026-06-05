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
//! `crate::opd_tape_shim::try_tape_opd_scalar_mean_cuda` (the OPD reverse-KL
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
//! the log-softmax Jacobian-vector product — a pure FORWARD computation
//! (softmax plus sparse label adds), which sidesteps the kt-CUDA `index_select` / `gather`
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
//!   `grpo_loss`. CUDA/ROCm additionally use a device-resident fast path for
//!   exact token-level coefficient cases and fall back to the uniform host
//!   derivation for sequence/CISPO/entropy-aware modes.

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
use crate::cd_types::Device;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
use crate::trainer::{
    GrpoLossParams, grpo_loss, selected_logits_from_chunk_sparse, token_log_probs,
};
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
use anyhow::{Context, Result};
// (#1082) candle Tensor import removed — the GRPO PG loss adapter returns
// `kiln_tensor::Tensor` (kt-native); no candle type remains in this module.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
use kiln_autograd::{BackwardOp, Tape, tape_forward_enabled, with_active_tape};

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
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
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
    /// kt device for the analytic backward (#1082 step 8: `Device` is the kt
    /// `Device` alias; was a candle device bridged per-call).
    device: Device,
    /// This loss is recorded as the tape-authoritative scalar root, so the
    /// upstream seed is the implicit unit scalar `dL/dL = 1`. Avoid reading
    /// that scalar back from the device on the production root path.
    unit_root_grad: bool,
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn grpo_scalar_seed_or_unit_kt(
    grad_output: &kiln_tensor::Tensor,
    op_name: &str,
    unit_root_grad: bool,
) -> kiln_tensor::Result<f64> {
    if unit_root_grad {
        if grad_output.element_count() != 1 {
            return Err(kiln_tensor::Error::Msg(format!(
                "{op_name}: unit scalar root expected 1-element grad_output, got shape {:?}",
                grad_output.shape()
            )));
        }
        return Ok(1.0);
    }

    grad_output
        .to_dtype(kiln_tensor::DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| kiln_tensor::Error::Msg(format!("{op_name}: grad scalar read: {e}")))?
        .first()
        .copied()
        .ok_or_else(|| {
            kiln_tensor::Error::Msg(format!("{op_name}: empty grad_output"))
        })
        .map(|v| v as f64)
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
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
        let grad_scalar = grpo_scalar_seed_or_unit_kt(
            grad_output,
            "GrpoPgLossFromLogitsBackward",
            self.unit_root_grad,
        )?;

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
/// all-ones mask when no quantile is configured (the common case). Used by the
/// analytic coeff path so the KL grad respects the same detached gate for every
/// IS mode.
///
/// ⚠ DRIFT COUPLING with `grpo_loss`'s entropy-quantile block — keep in lockstep.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn entropy_aware_kl_mask(plp_host: &[f32], loss_params: &GrpoLossParams) -> Vec<f64> {
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

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn active_positions_and_labels(
    input_ids: &[u32],
    action_mask: &[bool],
) -> Result<(Vec<usize>, Vec<u32>)> {
    anyhow::ensure!(
        action_mask.len() == input_ids.len(),
        "GRPO active positions: action_mask length {} != input length {}",
        action_mask.len(),
        input_ids.len()
    );
    if input_ids.len() < 2 {
        return Ok((Vec::new(), Vec::new()));
    }
    let active_positions: Vec<usize> = action_mask[1..]
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();
    let active_labels: Vec<u32> = active_positions.iter().map(|&i| input_ids[i + 1]).collect();
    Ok((active_positions, active_labels))
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn grpo_loss_coeff_from_policy_log_probs_kt(
    policy_log_probs: &kiln_tensor::Tensor,
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    num_active: usize,
) -> Result<Vec<f32>> {
    use kiln_tensor::DType as KtDType;

    if num_active == 0 {
        return Ok(Vec::new());
    }
    anyhow::ensure!(
        policy_log_probs.elem_count() == num_active && ref_log_probs_kt.elem_count() == num_active,
        "grpo_loss_coeff_from_policy_log_probs_kt: policy/ref len mismatch ({} / {} vs {num_active})",
        policy_log_probs.elem_count(),
        ref_log_probs_kt.elem_count()
    );

    let coeff: Vec<f32> = if loss_params.reinforce {
        let c = (-loss_params.advantage * loss_params.loss_normalizer) as f32;
        vec![c; num_active]
    } else {
        let plp_host: Vec<f32> = policy_log_probs
            .to_dtype(KtDType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| {
                anyhow::anyhow!("grpo_loss_coeff_from_policy_log_probs_kt: plp host: {e}")
            })?;
        let ref_host: Vec<f32> = ref_log_probs_kt
            .to_dtype(KtDType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| {
                anyhow::anyhow!("grpo_loss_coeff_from_policy_log_probs_kt: ref host: {e}")
            })?;
        anyhow::ensure!(
            plp_host.len() == num_active && ref_host.len() == num_active,
            "grpo_loss_coeff_from_policy_log_probs_kt: plp/ref len mismatch ({} / {} vs {num_active})",
            plp_host.len(),
            ref_host.len()
        );
        let lo = 1.0 - loss_params.clip_low;
        let hi = 1.0 + loss_params.clip_high;
        let log_ratios: Vec<f64> = plp_host
            .iter()
            .zip(ref_host.iter())
            .map(|(&p, &r)| (p - r) as f64)
            .collect();
        let kl_mask = entropy_aware_kl_mask(&plp_host, &loss_params);
        let kl_grad = |log_ratio: f64| -> f64 {
            match loss_params.kl_estimator {
                crate::KlEstimator::None => 0.0,
                crate::KlEstimator::K1 => loss_params.kl_coeff,
                crate::KlEstimator::K3 => loss_params.kl_coeff * (1.0 - (-log_ratio).exp()),
            }
        };

        match loss_params.is_level {
            crate::IsLevel::Token => log_ratios
                .iter()
                .enumerate()
                .map(|(a, &log_ratio)| {
                    let ratio = log_ratio.exp();
                    let pg_grad = if loss_params.advantage >= 0.0 {
                        if ratio <= hi {
                            -loss_params.advantage * ratio
                        } else {
                            0.0
                        }
                    } else if ratio >= lo {
                        -loss_params.advantage * ratio
                    } else {
                        0.0
                    };
                    let per_token = pg_grad + kl_mask[a] * kl_grad(log_ratio);
                    (loss_params.loss_normalizer * per_token) as f32
                })
                .collect(),
            crate::IsLevel::Sequence => {
                let mean_log_ratio = log_ratios.iter().sum::<f64>() / num_active as f64;
                let seq_ratio = mean_log_ratio.exp();
                let pg_grad = if loss_params.advantage >= 0.0 {
                    if seq_ratio <= hi {
                        -loss_params.advantage * seq_ratio / num_active as f64
                    } else {
                        0.0
                    }
                } else if seq_ratio >= lo {
                    -loss_params.advantage * seq_ratio / num_active as f64
                } else {
                    0.0
                };
                log_ratios
                    .iter()
                    .enumerate()
                    .map(|(a, &log_ratio)| {
                        let per_token = pg_grad + kl_mask[a] * kl_grad(log_ratio);
                        (loss_params.loss_normalizer * per_token) as f32
                    })
                    .collect()
            }
            crate::IsLevel::Cispo => log_ratios
                .iter()
                .enumerate()
                .map(|(a, &log_ratio)| {
                    let ratio = log_ratio.exp();
                    let clipped = ratio.clamp(lo, hi);
                    let weight = clipped * loss_params.advantage;
                    let per_token = -weight + kl_mask[a] * kl_grad(log_ratio);
                    (loss_params.loss_normalizer * per_token) as f32
                })
                .collect(),
        }
    };

    Ok(coeff)
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn grpo_loss_coeff_col_device_fast_path_kt(
    policy_log_probs: &kiln_tensor::Tensor,
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    num_active: usize,
    grad_scalar: f64,
    device: &Device,
) -> Result<Option<kiln_tensor::Tensor>> {
    use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

    if num_active == 0 {
        return Ok(Some(KtTensor::zeros(vec![0, 1], KtDType::F32, *device)?));
    }
    anyhow::ensure!(
        policy_log_probs.elem_count() == num_active && ref_log_probs_kt.elem_count() == num_active,
        "grpo_loss_coeff_col_device_fast_path_kt: policy/ref len mismatch ({} / {} vs {num_active})",
        policy_log_probs.elem_count(),
        ref_log_probs_kt.elem_count()
    );

    // Entropy-aware KL intentionally computes a detached quantile over active
    // policy log-probs. Keep that path on the exact host derivation for now.
    if loss_params.entropy_aware_kl_quantile.is_some() {
        return Ok(None);
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    let device_supported = matches!(device, Device::Cuda(_) | Device::Rocm(_));
    #[cfg(not(any(feature = "cuda", feature = "rocm")))]
    let device_supported = false;
    if !device_supported || !matches!(loss_params.is_level, crate::IsLevel::Token) {
        return Ok(None);
    }

    if loss_params.reinforce {
        let coeff = -loss_params.advantage * loss_params.loss_normalizer * grad_scalar;
        return KtTensor::ones(vec![num_active, 1], KtDType::F32, device)
            .and_then(|t| t.affine(coeff, 0.0))
            .map(Some)
            .map_err(Into::into);
    }

    let policy = policy_log_probs
        .to_dtype(KtDType::F32)?
        .flatten_all()?
        .reshape(vec![num_active])?
        .contiguous()?;
    let reference = ref_log_probs_kt
        .to_dtype(KtDType::F32)?
        .flatten_all()?
        .reshape(vec![num_active])?
        .contiguous()?;
    let log_ratio = (&policy - &reference)?.contiguous()?;
    let ratio = log_ratio.exp()?.contiguous()?;

    let kl_grad = match loss_params.kl_estimator {
        crate::KlEstimator::None => ratio.affine(0.0, 0.0)?,
        crate::KlEstimator::K1 => ratio.affine(0.0, loss_params.kl_coeff)?,
        crate::KlEstimator::K3 => {
            let exp_neg_log_ratio = log_ratio.neg()?.exp()?;
            exp_neg_log_ratio.affine(-loss_params.kl_coeff, loss_params.kl_coeff)?
        }
    };

    let pg_raw = ratio.affine(-loss_params.advantage, 0.0)?.contiguous()?;
    let zero = ratio.affine(0.0, 0.0)?.contiguous()?;
    let pg_grad = if loss_params.advantage >= 0.0 {
        let hi = ratio.affine(0.0, 1.0 + loss_params.clip_high)?.contiguous()?;
        let unclipped = kiln_tensor::ops::le(&ratio, &hi)?;
        unclipped.where_cond(&pg_raw, &zero)?
    } else {
        let lo = ratio.affine(0.0, 1.0 - loss_params.clip_low)?.contiguous()?;
        let unclipped = kiln_tensor::ops::ge(&ratio, &lo)?;
        unclipped.where_cond(&pg_raw, &zero)?
    };

    let coeff = (&pg_grad + &kl_grad)?
        .affine(loss_params.loss_normalizer * grad_scalar, 0.0)?
        .reshape(vec![num_active, 1])?
        .contiguous()?;
    Ok(Some(coeff))
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[derive(Debug)]
struct ChunkedSelectedLogProbState {
    policy_log_probs: kiln_tensor::Tensor,
    running_max: kiln_tensor::Tensor,
    running_sumexp: kiln_tensor::Tensor,
    /// Device-resident shifted positions for active tokens. Built once during
    /// the chunked forward and reused by the analytic backward to avoid a
    /// second long-context mask scan and H2D upload.
    active_idx: kiln_tensor::Tensor,
    active_labels: Vec<u32>,
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn selected_log_probs_from_normed_hidden_chunked_state_kt(
    normed_hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    chunk_size: usize,
    device: &Device,
) -> Result<ChunkedSelectedLogProbState> {
    use kiln_tensor::{D as KtDim, DType as KtDType, Tensor as KtTensor};

    if chunk_size == 0 {
        anyhow::bail!("selected_log_probs_from_normed_hidden_chunked_kt: chunk_size must be > 0");
    }
    let seq_len = input_ids.len();
    let dims = normed_hidden.dims().to_vec();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] == seq_len,
        "selected_log_probs_from_normed_hidden_chunked_kt: hidden must be [1, seq_len, hidden], got {dims:?} for seq_len {seq_len}"
    );
    let hidden_size = dims[2];
    anyhow::ensure!(
        head_t.dims().len() == 2 && head_t.dims()[0] == hidden_size,
        "selected_log_probs_from_normed_hidden_chunked_kt: head_t must be [hidden, vocab], got {:?}",
        head_t.dims()
    );
    let (active_positions, active_labels) = active_positions_and_labels(input_ids, action_mask)?;
    let num_active = active_positions.len();
    if num_active == 0 {
        return Ok(ChunkedSelectedLogProbState {
            policy_log_probs: KtTensor::zeros(vec![1], KtDType::F32, *device)?,
            running_max: KtTensor::zeros(vec![1, 1], KtDType::F32, *device)?,
            running_sumexp: KtTensor::zeros(vec![1, 1], KtDType::F32, *device)?,
            active_idx: KtTensor::from_vec_on(*device, Vec::<u32>::new(), vec![0])?,
            active_labels,
        });
    }

    let active_idx_u32: Vec<u32> = active_positions.iter().map(|&i| i as u32).collect();
    let active_idx = KtTensor::from_vec_on(*device, active_idx_u32, vec![num_active])?;
    let active_hidden = normed_hidden
        .squeeze(0)?
        .narrow(0, 0, seq_len - 1)?
        .index_select(&active_idx, 0)?
        .to_dtype(KtDType::F32)?;
    let head_t_f32 = head_t.to_dtype(KtDType::F32)?;
    let vocab_size = head_t_f32.dim(1)?;
    anyhow::ensure!(
        vocab_size > 0,
        "selected_log_probs_from_normed_hidden_chunked_kt: empty vocab"
    );

    let mut running_max: Option<KtTensor> = None;
    let mut running_sumexp: Option<KtTensor> = None;
    let mut correct_logits: Option<KtTensor> = None;
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let chunk_end = chunk_start + chunk_len;
        let head_chunk = head_t_f32.narrow(1, chunk_start, chunk_len)?.contiguous()?;
        let logits_chunk = active_hidden.matmul(&head_chunk)?;
        let chunk_max = logits_chunk.max_keepdim(KtDim::Minus1)?;
        let (new_max, new_sumexp) = match (running_max.as_ref(), running_sumexp.as_ref()) {
            (None, None) => {
                let shifted = (&logits_chunk - chunk_max.broadcast_as(logits_chunk.shape())?)?;
                let chunk_sumexp = shifted.exp()?.sum_keepdim(KtDim::Minus1)?;
                (chunk_max.detach(), chunk_sumexp.detach())
            }
            (Some(prev_max), Some(prev_sumexp)) => {
                let new_max = prev_max.maximum(&chunk_max)?;
                let prev_scale = (prev_max - &new_max)?.exp()?;
                let scaled_prev = prev_sumexp.broadcast_mul(&prev_scale)?;
                let shifted = (&logits_chunk - new_max.broadcast_as(logits_chunk.shape())?)?;
                let chunk_sumexp = shifted.exp()?.sum_keepdim(KtDim::Minus1)?;
                let new_sumexp = (scaled_prev + chunk_sumexp)?;
                (new_max.detach(), new_sumexp.detach())
            }
            _ => unreachable!("running max/sumexp are set together"),
        };
        running_max = Some(new_max);
        running_sumexp = Some(new_sumexp);

        let chunk_correct = selected_logits_from_chunk_sparse(
            &logits_chunk,
            &active_labels,
            chunk_start,
            chunk_len,
            vocab_size,
            device,
            "selected_log_probs_from_normed_hidden_chunked_kt",
        )?;
        correct_logits = Some(match correct_logits.as_ref() {
            Some(prev) => (prev + chunk_correct)?.detach(),
            None => chunk_correct.detach(),
        });
        chunk_start = chunk_end;
    }

    let running_max = running_max
        .context("selected_log_probs_from_normed_hidden_chunked_kt: vocab_size was zero")?;
    let running_sumexp = running_sumexp
        .context("selected_log_probs_from_normed_hidden_chunked_kt: vocab_size was zero")?;
    let correct_logits = correct_logits
        .context("selected_log_probs_from_normed_hidden_chunked_kt: vocab_size was zero")?;
    let running_log_sumexp = running_sumexp.log()?;
    let log_sum_exp = (&running_max + &running_log_sumexp)?;
    let policy_log_probs = (correct_logits - log_sum_exp)?.squeeze(1)?;
    Ok(ChunkedSelectedLogProbState {
        policy_log_probs,
        running_max,
        running_sumexp,
        active_idx,
        active_labels,
    })
}

#[cfg(all(
    test,
    any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )
))]
pub(crate) fn selected_log_probs_from_normed_hidden_chunked_kt(
    normed_hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    chunk_size: usize,
    device: &Device,
) -> Result<kiln_tensor::Tensor> {
    selected_log_probs_from_normed_hidden_chunked_state_kt(
        normed_hidden,
        head_t,
        input_ids,
        action_mask,
        chunk_size,
        device,
    )
    .map(|state| state.policy_log_probs)
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[derive(Debug)]
struct GrpoPgLossFromNormedHiddenBackward {
    normed_hidden: kiln_tensor::Tensor,
    head_t: kiln_tensor::Tensor,
    policy_log_probs: kiln_tensor::Tensor,
    running_max: kiln_tensor::Tensor,
    running_sumexp: kiln_tensor::Tensor,
    active_idx: kiln_tensor::Tensor,
    active_labels: Vec<u32>,
    ref_log_probs: kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    device: Device,
    chunk_size: usize,
    /// This loss is recorded as the tape-authoritative scalar root, so the
    /// upstream seed is the implicit unit scalar `dL/dL = 1`. Avoid reading
    /// that scalar back from the device on the production long-context path.
    unit_root_grad: bool,
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
impl BackwardOp for GrpoPgLossFromNormedHiddenBackward {
    fn name(&self) -> &'static str {
        "grpo_pg_loss_from_normed_hidden_backward"
    }

    fn input_count(&self) -> usize {
        2
    }

    fn requires_input(&self, _idx: usize) -> bool {
        false
    }

    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let grad_scalar = grpo_scalar_seed_or_unit_kt(
            grad_output,
            "GrpoPgLossFromNormedHiddenBackward",
            self.unit_root_grad,
        )?;

        let grad_hidden = grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt(
            &self.normed_hidden,
            &self.head_t,
            &self.active_idx,
            &self.active_labels,
            &self.policy_log_probs,
            &self.ref_log_probs,
            self.loss_params,
            grad_scalar,
            &self.device,
            self.chunk_size,
            &self.running_max,
            &self.running_sumexp,
        )
        .map_err(|e| {
            kiln_tensor::Error::Msg(format!(
                "GrpoPgLossFromNormedHiddenBackward: analytic hidden-grad: {e:#}"
            ))
        })?;

        Ok(vec![Some(grad_hidden), None])
    }
}

#[cfg(all(
    test,
    any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    )
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn grpo_pg_loss_from_normed_hidden_grad_kt(
    normed_hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    policy_log_probs_kt: &kiln_tensor::Tensor,
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    grad_scalar: f64,
    device: &Device,
    chunk_size: usize,
) -> Result<kiln_tensor::Tensor> {
    let state = selected_log_probs_from_normed_hidden_chunked_state_kt(
        normed_hidden,
        head_t,
        input_ids,
        action_mask,
        chunk_size,
        device,
    )
    .context("grpo_pg_loss_from_normed_hidden_grad_kt: selected log-prob state")?;
    grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt(
        normed_hidden,
        head_t,
        &state.active_idx,
        &state.active_labels,
        policy_log_probs_kt,
        ref_log_probs_kt,
        loss_params,
        grad_scalar,
        device,
        chunk_size,
        &state.running_max,
        &state.running_sumexp,
    )
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
fn grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt(
    normed_hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    active_idx: &kiln_tensor::Tensor,
    active_labels: &[u32],
    policy_log_probs_kt: &kiln_tensor::Tensor,
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    grad_scalar: f64,
    device: &Device,
    chunk_size: usize,
    running_max: &kiln_tensor::Tensor,
    running_sumexp: &kiln_tensor::Tensor,
) -> Result<kiln_tensor::Tensor> {
    use kiln_tensor::ops::{mul_scalar, scatter_add};
    use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

    if chunk_size == 0 {
        anyhow::bail!("grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: chunk_size must be > 0");
    }
    let dims = normed_hidden.dims().to_vec();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1,
        "grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: hidden must be [1, seq_len, hidden], got {dims:?}"
    );
    let seq_len = dims[1];
    let hidden_size = dims[2];
    anyhow::ensure!(
        head_t.dims().len() == 2 && head_t.dims()[0] == hidden_size,
        "grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: head_t must be [hidden, vocab], got {:?}",
        head_t.dims()
    );
    let vocab = head_t.dim(1)?;
    let hidden_dtype = normed_hidden.dtype();
    let num_active = active_labels.len();
    anyhow::ensure!(
        active_idx.dims() == [num_active],
        "grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: active_idx shape {:?} != [{num_active}]",
        active_idx.dims()
    );
    anyhow::ensure!(
        active_idx.dtype() == KtDType::U32,
        "grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: active_idx dtype {:?} != U32",
        active_idx.dtype()
    );
    if num_active == 0 {
        return KtTensor::zeros(vec![1, seq_len, hidden_size], KtDType::F32, *device)
            .and_then(|t| t.to_dtype(hidden_dtype))
            .map_err(Into::into);
    }
    anyhow::ensure!(
        running_max.dims() == [num_active, 1],
        "grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: running_max shape {:?} != [{num_active}, 1]",
        running_max.dims()
    );
    anyhow::ensure!(
        running_sumexp.dims() == [num_active, 1],
        "grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: running_sumexp shape {:?} != [{num_active}, 1]",
        running_sumexp.dims()
    );
    for &label in active_labels {
        anyhow::ensure!(
            (label as usize) < vocab,
            "grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: label {label} outside vocab size {vocab}"
        );
    }

    let active_hidden = normed_hidden
        .squeeze(0)?
        .narrow(0, 0, seq_len - 1)?
        .index_select(active_idx, 0)?
        .to_dtype(KtDType::F32)?;
    let head_t_f32 = head_t.to_dtype(KtDType::F32)?;
    anyhow::ensure!(
        vocab > 0,
        "grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: empty vocab"
    );

    let coeff_col = match grpo_loss_coeff_col_device_fast_path_kt(
        policy_log_probs_kt,
        ref_log_probs_kt,
        loss_params,
        num_active,
        grad_scalar,
        device,
    )
    .context("grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: coeff fast path")?
    {
        Some(coeff_col) => coeff_col,
        None => {
            let coeff = grpo_loss_coeff_from_policy_log_probs_kt(
                policy_log_probs_kt,
                ref_log_probs_kt,
                loss_params,
                num_active,
            )
            .context("grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: coeff")?;
            anyhow::ensure!(
                coeff.len() == num_active,
                "grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: coeff len {} != num_active {num_active}",
                coeff.len()
            );
            let coeff_scaled: Vec<f32> = coeff
                .iter()
                .map(|c| (*c as f64 * grad_scalar) as f32)
                .collect();
            KtTensor::from_vec_on(*device, coeff_scaled, vec![num_active, 1])
                .context("grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: coeff_col")?
        }
    };
    let mut grad_active_hidden =
        KtTensor::zeros(vec![num_active, hidden_size], KtDType::F32, *device)
            .context("grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt: grad_active_hidden zeros")?;

    let mut chunk_start = 0usize;
    while chunk_start < vocab {
        let chunk_len = chunk_size.min(vocab - chunk_start);
        let chunk_end = chunk_start + chunk_len;
        let head_chunk = head_t_f32.narrow(1, chunk_start, chunk_len)?.contiguous()?;
        let logits_chunk = active_hidden.matmul(&head_chunk)?;
        let shifted = (&logits_chunk - running_max.broadcast_as(logits_chunk.shape())?)?;
        let exp_chunk = shifted.exp()?;
        let softmax_chunk =
            exp_chunk.broadcast_div(&running_sumexp.broadcast_as(logits_chunk.shape())?)?;

        let head_chunk_t = head_chunk.t()?.contiguous()?;
        let softmax_rows = softmax_chunk.broadcast_mul(&coeff_col)?;
        let softmax_contrib = softmax_rows.matmul(&head_chunk_t)?;

        let mut row_hits = Vec::new();
        let mut rel_hits = Vec::new();
        for (row_idx, &label) in active_labels.iter().enumerate() {
            let label = label as usize;
            if label >= chunk_start && label < chunk_end {
                row_hits.push(row_idx as u32);
                rel_hits.push((label - chunk_start) as u32);
            }
        }
        let chunk_contrib = if row_hits.is_empty() {
            mul_scalar(&softmax_contrib, -1.0)?
        } else {
            let hits = row_hits.len();
            let row_idx = KtTensor::from_vec_on(*device, row_hits, vec![hits])?;
            let rel_idx = KtTensor::from_vec_on(*device, rel_hits, vec![hits])?;
            let selected_head_rows = head_chunk_t.index_select(&rel_idx, 0)?;
            let selected_coeff = coeff_col.index_select(&row_idx, 0)?;
            let selected_coeff_b = selected_coeff.broadcast_as(selected_head_rows.shape())?;
            let selected_rows = selected_head_rows.broadcast_mul(&selected_coeff_b)?;
            let selected_contrib = scatter_add(
                &selected_rows.contiguous()?,
                0,
                &row_idx,
                num_active,
            )?;
            (selected_contrib - softmax_contrib)?
        };
        grad_active_hidden = (&grad_active_hidden + chunk_contrib)?.detach();
        chunk_start = chunk_end;
    }

    let grad_hidden_2d = scatter_add(
        &grad_active_hidden.contiguous()?,
        0,
        active_idx,
        seq_len,
    )?;
    let grad_hidden = grad_hidden_2d.unsqueeze(0)?.to_dtype(hidden_dtype)?;
    Ok(grad_hidden)
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn grpo_pg_loss_from_normed_hidden_loss_and_grad_kt(
    normed_hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    grad_scalar: f64,
    device: &Device,
    chunk_size: usize,
) -> Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    let state = selected_log_probs_from_normed_hidden_chunked_state_kt(
        normed_hidden,
        head_t,
        input_ids,
        action_mask,
        chunk_size,
        device,
    )
    .context("grpo_pg_loss_from_normed_hidden_loss_and_grad_kt: selected log-prob state")?;
    let loss_kt = grpo_loss(
        &state.policy_log_probs,
        ref_log_probs_kt,
        loss_params,
        device,
    )
    .context("grpo_pg_loss_from_normed_hidden_loss_and_grad_kt: grpo_loss")?;
    let grad_hidden = grpo_pg_loss_from_normed_hidden_grad_with_logsum_kt(
        normed_hidden,
        head_t,
        &state.active_idx,
        &state.active_labels,
        &state.policy_log_probs,
        ref_log_probs_kt,
        loss_params,
        grad_scalar,
        device,
        chunk_size,
        &state.running_max,
        &state.running_sumexp,
    )
    .context("grpo_pg_loss_from_normed_hidden_loss_and_grad_kt: hidden grad")?;
    Ok((loss_kt, grad_hidden))
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
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
fn grpo_pg_loss_from_logits_grad_kt(
    logits_kt: &kiln_tensor::Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    grad_scalar: f64,
    device: &Device,
) -> Result<kiln_tensor::Tensor> {
    use kiln_tensor::ops::mul_scalar;
    use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

    let seq_len = input_ids.len();
    let dims = logits_kt.dims().to_vec();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1 && dims[1] == seq_len,
        "grpo_pg_loss_from_logits_grad_kt: logits must be [1, seq_len, vocab], got {dims:?} for \
         seq_len {seq_len}"
    );
    let vocab = dims[2];
    // The grad must come back in the SAVED logits dtype so it threads cleanly
    // into the (BF16) model/LoRA backward matmuls — exactly as the SFT
    // `cross_entropy_from_logits_grad_candle` casts its final grad to
    // `logits_dtype`. Returning F32 here severs the BF16 LoRA backward with a
    // `MatmulOp: dtype mismatch` (the prior parity gap on Metal). All interior
    // math stays F32; only the returned tensor is cast.
    let logits_dtype = logits_kt.dtype();

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
            .and_then(|t| t.to_dtype(logits_dtype))
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

    // --- 3) Per-active-position log-softmax JVP rows: coeff_a * (onehot - softmax). ---
    //
    // Mirror `token_log_probs`'s gather: shift_logits = logits[0 .. T-1],
    // index_select the active positions, cast F32, softmax over vocab. Build the
    // dense `-coeff * softmax` rows, then add the sparse `+coeff` label terms by
    // flattened `index_add` below. This avoids a second dense [num_active, vocab]
    // host one-hot allocation/upload in the backward.
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

    let coeff_col = match grpo_loss_coeff_col_device_fast_path_kt(
        &policy_log_probs,
        ref_log_probs_kt,
        loss_params,
        num_active,
        grad_scalar,
        device,
    )
    .context("grpo_pg_loss_from_logits_grad_kt: coeff fast path")?
    {
        Some(coeff_col) => coeff_col,
        None => {
            // coeff_a = dL/d(policy_log_prob_a).
            let coeff = grpo_loss_coeff_from_policy_log_probs_kt(
                &policy_log_probs,
                ref_log_probs_kt,
                loss_params,
                num_active,
            )
            .context("grpo_pg_loss_from_logits_grad_kt: coeff")?;
            let coeff_scaled: Vec<f32> = coeff
                .iter()
                .map(|c| (*c as f64 * grad_scalar) as f32)
                .collect();
            KtTensor::from_vec_on(*device, coeff_scaled, vec![num_active, 1])
                .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: coeff_col: {e}"))?
        }
    };
    let softmax_rows = softmax
        .broadcast_mul(&coeff_col) // [num_active, V]
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: softmax rows: {e}"))?;
    let rows = mul_scalar(&softmax_rows, -1.0)
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: neg softmax rows: {e}"))?;

    let mut label_flat_indices = Vec::with_capacity(num_active);
    for &p in &active_positions {
        let label = input_ids[p + 1] as usize;
        anyhow::ensure!(
            label < vocab,
            "grpo_pg_loss_from_logits_grad_kt: label {label} (pos {p}) >= vocab {vocab}"
        );
        let flat = p
            .checked_mul(vocab)
            .and_then(|base| base.checked_add(label))
            .ok_or_else(|| {
                anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: flat label index overflow")
            })?;
        label_flat_indices.push(u32::try_from(flat).with_context(|| {
            format!("grpo_pg_loss_from_logits_grad_kt: flat index {flat} exceeds u32 range")
        })?);
    }

    // --- 4) Scatter rows into a [T, V] zeros at the active seq positions. ---
    let grad_2d = KtTensor::zeros(vec![seq_len, vocab], KtDType::F32, *device)
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: grad zeros: {e}"))?;
    let grad_2d_neg_softmax = grad_2d
        .index_add(&active_idx, &rows, 0)
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: scatter softmax rows: {e}"))?;
    let label_idx = KtTensor::from_vec_on(*device, label_flat_indices, vec![num_active])
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: label_idx: {e}"))?;
    let label_coeff = coeff_col
        .squeeze(1)
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: label coeff: {e}"))?;
    let grad_2d = grad_2d_neg_softmax
        .flatten_all()
        .and_then(|flat| flat.index_add(&label_idx, &label_coeff, 0usize))
        .and_then(|flat| flat.reshape(vec![seq_len, vocab]))
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: scatter label coeffs: {e}"))?;
    let grad_logits = grad_2d
        .unsqueeze(0) // [1, T, V]
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: unsqueeze: {e}"))?;
    // Cast back to the saved logits dtype (BF16 in production) so the grad
    // threads into the LoRA backward matmuls (parity with the SFT CE grad).
    let grad_logits = grad_logits
        .to_dtype(logits_dtype)
        .map_err(|e| anyhow::anyhow!("grpo_pg_loss_from_logits_grad_kt: grad cast: {e}"))?;
    Ok(grad_logits)
}

/// Attempt to root the GRPO scalar PG (+ KL) loss at a SINGLE fused kt `Tape`
/// node taking the FULL `[1, T, V]` policy logits.
///
/// Mirrors `kiln_model::tape_forward::try_tape_cross_entropy_from_logits_cuda`
/// (SFT) and `crate::opd_tape_shim::try_tape_opd_scalar_mean_cuda` (OPD): the
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
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub(crate) fn try_tape_grpo_pg_loss_from_logits_kt(
    logits_kt: &kiln_tensor::Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    device: &Device,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }

    // Full model logits only: [1, T, V] on a GPU device. Defer any other
    // shape/device to the caller (the dispatch keeps non-GPU on the candle path
    // anyway). (#1082) Vulkan added: the GRPO backward
    // (`grpo_pg_loss_from_logits_grad_kt`) is a device-agnostic pure-kt
    // composite reachable on Vulkan, so the adapter must not decline Vulkan
    // logits here (declining would yield an empty grad store on F32 Vulkan
    // GRPO — the silent-empty bug PR6 left behind).
    let dims = logits_kt.dims().to_vec();
    if dims.len() != 3
        || dims[0] != 1
        || dims[1] != input_ids.len()
        || action_mask.len() != input_ids.len()
        || !matches!(
            logits_kt.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        )
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
                unit_root_grad: true,
            }) as Box<dyn BackwardOp>,
        );
        Ok(loss_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };
    let loss_kt =
        loss_kt.context("try_tape_grpo_pg_loss_from_logits_kt: kt-tape forward failed")?;

    // (#1082 keystone) Return the kt scalar loss DIRECTLY. The caller seeds it as
    // the tape root via `with_tape_authoritative_scope_kt` (ones_like at
    // `loss_kt.id()`) — no kt->candle copy, no `register_output_mapping` candle
    // round-trip. `logits_kt` is already the recorded tape input.
    Ok(Some(loss_kt))
}

/// Root the GRPO scalar PG (+ KL) loss at post-final-RMSNorm hidden instead of
/// full policy logits.
///
/// This is the long-context training path: forward computes selected log-probs
/// by chunking the frozen tied head over vocab, and backward emits only
/// `dL/d(normed_hidden)` by replaying those same chunks. The full `[T, V]`
/// policy logits and full `[T, V]` logit gradient are never materialized.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_tape_grpo_pg_loss_from_normed_hidden_kt(
    normed_hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    input_ids: &[u32],
    action_mask: &[bool],
    ref_log_probs_kt: &kiln_tensor::Tensor,
    loss_params: GrpoLossParams,
    device: &Device,
    chunk_size: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    if !tape_forward_enabled() {
        return Ok(None);
    }
    let dims = normed_hidden.dims().to_vec();
    if dims.len() != 3
        || dims[0] != 1
        || dims[1] != input_ids.len()
        || action_mask.len() != input_ids.len()
        || !matches!(
            normed_hidden.device(),
            kiln_tensor::Device::Cuda(_)
                | kiln_tensor::Device::Metal(_)
                | kiln_tensor::Device::Vulkan(_)
                | kiln_tensor::Device::Rocm(_)
        )
    {
        return Ok(None);
    }
    let hidden_size = dims[2];
    if head_t.dims().len() != 2 || head_t.dims()[0] != hidden_size {
        return Ok(None);
    }
    anyhow::ensure!(
        chunk_size > 0,
        "try_tape_grpo_pg_loss_from_normed_hidden_kt: chunk_size must be > 0"
    );

    let selected_state = selected_log_probs_from_normed_hidden_chunked_state_kt(
        normed_hidden,
        head_t,
        input_ids,
        action_mask,
        chunk_size,
        device,
    )
    .context("try_tape_grpo_pg_loss_from_normed_hidden_kt: selected log-probs")?;
    if selected_state.active_labels.is_empty() {
        return Ok(None);
    }
    let loss_kt_forward = grpo_loss(
        &selected_state.policy_log_probs,
        ref_log_probs_kt,
        loss_params,
        device,
    )
        .context("try_tape_grpo_pg_loss_from_normed_hidden_kt: grpo_loss")?;

    let loss_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        let loss_kt = loss_kt_forward;
        tape.record(
            &loss_kt,
            &[normed_hidden, head_t],
            Box::new(GrpoPgLossFromNormedHiddenBackward {
                normed_hidden: normed_hidden.clone(),
                head_t: head_t.clone(),
                policy_log_probs: selected_state.policy_log_probs.clone(),
                running_max: selected_state.running_max.clone(),
                running_sumexp: selected_state.running_sumexp.clone(),
                active_idx: selected_state.active_idx.clone(),
                active_labels: selected_state.active_labels.clone(),
                ref_log_probs: ref_log_probs_kt.clone(),
                loss_params,
                device: *device,
                chunk_size,
                unit_root_grad: true,
            }) as Box<dyn BackwardOp>,
        );
        Ok(loss_kt)
    }) {
        Some(result) => result,
        None => return Ok(None),
    };
    let loss_kt =
        loss_kt.context("try_tape_grpo_pg_loss_from_normed_hidden_kt: kt-tape forward failed")?;
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
    use crate::trainer::{GrpoLossParams, grpo_loss, token_log_probs};
    #[cfg(feature = "cuda")]
    use crate::{IsLevel, KlEstimator};
    #[cfg(feature = "cuda")]
    use kiln_tensor::{DType as KtDType, Tensor as KtTensor};
    #[cfg(feature = "cuda")]
    use rand::rngs::StdRng;
    #[cfg(feature = "cuda")]
    use rand::{RngExt, SeedableRng};

    #[cfg(feature = "cuda")]
    #[test]
    fn grpo_normed_hidden_chunked_matches_full_logits_cpu() {
        let device = kiln_tensor::Device::Cpu;
        let (seq_len, hidden_size, vocab) = (6usize, 5usize, 13usize);
        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2];
        let action_mask = vec![false, false, true, true, false, true];
        let num_active = action_mask[1..].iter().filter(|m| **m).count();
        let hidden_host: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (((i * 7) % 19) as f32 - 9.0) * 0.041)
            .collect();
        let head_host: Vec<f32> = (0..hidden_size * vocab)
            .map(|i| (((i * 11) % 23) as f32 - 11.0) * 0.037)
            .collect();
        let normed_hidden =
            KtTensor::from_vec_on(device, hidden_host, vec![1, seq_len, hidden_size])
                .expect("hidden");
        let head_t =
            KtTensor::from_vec_on(device, head_host, vec![hidden_size, vocab]).expect("head");
        let logits = normed_hidden
            .squeeze(0)
            .unwrap()
            .matmul(&head_t)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        let plp_full = token_log_probs(&logits, &input_ids, &action_mask, &device).unwrap();
        let plp_hidden = super::selected_log_probs_from_normed_hidden_chunked_kt(
            &normed_hidden,
            &head_t,
            &input_ids,
            &action_mask,
            3,
            &device,
        )
        .unwrap();
        let read = |t: &KtTensor| -> Vec<f32> {
            t.to_dtype(KtDType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let full_vals = read(&plp_full);
        let hidden_vals = read(&plp_hidden);
        assert_eq!(full_vals.len(), hidden_vals.len());
        for (a, (full, hidden)) in full_vals.iter().zip(hidden_vals.iter()).enumerate() {
            assert!(
                (full - hidden).abs() < 1e-5,
                "policy log-prob drift at active {a}: full={full} hidden={hidden}"
            );
        }

        let ref_host: Vec<f32> = full_vals.iter().map(|&p| p - 0.1).collect();
        let ref_kt = KtTensor::from_vec_on(device, ref_host, vec![num_active]).expect("ref");
        let variants: Vec<(&str, GrpoLossParams)> = vec![
            (
                "token-k1",
                GrpoLossParams {
                    advantage: -0.7,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.1,
                    kl_estimator: KlEstimator::K1,
                    loss_normalizer: 1.0,
                    is_level: IsLevel::Token,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
            ),
            (
                "sequence-k3",
                GrpoLossParams {
                    advantage: 0.6,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.05,
                    kl_estimator: KlEstimator::K3,
                    loss_normalizer: 0.5,
                    is_level: IsLevel::Sequence,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
            ),
            (
                "cispo-k1",
                GrpoLossParams {
                    advantage: -0.4,
                    clip_low: 0.15,
                    clip_high: 0.25,
                    kl_coeff: 0.03,
                    kl_estimator: KlEstimator::K1,
                    loss_normalizer: 0.75,
                    is_level: IsLevel::Cispo,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
            ),
            (
                "reinforce",
                GrpoLossParams {
                    advantage: 0.9,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.0,
                    kl_estimator: KlEstimator::None,
                    loss_normalizer: 1.25,
                    is_level: IsLevel::Token,
                    reinforce: true,
                    entropy_aware_kl_quantile: None,
                },
            ),
        ];

        let head_t_t = head_t.t().unwrap().contiguous().unwrap();
        for (name, params) in variants {
            let grad_logits = grpo_pg_loss_from_logits_grad_kt(
                &logits,
                &input_ids,
                &action_mask,
                &ref_kt,
                params,
                1.0,
                &device,
            )
            .unwrap();
            let expected_hidden = grad_logits
                .squeeze(0)
                .unwrap()
                .to_dtype(KtDType::F32)
                .unwrap()
                .matmul(&head_t_t)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let actual_hidden = super::grpo_pg_loss_from_normed_hidden_grad_kt(
                &normed_hidden,
                &head_t,
                &input_ids,
                &action_mask,
                &plp_hidden,
                &ref_kt,
                params,
                1.0,
                &device,
                3,
            )
            .unwrap();
            let loss_expected = grpo_loss(&plp_hidden, &ref_kt, params, &device).unwrap();
            let (loss_cached, cached_hidden) =
                super::grpo_pg_loss_from_normed_hidden_loss_and_grad_kt(
                    &normed_hidden,
                    &head_t,
                    &input_ids,
                    &action_mask,
                    &ref_kt,
                    params,
                    1.0,
                    &device,
                    3,
                )
                .unwrap();
            let loss_diff = (read(&loss_expected)[0] - read(&loss_cached)[0]).abs();
            assert!(
                loss_diff < 1e-6,
                "{name}: cached loss drift {loss_diff:e}"
            );
            let expected = read(&expected_hidden);
            let actual = read(&actual_hidden);
            let cached = read(&cached_hidden);
            assert_eq!(expected.len(), actual.len(), "{name}: grad length drift");
            assert_eq!(actual.len(), cached.len(), "{name}: cached grad length drift");
            for (i, ((e, a), c)) in expected.iter().zip(actual.iter()).zip(cached.iter()).enumerate() {
                let tol = 3e-4f32.max(3e-4 * e.abs());
                assert!(
                    (e - a).abs() <= tol,
                    "{name}: hidden grad drift at flat {i}: expected={e} actual={a} tol={tol}"
                );
                assert!(
                    (a - c).abs() <= tol,
                    "{name}: cached hidden grad drift at flat {i}: expected={a} actual={c} tol={tol}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn grpo_normed_hidden_sparse_backward_cuda_chunk_runs() {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!("[GRPO-CUDA] no CUDA device; skipping");
            return;
        }

        let device = kiln_tensor::Device::Cuda(0);
        let (seq_len, hidden_size, vocab) = (6usize, 5usize, 13usize);
        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2];
        let action_mask = vec![false, false, true, true, false, true];
        let hidden_host: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (((i * 7) % 19) as f32 - 9.0) * 0.041)
            .collect();
        let head_host: Vec<f32> = (0..hidden_size * vocab)
            .map(|i| (((i * 11) % 23) as f32 - 11.0) * 0.037)
            .collect();
        let normed_hidden =
            KtTensor::from_vec_on(device, hidden_host, vec![1, seq_len, hidden_size])
                .expect("cuda hidden");
        let head_t =
            KtTensor::from_vec_on(device, head_host, vec![hidden_size, vocab])
                .expect("cuda head");
        let read = |t: &KtTensor| -> Vec<f32> {
            t.to_dtype(KtDType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec()
                .unwrap()
        };

        let plp_hidden = super::selected_log_probs_from_normed_hidden_chunked_kt(
            &normed_hidden,
            &head_t,
            &input_ids,
            &action_mask,
            3,
            &device,
        )
        .unwrap();
        let ref_host: Vec<f32> = read(&plp_hidden).iter().map(|&p| p - 0.1).collect();
        let ref_kt = KtTensor::from_vec_on(device, ref_host, vec![3]).expect("cuda ref");
        let params = GrpoLossParams {
            advantage: -0.7,
            clip_low: 0.2,
            clip_high: 0.2,
            kl_coeff: 0.1,
            kl_estimator: KlEstimator::K1,
            loss_normalizer: 1.0,
            is_level: IsLevel::Token,
            reinforce: false,
            entropy_aware_kl_quantile: None,
        };

        let single = super::grpo_pg_loss_from_normed_hidden_grad_kt(
            &normed_hidden,
            &head_t,
            &input_ids,
            &action_mask,
            &plp_hidden,
            &ref_kt,
            params,
            1.0,
            &device,
            vocab,
        )
        .unwrap();
        let multi = super::grpo_pg_loss_from_normed_hidden_grad_kt(
            &normed_hidden,
            &head_t,
            &input_ids,
            &action_mask,
            &plp_hidden,
            &ref_kt,
            params,
            1.0,
            &device,
            3,
        )
        .unwrap();

        let single = read(&single);
        let multi = read(&multi);
        assert_eq!(single.len(), multi.len());
        for (i, (a, b)) in single.iter().zip(multi.iter()).enumerate() {
            let tol = 4e-4f32.max(4e-4 * a.abs());
            assert!(
                (a - b).abs() <= tol,
                "cuda sparse GRPO bwd drift at {i}: single={a} multi={b} tol={tol}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn grpo_loss_coeff_device_fast_path_matches_host_derivation_cuda() {
        if !kiln_tensor::probe::cuda_is_available() {
            eprintln!("[GRPO-CUDA] no CUDA device; skipping coeff fast path parity");
            return;
        }

        let device = kiln_tensor::Device::Cuda(0);
        let ratios = [0.5f64, 0.85, 1.05, 1.24, 1.4];
        let plp_host = vec![-1.20f32, -0.40, -2.00, -0.90, -1.60];
        let ref_host: Vec<f32> = plp_host
            .iter()
            .zip(ratios.iter())
            .map(|(&p, &r)| (p as f64 - r.ln()) as f32)
            .collect();
        let num_active = plp_host.len();
        let plp = KtTensor::from_vec_on(device, plp_host, vec![num_active]).unwrap();
        let ref_kt = KtTensor::from_vec_on(device, ref_host, vec![num_active]).unwrap();
        let variants: Vec<(&str, GrpoLossParams, f64)> = vec![
            (
                "token-positive-k1",
                GrpoLossParams {
                    advantage: 0.7,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.11,
                    kl_estimator: KlEstimator::K1,
                    loss_normalizer: 0.25,
                    is_level: IsLevel::Token,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
                1.0,
            ),
            (
                "token-negative-k3",
                GrpoLossParams {
                    advantage: -0.8,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.07,
                    kl_estimator: KlEstimator::K3,
                    loss_normalizer: 0.5,
                    is_level: IsLevel::Token,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
                0.75,
            ),
            (
                "token-positive-none",
                GrpoLossParams {
                    advantage: 0.4,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.0,
                    kl_estimator: KlEstimator::None,
                    loss_normalizer: 0.4,
                    is_level: IsLevel::Token,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
                1.25,
            ),
            (
                "reinforce",
                GrpoLossParams {
                    advantage: -0.6,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.0,
                    kl_estimator: KlEstimator::None,
                    loss_normalizer: 0.2,
                    is_level: IsLevel::Token,
                    reinforce: true,
                    entropy_aware_kl_quantile: None,
                },
                0.5,
            ),
        ];

        for (name, params, grad_scalar) in variants {
            let host = super::grpo_loss_coeff_from_policy_log_probs_kt(
                &plp,
                &ref_kt,
                params,
                num_active,
            )
            .unwrap();
            let expected: Vec<f32> = host
                .iter()
                .map(|c| (*c as f64 * grad_scalar) as f32)
                .collect();
            let got = super::grpo_loss_coeff_col_device_fast_path_kt(
                &plp,
                &ref_kt,
                params,
                num_active,
                grad_scalar,
                &device,
            )
            .unwrap()
            .unwrap_or_else(|| panic!("{name}: fast path declined"));
            assert_eq!(got.shape(), &[num_active, 1], "{name}: coeff_col shape");
            let got = got
                .flatten_all()
                .unwrap()
                .to_dtype(KtDType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (i, (&a, &b)) in expected.iter().zip(got.iter()).enumerate() {
                let tol = 2e-5f32.max(2e-5 * a.abs());
                assert!(
                    (a - b).abs() <= tol,
                    "{name}: coeff[{i}] host={a} device={b} tol={tol}"
                );
            }
        }

        let declined = super::grpo_loss_coeff_col_device_fast_path_kt(
            &plp,
            &ref_kt,
            GrpoLossParams {
                advantage: 0.7,
                clip_low: 0.2,
                clip_high: 0.2,
                kl_coeff: 0.11,
                kl_estimator: KlEstimator::K1,
                loss_normalizer: 0.25,
                is_level: IsLevel::Sequence,
                reinforce: false,
                entropy_aware_kl_quantile: None,
            },
            num_active,
            1.0,
            &device,
        )
        .unwrap();
        assert!(declined.is_none(), "sequence IS should keep the host fallback");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn grpo_loss_coeff_matches_finite_difference_cpu() {
        let device = kiln_tensor::Device::Cpu;
        let ratios = [0.5f64, 1.05, 1.4, 0.95];
        let plp_host = vec![-1.20f32, -0.40, -2.00, -0.90];
        let ref_host: Vec<f32> = plp_host
            .iter()
            .zip(ratios.iter())
            .map(|(&p, &r)| (p as f64 - r.ln()) as f32)
            .collect();
        let num_active = plp_host.len();
        let plp = KtTensor::from_vec_on(device, plp_host.clone(), vec![num_active]).unwrap();
        let ref_kt = KtTensor::from_vec_on(device, ref_host, vec![num_active]).unwrap();
        let variants: Vec<(&str, GrpoLossParams)> = vec![
            (
                "token-positive-k3",
                GrpoLossParams {
                    advantage: 0.7,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.11,
                    kl_estimator: KlEstimator::K3,
                    loss_normalizer: 0.25,
                    is_level: IsLevel::Token,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
            ),
            (
                "token-negative-k1",
                GrpoLossParams {
                    advantage: -0.8,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.07,
                    kl_estimator: KlEstimator::K1,
                    loss_normalizer: 0.5,
                    is_level: IsLevel::Token,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
            ),
            (
                "sequence-positive-k3",
                GrpoLossParams {
                    advantage: 0.6,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.05,
                    kl_estimator: KlEstimator::K3,
                    loss_normalizer: 0.75,
                    is_level: IsLevel::Sequence,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
            ),
            (
                "sequence-negative-none",
                GrpoLossParams {
                    advantage: -0.9,
                    clip_low: 0.2,
                    clip_high: 0.2,
                    kl_coeff: 0.0,
                    kl_estimator: KlEstimator::None,
                    loss_normalizer: 0.4,
                    is_level: IsLevel::Sequence,
                    reinforce: false,
                    entropy_aware_kl_quantile: None,
                },
            ),
        ];

        let read_scalar = |t: &KtTensor| -> f64 {
            t.to_dtype(KtDType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()[0] as f64
        };
        let loss_of = |host: &[f32], params: GrpoLossParams| -> f64 {
            let p = KtTensor::from_vec_on(device, host.to_vec(), vec![num_active]).unwrap();
            let loss = grpo_loss(&p, &ref_kt, params, &device).unwrap();
            read_scalar(&loss)
        };

        const EPS: f32 = 1e-2;
        for (name, params) in variants {
            let coeff = super::grpo_loss_coeff_from_policy_log_probs_kt(
                &plp,
                &ref_kt,
                params,
                num_active,
            )
            .unwrap();
            assert_eq!(coeff.len(), num_active, "{name}: coeff len");
            for a in 0..num_active {
                let mut plus = plp_host.clone();
                let mut minus = plp_host.clone();
                plus[a] += EPS;
                minus[a] -= EPS;
                let fd = (loss_of(&plus, params) - loss_of(&minus, params)) / (2.0 * EPS as f64);
                let got = coeff[a] as f64;
                let tol = 5e-3_f64.max(3e-2 * fd.abs());
                assert!(
                    (got - fd).abs() <= tol,
                    "{name}: coeff[{a}] got={got:+.6} fd={fd:+.6} tol={tol:.3e}"
                );
            }
        }
    }

    /// ROCm regression lock (#33): the GRPO tape-authoritative PG-loss recorder
    /// must STAY OPEN on ROCm. A refactor that drops `Device::Rocm(_)` from the
    /// device gate (grpo_tape_shim.rs:520) or `tape_auth_eligible`
    /// (trainer.rs:4919) would make this return Ok(None) → fall to the dead
    /// post-#1082 candle path → EMPTY grad store (silent broken GRPO). This
    /// catches that: it must record a node and emit a finite, non-zero d_logits.
    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_grpo_pg_loss_records_and_backprops() {
        use crate::trainer::{GrpoLossParams, token_log_probs};
        use crate::{IsLevel, KlEstimator};
        use kiln_tensor::{DType as KtDType, Tensor as KtTensor};

        if !kiln_tensor::rocm_is_available() {
            eprintln!("[GRPO-ROCm] no ROCm device — skipping");
            return;
        }
        unsafe {
            std::env::set_var("KILN_USE_TAPE_FORWARD", "1");
        }
        let device = kiln_tensor::Device::Rocm(0);
        let (seq_len, vocab) = (6usize, 16usize);
        let input_ids: Vec<u32> = vec![1, 5, 10, 3, 7, 2];
        let action_mask = vec![false, false, true, true, false, true];
        let num_active = action_mask[1..].iter().filter(|m| **m).count();
        // Deterministic BF16 logits [1, T, V] on ROCm (the server path is BF16).
        let logits_host: Vec<f32> = (0..seq_len * vocab)
            .map(|i| (((i * 31) % 19) as f32 - 9.0) * 0.07)
            .collect();
        let logits = KtTensor::from_vec_on(device, logits_host, vec![1, seq_len, vocab])
            .unwrap()
            .to_dtype(KtDType::BF16)
            .unwrap();
        // ref_log_probs = the fixture's own policy log-probs minus 0.1 (in-range ratio).
        let plp: Vec<f32> = token_log_probs(&logits, &input_ids, &action_mask, &device)
            .unwrap()
            .to_dtype(KtDType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec()
            .unwrap();
        let ref_host: Vec<f32> = plp.iter().map(|&p| p - 0.1).collect();
        let ref_kt = KtTensor::from_vec_on(device, ref_host, vec![num_active]).unwrap();
        let params = GrpoLossParams {
            advantage: -0.7,
            clip_low: 0.2,
            clip_high: 0.2,
            kl_coeff: 0.1,
            kl_estimator: KlEstimator::K1,
            loss_normalizer: 1.0,
            is_level: IsLevel::Token,
            reinforce: false,
            entropy_aware_kl_quantile: None,
        };

        let (res, tape) = kiln_autograd::with_thread_local_tape(|| {
            super::try_tape_grpo_pg_loss_from_logits_kt(
                &logits,
                &input_ids,
                &action_mask,
                &ref_kt,
                params,
                &device,
            )
        });
        let loss = res
            .expect("try_tape_grpo_pg_loss_from_logits_kt errored")
            .expect("returned None on ROCm — GRPO tape gate REJECTED Rocm (regression #1454)");
        assert!(tape.len() >= 1, "GRPO must record a tape node on ROCm");

        let seed = KtTensor::from_vec_on(device, vec![1.0f32], vec![]).unwrap();
        let grads = tape
            .backward(loss.id(), seed, |a, b| kiln_tensor::ops::add(a, b))
            .expect("GRPO tape backward on ROCm");
        let dl = grads.get(logits.id()).expect("d_logits present");
        let dl_v: Vec<f32> = dl
            .to_dtype(KtDType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec()
            .unwrap();
        assert!(
            dl_v.iter().all(|x| x.is_finite()),
            "non-finite GRPO d_logits on ROCm"
        );
        assert!(
            dl_v.iter().any(|&x| x != 0.0),
            "GRPO d_logits all-zero on ROCm (empty grads)"
        );
        eprintln!(
            "[GRPO-ROCm] OK: recorded {} node(s), finite non-zero d_logits",
            tape.len()
        );
    }

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
        let ref_kt = KtTensor::from_vec_on(device_kt, ref_host, vec![num_active]).expect("ref kt");

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
            (
                "Token+K1",
                mk_params(IsLevel::Token, KlEstimator::K1, false),
            ),
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

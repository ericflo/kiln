//! On-Policy Distillation top-K reverse-KL loss kernel.
//!
//! Computes the per-token reverse KL between a student LLM's distribution and
//! a teacher's distribution, restricted to the **teacher's top-K support**
//! (Fu et al. 2026, "Revisiting On-Policy Distillation", §3.1, Eq. 6–8).
//!
//! Given a sequence of student-sampled tokens, for each active position `t`
//! the teacher provides its top-K vocab indices `S_t = TopK_q(c_t)` and the
//! corresponding teacher logprobs over the full vocab. Both distributions
//! are renormalised over the K-element support and we compute
//!
//! ```text
//! KL_t = sum_{v in S_t} p_hat(v) * (log p_hat(v) - log q_hat(v))
//! ```
//!
//! where
//!
//! ```text
//! p_hat(v) = exp(s_v - logsumexp_{u in S_t}(s_u))                      (student renorm)
//! q_hat(v) = exp(t_v - logsumexp_{u in S_t}(t_u))                      (teacher renorm)
//! ```
//!
//! and `s_v = (hidden[t] @ head_t)[v]` is the student logit at the v-th
//! vocab position. The final loss is the mean of `KL_t` over **active**
//! positions (positions where `label_mask[t]` is true — typically only the
//! assistant tokens contribute to the loss).
//!
//! # Why a custom op and not just candle
//!
//! Naive candle: materialize the full `[T, V]` student logits tensor, gather
//! the K columns specified by the teacher's top-K indices, compute KL, and
//! backprop through the gather and the projection. For Qwen3.5-4B with
//! V = 151936 and T_active = 4096 this is **~9.7 GB of F32 logits**
//! before doing anything useful — the same memory pressure FLCE avoids
//! for standard cross-entropy.
//!
//! The OPD path is structurally cheaper than CE: we only need the K
//! per-token student logits the teacher cares about, not the full vocab.
//! We project `hidden[t]` against the `K` columns of `head_t` named by
//! `teacher_topk_indices[t, :]` — a per-token gather-then-matmul whose
//! peak intermediate is `[T_active, K]` (~5000× smaller than `[T_active, V]`).
//!
//! # API contract
//!
//! - `hidden`: `[1, T, H]` student hidden states (output of
//!   `model_forward_final_norm`), bf16 or f32 (kiln trainer uses bf16).
//! - `head_t`: `[H, V]` transposed LM head (matches kiln's `embed_tokens_t`
//!   layout), bf16 or f32.
//! - `teacher_topk_indices`: `[T_active, K]` flattened in row-major
//!   order. Holds the teacher's top-K vocab indices at each **active**
//!   position; positions in the same order as `active_positions` (see
//!   below). Dtype u32.
//! - `teacher_topk_logprobs`: `[T_active, K]` flattened in row-major
//!   order. Holds teacher log-probabilities at the K vocab positions
//!   (`log_softmax(teacher_logits)`). Dtype f32 (this is what every
//!   hosted-logprobs API returns and matches §3.2's `LogitSource` trait).
//! - `label_mask`: `[T]` booleans; the position-`t` logit contributes when
//!   `label_mask[t]` is true. The number of active positions must equal
//!   `T_active` and the order of active positions left-to-right in
//!   `hidden` must match the row order of `teacher_topk_indices`.
//!
//! Returns the **mean reverse KL** over active positions as a scalar f32
//! tensor. The trainer scales this by 1.0 (it is the loss directly) — note
//! that the per-token advantage used in the GRPO importance-sampling code
//! path is `-reverse_kl_per_token`, so a separate helper
//! [`opd_per_position_reverse_kl`] returns the per-position vector for
//! direct advantage construction.
//!
//! # Phase A vs Phase B
//!
//! Mirrors `kiln-flce-kernel`'s split:
//!
//! - **Phase A** ([`opd_top_k_reverse_kl_phase_a`]) — pure-candle reference
//!   implementation. Builds `[T_active, K]` student logits via per-token
//!   gather + batched matmul, runs the renormalised reverse-KL in candle
//!   ops, and lets candle autograd handle the backward. Used as the parity
//!   oracle and as the default path on CPU.
//! - **Phase B** ([`opd_top_k_reverse_kl_phase_b`]) — `CustomOp1` whose
//!   `bwd()` runs the manual analytic backward. Forward stores only the
//!   scalar loss; the chunk intermediates are dropped on return.
//!
//! [`opd_top_k_reverse_kl`] dispatches to Phase A or B based on the
//! `KILN_OPD_LOSS_PHASE_A=1` env var (default: Phase B).
//!
//! # Numerical contract
//!
//! Per [§9.2 of `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`]:
//! the same `(hidden, head_t, teacher_topk_indices, teacher_topk_logprobs,
//! label_mask)` tuple must produce KL values within 1e-5 across CPU / CUDA /
//! Metal. The parity tests in this crate enforce that at f32, and 1e-2
//! relative at bf16.
//!
//! [§9.2 of `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md`]: ../../docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md

use anyhow::{Context, Result, anyhow};
use candle_core::{D, DType, Device, Tensor};

mod phase_b;

pub use phase_b::{
    opd_top_k_reverse_kl_phase_b, opd_top_k_reverse_kl_phase_b_per_position,
};

/// Default chunk size when iterating along the active-token dimension. Used
/// by Phase B to bound the temporary `[chunk_T, K]` intermediate. For
/// typical OPD configs (T_active ≤ 8192, K = 32) the whole batch fits in
/// one chunk, but very-long-context training keeps the option open.
pub const DEFAULT_CHUNK_SIZE: usize = 4096;

/// Read the `KILN_OPD_LOSS_PHASE_A` env var. When set (`1` / `true` /
/// `yes`), the dispatch helper [`opd_top_k_reverse_kl`] routes to Phase A;
/// otherwise it routes to Phase B (the production default).
pub fn use_phase_a() -> bool {
    std::env::var("KILN_OPD_LOSS_PHASE_A")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// Read the `KILN_DISABLE_OPD_LOSS_KERNEL` env var. When set (`1` / `true`
/// / `yes`), [`opd_top_k_reverse_kl`] forces Phase A even when the caller
/// passes parameters that would otherwise activate Phase B. Mirrors the
/// `KILN_DISABLE_*` kill-switch convention used elsewhere in kiln (PR #92,
/// #133, #158, #166).
pub fn kernel_disabled() -> bool {
    std::env::var("KILN_DISABLE_OPD_LOSS_KERNEL")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// Validate the shape / dtype contract on the public entry points and
/// return the `(T, H, V, T_active, K)` quintuple for downstream code.
fn validate_inputs(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<(usize, usize, usize, usize, usize)> {
    let hidden_dims = hidden.dims();
    if hidden_dims.len() != 3 {
        return Err(anyhow!(
            "hidden must be 3-D [1, T, H]; got {:?}",
            hidden_dims
        ));
    }
    if hidden_dims[0] != 1 {
        return Err(anyhow!(
            "hidden batch dim must be 1 (kiln trainer convention); got {:?}",
            hidden_dims
        ));
    }
    let seq_len = hidden_dims[1];
    let hidden_size = hidden_dims[2];

    let head_dims = head_t.dims();
    if head_dims.len() != 2 {
        return Err(anyhow!(
            "head_t must be 2-D [H, V]; got {:?}",
            head_dims
        ));
    }
    if head_dims[0] != hidden_size {
        return Err(anyhow!(
            "hidden_size mismatch: hidden has H={} but head_t has H={}",
            hidden_size,
            head_dims[0]
        ));
    }
    let vocab_size = head_dims[1];

    if label_mask.len() != seq_len {
        return Err(anyhow!(
            "label_mask length {} does not match T {}",
            label_mask.len(),
            seq_len
        ));
    }
    if top_k == 0 {
        return Err(anyhow!("top_k must be > 0"));
    }

    let active_count = label_mask.iter().filter(|&&m| m).count();
    let expected_logits = active_count * top_k;
    if teacher_topk_indices.len() != expected_logits {
        return Err(anyhow!(
            "teacher_topk_indices length {} != T_active * K = {} * {} = {}",
            teacher_topk_indices.len(),
            active_count,
            top_k,
            expected_logits
        ));
    }
    if teacher_topk_logprobs.len() != expected_logits {
        return Err(anyhow!(
            "teacher_topk_logprobs length {} != T_active * K = {} * {} = {}",
            teacher_topk_logprobs.len(),
            active_count,
            top_k,
            expected_logits
        ));
    }
    for (i, &idx) in teacher_topk_indices.iter().enumerate() {
        if (idx as usize) >= vocab_size {
            return Err(anyhow!(
                "teacher_topk_indices[{}] = {} >= vocab_size {}",
                i,
                idx,
                vocab_size
            ));
        }
    }

    Ok((seq_len, hidden_size, vocab_size, active_count, top_k))
}

/// Dispatch to Phase A or Phase B based on the `KILN_OPD_LOSS_PHASE_A`
/// env var (default Phase B). Production trainer call sites should use
/// this entry point so a single env-var flip toggles every OPD-loss
/// call. The `KILN_DISABLE_OPD_LOSS_KERNEL` kill switch forces Phase A
/// even when the env var is unset.
pub fn opd_top_k_reverse_kl(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
    chunk_size: usize,
) -> Result<Tensor> {
    if use_phase_a() || kernel_disabled() {
        opd_top_k_reverse_kl_phase_a(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
        )
    } else {
        opd_top_k_reverse_kl_phase_b(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
            chunk_size,
        )
    }
}

/// Phase A entry point: pure-candle reference implementation, autograd
/// flows through the gather and matmul intermediates. Used as the parity
/// oracle for Phase B and as the default path when
/// `KILN_OPD_LOSS_PHASE_A=1` is set.
///
/// Returns scalar f32 mean reverse KL over active positions, or a zero
/// scalar tensor when no positions are active.
pub fn opd_top_k_reverse_kl_phase_a(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
) -> Result<Tensor> {
    let (_, _, _, active_count, _) = validate_inputs(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;
    if active_count == 0 {
        return Tensor::new(0.0f32, device).context("zero scalar loss (no active rows)");
    }

    let per_pos =
        per_position_phase_a(hidden, head_t, teacher_topk_indices, teacher_topk_logprobs, label_mask, top_k, device)?;
    Ok(per_pos.mean_all()?)
}

/// Phase A per-position reverse-KL. Returns a `[T_active]` f32 tensor.
/// Used by the trainer when constructing the per-token advantage
/// `A_t = -KL_t` for the GRPO importance-sampling loss (§3.1, step 4 of
/// the grand plan pseudocode).
pub fn opd_top_k_reverse_kl_phase_a_per_position(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
) -> Result<Tensor> {
    let (_, _, _, active_count, _) = validate_inputs(
        hidden,
        head_t,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;
    if active_count == 0 {
        return Tensor::zeros((0,), DType::F32, device)
            .context("empty per-position KL (no active rows)");
    }
    per_position_phase_a(hidden, head_t, teacher_topk_indices, teacher_topk_logprobs, label_mask, top_k, device)
}

/// Per-position helper shared by [`opd_top_k_reverse_kl_phase_a`] and
/// [`opd_top_k_reverse_kl_phase_a_per_position`].
///
/// Math:
///
/// 1. `head_chunk[t, h, k] = head_t[h, teacher_topk_indices[t, k]]` — a
///    per-token gather of K columns from the projection matrix. Built by
///    flattening the `[T_active * K]` index buffer once and calling
///    `head_t.index_select(...)` on dim 1, which yields
///    `[H, T_active * K]`. Reshape to `[T_active, K, H]` and transpose to
///    `[T_active, H, K]`.
/// 2. `s_logits[t, k] = sum_h hidden_active[t, h] * head_chunk[t, h, k]`
///    — batched matmul `[T_active, 1, H] @ [T_active, H, K]` → `[T_active, K]`.
/// 3. Renormalise both distributions over the K support:
///    ```text
///    log_p_hat = s_logits - logsumexp(s_logits, dim=-1)
///    log_q_hat = teacher_topk_logprobs - logsumexp(teacher_topk_logprobs, dim=-1)
///    p_hat = exp(log_p_hat)
///    ```
/// 4. Per-position KL: `KL_t = sum_k p_hat[t, k] * (log_p_hat[t, k] - log_q_hat[t, k])`.
fn per_position_phase_a(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
) -> Result<Tensor> {
    let active_positions: Vec<u32> = label_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    let active_count = active_positions.len();
    debug_assert!(active_count > 0, "caller short-circuits empty");

    let active_indices = Tensor::new(active_positions.as_slice(), device)
        .context("build active position indices")?;
    let hidden_2d = hidden.squeeze(0).context("squeeze hidden batch dim")?;
    let active_hidden = hidden_2d
        .index_select(&active_indices, 0)
        .context("gather active rows from hidden")?;
    let active_hidden_f32 = active_hidden
        .to_dtype(DType::F32)
        .context("cast hidden to f32")?;
    let head_t_f32 = head_t.to_dtype(DType::F32).context("cast head_t to f32")?;

    // Gather K columns per token, return `[T_active, H, K]`.
    let head_gather = gather_head_columns(&head_t_f32, teacher_topk_indices, active_count, top_k, device)?;

    // Batched matmul: [T_active, 1, H] @ [T_active, H, K] -> [T_active, 1, K]
    let lhs = active_hidden_f32
        .unsqueeze(1)
        .context("unsqueeze active_hidden")?;
    let s_logits = lhs
        .matmul(&head_gather)
        .context("batched matmul for student logits at K")?
        .squeeze(1)
        .context("squeeze student logits dim")?;

    // Teacher logprobs at the K support (already log-softmax over the full
    // teacher vocab from the LogitSource).
    let q_logprobs = Tensor::from_vec(
        teacher_topk_logprobs.to_vec(),
        (active_count, top_k),
        device,
    )
    .context("build teacher topk logprobs tensor")?;

    // Renormalise both distributions over the K-element support.
    let log_p_hat = log_softmax_last(&s_logits)?;
    let log_q_hat = log_softmax_last(&q_logprobs)?;

    // KL(p_hat || q_hat) = sum_k p_hat * (log_p_hat - log_q_hat)
    let p_hat = log_p_hat.exp().context("exp(log_p_hat)")?;
    let diff = (&log_p_hat - &log_q_hat).context("log_p_hat - log_q_hat")?;
    let per_token = (p_hat * diff)
        .context("p_hat * (log_p_hat - log_q_hat)")?
        .sum(D::Minus1)
        .context("sum over K support")?;
    Ok(per_token)
}

/// Gather K columns of `head_t` per active token. Returns a tensor of
/// shape `[T_active, H, K]` where slice `t` holds the K columns of `head_t`
/// pointed to by `teacher_topk_indices[t * K .. (t+1) * K]`.
///
/// We build this by `head_t.index_select(dim=1, indices=flat)` which yields
/// `[H, T_active * K]`, then reshape to `[H, T_active, K]` and transpose
/// to `[T_active, H, K]`. The expensive operation is the
/// `index_select` — `T_active * K` columns × `H` rows. For typical OPD
/// configs (T_active ≤ 4096, K = 32, H = 2560) this is 4096 × 32 × 2560 =
/// 335M f32 elements = ~1.3 GB, comparable to a single forward-chunk in
/// FLCE and far below the full-vocab projection.
fn gather_head_columns(
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    active_count: usize,
    top_k: usize,
    device: &Device,
) -> Result<Tensor> {
    let flat_indices = Tensor::new(teacher_topk_indices, device)
        .context("build flat indices")?;
    let gathered = head_t
        .index_select(&flat_indices, 1)
        .context("index_select head_t columns")?;
    // [H, T_active * K] -> [H, T_active, K] -> [T_active, H, K]
    let hidden_size = head_t.dim(0)?;
    let reshaped = gathered
        .reshape((hidden_size, active_count, top_k))
        .context("reshape gathered head columns")?;
    let transposed = reshaped
        .permute((1, 0, 2))
        .context("permute to [T_active, H, K]")?
        .contiguous()
        .context("contiguous after permute")?;
    Ok(transposed)
}

/// log_softmax along the last dimension. candle does have
/// `candle_nn::ops::log_softmax` but we re-implement it here to avoid an
/// extra dependency and to keep the autograd graph minimal: this version
/// produces a single subtraction node off the input.
pub(crate) fn log_softmax_last(x: &Tensor) -> Result<Tensor> {
    let lse = x.log_sum_exp(D::Minus1)?.unsqueeze(D::Minus1)?;
    let broadcast = lse.broadcast_as(x.shape())?;
    Ok((x - broadcast)?)
}

/// Re-exports used by parity tests and downstream call sites.
pub use crate::phase_b::OpdLossCustomOp;

pub use crate::phase_b::{compute_per_position_metrics, PerPositionMetrics};

/// One position's worth of distribution-alignment diagnostics, computed
/// over the **teacher's** K support (§3.8 of the grand plan).
///
/// `overlap_ratio` (the |S^p ∩ S^q| / K metric from Li et al. 2026 eq 6)
/// requires the student's own top-K, which is _not_ computed by the
/// fused kernel — that's a separate full-vocab pass via
/// [`compute_overlap_ratio_probe`]. The K-support metrics here are
/// cheap and computed on the same kernel launch as the loss.
#[derive(Debug, Clone, Copy, Default)]
pub struct PerPositionMetricsRow {
    /// `H(p_hat)` — entropy of the student distribution over the teacher's
    /// K support. In nats. Higher = student less concentrated.
    pub student_entropy: f32,
    /// `H(q_hat)` — entropy of the renormalised teacher distribution
    /// over the same K support. In nats.
    pub teacher_entropy: f32,
    /// `KL(p_hat || q_hat)` — the per-position reverse KL the trainer is
    /// minimising. Same value the loss kernel emits.
    pub reverse_kl: f32,
}

impl PerPositionMetricsRow {
    /// `entropy_gap = |H(q) - H(p)|`, the §3.8 diagnostic. Narrows as
    /// student converges to teacher.
    pub fn entropy_gap(&self) -> f32 {
        (self.teacher_entropy - self.student_entropy).abs()
    }

    /// `overlap_token_advantage` (Li et al. 2026 eq 7), restricted to
    /// the K-support: `E_p[log q_hat - log p_hat]`. For the K-support
    /// case this equals `-reverse_kl`; surfaced as a separate accessor
    /// so callers can use the Li-et-al name in dashboards without
    /// confusion.
    pub fn overlap_token_advantage(&self) -> f32 {
        -self.reverse_kl
    }
}

#[cfg(test)]
mod tests;

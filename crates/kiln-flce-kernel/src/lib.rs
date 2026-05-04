//! Fused Linear Cross-Entropy (FLCE).
//!
//! Computes cross-entropy loss over a projected head without materializing
//! the full `[T, V]` logits tensor. The mathematical trick is the log-sum-exp
//! identity: `log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))`, which
//! lets us reduce over the vocab dimension in chunks while keeping only
//! per-row max + running sum-exp + the gathered correct logit.
//!
//! # Phase A vs Phase B
//!
//! Phase A ([`fused_linear_cross_entropy`]) is a pure-candle reference
//! implementation that chunks the forward over the vocab dim. Forward peak
//! memory is roughly `T_active * chunk_size * 4 bytes` instead of `T_active *
//! V * 4 bytes` (~37× smaller for Qwen3.5-4B with chunk=4096 and V=151936).
//! Backward flows through candle autograd, so intermediate chunk tensors
//! (`logits_chunk`, `shifted`, `shifted.exp()`) are retained for the entire
//! forward — at T=8192 with V=248320 this is ~23 GiB held live across 61
//! vocab chunks, which OOMs SFT on A6000 (see
//! `docs/audits/PHASE10_MODE_B_TRACE.md`).
//!
//! Phase B ([`fused_linear_cross_entropy_phase_b`]) replaces the autograd
//! graph with a [`candle_core::CustomOp1`] whose `bwd()` recomputes each
//! vocab chunk on the fly. The forward stores only the scalar loss; chunk
//! intermediates created during the forward pass are local to the op's
//! `cpu_fwd`/`cuda_fwd` body and dropped on return, breaking the
//! ~23 GiB-of-FLCE-intermediates retention pattern. Estimated peak-VRAM
//! saving at T=8192 SFT on A6000: ~22 GiB. Math is identical to Phase A
//! up to floating-point associativity.
//!
//! [`fused_linear_cross_entropy_dispatch`] picks A or B based on the
//! `KILN_FLCE_PHASE_A=1` env var (default Phase B; opt back into Phase A
//! for parity debugging).
//!
//! # Target
//!
//! Phase 10 enables long-context SFT on Qwen3.5-4B on A6000. Preflight
//! (PR #235) showed the head materializes a `[T, V]` F32 logits tensor
//! that dominates peak VRAM at `T >= 8192` (OOM before reaching head).
//! FLCE is the prerequisite, not an optimization.

use anyhow::{Context, Result, anyhow};
use candle_core::{D, DType, Device, Tensor};

mod phase_b;
pub use phase_b::fused_linear_cross_entropy_phase_b;

/// Default chunk size along the vocab dimension.
///
/// The preflight math picked 4096 as a reasonable balance between kernel
/// launch overhead and peak intermediate footprint. For Qwen3.5-4B with
/// V=151936, a chunk of 4096 means ~37 chunks per forward — small enough
/// that per-chunk launch cost is absorbed.
pub const DEFAULT_CHUNK_SIZE: usize = 4096;

/// Read the `KILN_FLCE_PHASE_A` env var. When set (`1`/`true`/`yes`), the
/// dispatch helper [`fused_linear_cross_entropy_dispatch`] routes to Phase A
/// (this function); otherwise it routes to Phase B (the CustomOp1 path).
///
/// Phase B is the production default — the autograd-graph reduction is the
/// only audit-supported path to T=8192 SFT on A6000 (see
/// `docs/audits/PHASE10_MODE_B_TRACE.md`). Phase A is kept as the parity
/// reference and as an escape hatch for debugging.
pub fn use_phase_a() -> bool {
    std::env::var("KILN_FLCE_PHASE_A")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// Dispatch to either [`fused_linear_cross_entropy`] (Phase A) or
/// [`fused_linear_cross_entropy_phase_b`] (Phase B) based on the
/// `KILN_FLCE_PHASE_A` env var. Default is Phase B.
///
/// Trainer call sites should use this function instead of the explicit
/// Phase A/B helpers so a single env-var flip switches every FLCE call.
pub fn fused_linear_cross_entropy_dispatch(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
) -> Result<Tensor> {
    if use_phase_a() {
        fused_linear_cross_entropy(hidden, head_t, input_ids, label_mask, device, chunk_size)
    } else {
        fused_linear_cross_entropy_phase_b(
            hidden, head_t, input_ids, label_mask, device, chunk_size,
        )
    }
}

/// Compute cross-entropy loss using a fused linear + cross-entropy pass
/// (**Phase A** — pure-candle reference, autograd flows through chunk
/// intermediates).
///
/// Phase A is kept as the parity reference and as an opt-in escape hatch
/// for debugging via `KILN_FLCE_PHASE_A=1`. New training code should call
/// [`fused_linear_cross_entropy_dispatch`] (default Phase B).
///
/// # Arguments
///
/// * `hidden` — `[1, seq_len, hidden_size]` post-final-RMSNorm hidden states
///   (matches the input shape of `kiln_model::forward::model_forward_head`).
/// * `head_t` — `[hidden_size, vocab_size]` transposed head weight
///   (matches kiln's `embed_tokens_t` layout; this is `W.T` where `W` is
///   the standard `[vocab_size, hidden_size]` lm_head).
/// * `input_ids` — token ids; `input_ids[1..]` are the targets for
///   `logits[..seq_len-1]` (next-token prediction shift).
/// * `label_mask` — `[seq_len]` booleans; only positions where
///   `label_mask[i+1]` is true contribute to the loss.
/// * `device` — device on which the output scalar is allocated.
/// * `chunk_size` — chunk size along the vocab dim; use
///   `DEFAULT_CHUNK_SIZE` unless tuning.
///
/// # Returns
///
/// A scalar F32 [`Tensor`] — the mean cross-entropy over active positions.
/// Returns a zero tensor if no positions are active (no assistant tokens).
///
/// # Parity
///
/// This function is numerically equivalent to the naive
/// `log_sum_exp(logits) - gather(logits, labels)` path up to floating-point
/// associativity in the reduction across chunks. The CPU parity test
/// enforces `atol=1e-4 / rtol=1e-3` at bf16 and tighter at f32.
pub fn fused_linear_cross_entropy(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
) -> Result<Tensor> {
    let seq_len = input_ids.len();
    if seq_len < 2 {
        return Tensor::new(0.0f32, device).context("allocate zero loss scalar for seq_len < 2");
    }
    if label_mask.len() != seq_len {
        return Err(anyhow!(
            "label_mask length {} does not match input_ids length {}",
            label_mask.len(),
            seq_len,
        ));
    }
    if chunk_size == 0 {
        return Err(anyhow!("chunk_size must be > 0"));
    }

    // Squeeze batch dim: [seq_len, hidden_size]
    let hidden_2d = hidden.squeeze(0).context("squeeze batch dim from hidden")?;

    // Shift for next-token prediction. Use hidden[..seq_len-1] to predict
    // input_ids[1..]. Mask is also shifted to line up with the shifted labels.
    let shift_hidden = hidden_2d
        .narrow(0, 0, seq_len - 1)
        .context("narrow shift_hidden")?;
    let shift_labels: Vec<u32> = input_ids[1..].to_vec();
    let shift_mask: Vec<bool> = label_mask[1..].to_vec();

    // Gather active positions — these are the rows of `shift_hidden` we
    // score against their `shift_labels` entries.
    let active_positions: Vec<u32> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();

    if active_positions.is_empty() {
        return Tensor::new(0.0f32, device).context("allocate zero loss scalar");
    }

    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i as usize])
        .collect();

    let indices = Tensor::new(active_positions.as_slice(), device)
        .context("build active position indices")?;

    // `active_hidden`: [num_active, hidden_size]
    let active_hidden = shift_hidden
        .index_select(&indices, 0)
        .context("gather active_hidden rows")?;

    // Vocab size is the last dim of head_t ([hidden_size, vocab_size]).
    let head_dims = head_t.dims();
    if head_dims.len() != 2 {
        return Err(anyhow!(
            "head_t must be 2-D [hidden_size, vocab_size]; got {:?}",
            head_dims
        ));
    }
    let vocab_size = head_dims[1];

    // Accumulators in F32 for numerical stability. Phase A keeps these as
    // `Tensor`s so autograd can backprop into `active_hidden`; Phase B will
    // detach + recompute in a CustomOp.
    //
    // Invariant across chunks:
    //   running_max[i] = max_{j in [0, V_seen)} logits[i, j]
    //   running_sumexp[i] = sum_{j in [0, V_seen)} exp(logits[i, j] - running_max[i])
    //   correct_logit[i] = logits[i, labels[i]] (seen at most once across all chunks)
    let active_hidden_f32 = active_hidden.to_dtype(DType::F32)?;
    let head_t_f32 = head_t.to_dtype(DType::F32)?;

    let mut running_max: Option<Tensor> = None; // [num_active, 1]
    let mut running_sumexp: Option<Tensor> = None; // [num_active, 1] in exp-space relative to running_max
    let mut correct_logit: Option<Tensor> = None; // [num_active]

    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);

        // Head slice: [hidden_size, chunk_len].
        //
        // `narrow(1, off, chunk)` on a `[H, V]` tensor with stride `[V, 1]`
        // preserves stride `[V, 1]` for the slice rather than collapsing to
        // `[chunk, 1]`. CUDA matmul rejects strided right operands, so the
        // chunked-vocab path crashed on the first SFT step on Qwen3.5-4B
        // (V=248320). See PR #631 / docs/audits/PHASE10_FLCE_PREFLIGHT.md
        // Finding 1. CPU candle matmul is permissive about strides, which is
        // why the parity tests below missed it. Materialize a contiguous
        // chunk before the matmul.
        let head_chunk = head_t_f32
            .narrow(1, chunk_start, chunk_len)
            .context("slice head_t chunk")?
            .contiguous()
            .context("contiguous head_t chunk for matmul (CUDA matmul rejects strided rhs)")?;

        // Chunk logits: [num_active, chunk_len]. This is the ONE materialized
        // intermediate whose size scales with `chunk_len` instead of `vocab_size`.
        let logits_chunk = active_hidden_f32
            .matmul(&head_chunk)
            .context("matmul active_hidden_f32 @ head_chunk")?;

        // Per-row max within the chunk: [num_active, 1]
        let chunk_max = logits_chunk
            .max_keepdim(D::Minus1)
            .context("max_keepdim on logits_chunk")?;

        // Update running_max and rescale running_sumexp.
        let (new_max, new_sumexp) = match (running_max.as_ref(), running_sumexp.as_ref()) {
            (None, None) => {
                // First chunk: running_max = chunk_max, running_sumexp = sum(exp(chunk - chunk_max))
                let shifted = (&logits_chunk - chunk_max.broadcast_as(logits_chunk.shape())?)?;
                let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                (chunk_max.clone(), chunk_sumexp)
            }
            (Some(prev_max), Some(prev_sumexp)) => {
                // new_max = max(prev_max, chunk_max)
                // prev_sumexp *= exp(prev_max - new_max)
                // chunk_sumexp = sum(exp(logits_chunk - new_max))
                // new_sumexp = prev_sumexp + chunk_sumexp
                let new_max = prev_max.maximum(&chunk_max)?;
                let prev_scale = (prev_max - &new_max)?.exp()?;
                let scaled_prev = prev_sumexp.broadcast_mul(&prev_scale)?;
                let shifted = (&logits_chunk - new_max.broadcast_as(logits_chunk.shape())?)?;
                let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                let new_sumexp = (scaled_prev + chunk_sumexp)?;
                (new_max, new_sumexp)
            }
            _ => unreachable!("running_max and running_sumexp are set together"),
        };
        running_max = Some(new_max);
        running_sumexp = Some(new_sumexp);

        // For each active row whose label falls inside this chunk, gather the
        // correct logit from `logits_chunk`.
        let chunk_end = chunk_start + chunk_len;
        let mut chunk_hits: Vec<(u32, u32)> = Vec::new(); // (row_idx, label_local_in_chunk)
        for (row_idx, &label) in active_labels.iter().enumerate() {
            let label = label as usize;
            if label >= chunk_start && label < chunk_end {
                chunk_hits.push((row_idx as u32, (label - chunk_start) as u32));
            }
        }
        if !chunk_hits.is_empty() {
            let rows: Vec<u32> = chunk_hits.iter().map(|&(r, _)| r).collect();
            let cols: Vec<u32> = chunk_hits.iter().map(|&(_, c)| c).collect();
            let row_idx = Tensor::new(rows.as_slice(), device)?;
            let col_idx_2d = Tensor::new(cols.as_slice(), device)?.unsqueeze(1)?;

            // Gather first rows, then the specific column per row.
            let selected_rows = logits_chunk.index_select(&row_idx, 0)?; // [hits, chunk_len]
            let gathered = selected_rows.gather(&col_idx_2d, 1)?.squeeze(1)?; // [hits]

            // Scatter into a [num_active] F32 tensor. We initialize with zeros;
            // since each active row has exactly one label, each row is touched
            // exactly once across all chunks.
            let mut cur = match correct_logit.take() {
                Some(t) => t,
                None => Tensor::zeros(active_labels.len(), DType::F32, device)?,
            };
            // `index_add` along dim 0 with indices=row_idx accumulates gathered
            // into `cur`. Since each row appears in at most one chunk for its
            // one label, this is equivalent to a scatter.
            cur = cur.index_add(&row_idx, &gathered, 0)?;
            correct_logit = Some(cur);
        }

        chunk_start = chunk_end;
    }

    let running_max = running_max.ok_or_else(|| anyhow!("vocab_size was 0"))?;
    let running_sumexp = running_sumexp.ok_or_else(|| anyhow!("vocab_size was 0"))?;
    let correct_logit = correct_logit
        .ok_or_else(|| anyhow!("no labels fell inside any vocab chunk — label >= vocab_size?"))?;

    // log_sum_exp = running_max + log(running_sumexp). Squeeze the vocab dim.
    let log_sum_exp =
        (running_max.squeeze(D::Minus1)? + running_sumexp.squeeze(D::Minus1)?.log()?)?;

    // Per-token loss = log_sum_exp - correct_logit. Mean over active rows.
    let per_token_loss = (log_sum_exp - correct_logit)?;
    let loss = per_token_loss.mean_all()?;

    Ok(loss)
}

#[cfg(test)]
mod tests;

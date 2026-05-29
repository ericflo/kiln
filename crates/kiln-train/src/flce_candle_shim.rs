//! Candle-typed Fused Linear Cross-Entropy (FLCE) boundary for the
//! `kiln-train` SFT trainer ((#1082) — relocated out of
//! `kiln-flce-kernel`).
//!
//! # Why this module lives in `kiln-train` and not the kernel crate
//!
//! The THIRD kernel-crate candle drop ((#1082), after
//! `kiln-opd-loss-kernel` and `kiln-rmsnorm-kernel`):
//! `kiln-flce-kernel` is now 100% candle-free (pure `kiln_tensor` +
//! `kiln_autograd`). The candle-typed glue that the SFT trainer needs
//! — the pure-candle Phase A reference, the Phase B candle `CustomOp1`,
//! the `KtForwardOp1`-based kt-forward-op shim, and the kt-tape
//! production-caller adapter — moved UP into `kiln-train`, which
//! legitimately keeps `candle-core` (and already depends on
//! `kiln-kt-bridge`) for now. The kernel crate keeps the kt-typed
//! building blocks (`kt_api`, `kt_tape`) that this module calls; this
//! module is the candle↔kt boundary.
//!
//! Nothing in the FLCE math changed in the move: the Phase A composite,
//! the Phase B `CustomOp1`, the shim closures, and the tape adapter are
//! byte-identical in logic to their previous homes
//! (`kiln-flce-kernel/src/lib.rs`, `phase_b.rs`, `kt_forward_op.rs`,
//! `tape_forward.rs`). Only the crate location and the `crate::` →
//! `kiln_flce_kernel::` call paths changed.
//!
//! # Layout
//!
//! - **trait + dispatch** ([`FlceMatmulProvider`], [`FlceProvider`],
//!   [`fused_linear_cross_entropy_dispatch`],
//!   [`fused_linear_cross_entropy_dispatch_with_provider`]) — the
//!   candle-typed public surface the trainer calls.
//! - **Phase A** ([`fused_linear_cross_entropy`]) — the pure-candle
//!   reference path (autograd flows through chunk intermediates). Kept
//!   as the parity reference + `KILN_FLCE_PHASE_A=1` escape hatch.
//! - **Phase B** ([`fused_linear_cross_entropy_phase_b`] +
//!   [`fused_linear_cross_entropy_phase_b_with_provider`]) — the
//!   manual-backward candle `CustomOp1` (`FlceCustomOp`) whose `bwd()`
//!   routes through the kernel crate's kt bridge on CUDA.
//! - **kt-forward-op shim**
//!   ([`fused_linear_cross_entropy_phase_b_via_kt_forward_op`]) — a
//!   single candle `CustomOp1`
//!   ([`kiln_kt_bridge::forward_op::KtForwardOp1`]) wrapping the candle
//!   Phase-B forward + the kt-typed CUDA backward
//!   ([`kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_backward_kt`]).
//!   The production candle-autograd path.
//! - **kt-tape adapter** ([`try_tape_flce_phase_b_cuda`]) —
//!   `KILN_USE_TAPE_FORWARD`-gated adapter that records the FLCE
//!   backward onto a thread-local `kiln_autograd::Tape` via the kernel
//!   crate's kt-tape entry
//!   ([`kiln_flce_kernel::fused_linear_cross_entropy_phase_b_via_kt_tape`]).

use anyhow::{Context, Result, anyhow};
use candle_core::{D, DType, Device, Tensor};
use std::sync::Arc;

// Re-export the pure const from the kernel crate so existing call sites
// (`flce_candle_shim::DEFAULT_CHUNK_SIZE`) and external callers
// (`kiln_flce_kernel::DEFAULT_CHUNK_SIZE`) both resolve.
pub use kiln_flce_kernel::DEFAULT_CHUNK_SIZE;

// =========================================================================
// Candle-typed matmul provider trait + dispatch (relocated from
// kiln-flce-kernel/src/lib.rs)
// =========================================================================

/// Optional matmul override hook for the FLCE chunked head pass.
///
/// The default Phase B forward materializes the head as F32 (`head_t.to_dtype(F32)`,
/// ~2.5 GB on Qwen3.5-4B BF16), narrows it per-chunk, and dispatches each
/// `[active, hidden] @ [hidden, chunk_len]` matmul through candle's CPU
/// `broadcast_matmul`. On a unified-memory APU this is the dominant
/// remaining CPU compute in the training tail.
///
/// Implementations can route the per-chunk matmul through a Vulkan kernel
/// (or CUDA/Metal future-equivalents) without FLCE having to take a direct
/// dependency on the backend crate. Returning `Ok(None)` falls back to the
/// candle CPU path for that specific chunk; the caller's backward path
/// remains the analytic Phase B implementation.
///
/// The trait exposes chunk metadata (`full_rhs`, `chunk_start`,
/// `chunk_len`) so an implementation can upload `full_rhs` once and
/// reuse the same device buffer for every chunk via offset-aware
/// dispatch — the alternative (give the provider the already-narrowed
/// rhs Tensor) costs a fresh device-buffer upload per chunk because
/// candle's narrow yields a fresh `TensorId` and the underlying
/// per-tensor weight cache misses on every dispatch.
///
/// `lhs` is `[active, hidden]` F32. `full_rhs` is the original
/// `[hidden, vocab_size]` head_t in its original dtype. The chunk to
/// compute is `full_rhs[:, chunk_start .. chunk_start + chunk_len]`.
/// Expected output shape is `[active, chunk_len]` F32.
pub trait FlceMatmulProvider: Send + Sync + std::fmt::Debug {
    fn chunk_matmul(
        &self,
        lhs: &Tensor,
        full_rhs: &Tensor,
        chunk_start: usize,
        chunk_len: usize,
    ) -> Result<Option<Tensor>>;
}

/// Convenience boxed type used by the `_with_provider` entry points.
pub type FlceProvider = Arc<dyn FlceMatmulProvider>;

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
    fused_linear_cross_entropy_dispatch_with_provider(
        hidden, head_t, input_ids, label_mask, device, chunk_size, None,
    )
}

/// Same as [`fused_linear_cross_entropy_dispatch`] but accepts an optional
/// [`FlceProvider`] that the Phase B path consults for the per-chunk
/// matmul. Phase A ignores the provider (the env var path is the reference
/// implementation kept for parity debugging).
///
/// Trainer call sites that have a `BackendRuntime` handle build a
/// provider that wraps `backend.linear_prefill_apply` (or equivalent) and
/// pass it through here.
pub fn fused_linear_cross_entropy_dispatch_with_provider(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
    provider: Option<FlceProvider>,
) -> Result<Tensor> {
    if use_phase_a() {
        // Phase A is the reference path for parity debugging only; the
        // provider is intentionally not threaded through here.
        let _ = provider;
        fused_linear_cross_entropy(hidden, head_t, input_ids, label_mask, device, chunk_size)
    } else {
        // (#1082) Production path now routes through the
        // `KtForwardOp1` candle-autograd shim (commit `095f1c74`) over
        // the kt-typed forward + backward kernels. The shim falls
        // back to the candle Phase-B `CustomOp1` path when:
        //   - a `provider` is bound (the trainer's Vulkan FLCE escape;
        //     the shim has no provider plumbing),
        //   - `hidden` is not on CUDA,
        //   - `dtype` ∉ {F32, BF16} or hidden/head dtypes differ,
        //   - `active_count == 0` or `seq_len < 2`,
        //   - or the kill switch `KILN_DISABLE_FLCE_KT_FORWARD_OP=1`
        //     is set.
        // The autograd chain through `loss.backward()` is preserved
        // in either case — both the shim and the Phase-B path are
        // candle `CustomOp1`s parented on `hidden`.
        //
        // Wave-13 (#1082): when `KILN_USE_TAPE_FORWARD=1` AND a thread-
        // local `kiln_autograd::Tape` scope is active (and no provider
        // is bound — the tape entry has no provider plumbing), route
        // through `try_tape_flce_phase_b_cuda` first. The forward
        // result is bit-exact with the kt-shim (same kt-typed forward
        // underneath); the backward node is recorded on the tape for
        // `Tape::backward`. With any gate off (the default)
        // `try_tape_flce_phase_b_cuda` returns `Ok(None)` and we fall
        // through to the existing kt-shim — preserving the
        // candle-autograd chain for callers driving gradients via
        // `loss.backward()`.
        #[cfg(feature = "cuda")]
        {
            if provider.is_none() {
                if let Some(out) = try_tape_flce_phase_b_cuda(
                    hidden, head_t, input_ids, label_mask, chunk_size,
                )? {
                    return Ok(out);
                }
            }
        }
        fused_linear_cross_entropy_phase_b_via_kt_forward_op(
            hidden, head_t, input_ids, label_mask, device, chunk_size, provider,
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

// =========================================================================
// FLCE Phase B — manual-backward CustomOp1 (relocated from
// kiln-flce-kernel/src/phase_b.rs)
// =========================================================================
//
// # Why a CustomOp1
//
// Phase A keeps `logits_chunk`, `shifted`, and `shifted.exp()` live for
// candle autograd because the scalar loss depends on `running_sumexp`,
// which is a sum-tree across all chunks' `shifted.exp()` nodes. At T=8192
// with V=248320 / chunk=4096 = 61 chunks, that is 61 × 3 × 127.81 MiB ≈
// 23 GiB of intermediates pinned in the forward graph. PR #646 traced
// the OOM allocation directly to this pattern (see
// docs/audits/PHASE10_MODE_B_TRACE.md).
//
// Phase B replaces the autograd graph with a `CustomOp1` whose
// `cpu_fwd`/`cuda_fwd` runs the chunked forward in a function-local scope,
// producing only the scalar loss as output. Estimated peak-VRAM saving at
// T=8192: ~22 GiB.

use candle_core::backend::{BackendDevice, BackendStorage};
use candle_core::op::BackpropOp;
use candle_core::{
    CpuStorage, CudaStorage, CustomOp1, Layout, MetalStorage, Shape, Storage,
};

#[cfg(feature = "cuda")]
use std::sync::OnceLock;

/// Process-wide kill switch for [`fused_linear_cross_entropy_phase_b_backward_via_kt_bridge`].
///
/// Set `KILN_DISABLE_FLCE_BWD_KT_BRIDGE=1` to fall back to the candle-typed
/// `backward_dhidden` path (same math; this is purely a reversibility /
/// parity-test escape hatch). Mirrors the precedent established by
/// `fused_rmsnorm_backward_via_kt_bridge` (commit `341da876`),
/// `fused_rotary_one_backward_via_kt_bridge` (commit `d99a15a3`), and
/// `opd_loss_phase_b_backward_via_kt_bridge` (commit `0c1be227`).
#[cfg(feature = "cuda")]
fn flce_bwd_kt_bridge_disabled() -> bool {
    static DISABLED: OnceLock<bool> = OnceLock::new();
    *DISABLED.get_or_init(|| {
        matches!(
            std::env::var("KILN_DISABLE_FLCE_BWD_KT_BRIDGE")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("TRUE")
        )
    })
}

/// Phase B entry point: chunked FLCE with a manual-backward [`CustomOp1`].
///
/// Behaves identically to [`fused_linear_cross_entropy`] up to
/// floating-point associativity in the reduction across chunks, but routes
/// the autograd graph through a custom op so chunk intermediates do not
/// pin ~23 GiB of VRAM at T=8192 SFT.
pub fn fused_linear_cross_entropy_phase_b(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
) -> Result<Tensor> {
    fused_linear_cross_entropy_phase_b_with_provider(
        hidden, head_t, input_ids, label_mask, device, chunk_size, None,
    )
}

/// Provider-aware variant. The optional [`FlceProvider`] is consulted for
/// every chunk matmul in both forward and backward — when it returns
/// `Ok(Some(out))` the result is used directly; on `Ok(None)` the candle
/// CPU `broadcast_matmul` path runs as before.
pub fn fused_linear_cross_entropy_phase_b_with_provider(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
    provider: Option<FlceProvider>,
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
    let head_dims = head_t.dims();
    if head_dims.len() != 2 {
        return Err(anyhow!(
            "head_t must be 2-D [hidden_size, vocab_size]; got {:?}",
            head_dims
        ));
    }
    let hidden_dims = hidden.dims();
    if hidden_dims.len() != 3 {
        return Err(anyhow!(
            "hidden must be 3-D [1, seq_len, hidden_size]; got {:?}",
            hidden_dims
        ));
    }
    if hidden_dims[0] != 1 {
        return Err(anyhow!("hidden batch dim must be 1; got {:?}", hidden_dims));
    }
    if hidden_dims[2] != head_dims[0] {
        return Err(anyhow!(
            "hidden hidden_size {} != head_t hidden_size {}",
            hidden_dims[2],
            head_dims[0],
        ));
    }

    // Short-circuit when no positions are active. Phase A returns a zero
    // tensor with no autograd parent in this case; Phase B does the same so
    // calling .backward() on the result is a no-op for `hidden`.
    let active_count = label_mask[1..].iter().filter(|&&m| m).count();
    if active_count == 0 {
        return Tensor::new(0.0f32, device).context("allocate zero loss scalar (no active rows)");
    }

    // Apply the custom op. The op closes over head_t / input_ids / label_mask /
    // chunk_size; only `hidden` is the autograd input.
    let hidden_contig = hidden
        .contiguous()
        .context("contiguous hidden for FLCE phase B")?;
    let op = FlceCustomOp {
        head_t: head_t.clone(),
        input_ids: input_ids.to_vec(),
        label_mask: label_mask.to_vec(),
        chunk_size,
        provider,
    };
    hidden_contig.apply_op1(op).map_err(Into::into)
}

/// CustomOp1 wrapper for Phase B. `apply_op1(hidden)` -> scalar f32 loss.
#[derive(Debug)]
struct FlceCustomOp {
    /// `[hidden_size, vocab_size]` transposed lm_head — frozen during LoRA
    /// training, so it is captured here as op state rather than an autograd
    /// input.
    head_t: Tensor,
    /// Token ids for the sequence (length `seq_len`).
    input_ids: Vec<u32>,
    /// Loss mask aligned with `input_ids`; positions where `label_mask[i+1]`
    /// is true contribute to the loss.
    label_mask: Vec<bool>,
    /// Chunk size along the vocab dim. Use [`DEFAULT_CHUNK_SIZE`] unless
    /// tuning.
    chunk_size: usize,
    /// Optional matmul override for the per-chunk `[active, hidden] @
    /// [hidden, chunk_len]` step.
    provider: Option<FlceProvider>,
}

impl CustomOp1 for FlceCustomOp {
    fn name(&self) -> &'static str {
        "kiln-flce-phase-b"
    }

    fn cpu_fwd(
        &self,
        s_hidden: &CpuStorage,
        l_hidden: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let storage = Storage::Cpu(s_hidden.clone());
        let hidden_shape = Shape::from(l_hidden.shape().dims());
        let hidden_leaf = Tensor::from_storage(storage, hidden_shape, BackpropOp::none(), false);

        let loss = forward_loss(
            &hidden_leaf,
            &self.head_t,
            &self.input_ids,
            &self.label_mask,
            self.chunk_size,
            self.provider.as_ref(),
        )
        .map_err(|e| candle_core::Error::Msg(format!("flce phase b cpu_fwd: {e:#}")))?;

        Ok((CpuStorage::F32(vec![loss]), Shape::from(())))
    }

    fn cuda_fwd(
        &self,
        s_hidden: &CudaStorage,
        l_hidden: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (s_hidden, l_hidden);
            return Err(candle_core::Error::Msg(
                "flce phase b cuda_fwd: kiln-train built without `cuda` feature".into(),
            ));
        }
        #[cfg(feature = "cuda")]
        {
            let storage = Storage::Cuda(s_hidden.try_clone(l_hidden)?);
            let hidden_shape = Shape::from(l_hidden.shape().dims());
            let hidden_leaf =
                Tensor::from_storage(storage, hidden_shape, BackpropOp::none(), false);

            let loss_value = forward_loss(
                &hidden_leaf,
                &self.head_t,
                &self.input_ids,
                &self.label_mask,
                self.chunk_size,
                self.provider.as_ref(),
            )
            .map_err(|e| candle_core::Error::Msg(format!("flce phase b cuda_fwd: {e:#}")))?;

            let device = s_hidden.device();
            let out_slice = device.clone_htod(&[loss_value])?;
            Ok((
                CudaStorage::wrap_cuda_slice(out_slice, device.clone()),
                Shape::from(()),
            ))
        }
    }

    fn metal_fwd(
        &self,
        s_hidden: &MetalStorage,
        l_hidden: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let storage = Storage::Metal(s_hidden.try_clone(l_hidden)?);
        let hidden_shape = Shape::from(l_hidden.shape().dims());
        let hidden_leaf = Tensor::from_storage(storage, hidden_shape, BackpropOp::none(), false);

        let loss_value = forward_loss(
            &hidden_leaf,
            &self.head_t,
            &self.input_ids,
            &self.label_mask,
            self.chunk_size,
            self.provider.as_ref(),
        )
        .map_err(|e| candle_core::Error::Msg(format!("flce phase b metal_fwd: {e:#}")))?;

        let device = s_hidden.device();
        let out_storage = device.storage_from_slice(&[loss_value])?;
        Ok((out_storage, Shape::from(())))
    }

    fn bwd(
        &self,
        hidden: &Tensor,
        _loss: &Tensor,
        grad_loss: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        // CUDA kt-bridge fast path. Route through the kt-typed backward
        // (`fused_linear_cross_entropy_phase_b_backward_kt`) when:
        //   (a) hidden lives on CUDA,
        //   (b) no `FlceProvider` is bound,
        //   (c) the kill switch `KILN_DISABLE_FLCE_BWD_KT_BRIDGE=1` is not set.
        #[cfg(feature = "cuda")]
        {
            let on_cuda = matches!(hidden.device(), Device::Cuda(_));
            if on_cuda && self.provider.is_none() && !flce_bwd_kt_bridge_disabled() {
                match fused_linear_cross_entropy_phase_b_backward_via_kt_bridge(
                    self, hidden, grad_loss,
                ) {
                    Ok(dh) => return Ok(Some(dh)),
                    Err(e) => {
                        tracing::warn!(
                            "kiln-train flce: kt-bridge bwd path failed, falling back to candle: {e}"
                        );
                    }
                }
            }
        }

        backward_dhidden(
            hidden,
            &self.head_t,
            &self.input_ids,
            &self.label_mask,
            self.chunk_size,
            grad_loss,
            self.provider.as_ref(),
        )
        .map(Some)
        .map_err(|e| candle_core::Error::Msg(format!("flce phase b bwd: {e:#}")))
    }
}

/// kt-bridge variant of [`FlceCustomOp::bwd`] — borrows `hidden`/`head_t`/`grad_loss`
/// as kt-Tensors and dispatches the same two-pass chunked recompute as the
/// candle [`backward_dhidden`] via
/// [`kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_backward_kt`],
/// then copies the resulting `dhidden` back into a candle `Tensor`.
#[cfg(feature = "cuda")]
fn fused_linear_cross_entropy_phase_b_backward_via_kt_bridge(
    op: &FlceCustomOp,
    hidden: &Tensor,
    grad_loss: &Tensor,
) -> std::result::Result<Tensor, candle_core::Error> {
    use kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_backward_kt;

    let head_t_c = op
        .head_t
        .contiguous()
        .map_err(|e| candle_core::Error::Msg(format!("kt-bridge flce bwd: head_t contiguous: {e}")))?;

    let grad_loss_f32 = grad_loss
        .to_dtype(DType::F32)
        .map_err(|e| candle_core::Error::Msg(format!("kt-bridge flce bwd: cast grad_loss: {e}")))?;
    let grad_loss_c = grad_loss_f32
        .contiguous()
        .map_err(|e| candle_core::Error::Msg(format!("kt-bridge flce bwd: contiguous grad_loss: {e}")))?;

    let hidden_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(hidden).map_err(|e| {
        candle_core::Error::Msg(format!("kt-bridge flce bwd: borrow hidden failed: {e}"))
    })?;
    let head_t_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&head_t_c).map_err(|e| {
        candle_core::Error::Msg(format!("kt-bridge flce bwd: borrow head_t failed: {e}"))
    })?;
    let grad_loss_kt =
        kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&grad_loss_c).map_err(|e| {
            candle_core::Error::Msg(format!("kt-bridge flce bwd: borrow grad_loss failed: {e}"))
        })?;

    let d_hidden_kt = fused_linear_cross_entropy_phase_b_backward_kt(
        &hidden_kt,
        &head_t_kt,
        &op.input_ids,
        &op.label_mask,
        op.chunk_size,
        &grad_loss_kt,
    )
    .map_err(|e| candle_core::Error::Msg(format!("kt-bridge flce bwd: kt call failed: {e}")))?;

    let d_hidden_kt_contig = if d_hidden_kt.is_contiguous() {
        d_hidden_kt
    } else {
        d_hidden_kt.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!("kt-bridge flce bwd: contiguous d_hidden: {e}"))
        })?
    };

    kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&d_hidden_kt_contig).map_err(|e| {
        candle_core::Error::Msg(format!("kt-bridge flce bwd: copy-back d_hidden failed: {e}"))
    })
}

/// Run one chunk's `[active, hidden] @ [hidden, chunk_len]` matmul,
/// preferring the optional `FlceMatmulProvider` and falling back to
/// candle's `Tensor::matmul` when the provider declines.
fn forward_chunk_matmul(
    lhs: &Tensor,
    full_rhs: &Tensor,
    narrowed_rhs: &Tensor,
    chunk_start: usize,
    chunk_len: usize,
    provider: Option<&FlceProvider>,
) -> Result<Tensor> {
    if let Some(p) = provider {
        if let Some(out) = p.chunk_matmul(lhs, full_rhs, chunk_start, chunk_len)? {
            return Ok(out);
        }
    }
    lhs.matmul(narrowed_rhs).map_err(Into::into)
}

fn synchronize_metal_chunk(device: &Device, context: &'static str) -> Result<()> {
    if matches!(device, Device::Metal(_)) {
        device.synchronize().context(context)?;
    }
    Ok(())
}

fn forward_loss(
    hidden_leaf: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
    provider: Option<&FlceProvider>,
) -> Result<f32> {
    let device = hidden_leaf.device();
    let seq_len = input_ids.len();
    debug_assert!(seq_len >= 2);
    debug_assert_eq!(label_mask.len(), seq_len);

    let hidden_2d = hidden_leaf
        .squeeze(0)
        .context("squeeze batch dim from hidden")?;
    let shift_hidden = hidden_2d
        .narrow(0, 0, seq_len - 1)
        .context("narrow shift_hidden")?;
    let shift_labels: &[u32] = &input_ids[1..];
    let shift_mask: &[bool] = &label_mask[1..];

    let active_positions: Vec<u32> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    debug_assert!(!active_positions.is_empty(), "caller short-circuits empty");

    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i as usize])
        .collect();
    let num_active = active_positions.len();

    let active_indices = Tensor::new(active_positions.as_slice(), device)
        .context("build active position indices")?;
    let active_hidden = shift_hidden
        .index_select(&active_indices, 0)
        .context("gather active_hidden rows")?;

    let active_hidden_f32 = active_hidden.to_dtype(DType::F32)?;
    let head_t_f32 = head_t.to_dtype(DType::F32)?;
    let vocab_size = head_t.dim(1)?;

    let mut running_max: Option<Tensor> = None;
    let mut running_sumexp: Option<Tensor> = None;
    let mut correct_logit: Option<Tensor> = None;

    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let chunk_end = chunk_start + chunk_len;

        {
            let head_chunk = head_t_f32
                .narrow(1, chunk_start, chunk_len)
                .context("slice head_t chunk")?
                .contiguous()
                .context("contiguous head_t chunk for matmul")?;

            let logits_chunk = forward_chunk_matmul(
                &active_hidden_f32,
                head_t,
                &head_chunk,
                chunk_start,
                chunk_len,
                provider,
            )
            .context("matmul active_hidden_f32 @ head_chunk")?;

            let chunk_max = logits_chunk
                .max_keepdim(D::Minus1)
                .context("max_keepdim on logits_chunk")?;

            let (new_max, new_sumexp) = match (running_max.as_ref(), running_sumexp.as_ref()) {
                (None, None) => {
                    let shifted =
                        (&logits_chunk - chunk_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                    (chunk_max.detach(), chunk_sumexp.detach())
                }
                (Some(prev_max), Some(prev_sumexp)) => {
                    let new_max = prev_max.maximum(&chunk_max)?;
                    let prev_scale = (prev_max - &new_max)?.exp()?;
                    let scaled_prev = prev_sumexp.broadcast_mul(&prev_scale)?;
                    let shifted = (&logits_chunk - new_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                    let new_sumexp = (scaled_prev + chunk_sumexp)?;
                    (new_max.detach(), new_sumexp.detach())
                }
                _ => unreachable!("running_max and running_sumexp are set together"),
            };
            running_max = Some(new_max);
            running_sumexp = Some(new_sumexp);

            let mut chunk_hits: Vec<(u32, u32)> = Vec::new();
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
                let selected_rows = logits_chunk.index_select(&row_idx, 0)?;
                let gathered = selected_rows.gather(&col_idx_2d, 1)?.squeeze(1)?;
                let mut cur = match correct_logit.take() {
                    Some(t) => t,
                    None => Tensor::zeros(num_active, DType::F32, device)?,
                };
                cur = cur.index_add(&row_idx, &gathered, 0)?;
                correct_logit = Some(cur.detach());
            }
        }

        synchronize_metal_chunk(device, "synchronize FLCE phase B forward chunk")?;
        chunk_start = chunk_end;
    }

    let running_max = running_max.ok_or_else(|| anyhow!("vocab_size was 0"))?;
    let running_sumexp = running_sumexp.ok_or_else(|| anyhow!("vocab_size was 0"))?;
    let correct_logit = correct_logit
        .ok_or_else(|| anyhow!("no labels fell inside any vocab chunk — label >= vocab_size?"))?;

    let log_sum_exp =
        (running_max.squeeze(D::Minus1)? + running_sumexp.squeeze(D::Minus1)?.log()?)?;
    let per_token_loss = (log_sum_exp - correct_logit)?;
    let loss = per_token_loss.mean_all()?;
    Ok(loss.to_scalar::<f32>()?)
}

/// Backward implementation. Runs the chunk loop twice (recompute
/// running_max/sumexp, then accumulate dhidden by chunk). Returns
/// `dhidden` as a `[1, seq_len, hidden_size]` tensor in the original dtype.
fn backward_dhidden(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
    grad_loss: &Tensor,
    provider: Option<&FlceProvider>,
) -> Result<Tensor> {
    let device = hidden.device();
    let dtype = hidden.dtype();
    let seq_len = input_ids.len();
    debug_assert!(seq_len >= 2);
    debug_assert_eq!(label_mask.len(), seq_len);

    let hidden_dims = hidden.dims();
    let hidden_size = hidden_dims[2];

    let hidden_2d = hidden.squeeze(0)?;
    let shift_hidden = hidden_2d.narrow(0, 0, seq_len - 1)?;
    let shift_labels: &[u32] = &input_ids[1..];
    let shift_mask: &[bool] = &label_mask[1..];

    let active_positions: Vec<u32> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();

    if active_positions.is_empty() {
        return Ok(Tensor::zeros(hidden.shape(), dtype, device)?);
    }

    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i as usize])
        .collect();
    let num_active = active_positions.len();

    let active_indices = Tensor::new(active_positions.as_slice(), device)?;
    let active_hidden = shift_hidden.index_select(&active_indices, 0)?;
    let active_hidden_f32 = active_hidden.to_dtype(DType::F32)?;
    let head_t_f32 = head_t.to_dtype(DType::F32)?;
    let vocab_size = head_t.dim(1)?;

    // Pass 1: recompute running_max + running_sumexp.
    let mut running_max: Option<Tensor> = None;
    let mut running_sumexp: Option<Tensor> = None;
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        {
            let head_chunk = head_t_f32.narrow(1, chunk_start, chunk_len)?.contiguous()?;
            let logits_chunk = forward_chunk_matmul(
                &active_hidden_f32,
                head_t,
                &head_chunk,
                chunk_start,
                chunk_len,
                provider,
            )?;
            let chunk_max = logits_chunk.max_keepdim(D::Minus1)?;
            let (new_max, new_sumexp) = match (running_max.as_ref(), running_sumexp.as_ref()) {
                (None, None) => {
                    let shifted =
                        (&logits_chunk - chunk_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                    (chunk_max.detach(), chunk_sumexp.detach())
                }
                (Some(prev_max), Some(prev_sumexp)) => {
                    let new_max = prev_max.maximum(&chunk_max)?;
                    let prev_scale = (prev_max - &new_max)?.exp()?;
                    let scaled_prev = prev_sumexp.broadcast_mul(&prev_scale)?;
                    let shifted = (&logits_chunk - new_max.broadcast_as(logits_chunk.shape())?)?;
                    let chunk_sumexp = shifted.exp()?.sum_keepdim(D::Minus1)?;
                    let new_sumexp = (scaled_prev + chunk_sumexp)?;
                    (new_max.detach(), new_sumexp.detach())
                }
                _ => unreachable!(),
            };
            running_max = Some(new_max);
            running_sumexp = Some(new_sumexp);
        }
        synchronize_metal_chunk(device, "synchronize FLCE phase B backward normalizer chunk")?;
        chunk_start += chunk_len;
    }
    let running_max = running_max.ok_or_else(|| anyhow!("vocab_size was 0"))?;
    let running_sumexp = running_sumexp.ok_or_else(|| anyhow!("vocab_size was 0"))?;

    // Pass 2: accumulate dhidden_active by chunk.
    let grad_loss_f32 = grad_loss.to_dtype(DType::F32)?;
    let inv_n = 1.0f64 / (num_active as f64);

    let mut dhidden_active = Tensor::zeros((num_active, hidden_size), DType::F32, device)?;
    let mut chunk_start = 0usize;
    while chunk_start < vocab_size {
        let chunk_len = chunk_size.min(vocab_size - chunk_start);
        let chunk_end = chunk_start + chunk_len;
        {
            let head_chunk = head_t_f32.narrow(1, chunk_start, chunk_len)?.contiguous()?;
            let logits_chunk = forward_chunk_matmul(
                &active_hidden_f32,
                head_t,
                &head_chunk,
                chunk_start,
                chunk_len,
                provider,
            )?;
            let shifted = (&logits_chunk - running_max.broadcast_as(logits_chunk.shape())?)?;
            let exp_chunk = shifted.exp()?;
            let softmax_chunk =
                exp_chunk.broadcast_div(&running_sumexp.broadcast_as(logits_chunk.shape())?)?;

            let mut one_hot_data: Vec<f32> = vec![0.0; num_active * chunk_len];
            for (row_idx, &label) in active_labels.iter().enumerate() {
                let label = label as usize;
                if label >= chunk_start && label < chunk_end {
                    let col = label - chunk_start;
                    one_hot_data[row_idx * chunk_len + col] = 1.0;
                }
            }
            let one_hot = Tensor::from_vec(one_hot_data, (num_active, chunk_len), device)?;

            let diff = (softmax_chunk - one_hot)?;
            let scaled = diff.affine(inv_n, 0.0)?;
            let grad_logits_chunk = scaled.broadcast_mul(&grad_loss_f32)?;

            let head_chunk_t = head_chunk.t()?.contiguous()?;
            let chunk_contrib = grad_logits_chunk.matmul(&head_chunk_t)?;

            dhidden_active = (&dhidden_active + chunk_contrib)?.detach();
        }
        synchronize_metal_chunk(device, "synchronize FLCE phase B backward gradient chunk")?;

        chunk_start = chunk_end;
    }

    let mut grad_hidden_2d = Tensor::zeros((seq_len, hidden_size), DType::F32, device)?;
    grad_hidden_2d = grad_hidden_2d.index_add(&active_indices, &dhidden_active, 0)?;

    let grad_hidden_3d = grad_hidden_2d.unsqueeze(0)?;
    let dhidden = grad_hidden_3d.to_dtype(dtype)?;
    Ok(dhidden)
}

// =========================================================================
// kt-forward-op shim — production caller migration to KtForwardOp1
// (relocated from kiln-flce-kernel/src/kt_forward_op.rs)
// =========================================================================
//
// [`fused_linear_cross_entropy_phase_b_via_kt_forward_op`] replaces the
// Phase-A and Phase-B candle composites with a single candle `CustomOp1`
// — [`kiln_kt_bridge::forward_op::KtForwardOp1`] — whose forward closure
// runs the candle Phase-B forward as a leaf op and whose backward closure
// calls the kt-typed CUDA backward
// ([`kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_backward_kt`]).
// Mirrors the OPD migration (commit `f214f168`): forward via candle,
// backward via kt.

/// Read the `KILN_DISABLE_FLCE_KT_FORWARD_OP` kill switch. When set
/// (`1` / `true` / `yes` / `TRUE`), the production caller falls back
/// to the candle Phase-B `CustomOp1` path. Same convention as
/// `KILN_DISABLE_FLCE_BWD_KT_BRIDGE` (commit `ab2da23f`),
/// `KILN_DISABLE_OPD_KT_FORWARD_OP` (commit `f214f168`),
/// `KILN_DISABLE_RMSNORM_KERNEL`, `KILN_DISABLE_FUSED_CONV1D`, etc.
pub fn kt_forward_op_disabled() -> bool {
    std::env::var("KILN_DISABLE_FLCE_KT_FORWARD_OP")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// Returns `true` when the `(dtype, head_t.dtype())` triple is in
/// the fused kt-bwd envelope AND `hidden` is on CUDA.
fn shim_envelope_ok(hidden: &Tensor, head_t: &Tensor) -> bool {
    if !matches!(hidden.device(), Device::Cuda(_)) {
        return false;
    }
    let h_dt = hidden.dtype();
    let dtype_ok = matches!(h_dt, DType::F32 | DType::BF16);
    if !dtype_ok {
        return false;
    }
    if h_dt != head_t.dtype() {
        return false;
    }
    true
}

/// kt-shim FLCE phase-B with candle-autograd integration.
///
/// Behavioral envelope:
/// - CUDA + `dtype in {F32, BF16}` + matching `head_t` dtype + no
///   `FlceProvider` bound + non-empty active rows → routes through
///   [`KtForwardOp1`] over the kt-typed forward+backward (the
///   forward closure currently uses the candle Phase-B body).
/// - Anything outside the envelope → falls through to
///   [`fused_linear_cross_entropy_phase_b_with_provider`] (the
///   candle `CustomOp1` reference path). The autograd chain through
///   `loss.backward()` is preserved in either case.
///
/// [`KtForwardOp1`]: kiln_kt_bridge::forward_op::KtForwardOp1
pub fn fused_linear_cross_entropy_phase_b_via_kt_forward_op(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
    provider: Option<FlceProvider>,
) -> Result<Tensor> {
    // The provider hook is explicit Phase-B state; the shim has no
    // provider plumbing today. When a provider is bound, defer to the
    // Phase-B candle path so the trainer's Vulkan FLCE escape stays intact.
    if provider.is_some() {
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, provider,
        );
    }

    if kt_forward_op_disabled() || !shim_envelope_ok(hidden, head_t) {
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        );
    }

    let seq_len = input_ids.len();
    if seq_len < 2 {
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        );
    }
    if label_mask.len() != seq_len {
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        );
    }
    let active_count = label_mask[1..].iter().filter(|&&m| m).count();
    if active_count == 0 {
        return fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        );
    }

    #[cfg(feature = "cuda")]
    {
        return cuda_via_kt_forward_op(
            hidden,
            head_t,
            input_ids,
            label_mask,
            device,
            chunk_size,
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        fused_linear_cross_entropy_phase_b_with_provider(
            hidden, head_t, input_ids, label_mask, device, chunk_size, None,
        )
    }
}

#[cfg(feature = "cuda")]
fn cuda_via_kt_forward_op(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    device: &Device,
    chunk_size: usize,
) -> Result<Tensor> {
    use kiln_flce_kernel::kt_api::fused_linear_cross_entropy_phase_b_backward_kt;
    use kiln_kt_bridge::forward_op::KtForwardOp1;
    use kiln_kt_bridge::{
        kt_tensor_from_candle_cuda_borrow, kt_tensor_to_candle_cuda_copy,
    };

    let hidden_contig = hidden
        .contiguous()
        .context("force-contiguous hidden for FLCE kt-shim")?;

    let head_t_owned_fwd = head_t.clone();
    let head_t_owned_bwd = head_t.clone();
    let input_ids_fwd = input_ids.to_vec();
    let input_ids_bwd = input_ids.to_vec();
    let label_mask_fwd = label_mask.to_vec();
    let label_mask_bwd = label_mask.to_vec();
    let device_fwd = device.clone();

    // ----- Forward closure: candle Phase-B body on the leaf tensor -----
    let forward = move |hidden_in: &Tensor| -> candle_core::Result<Tensor> {
        let loss = fused_linear_cross_entropy_phase_b_with_provider(
            hidden_in,
            &head_t_owned_fwd,
            &input_ids_fwd,
            &label_mask_fwd,
            &device_fwd,
            chunk_size,
            None,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim fwd: phase-B candle composite: {e}"
            ))
        })?;
        loss.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim fwd: contiguous loss scalar: {e}"
            ))
        })
    };

    // ----- Backward closure: kt-typed CUDA backward -----
    let backward = move |arg: &Tensor,
                         _res: &Tensor,
                         grad_res: &Tensor|
          -> candle_core::Result<Option<Tensor>> {
        let hidden_c = arg.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: contiguous hidden: {e}"
            ))
        })?;
        let head_t_c = head_t_owned_bwd.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: contiguous head_t: {e}"
            ))
        })?;
        let grad_res_f32 = grad_res.to_dtype(DType::F32).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: cast grad_loss to F32: {e}"
            ))
        })?;
        let grad_res_c = grad_res_f32.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: contiguous grad_loss: {e}"
            ))
        })?;

        let hidden_kt = kt_tensor_from_candle_cuda_borrow(&hidden_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: borrow hidden: {e}"
            ))
        })?;
        let head_t_kt = kt_tensor_from_candle_cuda_borrow(&head_t_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: borrow head_t: {e}"
            ))
        })?;
        let grad_res_kt = kt_tensor_from_candle_cuda_borrow(&grad_res_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: borrow grad_loss: {e}"
            ))
        })?;

        let d_hidden_kt = fused_linear_cross_entropy_phase_b_backward_kt(
            &hidden_kt,
            &head_t_kt,
            &input_ids_bwd,
            &label_mask_bwd,
            chunk_size,
            &grad_res_kt,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: kt bwd call: {e}"
            ))
        })?;

        let d_hidden_kt_c = if d_hidden_kt.is_contiguous() {
            d_hidden_kt
        } else {
            d_hidden_kt.contiguous().map_err(|e| {
                candle_core::Error::Msg(format!(
                    "flce kt-shim bwd: contiguous d_hidden: {e}"
                ))
            })?
        };

        let d_hidden = kt_tensor_to_candle_cuda_copy(&d_hidden_kt_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "flce kt-shim bwd: copy-back d_hidden: {e}"
            ))
        })?;

        Ok(Some(d_hidden))
    };

    // ----- Apply -----
    let op = KtForwardOp1::new("kiln-flce-kt-forward-op", forward, backward);
    let _ = device; // unused; the device is implicit in `hidden_contig`.
    hidden_contig
        .apply_op1_arc(Arc::new(Box::new(op)))
        .context("apply FLCE kt-forward-op to hidden")
}

// =========================================================================
// kt-tape production-caller adapter (relocated from
// kiln-flce-kernel/src/tape_forward.rs)
// =========================================================================
//
// `KILN_USE_TAPE_FORWARD`-gated adapter that records the FLCE backward
// onto the active thread-local `kiln_autograd::Tape` via the kernel
// crate's kt-tape entry. Off by default — returns `Ok(None)` and the
// caller falls through to the kt-forward-op shim, preserving the candle
// autograd chain. See
// docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md for the
// architectural reason it does not replace the shim outright.

/// Attempt to run FLCE Phase-B through the kt-tape pilot instead of the
/// candle-typed [`fused_linear_cross_entropy_phase_b_via_kt_forward_op`]
/// shim.
///
/// Returns:
/// * `Ok(Some(out))` — the tape-forward path ran; the returned `Tensor`
///   is a copy of the kt-tape scalar loss into a candle CUDA tensor and
///   the backward node was recorded on the active thread-local tape.
/// * `Ok(None)` — env gate off, no active tape scope, out-of-envelope,
///   active count 0, or kt borrow failed. The caller falls through.
/// * `Err(...)` — a kt-tape forward error (envelope OK but FFI failed).
#[cfg(feature = "cuda")]
pub fn try_tape_flce_phase_b_cuda(
    hidden: &Tensor,
    head_t: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    chunk_size: usize,
) -> Result<Option<Tensor>> {
    use kiln_autograd::{tape_forward_enabled, with_active_tape, Tape};
    use kiln_flce_kernel::fused_linear_cross_entropy_phase_b_via_kt_tape;

    if !tape_forward_enabled() {
        return Ok(None);
    }

    // Active-count + seq_len short-circuit — match the kt-shim envelope.
    if label_mask.len() < 2 || label_mask.iter().filter(|&&m| m).count() == 0 {
        return Ok(None);
    }
    if input_ids.len() != label_mask.len() {
        return Ok(None);
    }

    let hidden_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(hidden) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let head_t_kt = match kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(head_t) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

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

// =========================================================================
// Unit tests for the kill switch (relocated from
// kiln-flce-kernel/src/kt_forward_op.rs).
// =========================================================================

#[cfg(test)]
mod kt_forward_op_tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate `KILN_DISABLE_FLCE_KT_FORWARD_OP`
    // so they don't race against each other.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|poisoned| {
            ENV_LOCK.clear_poison();
            poisoned.into_inner()
        })
    }

    #[test]
    fn kill_switch_default_off() {
        let _guard = env_lock();
        let prior = std::env::var("KILN_DISABLE_FLCE_KT_FORWARD_OP").ok();
        // SAFETY: `kt_forward_op_disabled()` reads the env on each call
        // (no caching); the ENV_LOCK guard serializes mutation.
        unsafe {
            std::env::remove_var("KILN_DISABLE_FLCE_KT_FORWARD_OP");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", "0");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", "false");
        }
        assert!(!kt_forward_op_disabled());

        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_FLCE_KT_FORWARD_OP"),
            }
        }
    }

    #[test]
    fn kill_switch_on() {
        let _guard = env_lock();
        let prior = std::env::var("KILN_DISABLE_FLCE_KT_FORWARD_OP").ok();
        for v in ["1", "true", "yes", "TRUE", "Yes"] {
            unsafe {
                std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", v);
            }
            assert!(
                kt_forward_op_disabled(),
                "expected disabled for env={v}"
            );
        }
        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_FLCE_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_FLCE_KT_FORWARD_OP"),
            }
        }
    }
}

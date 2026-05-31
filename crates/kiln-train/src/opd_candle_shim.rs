//! Candle-typed OPD top-K reverse-KL boundary for the `kiln-train` OPD
//! trainer ((#1082) — relocated out of `kiln-opd-loss-kernel`).
//!
//! # Why this module lives in `kiln-train` and not the kernel crate
//!
//! The first kernel-crate candle drop ((#1082)): `kiln-opd-loss-kernel`
//! is now 100% candle-free (pure `kiln_tensor` + `kiln_autograd`). The
//! candle-typed glue that the OPD trainer needs — the pure-candle Phase
//! A reference path, the candle `CustomOp1`-based kt-forward-op shim,
//! and the kt-tape production-caller adapters — moved UP into
//! `kiln-train`, which legitimately keeps `candle-core` (and already
//! depends on `kiln-kt-bridge`) for now. The kernel crate keeps the
//! kt-typed building blocks (`kt_api`, `kt_tape`) that this module
//! calls; this module is the candle↔kt boundary.
//!
//! Nothing in the OPD math changed in the move: the Phase A composite,
//! the shim closures, and the tape adapters are byte-identical in logic
//! to their previous homes (`kiln-opd-loss-kernel/src/lib.rs`,
//! `kt_forward_op.rs`, `tape_forward.rs`). Only the crate location and
//! the `crate::` → `kiln_opd_loss_kernel::` call paths changed.
//!
//! # Layout
//!
//! - **Phase A** ([`opd_top_k_reverse_kl_phase_a_per_position`]) — the
//!   pure-candle reference path. Builds `[T_active, K]` student logits
//!   via per-token gather + batched matmul, runs the renormalised
//!   reverse-KL in candle ops, and lets candle autograd handle the
//!   backward. Used only as the fallback inside the kt-forward-op shim
//!   when the kt envelope (`{K∈16,32} × {F32,BF16} × CUDA`) doesn't
//!   apply.
//! - **kt-forward-op shim** ([`opd_top_k_reverse_kl_per_position_via_kt_forward_op`])
//!   — a single candle `CustomOp1`
//!   ([`kiln_kt_bridge::forward_op::KtForwardOp1`]) wrapping the kt
//!   composite forward + the fused kt CUDA backward
//!   ([`kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_bwd_kt`]).
//!   The production candle-autograd path.
//! - **kt-tape adapters** ([`try_tape_opd_per_position_cuda`],
//!   [`try_tape_opd_scalar_mean_cuda_kt`]) — `KILN_USE_TAPE_FORWARD`-gated
//!   adapters that record the OPD backward onto a thread-local
//!   `kiln_autograd::Tape` via the kernel crate's kt-tape entries
//!   ([`kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`]
//!   / [`..._via_kt_tape`]).

use anyhow::{Context, Result, anyhow};
use candle_core::{D, DType, Device, Tensor};

// =========================================================================
// Phase A — pure-candle reference path
// =========================================================================

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

/// Per-position helper used by [`opd_top_k_reverse_kl_phase_a_per_position`].
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

/// log_softmax along the last dimension. We re-implement it here (rather
/// than pulling in a `candle_nn`-style helper) to keep the autograd graph
/// minimal: this version produces a single subtraction node off the input.
pub(crate) fn log_softmax_last(x: &Tensor) -> Result<Tensor> {
    let lse = x.log_sum_exp(D::Minus1)?.unsqueeze(D::Minus1)?;
    let broadcast = lse.broadcast_as(x.shape())?;
    Ok((x - broadcast)?)
}

// =========================================================================
// kt-forward-op shim — candle `CustomOp1` over the kt fused forward+backward
// =========================================================================
//
// Production-caller migration to `KtForwardOp` for OPD per-position
// reverse-KL ((#1082) — see `docs/CANDLE_REMOVAL_PLAN.md`).
//
// # What this wires together
//
// The OPD trainer (`crates/kiln-train/src/opd.rs`) previously called the
// pure-candle Phase A reference directly. Phase A is a pure-candle
// composite (`index_select` → `matmul` → `log_softmax_last` → `exp` →
// broadcast subtract → multiply → `sum`); it relies on candle's autograd
// graph to back-propagate `mean_kl.backward()` into the LoRA Vars that
// produced `student_hidden`.
//
// [`opd_top_k_reverse_kl_per_position_via_kt_forward_op`] replaces that
// candle composite with a **single** candle `CustomOp1` —
// [`kiln_kt_bridge::forward_op::KtForwardOp1`] (commit `095f1c74`) —
// whose forward closure runs the kt-typed
// [`kiln_opd_loss_kernel::opd_top_k_reverse_kl_per_position_kt`] composite
// end-to-end on CUDA (so the resulting candle autograd graph has one node
// per OPD call instead of ~8) and whose backward closure calls the kt-typed
// CUDA backward
// ([`kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_bwd_kt`]) on the
// fused `kiln_opd_topk_kl_bwd_*` FFI symbols.
//
// # Envelope and fallback
//
// The kt-typed backward is gated to `K ∈ {16, 32}` and
// `dtype ∈ {F32, BF16}` (matching the kernel crate's
// `phase_b::cuda_kernel_supports`). It also requires
// `hidden.dtype() == head_t.dtype()`. When ANY of the following hold we
// fall back to the candle Phase A path so the production caller stays
// correct for the full input envelope it supports today:
//
// - `KILN_DISABLE_OPD_KT_FORWARD_OP=1` (kill switch).
// - `hidden` is not on CUDA (the shim is CUDA-only by construction).
// - `top_k` is not 16 or 32.
// - `hidden.dtype()` is not F32 or BF16.
// - `hidden.dtype() != head_t.dtype()`.
// - `active_count == 0` (no rows to compute).
//
// Falling back preserves the `mean_kl.backward()` autograd chain the
// trainer relies on.
//
// # Numerical contract
//
// Up to floating-point associativity in the matmul and the log-softmax /
// KL reductions, the forward output matches Phase A to ≤ 1e-4 in F32 and
// ≤ 5e-2 in BF16. The backward d_hidden matches the candle reference to
// the same tolerances — it's the same FFI symbols
// (`kiln_opd_topk_kl_bwd_{bf16,f32}`).

/// Read the `KILN_DISABLE_OPD_KT_FORWARD_OP` kill switch. When set
/// (`1` / `true` / `yes` / `TRUE`), the production caller falls back
/// to the candle Phase A per-position path. Same convention as
/// `KILN_DISABLE_OPD_BWD_KT_BRIDGE` (commit `0c1be227`),
/// `KILN_DISABLE_RMSNORM_KERNEL`, `KILN_DISABLE_FUSED_CONV1D`, etc.
pub fn kt_forward_op_disabled() -> bool {
    std::env::var("KILN_DISABLE_OPD_KT_FORWARD_OP")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false)
}

/// Returns `true` when the `(top_k, dtype, head_t.dtype())` triple
/// is in the fused kt-bwd envelope AND `hidden` is on CUDA. The
/// envelope is the intersection of the kernel crate's
/// `phase_b::cuda_kernel_supports` (the K + dtype check) and the
/// dtype-matching constraint from
/// [`kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_bwd_kt`].
fn shim_envelope_ok(hidden: &Tensor, head_t: &Tensor, top_k: usize) -> bool {
    if !matches!(hidden.device(), Device::Cuda(_)) {
        return false;
    }
    if top_k != 16 && top_k != 32 {
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

/// kt-shim per-position OPD reverse-KL with candle-autograd integration.
///
/// Behavioral envelope:
/// - CUDA + `(top_k, dtype) in {16, 32} × {F32, BF16}` + matching
///   `head_t` dtype → routes through [`KtForwardOp1`] over the
///   kt-typed fused forward+backward.
/// - Anything outside the envelope → falls through to
///   [`opd_top_k_reverse_kl_phase_a_per_position`] (the pure-candle
///   reference path). The autograd chain through
///   `mean_kl.backward()` is preserved in either case.
///
/// The signature mirrors [`opd_top_k_reverse_kl_phase_a_per_position`]
/// so the OPD trainer call site is a one-line swap.
///
/// [`KtForwardOp1`]: kiln_kt_bridge::forward_op::KtForwardOp1
pub fn opd_top_k_reverse_kl_per_position_via_kt_forward_op(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
) -> Result<Tensor> {
    // Kill switch + non-CUDA fallback path: identical to the legacy
    // candle Phase A entry the trainer used before this migration.
    if kt_forward_op_disabled() || !shim_envelope_ok(hidden, head_t, top_k) {
        return opd_top_k_reverse_kl_phase_a_per_position(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
        );
    }

    #[cfg(feature = "cuda")]
    {
        return cuda_via_kt_forward_op(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
        );
    }

    // Non-cuda builds: shim_envelope_ok() returned true above only if
    // hidden.device() is CUDA — but candle without the `cuda` feature
    // can't have a CUDA device, so we'll never reach here. Leave a
    // belt-and-suspenders fallback so the function is still
    // well-typed.
    #[cfg(not(feature = "cuda"))]
    {
        opd_top_k_reverse_kl_phase_a_per_position(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
        )
    }
}

// ---------------------------------------------------------------------------
// CUDA fast path: KtForwardOp1 over kt-typed forward + backward.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
fn cuda_via_kt_forward_op(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    device: &Device,
) -> Result<Tensor> {
    use kiln_kt_bridge::forward_op::KtForwardOp1;
    use kiln_kt_bridge::{
        kt_tensor_from_candle_cuda_borrow, kt_tensor_to_candle_cuda_copy,
    };
    use kiln_opd_loss_kernel::{
        OpdLossOutputKt, opd_top_k_reverse_kl_per_position_kt,
        opd_top_k_reverse_kl_phase_b_bwd_kt,
    };
    use std::sync::Arc;

    // ----- Short-circuit on no active rows ---------------------------------
    //
    // Phase A returns `Tensor::zeros((0,), DType::F32, device)` when
    // `active_count == 0`. The kt forward closure would also handle
    // this internally (it returns an empty `[0]` F32 KtTensor via
    // `empty_per_position()`), and the shim wires that into candle
    // storage cleanly. But going through `apply_op1` on a hidden that
    // has no contribution to the loss attaches a dead branch to the
    // autograd graph; the trainer's `mean_kl.backward()` would then
    // try to evaluate that branch, producing an empty grad scatter.
    // Cheap and safer to defer to Phase A's short-circuit.
    let active_count = label_mask.iter().filter(|&&m| m).count();
    if active_count == 0 {
        return opd_top_k_reverse_kl_phase_a_per_position(
            hidden,
            head_t,
            teacher_topk_indices,
            teacher_topk_logprobs,
            label_mask,
            top_k,
            device,
        );
    }

    // ----- Force-contiguous `hidden` --------------------------------------
    //
    // `apply_op1` passes the input layout through to the CustomOp's
    // `cuda_fwd` hook; the kt-bridge borrow path requires contiguous
    // storage.
    let hidden_contig = hidden
        .contiguous()
        .context("force-contiguous hidden for OPD kt-shim")?;

    // ----- Captured shim state (cloned into the closures) -----------------
    //
    // `head_t` is frozen (transposed LM head). Both closures need it.
    // `teacher_topk_indices` / `teacher_topk_logprobs` / `label_mask`
    // are also captured by value so the shim outlives the call frame
    // — `KtForwardOp1` is held by `Arc<Box<dyn CustomOp1>>` inside
    // candle's autograd graph until `mean_kl.backward()` is invoked
    // (the parents drop the op when the graph is collected).
    let head_t_owned = head_t.clone();
    let head_t_owned_bwd = head_t.clone();
    let indices_fwd = teacher_topk_indices.to_vec();
    let indices_bwd = teacher_topk_indices.to_vec();
    let logprobs_fwd = teacher_topk_logprobs.to_vec();
    let logprobs_bwd = teacher_topk_logprobs.to_vec();
    let mask_fwd = label_mask.to_vec();
    let mask_bwd = label_mask.to_vec();

    // ----- Forward closure ------------------------------------------------
    //
    // Computes the per-position reverse-KL via the kt-typed forward
    // [`kiln_opd_loss_kernel::opd_top_k_reverse_kl_per_position_kt`]. The
    // forward borrows the candle CUDA tensors into kt (zero-copy), runs the
    // kt composite (`index_select` along axis 0 of `hidden` AND axis 1
    // of `head_t`, matmul, log_softmax, KL reduction), and copies the
    // result back to a candle CUDA tensor.
    //
    // This swap (over the previous candle composite) is unblocked by
    // the kt-tensor axis-N `IndexSelectOp::cuda_fwd` substrate change
    // earlier in (#1082). With axis-N CUDA gather available, the kt entry
    // runs end-to-end on the GPU and the OPD per-position path is fully
    // kt-typed (forward + backward both go through the fused kt FFI
    // symbols).
    let forward = move |hidden_in: &Tensor| -> candle_core::Result<Tensor> {
        // Force-contiguous both inputs before borrowing into kt; the
        // borrow path requires contiguous storage (same constraint
        // the backward closure enforces).
        let hidden_c = hidden_in.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: contiguous hidden: {e}"
            ))
        })?;
        let head_t_c = head_t_owned.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: contiguous head_t: {e}"
            ))
        })?;

        let hidden_kt = kt_tensor_from_candle_cuda_borrow(&hidden_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: borrow hidden: {e}"
            ))
        })?;
        let head_t_kt = kt_tensor_from_candle_cuda_borrow(&head_t_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: borrow head_t: {e}"
            ))
        })?;

        let per_position_kt = opd_top_k_reverse_kl_per_position_kt(
            &hidden_kt,
            &head_t_kt,
            &indices_fwd,
            &logprobs_fwd,
            &mask_fwd,
            top_k,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: kt per-position call: {e}"
            ))
        })?;

        // The kt entry returns `[T_active]` F32. The trailing
        // `sum_axis(1)` in `per_position_forward_kt` can yield a
        // non-contiguous view in some regimes, so be defensive
        // before the copy-back.
        let per_position_kt_c = if per_position_kt.is_contiguous() {
            per_position_kt
        } else {
            per_position_kt.contiguous().map_err(|e| {
                candle_core::Error::Msg(format!(
                    "opd kt-shim fwd: contiguous per_position: {e}"
                ))
            })?
        };

        let per_position = kt_tensor_to_candle_cuda_copy(&per_position_kt_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: copy-back per_position: {e}"
            ))
        })?;

        per_position.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim fwd: contiguous per-position: {e}"
            ))
        })
    };

    // ----- Backward closure ----------------------------------------------
    //
    // For PerPosition output mode, the kt bwd takes `grad_loss`
    // shape `[T_active]` (F32 contiguous on CUDA) and returns
    // `d_hidden` shape `[1, T, H]` in the input dtype. The kt entry
    // handles the active-row gather of `hidden`, the teacher tensor
    // uploads, the fused FFI dispatch, and the `scatter_add` from
    // `[T_active, H]` back into `[1, T, H]`.
    //
    // The shim passes us (`arg=hidden, res=per_position_kl,
    // grad_res=grad_per_position_kl`); we ignore `res` since the
    // backward doesn't depend on the forward output value.
    let backward = move |arg: &Tensor,
                         _res: &Tensor,
                         grad_res: &Tensor|
          -> candle_core::Result<Option<Tensor>> {
        let hidden_c = arg.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: contiguous hidden: {e}"
            ))
        })?;
        let head_t_c = head_t_owned_bwd.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: contiguous head_t: {e}"
            ))
        })?;
        // `grad_res` comes in as F32 already (the per-position KL is
        // F32 from `opd_top_k_reverse_kl_per_position_kt`'s output
        // dtype), but cast defensively in case an upstream candle op
        // re-typed the gradient (e.g. mixed-precision wrappers).
        let grad_res_f32 = grad_res.to_dtype(DType::F32).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: cast grad to F32: {e}"
            ))
        })?;
        let grad_res_c = grad_res_f32.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: contiguous grad: {e}"
            ))
        })?;

        let hidden_kt = kt_tensor_from_candle_cuda_borrow(&hidden_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: borrow hidden: {e}"
            ))
        })?;
        let head_t_kt = kt_tensor_from_candle_cuda_borrow(&head_t_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: borrow head_t: {e}"
            ))
        })?;
        let grad_res_kt = kt_tensor_from_candle_cuda_borrow(&grad_res_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: borrow grad: {e}"
            ))
        })?;

        let d_hidden_kt = opd_top_k_reverse_kl_phase_b_bwd_kt(
            &hidden_kt,
            &head_t_kt,
            &indices_bwd,
            &logprobs_bwd,
            &mask_bwd,
            &grad_res_kt,
            top_k,
            OpdLossOutputKt::PerPosition,
        )
        .map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: kt bwd call: {e}"
            ))
        })?;

        // `scatter_add` builds the `[1, T, H]` output contiguous from
        // a freshly zero-allocated buffer, but be defensive — the
        // bridge copy-back requires contiguous storage.
        let d_hidden_kt_c = if d_hidden_kt.is_contiguous() {
            d_hidden_kt
        } else {
            d_hidden_kt.contiguous().map_err(|e| {
                candle_core::Error::Msg(format!(
                    "opd kt-shim bwd: contiguous d_hidden: {e}"
                ))
            })?
        };

        let d_hidden = kt_tensor_to_candle_cuda_copy(&d_hidden_kt_c).map_err(|e| {
            candle_core::Error::Msg(format!(
                "opd kt-shim bwd: copy-back d_hidden: {e}"
            ))
        })?;

        Ok(Some(d_hidden))
    };

    // ----- Apply ----------------------------------------------------------
    let op = KtForwardOp1::new("kiln-opd-loss-kt-forward-op", forward, backward);
    let _ = device; // unused; the device is implicit in `hidden_contig`'s storage.
    hidden_contig
        .apply_op1_arc(Arc::new(Box::new(op)))
        .context("apply OPD kt-forward-op to hidden")
}

// =========================================================================
// kt-tape production-caller adapters
// =========================================================================
//
// Wave-13 (#1082) — OPD kt-tape production-caller adapter.
//
// Mirrors the rmsnorm sibling
// `kiln-model::tape_forward::try_tape_rms_norm_cuda`. The adapter takes
// the same candle-typed inputs the production caller in
// `kiln-train::opd::opd_step_loss` already passes to
// [`opd_top_k_reverse_kl_per_position_via_kt_forward_op`], checks the
// `KILN_USE_TAPE_FORWARD` env tristate + an active thread-local
// [`kiln_autograd::Tape`] scope, and on both gates open routes the forward
// through the kernel crate's kt-tape entry
// ([`kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`])
// (recording the backward node onto the active tape) and copies the kt
// output back into a candle CUDA tensor.
//
// # Production-safety
//
// Off by default. When `KILN_USE_TAPE_FORWARD` is unset or no thread-local
// `Tape` scope is active, [`try_tape_opd_per_position_cuda`] returns
// `Ok(None)` and the caller falls through to the existing
// [`opd_top_k_reverse_kl_per_position_via_kt_forward_op`] shim. The shim's
// autograd chain through candle's `loss.backward()` is preserved exactly as
// before.

/// Borrow a candle input as a kt tensor for the tape path, REUSING an
/// upstream adapter's kt output when this candle tensor was produced by one
/// in the active bridge scope (so the recorded kt `Tape` stays connected —
/// the consumer's input id becomes the producer's output id). Falls back to
/// a fresh zero-copy borrow otherwise (and outside any bridge scope, which is
/// the common case + every per-op parity test).
///
/// Mirrors `kiln_model::tape_forward::tape_kt_input` exactly. (#1082 CP-4
/// endgame, Step A.)
#[cfg(feature = "cuda")]
fn tape_kt_input(x: &Tensor) -> Option<kiln_tensor::Tensor> {
    if let Some(t) = kiln_kt_bridge::tape_bridge::kt_input_for_candle(x.id()) {
        return Some(t);
    }
    kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(x).ok()
}

/// Attempt to run OPD per-position top-K reverse-KL through the
/// kt-tape pilot
/// ([`kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`])
/// instead of the candle-typed
/// [`opd_top_k_reverse_kl_per_position_via_kt_forward_op`] shim.
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
/// [`kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`]:
/// CUDA + matching F32/BF16 `(hidden, head_t)` dtype + `top_k ∈ {16, 32}`.
/// Out-of-envelope inputs return `Ok(None)` rather than `Err` so the
/// adapter is callable from a forward path that wants to short-circuit
/// only on the happy path and fall through to the existing kt-shim on
/// everything else (matching the rmsnorm adapter's contract exactly).
#[cfg(feature = "cuda")]
pub fn try_tape_opd_per_position_cuda(
    hidden: &Tensor,
    head_t: &Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<Option<Tensor>> {
    use kiln_autograd::{tape_forward_enabled, with_active_tape, Tape};
    use kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape;

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

// (#1082 P-OPD) The candle-input scalar-mean adapter
// `try_tape_opd_scalar_mean_cuda` was removed: it had zero call sites.
// `opd_step_forward_backward_tape_authoritative` records the scalar OPD
// loss DIRECTLY against the kt `normed`/`head_t` via the kt-native
// [`try_tape_opd_scalar_mean_cuda_kt`] below (no candle copies /
// `normed_candle` retain dance), so the candle-typed sibling was dead.

/// **kt-native** OPD top-K reverse-KL **reduced to a scalar mean**, taking the
/// kt `hidden` (final-RMSNorm tape node output) and kt `head_t` (frozen lm_head)
/// DIRECTLY — no candle `hidden`/`head_t` copies at the call boundary.
///
/// This is the OPD analogue of
/// `kiln_model::tape_forward::try_tape_cross_entropy_from_logits_kt` (the H6
/// kt-native CE-from-logits loss root): the differentiable input is the
/// CONNECTED kt `hidden` (an already-recorded tape node output — the final
/// RMSNorm), so recording the OPD scalar loss against it roots
/// `dL/d(hidden)` straight on the model tape with NO candle id-mapping dance.
/// Only the SCALAR loss crosses back to candle (≈4 bytes) so the
/// tape-authoritative scope (`with_tape_authoritative_scope`) can resolve
/// `loss.id()` → `loss_kt` and seed `dL/dL = 1`.
///
/// Replaces the candle-shim caller's `normed`→`normed_candle` retain dance and
/// the per-run `head_t`→`head_t_candle` copy in
/// `opd_step_forward_backward_tape_authoritative`: that path bridged the kt
/// `normed`/`head_t` to candle ONLY because the (now-removed) candle-input
/// `try_tape_opd_scalar_mean_cuda` adapter took candle inputs. With kt
/// inputs, the bridge is gone.
///
/// Returns:
/// * `Ok(Some(out))` — the scalar tape path ran. The returned candle scalar
///   `Tensor` is a value-identical copy of the kt-tape loss (no candle autograd
///   lineage — the gradient lives on the tape); the backward node was recorded
///   on the active thread-local tape and the output IO mapping + retained
///   output were registered for the bridge.
/// * `Ok(None)` — `KILN_USE_TAPE_FORWARD` was off, no thread-local tape scope is
///   active, the active set was empty, or the kt envelope rejected the inputs.
///   The caller surfaces this as a clean error (the dispatch should not have
///   selected this path off the envelope).
/// * `Err(...)` — a kt-tape forward error (envelope OK but the FFI call failed).
///
/// # Envelope
///
/// Same as [`try_tape_opd_per_position_cuda`]: CUDA + matching F32/BF16
/// `(hidden, head_t)` dtype + `top_k ∈ {16, 32}`.
#[cfg(feature = "cuda")]
pub fn try_tape_opd_scalar_mean_cuda_kt(
    hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<Option<Tensor>> {
    use kiln_autograd::{tape_forward_enabled, with_active_tape, Tape};
    use kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_via_kt_tape;

    if !tape_forward_enabled() {
        return Ok(None);
    }

    // Active-count short-circuit — match the candle-typed adapters. An empty
    // active set has no loss contribution and recording a tape node for a
    // no-op forward is a footgun.
    let active_count = label_mask.iter().filter(|&&m| m).count();
    if active_count == 0 {
        return Ok(None);
    }

    // Record the SCALAR-mean OPD loss onto the active tape. The kt `hidden` is
    // the final-RMSNorm tape node output (passed straight through by
    // `opd_step_forward_backward_tape_authoritative`), so the recorded node's
    // `hidden` input id is ALREADY a tape node id — the tape stays connected
    // back through the LoRA chain WITHOUT any candle id-mapping (mirrors the H6
    // CE-from-logits-kt path, which records against the connected kt logits).
    // If no scope is open, fall through.
    let loss_kt = match with_active_tape(|tape: &mut Tape| {
        opd_top_k_reverse_kl_phase_b_via_kt_tape(
            hidden,
            head_t,
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
        .map_err(|e: kiln_tensor::Error| anyhow::anyhow!("opd_top_k scalar kt-tape (kt): {e}"))
        .context("try_tape_opd_scalar_mean_cuda_kt: kt-tape forward failed")?;

    // Value-identical candle copy of the scalar loss for the caller (loss_val +
    // metrics). The returned candle tensor carries NO candle autograd lineage —
    // the gradient lives on the tape.
    let out = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&loss_kt)
        .context("try_tape_opd_scalar_mean_cuda_kt: kt -> candle copy failed")?;

    // CP-4 (#1082) tape_bridge: map / retain the loss output keyed on the
    // RETURNED copy's id so `with_tape_authoritative_scope` resolves
    // `loss.id()` → `loss_kt` to seed the tape root. The differentiable input
    // (`hidden`) is itself a tape node output (the final RMSNorm), so it needs
    // no `register_input_mapping` here — `head_t` is non-differentiable in this
    // op (the kernel emits `d_hidden` only). No-ops cleanly outside a scope.
    kiln_kt_bridge::tape_bridge::register_output_mapping(loss_kt.id(), out.id());
    kiln_kt_bridge::tape_bridge::retain_output_for_chaining(&loss_kt, out.id());

    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// Unit tests (cfg-test only). End-to-end CUDA parity for the kt-forward-op
// shim lives in `kiln-train/tests/vk_cuda_opd_parity.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize tests that mutate `KILN_DISABLE_OPD_KT_FORWARD_OP` so
    // they don't race against each other. Cargo runs tests in threads
    // within a single process; without a lock, `kill_switch_on` can
    // set the env to "1" in the middle of `kill_switch_default_off`'s
    // sequence of `set_var` + `assert!(!kt_forward_op_disabled())`
    // calls, causing a flaky failure (observed on macOS CI run
    // 26572612427, commit 449fb6b1, #1082). The mutex is held for the
    // entire test body — including the prior-value capture and the
    // restore — so the env var is logically owned by exactly one
    // test at a time.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|poisoned| {
            // A previous test panicked while holding the lock. Clear
            // the poison and proceed — each test below re-establishes
            // its own starting env state before asserting.
            ENV_LOCK.clear_poison();
            poisoned.into_inner()
        })
    }

    #[test]
    fn kill_switch_default_off() {
        let _guard = env_lock();
        // Sanity: the default env should not trigger the kill switch.
        // (The harness may have set KILN_DISABLE_OPD_KT_FORWARD_OP=0,
        // which also reads as off.)
        // SAFETY: we don't set the env here — just check that an
        // unset (or "0") env yields false.
        let prior = std::env::var("KILN_DISABLE_OPD_KT_FORWARD_OP").ok();
        // SAFETY: env modification is intra-test and serialized via
        // `ENV_LOCK` against the other test in this module that
        // mutates the same var. The lock is held for the entire test
        // body (including the restore), so no concurrent reader in
        // this binary observes a partially-updated env.
        unsafe {
            std::env::remove_var("KILN_DISABLE_OPD_KT_FORWARD_OP");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", "0");
        }
        assert!(!kt_forward_op_disabled());
        unsafe {
            std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", "false");
        }
        assert!(!kt_forward_op_disabled());

        // Restore the prior value.
        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_OPD_KT_FORWARD_OP"),
            }
        }
    }

    #[test]
    fn kill_switch_on() {
        let _guard = env_lock();
        let prior = std::env::var("KILN_DISABLE_OPD_KT_FORWARD_OP").ok();
        for v in ["1", "true", "yes", "TRUE", "Yes"] {
            unsafe {
                std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", v);
            }
            assert!(
                kt_forward_op_disabled(),
                "expected disabled for env={v}"
            );
        }
        unsafe {
            match prior {
                Some(v) => std::env::set_var("KILN_DISABLE_OPD_KT_FORWARD_OP", v),
                None => std::env::remove_var("KILN_DISABLE_OPD_KT_FORWARD_OP"),
            }
        }
    }
}

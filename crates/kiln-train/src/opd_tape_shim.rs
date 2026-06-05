//! kt-native OPD top-K reverse-KL scalar-loss tape boundary for the
//! `kiln-train` OPD trainer ((#1082) — relocated out of `kiln-opd-loss-kernel`).
//! (Formerly `opd_candle_shim`; fully kt-native — candle was removed
//! workspace-wide in the #1082 finale. The vestigial `_cuda` suffixes on the
//! producer fn names are historical, not a candle/CUDA dependency.)
//!
//! # Why this module lives in `kiln-train` and not the kernel crate
//!
//! `kiln-opd-loss-kernel` is candle-free (pure `kiln_tensor` + `kiln_autograd`)
//! and holds the kt-typed building blocks (`kt_api`, `kt_tape`). The trainer-side
//! glue — the Phase A reference path and the kt-tape production-caller adapters
//! that root the OPD reverse-KL scalar on the `kiln_autograd::Tape` — lives here
//! next to the rest of the trainer. This module is the OPD tape-loss boundary:
//! it produces the single scalar node whose `Tape::backward` drives `dL/d(logits)`
//! back through the model chain into every LoRA parameter.
//!
//! The OPD math is byte-identical to its previous home; only the crate location
//! and the `crate::` → `kiln_opd_loss_kernel::` call paths changed.
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

use anyhow::{Context, Result};

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
/// Same as [`try_tape_opd_per_position_cuda`]: CUDA or Metal + matching
/// F32/BF16 `(hidden, head_t)` dtype + `top_k ∈ {16, 32}`. (#1082) Both the
/// FORWARD + loss and the recorded backward
/// (`CudaOpdTopKReverseKlPhaseBBackward::apply`) run on either device: CUDA
/// uses the fused FFI kernel, CPU/Metal the device-agnostic analytic
/// kt-composite backward.
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan", feature = "rocm"))]
pub fn try_tape_opd_scalar_mean_cuda_kt(
    hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    use kiln_autograd::{tape_forward_enabled, with_active_tape, Tape};
    use kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_unit_grad_via_kt_tape;

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

    // (#1443 step 3/5) Mixed-precision OPD on Vulkan: the model activations are
    // F32 (post embed BF16→F32 cast) so `hidden` is F32, but the tied lm_head
    // `head_t` is BF16 on a BF16 base. The OPD top-K reverse-KL kernel's envelope
    // requires `hidden.dtype() == head_t.dtype()` (it runs an internal
    // `hidden @ head_t` and the fused/composite backward assumes one dtype). The
    // kernel has no bf16w-style mixed path, so cast the FROZEN `head_t` to F32 for
    // this OPD-loss compute. The RESIDENT `embed_tokens_t` stays BF16 (the #1443
    // VRAM win for SFT/GRPO via the bf16w lm_head); this is only a per-step
    // transient F32 head used by the OPD loss — which already reads the full head
    // for its matmul. Vulkan-only + F32-hidden/BF16-head only; CUDA/Metal (BF16
    // hidden == BF16 head) and the F32-base Vulkan path are untouched.
    #[cfg(feature = "vulkan")]
    let head_owned;
    #[cfg(feature = "vulkan")]
    let head_t: &kiln_tensor::Tensor = if matches!(
        hidden.device(),
        kiln_tensor::Device::Vulkan(_)
    ) && hidden.dtype() == kiln_tensor::DType::F32
        && head_t.dtype() == kiln_tensor::DType::BF16
    {
        head_owned = head_t
            .to_dtype(kiln_tensor::DType::F32)
            .context("opd shim: cast BF16 head_t -> F32 for the OPD loss (Vulkan mixed precision)")?;
        &head_owned
    } else {
        head_t
    };

    // Record the SCALAR-mean OPD loss onto the active tape. The kt `hidden` is
    // the final-RMSNorm tape node output (passed straight through by
    // `opd_step_forward_backward_tape_authoritative`), so the recorded node's
    // `hidden` input id is ALREADY a tape node id — the tape stays connected
    // back through the LoRA chain WITHOUT any candle id-mapping (mirrors the H6
    // CE-from-logits-kt path, which records against the connected kt logits).
    // If no scope is open, fall through.
    let loss_kt = match with_active_tape(|tape: &mut Tape| {
        opd_top_k_reverse_kl_phase_b_unit_grad_via_kt_tape(
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

    // (#1082 keystone) Return the kt scalar loss DIRECTLY. The caller seeds it as
    // the tape root via `with_tape_authoritative_scope_kt` (ones_like at
    // `loss_kt.id()`) — no kt->candle copy, no `register_output_mapping`. The
    // differentiable input (`hidden`) is already a recorded tape node.
    Ok(Some(loss_kt))
}

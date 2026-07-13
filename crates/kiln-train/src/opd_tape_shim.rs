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
//!   [`try_tape_opd_scalar_mean_cuda_kt`]) — active-scope adapters that record
//!   the OPD backward onto a thread-local
//!   `kiln_autograd::Tape` via the kernel crate's kt-tape entries
//!   ([`kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_per_position_via_kt_tape`]
//!   / [`..._via_kt_tape`]).

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
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
/// * `Ok(None)` — no thread-local tape scope is active, the active set was
///   empty, or the kt envelope rejected the inputs.
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
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub fn try_tape_opd_scalar_mean_cuda_kt(
    hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    use kiln_autograd::{Tape, tape_scope_active, with_active_tape};
    use kiln_opd_loss_kernel::opd_top_k_reverse_kl_phase_b_unit_grad_via_kt_tape;

    if !tape_scope_active() {
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
    let head_t: &kiln_tensor::Tensor = if matches!(hidden.device(), kiln_tensor::Device::Vulkan(_))
        && hidden.dtype() == kiln_tensor::DType::F32
        && head_t.dtype() == kiln_tensor::DType::BF16
    {
        head_owned = head_t.to_dtype(kiln_tensor::DType::F32).context(
            "opd shim: cast BF16 head_t -> F32 for the OPD loss (Vulkan mixed precision)",
        )?;
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

#[cfg(feature = "vulkan")]
fn tensor_err(msg: impl Into<String>) -> kiln_tensor::Error {
    kiln_tensor::Error::Msg(msg.into())
}

#[cfg(feature = "vulkan")]
fn opd_active_positions(label_mask: &[bool]) -> kiln_tensor::Result<Vec<u32>> {
    label_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, &active)| active.then_some(idx))
        .map(|idx| {
            u32::try_from(idx)
                .map_err(|_| tensor_err(format!("vulkan OPD: active position {idx} exceeds u32")))
        })
        .collect()
}

#[cfg(feature = "vulkan")]
fn vulkan_device_index(t: &kiln_tensor::Tensor, context: &str) -> kiln_tensor::Result<usize> {
    match t.device() {
        kiln_tensor::Device::Vulkan(i) => Ok(i),
        other => Err(tensor_err(format!(
            "{context}: expected Vulkan tensor, got {other}"
        ))),
    }
}

#[cfg(feature = "vulkan")]
fn prepare_vulkan_opd_active_hidden(
    hidden: &kiln_tensor::Tensor,
    label_mask: &[bool],
) -> kiln_tensor::Result<(
    kiln_vulkan_kernel::vk_tensor::VkTensor,
    Vec<u32>,
    usize,
    usize,
    usize,
)> {
    let dims = hidden.dims();
    if dims.len() != 3 || dims[0] != 1 {
        return Err(tensor_err(format!(
            "vulkan OPD: hidden must be [1, seq_len, hidden], got {dims:?}"
        )));
    }
    let seq_len = dims[1];
    let hidden_size = dims[2];
    if label_mask.len() != seq_len {
        return Err(tensor_err(format!(
            "vulkan OPD: label_mask len {} != seq_len {seq_len}",
            label_mask.len()
        )));
    }
    if hidden.dtype() != kiln_tensor::DType::F32 {
        return Err(tensor_err(format!(
            "vulkan OPD: hidden must be F32 for vk_opd_topk_kl, got {}",
            hidden.dtype()
        )));
    }
    let active_positions = opd_active_positions(label_mask)?;
    if active_positions.is_empty() {
        return Err(tensor_err("vulkan OPD: no active positions"));
    }

    let hidden_2d = hidden.squeeze(0).and_then(|t| {
        if t.is_contiguous() {
            Ok(t)
        } else {
            t.contiguous()
        }
    })?;
    let hidden_vk = kiln_tensor::vk_tensor_from_kt(&hidden_2d)
        .map_err(|e| tensor_err(format!("vulkan OPD: bridge hidden: {e}")))?;
    let active_hidden_vk = kiln_vulkan_kernel::vk_ops::index_select::vk_index_select_rows(
        &hidden_vk,
        &active_positions,
    )
    .map_err(|e| tensor_err(format!("vulkan OPD: gather active rows: {e}")))?;
    Ok((
        active_hidden_vk,
        active_positions,
        seq_len,
        hidden_size,
        vulkan_device_index(hidden, "vulkan OPD hidden")?,
    ))
}

#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_opd_top_k_reverse_kl_scalar_loss_kt(
    hidden: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    let (active_hidden_vk, _active_positions, _seq_len, hidden_size, device_index) =
        prepare_vulkan_opd_active_hidden(hidden, label_mask)?;
    if weight.dims().len() != 2 || weight.dims()[1] != hidden_size {
        return Err(tensor_err(format!(
            "vulkan OPD: weight must be [vocab, hidden={hidden_size}], got {:?}",
            weight.dims()
        )));
    }
    if !matches!(
        weight.dtype(),
        kiln_tensor::DType::F32 | kiln_tensor::DType::BF16
    ) {
        return Err(tensor_err(format!(
            "vulkan OPD: weight must be F32/BF16, got {}",
            weight.dtype()
        )));
    }
    let weight_vk = kiln_tensor::vk_tensor_from_kt(weight)
        .map_err(|e| tensor_err(format!("vulkan OPD: bridge weight: {e}")))?;
    let loss_vk = kiln_vulkan_kernel::vk_ops::opd::vk_opd_top_k_reverse_kl_loss(
        &active_hidden_vk,
        &weight_vk,
        teacher_topk_indices,
        teacher_topk_logprobs,
        top_k,
    )
    .map_err(|e| tensor_err(format!("vulkan OPD fused loss: {e}")))?;
    kiln_tensor::kt_tensor_from_vk(&loss_vk, device_index)
        .map_err(|e| tensor_err(format!("vulkan OPD: bridge loss: {e}")))
}

#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_opd_top_k_reverse_kl_scalar_grad_kt(
    hidden: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
    grad_loss: &kiln_tensor::Tensor,
) -> kiln_tensor::Result<kiln_tensor::Tensor> {
    let (active_hidden_vk, active_positions, seq_len, hidden_size, device_index) =
        prepare_vulkan_opd_active_hidden(hidden, label_mask)?;
    if weight.dims().len() != 2 || weight.dims()[1] != hidden_size {
        return Err(tensor_err(format!(
            "vulkan OPD grad: weight must be [vocab, hidden={hidden_size}], got {:?}",
            weight.dims()
        )));
    }
    let weight_vk = kiln_tensor::vk_tensor_from_kt(weight)
        .map_err(|e| tensor_err(format!("vulkan OPD grad: bridge weight: {e}")))?;
    let grad_vk = kiln_tensor::vk_tensor_from_kt(grad_loss)
        .map_err(|e| tensor_err(format!("vulkan OPD grad: bridge grad_loss: {e}")))?;
    let grad_active_vk = kiln_vulkan_kernel::vk_ops::opd::vk_opd_top_k_reverse_kl_backward(
        &active_hidden_vk,
        &weight_vk,
        teacher_topk_indices,
        teacher_topk_logprobs,
        &grad_vk,
        top_k,
        kiln_vulkan_kernel::vk_ops::opd::OpdLossOutputMode::ScalarMean,
    )
    .map_err(|e| tensor_err(format!("vulkan OPD fused backward: {e}")))?;
    let grad_full_vk = kiln_vulkan_kernel::vk_ops::index_select::vk_scatter_rows_to_full(
        &grad_active_vk,
        &active_positions,
        seq_len,
    )
    .map_err(|e| tensor_err(format!("vulkan OPD grad: scatter active rows: {e}")))?;
    let grad_full = kiln_tensor::kt_tensor_from_vk(&grad_full_vk, device_index)
        .map_err(|e| tensor_err(format!("vulkan OPD grad: bridge full grad: {e}")))?;
    grad_full
        .unsqueeze(0)
        .map_err(|e| tensor_err(format!("vulkan OPD grad: unsqueeze: {e}")))
}

#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_opd_top_k_reverse_kl_scalar_loss_and_grad_kt(
    hidden: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> kiln_tensor::Result<(kiln_tensor::Tensor, kiln_tensor::Tensor)> {
    let loss = vulkan_opd_top_k_reverse_kl_scalar_loss_kt(
        hidden,
        weight,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )?;
    let grad_seed = kiln_tensor::Tensor::from_vec_on(hidden.device(), vec![1.0f32], vec![1])
        .map_err(|e| tensor_err(format!("vulkan OPD: grad seed: {e}")))?;
    let grad = vulkan_opd_top_k_reverse_kl_scalar_grad_kt(
        hidden,
        weight,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
        &grad_seed,
    )?;
    Ok((loss, grad))
}

#[cfg(feature = "vulkan")]
#[derive(Debug)]
struct VulkanOpdTopKReverseKlBackward {
    hidden: kiln_tensor::Tensor,
    weight: kiln_tensor::Tensor,
    teacher_topk_indices: Vec<u32>,
    teacher_topk_logprobs: Vec<f32>,
    label_mask: Vec<bool>,
    top_k: usize,
}

#[cfg(feature = "vulkan")]
impl kiln_autograd::BackwardOp for VulkanOpdTopKReverseKlBackward {
    fn name(&self) -> &'static str {
        "vulkan_opd_topk_kl_backward"
    }

    fn input_count(&self) -> usize {
        1
    }

    fn apply(
        &self,
        grad_output: &kiln_tensor::Tensor,
    ) -> kiln_tensor::Result<Vec<Option<kiln_tensor::Tensor>>> {
        let grad_hidden = vulkan_opd_top_k_reverse_kl_scalar_grad_kt(
            &self.hidden,
            &self.weight,
            &self.teacher_topk_indices,
            &self.teacher_topk_logprobs,
            &self.label_mask,
            self.top_k,
            grad_output,
        )?;
        Ok(vec![Some(grad_hidden)])
    }

    fn requires_input(&self, _idx: usize) -> bool {
        false
    }
}

/// Vulkan-native OPD scalar loss wrapper using `vk_opd_topk_kl_{fwd,bwd}`.
///
/// Unlike [`try_tape_opd_scalar_mean_cuda_kt`], this takes the canonical tied
/// LM-head weight `[vocab, hidden]` (`weights.embed_tokens`) because the Vulkan
/// fused OPD shader supports F32 activations with an F32/BF16 row-major weight.
#[cfg(feature = "vulkan")]
pub fn try_tape_opd_scalar_mean_vulkan_kt(
    hidden: &kiln_tensor::Tensor,
    weight: &kiln_tensor::Tensor,
    teacher_topk_indices: &[u32],
    teacher_topk_logprobs: &[f32],
    label_mask: &[bool],
    top_k: usize,
) -> Result<Option<kiln_tensor::Tensor>> {
    use kiln_autograd::{Tape, tape_scope_active, with_active_tape};

    if !tape_scope_active() {
        return Ok(None);
    }
    if !matches!(hidden.device(), kiln_tensor::Device::Vulkan(_))
        || !matches!(weight.device(), kiln_tensor::Device::Vulkan(_))
    {
        return Ok(None);
    }
    if label_mask.iter().filter(|&&m| m).count() == 0 {
        return Ok(None);
    }
    let loss_kt = vulkan_opd_top_k_reverse_kl_scalar_loss_kt(
        hidden,
        weight,
        teacher_topk_indices,
        teacher_topk_logprobs,
        label_mask,
        top_k,
    )
    .map_err(|e| anyhow::anyhow!("vulkan OPD scalar kt loss: {e}"))?;

    let loss_kt = match with_active_tape(|tape: &mut Tape| -> Result<_> {
        tape.record(
            &loss_kt,
            &[hidden],
            Box::new(VulkanOpdTopKReverseKlBackward {
                hidden: hidden.clone(),
                weight: weight.clone(),
                teacher_topk_indices: teacher_topk_indices.to_vec(),
                teacher_topk_logprobs: teacher_topk_logprobs.to_vec(),
                label_mask: label_mask.to_vec(),
                top_k,
            }),
        );
        Ok(loss_kt)
    }) {
        Some(result) => result?,
        None => return Ok(None),
    };

    Ok(Some(loss_kt))
}

/// ECHO env-CE composition for the OPD tape step (the OPD half of the
/// resurrection plan): wraps the recorded OPD scalar in a two-input node
/// whose value is `opd_loss + λ·env_CE` and whose backward (a) passes the
/// seed straight through to the OPD loss node and (b) emits the
/// constant-coefficient env-row gradient onto the recorded hidden — the
/// same closed form the GRPO fused root uses (`(λ/|O|)·(softmax − onehot)
/// @ head_tᵀ` at env rows, zero elsewhere).
///
/// Returns `None` (compose nothing) only when the spec selects no env rows or
/// λ = 0, so the OPD loss trains exactly as before. If the term has a
/// contribution but no active tape is available, this fails closed instead of
/// returning an unrecorded composed value or silently dropping ECHO.
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[allow(clippy::too_many_arguments)]
pub fn try_tape_opd_echo_env_compose_kt(
    opd_loss_kt: &kiln_tensor::Tensor,
    hidden: &kiln_tensor::Tensor,
    head_t: &kiln_tensor::Tensor,
    input_ids: &[u32],
    spec: &crate::grpo_tape_shim::EchoEnvSpec,
    chunk_size: usize,
    device: &kiln_tensor::Device,
) -> Result<Option<(kiln_tensor::Tensor, f64)>> {
    use kiln_autograd::{Tape, with_active_tape};

    let Some((node_state, env_ce_kt, env_ce_val)) =
        crate::grpo_tape_shim::echo_env_state_and_value_kt(
            hidden, head_t, input_ids, spec, chunk_size, device,
        )
        .context("opd echo compose: env state")?
    else {
        return Ok(None);
    };
    let weighted = env_ce_kt
        .affine(spec.lambda, 0.0)
        .map_err(|e| anyhow::anyhow!("opd echo compose: λ scale: {e}"))?;
    let composed =
        (opd_loss_kt + &weighted).map_err(|e| anyhow::anyhow!("opd echo compose: add: {e}"))?;

    let recorded = with_active_tape(|tape: &mut Tape| {
        tape.record(
            &composed,
            &[opd_loss_kt, hidden],
            Box::new(OpdEchoEnvComposeBackward {
                hidden: hidden.clone(),
                head_t: head_t.clone(),
                node_state,
                device: *device,
                chunk_size,
            }),
        );
        composed.clone()
    })
    .context("opd echo compose: a nonzero ECHO contribution requires an active tape scope")?;
    Ok(Some((recorded, env_ce_val)))
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
#[derive(Debug)]
struct OpdEchoEnvComposeBackward {
    hidden: kiln_tensor::Tensor,
    head_t: kiln_tensor::Tensor,
    node_state: crate::grpo_tape_shim::EchoEnvNodeState,
    device: kiln_tensor::Device,
    chunk_size: usize,
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
impl kiln_autograd::BackwardOp for OpdEchoEnvComposeBackward {
    fn name(&self) -> &'static str {
        "opd_echo_env_compose_backward"
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
        let grad_scalar = crate::grpo_tape_shim::grpo_scalar_seed_or_unit_kt(
            grad_output,
            "OpdEchoEnvComposeBackward",
            false,
        )?;
        // d(composed)/d(opd_loss) = 1 — the seed passes through unchanged
        // so the OPD node's own backward runs exactly as it would have.
        let passthrough = grad_output.clone();
        // The env-row hidden gradient. The PG-param arguments are never
        // read under coeff_override (the env coefficient is constant);
        // GrpoLossParams::reinforce-shaped defaults keep the call total.
        let env_grad = crate::grpo_tape_shim::echo_env_grad_from_normed_hidden_kt(
            &self.hidden,
            &self.head_t,
            &self.node_state,
            crate::trainer::GrpoLossParams {
                advantage: 0.0,
                clip_low: 1.0,
                clip_high: 1.0,
                kl_coeff: 0.0,
                kl_estimator: crate::KlEstimator::K1,
                loss_normalizer: 1.0,
                is_level: crate::IsLevel::Token,
                reinforce: true,
                entropy_aware_kl_quantile: None,
            },
            kiln_model::backend::GrpoKlAuxiliaryRoute::HostComposite,
            grad_scalar,
            &self.device,
            self.chunk_size,
        )
        .map_err(|e| kiln_tensor::Error::Msg(format!("opd echo compose backward: {e:#}")))?;
        Ok(vec![Some(passthrough), Some(env_grad)])
    }
}

/// Test-only constructor for the compose node (fields are private; the
/// closed-form test lives in grpo_tape_shim where the echo fixtures are).
#[cfg(test)]
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
pub(crate) mod test_support {
    pub(crate) fn opd_echo_compose_node(
        hidden: kiln_tensor::Tensor,
        head_t: kiln_tensor::Tensor,
        node_state: crate::grpo_tape_shim::EchoEnvNodeState,
        device: kiln_tensor::Device,
        chunk_size: usize,
    ) -> impl kiln_autograd::BackwardOp {
        super::OpdEchoEnvComposeBackward {
            hidden,
            head_t,
            node_state,
            device,
            chunk_size,
        }
    }
}

#[cfg(test)]
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm"
))]
mod recorder_contract_tests {
    use super::*;

    fn fixture() -> (
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        kiln_tensor::Tensor,
        Vec<u32>,
        crate::grpo_tape_shim::EchoEnvSpec,
        kiln_tensor::Device,
    ) {
        let device = kiln_tensor::Device::Cpu;
        let loss = kiln_tensor::Tensor::from_slice(&[1.0f32], Vec::<usize>::new())
            .expect("scalar OPD loss");
        let hidden =
            kiln_tensor::Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], vec![1, 3, 2])
                .expect("hidden fixture");
        let head_t = kiln_tensor::Tensor::from_slice(
            &[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            vec![2, 5],
        )
        .expect("head fixture");
        let input_ids = vec![0, 1, 2];
        let spec = crate::grpo_tape_shim::EchoEnvSpec {
            env_mask: vec![false, true, false],
            total_obs_len: 1,
            lambda: 0.1,
        };
        (loss, hidden, head_t, input_ids, spec, device)
    }

    #[test]
    fn nonzero_echo_contribution_requires_an_active_tape() {
        let (loss, hidden, head_t, input_ids, spec, device) = fixture();
        let error = try_tape_opd_echo_env_compose_kt(
            &loss, &hidden, &head_t, &input_ids, &spec, 2, &device,
        )
        .expect_err("a requested ECHO contribution must not disappear outside a tape scope");
        assert!(error.to_string().contains("requires an active tape scope"));
    }

    #[test]
    fn zero_echo_contribution_remains_a_valid_noop_without_a_tape() -> Result<()> {
        let (loss, hidden, head_t, input_ids, mut spec, device) = fixture();
        spec.lambda = 0.0;
        assert!(
            try_tape_opd_echo_env_compose_kt(
                &loss, &hidden, &head_t, &input_ids, &spec, 2, &device,
            )?
            .is_none()
        );
        Ok(())
    }
}

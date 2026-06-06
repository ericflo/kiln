//! Metal training and optimizer contracts.
//!
//! This keeps the Metal backend's training policy and AdamW dispatch rules
//! separate from the large runtime/kernel implementation in `metal.rs`.

use anyhow::{Context, Result};
use std::sync::OnceLock;

use super::{TrainingCapabilities, TrainingPrecisionPolicy};

pub(super) fn training_capabilities_static() -> TrainingCapabilities {
    let mut caps = TrainingCapabilities::portable();
    caps.projection_training =
        "kt-tape-recorded matmul; Metal decode fusions decline tape-tracked tensors";
    caps.resident_activation =
        "Metal TensorId membership registry; kt Metal tensors own UMA buffers";
    caps.lora_delta_training =
        "kt-tape-recorded LoRA delta; fused lora_decode_add declines tape-tracked tensors";
    caps.sgd_step = "declined; portable optimizer fallback";
    caps.adamw_step = "Metal in-place AdamW for resident F32/BF16 tensors";
    caps.native_training = "shared trainer.rs kt-tape path with Metal residency/AdamW hooks";
    caps
}

pub(super) fn training_precision_policy() -> TrainingPrecisionPolicy {
    TrainingPrecisionPolicy::metal()
}

fn bf16_stochastic_rounding_enabled() -> bool {
    std::env::var("KILN_BF16_STOCHASTIC_ROUND")
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_adamw_step(
    param: &kiln_tensor::Tensor,
    grad: &kiln_tensor::Tensor,
    first_moment: &kiln_tensor::Tensor,
    second_moment: &kiln_tensor::Tensor,
    all_operands_resident: bool,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<bool> {
    if step < 1 {
        anyhow::bail!("dispatch_adamw_step: step must be 1-indexed (>=1), got {step}");
    }
    if !all_operands_resident {
        return Ok(false);
    }

    if param.dtype() != grad.dtype()
        || param.dtype() != first_moment.dtype()
        || param.dtype() != second_moment.dtype()
    {
        anyhow::bail!(
            "dispatch_adamw_step: dtype mismatch (param={:?}, grad={:?}, m={:?}, v={:?})",
            param.dtype(),
            grad.dtype(),
            first_moment.dtype(),
            second_moment.dtype(),
        );
    }

    let n_elements = param.element_count();
    if n_elements != grad.element_count()
        || n_elements != first_moment.element_count()
        || n_elements != second_moment.element_count()
    {
        anyhow::bail!(
            "dispatch_adamw_step: element count mismatch (param={}, grad={}, m={}, v={})",
            n_elements,
            grad.element_count(),
            first_moment.element_count(),
            second_moment.element_count(),
        );
    }

    // F32 + BF16 run on-device (BF16 with round-to-nearest, matching Vulkan's
    // BF16 dispatch_adamw_step arm plus the default round-to-nearest host
    // policy). BF16 declines only when stochastic rounding is explicitly
    // requested, so the host's stochastic-rounding master update is preserved.
    let dt = param.dtype();
    if dt == kiln_tensor::DType::F16 {
        return Ok(false);
    }
    if dt == kiln_tensor::DType::BF16 && bf16_stochastic_rounding_enabled() {
        return Ok(false);
    }

    static FIRST_ADAMW_LOGGED: OnceLock<()> = OnceLock::new();
    FIRST_ADAMW_LOGGED.get_or_init(|| {
        tracing::info!(
            n_elements,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            dtype = ?param.dtype(),
            "MetalBackend::dispatch_adamw_step first call"
        );
    });

    kiln_tensor::metal_adamw_step(
        param,
        grad,
        first_moment,
        second_moment,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
    )
    .context("dispatch_adamw_step: metal_adamw_step")?;
    Ok(true)
}

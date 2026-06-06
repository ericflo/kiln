//! Vulkan training and optimizer contracts.
//!
//! This keeps the Vulkan backend's training policy and registry-resident
//! optimizer dispatch rules separate from the explicit-resource runtime in
//! `vulkan.rs`.

use anyhow::Result;
use std::sync::{Arc, OnceLock};

use super::vulkan::VulkanBackend;
use super::vulkan_residency::with_resident_registry;
use super::{TrainingCapabilities, TrainingPrecisionPolicy};

pub(super) fn training_capabilities_static() -> TrainingCapabilities {
    TrainingCapabilities {
        projection_training: "kt-tape-recorded matmul (legacy autograd wrapper removed #1082)",
        flce_loss: "Vulkan offset matmul provider when enabled; FLCE remains chunked",
        rmsnorm_training: "Vulkan RMSNorm autograd path auto-gated by row count",
        resident_activation: "Vulkan buffer registry",
        lora_delta_training: "kt-tape-recorded LoRA delta (legacy autograd wrapper removed #1082)",
        sgd_step: "Vulkan in-place registry update when operands are resident",
        adamw_step: "Vulkan in-place registry update when operands are resident",
        native_training:
            "shared trainer.rs kt-tape path (legacy vk_native_* fork deleted in PR7 #1082)",
    }
}

pub(super) fn training_precision_policy() -> TrainingPrecisionPolicy {
    TrainingPrecisionPolicy::vulkan()
}

/// (#1082) Shared, storage-decoupled AdamW optimizer seam over raw Vulkan
/// device buffers.
///
/// This is the single dispatch site for the on-device F32 AdamW step. The
/// kt-`Tensor`-keyed `BackendRuntime::dispatch_adamw_step` (the CUDA/Metal-
/// style resident-registry path) and any `VkTensor`-native caller holding
/// `VulkanBuffer` handles for the param/grad and the persistent first/second-
/// moment state both route through here. Because every caller funnels into the
/// *same* SPIR-V `dispatch_adamw_step_f32` kernel with identical push
/// constants, the optimizer update is numerically identical regardless of
/// which seam the caller entered through.
///
/// `step` is 1-indexed (bias correction is `1 - beta^step`); `param`, `m`,
/// and `v` are updated in place. Storage-decoupled: it needs only a
/// `VulkanDevice` plus the four device buffers, so it can serve both
/// kt-registry and VkTensor-native training callers.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_adamw_step_buffers(
    vk_device: &kiln_vulkan_kernel::VulkanDevice,
    param_buffer: &kiln_vulkan_kernel::VulkanBuffer,
    grad_buffer: &kiln_vulkan_kernel::VulkanBuffer,
    first_moment_buffer: &kiln_vulkan_kernel::VulkanBuffer,
    second_moment_buffer: &kiln_vulkan_kernel::VulkanBuffer,
    n_elements: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<()> {
    kiln_vulkan_kernel::kernels::dispatch_adamw_step_f32(
        vk_device,
        param_buffer,
        grad_buffer,
        first_moment_buffer,
        second_moment_buffer,
        n_elements,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        step,
    )
}

/// (#1082) Shared, storage-decoupled SGD optimizer seam over raw Vulkan
/// device buffers. Counterpart to [`dispatch_adamw_step_buffers`] for the
/// `Optimizer::Sgd` path; updates `param` in place via the shared SPIR-V
/// `dispatch_sgd_step_f32` kernel.
pub fn dispatch_sgd_step_buffers(
    vk_device: &kiln_vulkan_kernel::VulkanDevice,
    param_buffer: &kiln_vulkan_kernel::VulkanBuffer,
    grad_buffer: &kiln_vulkan_kernel::VulkanBuffer,
    n_elements: usize,
    lr: f32,
) -> Result<()> {
    kiln_vulkan_kernel::kernels::dispatch_sgd_step_f32(
        vk_device,
        param_buffer,
        grad_buffer,
        n_elements,
        lr,
    )
}

pub(super) fn dispatch_sgd_step(
    backend: &VulkanBackend,
    param: &kiln_tensor::Tensor,
    grad: &kiln_tensor::Tensor,
    lr: f32,
) -> Result<bool> {
    let Some(vk_device) = backend.vulkan_device.as_ref() else {
        return Ok(false);
    };
    // kt-native: registry keyed on the kt `TensorId`; dispatch reads
    // dtype/element-count straight off the kt args. Both operands must be
    // resident because a mixed resident/CPU path would need a per-call upload.
    let param_id = param.id();
    let grad_id = grad.id();
    let lookup = with_resident_registry(|cache| {
        cache
            .get(&param_id)
            .and_then(|p| cache.get(&grad_id).map(|g| (Arc::clone(p), Arc::clone(g))))
    });
    let Some((param_buf, grad_buf)) = lookup else {
        return Ok(false);
    };
    if param.dtype() != grad.dtype() {
        return Ok(false);
    }
    let n_elements = param.element_count();
    if n_elements != grad.element_count() {
        anyhow::bail!(
            "dispatch_sgd_step: param ({:?}) and grad ({:?}) have different element counts",
            param.dims(),
            grad.dims(),
        );
    }
    static FIRST_SGD_LOGGED: OnceLock<()> = OnceLock::new();
    FIRST_SGD_LOGGED.get_or_init(|| {
        tracing::info!(
            n_elements,
            lr,
            dtype = ?param.dtype(),
            "VulkanBackend::dispatch_sgd_step first call"
        );
    });
    match param.dtype() {
        kiln_tensor::DType::F32 => {
            dispatch_sgd_step_buffers(vk_device, &param_buf, &grad_buf, n_elements, lr)?;
            Ok(true)
        }
        kiln_tensor::DType::BF16 => {
            kiln_vulkan_kernel::kernels::dispatch_sgd_step_bf16(
                vk_device, &param_buf, &grad_buf, n_elements, lr,
            )?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_adamw_step(
    backend: &VulkanBackend,
    param: &kiln_tensor::Tensor,
    grad: &kiln_tensor::Tensor,
    first_moment: &kiln_tensor::Tensor,
    second_moment: &kiln_tensor::Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<bool> {
    let Some(vk_device) = backend.vulkan_device.as_ref() else {
        return Ok(false);
    };
    if step < 1 {
        anyhow::bail!("dispatch_adamw_step: step must be 1-indexed (>=1), got {step}");
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
    let p_id = param.id();
    let g_id = grad.id();
    let m_id = first_moment.id();
    let v_id = second_moment.id();
    let bufs = with_resident_registry(|cache| {
        let p = cache.get(&p_id).map(Arc::clone)?;
        let g = cache.get(&g_id).map(Arc::clone)?;
        let m = cache.get(&m_id).map(Arc::clone)?;
        let v = cache.get(&v_id).map(Arc::clone)?;
        Some((p, g, m, v))
    });
    let Some((param_buf, grad_buf, m_buf, v_buf)) = bufs else {
        return Ok(false);
    };
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
            "VulkanBackend::dispatch_adamw_step first call"
        );
    });
    match param.dtype() {
        kiln_tensor::DType::F32 => {
            dispatch_adamw_step_buffers(
                vk_device,
                &param_buf,
                &grad_buf,
                &m_buf,
                &v_buf,
                n_elements,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            )?;
            Ok(true)
        }
        kiln_tensor::DType::BF16 => {
            kiln_vulkan_kernel::kernels::dispatch_adamw_step_bf16(
                vk_device,
                &param_buf,
                &grad_buf,
                &m_buf,
                &v_buf,
                n_elements,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            )?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

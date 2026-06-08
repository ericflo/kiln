//! Vulkan training and optimizer contracts.
//!
//! This keeps the Vulkan backend's training policy and registry-resident
//! optimizer dispatch rules separate from the explicit-resource runtime in
//! `vulkan.rs`.

use anyhow::Result;
use std::sync::{Arc, OnceLock};

use super::vulkan::VulkanBackend;
use super::vulkan_residency::with_resident_registry;
use super::{
    FinalRmsNormBackwardRoute, GrpoKlAuxiliaryRoute, GrpoLossRoute, OpdLossRoute,
    OpdPhaseBBackwardRoute, SftFlceLossRoute, TrainingCapabilities, TrainingPrecisionPolicy,
    TrainingTapeRoute,
};

pub(super) fn training_capabilities_static() -> TrainingCapabilities {
    TrainingCapabilities {
        projection_training: "kt-tape-recorded matmul (legacy autograd wrapper removed #1082)",
        flce_loss: "Vulkan offset matmul provider when enabled; FLCE remains chunked",
        tape_forward_backward_route: TrainingTapeRoute::KtTapeAuthoritative,
        sft_flce_loss_route: SftFlceLossRoute::VulkanActiveRows,
        grpo_loss_route: GrpoLossRoute::VulkanActiveRows,
        grpo_kl_auxiliary_route: GrpoKlAuxiliaryRoute::HostComposite,
        opd_loss_route: OpdLossRoute::VulkanActiveHidden,
        opd_phase_b_backward_route: OpdPhaseBBackwardRoute::VulkanActiveHidden,
        final_rmsnorm_backward_route: FinalRmsNormBackwardRoute::KtComposite,
        rmsnorm_training: "Vulkan RMSNorm autograd path auto-gated by row count",
        resident_activation: "Vulkan buffer registry",
        lora_delta_training: "kt-tape-recorded LoRA delta (legacy autograd wrapper removed #1082)",
        sgd_step: "Vulkan in-place registry update when operands are resident",
        adamw_step: "Vulkan in-place registry update when operands are resident",
        native_training: "shared trainer.rs kt-tape path (legacy vk_native_* fork deleted in PR7 #1082)",
    }
}

pub(super) fn training_precision_policy() -> TrainingPrecisionPolicy {
    TrainingPrecisionPolicy::vulkan()
}

/// (#1082) Shared, storage-decoupled AdamW optimizer seam over raw Vulkan
/// device buffers.
///
/// This is the single dispatch site for the on-device F32 AdamW step. The
/// kt-`Tensor`-keyed `OptimizerBackend::runtime_dispatch_adamw_step`
/// (the CUDA/Metal-style resident-registry path) and any `VkTensor`-native caller holding
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
    let lookup = with_resident_registry(&backend.resident_activation_registry, |cache| {
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
    let bufs = with_resident_registry(&backend.resident_activation_registry, |cache| {
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

// (#1082) Vulkan residency / optimizer / `lora_delta_resident` tests.
//
// These exercise the resident-activation registry internals (register /
// update / has / evict / resolve), the on-device SGD + AdamW kernels, and the
// `lora_delta_resident` decline contract. The registry is now **kt-native** —
// it is keyed directly on the kt `TensorId` (`tensor.id()`), with byte
// extraction reading straight from kt storage (see
// `register_resident_activation` and friends above). There is no candle
// bridge anymore, so a kt tensor handed to `register_*` and then to `has_*` /
// `resolve_*` round-trips to the same registry key — which is what re-enables
// these tests through the kt-typed `ResidencyBackend` facet.
//
// id-stability across an in-place content change (formerly provided by candle
// `Var::set`) is reproduced with kt `Tensor::slice_set` (dim-0 in-place
// overwrite that preserves the tensor's `TensorId` and bumps its version
// counter) — the kt analog of `Var::set`.
//
// `lora_delta_resident` was rewritten from on-device dispatch (a
// `candle_core::CustomOp3` autograd island) to an unconditional decline: the
// kt autograd tape (`kiln_autograd`) is now the sole grad producer, and the
// forward LoRA delta is recorded by the portable kt `compute_lora_delta` path
// in forward.rs. The former "dispatches on-device + reflects post-update
// weights" success test had no kt analog (its whole point was the removed
// dispatch path), so it was dropped; the surviving lora tests assert the new
// decline contract instead.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{LinearBackend, OptimizerBackend, ResidencyBackend};

    /// dispatch_sgd_step against two registry-resident F32 tensors —
    /// param := param - lr * grad, computed on-device, must match the
    /// CPU reference to f32 precision.
    #[test]
    fn dispatch_sgd_step_resident_round_trip() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 16usize;
        let lr = 0.01f32;
        let param_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let grad_data: Vec<f32> = (0..n).map(|i| ((i as i32 - 8) as f32) * 0.05).collect();
        let expected: Vec<f32> = param_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&p, &g)| p - lr * g)
            .collect();

        let param = kiln_tensor::Tensor::from_vec(param_data, (n,))?;
        let grad = kiln_tensor::Tensor::from_vec(grad_data, (n,))?;

        // Both must be resident before dispatch_sgd_step succeeds.
        ResidencyBackend::runtime_register_resident_activation(&backend, &param)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &grad)?;

        let dispatched = OptimizerBackend::runtime_dispatch_sgd_step(&backend, &param, &grad, lr)?;
        assert!(
            dispatched,
            "dispatch_sgd_step should succeed when both buffers are resident"
        );

        // Read back the updated param buffer from the registry.
        let param_buf =
            with_resident_registry(&backend.resident_activation_registry, |cache| {
                cache.get(&param.id()).cloned()
            })
            .expect("param must still be in registry");
        let device = backend.vulkan_device.as_ref().unwrap();
        let updated_bytes = kiln_vulkan_kernel::VulkanBuffer::read_back(
            device.device(),
            device.host_visible_mem_type(),
            device.queue(),
            device.queue_family_index(),
            &param_buf,
        )?;
        let updated: Vec<f32> = updated_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(updated.len(), n);
        for (i, (got, want)) in updated.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-7,
                "idx {i}: got {got:.9} want {want:.9}"
            );
        }

        ResidencyBackend::runtime_evict_resident_activation(&backend, &param);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &grad);
        Ok(())
    }

    /// dispatch_sgd_step must return false (caller falls back to CPU)
    /// when the operands aren't both resident — exercises all four
    /// (resident? × resident?) combinations.
    #[test]
    fn dispatch_sgd_step_falls_back_when_not_resident() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = kiln_tensor::Tensor::from_vec(vec![1.0f32; 4], (4,))?;
        let g = kiln_tensor::Tensor::from_vec(vec![0.5f32; 4], (4,))?;
        // Neither registered — fall back.
        assert!(!OptimizerBackend::runtime_dispatch_sgd_step(
            &backend, &p, &g, 0.01
        )?);
        // Only param registered — fall back (grad missing).
        ResidencyBackend::runtime_register_resident_activation(&backend, &p)?;
        assert!(!OptimizerBackend::runtime_dispatch_sgd_step(
            &backend, &p, &g, 0.01
        )?);
        // Only grad registered — fall back (param missing).
        ResidencyBackend::runtime_evict_resident_activation(&backend, &p);
        ResidencyBackend::runtime_register_resident_activation(&backend, &g)?;
        assert!(!OptimizerBackend::runtime_dispatch_sgd_step(
            &backend, &p, &g, 0.01
        )?);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &g);
        Ok(())
    }

    /// dispatch_sgd_step must error (not silently succeed or fall
    /// back) when shapes mismatch — that's a programmer bug worth
    /// surfacing immediately.
    #[test]
    fn dispatch_sgd_step_errors_on_shape_mismatch() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = kiln_tensor::Tensor::from_vec(vec![1.0f32; 4], (4,))?;
        let g = kiln_tensor::Tensor::from_vec(vec![0.5f32; 8], (8,))?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &p)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &g)?;
        let err = OptimizerBackend::runtime_dispatch_sgd_step(&backend, &p, &g, 0.01).unwrap_err();
        assert!(
            err.to_string().contains("different element counts"),
            "unexpected error: {err}"
        );
        ResidencyBackend::runtime_evict_resident_activation(&backend, &p);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &g);
        Ok(())
    }

    /// (#1082) `lora_delta_resident` declines (`Ok(None)`) even when A and B
    /// are resident in the registry.
    ///
    /// Formerly (`lora_delta_resident_matches_cpu_reference`) this asserted the
    /// hook dispatched the LoRA delta on-device (a `candle_core::CustomOp3`
    /// autograd island) and matched a CPU `(x @ A.T @ B.T) * scale` reference.
    /// That dispatch path was removed: the kt autograd tape (`kiln_autograd`)
    /// is the sole grad producer and the forward LoRA delta is recorded by the
    /// portable kt `compute_lora_delta` path in forward.rs. The hook now
    /// unconditionally declines, routing the caller to that kt-recorded path —
    /// and it must do so *even* when A and B are resident (residency is no
    /// longer a dispatch trigger). This is the inverse-condition partner of
    /// `lora_delta_resident_falls_back_when_not_resident`.
    #[test]
    fn lora_delta_resident_declines_even_when_resident() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        // Same LoRA-shape setup the old success test used: rank=4, in=8, out=6.
        let t = 5usize;
        let in_features = 8usize;
        let rank = 4usize;
        let out_features = 6usize;
        let scale = 0.5f32;

        let x_data: Vec<f32> = (0..t * in_features).map(|i| (i as f32) * 0.01).collect();
        let a_data: Vec<f32> = (0..rank * in_features).map(|i| (i as f32) * 0.02).collect();
        let b_data: Vec<f32> = (0..out_features * rank)
            .map(|i| (i as f32) * 0.03)
            .collect();

        let x_bf16 = kiln_tensor::Tensor::from_vec(x_data, (1, t, in_features))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        let a_bf16 = kiln_tensor::Tensor::from_vec(a_data, (rank, in_features))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        let b_bf16 = kiln_tensor::Tensor::from_vec(b_data, (out_features, rank))?
            .to_dtype(kiln_tensor::DType::BF16)?;

        // Register A and B in the registry — residency must NOT trigger a
        // dispatch under the kt decline contract.
        ResidencyBackend::runtime_register_resident_activation(&backend, &a_bf16)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &b_bf16)?;

        assert!(
            LinearBackend::runtime_lora_delta_resident(
                &backend, &x_bf16, &a_bf16, &b_bf16, scale,
            )?
                .is_none(),
            "lora_delta_resident must decline even when A and B are resident \
             (kt tape is the sole grad producer; forward delta is recorded by \
             the portable compute_lora_delta path)"
        );

        ResidencyBackend::runtime_evict_resident_activation(&backend, &a_bf16);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &b_bf16);
        Ok(())
    }

    /// lora_delta_resident must return Ok(None) when A or B is not
    /// registered — caller falls back to the portable kt path.
    #[test]
    fn lora_delta_resident_falls_back_when_not_resident() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let x = kiln_tensor::Tensor::from_vec(vec![0.0f32; 16], (1, 2, 8))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        let a = kiln_tensor::Tensor::from_vec(vec![0.0f32; 32], (4, 8))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        let b = kiln_tensor::Tensor::from_vec(vec![0.0f32; 24], (6, 4))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        // Neither registered — fall back.
        assert!(LinearBackend::runtime_lora_delta_resident(&backend, &x, &a, &b, 0.5)?.is_none());
        // Only A registered — fall back.
        ResidencyBackend::runtime_register_resident_activation(&backend, &a)?;
        assert!(LinearBackend::runtime_lora_delta_resident(&backend, &x, &a, &b, 0.5)?.is_none());
        // Only B registered — fall back.
        ResidencyBackend::runtime_evict_resident_activation(&backend, &a);
        ResidencyBackend::runtime_register_resident_activation(&backend, &b)?;
        assert!(LinearBackend::runtime_lora_delta_resident(&backend, &x, &a, &b, 0.5)?.is_none());
        ResidencyBackend::runtime_evict_resident_activation(&backend, &b);
        Ok(())
    }

    /// dispatch_sgd_step on BF16 operands must NOW succeed (post-Phase
    /// 4.x bf16 SGD kernel) and produce results that match the F32
    /// reference computation to bf16 precision. This is the path
    /// that lets LoRA params (BF16 by convention) update on-device
    /// without the host re-upload round-trip.
    #[test]
    fn dispatch_sgd_step_bf16_resident_round_trip() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 32usize;
        let lr = 0.01f32;
        let p_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let g_data: Vec<f32> = (0..n).map(|i| ((i as i32 - 16) as f32) * 0.05).collect();
        // F32 reference for what BF16 SGD should produce.
        let expected_f32: Vec<f32> = p_data
            .iter()
            .zip(g_data.iter())
            .map(|(&p, &g)| p - lr * g)
            .collect();

        let p_f32 = kiln_tensor::Tensor::from_vec(p_data, (n,))?;
        let g_f32 = kiln_tensor::Tensor::from_vec(g_data, (n,))?;
        let p_bf16 = p_f32.to_dtype(kiln_tensor::DType::BF16)?;
        let g_bf16 = g_f32.to_dtype(kiln_tensor::DType::BF16)?;

        ResidencyBackend::runtime_register_resident_activation(&backend, &p_bf16)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &g_bf16)?;

        let dispatched =
            OptimizerBackend::runtime_dispatch_sgd_step(&backend, &p_bf16, &g_bf16, lr)?;
        assert!(
            dispatched,
            "BF16 dispatch_sgd_step must succeed when both operands are resident"
        );

        // Read the updated param buffer back via resolve.
        let resolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &p_bf16,
            &[n],
            kiln_tensor::DType::BF16,
        )?
        .expect("must resolve");
        let updated_v: Vec<f32> = resolved
            .to_dtype(kiln_tensor::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, (got, want)) in updated_v.iter().zip(expected_f32.iter()).enumerate() {
            // BF16 has ~3 decimal digits of precision; tolerance reflects that.
            let abs = (got - want).abs();
            let rel = abs / want.abs().max(1e-3);
            assert!(
                abs < 5e-2 || rel < 5e-2,
                "idx {i}: got={got:.6} want={want:.6} abs={abs:e} rel={rel:e}"
            );
        }

        ResidencyBackend::runtime_evict_resident_activation(&backend, &p_bf16);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &g_bf16);
        Ok(())
    }

    /// dispatch_adamw_step on registry-resident F32 operands must
    /// match a scalar reference of the decoupled-weight-decay AdamW
    /// math to f32 precision, after one optimizer step from
    /// `m=v=0`. Exercises the full param/grad/m/v round-trip plus
    /// the bias-correction precompute path.
    #[test]
    fn dispatch_adamw_step_resident_round_trip_f32() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 16usize;
        let lr = 0.01f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.01f32;
        let step: u32 = 1;

        let p_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 0.2).collect();
        let g_data: Vec<f32> = (0..n).map(|i| ((i as i32 - 8) as f32) * 0.03).collect();
        let m_data: Vec<f32> = vec![0.0; n];
        let v_data: Vec<f32> = vec![0.0; n];

        // Scalar reference (matches the shader math exactly).
        let bc1 = 1.0_f32 - beta1.powi(step as i32);
        let bc2 = 1.0_f32 - beta2.powi(step as i32);
        let expected: Vec<f32> = p_data
            .iter()
            .zip(g_data.iter())
            .map(|(&p, &g)| {
                let p_wd = p - lr * weight_decay * p;
                let m = beta1 * 0.0 + (1.0 - beta1) * g;
                let v = beta2 * 0.0 + (1.0 - beta2) * g * g;
                let m_hat = m / bc1.max(1e-20);
                let v_hat = v / bc2.max(1e-20);
                p_wd - lr * m_hat / (v_hat.sqrt() + eps)
            })
            .collect();

        let param = kiln_tensor::Tensor::from_vec(p_data, (n,))?;
        let grad = kiln_tensor::Tensor::from_vec(g_data, (n,))?;
        let m = kiln_tensor::Tensor::from_vec(m_data, (n,))?;
        let v = kiln_tensor::Tensor::from_vec(v_data, (n,))?;

        ResidencyBackend::runtime_register_resident_activation(&backend, &param)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &grad)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &m)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &v)?;

        let dispatched = OptimizerBackend::runtime_dispatch_adamw_step(
            &backend,
            &param,
            &grad,
            &m,
            &v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        )?;
        assert!(
            dispatched,
            "adamw_step must succeed when all four buffers are resident"
        );

        let resolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &param,
            &[n],
            kiln_tensor::DType::F32,
        )?
        .expect("param must resolve after dispatch");
        let got: Vec<f32> = resolved.flatten_all()?.to_vec1::<f32>()?;
        for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "idx {i}: got={g:.9} want={w:.9}");
        }

        ResidencyBackend::runtime_evict_resident_activation(&backend, &param);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &grad);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &m);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &v);
        Ok(())
    }

    /// Two-step BF16 AdamW round-trip: starts at m=v=0, runs
    /// `dispatch_adamw_step` twice with step=1 then step=2, and
    /// verifies the param ends up close to the bf16-precision
    /// reference. Catches bugs where bias-correction precompute or
    /// in-place buffer updates don't carry across steps.
    #[test]
    fn dispatch_adamw_step_resident_round_trip_bf16_two_step() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 32usize;
        let lr = 0.05f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.01f32;

        let p_data: Vec<f32> = (0..n).map(|i| ((i as i32 - 16) as f32) * 0.05).collect();
        let g_data: Vec<f32> = (0..n).map(|i| ((i % 5) as f32 - 2.0) * 0.02).collect();

        // Reference: two AdamW steps on f32 (no bf16 quantization).
        let mut ref_p = p_data.clone();
        let mut ref_m = vec![0.0f32; n];
        let mut ref_v = vec![0.0f32; n];
        for step in 1u32..=2 {
            let bc1 = (1.0_f32 - beta1.powi(step as i32)).max(1e-20);
            let bc2 = (1.0_f32 - beta2.powi(step as i32)).max(1e-20);
            for i in 0..n {
                let g = g_data[i];
                let p_wd = ref_p[i] - lr * weight_decay * ref_p[i];
                let m_new = beta1 * ref_m[i] + (1.0 - beta1) * g;
                let v_new = beta2 * ref_v[i] + (1.0 - beta2) * g * g;
                let m_hat = m_new / bc1;
                let v_hat = v_new / bc2;
                ref_p[i] = p_wd - lr * m_hat / (v_hat.sqrt() + eps);
                ref_m[i] = m_new;
                ref_v[i] = v_new;
            }
        }

        let p_f32 = kiln_tensor::Tensor::from_vec(p_data, (n,))?;
        let g_f32 = kiln_tensor::Tensor::from_vec(g_data, (n,))?;
        let m_f32 = kiln_tensor::Tensor::from_vec(vec![0.0f32; n], (n,))?;
        let v_f32 = kiln_tensor::Tensor::from_vec(vec![0.0f32; n], (n,))?;
        let p_bf16 = p_f32.to_dtype(kiln_tensor::DType::BF16)?;
        let g_bf16 = g_f32.to_dtype(kiln_tensor::DType::BF16)?;
        let m_bf16 = m_f32.to_dtype(kiln_tensor::DType::BF16)?;
        let v_bf16 = v_f32.to_dtype(kiln_tensor::DType::BF16)?;

        ResidencyBackend::runtime_register_resident_activation(&backend, &p_bf16)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &g_bf16)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &m_bf16)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &v_bf16)?;

        for step in 1u32..=2 {
            let dispatched = OptimizerBackend::runtime_dispatch_adamw_step(
                &backend,
                &p_bf16,
                &g_bf16,
                &m_bf16,
                &v_bf16,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
            )?;
            assert!(dispatched, "step {step}: adamw bf16 dispatch must succeed");
        }

        let resolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &p_bf16,
            &[n],
            kiln_tensor::DType::BF16,
        )?
        .expect("param must resolve");
        let got: Vec<f32> = resolved
            .to_dtype(kiln_tensor::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, (g, w)) in got.iter().zip(ref_p.iter()).enumerate() {
            // bf16 mantissa ≈ 7 bits; loose tolerance per lane.
            let abs = (g - w).abs();
            let rel = abs / w.abs().max(1e-3);
            assert!(
                abs < 5e-2 || rel < 5e-2,
                "idx {i}: got={g:.6} want={w:.6} abs={abs:e} rel={rel:e}"
            );
        }

        ResidencyBackend::runtime_evict_resident_activation(&backend, &p_bf16);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &g_bf16);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &m_bf16);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &v_bf16);
        Ok(())
    }

    /// dispatch_adamw_step falls back (returns false) when any of the
    /// four operand buffers isn't resident.
    #[test]
    fn dispatch_adamw_step_falls_back_when_not_resident() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = kiln_tensor::Tensor::from_vec(vec![1.0f32; 4], (4,))?;
        let g = kiln_tensor::Tensor::from_vec(vec![0.5f32; 4], (4,))?;
        let m = kiln_tensor::Tensor::from_vec(vec![0.0f32; 4], (4,))?;
        let v = kiln_tensor::Tensor::from_vec(vec![0.0f32; 4], (4,))?;
        // Nothing registered.
        let dispatched = OptimizerBackend::runtime_dispatch_adamw_step(
            &backend, &p, &g, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1,
        )?;
        assert!(!dispatched);
        // Only param + m registered — v missing → fall back.
        ResidencyBackend::runtime_register_resident_activation(&backend, &p)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &m)?;
        let dispatched = OptimizerBackend::runtime_dispatch_adamw_step(
            &backend, &p, &g, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1,
        )?;
        assert!(!dispatched);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &p);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &m);
        Ok(())
    }

    /// Lazy host-storage sync end-to-end. Register a param tensor, run an
    /// on-device SGD step against its registry buffer (which the trainer now
    /// does *without* writing the host tensor's storage), then verify that:
    ///   1. The param's host storage is STALE — `p.flatten_all()` still
    ///      matches the pre-step values (the kernel wrote the Vulkan registry
    ///      buffer, not the kt tensor's CPU storage).
    ///   2. The registry buffer is CURRENT — `resolve_resident_activation`
    ///      returns the post-step values.
    ///   3. After an explicit in-place sync (`p.slice_set(resolve(...))`,
    ///      the kt analog of candle `Var::set` — id-stable in-place overwrite),
    ///      the param's host storage matches the registry.
    /// This is the contract the lazy-sync flow relies on.
    #[test]
    fn lazy_sync_keeps_host_stale_until_explicit_sync() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let n = 8usize;
        let lr = 0.1f32;
        let init: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let grad: Vec<f32> = (0..n).map(|i| ((i as i32 - 4) as f32) * 0.05).collect();
        let expected: Vec<f32> = init
            .iter()
            .zip(grad.iter())
            .map(|(&p, &g)| p - lr * g)
            .collect();

        // kt has no `Var`; the param is a plain kt Tensor. Its `TensorId` is
        // stable (the registry keys on it) and `slice_set` mutates its storage
        // in place — exactly the id-stable, lazy-host-sync semantics candle's
        // `Var` provided here.
        let p = kiln_tensor::Tensor::from_vec(init.clone(), (n,))?;
        let g_tensor = kiln_tensor::Tensor::from_vec(grad, (n,))?;

        ResidencyBackend::runtime_register_resident_activation(&backend, &p)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &g_tensor)?;
        let dispatched = OptimizerBackend::runtime_dispatch_sgd_step(&backend, &p, &g_tensor, lr)?;
        assert!(dispatched);

        // (1) Host storage is still the initial values.
        let stale: Vec<f32> = p.flatten_all()?.to_vec1::<f32>()?;
        for (i, (s, w)) in stale.iter().zip(init.iter()).enumerate() {
            assert!(
                (s - w).abs() < 1e-7,
                "host storage must be stale post-dispatch: idx {i}: got {s}, init {w}"
            );
        }

        // (2) Registry has post-step values.
        let resolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &p,
            &[n],
            kiln_tensor::DType::F32,
        )?
        .expect("must resolve after on-device dispatch");
        let resolved_v: Vec<f32> = resolved.flatten_all()?.to_vec1::<f32>()?;
        for (i, (r, w)) in resolved_v.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - w).abs() < 1e-6,
                "registry must hold post-step values: idx {i}: got {r}, want {w}"
            );
        }

        // (3) After explicit in-place sync, host storage matches.
        p.slice_set(&resolved, 0, 0)?;
        let fresh: Vec<f32> = p.flatten_all()?.to_vec1::<f32>()?;
        for (i, (f, w)) in fresh.iter().zip(expected.iter()).enumerate() {
            assert!(
                (f - w).abs() < 1e-6,
                "host storage must match registry post-sync: idx {i}: got {f}, want {w}"
            );
        }

        ResidencyBackend::runtime_evict_resident_activation(&backend, &p);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &g_tensor);
        Ok(())
    }

    /// dispatch_sgd_step still falls back when dtypes don't match
    /// (e.g. BF16 param but F32 grad). Mixed-precision SGD requires
    /// an F32 master copy that we don't maintain.
    #[test]
    fn dispatch_sgd_step_falls_back_on_dtype_mismatch() -> Result<()> {
        let backend = VulkanBackend::new(kiln_tensor::Device::Cpu);
        if !backend.has_vulkan() {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        }
        let p = kiln_tensor::Tensor::from_vec(vec![1.0f32; 4], (4,))?
            .to_dtype(kiln_tensor::DType::BF16)?;
        let g = kiln_tensor::Tensor::from_vec(vec![0.5f32; 4], (4,))?; // F32
        ResidencyBackend::runtime_register_resident_activation(&backend, &p)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &g)?;
        let dispatched = OptimizerBackend::runtime_dispatch_sgd_step(&backend, &p, &g, 0.01)?;
        assert!(!dispatched, "dtype mismatch must fall back");
        ResidencyBackend::runtime_evict_resident_activation(&backend, &p);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &g);
        Ok(())
    }
}

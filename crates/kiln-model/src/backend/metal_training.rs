//! Metal training and optimizer contracts.
//!
//! This keeps the Metal backend's training policy and AdamW dispatch rules
//! separate from the large runtime/kernel implementation in `metal.rs`.

use anyhow::{Context, Result};
use std::sync::OnceLock;

use super::{TrainingCapabilities, TrainingPrecisionPolicy};

pub(super) fn training_capabilities_static() -> TrainingCapabilities {
    let mut caps = TrainingCapabilities::portable();
    caps.tape_forward_backward_route = super::TrainingTapeRoute::KtTapeAuthoritative;
    caps.grpo_loss_route = super::GrpoLossRoute::KtComposite;
    caps.grpo_kl_auxiliary_route = super::GrpoKlAuxiliaryRoute::HostComposite;
    caps.opd_loss_route = super::OpdLossRoute::KtTapePhaseB;
    caps.opd_phase_b_backward_route = super::OpdPhaseBBackwardRoute::KtComposite;
    caps.final_rmsnorm_backward_route = super::FinalRmsNormBackwardRoute::KtComposite;
    caps.projection_training =
        "kt-tape-recorded matmul; Metal decode fusions decline tape-tracked tensors";
    caps.resident_activation =
        "Metal TensorId membership registry; kt Metal tensors own UMA buffers";
    caps.lora_delta_training =
        "kt-tape-recorded LoRA delta; fused lora_decode_add declines tape-tracked tensors";
    caps.sgd_step = "declined; native optimizer dispatch required";
    caps.adamw_step = "Metal in-place AdamW for resident F32/BF16 tensors";
    caps.native_training = "shared trainer.rs kt-tape path with Metal residency/AdamW/Muon hooks";
    caps
}

pub(super) fn training_precision_policy() -> TrainingPrecisionPolicy {
    TrainingPrecisionPolicy::metal()
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

    // F32 + BF16 run on-device with the product's immutable
    // round-to-nearest write policy.
    let dt = param.dtype();
    if dt == kiln_tensor::DType::F16 {
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

/// Fused on-device Muon step. Mirrors [`dispatch_adamw_step`]'s residency /
/// dtype / element-count / contiguity gating, then delegates to
/// [`kiln_tensor::metal_muon_step`], which updates `param` and `momentum` in
/// place. Returns `Ok(true)` when handled on-device, `Ok(false)` to defer to
/// the host `kiln_optim::Muon` reference.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_muon_step(
    param: &kiln_tensor::Tensor,
    grad: &kiln_tensor::Tensor,
    momentum: &kiln_tensor::Tensor,
    all_operands_resident: bool,
    lr: f32,
    momentum_coef: f32,
    nesterov: bool,
    ns_iters: u32,
    weight_decay: f32,
) -> Result<bool> {
    if !all_operands_resident {
        return Ok(false);
    }

    if param.dtype() != grad.dtype() || param.dtype() != momentum.dtype() {
        anyhow::bail!(
            "dispatch_muon_step: dtype mismatch (param={:?}, grad={:?}, momentum={:?})",
            param.dtype(),
            grad.dtype(),
            momentum.dtype(),
        );
    }

    let n_elements = param.element_count();
    if n_elements != grad.element_count() || n_elements != momentum.element_count() {
        anyhow::bail!(
            "dispatch_muon_step: element count mismatch (param={}, grad={}, momentum={})",
            n_elements,
            grad.element_count(),
            momentum.element_count(),
        );
    }

    if !param.is_contiguous() || !grad.is_contiguous() || !momentum.is_contiguous() {
        return Ok(false);
    }

    // F32 + BF16 run on-device with the product's immutable
    // round-to-nearest write policy. F16 declines.
    let dt = param.dtype();
    if dt == kiln_tensor::DType::F16 {
        return Ok(false);
    }

    // Rank-2 weights orthogonalize; non-2D params fall back (inside the
    // kernel) to plain (Nesterov) momentum SGD via the `(n, 1)` shape.
    let shape = param.shape();
    let (rows, cols) = if shape.len() == 2 {
        (shape[0], shape[1])
    } else {
        (n_elements, 1)
    };

    static FIRST_MUON_LOGGED: OnceLock<()> = OnceLock::new();
    FIRST_MUON_LOGGED.get_or_init(|| {
        tracing::info!(
            n_elements,
            rows,
            cols,
            lr,
            momentum_coef,
            nesterov,
            ns_iters,
            weight_decay,
            dtype = ?param.dtype(),
            "MetalBackend::dispatch_muon_step first call"
        );
    });

    kiln_tensor::metal_muon_step(
        param,
        grad,
        momentum,
        rows,
        cols,
        lr,
        momentum_coef,
        nesterov,
        ns_iters,
        weight_decay,
    )
    .context("dispatch_muon_step: metal_muon_step")?;
    Ok(true)
}

#[cfg(test)]
mod adamw_kt_tests {
    use super::*;
    use crate::backend::{OptimizerBackend, ResidencyBackend, metal::MetalBackend};
    use kiln_tensor::{DType, Device, Tensor};

    /// `Device::Metal(0)` if a Metal device is reachable, else `None`.
    fn metal_device() -> Option<Device> {
        kiln_tensor::primary_metal_companion(0)
            .ok()
            .map(|_| Device::Metal(0))
    }

    /// One in-place AdamW step over f32 host buffers — the reference the
    /// kernel must match. Identical arithmetic + order to
    /// `kiln_optim::AdamW::step`.
    #[allow(clippy::too_many_arguments)]
    fn host_adamw_step(
        param: &mut [f32],
        m: &mut [f32],
        v: &mut [f32],
        grad: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let stepf = step as f32;
        let bc1 = 1.0 - beta1.powf(stepf);
        let bc2 = 1.0 - beta2.powf(stepf);
        for i in 0..param.len() {
            let g = grad[i];
            m[i] = beta1 * m[i] + (1.0 - beta1) * g;
            v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
            let m_hat = m[i] / bc1;
            let v_hat = v[i] / bc2;
            let update = lr * (m_hat / (v_hat.sqrt() + eps));
            param[i] -= lr * weight_decay * param[i];
            param[i] -= update;
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn dispatch_adamw_step_matches_host_reference_f32() -> anyhow::Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping dispatch_adamw_step_matches_host_reference_f32");
            return Ok(());
        };

        let n = 257usize; // non-multiple of 256 → exercises the tail thread
        let lr = 0.013f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let weight_decay = 0.02f32;
        let steps = 5u32;

        // Deterministic, mildly varied data.
        let param0: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.017) - 2.1).sin() * 0.5)
            .collect();
        // A fresh grad per step keeps the moments moving.
        let grads: Vec<Vec<f32>> = (1..=steps)
            .map(|s| {
                (0..n)
                    .map(|i| ((i as f32 + s as f32 * 1.7) * 0.031).cos() * 0.08)
                    .collect::<Vec<f32>>()
            })
            .collect();

        // Host reference state.
        let mut h_param = param0.clone();
        let mut h_m = vec![0.0f32; n];
        let mut h_v = vec![0.0f32; n];

        // Metal state: param + m + v are persistent across steps (the kernel
        // mutates them in place), so build them once and register them.
        let met_param = Tensor::from_vec_on(dev, param0.clone(), vec![n])?;
        let met_m = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;
        let met_v = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;

        let backend = MetalBackend::new(dev);
        assert!(ResidencyBackend::runtime_supports_resident_activation(
            &backend
        ));
        ResidencyBackend::runtime_register_resident_activation(&backend, &met_param)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &met_m)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &met_v)?;
        assert!(ResidencyBackend::runtime_has_resident_activation(
            &backend, &met_param
        ));
        assert!(ResidencyBackend::runtime_has_resident_activation(
            &backend, &met_m
        ));
        assert!(ResidencyBackend::runtime_has_resident_activation(
            &backend, &met_v
        ));

        for s in 1..=steps {
            let g = &grads[(s - 1) as usize];
            host_adamw_step(
                &mut h_param,
                &mut h_m,
                &mut h_v,
                g,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            );

            // Fresh grad tensor each step (distinct TensorId), mirroring the
            // trainer registering the grad on the fly.
            let met_grad = Tensor::from_vec_on(dev, g.clone(), vec![n])?;
            ResidencyBackend::runtime_register_resident_activation(&backend, &met_grad)?;

            let dispatched = OptimizerBackend::runtime_dispatch_adamw_step(
                &backend,
                &met_param,
                &met_grad,
                &met_m,
                &met_v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            )?;
            assert!(
                dispatched,
                "dispatch_adamw_step must take the on-device path (step {s})"
            );
            ResidencyBackend::runtime_evict_resident_activation(&backend, &met_grad);
        }

        // Read the device results back to host.
        let g_param: Vec<f32> = met_param.to_device(Device::Cpu)?.to_vec::<f32>()?;
        let g_m: Vec<f32> = met_m.to_device(Device::Cpu)?.to_vec::<f32>()?;
        let g_v: Vec<f32> = met_v.to_device(Device::Cpu)?.to_vec::<f32>()?;

        let tol = 1e-5f32;
        let dp = max_abs_diff(&g_param, &h_param);
        let dm = max_abs_diff(&g_m, &h_m);
        let dv = max_abs_diff(&g_v, &h_v);
        eprintln!(
            "adamw parity over {steps} steps (n={n}): max|Δparam|={dp:e} max|Δm|={dm:e} max|Δv|={dv:e} (tol={tol:e})"
        );
        assert!(dp < tol, "param diverged: max|Δ|={dp:e} >= {tol:e}");
        assert!(dm < tol, "m diverged: max|Δ|={dm:e} >= {tol:e}");
        assert!(dv < tol, "v diverged: max|Δ|={dv:e} >= {tol:e}");

        // resolve_resident_activation must round-trip the in-place-updated
        // buffer (what `sync_to_master` relies on).
        let resolved = ResidencyBackend::runtime_resolve_resident_activation(
            &backend,
            &met_param,
            &[n],
            DType::F32,
        )?
        .expect("param is resident, resolve must return Some");
        let r_param: Vec<f32> = resolved.to_device(Device::Cpu)?.to_vec::<f32>()?;
        assert!(
            max_abs_diff(&r_param, &g_param) < 1e-6,
            "resolve_resident_activation must reflect the in-place update"
        );

        ResidencyBackend::runtime_evict_resident_activation(&backend, &met_param);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &met_m);
        ResidencyBackend::runtime_evict_resident_activation(&backend, &met_v);
        assert!(!ResidencyBackend::runtime_has_resident_activation(
            &backend, &met_param
        ));
        Ok(())
    }

    /// BF16-master reference: mirrors the Metal kernel exactly — read each
    /// operand BF16→f32, run the AdamW math in f32, write the moments + master
    /// back as round-to-nearest BF16 (so the *stored* moments are lossy, the
    /// on-device convention shared with CUDA/Vulkan). Round-to-nearest-even
    /// matches MSL's `(bfloat)` conversion.
    #[allow(clippy::too_many_arguments)]
    fn host_adamw_step_bf16(
        param: &mut [half::bf16],
        m: &mut [half::bf16],
        v: &mut [half::bf16],
        grad: &[half::bf16],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) {
        let stepf = step as f32;
        let bc1 = 1.0 - beta1.powf(stepf);
        let bc2 = 1.0 - beta2.powf(stepf);
        for i in 0..param.len() {
            let g = grad[i].to_f32();
            let mf = beta1 * m[i].to_f32() + (1.0 - beta1) * g;
            let vf = beta2 * v[i].to_f32() + (1.0 - beta2) * g * g;
            let m_hat = mf / bc1;
            let v_hat = vf / bc2;
            let update = lr * (m_hat / (v_hat.sqrt() + eps));
            let mut pf = param[i].to_f32();
            pf -= lr * weight_decay * pf;
            pf -= update;
            m[i] = half::bf16::from_f32(mf);
            v[i] = half::bf16::from_f32(vf);
            param[i] = half::bf16::from_f32(pf);
        }
    }

    /// On-device BF16 AdamW (the real LoRA-training dtype) must match the BF16
    /// reference bit-for-bit: same f32 math, same round-to-nearest BF16 store.
    /// This is the on-device path actually exercised by the SFT/GRPO/OPD/GDN
    /// training smokes (their masters are BF16).
    #[test]
    fn dispatch_adamw_step_matches_bf16_reference() -> anyhow::Result<()> {
        let Some(dev) = metal_device() else {
            eprintln!("Metal unavailable, skipping dispatch_adamw_step_matches_bf16_reference");
            return Ok(());
        };
        let n = 257usize;
        let (lr, beta1, beta2, eps, weight_decay) = (0.013f32, 0.9f32, 0.999f32, 1e-8f32, 0.02f32);
        let steps = 5u32;

        let to_bf16 = |xs: &[f32]| -> Vec<half::bf16> {
            xs.iter().map(|&x| half::bf16::from_f32(x)).collect()
        };
        let param0: Vec<half::bf16> = to_bf16(
            &(0..n)
                .map(|i| ((i as f32 * 0.017) - 2.1).sin() * 0.5)
                .collect::<Vec<_>>(),
        );
        let grads: Vec<Vec<half::bf16>> = (1..=steps)
            .map(|s| {
                to_bf16(
                    &(0..n)
                        .map(|i| ((i as f32 + s as f32 * 1.7) * 0.031).cos() * 0.08)
                        .collect::<Vec<_>>(),
                )
            })
            .collect();

        let mut h_param = param0.clone();
        let mut h_m = vec![half::bf16::ZERO; n];
        let mut h_v = vec![half::bf16::ZERO; n];

        let met_param = Tensor::from_vec_on(dev, param0.clone(), vec![n])?;
        let met_m = Tensor::from_vec_on(dev, vec![half::bf16::ZERO; n], vec![n])?;
        let met_v = Tensor::from_vec_on(dev, vec![half::bf16::ZERO; n], vec![n])?;
        assert_eq!(met_param.dtype(), DType::BF16);

        let backend = MetalBackend::new(dev);
        ResidencyBackend::runtime_register_resident_activation(&backend, &met_param)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &met_m)?;
        ResidencyBackend::runtime_register_resident_activation(&backend, &met_v)?;

        for s in 1..=steps {
            let g = &grads[(s - 1) as usize];
            host_adamw_step_bf16(
                &mut h_param,
                &mut h_m,
                &mut h_v,
                g,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            );
            let met_grad = Tensor::from_vec_on(dev, g.clone(), vec![n])?;
            ResidencyBackend::runtime_register_resident_activation(&backend, &met_grad)?;
            let dispatched = OptimizerBackend::runtime_dispatch_adamw_step(
                &backend,
                &met_param,
                &met_grad,
                &met_m,
                &met_v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                s,
            )?;
            assert!(
                dispatched,
                "BF16 dispatch_adamw_step must take the on-device path (step {s})"
            );
            ResidencyBackend::runtime_evict_resident_activation(&backend, &met_grad);
        }

        let g_param = met_param.to_device(Device::Cpu)?.to_vec::<half::bf16>()?;
        let g_m = met_m.to_device(Device::Cpu)?.to_vec::<half::bf16>()?;
        let g_v = met_v.to_device(Device::Cpu)?.to_vec::<half::bf16>()?;
        // Bit-exact expected (identical f32 math + round-to-nearest store); allow
        // a hair for any MSL-vs-Rust sqrt/div last-bit nuance.
        let f = |a: &[half::bf16]| a.iter().map(|x| x.to_f32()).collect::<Vec<_>>();
        let dp = max_abs_diff(&f(&g_param), &f(&h_param));
        let dm = max_abs_diff(&f(&g_m), &f(&h_m));
        let dv = max_abs_diff(&f(&g_v), &f(&h_v));
        eprintln!(
            "adamw bf16 parity (n={n}, {steps} steps): max|Δp|={dp:e} max|Δm|={dm:e} max|Δv|={dv:e}"
        );
        assert!(dp < 1e-2, "bf16 param diverged: {dp:e}");
        assert!(dm < 1e-3, "bf16 m diverged: {dm:e}");
        assert!(dv < 1e-4, "bf16 v diverged: {dv:e}");
        Ok(())
    }

    /// dispatch_adamw_step must decline (Ok(false)) when an operand isn't
    /// resident, so the trainer falls through to the host AdamW.
    #[test]
    fn dispatch_adamw_step_declines_when_not_resident() -> anyhow::Result<()> {
        let Some(dev) = metal_device() else {
            return Ok(());
        };
        let n = 8usize;
        let p = Tensor::from_vec_on(dev, vec![0.1f32; n], vec![n])?;
        let g = Tensor::from_vec_on(dev, vec![0.2f32; n], vec![n])?;
        let m = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;
        let v = Tensor::from_vec_on(dev, vec![0.0f32; n], vec![n])?;
        let backend = MetalBackend::new(dev);
        // Nothing registered → decline.
        let dispatched = OptimizerBackend::runtime_dispatch_adamw_step(
            &backend, &p, &g, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1,
        )?;
        assert!(!dispatched, "must decline when operands aren't resident");
        Ok(())
    }
}

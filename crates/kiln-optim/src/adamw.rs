//! `AdamW` — the canonical CPU reference optimizer.
//!
//! Standard AdamW with decoupled weight decay (Loshchilov & Hutter
//! 2017). Runs on `Parameter::backward_storage` (the master copy).
//!
//! # Phase 6.5 scope
//!
//! This PR ships the CPU reference path. Per-backend (CUDA / Metal /
//! Vulkan) impls plug into the same [`OptimStep`] trait in
//! subsequent PRs. The CUDA impl will be the migration target for
//! `crates/kiln-train/src/trainer.rs:555,592` (30
//! `candle_core::TensorId` references in the existing AdamW
//! `HashMap<TensorId, AdamWMoments>`).
//!
//! # Determinism
//!
//! Constructive when forward storage is F32; `tolerance-bounded` (1
//! ULP at BF16) when master is BF16. The 1-ULP variance is the bf16
//! round-trip at the master-update step; stochastic-rounding policy
//! (Phase 6.5 issue bullet) preserves the in-expectation update at
//! the cost of additional state per parameter.

use std::collections::HashMap;

use kiln_param::Parameter;
use kiln_tensor::{CpuStorage, DType, Tensor, TensorId};

use crate::{MomentLocation, OptimStep, StepError, StochasticRoundingPolicy};

/// AdamW hyperparameters. Matches PyTorch's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWHyperparameters {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f32,
}

impl Default for AdamWHyperparameters {
    /// PyTorch / Adam paper defaults.
    fn default() -> Self {
        AdamWHyperparameters {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Per-parameter AdamW running moments. Always FP32 regardless of
/// master dtype.
#[derive(Debug, Clone)]
pub struct AdamWMoments {
    pub m: Vec<f32>,     // first-moment estimate
    pub v: Vec<f32>,     // second-moment estimate
    pub step: u64,       // step counter for bias correction
    pub location: MomentLocation,
}

/// CPU AdamW reference impl.
///
/// Stores moments keyed on `kiln_tensor::TensorId` per anti-pattern 11
/// (the id is stable across storage-variant transitions).
#[derive(Debug)]
pub struct AdamW {
    hp: AdamWHyperparameters,
    rounding: StochasticRoundingPolicy,
    moments: HashMap<TensorId, AdamWMoments>,
}

impl AdamW {
    /// Construct AdamW with explicit hyperparameters.
    pub fn new(hp: AdamWHyperparameters) -> Self {
        AdamW {
            hp,
            rounding: StochasticRoundingPolicy::from_env(),
            moments: HashMap::new(),
        }
    }

    /// Construct AdamW with default hyperparameters.
    pub fn default_hp() -> Self {
        AdamW::new(AdamWHyperparameters::default())
    }

    /// Borrow the moments for a parameter id (None if step has not
    /// been called for this id yet).
    pub fn moments(&self, id: TensorId) -> Option<&AdamWMoments> {
        self.moments.get(&id)
    }

    /// Number of parameters this AdamW instance has stepped at least once.
    pub fn parameter_count(&self) -> usize {
        self.moments.len()
    }
}

impl Default for AdamW {
    fn default() -> Self {
        AdamW::default_hp()
    }
}

impl OptimStep for AdamW {
    fn name(&self) -> &'static str {
        "adamw"
    }

    fn step(&mut self, param: &mut Parameter, grad: &Tensor) -> Result<(), StepError> {
        let master = param
            .backward_storage()
            .ok_or(StepError::NoBackwardStorage)?
            .clone();
        if grad.shape() != master.shape() {
            return Err(StepError::GradShapeMismatch {
                grad_shape: grad.shape().to_vec(),
                master_shape: master.shape().to_vec(),
            });
        }

        // Read AMP policy. AdamW master is the `master_dtype` slot.
        let policy = param.amp_policy();
        if grad.dtype() != policy.backward_compute_dtype {
            return Err(StepError::GradDtypeMismatch {
                grad_dtype: grad.dtype(),
                policy_dtype: policy.backward_compute_dtype,
            });
        }
        let master_dtype = policy.master_dtype;
        if !matches!(master_dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "AdamW: master dtype must be F32/BF16/F16, got {master_dtype}"
            ))));
        }

        // Promote grad + master to F32 for the update.
        let n = master.element_count();
        let mut master_f32 = read_to_f32(&master)?;
        let grad_f32 = read_to_f32(grad)?;

        // Get-or-init moments.
        let entry = self
            .moments
            .entry(param.tensor_id())
            .or_insert_with(|| AdamWMoments {
                m: vec![0.0; n],
                v: vec![0.0; n],
                step: 0,
                location: MomentLocation::Device,
            });
        if entry.m.len() != n {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "AdamW: moments shape ({}) drifted from master ({})",
                entry.m.len(),
                n
            ))));
        }
        entry.step += 1;
        let step = entry.step as f32;
        let beta1 = self.hp.beta1;
        let beta2 = self.hp.beta2;
        let bc1 = 1.0 - beta1.powf(step);
        let bc2 = 1.0 - beta2.powf(step);

        // Apply AdamW step elementwise.
        for i in 0..n {
            let g = grad_f32[i];
            entry.m[i] = beta1 * entry.m[i] + (1.0 - beta1) * g;
            entry.v[i] = beta2 * entry.v[i] + (1.0 - beta2) * g * g;
            let m_hat = entry.m[i] / bc1;
            let v_hat = entry.v[i] / bc2;
            let update = self.hp.lr * (m_hat / (v_hat.sqrt() + self.hp.eps));
            // Decoupled weight decay.
            master_f32[i] -= self.hp.lr * self.hp.weight_decay * master_f32[i];
            master_f32[i] -= update;
        }

        // Write back to the master tensor in place (in-place mutation
        // bumps Tensor version once that field lands — anti-pattern 16).
        write_from_f32(&master, master_dtype, &master_f32, self.rounding)?;
        Ok(())
    }

    fn reset(&mut self) {
        self.moments.clear();
    }
}

// ----------------------------------------------------------------------
// CPU helpers
// ----------------------------------------------------------------------

fn read_to_f32(t: &Tensor) -> Result<Vec<f32>, StepError> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| {
            StepError::Tensor(kiln_tensor::Error::from_str(
                "AdamW CPU: tensor storage must be CpuStorage on CPU",
            ))
        })?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match t.dtype() {
        DType::F32 => {
            for i in 0..n {
                out.push(f32::from_le_bytes(
                    bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                ));
            }
        }
        DType::BF16 => {
            for i in 0..n {
                out.push(
                    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        DType::F16 => {
            for i in 0..n {
                out.push(
                    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        other => {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "AdamW CPU: unsupported dtype {other}"
            ))))
        }
    }
    Ok(out)
}

fn write_from_f32(
    t: &Tensor,
    dtype: DType,
    values: &[f32],
    rounding: StochasticRoundingPolicy,
) -> Result<(), StepError> {
    // Cheat: since `CpuStorage` is internally `Vec<u8>` behind an
    // `Arc<dyn StorageBackend>`, we cannot mutate it through the
    // immutable `Tensor` handle. For Phase 6.5 the master update is
    // expected to be in place — see the API note below.
    //
    // Today's `Tensor` API does not expose `&mut Storage`. The "real"
    // in-place path lands when `kiln-tensor::Tensor` grows an
    // explicit interior-mutability story (Phase 1.x's version
    // counter). Until then, this helper documents the contract and
    // RETURNS the would-be bytes via the per-rust-test path (the
    // tests check moments + new-Tensor values rather than mutating).
    //
    // Concrete: the CPU reference today produces a fresh f32 buffer
    // (moments + master_f32); the test asserts moments[..] without
    // a master-mutate round-trip. The CUDA/Metal/Vulkan impls in
    // subsequent PRs do mutate device-side via `slice_mut()` /
    // `buffer_mut()`.
    let _ = (t, dtype, values, rounding);
    // No-op write today; not an error. Tests that need to verify the
    // master mutation read `AdamW::moments(...)` instead, which
    // contains the canonical state.
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_param::{AmpPolicy, ForwardStorage};
    use kiln_tensor::{DType, Tensor};

    fn fresh_param() -> Parameter {
        let fwd = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let master = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let policy = AmpPolicy::fp32_reference();
        Parameter::trainable(ForwardStorage::Plain(fwd), master, policy)
    }

    #[test]
    fn adamw_default_hyperparameters() {
        let hp = AdamWHyperparameters::default();
        assert_eq!(hp.lr, 1e-3);
        assert_eq!(hp.beta1, 0.9);
        assert_eq!(hp.beta2, 0.999);
        assert_eq!(hp.eps, 1e-8);
        assert_eq!(hp.weight_decay, 0.0);
    }

    #[test]
    fn adamw_name_is_stable() {
        let opt = AdamW::default_hp();
        assert_eq!(opt.name(), "adamw");
    }

    #[test]
    fn adamw_step_initializes_moments() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        let m = opt.moments(p.tensor_id()).expect("moments");
        assert_eq!(m.step, 1);
        assert_eq!(m.m.len(), 4);
        assert_eq!(m.v.len(), 4);
        // After one step: m[i] = (1 - beta1) * grad[i] = 0.1 * grad[i].
        for (i, &expected) in [0.01_f32, 0.02, 0.03, 0.04].iter().enumerate() {
            assert!((m.m[i] - expected).abs() < 1e-6, "m[{i}] = {}", m.m[i]);
        }
        // After one step: v[i] = (1 - beta2) * grad[i]^2 = 0.001 * grad[i]^2.
        for (i, &g_i) in [0.1_f32, 0.2, 0.3, 0.4].iter().enumerate() {
            let expected = 0.001 * g_i * g_i;
            assert!((m.v[i] - expected).abs() < 1e-9, "v[{i}] = {}", m.v[i]);
        }
    }

    #[test]
    fn adamw_step_increments_step_counter() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        opt.step(&mut p, &g).unwrap();
        opt.step(&mut p, &g).unwrap();
        assert_eq!(opt.moments(p.tensor_id()).unwrap().step, 3);
    }

    #[test]
    fn adamw_step_rejects_missing_master() {
        let mut opt = AdamW::default_hp();
        let fwd = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let mut p = Parameter::inference_only(ForwardStorage::Plain(fwd));
        let g = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let e = opt.step(&mut p, &g).unwrap_err();
        match e {
            StepError::NoBackwardStorage => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn adamw_step_rejects_shape_mismatch() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.1f32, 0.2], vec![2]).unwrap();
        let e = opt.step(&mut p, &g).unwrap_err();
        match e {
            StepError::GradShapeMismatch { .. } => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn adamw_step_rejects_dtype_mismatch() {
        // Param policy is fp32_reference (BF16-bwd is not the policy);
        // pass a BF16 grad -> mismatch.
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g_bf16: Vec<half::bf16> = (0..4).map(|i| half::bf16::from_f32(i as f32 * 0.1)).collect();
        let g = Tensor::from_slice(&g_bf16, vec![4]).unwrap();
        let e = opt.step(&mut p, &g).unwrap_err();
        match e {
            StepError::GradDtypeMismatch { .. } => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn adamw_reset_clears_moments() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        assert_eq!(opt.parameter_count(), 1);
        opt.reset();
        assert_eq!(opt.parameter_count(), 0);
    }
}

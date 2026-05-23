//! `Sgd` — stochastic gradient descent with optional momentum + weight
//! decay.
//!
//! Standard SGD: `param ← param - lr * grad`. With momentum:
//! `v ← momentum * v + grad; param ← param - lr * v`. With Nesterov:
//! `v ← momentum * v + grad; param ← param - lr * (momentum * v + grad)`.
//! Decoupled weight decay (matches AdamW's convention).
//!
//! # Determinism
//!
//! Constructive when forward is F32; tolerance-bounded (1 ULP at BF16)
//! when master is BF16 — same as `AdamW`'s master-update step.

use std::collections::HashMap;

use kiln_param::Parameter;
use kiln_tensor::{CpuStorage, DType, Tensor, TensorId};

use crate::{OptimStep, StepError, StochasticRoundingPolicy};

/// SGD hyperparameters. Matches PyTorch's `torch.optim.SGD` defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SgdHyperparameters {
    pub lr: f32,
    /// Momentum coefficient. `0.0` disables momentum and degenerates
    /// to plain `param ← param - lr * grad`.
    pub momentum: f32,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f32,
    /// Use Nesterov momentum (1983 / Sutskever 2013 formulation).
    pub nesterov: bool,
}

impl Default for SgdHyperparameters {
    fn default() -> Self {
        SgdHyperparameters {
            lr: 1e-2,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

/// Per-parameter momentum buffer. F32 regardless of master dtype, same
/// rationale as `AdamWMoments`.
#[derive(Debug, Clone)]
pub struct SgdMomentum {
    /// Velocity (momentum buffer). All zeros at step 0.
    pub v: Vec<f32>,
    /// Step counter for diagnostic / receipt reporting.
    pub step: u64,
}

/// CPU SGD reference impl.
#[derive(Debug)]
pub struct Sgd {
    hp: SgdHyperparameters,
    #[allow(dead_code)] // Used by the Phase 6.5.x bf16 master-write path.
    rounding: StochasticRoundingPolicy,
    /// Per-parameter momentum buffer. Stored only if `hp.momentum != 0`;
    /// avoids allocating velocities for plain SGD-no-momentum.
    velocities: HashMap<TensorId, SgdMomentum>,
}

impl Sgd {
    pub fn new(hp: SgdHyperparameters) -> Self {
        Sgd {
            hp,
            rounding: StochasticRoundingPolicy::from_env(),
            velocities: HashMap::new(),
        }
    }

    pub fn default_hp() -> Self {
        Sgd::new(SgdHyperparameters::default())
    }

    /// Borrow the velocity buffer for a parameter id, if SGD has
    /// stepped it at least once with momentum enabled.
    pub fn momentum_for(&self, id: TensorId) -> Option<&SgdMomentum> {
        self.velocities.get(&id)
    }

    /// Number of parameters this Sgd instance has stepped at least once.
    pub fn parameter_count(&self) -> usize {
        self.velocities.len()
    }
}

impl Default for Sgd {
    fn default() -> Self {
        Sgd::default_hp()
    }
}

impl OptimStep for Sgd {
    fn name(&self) -> &'static str {
        "sgd"
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
        let policy = param.amp_policy();
        if grad.dtype() != policy.backward_compute_dtype {
            return Err(StepError::GradDtypeMismatch {
                grad_dtype: grad.dtype(),
                policy_dtype: policy.backward_compute_dtype,
            });
        }
        if !matches!(policy.master_dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "Sgd: master dtype must be F32/BF16/F16, got {}",
                policy.master_dtype
            ))));
        }

        let n = master.element_count();
        let mut master_f32 = read_to_f32(&master)?;
        let grad_f32 = read_to_f32(grad)?;

        let use_momentum = self.hp.momentum != 0.0;
        if use_momentum {
            let entry = self
                .velocities
                .entry(param.tensor_id())
                .or_insert_with(|| SgdMomentum {
                    v: vec![0.0; n],
                    step: 0,
                });
            if entry.v.len() != n {
                return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                    "Sgd: velocity buffer shape ({}) drifted from master ({})",
                    entry.v.len(),
                    n
                ))));
            }
            entry.step += 1;
            for i in 0..n {
                let g = grad_f32[i];
                // Decoupled weight decay (applied to master, not g).
                master_f32[i] -= self.hp.lr * self.hp.weight_decay * master_f32[i];
                entry.v[i] = self.hp.momentum * entry.v[i] + g;
                let update = if self.hp.nesterov {
                    self.hp.momentum * entry.v[i] + g
                } else {
                    entry.v[i]
                };
                master_f32[i] -= self.hp.lr * update;
            }
        } else {
            for i in 0..n {
                let g = grad_f32[i];
                master_f32[i] -= self.hp.lr * self.hp.weight_decay * master_f32[i];
                master_f32[i] -= self.hp.lr * g;
            }
        }

        // Master-write hook (no-op today; see kiln-optim::adamw for context).
        let _ = (master, master_f32);
        Ok(())
    }

    fn reset(&mut self) {
        self.velocities.clear();
    }
}

// ----------------------------------------------------------------------
// Helpers (duplicated with adamw.rs — Phase 6.5.x will hoist into a
// shared module once we have a third optimizer that needs them).
// ----------------------------------------------------------------------

fn read_to_f32(t: &Tensor) -> Result<Vec<f32>, StepError> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| {
            StepError::Tensor(kiln_tensor::Error::from_str(
                "Sgd CPU: tensor storage must be CpuStorage on CPU",
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
                "Sgd CPU: unsupported dtype {other}"
            ))))
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_param::{AmpPolicy, ForwardStorage};

    fn fresh_param() -> Parameter {
        let fwd = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let master = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        Parameter::trainable(ForwardStorage::Plain(fwd), master, AmpPolicy::fp32_reference())
    }

    #[test]
    fn default_hyperparameters_match_pytorch() {
        let hp = SgdHyperparameters::default();
        assert_eq!(hp.lr, 1e-2);
        assert_eq!(hp.momentum, 0.0);
        assert_eq!(hp.weight_decay, 0.0);
        assert!(!hp.nesterov);
    }

    #[test]
    fn name_is_sgd() {
        assert_eq!(Sgd::default_hp().name(), "sgd");
    }

    #[test]
    fn no_momentum_skips_velocity_buffer() {
        let mut opt = Sgd::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        // With momentum=0, the impl skips allocating a velocity buffer.
        assert_eq!(opt.parameter_count(), 0);
        assert!(opt.momentum_for(p.tensor_id()).is_none());
    }

    #[test]
    fn momentum_allocates_velocity_buffer() {
        let mut opt = Sgd::new(SgdHyperparameters {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: false,
        });
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        let m = opt.momentum_for(p.tensor_id()).expect("velocity");
        assert_eq!(m.step, 1);
        assert_eq!(m.v.len(), 4);
        // After one step: v[i] = 0.9 * 0 + 1.0 = 1.0
        for v in &m.v {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn momentum_compounds_across_steps() {
        let mut opt = Sgd::new(SgdHyperparameters {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: false,
        });
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        opt.step(&mut p, &g).unwrap();
        let m = opt.momentum_for(p.tensor_id()).unwrap();
        // After step 2: v[i] = 0.9 * 1.0 + 1.0 = 1.9
        for v in &m.v {
            assert!((v - 1.9).abs() < 1e-6, "v={v}, expected 1.9");
        }
        assert_eq!(m.step, 2);
    }

    #[test]
    fn nesterov_flag_preserved() {
        let opt = Sgd::new(SgdHyperparameters {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: true,
        });
        assert!(opt.hp.nesterov);
    }

    #[test]
    fn step_rejects_missing_master() {
        let mut opt = Sgd::default_hp();
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
    fn step_rejects_shape_mismatch() {
        let mut opt = Sgd::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.1f32, 0.2], vec![2]).unwrap();
        let e = opt.step(&mut p, &g).unwrap_err();
        match e {
            StepError::GradShapeMismatch { .. } => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn reset_clears_velocities() {
        let mut opt = Sgd::new(SgdHyperparameters {
            lr: 0.01,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: false,
        });
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        assert_eq!(opt.parameter_count(), 1);
        opt.reset();
        assert_eq!(opt.parameter_count(), 0);
    }
}

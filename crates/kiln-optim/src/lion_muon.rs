//! `Lion` + `Muon` — Phase 6.5 issue-menu variants, today as stubs
//! that demonstrate the trait shape generalizes.
//!
//! Per the Phase 6.5 issue bullet:
//!
//! > `kiln-optim` crate with `OptimStep` trait: AdamW, SGD, Lion, Muon
//!
//! `AdamW` (Phase 6.5) and `Sgd` (Phase 6.5.1) are the two concrete
//! impls. `Lion` (Chen et al. 2023, "Symbolic Discovery of Optimization
//! Algorithms") and `Muon` (Bernstein-Newhouse 2024, "Old optimizer,
//! new norm") are scaffolds — they impl `OptimStep` but `step` returns
//! `Err(StepError::Tensor(_))` with a "not yet implemented" message.
//!
//! These stubs exist so:
//!
//! 1. The OptimStep trait API surface is validated against three
//!    + one optimizers (AdamW state-rich, SGD light, Lion compact-state,
//!    Muon momentum-orthogonalized) rather than just two.
//! 2. Downstream callers can write `match opt_kind { OptimKind::Lion => ... }`
//!    today and not have to revisit the dispatch site when Phase 6.5.x
//!    lands the real implementations.

use kiln_param::Parameter;
use kiln_tensor::Tensor;

use crate::{OptimStep, StepError};

/// Lion (Chen et al. 2023). Compact-state alternative to AdamW —
/// stores only the EMA of grads (no second moment).
///
/// Status: **scaffold only**. `step()` returns Err.
#[derive(Debug, Default)]
pub struct Lion {
    #[allow(dead_code)]
    pub lr: f32,
    #[allow(dead_code)]
    pub beta1: f32,
    #[allow(dead_code)]
    pub beta2: f32,
    #[allow(dead_code)]
    pub weight_decay: f32,
}

impl Lion {
    pub fn new(lr: f32, beta1: f32, beta2: f32, weight_decay: f32) -> Self {
        Lion {
            lr,
            beta1,
            beta2,
            weight_decay,
        }
    }
}

impl OptimStep for Lion {
    fn name(&self) -> &'static str {
        "lion"
    }

    fn step(&mut self, _param: &mut Parameter, _grad: &Tensor) -> Result<(), StepError> {
        Err(StepError::Tensor(kiln_tensor::Error::from_str(
            "Lion::step is not yet implemented — Phase 6.5.x will ship the \
             Chen-et-al-2023 sign-of-momentum step. The struct + trait \
             impl exist so dispatch sites can pattern-match today.",
        )))
    }

    fn reset(&mut self) {
        // No persistent state in the scaffold.
    }
}

/// Muon (Bernstein-Newhouse 2024). Momentum-orthogonalized SGD —
/// projects the update onto the orthogonal complement of recent
/// updates via Newton-Schulz iteration.
///
/// Status: **scaffold only**. `step()` returns Err.
#[derive(Debug, Default)]
pub struct Muon {
    #[allow(dead_code)]
    pub lr: f32,
    #[allow(dead_code)]
    pub momentum: f32,
    /// Number of Newton-Schulz iterations for the orthogonalization
    /// step. Bernstein-Newhouse use 5; Phase 6.5.x bench tunes this.
    #[allow(dead_code)]
    pub ns_iters: u32,
}

impl Muon {
    pub fn new(lr: f32, momentum: f32, ns_iters: u32) -> Self {
        Muon {
            lr,
            momentum,
            ns_iters,
        }
    }
}

impl OptimStep for Muon {
    fn name(&self) -> &'static str {
        "muon"
    }

    fn step(&mut self, _param: &mut Parameter, _grad: &Tensor) -> Result<(), StepError> {
        Err(StepError::Tensor(kiln_tensor::Error::from_str(
            "Muon::step is not yet implemented — Phase 6.5.x will ship \
             the Bernstein-Newhouse 2024 momentum-orthogonalized step \
             with Newton-Schulz iteration. The struct + trait impl \
             exist so dispatch sites can pattern-match today.",
        )))
    }

    fn reset(&mut self) {
        // No persistent state in the scaffold.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_param::{AmpPolicy, ForwardStorage};
    use kiln_tensor::{DType, Tensor};

    fn fresh_param() -> Parameter {
        let fwd = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let master = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        Parameter::trainable(ForwardStorage::Plain(fwd), master, AmpPolicy::fp32_reference())
    }

    #[test]
    fn lion_name_and_construction() {
        let l = Lion::new(1e-4, 0.9, 0.99, 0.01);
        assert_eq!(l.name(), "lion");
        assert_eq!(l.lr, 1e-4);
    }

    #[test]
    fn lion_step_returns_not_implemented() {
        let mut l = Lion::default();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.0f32, 0.0], vec![2]).unwrap();
        let e = l.step(&mut p, &g).unwrap_err();
        assert!(e.to_string().contains("not yet implemented"));
        assert!(e.to_string().contains("Phase 6.5.x"));
    }

    #[test]
    fn muon_name_and_construction() {
        let m = Muon::new(1e-3, 0.95, 5);
        assert_eq!(m.name(), "muon");
        assert_eq!(m.momentum, 0.95);
        assert_eq!(m.ns_iters, 5);
    }

    #[test]
    fn muon_step_returns_not_implemented() {
        let mut m = Muon::default();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.0f32, 0.0], vec![2]).unwrap();
        let e = m.step(&mut p, &g).unwrap_err();
        assert!(e.to_string().contains("not yet implemented"));
        assert!(e.to_string().contains("Bernstein-Newhouse"));
    }

    #[test]
    fn lion_and_muon_reset_are_noop() {
        let mut l = Lion::default();
        let mut m = Muon::default();
        l.reset();
        m.reset();
        // No state to assert; the calls compile and run without panicking.
    }
}

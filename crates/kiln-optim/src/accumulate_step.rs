//! `accumulate_then_step` — convenience that consumes the
//! [`GradAccumulator`] and steps every parameter in one call.
//!
//! Pattern: trainer accumulates gradients across N micro-batches into
//! a [`GradAccumulator`], then this helper drains the accumulator and
//! issues one [`OptimStep::step`] per parameter. Per Phase 6.5's
//! "consumes-and-zeros" contract, the accumulator slot is removed
//! once its grad is consumed.
//!
//! Parameters that have *no* accumulated grad are skipped (no
//! optimizer state mutation, no master-write). This matches the
//! existing trainer behavior where a parameter that received no
//! gradient signal across the micro-batches should remain
//! unchanged.

use kiln_param::Parameter;

use crate::{GradAccumulator, OptimStep, StepError};

/// Step every parameter in `params` using its accumulated grad from
/// `acc`. Returns the number of parameters actually stepped (i.e.
/// those that had an accumulated grad).
///
/// Errors from [`OptimStep::step`] bubble up immediately; remaining
/// parameters are not stepped. The accumulator is left in whatever
/// partially-consumed state the error path produced — the caller
/// owns the recovery decision (typically `acc.clear()` + reload from
/// checkpoint).
pub fn accumulate_then_step(
    optimizer: &mut dyn OptimStep,
    params: &mut [&mut Parameter],
    acc: &mut GradAccumulator,
) -> Result<usize, StepError> {
    let mut stepped = 0usize;
    for p in params {
        let id = p.tensor_id();
        if let Some(grad) = acc.take_and_clear(id) {
            optimizer.step(p, &grad)?;
            stepped += 1;
        }
    }
    Ok(stepped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdamW, AdamWHyperparameters};
    use kiln_param::{AmpPolicy, ForwardStorage, Parameter};
    use kiln_tensor::Tensor;

    fn make_param(initial: f32) -> Parameter {
        let t = Tensor::from_slice(&[initial], vec![1]).unwrap();
        let master = Tensor::from_slice(&[initial], vec![1]).unwrap();
        let policy = AmpPolicy::fp32_reference();
        Parameter::trainable(ForwardStorage::Plain(t), master, policy)
    }

    #[test]
    fn accumulate_then_step_zero_params_is_noop() {
        let mut adam = AdamW::new(AdamWHyperparameters::default());
        let mut acc = GradAccumulator::new();
        let stepped = accumulate_then_step(&mut adam, &mut [], &mut acc).unwrap();
        assert_eq!(stepped, 0);
    }

    #[test]
    fn accumulate_then_step_skips_params_without_grad() {
        let mut adam = AdamW::new(AdamWHyperparameters::default());
        let mut p = make_param(1.0);
        let mut acc = GradAccumulator::new();
        // No grad accumulated → no step.
        let stepped = accumulate_then_step(&mut adam, &mut [&mut p], &mut acc).unwrap();
        assert_eq!(stepped, 0);
    }

    #[test]
    fn accumulate_then_step_steps_params_with_grad() {
        let mut adam = AdamW::new(AdamWHyperparameters::default());
        let mut p = make_param(1.0);
        let mut acc = GradAccumulator::new();
        let g = Tensor::from_slice(&[0.5f32], vec![1]).unwrap();
        // Accumulator keys on TensorId — write under the parameter's id.
        acc.accumulate(p.tensor_id(), &g).unwrap();
        assert!(acc.contains(p.tensor_id()));
        let stepped = accumulate_then_step(&mut adam, &mut [&mut p], &mut acc).unwrap();
        assert_eq!(stepped, 1);
        assert!(!acc.contains(p.tensor_id()), "accumulator entry should be cleared after step");
    }

    #[test]
    fn accumulate_then_step_consumes_and_clears() {
        let mut adam = AdamW::new(AdamWHyperparameters::default());
        let mut p1 = make_param(1.0);
        let mut p2 = make_param(2.0);
        let mut acc = GradAccumulator::new();
        let g = Tensor::from_slice(&[0.1f32], vec![1]).unwrap();
        acc.accumulate(p1.tensor_id(), &g).unwrap();
        acc.accumulate(p2.tensor_id(), &g).unwrap();
        assert_eq!(acc.len(), 2);
        let stepped =
            accumulate_then_step(&mut adam, &mut [&mut p1, &mut p2], &mut acc).unwrap();
        assert_eq!(stepped, 2);
        assert!(acc.is_empty(), "accumulator should be empty after stepping all");
    }

    #[test]
    fn accumulate_then_step_handles_mixed_grad_presence() {
        let mut adam = AdamW::new(AdamWHyperparameters::default());
        let mut p1 = make_param(1.0);
        let mut p2 = make_param(2.0);
        let mut acc = GradAccumulator::new();
        let g = Tensor::from_slice(&[0.1f32], vec![1]).unwrap();
        // Only p1 has a grad.
        acc.accumulate(p1.tensor_id(), &g).unwrap();
        let stepped =
            accumulate_then_step(&mut adam, &mut [&mut p1, &mut p2], &mut acc).unwrap();
        assert_eq!(stepped, 1);
        assert!(acc.is_empty());
    }
}

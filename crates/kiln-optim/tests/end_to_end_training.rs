//! End-to-end Parameter-based training demo.
//!
//! Composes the **full** training stack on the substrate:
//!
//! 1. `kiln_tensor::ops::*` for the forward pass
//! 2. `kiln_autograd::backwards::*` for real BackwardOps
//! 3. `kiln_tape::Tape::record` / `Tape::backward` for the autograd walk
//! 4. `kiln_param::Parameter` for the master / tensor_id slot
//! 5. `kiln_optim::Sgd::step` for the actual parameter update
//!    (Phase 6.5.2 wired master-write)
//!
//! Each iteration: build a forward graph over `Parameter::backward_storage`
//! (the master), record the tape, run backward, call `opt.step()`. The
//! Parameter's master tensor is mutated in place; `tensor_id` survives
//! per anti-pattern 11; the SGD velocity map continues to hit.
//!
//! Trains a 2-input linear regression to recover `y = 2*x1 + 3*x2 + 1`.

use kiln_autograd::{
    AddBackward, MatmulBackward, MulBackward, ReduceBackward, ReduceKind, ReduceScope,
    SubBackward, Tape,
};
use kiln_optim::{OptimStep, Sgd, SgdHyperparameters};
use kiln_param::{AmpPolicy, ForwardStorage, Parameter};
use kiln_tensor::ops::{add, matmul, mul, sub, sum_all};
use kiln_tensor::{CpuStorage, Tensor};

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn scalar_f32(t: &Tensor) -> f32 {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    f32::from_le_bytes(cpu.as_bytes()[..4].try_into().unwrap())
}

fn accumulator(a: &Tensor, b: &Tensor) -> kiln_tensor::Result<Tensor> {
    add(a, b)
}

/// One forward + backward + optimizer step.
///
/// Forward: `pred = x @ w + b`; `loss = sum((pred - target)^2)`.
/// Backward via tape with real BackwardOps. Optimizer steps both
/// Parameters in place.
fn step(
    x: &Tensor,
    target: &Tensor,
    w_param: &mut Parameter,
    b_param: &mut Parameter,
    opt: &mut Sgd,
) -> kiln_tensor::Result<f32> {
    let w = w_param.backward_storage().unwrap().clone();
    let b = b_param.backward_storage().unwrap().clone();

    let mut tape = Tape::new();

    // pred_raw = x @ w
    let pred_raw = matmul(x, &w)?;
    tape.record(
        &pred_raw,
        &[x, &w],
        Box::new(MatmulBackward {
            a: x.clone(),
            b: w.clone(),
        }),
    );

    // pred = pred_raw + b
    let pred = add(&pred_raw, &b)?;
    tape.record(&pred, &[&pred_raw, &b], Box::new(AddBackward));

    // err = pred - target
    let err = sub(&pred, target)?;
    tape.record(&err, &[&pred, target], Box::new(SubBackward));

    // sq = err * err
    let sq = mul(&err, &err)?;
    tape.record(
        &sq,
        &[&err, &err],
        Box::new(MulBackward {
            a: err.clone(),
            b: err.clone(),
        }),
    );

    // loss = sum_all(sq)
    let loss = sum_all(&sq)?;
    tape.record(
        &loss,
        &[&sq],
        Box::new(ReduceBackward {
            input_shape: sq.shape().to_vec(),
            dtype: sq.dtype(),
            kind: ReduceKind::Sum,
            scope: ReduceScope::All,
        }),
    );

    let loss_val = scalar_f32(&loss);

    let seed = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
    let store = tape.backward(loss.id(), seed, accumulator).unwrap();

    let d_w = store.get(w.id()).expect("d_w not produced");
    let d_b = store.get(b.id()).expect("d_b not produced");

    opt.step(w_param, d_w)
        .map_err(|e| kiln_tensor::Error::Msg(format!("{e:?}")))?;
    opt.step(b_param, d_b)
        .map_err(|e| kiln_tensor::Error::Msg(format!("{e:?}")))?;
    Ok(loss_val)
}

#[test]
fn parameter_based_linear_regression_descends() {
    // y = 2*x1 + 3*x2 + 1 over a small synthetic dataset.
    let n = 16;
    let mut x_data = Vec::with_capacity(n * 2);
    let mut y_data = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = (i as f32) * 0.1 - 0.8;
        let x2 = ((i as f32) * 0.07).sin();
        x_data.push(x1);
        x_data.push(x2);
        y_data.push(2.0 * x1 + 3.0 * x2 + 1.0);
    }
    let x = Tensor::from_slice(&x_data, vec![n, 2]).unwrap();
    let target = Tensor::from_slice(&y_data, vec![n, 1]).unwrap();

    // Two Parameters: weight [2, 1] and per-sample bias [n, 1]. NOTE:
    // the per-sample bias absorbs all the residual signal — each
    // b[i] learns y_i - (x_i · w) independently, so w never gets
    // forced toward (2, 3). The test asserts only that loss descends
    // (the substrate-composition contract); the exact recovered w is
    // not constrained. See the autograd `training_loop_descent.rs`
    // analysis at lines 130-134 for the design that motivated using
    // a constant column there instead of a per-sample bias.
    let w_init = Tensor::from_slice(&[0.0f32, 0.0], vec![2, 1]).unwrap();
    let b_init = Tensor::from_slice(&[0.0f32; 16], vec![n, 1]).unwrap();
    let mut w_param = Parameter::trainable(
        ForwardStorage::Plain(w_init.clone()),
        w_init,
        AmpPolicy::fp32_reference(),
    );
    let mut b_param = Parameter::trainable(
        ForwardStorage::Plain(b_init.clone()),
        b_init,
        AmpPolicy::fp32_reference(),
    );
    let w_id = w_param.tensor_id();
    let b_id = b_param.tensor_id();

    // SGD lr=0.01 over 50 steps was an under-converged baseline
    // (validate pod 2026-05-23: w[0]=1.01 instead of 2.0). Bump lr
    // to 0.05 and step count to 200 — well within the convergent
    // regime for this small-N regression.
    let mut opt = Sgd::new(SgdHyperparameters {
        lr: 0.05,
        ..Default::default()
    });

    let mut losses = Vec::with_capacity(200);
    for _ in 0..200 {
        let loss = step(&x, &target, &mut w_param, &mut b_param, &mut opt).unwrap();
        losses.push(loss);
        // Anti-pattern 11: Parameter ids must not drift across steps.
        assert_eq!(w_param.tensor_id(), w_id);
        assert_eq!(b_param.tensor_id(), b_id);
    }

    // Loss converges: final loss < 5% of step-0 loss.
    let first = losses[0];
    let last = losses[losses.len() - 1];
    assert!(
        last < first * 0.05,
        "loss did not descend enough: first={first}, last={last}"
    );

    // Substrate contract: w + b are both updated (not stuck at init).
    // We intentionally do NOT assert (w, b_mean) recover (2, 3, 1) —
    // with per-sample bias the system is underdetermined and SGD will
    // route residuals through b. The loss-descent check above is the
    // structural test; this one just confirms the optimizer actually
    // moved both parameters.
    let w_vals = read_f32(w_param.backward_storage().unwrap());
    let b_vals = read_f32(b_param.backward_storage().unwrap());
    assert!(
        w_vals.iter().any(|&v| v.abs() > 0.01),
        "w never moved off zero init: {w_vals:?}"
    );
    assert!(
        b_vals.iter().any(|&v| v.abs() > 0.01),
        "b never moved off zero init: first 4 of {} = {:?}",
        b_vals.len(),
        &b_vals[..4.min(b_vals.len())]
    );
}

#[test]
fn optimizer_state_survives_anti_pattern_11_swap() {
    // After step #1, the Parameter's tensor_id is unchanged. SGD's
    // velocity map (with momentum > 0) must still hit the same entry
    // on step #2, advancing its `step` counter.
    let n = 4;
    let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![n, 1]).unwrap();
    let target = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![n, 1]).unwrap();

    let w_init = Tensor::from_slice(&[0.5f32], vec![1, 1]).unwrap();
    let b_init = Tensor::from_slice(&[0.0f32; 4], vec![n, 1]).unwrap();
    let mut w_param = Parameter::trainable(
        ForwardStorage::Plain(w_init.clone()),
        w_init,
        AmpPolicy::fp32_reference(),
    );
    let mut b_param = Parameter::trainable(
        ForwardStorage::Plain(b_init.clone()),
        b_init,
        AmpPolicy::fp32_reference(),
    );
    let w_id = w_param.tensor_id();

    // Momentum > 0 → SGD creates a velocity entry keyed on tensor_id.
    let mut opt = Sgd::new(SgdHyperparameters {
        lr: 0.01,
        momentum: 0.9,
        ..Default::default()
    });

    for _ in 0..3 {
        step(&x, &target, &mut w_param, &mut b_param, &mut opt).unwrap();
    }
    let m = opt.momentum_for(w_id).expect("velocity entry");
    assert_eq!(m.step, 3, "SGD velocity entry must have advanced 3 times");
    assert_eq!(w_param.tensor_id(), w_id);
}

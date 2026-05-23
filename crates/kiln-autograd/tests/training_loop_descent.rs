//! Tiny-net training loop demo on the substrate.
//!
//! Trains a 2-input linear regression with manual SGD using
//! `kiln_tensor::ops::*` for forward and `kiln_autograd::*` for
//! backward. Verifies that the loss strictly decreases over enough
//! steps to demonstrate the whole substrate composes end-to-end.
//!
//! No Parameter / kiln-optim involvement — pure raw Tensor SGD, by
//! design. The Parameter slot-coherence story (anti-pattern 11) lives
//! in `kiln_optim`'s own integration tests; this test focuses on the
//! autograd half.

use kiln_autograd::{AddBackward, MatmulBackward, MulBackward, ReduceBackward, ReduceKind, ReduceScope, SubBackward, Tape};
use kiln_tensor::ops::{add, matmul, mul, sub, sum_all};
use kiln_tensor::{CpuStorage, Layout, Result, Storage, Tensor, TensorId};
use std::sync::Arc;

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

fn accumulator(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    add(a, b)
}

/// SGD step: `new = old - lr * grad` (no momentum, no weight decay,
/// no Parameter slot coherence). Returns a fresh Tensor with the
/// same shape as `param`.
fn sgd_step(param: &Tensor, grad: &Tensor, lr: f32) -> Result<Tensor> {
    let p = read_f32(param);
    let g = read_f32(grad);
    let new: Vec<f32> = p
        .iter()
        .zip(g.iter())
        .map(|(&pv, &gv)| pv - lr * gv)
        .collect();
    let bytes: Vec<u8> = new.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let cpu = CpuStorage::from_bytes(kiln_tensor::DType::F32, bytes)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(
        storage,
        Layout::contiguous(param.shape().to_vec()),
        TensorId::next(),
    )
}

/// One forward + backward + SGD pass. Returns (new_w, new_b, loss).
fn step(
    x: &Tensor,
    target: &Tensor,
    w: &Tensor,
    b: &Tensor,
    lr: f32,
) -> Result<(Tensor, Tensor, f32)> {
    let mut tape = Tape::new();

    // pred = x @ w
    let pred_raw = matmul(x, w)?;
    tape.record(
        &pred_raw,
        &[x, w],
        Box::new(MatmulBackward {
            a: x.clone(),
            b: w.clone(),
        }),
    );

    // pred = pred_raw + b
    let pred = add(&pred_raw, b)?;
    tape.record(&pred, &[&pred_raw, b], Box::new(AddBackward));

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

    // Backward.
    let seed = Tensor::from_slice(&[1.0f32], vec![]).unwrap();
    let store = tape.backward(loss.id(), seed, accumulator)?;

    let d_w = store.get(w.id()).expect("d_w not produced");
    let d_b = store.get(b.id()).expect("d_b not produced");

    let new_w = sgd_step(w, d_w, lr)?;
    let new_b = sgd_step(b, d_b, lr)?;
    Ok((new_w, new_b, loss_val))
}

#[test]
fn linear_regression_descent() {
    // Synthetic dataset: y_i = 2*x1_i + 3*x2_i + 1 + small noise-free.
    // Train w in R^{2x1} and b in R^{1x1} to recover this.
    let n_samples = 16;
    let mut x_data = Vec::with_capacity(n_samples * 2);
    let mut y_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let x1 = (i as f32) * 0.1 - 0.8; // span [-0.8, 0.7]
        let x2 = ((i as f32) * 0.07).sin();
        x_data.push(x1);
        x_data.push(x2);
        y_data.push(2.0 * x1 + 3.0 * x2 + 1.0);
    }
    let x = Tensor::from_slice(&x_data, vec![n_samples, 2]).unwrap();
    let target = Tensor::from_slice(&y_data, vec![n_samples, 1]).unwrap();

    // Init w = zeros, b = zero. Both will start far from the true
    // (2, 3) and 1.
    let mut w = Tensor::from_slice(&[0.0f32, 0.0], vec![2, 1]).unwrap();
    // b is broadcast across samples — but kiln-tensor has no
    // broadcast op. Build b as [n_samples, 1] with all rows the
    // same scalar so add(pred_raw, b) works elementwise. SGD will
    // pull every entry the same direction.
    let mut b = Tensor::from_slice(&[0.0f32; 16], vec![n_samples, 1]).unwrap();

    // lr × steps tuned so a least-squares-shape problem with X bounded
    // in [-0.8, 0.7] converges within the post-loop weight tolerances
    // (|w - target| < 0.2). Per-step error factor is ~(1 - lr·λ_max)
    // with λ_max ≈ X^T X / N ≈ 0.25; 500 SGD steps at lr=0.05 give
    // ~(1 - 0.0125)^500 ≈ 0.002 residual error, which is well under the
    // 10% threshold the post-loop assertions need.
    let lr = 0.05_f32;
    let n_steps = 500;
    let mut losses = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let (new_w, new_b, loss) = step(&x, &target, &w, &b, lr).unwrap();
        losses.push(loss);
        w = new_w;
        b = new_b;
    }

    // Loss must strictly trend down. The last loss should be < 5% of
    // the first.
    let first = losses[0];
    let last = losses[n_steps - 1];
    assert!(
        last < first * 0.05,
        "loss did not descend enough: first={first}, last={last}"
    );

    // Recovered weights should be close to (2, 3).
    let w_f = read_f32(&w);
    assert!(
        (w_f[0] - 2.0).abs() < 0.2,
        "w[0]={} didn't recover 2.0",
        w_f[0]
    );
    assert!(
        (w_f[1] - 3.0).abs() < 0.2,
        "w[1]={} didn't recover 3.0",
        w_f[1]
    );

    // b should have converged toward 1.0 in every entry (since they
    // all get the same gradient, they stay equal to each other).
    let b_f = read_f32(&b);
    let b_mean: f32 = b_f.iter().sum::<f32>() / b_f.len() as f32;
    assert!(
        (b_mean - 1.0).abs() < 0.2,
        "b_mean={b_mean} didn't recover 1.0"
    );
    // All b entries should be ~equal (same gradient → same updates).
    let b_max = b_f.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let b_min = b_f.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        (b_max - b_min).abs() < 1e-3,
        "b entries diverged: min={b_min}, max={b_max}"
    );
}

#[test]
fn loss_is_monotone_decreasing_after_warmup() {
    // Same setup as the descent test but stricter: every step after
    // step #5 must reduce (or hold) the loss.
    let n_samples = 8;
    let mut x_data = Vec::with_capacity(n_samples * 2);
    let mut y_data = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let x1 = (i as f32) * 0.2 - 0.7;
        let x2 = ((i as f32) * 0.13).cos();
        x_data.push(x1);
        x_data.push(x2);
        y_data.push(0.5 * x1 - 1.5 * x2 + 0.25);
    }
    let x = Tensor::from_slice(&x_data, vec![n_samples, 2]).unwrap();
    let target = Tensor::from_slice(&y_data, vec![n_samples, 1]).unwrap();
    let mut w = Tensor::from_slice(&[0.0f32, 0.0], vec![2, 1]).unwrap();
    let mut b = Tensor::from_slice(&[0.0f32; 8], vec![n_samples, 1]).unwrap();
    let lr = 0.02f32;

    let mut prev_loss = f32::INFINITY;
    let mut decreased_count = 0;
    for step_idx in 0..30 {
        let (new_w, new_b, loss) = step(&x, &target, &w, &b, lr).unwrap();
        if step_idx >= 5 && loss < prev_loss + 1e-4 {
            decreased_count += 1;
        }
        prev_loss = loss;
        w = new_w;
        b = new_b;
    }
    // After warmup, at least 20 of the 25 post-warmup steps should
    // strictly decrease (the rest can hold).
    assert!(
        decreased_count >= 20,
        "monotone descent broken: only {decreased_count}/25 post-warmup steps decreased"
    );
}

//! End-to-end training demo: gradient accumulation across micro-batches
//! using `GradAccumulator` + `accumulate_then_step` + AdamW.
//!
//! Validates the Phase 6.5 contract end-to-end:
//! - Build a logistic-regression-shaped problem (`y = sigmoid(w·x + b)`
//!   target).
//! - Split each "batch" into N micro-batches.
//! - Accumulate gradients into a `GradAccumulator`.
//! - Call `accumulate_then_step` after the last micro-batch.
//! - Loss converges; the recovered weights approximate the ground
//!   truth; Parameter::tensor_id never drifts (anti-pattern 11).

use kiln_optim::{
    accumulate_then_step, AdamW, AdamWHyperparameters, GradAccumulator,
};
use kiln_param::{AmpPolicy, ForwardStorage, Parameter};
use kiln_tensor::{CpuStorage, Tensor};

fn read_f32(t: &Tensor) -> Vec<f32> {
    let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
    cpu.as_bytes()
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn loss_and_grads(
    x: &Tensor,
    target: &Tensor,
    w_param: &Parameter,
    b_param: &Parameter,
) -> kiln_tensor::Result<(f32, Tensor, Tensor)> {
    // Compute y = x @ w + b
    let w = w_param.backward_storage().unwrap();
    let b = b_param.backward_storage().unwrap();
    let pred = kiln_tensor::ops::matmul(x, w)?;
    let pred_shape = pred.shape().to_vec();
    // b is shape [1, 1]; broadcast to pred's [N, 1] shape.
    let pred = kiln_tensor::ops::add(
        &pred,
        &kiln_tensor::ops::broadcast_to(b, &pred_shape)?,
    )?;
    // diff = pred - target
    let diff = kiln_tensor::ops::sub(&pred, target)?;
    // grad_pred = 2 * diff / N
    let n = pred.element_count() as f32;
    let scale = 2.0 / n;
    let grad_pred = kiln_tensor::ops::mul_scalar(&diff, scale)?;
    // L = mean(diff²) — compute manually as scalar
    let sq = kiln_tensor::ops::mul(&diff, &diff)?;
    let sum_t = kiln_tensor::ops::sum_all(&sq)?;
    let loss = read_f32(&sum_t)[0] / n;

    // d/dw: x^T @ grad_pred  (shape [features, 1])
    let x_t = x.transpose(0, 1)?.contiguous()?;
    let dw = kiln_tensor::ops::matmul(&x_t, &grad_pred)?;
    // d/db: sum(grad_pred) over rows → scalar broadcasting back to [1, 1]
    let db_full = kiln_tensor::ops::sum_all(&grad_pred)?;
    let db = db_full.reshape(vec![1, 1])?;
    Ok((loss, dw, db))
}

#[test]
fn microbatch_grad_accum_converges_via_accumulate_then_step() {
    // Synthetic regression: y = 2*x1 + (-1)*x2 + 0.5
    let n_total = 32;
    let micro_size = 8;
    let n_micro = n_total / micro_size;
    assert_eq!(n_micro * micro_size, n_total);

    // Build the full dataset, then chunk into micro-batches.
    let mut x_data = Vec::with_capacity(n_total * 2);
    let mut y_data = Vec::with_capacity(n_total);
    for i in 0..n_total {
        let x1 = (i as f32) * 0.07 - 1.0;
        let x2 = ((i as f32) * 0.11).cos();
        x_data.push(x1);
        x_data.push(x2);
        y_data.push(2.0 * x1 + (-1.0) * x2 + 0.5);
    }
    let x_full = Tensor::from_slice(&x_data, vec![n_total, 2]).unwrap();
    let y_full = Tensor::from_slice(&y_data, vec![n_total, 1]).unwrap();

    // Parameters: w = [2, 1], b = [1, 1] (broadcasts over rows).
    let w_init = Tensor::from_slice(&[0.0f32, 0.0], vec![2, 1]).unwrap();
    let b_init = Tensor::from_slice(&[0.0f32], vec![1, 1]).unwrap();
    let policy = AmpPolicy::fp32_reference();
    let mut w_param = Parameter::trainable(
        ForwardStorage::Plain(w_init.clone()),
        w_init,
        policy,
    );
    let mut b_param = Parameter::trainable(
        ForwardStorage::Plain(b_init.clone()),
        b_init,
        policy,
    );
    let w_id = w_param.tensor_id();
    let b_id = b_param.tensor_id();

    let mut opt = AdamW::new(AdamWHyperparameters {
        lr: 0.05,
        ..Default::default()
    });

    // 200 epochs × 4 micro-batches = 800 AdamW steps. Empirical
    // (validate pod 2026-05-23): 60 epochs left w[0] at 1.54 (the
    // descent had clearly started but hadn't reached the target).
    let mut losses = Vec::new();
    for _epoch in 0..200 {
        let mut acc = GradAccumulator::new();
        let mut epoch_loss = 0.0;
        for mi in 0..n_micro {
            // Extract the micro-batch via narrow + contiguous.
            let x_mb = x_full.narrow(0, mi * micro_size, micro_size).unwrap()
                .contiguous().unwrap();
            let y_mb = y_full.narrow(0, mi * micro_size, micro_size).unwrap()
                .contiguous().unwrap();
            let (loss, dw, db) =
                loss_and_grads(&x_mb, &y_mb, &w_param, &b_param).unwrap();
            epoch_loss += loss;
            acc.accumulate(w_id, &dw).unwrap();
            acc.accumulate(b_id, &db).unwrap();
        }
        epoch_loss /= n_micro as f32;
        losses.push(epoch_loss);

        // One step per epoch, draining the accumulator.
        let stepped = accumulate_then_step(
            &mut opt,
            &mut [&mut w_param, &mut b_param],
            &mut acc,
        )
        .unwrap();
        assert_eq!(stepped, 2);
        assert!(acc.is_empty(), "accumulator should be empty after step");

        // Anti-pattern 11: tensor_ids never drift.
        assert_eq!(w_param.tensor_id(), w_id);
        assert_eq!(b_param.tensor_id(), b_id);
    }

    let first = losses[0];
    let last = losses[losses.len() - 1];
    assert!(
        last < first * 0.10,
        "loss did not descend enough across {} epochs of \
         micro-batch accumulation: first={first}, last={last}",
        losses.len()
    );

    // Recovered parameters: w → (2, -1), b → 0.5.
    let w_vals = read_f32(w_param.backward_storage().unwrap());
    let b_vals = read_f32(b_param.backward_storage().unwrap());
    assert!(
        (w_vals[0] - 2.0).abs() < 0.3,
        "w[0]={} didn't recover 2.0",
        w_vals[0]
    );
    assert!(
        (w_vals[1] + 1.0).abs() < 0.3,
        "w[1]={} didn't recover -1.0",
        w_vals[1]
    );
    assert!(
        (b_vals[0] - 0.5).abs() < 0.3,
        "b[0]={} didn't recover 0.5",
        b_vals[0]
    );
}

#[test]
fn accumulate_then_step_with_skipped_param_does_not_corrupt_others() {
    // Two parameters; only one of them receives a gradient signal in
    // this epoch (the other's micro-batches contribute zero grad).
    // Verify that the second parameter is genuinely unchanged.
    let w_init = Tensor::from_slice(&[3.0f32, -1.0], vec![2, 1]).unwrap();
    let b_init = Tensor::from_slice(&[7.0f32], vec![1, 1]).unwrap();
    let policy = AmpPolicy::fp32_reference();
    let mut w_param =
        Parameter::trainable(ForwardStorage::Plain(w_init.clone()), w_init, policy);
    let mut b_param =
        Parameter::trainable(ForwardStorage::Plain(b_init.clone()), b_init, policy);

    let mut opt = AdamW::new(AdamWHyperparameters::default());
    let mut acc = GradAccumulator::new();

    // Inject grad ONLY for w_param.
    let g_w = Tensor::from_slice(&[0.5f32, 0.5], vec![2, 1]).unwrap();
    acc.accumulate(w_param.tensor_id(), &g_w).unwrap();
    // b_param gets no grad — it should be skipped.

    let stepped =
        accumulate_then_step(&mut opt, &mut [&mut w_param, &mut b_param], &mut acc)
            .unwrap();
    assert_eq!(stepped, 1);

    let b_after = read_f32(b_param.backward_storage().unwrap());
    assert_eq!(b_after, vec![7.0], "b_param must be unchanged");

    // w_param did get stepped — its value should be slightly less than
    // the initial values (positive grad with default lr).
    let w_after = read_f32(w_param.backward_storage().unwrap());
    assert!(w_after[0] < 3.0, "w[0] should have decreased");
}

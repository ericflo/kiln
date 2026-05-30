//! CPU numeric-parity test for the fused "cross-entropy from full logits"
//! tape node's candle-composite backward
//! ([`kiln_model::forward::cross_entropy_from_logits_grad_candle`]).
//!
//! # Why CPU (and why this runs in CI)
//!
//! The kt-tape op `kiln_model::tape_forward::CrossEntropyFromLogitsBackward`
//! (#1082 CP-4 Increment 1) cannot run its `apply` on CPU: its kt↔candle
//! bridge (`kt_tensor_to/from_candle_cuda_copy`) is CUDA-only, and the whole
//! `tape_forward` module is `#![cfg(feature = "cuda")]`. Following the
//! established split (the candle composite `gdn_recurrent_backward_no_grad`
//! lives in the non-gated `forward` module and is CPU-parity-tested; the
//! cuda-gated `GdnRecurrentBackward::apply` just bridges to it), the NUMERICS
//! of this op live in the pure-candle, device-agnostic helper
//! `cross_entropy_from_logits_grad_candle`. The cuda-gated `apply` converts the
//! kt grad → candle, calls this helper, and converts back.
//!
//! This test exercises that helper on CPU with candle's own autograd as the
//! oracle, so it runs in the default-feature CI matrix (no CUDA/GPU). It pins
//! the gradient formula the op depends on: mean reduction (`1/num_active`), the
//! `p - one_hot` per-active-row term scaled by the incoming scalar seed, the
//! `index_select` scatter adjoint, and the trailing zero row for the dropped
//! final logit.

use candle_core::{DType, Device, Tensor, Var, D};

// #1082 type-flip: `kiln_model::forward::cross_entropy_from_logits_grad_candle`
// now takes/returns kt (`kiln_tensor`) tensors even though it is a
// pure-candle-numerics composite under the hood. This CPU parity test keeps the
// candle autograd oracle exactly as-is and builds a parallel kt input from the
// same deterministic data so it can call the now-kt-typed function. The kt
// output is read back to a `Vec<f32>` and compared against the candle oracle,
// preserving the same numeric assertions.
use kiln_tensor::Tensor as KtTensor;

/// Deterministic F32 tensor on `device`, shaped `dims`, filled `base + i*step`.
fn det_f32(device: &Device, dims: &[usize], base: f32, step: f32) -> Tensor {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| base + (i as f32) * step).collect();
    Tensor::from_vec(data, dims.to_vec(), device).expect("det tensor")
}

/// Build a kt CPU tensor with the same deterministic data + shape as `det_f32`.
fn det_f32_kt(dims: &[usize], base: f32, step: f32) -> KtTensor {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| base + (i as f32) * step).collect();
    KtTensor::from_vec(data, dims.to_vec()).expect("det kt tensor")
}

/// Build a kt CPU tensor mirroring a candle tensor's data + shape (CPU only).
fn kt_from_candle_cpu(t: &Tensor) -> KtTensor {
    let dims = t.dims().to_vec();
    let data = t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    KtTensor::from_vec(data, dims).expect("kt from candle cpu")
}

/// kt vs candle comparison: the under-test fn now returns a kt tensor while the
/// oracle stays candle. Extract both to `Vec<f32>` and compare (same metric).
fn max_abs_rel_err_kt_vs_candle(got: &KtTensor, want: &Tensor) -> f32 {
    let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    max_abs_rel_err_slices(&g, &w)
}

fn max_abs_rel_err_kt(got: &KtTensor, want: &KtTensor) -> f32 {
    let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let w = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    max_abs_rel_err_slices(&g, &w)
}

fn max_abs_rel_err_slices(g: &[f32], w: &[f32]) -> f32 {
    assert_eq!(g.len(), w.len(), "shape mismatch got {} want {}", g.len(), w.len());
    let mut max = 0.0f32;
    for (a, b) in g.iter().zip(w.iter()) {
        let denom = b.abs().max(1e-6);
        let rel = (a - b).abs() / denom;
        if rel > max {
            max = rel;
        }
    }
    max
}

/// Replicate `kiln_train::trainer::cross_entropy_loss`'s forward EXACTLY (the
/// candle-autograd oracle): mean over active shifted positions of
/// `log_sum_exp(active_logits) - active_logits[label]`.
fn oracle_loss(logits: &Tensor, input_ids: &[u32], label_mask: &[bool]) -> Tensor {
    let seq_len = input_ids.len();
    let device = logits.device();

    let lg = logits.squeeze(0).unwrap(); // [T, V]
    let shift_logits = lg.narrow(0, 0, seq_len - 1).unwrap(); // [T-1, V]
    let shift_labels: Vec<u32> = input_ids[1..].to_vec();
    let shift_mask: Vec<bool> = label_mask[1..].to_vec();

    let active_positions: Vec<u32> = shift_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    let indices = Tensor::new(active_positions.as_slice(), device).unwrap();
    let active_logits = shift_logits.index_select(&indices, 0).unwrap(); // [A, V]

    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| shift_labels[i as usize])
        .collect();
    let labels_tensor = Tensor::new(active_labels.as_slice(), device)
        .unwrap()
        .to_dtype(DType::U32)
        .unwrap();

    let active_logits_f32 = active_logits.to_dtype(DType::F32).unwrap();
    let log_sum_exp = active_logits_f32.log_sum_exp(D::Minus1).unwrap(); // [A]
    let labels_2d = labels_tensor.unsqueeze(1).unwrap(); // [A, 1]
    let correct_logits = active_logits_f32
        .gather(&labels_2d.to_dtype(DType::U32).unwrap(), 1)
        .unwrap()
        .squeeze(1)
        .unwrap(); // [A]
    let per_token_loss = (log_sum_exp - correct_logits).unwrap();
    per_token_loss.mean_all().unwrap()
}

#[test]
fn cross_entropy_from_logits_grad_matches_candle_autograd() {
    let device = Device::Cpu;
    let (t, v) = (6usize, 8usize);

    // Deterministic logits leaf [1, T, V] as a tracked Var so candle autograd
    // can produce the oracle gradient.
    let logits_data = det_f32(&device, &[1, t, v], -0.3, 0.017);
    let logits_var = Var::from_tensor(&logits_data).expect("var");
    let logits = logits_var.as_tensor();

    // T tokens; label_mask with >=2 active SHIFTED positions. active shifted
    // positions = { i in 0..T-1 : label_mask[i+1] }. With this mask:
    //   label_mask = [F, T, F, T, T, F]
    //   shifted (label_mask[1..]) = [T, F, T, T, F]  -> active i = {0, 2, 3} (3 active)
    let input_ids: Vec<u32> = vec![1, 5, 2, 7, 3, 0];
    let label_mask: Vec<bool> = vec![false, true, false, true, true, false];

    // Oracle: candle autograd of the EXACT forward loss.
    let loss = oracle_loss(logits, &input_ids, &label_mask);
    let grads = loss.backward().expect("backward");
    let oracle_grad = grads
        .get(logits)
        .expect("logits grad present")
        .to_dtype(DType::F32)
        .unwrap(); // [1, T, V]

    // Under test: the candle composite the kt-tape op wraps. grad_scalar = 1.0
    // is the dL/dloss seed (the tape-authoritative backward seeds the loss with
    // ones). #1082: the fn is now kt-typed — feed the same logits as a kt CPU
    // tensor (logits leaf is the same deterministic data as `logits_data`).
    let logits_kt = kt_from_candle_cpu(logits);
    let got_grad = kiln_model::forward::cross_entropy_from_logits_grad_candle(
        &logits_kt,
        &input_ids,
        &label_mask,
        1.0,
    )
    .expect("composite grad");

    assert_eq!(
        got_grad.dims(),
        &[1, t, v],
        "composite grad shape must be [1, T, V]"
    );

    let err = max_abs_rel_err_kt_vs_candle(&got_grad, &oracle_grad);
    assert!(
        err < 1e-4,
        "cross_entropy_from_logits_grad_candle vs candle autograd: max rel err {err} >= 1e-4"
    );

    // The dropped final logit row (lg[T-1]) and inactive shifted rows must be
    // exactly zero in the composite grad (the narrow + index_select adjoints).
    let g = got_grad.squeeze(0).unwrap().to_vec2::<f32>().unwrap(); // [T, V]
    for col in &g[t - 1] {
        assert_eq!(*col, 0.0, "final logit row must be zero (narrow adjoint)");
    }
    // shifted row i corresponds to lg row i; inactive shifted positions
    // (i in 0..T-1 with !label_mask[i+1]) get zero grad. Here i=1 and i=4 are
    // inactive (label_mask[2]=F, label_mask[5]=F).
    for &i in &[1usize, 4usize] {
        for col in &g[i] {
            assert_eq!(*col, 0.0, "inactive shifted row {i} must be zero");
        }
    }
}

/// The scalar seed scales the gradient linearly (the backward folds dL/dloss
/// into the per-row mean gradient). grad(2.0) == 2 * grad(1.0).
#[test]
fn cross_entropy_from_logits_grad_scales_with_seed() {
    let (t, v) = (5usize, 4usize);
    // #1082: fn is kt-typed — build the logits leaf directly as a kt CPU tensor
    // with the same deterministic data as `det_f32`.
    let logits = det_f32_kt(&[1, t, v], 0.1, 0.03);
    let input_ids: Vec<u32> = vec![0, 2, 1, 3, 2];
    let label_mask: Vec<bool> = vec![false, true, true, false, true];

    let g1 = kiln_model::forward::cross_entropy_from_logits_grad_candle(
        &logits,
        &input_ids,
        &label_mask,
        1.0,
    )
    .unwrap();
    let g2 = kiln_model::forward::cross_entropy_from_logits_grad_candle(
        &logits,
        &input_ids,
        &label_mask,
        2.0,
    )
    .unwrap();

    let two_g1 = g1.affine(2.0, 0.0).unwrap();
    let err = max_abs_rel_err_kt(&g2, &two_g1);
    assert!(err < 1e-5, "grad(2.0) != 2*grad(1.0): max rel err {err}");
}

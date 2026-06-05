//! CPU numeric-parity test for the fused "cross-entropy from full logits"
//! tape node's composite backward
//! ([`kiln_model::forward::cross_entropy_from_logits_grad_candle`]).
//!
//! # Why CPU (and why this runs in CI)
//!
//! The kt-tape op `kiln_model::tape_forward::CrossEntropyFromLogitsBackward`
//! (#1082 CP-4 Increment 1) cannot run its `apply` on CPU: its kt↔candle
//! bridge (`kt_tensor_to/from_candle_cuda_copy`) is CUDA-only, and the whole
//! `tape_forward` module is `#![cfg(feature = "cuda")]`. Following the
//! established split (the device-agnostic composite `gdn_recurrent_backward_no_grad`
//! lives in the non-gated `forward` module and is CPU-parity-tested; the
//! cuda-gated `GdnRecurrentBackward::apply` just bridges to it), the NUMERICS
//! of this op live in the pure-kt, device-agnostic helper
//! `cross_entropy_from_logits_grad_candle` (a misnomer kept for stability — it
//! is kt logits in, kt grad out). The cuda-gated `apply` converts the kt grad →
//! candle, calls this helper, and converts back.
//!
//! # Oracle (#1082 candle removal)
//!
//! The original oracle was candle's autograd (`Var` + `.backward()`). After the
//! candle drop this file is candle-free, so the oracle is now a **central
//! finite-difference** gradient of the SAME scalar CE loss, computed in pure
//! host f32: for each logit `i`, `∂L/∂x_i ≈ (L(x + eps·e_i) − L(x − eps·e_i)) /
//! (2·eps)`. The CE loss matches `kiln_train::trainer::cross_entropy_loss`'s
//! forward EXACTLY (mean over active shifted positions of
//! `log_sum_exp(active_logits) − active_logits[label]`), so the FD gradient is
//! the analytic gradient up to O(eps²) truncation. This pins the same formula
//! the op depends on: mean reduction (`1/num_active`), the `p − one_hot`
//! per-active-row term scaled by the incoming scalar seed, the `index_select`
//! scatter adjoint, and the trailing zero row for the dropped final logit.
//!
//! Runs in the default-feature CI matrix (no CUDA/GPU) under plain `cargo test`.

#![cfg(feature = "cuda")]

use kiln_tensor::Tensor as KtTensor;

/// Build a kt CPU tensor, shaped `dims`, filled `base + i*step`.
fn det_f32_kt(dims: &[usize], base: f32, step: f32) -> KtTensor {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| base + (i as f32) * step).collect();
    KtTensor::from_vec(data, dims.to_vec()).expect("det kt tensor")
}

/// kt vs host-f32 comparison: the under-test fn returns a kt tensor; the FD
/// oracle is a host `Vec<f32>`. Extract the kt grad to `Vec<f32>` and compare.
fn max_abs_rel_err_kt_vs_host(got: &KtTensor, want: &[f32]) -> f32 {
    let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    max_abs_rel_err_slices(&g, want)
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

/// Pure-host f32 replica of `kiln_train::trainer::cross_entropy_loss`'s forward
/// (the FD oracle's scalar loss). `logits` is row-major `[1, T, V]` (the leading
/// batch dim is always 1 here). Returns the mean over active shifted positions
/// of `log_sum_exp(active_logits) − active_logits[label]`.
///
/// active shifted positions = `{ i in 0..T-1 : label_mask[i+1] }`; the label for
/// shifted position `i` is `input_ids[i+1]`. The final logit row `lg[T-1]` is
/// dropped (the shift), and inactive shifted rows contribute nothing — the
/// gradient w.r.t. those rows is therefore exactly zero (pinned below).
fn ce_loss_host(logits: &[f32], t: usize, v: usize, input_ids: &[u32], label_mask: &[bool]) -> f64 {
    assert_eq!(logits.len(), t * v);
    let mut sum = 0.0f64;
    let mut active = 0usize;
    // shifted index i in 0..T-1 corresponds to logit row i; its label is
    // input_ids[i+1] and it is active iff label_mask[i+1].
    for i in 0..(t - 1) {
        if !label_mask[i + 1] {
            continue;
        }
        let row = &logits[i * v..(i + 1) * v];
        // numerically stable log-sum-exp in f64.
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut se = 0.0f64;
        for &x in row {
            se += ((x as f64) - m).exp();
        }
        let lse = m + se.ln();
        let label = input_ids[i + 1] as usize;
        let correct = row[label] as f64;
        sum += lse - correct;
        active += 1;
    }
    assert!(active > 0, "ce_loss_host: need >=1 active shifted position");
    sum / (active as f64)
}

/// Central finite-difference gradient of `ce_loss_host` w.r.t. every logit,
/// scaled by `grad_scalar` (the dL/dloss seed). Returns a `[1, T, V]`-flat
/// `Vec<f32>`. `eps = 1e-3` (f32) per the #1082 spec; the resulting band is
/// O(eps²) ≈ 1e-6 truncation on top of f32 readback noise, which the loosened
/// `< 2e-3` tolerance below absorbs.
fn fd_grad_host(
    logits: &[f32],
    t: usize,
    v: usize,
    input_ids: &[u32],
    label_mask: &[bool],
    grad_scalar: f32,
) -> Vec<f32> {
    let eps = 1e-3f32;
    let mut grad = vec![0.0f32; t * v];
    // Only active shifted rows have nonzero gradient; perturbing any other
    // logit leaves the loss unchanged, so the FD there is exactly 0.0 (which is
    // also the analytic answer — pinned by the zero-row asserts in the tests).
    let mut buf = logits.to_vec();
    for idx in 0..(t * v) {
        let orig = buf[idx];
        buf[idx] = orig + eps;
        let lp = ce_loss_host(&buf, t, v, input_ids, label_mask);
        buf[idx] = orig - eps;
        let lm = ce_loss_host(&buf, t, v, input_ids, label_mask);
        buf[idx] = orig;
        let d = ((lp - lm) / (2.0 * eps as f64)) as f32;
        grad[idx] = d * grad_scalar;
    }
    grad
}

#[test]
fn cross_entropy_from_logits_grad_matches_finite_difference() {
    let (t, v) = (6usize, 8usize);

    // Deterministic logits leaf [1, T, V] (same data the candle oracle used).
    let logits_data: Vec<f32> = (0..(t * v)).map(|i| -0.3 + (i as f32) * 0.017).collect();

    // T tokens; label_mask with >=2 active SHIFTED positions. active shifted
    // positions = { i in 0..T-1 : label_mask[i+1] }. With this mask:
    //   label_mask = [F, T, F, T, T, F]
    //   shifted (label_mask[1..]) = [T, F, T, T, F]  -> active i = {0, 2, 3} (3 active)
    let input_ids: Vec<u32> = vec![1, 5, 2, 7, 3, 0];
    let label_mask: Vec<bool> = vec![false, true, false, true, true, false];

    // Oracle: central finite-difference of the EXACT forward CE loss.
    // grad_scalar = 1.0 is the dL/dloss seed (the tape-authoritative backward
    // seeds the loss with ones).
    let oracle_grad = fd_grad_host(&logits_data, t, v, &input_ids, &label_mask, 1.0);

    // Under test: the kt composite the kt-tape op wraps. Feed the same logits as
    // a kt CPU tensor (#1082 — the fn is kt-typed).
    let logits_kt = KtTensor::from_vec(logits_data, vec![1, t, v]).expect("logits kt");
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

    let err = max_abs_rel_err_kt_vs_host(&got_grad, &oracle_grad);
    assert!(
        err < 2e-3,
        "cross_entropy_from_logits_grad_candle vs finite-difference: max rel err {err} >= 2e-3 \
         (FD truncation band, eps=1e-3)"
    );

    // The dropped final logit row (lg[T-1]) and inactive shifted rows must be
    // exactly zero in the composite grad (the narrow + index_select adjoints).
    let g_flat = got_grad.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let g: Vec<&[f32]> = (0..t).map(|r| &g_flat[r * v..(r + 1) * v]).collect(); // [T, V]
    for col in g[t - 1] {
        assert_eq!(*col, 0.0, "final logit row must be zero (narrow adjoint)");
    }
    // shifted row i corresponds to lg row i; inactive shifted positions
    // (i in 0..T-1 with !label_mask[i+1]) get zero grad. Here i=1 and i=4 are
    // inactive (label_mask[2]=F, label_mask[5]=F).
    for &i in &[1usize, 4usize] {
        for col in g[i] {
            assert_eq!(*col, 0.0, "inactive shifted row {i} must be zero");
        }
    }
}

/// The scalar seed scales the gradient linearly (the backward folds dL/dloss
/// into the per-row mean gradient). grad(2.0) == 2 * grad(1.0).
#[test]
fn cross_entropy_from_logits_grad_scales_with_seed() {
    let (t, v) = (5usize, 4usize);
    // #1082: fn is kt-typed — build the logits leaf directly as a kt CPU tensor.
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

#[test]
fn cross_entropy_from_logits_grad_cuda_matches_cpu() {
    if !kiln_tensor::probe::cuda_is_available() {
        eprintln!("[CE-CUDA] no CUDA device; skipping");
        return;
    }

    let (t, v) = (6usize, 11usize);
    let logits_data: Vec<f32> = (0..(t * v))
        .map(|i| ((i as f32 + 2.0) * 0.021).sin() * 0.7)
        .collect();
    let input_ids: Vec<u32> = vec![1, 5, 2, 7, 3, 0];
    let label_mask: Vec<bool> = vec![false, true, false, true, true, false];
    let cpu_logits = KtTensor::from_vec(logits_data.clone(), vec![1, t, v]).expect("cpu logits");
    let cuda_logits = KtTensor::from_vec_on(
        kiln_tensor::Device::Cuda(0),
        logits_data,
        vec![1, t, v],
    )
    .expect("cuda logits");

    let cpu_grad = kiln_model::forward::cross_entropy_from_logits_grad_candle(
        &cpu_logits,
        &input_ids,
        &label_mask,
        1.0,
    )
    .expect("cpu composite grad");
    let cuda_grad = kiln_model::forward::cross_entropy_from_logits_grad_candle(
        &cuda_logits,
        &input_ids,
        &label_mask,
        1.0,
    )
    .expect("cuda composite grad");

    let err = max_abs_rel_err_kt(&cuda_grad, &cpu_grad);
    assert!(err < 1e-5, "cuda CE grad != cpu grad: max rel err {err}");
}

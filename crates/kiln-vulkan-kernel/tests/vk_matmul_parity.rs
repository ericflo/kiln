//! Phase B parity tests: vk_matmul forward + backward vs an analytical
//! reference. Fully candle-free: the LoRA composition backward in
//! `vk_lora_style_composition_backward_parity` is checked against a
//! finite-difference numerical gradient of the same scalar loss,
//! replacing the former candle Var-based oracle. (#1082)
//!
//! Test factories are candle-free via the kt-native
//! `VkTensor::from_f32_slice` / `from_f32_slice_as_bf16` /
//! `parameter_from_f32_slice` constructors. (#1082)
//!
//! Also covers the LoRA-style composition `delta = (x @ A.T) @ B.T`
//! since it's the Phase B canonical use case.
//! Normal developer runs skip if no Vulkan device is available; runs with
//! `KILN_QUALIFICATION=1` fail on an unavailable or uninitializable device.

use anyhow::Result;
use half::bf16;
use kiln_vulkan_kernel::VulkanDevice;
use kiln_vulkan_kernel::vk_autograd::vk_backward;
use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
use kiln_vulkan_kernel::vk_ops::matmul::{
    vk_matmul, vk_matmul_lhs_t_no_grad, vk_matmul_no_grad, vk_matmul_rhs_t_no_grad,
};
use kiln_vulkan_kernel::vk_ops::matmul_batched::{
    vk_matmul_lhs_t_batched_bf16_no_grad, vk_matmul_rhs_t_batched_bf16_no_grad,
};
use kiln_vulkan_kernel::vk_ops::matmul_bf16w::vk_matmul_bf16w;
use kiln_vulkan_kernel::vk_ops::reduce::vk_mean_all;
use kiln_vulkan_kernel::vk_ops::shape::vk_transpose_2d_no_grad;
use kiln_vulkan_kernel::vk_tensor::VkTensor;
use std::sync::Arc;

fn qualification_required(value: Option<&str>) -> bool {
    value == Some("1")
}

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        if qualification_required(std::env::var("KILN_QUALIFICATION").ok().as_deref()) {
            panic!("Vulkan device unavailable while KILN_QUALIFICATION=1");
        }
        eprintln!("no Vulkan device available; skipping matmul parity test");
        return None;
    }
    match VulkanDevice::new() {
        Ok(device) => Some(Arc::new(device)),
        Err(error) => {
            if qualification_required(std::env::var("KILN_QUALIFICATION").ok().as_deref()) {
                panic!("Vulkan device initialization failed while KILN_QUALIFICATION=1: {error:#}");
            }
            eprintln!(
                "Vulkan device initialization failed; skipping matmul parity test: {error:#}"
            );
            None
        }
    }
}

fn upload_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    VkTensor::from_f32_slice(data, shape.to_vec(), Arc::clone(dev))
}

fn upload_bf16(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    VkTensor::from_f32_slice_as_bf16(data, shape.to_vec(), Arc::clone(dev))
}

fn upload_param_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    VkTensor::parameter_from_f32_slice(data, shape.to_vec(), Arc::clone(dev))
}

fn max_abs_diff(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len(), "len mismatch");
    got.iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max)
}

fn naive_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0_f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ki * n + ni];
            }
            c[mi * n + ni] = acc;
        }
    }
    c
}

fn naive_lhs_t_matmul(a: &[f32], b: &[f32], k: usize, m: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0_f32;
            for ki in 0..k {
                acc += a[ki * m + mi] * b[ki * n + ni];
            }
            c[mi * n + ni] = acc;
        }
    }
    c
}

fn naive_rhs_t_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut acc = 0.0_f32;
            for ki in 0..k {
                acc += a[mi * k + ki] * b[ni * k + ki];
            }
            c[mi * n + ni] = acc;
        }
    }
    c
}

/// Scalar loss of the LoRA-style composition as a pure function of its
/// three inputs: loss = mean( ((x @ A.T) @ B.T) * scale ).
///   x: [batch, in_features], A: [rank, in_features], B: [out_features, rank]
/// Used as the inner function for finite-difference gradients (candle-free).
#[allow(clippy::too_many_arguments)]
fn lora_scalar_loss(
    x: &[f32],
    a: &[f32],
    b: &[f32],
    scale: f32,
    batch: usize,
    in_features: usize,
    rank: usize,
    out_features: usize,
) -> f32 {
    // A.T : [in_features, rank]
    let mut a_t = vec![0.0_f32; in_features * rank];
    for r in 0..rank {
        for c in 0..in_features {
            a_t[c * rank + r] = a[r * in_features + c];
        }
    }
    // B.T : [rank, out_features]
    let mut b_t = vec![0.0_f32; rank * out_features];
    for r in 0..out_features {
        for c in 0..rank {
            b_t[c * out_features + r] = b[r * rank + c];
        }
    }
    let h = naive_matmul(x, &a_t, batch, rank, in_features);
    let mm = naive_matmul(&h, &b_t, batch, out_features, rank);
    let n_total = (batch * out_features) as f32;
    mm.iter().map(|v| v * scale).sum::<f32>() / n_total
}

/// Central finite-difference gradient of `loss(param)` w.r.t. each entry of
/// `param`, where `loss` is the closure recomputing the scalar loss after
/// mutating `param` in place. eps ~ 1e-3 for f32.
fn fd_grad(param: &mut [f32], eps: f32, mut loss: impl FnMut(&[f32]) -> f32) -> Vec<f32> {
    let mut grad = vec![0.0_f32; param.len()];
    for i in 0..param.len() {
        let orig = param[i];
        param[i] = orig + eps;
        let lp = loss(param);
        param[i] = orig - eps;
        let lm = loss(param);
        param[i] = orig;
        grad[i] = (lp - lm) / (2.0 * eps);
    }
    grad
}

#[test]
fn vk_matmul_bf16w_canonical_weight_forward_and_dx() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 3;
    let hidden = 5;
    let out_dim = 4;
    let x_data: Vec<f32> = (0..(batch * hidden))
        .map(|i| ((i as f32) * 0.17).sin())
        .collect();
    // Canonical frozen projection layout: [out_dim, hidden].
    let w_data: Vec<f32> = (0..(out_dim * hidden))
        .map(|i| ((i as f32) * 0.11).cos() * 0.5)
        .collect();
    let w_bf16: Vec<f32> = w_data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();

    let x = upload_param_f32(&dev, &x_data, &[batch, hidden])?;
    let w = upload_bf16(&dev, &w_data, &[out_dim, hidden])?;
    let out = vk_matmul_bf16w(&x, &w)?;
    assert_eq!(out.shape(), &[batch, out_dim]);

    let mut w_t = vec![0.0_f32; hidden * out_dim];
    for o in 0..out_dim {
        for h in 0..hidden {
            w_t[h * out_dim + o] = w_bf16[o * hidden + h];
        }
    }
    let expected = naive_matmul(&x_data, &w_t, batch, out_dim, hidden);
    let got = out.to_vec_f32()?;
    let mad = max_abs_diff(&got, &expected);
    assert!(mad < 2e-3, "bf16w forward max diff {mad}");

    let loss = vk_mean_all(&out)?;
    let grads = vk_backward(&loss)?;
    let grad_x = grads.get(x.param_id().unwrap()).expect("dx").to_vec_f32()?;
    let d_out = vec![1.0_f32 / (batch * out_dim) as f32; batch * out_dim];
    let expected_dx = naive_matmul(&d_out, &w_bf16, batch, hidden, out_dim);
    let mad_dx = max_abs_diff(&grad_x, &expected_dx);
    assert!(mad_dx < 2e-3, "bf16w dx max diff {mad_dx}");
    Ok(())
}

#[test]
fn vk_matmul_forward_small() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    // [3, 4] @ [4, 2] → [3, 2]
    let a_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.5).collect();
    let b_data: Vec<f32> = (0..8).map(|i| (i as f32) - 3.0).collect();
    let a = upload_f32(&dev, &a_data, &[3, 4])?;
    let b = upload_f32(&dev, &b_data, &[4, 2])?;
    let c = vk_matmul_no_grad(&a, &b)?;
    assert_eq!(c.shape(), &[3, 2]);
    let got = c.to_vec_f32()?;
    let expected = naive_matmul(&a_data, &b_data, 3, 2, 4);
    let mad = max_abs_diff(&got, &expected);
    assert!(
        mad < 1e-5,
        "max abs diff {mad}; got {got:?} vs {expected:?}"
    );
    Ok(())
}

#[test]
fn vk_matmul_forward_tile_boundary() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    // Shapes intentionally non-aligned to 16 to exercise tile masking.
    // [17, 33] @ [33, 19] → [17, 19]
    let m = 17;
    let k = 33;
    let n = 19;
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.013).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.027).cos()).collect();
    let a = upload_f32(&dev, &a_data, &[m, k])?;
    let b = upload_f32(&dev, &b_data, &[k, n])?;
    let c = vk_matmul_no_grad(&a, &b)?;
    let got = c.to_vec_f32()?;
    let expected = naive_matmul(&a_data, &b_data, m, n, k);
    let mad = max_abs_diff(&got, &expected);
    // K=33 sums of small products; F32 should be ~1e-5 max
    assert!(mad < 1e-4, "max abs diff {mad}");
    Ok(())
}

#[test]
fn vk_matmul_lhs_t_forward_tile_boundary() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    // A [K, M]^T @ B [K, N] -> [M, N], with non-16-aligned dimensions.
    let k = 33;
    let m = 17;
    let n = 19;
    let a_data: Vec<f32> = (0..(k * m)).map(|i| ((i as f32) * 0.013).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.027).cos()).collect();
    let a = upload_f32(&dev, &a_data, &[k, m])?;
    let b = upload_f32(&dev, &b_data, &[k, n])?;
    let c = vk_matmul_lhs_t_no_grad(&a, &b)?;
    assert_eq!(c.shape(), &[m, n]);
    let got = c.to_vec_f32()?;
    let expected = naive_lhs_t_matmul(&a_data, &b_data, k, m, n);
    let mad = max_abs_diff(&got, &expected);
    assert!(mad < 1e-4, "lhs_t max abs diff {mad}");
    Ok(())
}

#[test]
fn vk_matmul_rhs_t_forward_tile_boundary() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    // A [M, K] @ B [N, K]^T -> [M, N], with non-16-aligned dimensions.
    let m = 17;
    let k = 33;
    let n = 19;
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.013).sin()).collect();
    let b_data: Vec<f32> = (0..(n * k)).map(|i| ((i as f32) * 0.027).cos()).collect();
    let a = upload_f32(&dev, &a_data, &[m, k])?;
    let b = upload_f32(&dev, &b_data, &[n, k])?;
    let c = vk_matmul_rhs_t_no_grad(&a, &b)?;
    assert_eq!(c.shape(), &[m, n]);
    let got = c.to_vec_f32()?;
    let expected = naive_rhs_t_matmul(&a_data, &b_data, m, k, n);
    let mad = max_abs_diff(&got, &expected);
    assert!(mad < 1e-4, "rhs_t max abs diff {mad}");
    Ok(())
}

#[test]
fn vk_matmul_lhs_t_batched_bf16_forward() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let (batch, k, m, n) = (3usize, 21usize, 7usize, 11usize);
    let a_data: Vec<f32> = (0..(batch * k * m))
        .map(|i| ((i as f32) * 0.031).sin())
        .collect();
    let b_data: Vec<f32> = (0..(batch * k * n))
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();
    let a = upload_bf16(&dev, &a_data, &[batch, k, m])?;
    let b = upload_bf16(&dev, &b_data, &[batch, k, n])?;
    let c = vk_matmul_lhs_t_batched_bf16_no_grad(&a, &b)?;
    assert_eq!(c.shape(), &[batch, m, n]);
    let got = c.to_vec_f32()?;

    let a_rounded: Vec<f32> = a_data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let b_rounded: Vec<f32> = b_data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let mut expected = Vec::with_capacity(batch * m * n);
    for bx in 0..batch {
        let a0 = bx * k * m;
        let b0 = bx * k * n;
        expected.extend(naive_lhs_t_matmul(
            &a_rounded[a0..a0 + k * m],
            &b_rounded[b0..b0 + k * n],
            k,
            m,
            n,
        ));
    }
    let mad = max_abs_diff(&got, &expected);
    assert!(mad < 2e-2, "lhs_t bf16 batched max abs diff {mad}");
    Ok(())
}

#[test]
fn vk_matmul_rhs_t_batched_bf16_forward() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let (batch, m, k, n) = (3usize, 7usize, 21usize, 11usize);
    let a_data: Vec<f32> = (0..(batch * m * k))
        .map(|i| ((i as f32) * 0.031).sin())
        .collect();
    let b_data: Vec<f32> = (0..(batch * n * k))
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();
    let a = upload_bf16(&dev, &a_data, &[batch, m, k])?;
    let b = upload_bf16(&dev, &b_data, &[batch, n, k])?;
    let c = vk_matmul_rhs_t_batched_bf16_no_grad(&a, &b)?;
    assert_eq!(c.shape(), &[batch, m, n]);
    let got = c.to_vec_f32()?;

    let a_rounded: Vec<f32> = a_data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let b_rounded: Vec<f32> = b_data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let mut expected = Vec::with_capacity(batch * m * n);
    for bx in 0..batch {
        let a0 = bx * m * k;
        let b0 = bx * n * k;
        expected.extend(naive_rhs_t_matmul(
            &a_rounded[a0..a0 + m * k],
            &b_rounded[b0..b0 + n * k],
            m,
            k,
            n,
        ));
    }
    let mad = max_abs_diff(&got, &expected);
    assert!(mad < 2e-2, "rhs_t bf16 batched max abs diff {mad}");
    Ok(())
}

/// Backward parity: loss = mean(A @ B), then check dA and dB analytically.
/// For loss = mean(A @ B) with shape [M, N] and N_total = M*N:
///   d loss / d C[i, j] = 1 / N_total
///   d loss / d A = (1/N_total) * ones_C @ B.T
///   d loss / d B = (1/N_total) * A.T @ ones_C
#[test]
fn vk_matmul_backward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let m = 4;
    let k = 5;
    let n = 3;
    let a_data: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.2 - 0.5).collect();
    let a = upload_param_f32(&dev, &a_data, &[m, k])?;
    let b = upload_param_f32(&dev, &b_data, &[k, n])?;
    let c = vk_matmul(&a, &b)?;
    let loss = vk_mean_all(&c)?;
    let grads = vk_backward(&loss)?;
    let grad_a = grads.get(a.param_id().unwrap()).expect("dA").to_vec_f32()?;
    let grad_b = grads.get(b.param_id().unwrap()).expect("dB").to_vec_f32()?;

    let n_total = (m * n) as f32;
    let dc = vec![1.0_f32 / n_total; m * n];
    // dA = dC @ B.T  → [M, K]; B.T has shape [N, K]
    let mut b_t = vec![0.0; n * k];
    for ki in 0..k {
        for ni in 0..n {
            b_t[ni * k + ki] = b_data[ki * n + ni];
        }
    }
    let exp_da = naive_matmul(&dc, &b_t, m, k, n);
    // dB = A.T @ dC  → [K, N]; A.T has shape [K, M]
    let mut a_t = vec![0.0; k * m];
    for mi in 0..m {
        for ki in 0..k {
            a_t[ki * m + mi] = a_data[mi * k + ki];
        }
    }
    let exp_db = naive_matmul(&a_t, &dc, k, n, m);

    let mad_a = max_abs_diff(&grad_a, &exp_da);
    let mad_b = max_abs_diff(&grad_b, &exp_db);
    assert!(mad_a < 1e-5, "dA max diff {mad_a}");
    assert!(mad_b < 1e-5, "dB max diff {mad_b}");
    Ok(())
}

/// LoRA-style composition: delta = (x @ A.T) @ B.T, scaled by `scale`.
/// All three (x, A, B) are parameters; verify all three grads via the
/// finite-difference reference.
///
/// Shapes: x: [batch, in_features], A: [rank, in_features], B: [out, rank]
/// delta: [batch, out]
#[test]
fn vk_lora_style_composition_backward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 6;
    let in_features = 8;
    let rank = 3;
    let out_features = 5;
    let scale = 0.5_f32;

    let x_data: Vec<f32> = (0..(batch * in_features))
        .map(|i| (i as f32) * 0.05 - 0.2)
        .collect();
    let a_data: Vec<f32> = (0..(rank * in_features))
        .map(|i| ((i as f32) * 0.073).sin())
        .collect();
    let b_data: Vec<f32> = (0..(out_features * rank))
        .map(|i| ((i as f32) * 0.029).cos() * 0.3)
        .collect();

    let x = upload_param_f32(&dev, &x_data, &[batch, in_features])?;
    let a_mat = upload_param_f32(&dev, &a_data, &[rank, in_features])?;
    let b_mat = upload_param_f32(&dev, &b_data, &[out_features, rank])?;

    // h = x @ A.T     (batch, rank)
    let a_t = vk_transpose_2d_no_grad(&a_mat)?;
    // Need grad through A → can't use no_grad here. Wire transposes with grad.
    // For Phase B, easiest: have a_mat as parameter, transpose with autograd:
    use kiln_vulkan_kernel::vk_ops::shape::vk_transpose_2d;
    let _ = a_t; // discard the no-grad version
    let a_t = vk_transpose_2d(&a_mat)?;
    let h = vk_matmul(&x, &a_t)?;
    // delta = h @ B.T  (batch, out)
    let b_t = vk_transpose_2d(&b_mat)?;
    let mm = vk_matmul(&h, &b_t)?;
    // scale via element-wise multiply by a same-shape constant tensor of `scale`.
    let scale_data: Vec<f32> = vec![scale; batch * out_features];
    let scale_t = upload_f32(&dev, &scale_data, &[batch, out_features])?;
    let delta = vk_mul(&mm, &scale_t)?;
    let loss = vk_mean_all(&delta)?;
    let grads = vk_backward(&loss)?;
    let grad_x = grads.get(x.param_id().unwrap()).expect("dx").to_vec_f32()?;
    let grad_a = grads
        .get(a_mat.param_id().unwrap())
        .expect("dA")
        .to_vec_f32()?;
    let grad_b = grads
        .get(b_mat.param_id().unwrap())
        .expect("dB")
        .to_vec_f32()?;

    // Reference via naive CPU.
    let mut a_t_data = vec![0.0_f32; in_features * rank];
    for r in 0..rank {
        for c in 0..in_features {
            a_t_data[c * rank + r] = a_data[r * in_features + c];
        }
    }
    let mut b_t_data = vec![0.0_f32; rank * out_features];
    for r in 0..out_features {
        for c in 0..rank {
            b_t_data[c * out_features + r] = b_data[r * rank + c];
        }
    }
    let h_data = naive_matmul(&x_data, &a_t_data, batch, rank, in_features);
    let mm_data = naive_matmul(&h_data, &b_t_data, batch, out_features, rank);
    let delta_data: Vec<f32> = mm_data.iter().map(|v| v * scale).collect();
    let n_total = (batch * out_features) as f32;
    let loss_val: f32 = delta_data.iter().sum::<f32>() / n_total;
    let got_loss = loss.to_vec_f32()?[0];
    assert!(
        (got_loss - loss_val).abs() < 1e-4,
        "loss {} vs {}",
        got_loss,
        loss_val
    );

    // d loss / d delta = scale / n_total  (since loss = mean(delta) ... wait,
    // delta = mm * scale, mean(delta) = scale * mean(mm)). Actually
    // d loss / d mm[i,j] = scale / n_total. Then propagate:
    //   d loss / d h = (scale/n_total * ones) @ B
    //   d loss / d B = h.T @ (scale/n_total * ones)
    //   d loss / d x = (d loss / d h) @ A
    //   d loss / d A = h_for_a path: d loss / d A = (d loss / d h).T @ x ... etc.
    //
    // The closed-form reference here is intricate, so instead of hand-coding
    // each path we cross-check the Vulkan backward against a central
    // finite-difference numerical gradient of the same scalar loss
    // `loss = mean( ((x @ A.T) @ B.T) * scale )`. This replaces the former
    // candle Var-based autograd oracle, leaving the file candle-free. (#1082)
    let eps = 1e-3_f32;
    let mut x_pert = x_data.clone();
    let exp_dx = fd_grad(&mut x_pert, eps, |xp| {
        lora_scalar_loss(
            xp,
            &a_data,
            &b_data,
            scale,
            batch,
            in_features,
            rank,
            out_features,
        )
    });
    let mut a_pert = a_data.clone();
    let exp_da = fd_grad(&mut a_pert, eps, |ap| {
        lora_scalar_loss(
            &x_data,
            ap,
            &b_data,
            scale,
            batch,
            in_features,
            rank,
            out_features,
        )
    });
    let mut b_pert = b_data.clone();
    let exp_db = fd_grad(&mut b_pert, eps, |bp| {
        lora_scalar_loss(
            &x_data,
            &a_data,
            bp,
            scale,
            batch,
            in_features,
            rank,
            out_features,
        )
    });

    // Tolerances loosened from 1e-4 to 2e-3 to absorb finite-difference
    // truncation/rounding error; the loss is smooth (pure matmuls), so the
    // central difference is accurate well within this band.
    let mad_x = max_abs_diff(&grad_x, &exp_dx);
    let mad_a = max_abs_diff(&grad_a, &exp_da);
    let mad_b = max_abs_diff(&grad_b, &exp_db);
    assert!(mad_x < 2e-3, "dx mad {mad_x}");
    assert!(mad_a < 2e-3, "dA mad {mad_a}");
    assert!(mad_b < 2e-3, "dB mad {mad_b}");
    Ok(())
}

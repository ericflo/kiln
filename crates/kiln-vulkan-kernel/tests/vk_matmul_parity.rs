//! Phase B parity tests: vk_matmul forward + backward vs candle.
//!
//! Also covers the LoRA-style composition `delta = (x @ A.T) @ B.T`
//! since it's the Phase B canonical use case.

use anyhow::Result;
use candle_core::{Device, Tensor};
use kiln_vulkan_kernel::vk_autograd::vk_backward;
use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
use kiln_vulkan_kernel::vk_ops::matmul::{vk_matmul, vk_matmul_no_grad};
use kiln_vulkan_kernel::vk_ops::reduce::vk_mean_all;
use kiln_vulkan_kernel::vk_ops::shape::vk_transpose_2d_no_grad;
use kiln_vulkan_kernel::vk_tensor::VkTensor;
use kiln_vulkan_kernel::VulkanDevice;
use std::sync::Arc;

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

fn upload_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    VkTensor::from_candle(&t, Arc::clone(dev))
}

fn upload_param_f32(
    dev: &Arc<VulkanDevice>,
    data: &[f32],
    shape: &[usize],
) -> Result<(candle_core::Var, VkTensor)> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    let var = candle_core::Var::from_tensor(&t)?;
    let vk = VkTensor::from_candle(&t, Arc::clone(dev))?;
    let pid = var.id();
    let param = VkTensor::parameter(
        Arc::clone(vk.buffer()),
        vk.shape().to_vec(),
        vk.dtype(),
        Arc::clone(vk.device()),
        pid,
    );
    Ok((var, param))
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
    assert!(mad < 1e-5, "max abs diff {mad}; got {got:?} vs {expected:?}");
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
    let a_data: Vec<f32> = (0..(m * k))
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let b_data: Vec<f32> = (0..(k * n))
        .map(|i| ((i as f32) * 0.027).cos())
        .collect();
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
    let (_a_var, a) = upload_param_f32(&dev, &a_data, &[m, k])?;
    let (_b_var, b) = upload_param_f32(&dev, &b_data, &[k, n])?;
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
/// All three (x, A, B) are parameters; verify all three grads via
/// candle reference.
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

    let (_x_var, x) = upload_param_f32(&dev, &x_data, &[batch, in_features])?;
    let (_a_var, a_mat) = upload_param_f32(&dev, &a_data, &[rank, in_features])?;
    let (_b_var, b_mat) = upload_param_f32(&dev, &b_data, &[out_features, rank])?;

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
    // The candle-free reference here is intricate; use a fresh candle
    // path for the reference instead.
    use candle_core::{DType, Var};
    let dev_c = Device::Cpu;
    let xv = Var::from_tensor(&Tensor::from_vec(
        x_data.clone(),
        (batch, in_features),
        &dev_c,
    )?)?;
    let av = Var::from_tensor(&Tensor::from_vec(
        a_data.clone(),
        (rank, in_features),
        &dev_c,
    )?)?;
    let bv = Var::from_tensor(&Tensor::from_vec(
        b_data.clone(),
        (out_features, rank),
        &dev_c,
    )?)?;
    let x_c = xv.as_tensor();
    let a_c = av.as_tensor();
    let b_c = bv.as_tensor();
    let h_c = x_c.matmul(&a_c.transpose(0, 1)?)?;
    let delta_c = h_c.matmul(&b_c.transpose(0, 1)?)?;
    let delta_c = (delta_c * (scale as f64))?;
    let loss_c = delta_c.mean_all()?.to_dtype(DType::F32)?;
    let grads_c = loss_c.backward()?;
    let exp_dx = grads_c.get(x_c).unwrap().flatten_all()?.to_vec1::<f32>()?;
    let exp_da = grads_c.get(a_c).unwrap().flatten_all()?.to_vec1::<f32>()?;
    let exp_db = grads_c.get(b_c).unwrap().flatten_all()?.to_vec1::<f32>()?;

    let mad_x = max_abs_diff(&grad_x, &exp_dx);
    let mad_a = max_abs_diff(&grad_a, &exp_da);
    let mad_b = max_abs_diff(&grad_b, &exp_db);
    assert!(mad_x < 1e-4, "dx mad {mad_x}");
    assert!(mad_a < 1e-4, "dA mad {mad_a}");
    assert!(mad_b < 1e-4, "dB mad {mad_b}");
    Ok(())
}

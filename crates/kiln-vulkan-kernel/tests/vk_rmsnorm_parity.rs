//! Phase B.2: vk_rmsnorm forward + backward parity vs analytic
//! reference (the existing CustomOp wrapping the same kernels is
//! already battle-tested in the inference and SFT paths; here we
//! verify VkTensor wiring against the analytic Qwen3.5 RMSNorm
//! definition + numeric finite-difference for backward).

use anyhow::Result;
use candle_core::{Device, Tensor};
use kiln_vulkan_kernel::vk_autograd::vk_backward;
use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
use kiln_vulkan_kernel::vk_ops::reduce::vk_mean_all;
use kiln_vulkan_kernel::vk_ops::rmsnorm::{vk_rmsnorm, vk_rmsnorm_no_grad};
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

/// CPU reference: out[i, j] = (1 + w[j]) * x[i, j] / sqrt(mean(x[i, :]^2) + eps)
fn cpu_rmsnorm(x: &[f32], w: &[f32], rows: usize, hidden: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; rows * hidden];
    for r in 0..rows {
        let mut sq = 0.0;
        for c in 0..hidden {
            let v = x[r * hidden + c];
            sq += v * v;
        }
        let s_inv = 1.0 / ((sq / hidden as f32) + eps).sqrt();
        for c in 0..hidden {
            out[r * hidden + c] = (1.0 + w[c]) * x[r * hidden + c] * s_inv;
        }
    }
    out
}

#[test]
fn vk_rmsnorm_forward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 4;
    let hidden = 32;
    let x_data: Vec<f32> = (0..(rows * hidden))
        .map(|i| ((i as f32) * 0.013).sin() * 1.5)
        .collect();
    let w_data: Vec<f32> = (0..hidden).map(|i| ((i as f32) * 0.05).cos() * 0.1).collect();
    let eps = 1e-5_f32;
    let x = upload_f32(&dev, &x_data, &[rows, hidden])?;
    let w = upload_f32(&dev, &w_data, &[hidden])?;
    let y = vk_rmsnorm_no_grad(&x, &w, eps)?;
    assert_eq!(y.shape(), &[rows, hidden]);
    let got = y.to_vec_f32()?;
    let expected = cpu_rmsnorm(&x_data, &w_data, rows, hidden, eps);
    let mad = max_abs_diff(&got, &expected);
    assert!(mad < 1e-5, "max abs diff {mad}");
    Ok(())
}

/// Backward parity: loss = mean(rmsnorm(x, w) ^ 2), check dx via
/// finite-difference (weight is intentionally frozen — kernel does
/// not return dw).
#[test]
fn vk_rmsnorm_backward_dx_finite_diff() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 2;
    let hidden = 16;
    let x_data: Vec<f32> = (0..(rows * hidden))
        .map(|i| ((i as f32) * 0.07).cos() * 0.9 + 0.1)
        .collect();
    let w_data: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.13).sin() * 0.15)
        .collect();
    let eps = 1e-5_f32;

    let (_x_var, x_param) = upload_param_f32(&dev, &x_data, &[rows, hidden])?;
    let w = upload_f32(&dev, &w_data, &[hidden])?;
    let y = vk_rmsnorm(&x_param, &w, eps)?;
    let sq = vk_mul(&y, &y)?;
    let loss = vk_mean_all(&sq)?;
    let grads = vk_backward(&loss)?;
    let grad_x = grads
        .get(x_param.param_id().unwrap())
        .expect("dx")
        .to_vec_f32()?;

    // Finite-difference check at a handful of indices.
    let h = 1e-3_f32;
    let n_total = (rows * hidden) as f32;
    let base_loss = {
        let y = cpu_rmsnorm(&x_data, &w_data, rows, hidden, eps);
        y.iter().map(|v| v * v).sum::<f32>() / n_total
    };
    let mut max_diff = 0.0_f32;
    for idx in [0, 1, 7, 15, 16, 31] {
        let mut x_plus = x_data.clone();
        x_plus[idx] += h;
        let y_plus = cpu_rmsnorm(&x_plus, &w_data, rows, hidden, eps);
        let loss_plus: f32 = y_plus.iter().map(|v| v * v).sum::<f32>() / n_total;
        let mut x_minus = x_data.clone();
        x_minus[idx] -= h;
        let y_minus = cpu_rmsnorm(&x_minus, &w_data, rows, hidden, eps);
        let loss_minus: f32 = y_minus.iter().map(|v| v * v).sum::<f32>() / n_total;
        let fd_grad = (loss_plus - loss_minus) / (2.0 * h);
        let diff = (grad_x[idx] - fd_grad).abs();
        max_diff = max_diff.max(diff);
        let _ = base_loss; // referenced for clarity
    }
    assert!(
        max_diff < 5e-3,
        "rmsnorm backward fd discrepancy {max_diff}"
    );
    Ok(())
}

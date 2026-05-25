//! Phase C.1: vk_softmax forward + backward parity vs an analytical
//! CPU reference. (candle-free; #1082)

use anyhow::Result;
use kiln_vulkan_kernel::VulkanDevice;
use kiln_vulkan_kernel::vk_autograd::vk_backward;
use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
use kiln_vulkan_kernel::vk_ops::reduce::vk_mean_all;
use kiln_vulkan_kernel::vk_ops::softmax::{vk_softmax_lastdim, vk_softmax_lastdim_no_grad};
use kiln_vulkan_kernel::vk_tensor::VkTensor;
use std::sync::Arc;

fn cpu_softmax_lastdim(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let mx = x[base..base + cols]
            .iter()
            .cloned()
            .fold(f32::MIN, f32::max);
        let mut z = 0.0_f32;
        for c in 0..cols {
            z += (x[base + c] - mx).exp();
        }
        for c in 0..cols {
            y[base + c] = ((x[base + c] - mx).exp()) / z;
        }
    }
    y
}

/// Reference dx for loss = mean(softmax(x) ** 2) computed analytically:
///   y = softmax(x); s = sum_j y_j * (2*y_j / N)  (where N = rows*cols)
///   dx_i = y_i * (2*y_i / N - s)
/// since d(loss)/dy_i = 2*y_i/N, and softmax-bwd uses s = sum_j y_j * dy_j.
fn cpu_softmax_sq_loss_dx(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let y = cpu_softmax_lastdim(x, rows, cols);
    let n_total = (rows * cols) as f32;
    let mut dx = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let mut s = 0.0_f32;
        for c in 0..cols {
            let dy = 2.0 * y[base + c] / n_total;
            s += y[base + c] * dy;
        }
        for c in 0..cols {
            let dy = 2.0 * y[base + c] / n_total;
            dx[base + c] = y[base + c] * (dy - s);
        }
    }
    dx
}

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

fn upload_f32(dev: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    VkTensor::from_f32_slice(data, shape.to_vec(), Arc::clone(dev))
}

fn upload_param_f32(
    dev: &Arc<VulkanDevice>,
    data: &[f32],
    shape: &[usize],
) -> Result<VkTensor> {
    VkTensor::parameter_from_f32_slice(data, shape.to_vec(), Arc::clone(dev))
}

fn max_abs_diff(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len(), "len mismatch");
    got.iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn vk_softmax_forward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 3;
    let cols = 17;
    let data: Vec<f32> = (0..(rows * cols))
        .map(|i| ((i as f32) * 0.11).sin() * 3.0)
        .collect();
    let x_vk = upload_f32(&dev, &data, &[rows, cols])?;
    let y_vk = vk_softmax_lastdim_no_grad(&x_vk)?.to_vec_f32()?;

    let y_c = cpu_softmax_lastdim(&data, rows, cols);
    let mad = max_abs_diff(&y_vk, &y_c);
    assert!(mad < 1e-5, "softmax forward mad {mad}");
    Ok(())
}

#[test]
fn vk_softmax_backward_parity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 4;
    let cols = 13;
    let n_total = (rows * cols) as f32;
    let data: Vec<f32> = (0..(rows * cols))
        .map(|i| ((i as f32) * 0.07).cos() * 2.0)
        .collect();

    // VK path
    let x = upload_param_f32(&dev, &data, &[rows, cols])?;
    let y = vk_softmax_lastdim(&x)?;
    let sq = vk_mul(&y, &y)?;
    let loss = vk_mean_all(&sq)?;
    let grads = vk_backward(&loss)?;
    let grad_x = grads.get(x.param_id().unwrap()).expect("dx").to_vec_f32()?;

    // Analytical reference (softmax + mean(y*y) loss)
    let exp_dx = cpu_softmax_sq_loss_dx(&data, rows, cols);
    let _ = n_total;
    let mad = max_abs_diff(&grad_x, &exp_dx);
    assert!(mad < 1e-5, "softmax backward mad {mad}");
    Ok(())
}

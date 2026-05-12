//! Forward-only parity tests for the GDN foundation wrappers added in
//! Phase 2:
//!   - vk_causal_conv1d_no_grad     (with fused SiLU)
//!   - vk_gdn_gates_no_grad         (sigmoid β, softplus·exp g)
//!   - vk_gdn_gated_rms_norm_no_grad
//!
//! Each test computes a CPU reference then asserts max-abs error
//! against the GPU output within F32 tolerance. Skips cleanly if no
//! Vulkan device is available.

#![cfg(test)]

use anyhow::Result;
use candle_core::{Device, Tensor};
use kiln_vulkan_kernel::vk_ops::conv1d::vk_causal_conv1d_no_grad;
use kiln_vulkan_kernel::vk_ops::gdn_gated_rms_norm::vk_gdn_gated_rms_norm_no_grad;
use kiln_vulkan_kernel::vk_ops::gdn_gates::vk_gdn_gates_no_grad;
use kiln_vulkan_kernel::vk_ops::gdn_state::VkLinearAttentionState;
use kiln_vulkan_kernel::{VkTensor, VulkanBuffer, VulkanDevice};
use std::sync::Arc;

fn vk_dev() -> Option<Arc<VulkanDevice>> {
    if !VulkanDevice::probe() {
        return None;
    }
    VulkanDevice::new().ok().map(Arc::new)
}

fn upload(device: &Arc<VulkanDevice>, data: &[f32], shape: &[usize]) -> Result<VkTensor> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    VkTensor::from_candle(&t, Arc::clone(device))
}

fn upload_buffer(device: &Arc<VulkanDevice>, data: &[f32]) -> Result<Arc<VulkanBuffer>> {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let buf = VulkanBuffer::create_device_local(
        device.device(),
        device.device_local_mem_type(),
        bytes.len().max(4) as u64,
    )?;
    VulkanBuffer::upload_data(
        device.device(),
        device.host_visible_mem_type(),
        device.queue(),
        device.queue_family_index(),
        &buf,
        &bytes,
    )?;
    Ok(Arc::new(buf))
}

fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

// ---------------- VkLinearAttentionState ----------------

#[test]
fn vk_linear_attention_state_zeros_initialized() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let state = VkLinearAttentionState::zeros(&dev, 3, 1, 2, 4, 4, 6, 4)?;
    assert_eq!(state.layers.len(), 3);
    for layer in &state.layers {
        assert_eq!(layer.recurrent_n_elements, 1 * 2 * 4 * 4);
        assert_eq!(layer.conv_n_elements, 1 * 6 * 3); // kernel-1 = 3
    }
    Ok(())
}

// ---------------- vk_gdn_gates ----------------

fn cpu_gdn_gates(a: &[f32], b: &[f32], a_log: &[f32], dt_bias: &[f32], nv: usize) -> (Vec<f32>, Vec<f32>) {
    let total = a.len();
    let mut beta = Vec::with_capacity(total);
    let mut g = Vec::with_capacity(total);
    for i in 0..total {
        let nv_idx = i % nv;
        let bv = b[i];
        let beta_v = if bv >= 0.0 {
            1.0 / (1.0 + (-bv).exp())
        } else {
            let e = bv.exp();
            e / (1.0 + e)
        };
        beta.push(beta_v);
        let x = a[i] + dt_bias[nv_idx];
        let softplus = if x > 20.0 { x } else { (1.0 + x.exp()).ln() };
        g.push(-a_log[nv_idx].exp() * softplus);
    }
    (beta, g)
}

#[test]
fn vk_gdn_gates_matches_cpu_reference() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let b_dim = 1;
    let t = 8;
    let nv = 2;
    let total = b_dim * t * nv;
    let a_data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.13 - 0.5).sin()).collect();
    let b_data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.21 + 0.7).cos()).collect();
    let a_log_data: Vec<f32> = (0..nv).map(|i| -((i as f32) + 1.0) * 0.5).collect();
    let dt_bias_data: Vec<f32> = (0..nv).map(|i| ((i as f32) - 0.5) * 0.1).collect();

    let a = upload(&dev, &a_data, &[b_dim, t, nv])?;
    let b = upload(&dev, &b_data, &[b_dim, t, nv])?;
    let a_log = upload(&dev, &a_log_data, &[nv])?;
    let dt_bias = upload(&dev, &dt_bias_data, &[nv])?;

    let (beta_gpu, g_gpu) = vk_gdn_gates_no_grad(&a, &b, &a_log, &dt_bias, nv)?;
    let (beta_ref, g_ref) =
        cpu_gdn_gates(&a_data, &b_data, &a_log_data, &dt_bias_data, nv);

    let beta_actual = beta_gpu.to_vec_f32()?;
    let g_actual = g_gpu.to_vec_f32()?;
    assert!(
        max_abs_err(&beta_actual, &beta_ref) < 1e-5,
        "beta max abs err {}",
        max_abs_err(&beta_actual, &beta_ref)
    );
    assert!(
        max_abs_err(&g_actual, &g_ref) < 1e-5,
        "g max abs err {}",
        max_abs_err(&g_actual, &g_ref)
    );
    Ok(())
}

// ---------------- vk_causal_conv1d ----------------

fn cpu_causal_conv1d(
    x: &[f32],
    weight: &[f32],
    conv_state: &[f32],
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> Vec<f32> {
    let state_len = kernel_size - 1;
    let mut out = vec![0.0_f32; batch * channels * seq_len];
    for b in 0..batch {
        for c in 0..channels {
            for t in 0..seq_len {
                let mut sum = 0.0_f32;
                for k in 0..kernel_size {
                    let logical_t = t + k;
                    let xv = if logical_t < state_len {
                        conv_state[(b * channels + c) * state_len + logical_t]
                    } else {
                        x[(b * channels + c) * seq_len + (logical_t - state_len)]
                    };
                    sum += xv * weight[c * kernel_size + k];
                }
                let silu = if sum >= 0.0 {
                    sum / (1.0 + (-sum).exp())
                } else {
                    let e = sum.exp();
                    sum * e / (1.0 + e)
                };
                out[(b * channels + c) * seq_len + t] = silu;
            }
        }
    }
    out
}

#[test]
fn vk_causal_conv1d_matches_cpu_reference() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 1;
    let channels = 4;
    let seq_len = 8;
    let kernel_size = 4;
    let state_len = kernel_size - 1;
    let x_data: Vec<f32> = (0..(batch * channels * seq_len))
        .map(|i| ((i as f32) * 0.07 - 0.4).sin())
        .collect();
    let w_data: Vec<f32> = (0..(channels * kernel_size))
        .map(|i| ((i as f32) * 0.11 + 0.2).cos() * 0.3)
        .collect();
    let cs_data: Vec<f32> = (0..(batch * channels * state_len))
        .map(|i| ((i as f32) * 0.05 + 0.1).sin())
        .collect();

    let x = upload(&dev, &x_data, &[batch, channels, seq_len])?;
    let weight = upload(&dev, &w_data, &[channels, kernel_size])?;
    let conv_state = upload_buffer(&dev, &cs_data)?;

    let out = vk_causal_conv1d_no_grad(
        &x,
        &weight,
        &conv_state,
        batch,
        channels,
        seq_len,
        kernel_size,
    )?;
    let out_actual = out.to_vec_f32()?;
    let out_ref = cpu_causal_conv1d(
        &x_data,
        &w_data,
        &cs_data,
        batch,
        channels,
        seq_len,
        kernel_size,
    );
    let err = max_abs_err(&out_actual, &out_ref);
    assert!(err < 1e-5, "conv1d max abs err {err}");
    Ok(())
}

// ---------------- vk_gdn_gated_rms_norm ----------------

fn cpu_gated_rms_norm(
    x: &[f32],
    z: &[f32],
    weight: &[f32],
    rows: usize,
    hidden: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * hidden];
    for r in 0..rows {
        let base = r * hidden;
        let mut sum_sq = 0.0_f32;
        for c in 0..hidden {
            sum_sq += x[base + c] * x[base + c];
        }
        let rms_inv = 1.0 / (sum_sq / (hidden as f32) + eps).sqrt();
        for c in 0..hidden {
            let zv = z[base + c];
            let silu_z = if zv >= 0.0 {
                zv / (1.0 + (-zv).exp())
            } else {
                let e = zv.exp();
                zv * e / (1.0 + e)
            };
            out[base + c] = x[base + c] * rms_inv * silu_z * weight[c];
        }
    }
    out
}

#[test]
fn vk_gdn_gated_rms_norm_matches_cpu_reference() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 6;
    let hidden = 32;
    let eps = 1e-6_f32;
    let x_data: Vec<f32> = (0..(rows * hidden))
        .map(|i| ((i as f32) * 0.03 + 0.1).sin())
        .collect();
    let z_data: Vec<f32> = (0..(rows * hidden))
        .map(|i| ((i as f32) * 0.05 - 0.2).cos())
        .collect();
    let w_data: Vec<f32> = (0..hidden)
        .map(|i| ((i as f32) * 0.07).sin() * 0.5 + 1.0)
        .collect();

    let x = upload(&dev, &x_data, &[rows, hidden])?;
    let z = upload(&dev, &z_data, &[rows, hidden])?;
    let w = upload(&dev, &w_data, &[hidden])?;
    let out = vk_gdn_gated_rms_norm_no_grad(&x, &z, &w, eps)?;
    let out_actual = out.to_vec_f32()?;
    let out_ref = cpu_gated_rms_norm(&x_data, &z_data, &w_data, rows, hidden, eps);
    let err = max_abs_err(&out_actual, &out_ref);
    assert!(err < 1e-5, "gated_rms_norm max abs err {err}");
    Ok(())
}

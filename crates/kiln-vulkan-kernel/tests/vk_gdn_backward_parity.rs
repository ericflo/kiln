//! Backward-kernel parity tests for Phase 4 backward shaders.
//!
//! Each test computes a CPU reference for the backward and asserts
//! GPU output matches within 1e-5 (F32).

#![cfg(test)]

use anyhow::Result;
use candle_core::{Device, Tensor};
use kiln_vulkan_kernel::vk_ops::conv1d::vk_causal_conv1d_bwd_no_grad;
use kiln_vulkan_kernel::vk_ops::gdn_gates::vk_gdn_gates_bwd_no_grad;
use kiln_vulkan_kernel::vk_ops::reverse_cumsum::vk_reverse_cumsum_no_grad;
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

// ---------------- vk_reverse_cumsum ----------------

#[test]
fn vk_reverse_cumsum_matches_cpu() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 3;
    let cols = 8;
    let data: Vec<f32> = (0..(rows * cols)).map(|i| (i as f32) * 0.1).collect();
    let t = upload(&dev, &data, &[rows, cols])?;
    let out = vk_reverse_cumsum_no_grad(&t)?;
    let actual = out.to_vec_f32()?;

    let mut expected = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        let mut acc = 0.0_f32;
        for c in (0..cols).rev() {
            acc += data[r * cols + c];
            expected[r * cols + c] = acc;
        }
    }
    let err = max_abs_err(&actual, &expected);
    assert!(err < 1e-5, "reverse_cumsum max err {err}");
    Ok(())
}

// ---------------- vk_gdn_gates_bwd ----------------

fn cpu_gates_bwd(
    d_beta: &[f32],
    d_g: &[f32],
    a: &[f32],
    b: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    nv: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let total = a.len();
    let mut d_a = vec![0.0_f32; total];
    let mut d_b = vec![0.0_f32; total];
    let mut d_a_log = vec![0.0_f32; nv];
    let mut d_dt = vec![0.0_f32; nv];
    for i in 0..total {
        let n = i % nv;
        let a_v = a[i];
        let b_v = b[i];
        let a_log_v = a_log[n];
        let dt_v = dt_bias[n];

        let sig_b = if b_v >= 0.0 {
            1.0 / (1.0 + (-b_v).exp())
        } else {
            let e = b_v.exp();
            e / (1.0 + e)
        };
        d_b[i] = d_beta[i] * sig_b * (1.0 - sig_b);

        let a_plus = a_v + dt_v;
        let sig_apd = if a_plus >= 0.0 {
            1.0 / (1.0 + (-a_plus).exp())
        } else {
            let e = a_plus.exp();
            e / (1.0 + e)
        };
        let sp = if a_plus > 20.0 {
            a_plus
        } else {
            (1.0 + a_plus.exp()).ln()
        };
        let minus_exp_alog = -a_log_v.exp();
        let dg_dap = minus_exp_alog * sig_apd;
        let dg_dalog = minus_exp_alog * sp;

        d_a[i] = d_g[i] * dg_dap;
        d_a_log[n] += d_g[i] * dg_dalog;
        d_dt[n] += d_g[i] * dg_dap;
    }
    (d_a, d_b, d_a_log, d_dt)
}

#[test]
fn vk_gdn_gates_bwd_matches_cpu() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let b_dim = 1;
    let t = 4;
    let nv = 2;
    let total = b_dim * t * nv;
    let a_data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.13 - 0.5).sin()).collect();
    let b_data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.21 + 0.7).cos()).collect();
    let a_log_data: Vec<f32> = (0..nv).map(|i| -((i as f32) + 1.0) * 0.5).collect();
    let dt_bias_data: Vec<f32> = (0..nv).map(|i| ((i as f32) - 0.5) * 0.1).collect();
    let d_beta_data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.07).sin()).collect();
    let d_g_data: Vec<f32> = (0..total).map(|i| ((i as f32) * 0.11).cos()).collect();

    let a = upload(&dev, &a_data, &[b_dim, t, nv])?;
    let b = upload(&dev, &b_data, &[b_dim, t, nv])?;
    let a_log = upload(&dev, &a_log_data, &[nv])?;
    let dt_bias = upload(&dev, &dt_bias_data, &[nv])?;
    let d_beta = upload(&dev, &d_beta_data, &[b_dim, t, nv])?;
    let d_g = upload(&dev, &d_g_data, &[b_dim, t, nv])?;

    let (gd_a, gd_b, gd_alog, gd_dt) =
        vk_gdn_gates_bwd_no_grad(&d_beta, &d_g, &a, &b, &a_log, &dt_bias, nv)?;
    let (cd_a, cd_b, cd_alog, cd_dt) =
        cpu_gates_bwd(&d_beta_data, &d_g_data, &a_data, &b_data, &a_log_data, &dt_bias_data, nv);

    assert!(max_abs_err(&gd_a.to_vec_f32()?, &cd_a) < 1e-5);
    assert!(max_abs_err(&gd_b.to_vec_f32()?, &cd_b) < 1e-5);
    assert!(max_abs_err(&gd_alog.to_vec_f32()?, &cd_alog) < 1e-5);
    assert!(max_abs_err(&gd_dt.to_vec_f32()?, &cd_dt) < 1e-5);
    Ok(())
}

// ---------------- vk_causal_conv1d_bwd ----------------

fn cpu_conv1d_linear_bwd(
    d_out: &[f32],
    weight: &[f32],
    x: &[f32],
    cs: &[f32],
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let state_len = kernel_size - 1;
    let mut d_x = vec![0.0_f32; batch * channels * seq_len];
    let mut d_weight = vec![0.0_f32; channels * kernel_size];
    let mut d_cs = vec![0.0_f32; batch * channels * state_len];
    for b in 0..batch {
        for c in 0..channels {
            for t in 0..seq_len {
                let dout_v = d_out[(b * channels + c) * seq_len + t];
                for k in 0..kernel_size {
                    let logical_t = t + k;
                    if logical_t < state_len {
                        d_cs[(b * channels + c) * state_len + logical_t] +=
                            dout_v * weight[c * kernel_size + k];
                        d_weight[c * kernel_size + k] +=
                            dout_v * cs[(b * channels + c) * state_len + logical_t];
                    } else {
                        let xi = logical_t - state_len;
                        d_x[(b * channels + c) * seq_len + xi] +=
                            dout_v * weight[c * kernel_size + k];
                        d_weight[c * kernel_size + k] +=
                            dout_v * x[(b * channels + c) * seq_len + xi];
                    }
                }
            }
        }
    }
    (d_x, d_weight, d_cs)
}

#[test]
fn vk_causal_conv1d_bwd_matches_cpu() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 1;
    let channels = 4;
    let seq_len = 8;
    let kernel_size = 4;
    let state_len = kernel_size - 1;
    let dout_data: Vec<f32> = (0..(batch * channels * seq_len))
        .map(|i| ((i as f32) * 0.05 - 0.3).sin())
        .collect();
    let w_data: Vec<f32> = (0..(channels * kernel_size))
        .map(|i| ((i as f32) * 0.07 + 0.1).cos())
        .collect();
    let x_data: Vec<f32> = (0..(batch * channels * seq_len))
        .map(|i| ((i as f32) * 0.03).sin())
        .collect();
    let cs_data: Vec<f32> = (0..(batch * channels * state_len))
        .map(|i| ((i as f32) * 0.09 + 0.2).cos())
        .collect();

    let dout = upload(&dev, &dout_data, &[batch, channels, seq_len])?;
    let weight = upload(&dev, &w_data, &[channels, kernel_size])?;
    let x = upload(&dev, &x_data, &[batch, channels, seq_len])?;
    let cs = upload_buffer(&dev, &cs_data)?;

    let (d_x, d_w, _d_cs_buf) = vk_causal_conv1d_bwd_no_grad(
        &dout, &weight, &x, &cs, batch, channels, seq_len, kernel_size,
    )?;
    let (cpu_dx, cpu_dw, _cpu_dcs) = cpu_conv1d_linear_bwd(
        &dout_data,
        &w_data,
        &x_data,
        &cs_data,
        batch,
        channels,
        seq_len,
        kernel_size,
    );

    assert!(
        max_abs_err(&d_x.to_vec_f32()?, &cpu_dx) < 1e-5,
        "d_x err {}",
        max_abs_err(&d_x.to_vec_f32()?, &cpu_dx)
    );
    assert!(
        max_abs_err(&d_w.to_vec_f32()?, &cpu_dw) < 1e-5,
        "d_weight err {}",
        max_abs_err(&d_w.to_vec_f32()?, &cpu_dw)
    );
    Ok(())
}

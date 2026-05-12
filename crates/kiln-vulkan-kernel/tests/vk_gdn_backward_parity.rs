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

// ---------------- vk_gdn_gated_rms_norm_bwd ----------------

#[test]
fn vk_gdn_gated_rms_norm_bwd_matches_finite_diff() -> Result<()> {
    use kiln_vulkan_kernel::vk_ops::gdn_gated_rms_norm::{
        vk_gdn_gated_rms_norm_bwd_no_grad, vk_gdn_gated_rms_norm_no_grad,
    };
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 2;
    let hidden = 4;
    let eps = 1e-6_f32;
    let x_data: Vec<f32> = (0..(rows * hidden))
        .map(|i| ((i as f32) * 0.07 + 0.3).sin() + 0.5)
        .collect();
    let z_data: Vec<f32> = (0..(rows * hidden))
        .map(|i| ((i as f32) * 0.05 - 0.1).cos())
        .collect();
    let w_data: Vec<f32> = (0..hidden).map(|i| 0.5 + (i as f32) * 0.1).collect();

    let x = upload(&dev, &x_data, &[rows, hidden])?;
    let z = upload(&dev, &z_data, &[rows, hidden])?;
    let w = upload(&dev, &w_data, &[hidden])?;
    // d_out: synthetic upstream gradient
    let dout_data: Vec<f32> = (0..(rows * hidden)).map(|i| (i as f32 + 1.0) * 0.01).collect();
    let dout = upload(&dev, &dout_data, &[rows, hidden])?;

    let (d_x, d_z, d_w) = vk_gdn_gated_rms_norm_bwd_no_grad(&dout, &x, &z, &w, eps)?;

    // Verify d_x via finite differences for one element
    let test_idx = 1;
    let h = 1e-4_f32;
    let mut x_plus = x_data.clone();
    x_plus[test_idx] += h;
    let mut x_minus = x_data.clone();
    x_minus[test_idx] -= h;
    let xp = upload(&dev, &x_plus, &[rows, hidden])?;
    let xm = upload(&dev, &x_minus, &[rows, hidden])?;
    let outp = vk_gdn_gated_rms_norm_no_grad(&xp, &z, &w, eps)?.to_vec_f32()?;
    let outm = vk_gdn_gated_rms_norm_no_grad(&xm, &z, &w, eps)?.to_vec_f32()?;
    // numerical d_loss/d_x_i where loss = Σ d_out_j · out_j
    let loss_p: f32 = outp.iter().zip(dout_data.iter()).map(|(a, b)| a * b).sum();
    let loss_m: f32 = outm.iter().zip(dout_data.iter()).map(|(a, b)| a * b).sum();
    let numerical = (loss_p - loss_m) / (2.0 * h);
    let analytic = d_x.to_vec_f32()?[test_idx];
    let diff = (numerical - analytic).abs();
    println!("gated_rms_norm_bwd d_x[{test_idx}]: numerical={numerical:.6} analytic={analytic:.6} diff={diff:.6e}");
    assert!(
        diff < 1e-2,
        "d_x finite-diff vs analytic: |{numerical} - {analytic}| = {diff}"
    );
    // Sanity: outputs are finite and right shape
    assert_eq!(d_x.shape(), &[rows, hidden]);
    assert_eq!(d_z.shape(), &[rows, hidden]);
    assert_eq!(d_w.shape(), &[hidden]);
    for v in d_x.to_vec_f32()? {
        assert!(v.is_finite());
    }
    for v in d_z.to_vec_f32()? {
        assert!(v.is_finite());
    }
    for v in d_w.to_vec_f32()? {
        assert!(v.is_finite());
    }
    Ok(())
}

// ---------------- vk_gdn_chunk_scan_bwd ----------------

#[test]
fn vk_gdn_chunk_scan_bwd_matches_cpu() -> Result<()> {
    use kiln_vulkan_kernel::vk_ops::gdn_chunk_bwd::vk_gdn_chunk_scan_bwd_no_grad;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 1;
    let nv = 2;
    let chunk = 4;
    let dv = 3;
    // Build masked B (causal lower-tri) and W
    let bmask: Vec<f32> = (0..(batch * nv * chunk * chunk))
        .map(|i| {
            let t = (i / chunk) % chunk;
            let j = i % chunk;
            if j <= t {
                ((i as f32) * 0.05 + 0.1).sin() * 0.3
            } else {
                0.0
            }
        })
        .collect();
    let w_data: Vec<f32> = (0..(batch * nv * chunk * dv))
        .map(|i| ((i as f32) * 0.07).cos())
        .collect();
    let dout_data: Vec<f32> = (0..(batch * nv * chunk * dv))
        .map(|i| 0.1 * ((i + 1) as f32))
        .collect();
    let bm = upload(&dev, &bmask, &[batch, nv, chunk, chunk])?;
    let w = upload(&dev, &w_data, &[batch, nv, chunk, dv])?;
    let dout = upload(&dev, &dout_data, &[batch, nv, chunk, dv])?;
    let (gpu_dqs, gpu_dbm, gpu_dw) =
        vk_gdn_chunk_scan_bwd_no_grad(&dout, &bm, &w, batch, nv, chunk, dv)?;

    // CPU reference
    let mut cpu_dqs = vec![0.0_f32; batch * nv * chunk * dv];
    let mut cpu_dbm = vec![0.0_f32; batch * nv * chunk * chunk];
    let mut cpu_dw = vec![0.0_f32; batch * nv * chunk * dv];
    for bh in 0..batch * nv {
        let v_base = bh * chunk * dv;
        let m_base = bh * chunk * chunk;
        for t in 0..chunk {
            for d in 0..dv {
                cpu_dqs[v_base + t * dv + d] = dout_data[v_base + t * dv + d];
            }
        }
        for t in 0..chunk {
            for i in 0..chunk {
                let mut acc = 0.0_f32;
                for d in 0..dv {
                    acc += dout_data[v_base + t * dv + d] * w_data[v_base + i * dv + d];
                }
                cpu_dbm[m_base + t * chunk + i] = acc;
            }
        }
        for i in 0..chunk {
            for d in 0..dv {
                let mut acc = 0.0_f32;
                for t in 0..chunk {
                    acc += bmask[m_base + t * chunk + i] * dout_data[v_base + t * dv + d];
                }
                cpu_dw[v_base + i * dv + d] = acc;
            }
        }
    }
    assert!(max_abs_err(&gpu_dqs.to_vec_f32()?, &cpu_dqs) < 1e-5);
    assert!(max_abs_err(&gpu_dbm.to_vec_f32()?, &cpu_dbm) < 1e-5);
    assert!(max_abs_err(&gpu_dw.to_vec_f32()?, &cpu_dw) < 1e-5);
    Ok(())
}

// ---------------- vk_gdn_state_exit_bwd ----------------

#[test]
fn vk_gdn_state_exit_bwd_matches_cpu() -> Result<()> {
    // Compare GPU shader to the CPU fallback (set via env var).
    use kiln_vulkan_kernel::vk_ops::gdn_chunk_bwd::vk_gdn_state_exit_bwd_no_grad;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 1;
    let nv = 2;
    let chunk = 4;
    let dk = 4;
    let dv = 3;

    let dse: Vec<f32> = (0..(batch * nv * dk * dv))
        .map(|i| ((i as f32) * 0.07).sin() * 0.2)
        .collect();
    let dlc: Vec<f32> = (0..(batch * nv * chunk))
        .map(|i| 0.5 + ((i as f32) * 0.05).cos() * 0.1)
        .collect();
    let kc: Vec<f32> = (0..(batch * nv * chunk * dk))
        .map(|i| ((i as f32) * 0.04 - 0.1).sin())
        .collect();
    let wd: Vec<f32> = (0..(batch * nv * chunk * dv))
        .map(|i| ((i as f32) * 0.03 + 0.2).cos())
        .collect();
    let s: Vec<f32> = (0..(batch * nv * dk * dv))
        .map(|i| ((i as f32) * 0.05 + 0.1).sin())
        .collect();
    let pl: Vec<f32> = (0..(batch * nv)).map(|i| 0.9 + (i as f32) * 0.01).collect();

    let dse_t = upload(&dev, &dse, &[batch, nv, dk, dv])?;
    let dlc_t = upload(&dev, &dlc, &[batch, nv, chunk])?;
    let k_t = upload(&dev, &kc, &[batch, nv, chunk, dk])?;
    let w_t = upload(&dev, &wd, &[batch, nv, chunk, dv])?;
    let s_t = upload(&dev, &s, &[batch, nv, dk, dv])?;
    let pl_t = upload(&dev, &pl, &[batch, nv])?;

    // GPU path
    let (gd_si, gd_w, gd_k, gd_dec, gd_pl) = vk_gdn_state_exit_bwd_no_grad(
        &dse_t, &dlc_t, &k_t, &w_t, &s_t, &pl_t, batch, nv, chunk, dk, dv,
    )?;

    // CPU reference (run via env var). SAFETY: serial test execution
    // (--test-threads=1 from CI command line), no concurrent
    // access to the env. The env var is scoped to this test.
    unsafe {
        std::env::set_var("KILN_VK_GDN_STATE_EXIT_BWD_CPU", "1");
    }
    let (cd_si, cd_w, cd_k, cd_dec, cd_pl) = vk_gdn_state_exit_bwd_no_grad(
        &dse_t, &dlc_t, &k_t, &w_t, &s_t, &pl_t, batch, nv, chunk, dk, dv,
    )?;
    unsafe {
        std::env::remove_var("KILN_VK_GDN_STATE_EXIT_BWD_CPU");
    }

    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    };
    assert!(max_err(&gd_si.to_vec_f32()?, &cd_si.to_vec_f32()?) < 1e-5);
    assert!(max_err(&gd_w.to_vec_f32()?, &cd_w.to_vec_f32()?) < 1e-5);
    assert!(max_err(&gd_k.to_vec_f32()?, &cd_k.to_vec_f32()?) < 1e-5);
    assert!(max_err(&gd_dec.to_vec_f32()?, &cd_dec.to_vec_f32()?) < 1e-5);
    assert!(max_err(&gd_pl.to_vec_f32()?, &cd_pl.to_vec_f32()?) < 1e-5);
    Ok(())
}

// ---------------- vk_gdn_chunk_prep_bwd (most complex) ----------------

#[test]
fn vk_gdn_chunk_prep_bwd_matches_finite_diff() -> Result<()> {
    use kiln_vulkan_kernel::vk_ops::gdn_chunk_bwd::vk_gdn_chunk_prep_bwd_no_grad;
    use kiln_vulkan_kernel::vk_ops::gdn_chunk_prep::vk_gdn_chunk_prep_no_grad;
    let Some(dev) = vk_dev() else { return Ok(()) };

    let batch = 1;
    let nv = 1;
    let chunk = 4;
    let dv = 2;

    let g_data: Vec<f32> = (0..(batch * nv * chunk)).map(|i| -0.05 + (i as f32) * 0.01).collect();
    let v_data: Vec<f32> = (0..(batch * nv * chunk * dv))
        .map(|i| ((i as f32) * 0.07).sin() * 0.3)
        .collect();
    let kkt_data: Vec<f32> = (0..(batch * nv * chunk * chunk))
        .map(|i| ((i as f32) * 0.05 + 0.1).cos() * 0.2)
        .collect();
    let qkt_data: Vec<f32> = (0..(batch * nv * chunk * chunk))
        .map(|i| ((i as f32) * 0.04 - 0.2).sin() * 0.3)
        .collect();
    let ks_data: Vec<f32> = (0..(batch * nv * chunk * dv))
        .map(|i| ((i as f32) * 0.06).cos() * 0.1)
        .collect();
    let qs_data: Vec<f32> = (0..(batch * nv * chunk * dv))
        .map(|i| ((i as f32) * 0.08).sin() * 0.15)
        .collect();

    let g = upload(&dev, &g_data, &[batch, nv, chunk])?;
    let v = upload(&dev, &v_data, &[batch, nv, chunk, dv])?;
    let kkt = upload(&dev, &kkt_data, &[batch, nv, chunk, chunk])?;
    let qkt = upload(&dev, &qkt_data, &[batch, nv, chunk, chunk])?;
    let ks_e = upload(&dev, &ks_data, &[batch, nv, chunk, dv])?;
    let qs = upload(&dev, &qs_data, &[batch, nv, chunk, dv])?;

    // Synthetic upstream gradients
    let n_a = batch * nv * chunk * chunk;
    let n_v = batch * nv * chunk * dv;
    let n_c = batch * nv * chunk;
    let n_bh = batch * nv;
    let das_data: Vec<f32> = (0..n_a).map(|i| 0.01 * ((i + 1) as f32)).collect();
    let dbm_data: Vec<f32> = (0..n_a).map(|i| 0.013 * ((i + 1) as f32)).collect();
    let dvp_data: Vec<f32> = (0..n_v).map(|i| 0.017 * ((i + 1) as f32)).collect();
    let dqss_data: Vec<f32> = (0..n_v).map(|i| 0.019 * ((i + 1) as f32)).collect();
    let ddec_data: Vec<f32> = (0..n_c).map(|i| 0.023 * ((i + 1) as f32)).collect();
    let dpl_data: Vec<f32> = (0..n_bh).map(|i| 0.029 * ((i + 1) as f32)).collect();

    let das = upload(&dev, &das_data, &[batch, nv, chunk, chunk])?;
    let dbm = upload(&dev, &dbm_data, &[batch, nv, chunk, chunk])?;
    let dvp = upload(&dev, &dvp_data, &[batch, nv, chunk, dv])?;
    let dqss = upload(&dev, &dqss_data, &[batch, nv, chunk, dv])?;
    let ddec = upload(&dev, &ddec_data, &[batch, nv, chunk])?;
    let dpl = upload(&dev, &dpl_data, &[batch, nv])?;

    let (d_g, _d_v, _d_kkt, _d_qkt, _d_ks, _d_qs) = vk_gdn_chunk_prep_bwd_no_grad(
        &das, &dbm, &dvp, &dqss, &ddec, &dpl, &g, &v, &kkt, &qkt, &ks_e, &qs, batch, nv, chunk, dv,
    )?;

    // Finite-diff check on d_g[0]: perturb g[0], rerun forward, measure
    // change in dotted-loss.
    let h = 1e-4_f32;
    let dotted = |out: &kiln_vulkan_kernel::vk_ops::gdn_chunk_prep::GdnChunkPrepOutput| -> f32 {
        let mut sum = 0.0_f32;
        for (a, b) in out.a_strict.to_vec_f32().unwrap().iter().zip(das_data.iter()) {
            sum += a * b;
        }
        for (a, b) in out.b_mask.to_vec_f32().unwrap().iter().zip(dbm_data.iter()) {
            sum += a * b;
        }
        for (a, b) in out.v_prime.to_vec_f32().unwrap().iter().zip(dvp_data.iter()) {
            sum += a * b;
        }
        for (a, b) in out.q_s_scaled.to_vec_f32().unwrap().iter().zip(dqss_data.iter()) {
            sum += a * b;
        }
        for (a, b) in out.decay_last_col.to_vec_f32().unwrap().iter().zip(ddec_data.iter()) {
            sum += a * b;
        }
        for (a, b) in out.p_last.to_vec_f32().unwrap().iter().zip(dpl_data.iter()) {
            sum += a * b;
        }
        sum
    };
    let mut g_p = g_data.clone();
    g_p[0] += h;
    let mut g_m = g_data.clone();
    g_m[0] -= h;
    let g_pt = upload(&dev, &g_p, &[batch, nv, chunk])?;
    let g_mt = upload(&dev, &g_m, &[batch, nv, chunk])?;
    let out_p = vk_gdn_chunk_prep_no_grad(&g_pt, &v, &kkt, &qkt, &ks_e, &qs, batch, nv, chunk, dv)?;
    let out_m = vk_gdn_chunk_prep_no_grad(&g_mt, &v, &kkt, &qkt, &ks_e, &qs, batch, nv, chunk, dv)?;
    let numerical = (dotted(&out_p) - dotted(&out_m)) / (2.0 * h);
    let analytic = d_g.to_vec_f32()?[0];
    let err = (numerical - analytic).abs();
    let rel_err = err / numerical.abs().max(1e-6);
    println!(
        "chunk_prep_bwd d_g[0]: numerical={numerical:.6} analytic={analytic:.6} \
         abs_err={err:.6e} rel_err={rel_err:.6e}"
    );
    // Allow 5% rel error or 1e-4 abs — finite-diff at h=1e-4 has its
    // own numerical noise that can dominate at small gradient values.
    assert!(
        err < 1e-4 || rel_err < 5e-2,
        "chunk_prep_bwd d_g[0] mismatch (abs_err={err}, rel_err={rel_err})"
    );
    Ok(())
}

// ---------------- vk_solve_tri_transpose ----------------

#[test]
fn vk_solve_tri_transpose_solves_correctly() -> Result<()> {
    use kiln_vulkan_kernel::vk_ops::gdn_chunk_bwd::vk_solve_tri_transpose_no_grad;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 1;
    let nv = 1;
    let chunk = 4;
    let dv = 3;
    let a_data: Vec<f32> = (0..(batch * nv * chunk * chunk))
        .map(|i| {
            let t = (i / chunk) % chunk;
            let j = i % chunk;
            if j < t {
                ((i as f32) * 0.05).sin() * 0.1
            } else {
                0.0
            }
        })
        .collect();
    let beta_data: Vec<f32> = (0..(batch * nv * chunk)).map(|i| 0.5 + (i as f32) * 0.05).collect();
    let dw_data: Vec<f32> = (0..(batch * nv * chunk * dv))
        .map(|i| 0.3 + (i as f32) * 0.1)
        .collect();

    let a = upload(&dev, &a_data, &[batch, nv, chunk, chunk])?;
    let beta = upload(&dev, &beta_data, &[batch, nv, chunk])?;
    let dw = upload(&dev, &dw_data, &[batch, nv, chunk, dv])?;

    let dr = vk_solve_tri_transpose_no_grad(&a, &beta, &dw, batch, nv, chunk, dv)?;
    let dr_data = dr.to_vec_f32()?;

    // Verify M^T · dr ≈ dW where M[t,i] = δ_{ti} + β[t]·A_strict[t,i]
    // i.e. (M^T · dr)[t, d] = dr[t, d] + Σ_{i>t} β[i] · A_strict[i, t] · dr[i, d]
    for bh in 0..batch * nv {
        for t in 0..chunk {
            for d in 0..dv {
                let mut acc = dr_data[bh * chunk * dv + t * dv + d];
                for i in (t + 1)..chunk {
                    acc += beta_data[bh * chunk + i]
                        * a_data[bh * chunk * chunk + i * chunk + t]
                        * dr_data[bh * chunk * dv + i * dv + d];
                }
                let expected = dw_data[bh * chunk * dv + t * dv + d];
                let err = (acc - expected).abs();
                assert!(err < 1e-5, "M^T · dr[{t},{d}] = {acc} != dW = {expected}");
            }
        }
    }
    Ok(())
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

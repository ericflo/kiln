//! Phase C.5 parity test: vk_sdpa_prefill forward + backward vs CPU reference.

use anyhow::Result;
use candle_core::{Device, Tensor, Var};
use kiln_vulkan_kernel::VulkanDevice;
use kiln_vulkan_kernel::vk_autograd::vk_backward;
use kiln_vulkan_kernel::vk_ops::attention::{vk_flash_sdpa_prefill_flat, vk_sdpa_prefill};
use kiln_vulkan_kernel::vk_ops::elementwise::vk_mul;
use kiln_vulkan_kernel::vk_ops::permute::{
    vk_permute_hr_to_rh_no_grad, vk_permute_rh_to_hr_no_grad, vk_repeat_kv_heads_no_grad,
};
use kiln_vulkan_kernel::vk_ops::reduce::vk_mean_all;
use kiln_vulkan_kernel::vk_ops::silu::vk_silu;
use kiln_vulkan_kernel::vk_tensor::VkTensor;
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
) -> Result<(Var, VkTensor)> {
    let t = Tensor::from_vec(data.to_vec(), shape.to_vec(), &Device::Cpu)?;
    let var = Var::from_tensor(&t)?;
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

/// CPU reference for SDPA causal prefill, GQA with `groups = heads_q/heads_kv`.
fn cpu_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    rows: usize,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let groups = heads_q / heads_kv;
    let mut out = vec![0.0_f32; rows * heads_q * head_dim];
    for h_q in 0..heads_q {
        let h_kv = h_q / groups;
        // scores[r, c] = sum_d q[r, h_q, d] * k[c, h_kv, d] * scale
        let mut scores = vec![0.0_f32; rows * rows];
        for r in 0..rows {
            for c in 0..rows {
                if c > r {
                    scores[r * rows + c] = -1.0e30;
                    continue;
                }
                let mut s = 0.0_f32;
                for d in 0..head_dim {
                    s += q[(r * heads_q + h_q) * head_dim + d]
                        * k[(c * heads_kv + h_kv) * head_dim + d];
                }
                scores[r * rows + c] = s * scale;
            }
        }
        // softmax per row
        for r in 0..rows {
            let mut mx = f32::NEG_INFINITY;
            for c in 0..rows {
                if scores[r * rows + c] > mx {
                    mx = scores[r * rows + c];
                }
            }
            let mut z = 0.0;
            for c in 0..rows {
                scores[r * rows + c] = (scores[r * rows + c] - mx).exp();
                z += scores[r * rows + c];
            }
            for c in 0..rows {
                scores[r * rows + c] /= z;
            }
        }
        // out[r, h_q, d] = sum_c scores[r, c] * v[c, h_kv, d]
        for r in 0..rows {
            for d in 0..head_dim {
                let mut acc = 0.0_f32;
                for c in 0..rows {
                    acc += scores[r * rows + c] * v[(c * heads_kv + h_kv) * head_dim + d];
                }
                out[(r * heads_q + h_q) * head_dim + d] = acc;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cpu_sdpa_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad_out: &[f32],
    out: &[f32],
    rows: usize,
    heads_q: usize,
    heads_kv: usize,
    head_dim: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let groups = heads_q / heads_kv;
    let mut dq = vec![0.0_f32; q.len()];
    let mut dk = vec![0.0_f32; k.len()];
    let mut dv = vec![0.0_f32; v.len()];
    for hq in 0..heads_q {
        let hkv = hq / groups;
        for row in 0..rows {
            let mut logits = vec![0.0_f32; row + 1];
            let mut max_logit = f32::NEG_INFINITY;
            for key_row in 0..=row {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[(row * heads_q + hq) * head_dim + d]
                        * k[(key_row * heads_kv + hkv) * head_dim + d];
                }
                let logit = dot * scale;
                logits[key_row] = logit;
                max_logit = max_logit.max(logit);
            }
            let mut denom = 0.0_f32;
            for logit in logits.iter_mut() {
                *logit = (*logit - max_logit).exp();
                denom += *logit;
            }
            let mut delta = 0.0_f32;
            for d in 0..head_dim {
                let idx = (row * heads_q + hq) * head_dim + d;
                delta += grad_out[idx] * out[idx];
            }
            for key_row in 0..=row {
                let p = logits[key_row] / denom;
                let mut dp = 0.0_f32;
                for d in 0..head_dim {
                    dp += grad_out[(row * heads_q + hq) * head_dim + d]
                        * v[(key_row * heads_kv + hkv) * head_dim + d];
                }
                let ds = p * (dp - delta);
                for d in 0..head_dim {
                    dq[(row * heads_q + hq) * head_dim + d] +=
                        ds * k[(key_row * heads_kv + hkv) * head_dim + d] * scale;
                    dk[(key_row * heads_kv + hkv) * head_dim + d] +=
                        ds * q[(row * heads_q + hq) * head_dim + d] * scale;
                    dv[(key_row * heads_kv + hkv) * head_dim + d] +=
                        p * grad_out[(row * heads_q + hq) * head_dim + d];
                }
            }
        }
    }
    (dq, dk, dv)
}

fn cpu_silu_grad_loss(out: &[f32]) -> Vec<f32> {
    let inv_n = 1.0_f32 / out.len() as f32;
    out.iter()
        .map(|&x| {
            let sig = if x >= 0.0 {
                1.0 / (1.0 + (-x).exp())
            } else {
                let e = x.exp();
                e / (1.0 + e)
            };
            let silu = x * sig;
            let silu_deriv = sig + silu * (1.0 - sig);
            2.0 * silu * silu_deriv * inv_n
        })
        .collect()
}

#[test]
fn vk_sdpa_forward_parity_gqa() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 6;
    let heads_q = 4;
    let heads_kv = 2;
    let head_dim = 8;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_data: Vec<f32> = (0..(rows * heads_q * head_dim))
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let k_data: Vec<f32> = (0..(rows * heads_kv * head_dim))
        .map(|i| ((i as f32) * 0.027).cos())
        .collect();
    let v_data: Vec<f32> = (0..(rows * heads_kv * head_dim))
        .map(|i| ((i as f32) * 0.041).sin() + 0.1)
        .collect();
    let q = upload_f32(&dev, &q_data, &[rows, heads_q, head_dim])?;
    let k = upload_f32(&dev, &k_data, &[rows, heads_kv, head_dim])?;
    let v = upload_f32(&dev, &v_data, &[rows, heads_kv, head_dim])?;
    let out_vk = vk_sdpa_prefill(&q, &k, &v, scale)?;
    assert_eq!(out_vk.shape(), &[rows, heads_q, head_dim]);
    let got = out_vk.to_vec_f32()?;
    let expected = cpu_sdpa(
        &q_data, &k_data, &v_data, rows, heads_q, heads_kv, head_dim, scale,
    );
    let mad = max_abs_diff(&got, &expected);
    assert!(mad < 1e-4, "sdpa forward mad {mad}");
    Ok(())
}

#[test]
fn vk_sdpa_backward_runs_through_three_params() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 4;
    let heads_q = 2;
    let heads_kv = 1;
    let head_dim = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_data: Vec<f32> = (0..(rows * heads_q * head_dim))
        .map(|i| 0.05 + (i as f32) * 0.01)
        .collect();
    let k_data: Vec<f32> = (0..(rows * heads_kv * head_dim))
        .map(|i| -0.03 + (i as f32) * 0.02)
        .collect();
    let v_data: Vec<f32> = (0..(rows * heads_kv * head_dim))
        .map(|i| 0.1 + (i as f32) * 0.005)
        .collect();
    let (_qv, q) = upload_param_f32(&dev, &q_data, &[rows, heads_q, head_dim])?;
    let (_kv, k) = upload_param_f32(&dev, &k_data, &[rows, heads_kv, head_dim])?;
    let (_vv, v) = upload_param_f32(&dev, &v_data, &[rows, heads_kv, head_dim])?;
    let out = vk_sdpa_prefill(&q, &k, &v, scale)?;
    // sanity loss: mean(silu(out)^2)
    let s = vk_silu(&out)?;
    let sq = vk_mul(&s, &s)?;
    let loss = vk_mean_all(&sq)?;
    let grads = vk_backward(&loss)?;
    // We don't need exact parity here (would require a candle reference),
    // but we *do* need all three param grads to be present and finite.
    assert_eq!(grads.len(), 3, "expected 3 param grads");
    for (_, g) in grads.iter() {
        let v = g.to_vec_f32()?;
        for x in v {
            assert!(x.is_finite(), "non-finite grad");
        }
    }
    Ok(())
}

#[test]
fn vk_flash_sdpa_flat_matches_cpu_forward_and_backward() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 5;
    let heads_q = 4;
    let heads_kv = 2;
    let head_dim = 8;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q_data: Vec<f32> = (0..(rows * heads_q * head_dim))
        .map(|i| ((i as f32) * 0.017).sin() * 0.25)
        .collect();
    let k_data: Vec<f32> = (0..(rows * heads_kv * head_dim))
        .map(|i| ((i as f32) * 0.023).cos() * 0.2)
        .collect();
    let v_data: Vec<f32> = (0..(rows * heads_kv * head_dim))
        .map(|i| ((i as f32) * 0.031).sin() * 0.15 + 0.05)
        .collect();

    let (_qv_f, q_f) = upload_param_f32(&dev, &q_data, &[rows, heads_q * head_dim])?;
    let (_kv_f, k_f) = upload_param_f32(&dev, &k_data, &[rows, heads_kv * head_dim])?;
    let (_vv_f, v_f) = upload_param_f32(&dev, &v_data, &[rows, heads_kv * head_dim])?;
    let out_f = vk_flash_sdpa_prefill_flat(&q_f, &k_f, &v_f, heads_q, heads_kv, head_dim, scale)?;

    let expected_out = cpu_sdpa(
        &q_data, &k_data, &v_data, rows, heads_q, heads_kv, head_dim, scale,
    );
    let fwd_mad = max_abs_diff(&out_f.to_vec_f32()?, &expected_out);
    assert!(fwd_mad < 2e-5, "flash sdpa forward mad {fwd_mad}");

    let loss_f = {
        let s = vk_silu(&out_f)?;
        let sq = vk_mul(&s, &s)?;
        vk_mean_all(&sq)?
    };
    let grads_f = vk_backward(&loss_f)?;
    let grad_out = cpu_silu_grad_loss(&expected_out);
    let (dq_expected, dk_expected, dv_expected) = cpu_sdpa_backward(
        &q_data,
        &k_data,
        &v_data,
        &grad_out,
        &expected_out,
        rows,
        heads_q,
        heads_kv,
        head_dim,
        scale,
    );

    for (name, pid_f, expected) in [
        ("q", q_f.param_id().unwrap(), dq_expected),
        ("k", k_f.param_id().unwrap(), dk_expected),
        ("v", v_f.param_id().unwrap(), dv_expected),
    ] {
        let gf = grads_f.get(pid_f).expect("flash grad missing");
        let mad = max_abs_diff(&gf.to_vec_f32()?, &expected);
        assert!(mad < 2e-5, "flash sdpa {name} backward mad {mad}");
    }
    Ok(())
}

#[test]
fn vk_permute_roundtrip_identity() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let rows = 5;
    let heads = 3;
    let head_dim = 7;
    let data: Vec<f32> = (0..(rows * heads * head_dim)).map(|i| i as f32).collect();
    let t = upload_f32(&dev, &data, &[rows, heads, head_dim])?;
    let perm = vk_permute_rh_to_hr_no_grad(&t)?;
    let back = vk_permute_hr_to_rh_no_grad(&perm)?;
    let got = back.to_vec_f32()?;
    assert_eq!(got, data);
    Ok(())
}

#[test]
fn vk_repeat_kv_heads_smoke() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };
    let heads_kv = 2;
    let groups = 3;
    let rows = 4;
    let head_dim = 5;
    let data: Vec<f32> = (0..(heads_kv * rows * head_dim))
        .map(|i| i as f32)
        .collect();
    let t = upload_f32(&dev, &data, &[heads_kv, rows, head_dim])?;
    let r = vk_repeat_kv_heads_no_grad(&t, groups)?;
    assert_eq!(r.shape(), &[heads_kv * groups, rows, head_dim]);
    let got = r.to_vec_f32()?;
    // Verify that head h_q maps to source head h_kv = h_q / groups.
    for h_q in 0..(heads_kv * groups) {
        let h_kv = h_q / groups;
        for r_i in 0..rows {
            for d in 0..head_dim {
                let g = got[(h_q * rows + r_i) * head_dim + d];
                let e = data[(h_kv * rows + r_i) * head_dim + d];
                assert!((g - e).abs() < 1e-6, "h_q={h_q} r={r_i} d={d}: {g} vs {e}");
            }
        }
    }
    Ok(())
}

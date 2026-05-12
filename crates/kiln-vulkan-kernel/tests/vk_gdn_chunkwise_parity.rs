//! Parity test for `vk_gdn_chunkwise_forward_no_grad` vs a CPU
//! per-token reference implementation of the gated DeltaNet recurrence.
//!
//! The chunkwise math is mathematically equivalent to the per-token
//! recurrence, so the GPU output must match within F32 tolerance.

#![cfg(test)]

use anyhow::Result;
use candle_core::{Device, Tensor};
use kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise_forward_no_grad;
use kiln_vulkan_kernel::{VkTensor, VulkanDevice};
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

/// Per-token reference, mirroring `gdn_single_token_recurrence` in
/// candle. Mutates `state` in place.
///
/// q, k: [B, nv, T, dk]   — batch flat indexing: ((b*nv+h)*T+t)*dk+i
/// v: [B, nv, T, dv]
/// beta, g: [B, nv, T]
/// state: [B, nv, dk, dv]
fn cpu_per_token_recurrence(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    state: &mut [f32],
    batch: usize,
    nv: usize,
    seq_len: usize,
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; batch * nv * seq_len * dv];
    for b in 0..batch {
        for h in 0..nv {
            for t in 0..seq_len {
                // Indices
                let qkt_off = ((b * nv + h) * seq_len + t) * dk;
                let v_off = ((b * nv + h) * seq_len + t) * dv;
                let bg_off = (b * nv + h) * seq_len + t;
                let s_off = (b * nv + h) * dk * dv;

                let g_t = g[bg_off];
                let beta_t = beta[bg_off];
                let p = g_t.exp();

                // ks = k_t · S_{t-1} → [dv]
                let mut ks = vec![0.0_f32; dv];
                let mut q_s = vec![0.0_f32; dv];
                for i in 0..dk {
                    let k_val = k[qkt_off + i];
                    let q_val = q[qkt_off + i];
                    for d in 0..dv {
                        let s_val = state[s_off + i * dv + d];
                        ks[d] += k_val * s_val;
                        q_s[d] += q_val * s_val;
                    }
                }

                // q_t · k_t → scalar
                let mut qk = 0.0_f32;
                for i in 0..dk {
                    qk += q[qkt_off + i] * k[qkt_off + i];
                }

                // v'_t = v_t - p * ks_t
                // w_t = β_t * v'_t
                // out_t = p * q_s_t + qk * w_t
                for d in 0..dv {
                    let v_prime = v[v_off + d] - p * ks[d];
                    let w = beta_t * v_prime;
                    out[v_off + d] = p * q_s[d] + qk * w;

                    // S_t = p * S_{t-1} + k_t.T · w_t (outer product)
                    // S[i, d] += k[i] * w
                    for i in 0..dk {
                        let s_idx = s_off + i * dv + d;
                        let new_s = p * state[s_idx] + k[qkt_off + i] * w;
                        state[s_idx] = new_s;
                    }
                }
            }
        }
    }
    out
}

#[test]
fn vk_gdn_chunkwise_matches_cpu_per_token_c64_t128() -> Result<()> {
    let Some(dev) = vk_dev() else { return Ok(()) };

    let batch = 1;
    let nv = 2;
    let seq_len = 128;
    let dk = 16;
    let dv = 16;
    let chunk_size = 64;

    // Deterministic synthetic input via simple LCG (avoids rand dep)
    let mk = |seed: u64, n: usize, scale: f32| -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9e3779b97f4a7c15) ^ 0xdeadbeefcafef00d;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let normalized = ((state >> 33) as f32) / (u32::MAX as f32);
                (normalized - 0.5) * 2.0 * scale
            })
            .collect()
    };
    let q_data = mk(11, batch * nv * seq_len * dk, 0.3);
    let k_data = mk(13, batch * nv * seq_len * dk, 0.3);
    let v_data = mk(17, batch * nv * seq_len * dv, 0.3);
    // β in (0, 1) via sigmoid-ish scaling; for the test, pick small positive
    let beta_data: Vec<f32> = mk(19, batch * nv * seq_len, 0.2)
        .iter()
        .map(|x| 0.5 + 0.4 * x)
        .collect();
    // g should be slightly negative (decay); range [-0.1, 0]
    let g_data: Vec<f32> = mk(23, batch * nv * seq_len, 0.05)
        .iter()
        .map(|x| -0.05 + x)
        .collect();
    let state_init = vec![0.0_f32; batch * nv * dk * dv];

    // CPU reference
    let mut cpu_state = state_init.clone();
    let cpu_out = cpu_per_token_recurrence(
        &q_data,
        &k_data,
        &v_data,
        &beta_data,
        &g_data,
        &mut cpu_state,
        batch,
        nv,
        seq_len,
        dk,
        dv,
    );

    // GPU vk-native
    let q = upload(&dev, &q_data, &[batch, nv, seq_len, dk])?;
    let k = upload(&dev, &k_data, &[batch, nv, seq_len, dk])?;
    let v = upload(&dev, &v_data, &[batch, nv, seq_len, dv])?;
    let beta = upload(&dev, &beta_data, &[batch, nv, seq_len])?;
    let g = upload(&dev, &g_data, &[batch, nv, seq_len])?;
    let mut state = upload(&dev, &state_init, &[batch, nv, dk, dv])?;

    let gpu_out = vk_gdn_chunkwise_forward_no_grad(&q, &k, &v, &beta, &g, &mut state, chunk_size)?;
    let gpu_out_data = gpu_out.to_vec_f32()?;
    let gpu_state_data = state.to_vec_f32()?;

    assert_eq!(gpu_out_data.len(), cpu_out.len(), "out shape mismatch");
    let max_err = gpu_out_data
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("vk_gdn_chunkwise out max abs err = {max_err:.6e}");
    assert!(
        max_err < 5e-4,
        "vk-native chunkwise out max err {max_err} exceeds 5e-4 tolerance"
    );

    let state_err = gpu_state_data
        .iter()
        .zip(cpu_state.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("vk_gdn_chunkwise state max abs err = {state_err:.6e}");
    assert!(
        state_err < 5e-4,
        "vk-native chunkwise state max err {state_err} exceeds 5e-4 tolerance"
    );

    Ok(())
}

#[test]
fn vk_chunk_prep_isolated() -> Result<()> {
    use kiln_vulkan_kernel::vk_ops::gdn_chunk_prep::vk_gdn_chunk_prep_no_grad;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 1;
    let nv = 1;
    let chunk = 2;
    let dv = 2;
    let g = upload(
        &dev,
        &vec![-0.01_f32; batch * nv * chunk],
        &[batch, nv, chunk],
    )?;
    let v = upload(
        &dev,
        &vec![0.1_f32; batch * nv * chunk * dv],
        &[batch, nv, chunk, dv],
    )?;
    let kkt = upload(
        &dev,
        &vec![0.05_f32; batch * nv * chunk * chunk],
        &[batch, nv, chunk, chunk],
    )?;
    let qkt = upload(
        &dev,
        &vec![0.05_f32; batch * nv * chunk * chunk],
        &[batch, nv, chunk, chunk],
    )?;
    let ks = upload(
        &dev,
        &vec![0.0_f32; batch * nv * chunk * dv],
        &[batch, nv, chunk, dv],
    )?;
    let qs = upload(
        &dev,
        &vec![0.0_f32; batch * nv * chunk * dv],
        &[batch, nv, chunk, dv],
    )?;
    let out = vk_gdn_chunk_prep_no_grad(&g, &v, &kkt, &qkt, &ks, &qs, batch, nv, chunk, dv)?;
    let v_prime = out.v_prime.to_vec_f32()?;
    let p_last = out.p_last.to_vec_f32()?;
    println!("v_prime = {v_prime:?}");
    println!("p_last  = {p_last:?}");
    assert!(v_prime.iter().all(|x| x.is_finite()));
    Ok(())
}

#[test]
fn vk_solve_tri_isolated() -> Result<()> {
    use kiln_vulkan_kernel::vk_ops::solve_tri::vk_solve_tri_no_grad;
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 1;
    let nv = 1;
    let chunk = 2;
    let dv = 2;
    let a_strict = upload(
        &dev,
        &vec![0.0_f32; batch * nv * chunk * chunk],
        &[batch, nv, chunk, chunk],
    )?;
    let v_prime = upload(
        &dev,
        &vec![0.1_f32; batch * nv * chunk * dv],
        &[batch, nv, chunk, dv],
    )?;
    let beta = upload(
        &dev,
        &vec![0.5_f32; batch * nv * chunk],
        &[batch, nv, chunk],
    )?;
    let w = vk_solve_tri_no_grad(&a_strict, &v_prime, &beta, batch, nv, chunk, dv)?;
    let data = w.to_vec_f32()?;
    println!("solve_tri w = {data:?}");
    assert!(data.iter().all(|x| x.is_finite()));
    Ok(())
}

#[test]
fn vk_gdn_chunkwise_minimal_sanity() -> Result<()> {
    // Smallest possible non-trivial input to isolate where the dispatch
    // chain breaks if it does.
    let Some(dev) = vk_dev() else { return Ok(()) };
    let batch = 1;
    let nv = 1;
    let seq_len = 2;
    let dk = 2;
    let dv = 2;
    let chunk_size = 2;

    let q_data = vec![0.1_f32; batch * nv * seq_len * dk];
    let k_data = vec![0.1_f32; batch * nv * seq_len * dk];
    let v_data = vec![0.1_f32; batch * nv * seq_len * dv];
    let beta_data = vec![0.5_f32; batch * nv * seq_len];
    let g_data = vec![-0.01_f32; batch * nv * seq_len];
    let state_init = vec![0.0_f32; batch * nv * dk * dv];

    let q = upload(&dev, &q_data, &[batch, nv, seq_len, dk])?;
    let k = upload(&dev, &k_data, &[batch, nv, seq_len, dk])?;
    let v = upload(&dev, &v_data, &[batch, nv, seq_len, dv])?;
    let beta = upload(&dev, &beta_data, &[batch, nv, seq_len])?;
    let g = upload(&dev, &g_data, &[batch, nv, seq_len])?;
    let mut state = upload(&dev, &state_init, &[batch, nv, dk, dv])?;

    let gpu_out = vk_gdn_chunkwise_forward_no_grad(&q, &k, &v, &beta, &g, &mut state, chunk_size)?;
    let data = gpu_out.to_vec_f32()?;
    println!("minimal out = {data:?}");
    assert_eq!(data.len(), batch * nv * seq_len * dv);
    for v in &data {
        assert!(v.is_finite(), "non-finite output: {v}");
    }
    Ok(())
}

#[test]
fn vk_gdn_chunkwise_autograd_smoke() -> Result<()> {
    use kiln_vulkan_kernel::vk_autograd::vk_backward;
    use kiln_vulkan_kernel::vk_ops::gdn_chunkwise::vk_gdn_chunkwise;
    use kiln_vulkan_kernel::vk_ops::reduce::vk_mean_all;
    let Some(dev) = vk_dev() else { return Ok(()) };

    let batch = 1;
    let nv = 1;
    let seq_len = 8;
    let dk = 4;
    let dv = 4;
    let chunk_size = 8;

    // Build VkTensors that participate in autograd: synthesize them as
    // outputs of a no-op upload (we don't need them as parameters here;
    // we just want the autograd tape to terminate at the GDN bwd op
    // and verify gradients shape out correctly).
    use candle_core::{Device, Tensor, Var};
    let mk_param = |seed: u64, n: usize, shape: Vec<usize>| -> Result<VkTensor> {
        let data: Vec<f32> = (0..n)
            .map(|i| (((i + seed as usize) as f32) * 0.05).sin())
            .collect();
        let t = Tensor::from_vec(data, shape.clone(), &Device::Cpu)?;
        let var = Var::from_tensor(&t)?;
        let vk = VkTensor::from_candle(&t, Arc::clone(&dev))?;
        Ok(VkTensor::parameter(
            Arc::clone(vk.buffer()),
            shape,
            vk.dtype(),
            Arc::clone(vk.device()),
            var.id(),
        ))
    };

    let q = mk_param(1, batch * nv * seq_len * dk, vec![batch, nv, seq_len, dk])?;
    let k = mk_param(2, batch * nv * seq_len * dk, vec![batch, nv, seq_len, dk])?;
    let v = mk_param(3, batch * nv * seq_len * dv, vec![batch, nv, seq_len, dv])?;
    let beta = mk_param(4, batch * nv * seq_len, vec![batch, nv, seq_len])?;
    let g = mk_param(5, batch * nv * seq_len, vec![batch, nv, seq_len])?;
    let mut state = upload(
        &dev,
        &vec![0.0_f32; batch * nv * dk * dv],
        &[batch, nv, dk, dv],
    )?;

    let out = vk_gdn_chunkwise(&q, &k, &v, &beta, &g, &mut state, chunk_size)?;
    assert!(out.requires_grad());
    let loss = vk_mean_all(&out)?;
    let grads = vk_backward(&loss)?;
    println!("vk_gdn_chunkwise autograd smoke: grads={}", grads.len());
    // Expect 5 gradients: q, k, v, beta, g
    assert!(grads.len() >= 5, "expected ≥5 grads, got {}", grads.len());
    Ok(())
}

#[test]
fn vk_gdn_chunkwise_matches_cpu_per_token_c8_t8() -> Result<()> {
    // Smaller smoke test exercising 1 chunk only (T == chunk_size)
    let Some(dev) = vk_dev() else { return Ok(()) };

    let batch = 1;
    let nv = 1;
    let seq_len = 8;
    let dk = 4;
    let dv = 4;
    let chunk_size = 8;

    let q_data: Vec<f32> = (0..(batch * nv * seq_len * dk))
        .map(|i| ((i as f32) * 0.05).sin())
        .collect();
    let k_data: Vec<f32> = (0..(batch * nv * seq_len * dk))
        .map(|i| ((i as f32) * 0.07 + 0.1).cos())
        .collect();
    let v_data: Vec<f32> = (0..(batch * nv * seq_len * dv))
        .map(|i| ((i as f32) * 0.03 - 0.2).sin())
        .collect();
    let beta_data: Vec<f32> = (0..(batch * nv * seq_len)).map(|_| 0.5).collect();
    let g_data: Vec<f32> = (0..(batch * nv * seq_len)).map(|_| -0.02).collect();
    let state_init = vec![0.0_f32; batch * nv * dk * dv];

    let mut cpu_state = state_init.clone();
    let cpu_out = cpu_per_token_recurrence(
        &q_data,
        &k_data,
        &v_data,
        &beta_data,
        &g_data,
        &mut cpu_state,
        batch,
        nv,
        seq_len,
        dk,
        dv,
    );

    let q = upload(&dev, &q_data, &[batch, nv, seq_len, dk])?;
    let k = upload(&dev, &k_data, &[batch, nv, seq_len, dk])?;
    let v = upload(&dev, &v_data, &[batch, nv, seq_len, dv])?;
    let beta = upload(&dev, &beta_data, &[batch, nv, seq_len])?;
    let g = upload(&dev, &g_data, &[batch, nv, seq_len])?;
    let mut state = upload(&dev, &state_init, &[batch, nv, dk, dv])?;

    let gpu_out = vk_gdn_chunkwise_forward_no_grad(&q, &k, &v, &beta, &g, &mut state, chunk_size)?;
    let gpu_out_data = gpu_out.to_vec_f32()?;
    let max_err = gpu_out_data
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    println!("vk_gdn_chunkwise small-test out max abs err = {max_err:.6e}");
    assert!(max_err < 1e-4, "small-test out max err {max_err}");
    Ok(())
}

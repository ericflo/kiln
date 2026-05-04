use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, shape::ShapeWithOneHole};
use kiln_vulkan_kernel::VulkanDevice;

fn cpu_bf16(data: Vec<f32>, shape: impl ShapeWithOneHole) -> Result<Tensor> {
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?.to_dtype(DType::BF16)?)
}

fn cpu_f32(data: Vec<f32>, shape: impl ShapeWithOneHole) -> Result<Tensor> {
    Ok(Tensor::from_vec(data, shape, &Device::Cpu)?)
}

fn tensor_data_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn assert_close(name: &str, got: &Tensor, expected: &Tensor, tol: f32) -> Result<()> {
    let got = tensor_data_f32(got)?;
    let expected = tensor_data_f32(expected)?;
    anyhow::ensure!(
        got.len() == expected.len(),
        "{name}: len mismatch got {} expected {}",
        got.len(),
        expected.len()
    );
    let mut worst = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for (idx, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        if diff > worst.3 {
            worst = (idx, g, e, diff);
        }
    }
    anyhow::ensure!(
        worst.3 <= tol,
        "{name}: max abs diff {} at {} (got {}, expected {}) > {}",
        worst.3,
        worst.0,
        worst.1,
        worst.2,
        tol
    );
    Ok(())
}

fn silu_f32(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

fn causal_conv1d_reference(
    x: &[f32],
    weight: &[f32],
    state: &[f32],
    batch: usize,
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let state_len = kernel_size - 1;
    let mut out = vec![0.0; batch * channels * seq_len];
    let mut new_state = state.to_vec();

    for b in 0..batch {
        for c in 0..channels {
            let state_base = (b * channels + c) * state_len;
            let x_base = (b * channels + c) * seq_len;
            for t in 0..seq_len {
                let mut sum = 0.0f32;
                for k in 0..kernel_size {
                    let logical_t = t + k;
                    let x_val = if logical_t < state_len {
                        state[state_base + logical_t]
                    } else {
                        x[x_base + logical_t - state_len]
                    };
                    sum += x_val * weight[c * kernel_size + k];
                }
                out[x_base + t] = silu_f32(sum);
            }

            if seq_len >= state_len {
                let x_start = seq_len - state_len;
                for s in 0..state_len {
                    new_state[state_base + s] = x[x_base + x_start + s];
                }
            } else {
                let keep = state_len - seq_len;
                for s in 0..keep {
                    new_state[state_base + s] = state[state_base + seq_len + s];
                }
                for s in keep..state_len {
                    new_state[state_base + s] = x[x_base + s - keep];
                }
            }
        }
    }

    (out, new_state)
}

fn maybe_vulkan() -> Option<VulkanDevice> {
    VulkanDevice::new().ok()
}

#[test]
fn gdn_in_proj_decode_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (hidden, qkv_dim, z_dim, a_dim, b_dim) = (8usize, 5usize, 3usize, 2usize, 4usize);
    let x = Tensor::from_vec(
        (0..hidden)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.125)
            .collect::<Vec<_>>(),
        (1, 1, hidden),
        &Device::Cpu,
    )?;
    let make_weight = |out_dim: usize, scale: f32| -> Result<Tensor> {
        Tensor::from_vec(
            (0..hidden * out_dim)
                .map(|i| ((i as f32 % 11.0) - 5.0) * scale)
                .collect::<Vec<_>>(),
            (hidden, out_dim),
            &Device::Cpu,
        )
        .map_err(Into::into)
    };
    let qkv_w = make_weight(qkv_dim, 0.03125)?;
    let z_w = make_weight(z_dim, 0.043)?;
    let a_w = make_weight(a_dim, -0.027)?;
    let b_w = make_weight(b_dim, 0.019)?;

    let qkv_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &qkv_w)?;
    let z_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &z_w)?;
    let a_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &a_w)?;
    let b_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &b_w)?;

    let (got_qkv, got_z, got_a, got_b) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached(
            &vk, &x, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim, z_dim, a_dim, b_dim,
        )
        .context("dispatch_gdn_in_proj_decode_cached")?;

    assert_close("in_proj qkv", &got_qkv, &x.broadcast_matmul(&qkv_w)?, 1e-5)?;
    assert_close("in_proj z", &got_z, &x.broadcast_matmul(&z_w)?, 1e-5)?;
    assert_close("in_proj a", &got_a, &x.broadcast_matmul(&a_w)?, 1e-5)?;
    assert_close("in_proj b", &got_b, &x.broadcast_matmul(&b_w)?, 1e-5)?;
    Ok(())
}

#[test]
fn gdn_in_proj_decode_batched_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, hidden, qkv_dim, z_dim, a_dim, b_dim) =
        (3usize, 8usize, 5usize, 3usize, 2usize, 4usize);
    let x = Tensor::from_vec(
        (0..batch * hidden)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.071)
            .collect::<Vec<_>>(),
        (batch, 1, hidden),
        &Device::Cpu,
    )?;
    let make_weight = |out_dim: usize, scale: f32| -> Result<Tensor> {
        Tensor::from_vec(
            (0..hidden * out_dim)
                .map(|i| ((i as f32 % 11.0) - 5.0) * scale)
                .collect::<Vec<_>>(),
            (hidden, out_dim),
            &Device::Cpu,
        )
        .map_err(Into::into)
    };
    let qkv_w = make_weight(qkv_dim, 0.03125)?;
    let z_w = make_weight(z_dim, 0.043)?;
    let a_w = make_weight(a_dim, -0.027)?;
    let b_w = make_weight(b_dim, 0.019)?;

    let qkv_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &qkv_w)?;
    let z_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &z_w)?;
    let a_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &a_w)?;
    let b_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &b_w)?;

    let (got_qkv, got_z, got_a, got_b) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_in_proj_decode_cached(
            &vk, &x, &qkv_buf, &z_buf, &a_buf, &b_buf, hidden, qkv_dim, z_dim, a_dim, b_dim,
        )
        .context("dispatch_gdn_in_proj_decode_cached batched")?;

    assert_close(
        "batched in_proj qkv",
        &got_qkv,
        &x.broadcast_matmul(&qkv_w)?,
        1e-5,
    )?;
    assert_close(
        "batched in_proj z",
        &got_z,
        &x.broadcast_matmul(&z_w)?,
        1e-5,
    )?;
    assert_close(
        "batched in_proj a",
        &got_a,
        &x.broadcast_matmul(&a_w)?,
        1e-5,
    )?;
    assert_close(
        "batched in_proj b",
        &got_b,
        &x.broadcast_matmul(&b_w)?,
        1e-5,
    )?;
    Ok(())
}

#[test]
fn linear_decode_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (hidden, out_dim) = (9usize, 7usize);
    let x = cpu_f32(
        (0..hidden)
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.17)
            .collect(),
        (1, 1, hidden),
    )?;
    let weight = cpu_f32(
        (0..hidden * out_dim)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.023)
            .collect(),
        (hidden, out_dim),
    )?;
    let weight_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &weight)?;
    let got = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached(
        &vk,
        &x,
        &weight_buf,
        1,
        hidden,
        out_dim,
    )
    .context("dispatch_linear_decode_cached")?;
    assert_close("linear decode", &got, &x.broadcast_matmul(&weight)?, 1e-5)?;
    Ok(())
}

#[test]
fn linear_decode_batched_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, hidden, out_dim) = (4usize, 11usize, 9usize);
    let x = cpu_f32(
        (0..batch * hidden)
            .map(|i| ((i as f32 % 19.0) - 9.0) * 0.071)
            .collect(),
        (batch, 1, hidden),
    )?;
    let weight = cpu_f32(
        (0..hidden * out_dim)
            .map(|i| ((i as f32 % 17.0) - 8.0) * -0.013)
            .collect(),
        (hidden, out_dim),
    )?;
    let weight_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &weight)?;
    let got = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached(
        &vk,
        &x,
        &weight_buf,
        batch,
        hidden,
        out_dim,
    )
    .context("dispatch_linear_decode_cached batched")?;
    assert_close(
        "linear decode batched",
        &got,
        &x.broadcast_matmul(&weight)?,
        1e-5,
    )?;
    Ok(())
}

#[test]
fn linear_decode_argmax_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (hidden, out_dim) = (10usize, 35usize);
    let x = cpu_f32(
        (0..hidden)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.13)
            .collect(),
        (1, 1, hidden),
    )?;
    let weight = cpu_f32(
        (0..hidden * out_dim)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.019)
            .collect(),
        (hidden, out_dim),
    )?;
    let weight_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &weight)?;
    let got = kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_cached(
        &vk,
        &x,
        &weight_buf,
        hidden,
        out_dim,
    )
    .context("dispatch_linear_decode_argmax_cached")?;

    let logits = tensor_data_f32(&x.broadcast_matmul(&weight)?)?;
    let (expected, _) =
        logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (idx, &score)| {
                if score > best.1 { (idx, score) } else { best }
            });
    anyhow::ensure!(
        got == expected as u32,
        "linear argmax mismatch got {got} expected {expected}"
    );
    Ok(())
}

#[test]
fn linear_decode_argmax_batched_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, hidden, out_dim) = (4usize, 13usize, 71usize);
    let x = cpu_f32(
        (0..batch * hidden)
            .map(|i| ((i as f32 % 23.0) - 11.0) * 0.047)
            .collect(),
        (batch, 1, hidden),
    )?;
    let weight = cpu_f32(
        (0..hidden * out_dim)
            .map(|i| ((i as f32 % 29.0) - 14.0) * -0.017)
            .collect(),
        (hidden, out_dim),
    )?;
    let weight_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &weight)?;
    let got = kiln_vulkan_kernel::kernels::dispatch_linear_decode_argmax_batched_cached(
        &vk,
        &x,
        &weight_buf,
        batch,
        hidden,
        out_dim,
    )
    .context("dispatch_linear_decode_argmax_batched_cached")?;

    let logits = tensor_data_f32(&x.broadcast_matmul(&weight)?)?;
    let expected: Vec<u32> = (0..batch)
        .map(|row| {
            let start = row * out_dim;
            let row_logits = &logits[start..start + out_dim];
            let (token, _) = row_logits.iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |best, (idx, &score)| {
                    if score > best.1 { (idx, score) } else { best }
                },
            );
            token as u32
        })
        .collect();
    anyhow::ensure!(
        got == expected,
        "batched linear argmax mismatch got {:?} expected {:?}",
        got,
        expected
    );
    Ok(())
}

#[test]
fn causal_conv1d_update_matches_stateful_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, channels, seq_len, kernel_size) = (1usize, 2usize, 1usize, 4usize);
    let x_data = vec![0.35, -0.42];
    let weight_data = vec![0.25, -0.5, 0.75, 0.125, -0.2, 0.4, -0.6, 0.8];
    let state_data = vec![1.0, -2.0, 3.0, -0.5, 0.75, -1.25];

    let x = cpu_f32(x_data.clone(), (batch, channels, seq_len))?;
    let weight = cpu_f32(weight_data.clone(), (channels, 1, kernel_size))?;
    let state = cpu_f32(state_data.clone(), (batch, channels, kernel_size - 1))?;
    let (got_out, got_state) = kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_update(
        &vk,
        &x,
        &weight,
        &state,
        kernel_size,
    )
    .context("dispatch_causal_conv1d_update")?;

    let (exp_out, exp_state) = causal_conv1d_reference(
        &x_data,
        &weight_data,
        &state_data,
        batch,
        channels,
        seq_len,
        kernel_size,
    );
    assert_close(
        "causal conv1d update out",
        &got_out,
        &cpu_f32(exp_out, (batch, channels, seq_len))?,
        1e-5,
    )?;
    assert_close(
        "causal conv1d update state",
        &got_state,
        &cpu_f32(exp_state, (batch, channels, kernel_size - 1))?,
        1e-5,
    )?;
    Ok(())
}

#[test]
fn causal_conv1d_prefill_matches_stateful_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    for seq_len in [2usize, 5usize] {
        let (batch, channels, kernel_size) = (1usize, 2usize, 4usize);
        let x_data = (0..batch * channels * seq_len)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.19)
            .collect::<Vec<_>>();
        let weight_data = vec![0.25, -0.5, 0.75, 0.125, -0.2, 0.4, -0.6, 0.8];
        let state_data = vec![1.0, -2.0, 3.0, -0.5, 0.75, -1.25];

        let x = cpu_f32(x_data.clone(), (batch, channels, seq_len))?;
        let weight = cpu_f32(weight_data.clone(), (channels, 1, kernel_size))?;
        let state = cpu_f32(state_data.clone(), (batch, channels, kernel_size - 1))?;
        let (got_out, got_state) = kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill(
            &vk,
            &x,
            &weight,
            &state,
            kernel_size,
        )
        .with_context(|| format!("dispatch_causal_conv1d_prefill seq_len={seq_len}"))?;

        let (exp_out, exp_state) = causal_conv1d_reference(
            &x_data,
            &weight_data,
            &state_data,
            batch,
            channels,
            seq_len,
            kernel_size,
        );
        assert_close(
            &format!("causal conv1d prefill out seq_len={seq_len}"),
            &got_out,
            &cpu_f32(exp_out, (batch, channels, seq_len))?,
            1e-5,
        )?;
        assert_close(
            &format!("causal conv1d prefill state seq_len={seq_len}"),
            &got_state,
            &cpu_f32(exp_state, (batch, channels, kernel_size - 1))?,
            1e-5,
        )?;
    }
    Ok(())
}

#[test]
fn full_attn_qkv_decode_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (hidden, q_dim, k_dim, v_dim) = (8usize, 7usize, 3usize, 4usize);
    let x = cpu_f32(
        (0..hidden)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.11)
            .collect(),
        (1, 1, hidden),
    )?;
    let make_weight = |out_dim: usize, scale: f32| -> Result<Tensor> {
        cpu_f32(
            (0..hidden * out_dim)
                .map(|i| ((i as f32 % 13.0) - 6.0) * scale)
                .collect(),
            (hidden, out_dim),
        )
    };
    let q_w = make_weight(q_dim, 0.019)?;
    let k_w = make_weight(k_dim, -0.031)?;
    let v_w = make_weight(v_dim, 0.023)?;
    let q_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &q_w)?;
    let k_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &k_w)?;
    let v_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &v_w)?;
    let (got_q, got_k, got_v) = kiln_vulkan_kernel::kernels::dispatch_full_attn_qkv_decode_cached(
        &vk, &x, &q_buf, &k_buf, &v_buf, hidden, q_dim, k_dim, v_dim,
    )
    .context("dispatch_full_attn_qkv_decode_cached")?;

    assert_close("full attn q", &got_q, &x.broadcast_matmul(&q_w)?, 1e-5)?;
    assert_close("full attn k", &got_k, &x.broadcast_matmul(&k_w)?, 1e-5)?;
    assert_close("full attn v", &got_v, &x.broadcast_matmul(&v_w)?, 1e-5)?;
    Ok(())
}

#[test]
fn mlp_gate_up_decode_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (hidden, intermediate) = (6usize, 5usize);
    let x = cpu_f32(
        (0..hidden)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.13)
            .collect(),
        (1, 1, hidden),
    )?;
    let gate_w = cpu_f32(
        (0..hidden * intermediate)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.029)
            .collect(),
        (hidden, intermediate),
    )?;
    let up_w = cpu_f32(
        (0..hidden * intermediate)
            .map(|i| ((i as f32 % 13.0) - 6.0) * -0.017)
            .collect(),
        (hidden, intermediate),
    )?;
    let gate_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &gate_w)?;
    let up_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &up_w)?;
    let got = kiln_vulkan_kernel::kernels::dispatch_mlp_gate_up_decode_cached(
        &vk,
        &x,
        &gate_buf,
        &up_buf,
        hidden,
        intermediate,
    )
    .context("dispatch_mlp_gate_up_decode_cached")?;

    let gate = x.broadcast_matmul(&gate_w)?;
    let up = x.broadcast_matmul(&up_w)?;
    let sigmoid = (gate.neg()?.exp()? + 1.0)?.recip()?;
    let silu = (gate * sigmoid)?;
    let expected = (silu * up)?;
    assert_close("mlp gate/up decode", &got, &expected, 1e-5)?;
    Ok(())
}

#[test]
fn mlp_decode_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (hidden, intermediate, out_dim) = (6usize, 5usize, 4usize);
    let x = cpu_f32(
        (0..hidden)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.13)
            .collect(),
        (1, 1, hidden),
    )?;
    let gate_w = cpu_f32(
        (0..hidden * intermediate)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.029)
            .collect(),
        (hidden, intermediate),
    )?;
    let up_w = cpu_f32(
        (0..hidden * intermediate)
            .map(|i| ((i as f32 % 13.0) - 6.0) * -0.017)
            .collect(),
        (hidden, intermediate),
    )?;
    let down_w = cpu_f32(
        (0..intermediate * out_dim)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.011)
            .collect(),
        (intermediate, out_dim),
    )?;
    let gate_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &gate_w)?;
    let up_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &up_w)?;
    let down_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &down_w)?;
    let got = kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached(
        &vk,
        &x,
        &gate_buf,
        &up_buf,
        &down_buf,
        hidden,
        intermediate,
        out_dim,
    )
    .context("dispatch_mlp_decode_cached")?;

    let gate = x.broadcast_matmul(&gate_w)?;
    let up = x.broadcast_matmul(&up_w)?;
    let sigmoid = (gate.neg()?.exp()? + 1.0)?.recip()?;
    let hidden_t = ((gate * sigmoid)? * up)?;
    let expected = hidden_t.broadcast_matmul(&down_w)?;
    assert_close("mlp decode", &got, &expected, 1e-5)?;
    Ok(())
}

#[test]
fn mlp_decode_batched_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, hidden, intermediate, out_dim) = (3usize, 6usize, 5usize, 4usize);
    let x = cpu_f32(
        (0..batch * hidden)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.047)
            .collect(),
        (batch, 1, hidden),
    )?;
    let gate_w = cpu_f32(
        (0..hidden * intermediate)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.029)
            .collect(),
        (hidden, intermediate),
    )?;
    let up_w = cpu_f32(
        (0..hidden * intermediate)
            .map(|i| ((i as f32 % 13.0) - 6.0) * -0.017)
            .collect(),
        (hidden, intermediate),
    )?;
    let down_w = cpu_f32(
        (0..intermediate * out_dim)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.011)
            .collect(),
        (intermediate, out_dim),
    )?;
    let gate_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &gate_w)?;
    let up_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &up_w)?;
    let down_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &down_w)?;

    let got_gate_up = kiln_vulkan_kernel::kernels::dispatch_mlp_gate_up_decode_cached(
        &vk,
        &x,
        &gate_buf,
        &up_buf,
        hidden,
        intermediate,
    )
    .context("dispatch_mlp_gate_up_decode_cached batched")?;
    let got = kiln_vulkan_kernel::kernels::dispatch_mlp_decode_cached(
        &vk,
        &x,
        &gate_buf,
        &up_buf,
        &down_buf,
        hidden,
        intermediate,
        out_dim,
    )
    .context("dispatch_mlp_decode_cached batched")?;

    let gate = x.broadcast_matmul(&gate_w)?;
    let up = x.broadcast_matmul(&up_w)?;
    let sigmoid = (gate.neg()?.exp()? + 1.0)?.recip()?;
    let hidden_t = ((gate * sigmoid)? * up)?;
    let expected = hidden_t.broadcast_matmul(&down_w)?;
    assert_close("mlp gate/up batched", &got_gate_up, &hidden_t, 1e-5)?;
    assert_close("mlp decode batched", &got, &expected, 1e-5)?;
    Ok(())
}

#[test]
fn gdn_gates_and_gated_rms_norm_match_f32_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let a = cpu_f32(vec![-0.3, 0.2, 1.1, -1.4], (1, 2, 2))?;
    let b = cpu_f32(vec![0.4, -0.7, 1.3, -1.2], (1, 2, 2))?;
    let a_log = cpu_f32(vec![0.1, -0.2], (2,))?;
    let dt_bias = cpu_f32(vec![0.05, -0.15], (2,))?;

    let (beta, g) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_gates(&vk, &a, &b, &a_log, &dt_bias, &[1, 2, 2])
            .context("dispatch_gdn_gates")?;
    let a_log_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &a_log)?;
    let dt_bias_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &dt_bias)?;
    let (beta_cached, g_cached) = kiln_vulkan_kernel::kernels::dispatch_gdn_gates_cached(
        &vk,
        &a,
        &b,
        &a_log_buf,
        &dt_bias_buf,
        2,
        &[1, 2, 2],
    )
    .context("dispatch_gdn_gates_cached")?;

    let ad = tensor_data_f32(&a)?;
    let bd = tensor_data_f32(&b)?;
    let ald = tensor_data_f32(&a_log)?;
    let dbd = tensor_data_f32(&dt_bias)?;
    let mut exp_beta = vec![0.0; bd.len()];
    let mut exp_g = vec![0.0; ad.len()];
    for i in 0..ad.len() {
        let head = i % 2;
        exp_beta[i] = 1.0 / (1.0 + (-bd[i]).exp());
        let biased = ad[i] + dbd[head];
        let softplus = biased.max(0.0) + (-biased.abs()).exp().ln_1p();
        exp_g[i] = -ald[head].exp() * softplus;
    }
    assert_close(
        "gates beta f32",
        &beta,
        &cpu_f32(exp_beta, (1, 2, 2))?,
        1e-5,
    )?;
    assert_close("gates g f32", &g, &cpu_f32(exp_g, (1, 2, 2))?, 1e-5)?;
    assert_close("gates cached beta f32", &beta_cached, &beta, 1e-5)?;
    assert_close("gates cached g f32", &g_cached, &g, 1e-5)?;

    let x = cpu_f32(vec![0.2, -0.4, 0.8, -1.2, 1.1, -0.9, 0.5, -0.3], (1, 2, 4))?;
    let z = cpu_f32(vec![-0.5, 0.7, 1.4, -1.6, 0.3, -0.8, 1.2, -0.2], (1, 2, 4))?;
    let weight = cpu_f32(vec![0.9, -0.7, 1.3, 0.5], (4,))?;
    let got_norm = kiln_vulkan_kernel::kernels::dispatch_gdn_gated_rms_norm(
        &vk,
        &x,
        &z,
        &weight,
        1e-6,
        &[1, 2, 4],
    )
    .context("dispatch_gdn_gated_rms_norm")?;
    let weight_buf = kiln_vulkan_kernel::kernels::upload_tensor_f32_buffer(&vk, &weight)?;
    let got_norm_cached = kiln_vulkan_kernel::kernels::dispatch_gdn_gated_rms_norm_cached(
        &vk,
        &x,
        &z,
        &weight_buf,
        4,
        1e-6,
        &[1, 2, 4],
    )
    .context("dispatch_gdn_gated_rms_norm_cached")?;

    let xd = tensor_data_f32(&x)?;
    let zd = tensor_data_f32(&z)?;
    let wd = tensor_data_f32(&weight)?;
    let mut exp = vec![0.0; xd.len()];
    for row in 0..2 {
        let base = row * 4;
        let mean_sq = (0..4).map(|i| xd[base + i] * xd[base + i]).sum::<f32>() / 4.0;
        let rms_inv = (mean_sq + 1e-6).sqrt().recip();
        for i in 0..4 {
            let zv = zd[base + i];
            let sigmoid = if zv >= 0.0 {
                1.0 / (1.0 + (-zv).exp())
            } else {
                let ez = zv.exp();
                ez / (1.0 + ez)
            };
            exp[base + i] = xd[base + i] * rms_inv * wd[i] * zv * sigmoid;
        }
    }
    assert_close(
        "gated rms norm f32",
        &got_norm,
        &cpu_f32(exp, (1, 2, 4))?,
        1e-5,
    )?;
    assert_close(
        "gated rms norm cached f32",
        &got_norm_cached,
        &got_norm,
        1e-5,
    )?;
    Ok(())
}

#[test]
fn gdn_recurrent_step_matches_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, heads, dk, dv) = (1usize, 2usize, 8usize, 6usize);
    let q = cpu_bf16(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.03125)
            .collect(),
        (batch, heads, dk),
    )?;
    let k = cpu_bf16(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.041)
            .collect(),
        (batch, heads, dk),
    )?;
    let v = cpu_bf16(
        (0..batch * heads * dv)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.027)
            .collect(),
        (batch, heads, dv),
    )?;
    let beta = cpu_bf16(vec![0.35, 0.72], (batch, heads))?;
    let g = cpu_bf16(vec![-0.11, -0.23], (batch, heads))?;
    let state = cpu_bf16(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.009)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let (got_out, got_state) = kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step(
        &vk, &q, &k, &v, &beta, &g, &state,
    )
    .context("dispatch_gdn_recurrent_step")?;

    let qd = tensor_data_f32(&q)?;
    let kd = tensor_data_f32(&k)?;
    let vd = tensor_data_f32(&v)?;
    let bd = tensor_data_f32(&beta)?;
    let gd = tensor_data_f32(&g)?;
    let sd = tensor_data_f32(&state)?;
    let mut expected_out = vec![0.0f32; batch * heads * dv];
    let mut expected_state = sd.clone();

    for b in 0..batch {
        for h in 0..heads {
            let bh = b * heads + h;
            let q_base = bh * dk;
            let k_base = bh * dk;
            let v_base = bh * dv;
            let state_base = bh * dk * dv;
            let decay = gd[bh].exp();
            for d in 0..dv {
                let mut v_pred = 0.0f32;
                for i in 0..dk {
                    v_pred += kd[k_base + i] * decay * sd[state_base + i * dv + d];
                }
                let delta = bd[bh] * (vd[v_base + d] - v_pred);
                let mut out_acc = 0.0f32;
                for i in 0..dk {
                    let new_s = decay * sd[state_base + i * dv + d] + kd[k_base + i] * delta;
                    expected_state[state_base + i * dv + d] = new_s;
                    out_acc += qd[q_base + i] * new_s;
                }
                expected_out[v_base + d] = out_acc;
            }
        }
    }

    let expected_out = cpu_bf16(expected_out, (batch, heads, dv))?;
    let expected_state = cpu_bf16(expected_state, (batch, heads, dk, dv))?;
    assert_close("recurrent out", &got_out, &expected_out, 1e-2)?;
    assert_close("recurrent state", &got_state, &expected_state, 1e-2)?;
    Ok(())
}

#[test]
fn gdn_recurrent_step_matches_f32_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, heads, dk, dv) = (1usize, 1usize, 4usize, 3usize);
    let q = cpu_f32(vec![0.12, -0.08, 0.21, -0.17], (batch, heads, dk))?;
    let k = cpu_f32(vec![-0.11, 0.24, -0.19, 0.07], (batch, heads, dk))?;
    let v = cpu_f32(vec![0.31, -0.23, 0.14], (batch, heads, dv))?;
    let beta = cpu_f32(vec![0.42], (batch, heads))?;
    let g = cpu_f32(vec![-0.09], (batch, heads))?;
    let state = cpu_f32(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.021)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let (got_out, got_state) = kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step(
        &vk, &q, &k, &v, &beta, &g, &state,
    )
    .context("dispatch_gdn_recurrent_step f32")?;

    let qd = tensor_data_f32(&q)?;
    let kd = tensor_data_f32(&k)?;
    let vd = tensor_data_f32(&v)?;
    let bd = tensor_data_f32(&beta)?;
    let gd = tensor_data_f32(&g)?;
    let sd = tensor_data_f32(&state)?;
    let mut expected_out = vec![0.0f32; batch * heads * dv];
    let mut expected_state = sd.clone();
    let decay = gd[0].exp();
    for d in 0..dv {
        let mut v_pred = 0.0f32;
        for i in 0..dk {
            v_pred += kd[i] * decay * sd[i * dv + d];
        }
        let delta = bd[0] * (vd[d] - v_pred);
        let mut out_acc = 0.0f32;
        for i in 0..dk {
            let new_s = decay * sd[i * dv + d] + kd[i] * delta;
            expected_state[i * dv + d] = new_s;
            out_acc += qd[i] * new_s;
        }
        expected_out[d] = out_acc;
    }

    assert_close(
        "recurrent out f32",
        &got_out,
        &cpu_f32(expected_out, (batch, heads, dv))?,
        1e-5,
    )?;
    assert_close(
        "recurrent state f32",
        &got_state,
        &cpu_f32(expected_state, (batch, heads, dk, dv))?,
        1e-5,
    )?;
    Ok(())
}

#[test]
fn gdn_recurrent_resident_state_matches_two_step_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, heads, dk, dv) = (1usize, 1usize, 4usize, 3usize);
    let q1 = cpu_f32(vec![0.12, -0.08, 0.21, -0.17], (batch, heads, dk))?;
    let k1 = cpu_f32(vec![-0.11, 0.24, -0.19, 0.07], (batch, heads, dk))?;
    let v1 = cpu_f32(vec![0.31, -0.23, 0.14], (batch, heads, dv))?;
    let beta1 = cpu_f32(vec![0.42], (batch, heads))?;
    let g1 = cpu_f32(vec![-0.09], (batch, heads))?;
    let q2 = cpu_f32(vec![0.05, 0.19, -0.13, 0.29], (batch, heads, dk))?;
    let k2 = cpu_f32(vec![0.17, -0.03, 0.11, -0.21], (batch, heads, dk))?;
    let v2 = cpu_f32(vec![-0.07, 0.25, 0.18], (batch, heads, dv))?;
    let beta2 = cpu_f32(vec![0.57], (batch, heads))?;
    let g2 = cpu_f32(vec![-0.14], (batch, heads))?;
    let state = cpu_f32(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.021)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let (expected_out1, expected_state1) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step(
            &vk, &q1, &k1, &v1, &beta1, &g1, &state,
        )
        .context("dispatch_gdn_recurrent_step reference step 1")?;
    let (expected_out2, _expected_state2) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step(
            &vk,
            &q2,
            &k2,
            &v2,
            &beta2,
            &g2,
            &expected_state1,
        )
        .context("dispatch_gdn_recurrent_step reference step 2")?;

    let (got_out1, resident_state) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_resident_state(
            &vk, &q1, &k1, &v1, &beta1, &g1, &state, None,
        )
        .context("dispatch_gdn_recurrent_step_resident_state step 1")?;
    let (got_out2, _resident_state) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_resident_state(
            &vk,
            &q2,
            &k2,
            &v2,
            &beta2,
            &g2,
            &state,
            Some(resident_state),
        )
        .context("dispatch_gdn_recurrent_step_resident_state step 2")?;

    assert_close(
        "resident recurrent out step 1",
        &got_out1,
        &expected_out1,
        1e-5,
    )?;
    assert_close(
        "resident recurrent out step 2",
        &got_out2,
        &expected_out2,
        1e-5,
    )?;
    Ok(())
}

#[test]
fn gdn_decode_gates_recurrent_rmsnorm_matches_f32_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, nv, dk, dv) = (3usize, 2usize, 4usize, 3usize);
    let q = cpu_f32(
        (0..batch * nv * dk)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.041)
            .collect(),
        (batch, 1, nv, dk),
    )?;
    let k = cpu_f32(
        (0..batch * nv * dk)
            .map(|i| ((i as f32 % 5.0) - 2.0) * -0.037)
            .collect(),
        (batch, 1, nv, dk),
    )?;
    let v = cpu_f32(
        (0..batch * nv * dv)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.029)
            .collect(),
        (batch, 1, nv, dv),
    )?;
    let a = cpu_f32(
        (0..batch * nv)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.061)
            .collect(),
        (batch, 1, nv),
    )?;
    let b = cpu_f32(
        (0..batch * nv)
            .map(|i| ((i as f32 % 5.0) - 2.0) * -0.073)
            .collect(),
        (batch, 1, nv),
    )?;
    let a_log = cpu_f32(vec![0.08, -0.17], (nv,))?;
    let dt_bias = cpu_f32(vec![0.03, -0.05], (nv,))?;
    let state = cpu_f32(
        (0..batch * nv * dk * dv)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.013)
            .collect(),
        (batch, nv, dk, dv),
    )?;
    let z = cpu_f32(
        (0..batch * nv * dv)
            .map(|i| ((i as f32 % 9.0) - 4.0) * 0.071)
            .collect(),
        (batch, 1, nv, dv),
    )?;
    let weight = cpu_f32(vec![0.7, -1.1, 0.9], (dv,))?;

    let (got_out, got_state) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm(
            &vk, &q, &k, &v, &a, &b, &a_log, &dt_bias, &state, &z, &weight, 1e-6, false,
        )
        .context("dispatch_gdn_decode_gates_recurrent_rmsnorm")?;

    let qd = tensor_data_f32(&q)?;
    let kd = tensor_data_f32(&k)?;
    let vd = tensor_data_f32(&v)?;
    let ad = tensor_data_f32(&a)?;
    let bd = tensor_data_f32(&b)?;
    let ald = tensor_data_f32(&a_log)?;
    let dbd = tensor_data_f32(&dt_bias)?;
    let sd = tensor_data_f32(&state)?;
    let zd = tensor_data_f32(&z)?;
    let wd = tensor_data_f32(&weight)?;
    let mut exp_state = sd.clone();
    let mut raw = vec![0.0f32; batch * nv * dv];

    for row in 0..batch {
        for h in 0..nv {
            let head = row * nv + h;
            let beta = 1.0 / (1.0 + (-bd[head]).exp());
            let biased = ad[head] + dbd[h];
            let softplus = biased.max(0.0) + (-biased.abs()).exp().ln_1p();
            let decay = (-ald[h].exp() * softplus).exp();
            for d in 0..dv {
                let mut v_pred = 0.0;
                for i in 0..dk {
                    v_pred += kd[head * dk + i] * decay * sd[head * dk * dv + i * dv + d];
                }
                let delta = beta * (vd[head * dv + d] - v_pred);
                let mut out_acc = 0.0;
                for i in 0..dk {
                    let idx = head * dk * dv + i * dv + d;
                    let new_s = decay * sd[idx] + kd[head * dk + i] * delta;
                    exp_state[idx] = new_s;
                    out_acc += qd[head * dk + i] * new_s;
                }
                raw[head * dv + d] = out_acc;
            }
        }
    }

    let mut exp_out = vec![0.0f32; batch * nv * dv];
    for row in 0..batch {
        for h in 0..nv {
            let head = row * nv + h;
            let mean_sq = (0..dv)
                .map(|d| raw[head * dv + d] * raw[head * dv + d])
                .sum::<f32>()
                / dv as f32;
            let rms_inv = (mean_sq + 1e-6).sqrt().recip();
            for d in 0..dv {
                let zv = zd[head * dv + d];
                let sigmoid = if zv >= 0.0 {
                    1.0 / (1.0 + (-zv).exp())
                } else {
                    let ez = zv.exp();
                    ez / (1.0 + ez)
                };
                exp_out[head * dv + d] = raw[head * dv + d] * rms_inv * wd[d] * zv * sigmoid;
            }
        }
    }

    let (skip_out, skip_state) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm(
            &vk, &q, &k, &v, &a, &b, &a_log, &dt_bias, &state, &z, &weight, 1e-6, true,
        )
        .context("dispatch_gdn_decode_gates_recurrent_rmsnorm skip state readback")?;

    assert_close(
        "decode gates recurrent rmsnorm out",
        &got_out,
        &cpu_f32(exp_out, (batch, 1, nv, dv))?,
        1e-5,
    )?;
    assert_close(
        "decode gates recurrent rmsnorm state",
        &got_state,
        &cpu_f32(exp_state, (batch, nv, dk, dv))?,
        1e-5,
    )?;
    assert_close(
        "decode gates recurrent rmsnorm skip-readback out",
        &skip_out,
        &got_out,
        1e-5,
    )?;
    assert_close(
        "decode gates recurrent rmsnorm skip-readback state",
        &skip_state,
        &state,
        1e-5,
    )?;
    Ok(())
}

#[test]
fn gdn_decode_gates_recurrent_rmsnorm_resident_state_matches_two_step_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, nv, dk, dv) = (3usize, 2usize, 4usize, 3usize);
    let make_q = |scale: f32| -> Result<Tensor> {
        cpu_f32(
            (0..batch * nv * dk)
                .map(|i| ((i as f32 % 7.0) - 3.0) * scale)
                .collect(),
            (batch, 1, nv, dk),
        )
    };
    let make_v = |scale: f32| -> Result<Tensor> {
        cpu_f32(
            (0..batch * nv * dv)
                .map(|i| ((i as f32 % 11.0) - 5.0) * scale)
                .collect(),
            (batch, 1, nv, dv),
        )
    };
    let q1 = make_q(0.041)?;
    let k1 = make_q(-0.037)?;
    let v1 = make_v(0.029)?;
    let a1 = cpu_f32(
        (0..batch * nv)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.061)
            .collect(),
        (batch, 1, nv),
    )?;
    let b1 = cpu_f32(
        (0..batch * nv)
            .map(|i| ((i as f32 % 5.0) - 2.0) * -0.073)
            .collect(),
        (batch, 1, nv),
    )?;
    let z1 = make_v(0.071)?;
    let q2 = make_q(-0.025)?;
    let k2 = make_q(0.033)?;
    let v2 = make_v(-0.021)?;
    let a2 = cpu_f32(
        (0..batch * nv)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.047)
            .collect(),
        (batch, 1, nv),
    )?;
    let b2 = cpu_f32(
        (0..batch * nv)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.039)
            .collect(),
        (batch, 1, nv),
    )?;
    let z2 = make_v(-0.052)?;
    let a_log = cpu_f32(vec![0.08, -0.17], (nv,))?;
    let dt_bias = cpu_f32(vec![0.03, -0.05], (nv,))?;
    let state = cpu_f32(
        (0..batch * nv * dk * dv)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.013)
            .collect(),
        (batch, nv, dk, dv),
    )?;
    let weight = cpu_f32(vec![0.7, -1.1, 0.9], (dv,))?;

    let (expected_out1, expected_state1) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm(
            &vk, &q1, &k1, &v1, &a1, &b1, &a_log, &dt_bias, &state, &z1, &weight, 1e-6, false,
        )
        .context("dispatch_gdn_decode_gates_recurrent_rmsnorm reference step 1")?;
    let (expected_out2, _expected_state2) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm(
            &vk,
            &q2,
            &k2,
            &v2,
            &a2,
            &b2,
            &a_log,
            &dt_bias,
            &expected_state1,
            &z2,
            &weight,
            1e-6,
            false,
        )
        .context("dispatch_gdn_decode_gates_recurrent_rmsnorm reference step 2")?;

    let (got_out1, resident_state) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state(
            &vk, &q1, &k1, &v1, &a1, &b1, &a_log, &dt_bias, &state, &z1, &weight, 1e-6, None,
        )
        .context("dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state step 1")?;
    let (got_out2, _resident_state) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state(
            &vk,
            &q2,
            &k2,
            &v2,
            &a2,
            &b2,
            &a_log,
            &dt_bias,
            &state,
            &z2,
            &weight,
            1e-6,
            Some(resident_state),
        )
        .context("dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state step 2")?;

    assert_close(
        "resident fused gdn out step 1",
        &got_out1,
        &expected_out1,
        1e-5,
    )?;
    assert_close(
        "resident fused gdn out step 2",
        &got_out2,
        &expected_out2,
        1e-5,
    )?;
    Ok(())
}

#[test]
fn gdn_chunk_prep_and_scan_match_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, heads, chunk, dv) = (1usize, 1usize, 4usize, 3usize);
    let g = cpu_bf16(vec![-0.08, -0.04, -0.12, -0.02], (batch, heads, chunk))?;
    let v = cpu_bf16(
        (0..batch * heads * chunk * dv)
            .map(|i| ((i as f32 % 9.0) - 4.0) * 0.052)
            .collect(),
        (batch, heads, chunk, dv),
    )?;
    let kkt = cpu_bf16(
        (0..batch * heads * chunk * chunk)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.033)
            .collect(),
        (batch, heads, chunk, chunk),
    )?;
    let qkt = cpu_bf16(
        (0..batch * heads * chunk * chunk)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.027)
            .collect(),
        (batch, heads, chunk, chunk),
    )?;
    let ks_entry = cpu_bf16(
        (0..batch * heads * chunk * dv)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.019)
            .collect(),
        (batch, heads, chunk, dv),
    )?;
    let q_s = cpu_bf16(
        (0..batch * heads * chunk * dv)
            .map(|i| ((i as f32 % 5.0) - 2.0) * 0.061)
            .collect(),
        (batch, heads, chunk, dv),
    )?;
    let beta = cpu_bf16(vec![0.25, 0.55, 0.38, 0.7], (batch, heads, chunk))?;

    let (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_prep(
            &vk, &g, &v, &kkt, &qkt, &ks_entry, &q_s,
        )
        .context("dispatch_gdn_chunk_prep")?;

    let gd = tensor_data_f32(&g)?;
    let vd = tensor_data_f32(&v)?;
    let kktd = tensor_data_f32(&kkt)?;
    let qktd = tensor_data_f32(&qkt)?;
    let ksd = tensor_data_f32(&ks_entry)?;
    let qsd = tensor_data_f32(&q_s)?;

    let mut big_g = vec![0.0f32; chunk];
    let mut acc = 0.0f32;
    for t in 0..chunk {
        acc += gd[t];
        big_g[t] = acc;
    }
    let mut exp_a = vec![0.0f32; chunk * chunk];
    let mut exp_b = vec![0.0f32; chunk * chunk];
    let mut exp_vp = vec![0.0f32; chunk * dv];
    let mut exp_qs = vec![0.0f32; chunk * dv];
    let mut exp_decay = vec![0.0f32; chunk];
    let exp_plast = vec![big_g[chunk - 1].exp()];

    for t in 0..chunk {
        for i in 0..chunk {
            let decay = (big_g[t] - big_g[i]).exp();
            let off = t * chunk + i;
            exp_a[off] = if t > i { kktd[off] * decay } else { 0.0 };
            exp_b[off] = if t >= i { qktd[off] * decay } else { 0.0 };
        }
        let p = big_g[t].exp();
        for d in 0..dv {
            let off = t * dv + d;
            exp_vp[off] = vd[off] - ksd[off] * p;
            exp_qs[off] = qsd[off] * p;
        }
        exp_decay[t] = (big_g[chunk - 1] - big_g[t]).exp();
    }

    let exp_a = cpu_bf16(exp_a, (batch, heads, chunk, chunk))?;
    let exp_b = cpu_bf16(exp_b, (batch, heads, chunk, chunk))?;
    let exp_vp = cpu_bf16(exp_vp, (batch, heads, chunk, dv))?;
    let exp_qs = cpu_bf16(exp_qs, (batch, heads, chunk, dv))?;
    let exp_decay = cpu_bf16(exp_decay, (batch, heads, chunk))?;
    let exp_plast = cpu_bf16(exp_plast, (batch, heads))?;

    assert_close("prep a_strict", &a_strict, &exp_a, 1e-2)?;
    assert_close("prep b_mask", &b_mask, &exp_b, 1e-2)?;
    assert_close("prep v_prime", &v_prime, &exp_vp, 1e-2)?;
    assert_close("prep q_s_scaled", &q_s_scaled, &exp_qs, 1e-2)?;
    assert_close("prep decay_last_col", &decay_last_col, &exp_decay, 1e-2)?;
    assert_close("prep p_last", &p_last, &exp_plast, 1e-2)?;

    let (got_out, got_w_weighted) = kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_scan(
        &vk,
        &a_strict,
        &b_mask,
        &v_prime,
        &q_s_scaled,
        &beta,
        &decay_last_col,
    )
    .context("dispatch_gdn_chunk_scan")?;

    let ad = tensor_data_f32(&a_strict)?;
    let bd = tensor_data_f32(&b_mask)?;
    let vpd = tensor_data_f32(&v_prime)?;
    let qssd = tensor_data_f32(&q_s_scaled)?;
    let betad = tensor_data_f32(&beta)?;
    let decayd = tensor_data_f32(&decay_last_col)?;
    let mut expected_out = vec![0.0f32; chunk * dv];
    let mut expected_w_weighted = vec![0.0f32; chunk * dv];

    for c in 0..chunk {
        for d in 0..dv {
            let mut w = vec![0.0f32; c + 1];
            for t in 0..=c {
                let mut acc_a = 0.0f32;
                for i in 0..t {
                    acc_a += ad[t * chunk + i] * w[i];
                }
                w[t] = betad[t] * (vpd[t * dv + d] - acc_a);
            }
            let mut intra = 0.0f32;
            for i in 0..=c {
                intra += bd[c * chunk + i] * w[i];
            }
            expected_out[c * dv + d] = qssd[c * dv + d] + intra;
            expected_w_weighted[c * dv + d] = w[c] * decayd[c];
        }
    }

    let expected_out = cpu_bf16(expected_out, (batch, heads, chunk, dv))?;
    let expected_w_weighted = cpu_bf16(expected_w_weighted, (batch, heads, chunk, dv))?;
    assert_close("scan out", &got_out, &expected_out, 1e-2)?;
    assert_close(
        "scan w_weighted",
        &got_w_weighted,
        &expected_w_weighted,
        1e-2,
    )?;
    Ok(())
}

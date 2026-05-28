use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, shape::ShapeWithOneHole};
use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice};

// Local test-only candle ↔ bytes/buffer helpers. These mirror the
// behaviour of the historical `kernels::{extract_tensor_bytes,
// create_tensor_from_data, upload_tensor_f32_buffer}` surface, which
// went away with the `candle_bridge` module in #1082. The production
// crate is now candle-free; these helpers stay scoped to this
// integration test file so candle-core can remain a dev-dependency
// only. (#1082)

fn extract_tensor_bytes(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let flat = tensor.flatten_all()?;
    let f32_data = flat.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok((bytemuck::cast_slice(&f32_data).to_vec(), shape))
}

fn create_tensor_from_data(data: &[u8], shape: &[usize], dtype: DType) -> Result<Tensor> {
    let f32_data: &[f32] = bytemuck::cast_slice(data);
    let tensor =
        Tensor::from_vec(f32_data.to_vec(), f32_data.len(), &Device::Cpu)?.reshape(shape)?;
    if dtype == DType::BF16 {
        Ok(tensor.to_dtype(DType::BF16)?)
    } else {
        Ok(tensor)
    }
}

fn upload_tensor_f32_buffer(vk_device: &VulkanDevice, tensor: &Tensor) -> Result<VulkanBuffer> {
    let tensor_f32;
    let tensor = if tensor.dtype() == DType::F32 {
        tensor
    } else {
        tensor_f32 = tensor.to_dtype(DType::F32)?;
        &tensor_f32
    };
    let (data, _) = extract_tensor_bytes(tensor)?;
    let f32_slice: &[f32] = bytemuck::cast_slice(&data);
    Ok(kiln_vulkan_kernel::kernels::upload_f32_buffer_from_slice(
        vk_device, f32_slice,
    )?)
}

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

/// Test-only candle wrapper for
/// `dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state_bytes`.
#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state_tensor(
    vk: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
    resident_state: Option<std::sync::Arc<kiln_vulkan_kernel::VulkanBuffer>>,
) -> Result<(Tensor, std::sync::Arc<kiln_vulkan_kernel::VulkanBuffer>)> {
    let (batch, _, nv, dk) = q.dims4()?;
    let dv = v.dims4()?.3;
    let q_dtype = q.dtype();
    let q_b = extract_tensor_bytes(q)?.0;
    let k_b = extract_tensor_bytes(k)?.0;
    let v_b = extract_tensor_bytes(v)?.0;
    let a_b = extract_tensor_bytes(a)?.0;
    let b_b = extract_tensor_bytes(b)?.0;
    let a_log_b = extract_tensor_bytes(a_log)?.0;
    let dt_bias_b = extract_tensor_bytes(dt_bias)?.0;
    let z_b = extract_tensor_bytes(z)?.0;
    let weight_b = extract_tensor_bytes(weight)?.0;
    let state_b = if resident_state.is_none() {
        Some(extract_tensor_bytes(state)?.0)
    } else {
        None
    };
    let (out_data, resident_state) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state_bytes(
            vk,
            &q_b, &k_b, &v_b, &a_b, &b_b, &a_log_b, &dt_bias_b,
            state_b.as_deref(),
            &z_b, &weight_b,
            batch, nv, dk, dv,
            eps,
            resident_state,
        )?;
    let out = create_tensor_from_data(
        &out_data,
        &[batch, 1, nv, dv],
        q_dtype,
    )?;
    Ok((out, resident_state))
}

/// Test-only candle wrapper for `dispatch_gdn_decode_gates_recurrent_rmsnorm_bytes`.
/// Falls back to `state.clone()` for the no-readback path to match the
/// pre-inversion `(Tensor, Tensor)` return shape used by the parity tests.
#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_decode_gates_recurrent_rmsnorm_tensor(
    vk: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
    skip_state_readback: bool,
) -> Result<(Tensor, Tensor)> {
    let (batch, _, nv, dk) = q.dims4()?;
    let dv = v.dims4()?.3;
    let q_dtype = q.dtype();
    let state_dtype = state.dtype();
    let state_dims = state.dims().to_vec();
    let input_tensors: [&Tensor; 10] = [q, k, v, a, b, a_log, dt_bias, state, z, weight];
    let mut input_data: Vec<Vec<u8>> = Vec::with_capacity(input_tensors.len());
    for tensor in &input_tensors {
        input_data.push(extract_tensor_bytes(tensor)?.0);
    }
    let (out_data, new_state_data) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_decode_gates_recurrent_rmsnorm_bytes(
            vk,
            &input_data,
            batch,
            nv,
            dk,
            dv,
            eps,
            skip_state_readback,
        )?;
    let out = create_tensor_from_data(
        &out_data,
        &[batch, 1, nv, dv],
        q_dtype,
    )?;
    let new_state = if let Some(sd) = new_state_data {
        create_tensor_from_data(&sd, &state_dims, state_dtype)?
    } else {
        state.clone()
    };
    Ok((out, new_state))
}

/// Test-only candle wrapper for the bytes-only
/// `dispatch_gdn_recurrent_step_with_options_bytes`. Keeps the candle-typed
/// parity tests readable without re-exposing candle types in the kernel
/// crate's public API.
fn dispatch_gdn_recurrent_step_with_options_tensor(
    vk: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
    skip_state_readback: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let q_dims = q.dims();
    let (batch, heads, dk) = (q_dims[0], q_dims[1], q_dims[2]);
    let dv = v.dims()[2];
    let q_dtype = q.dtype();
    let state_dtype = state.dtype();
    let state_dims = state.dims().to_vec();
    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let g_data = extract_tensor_bytes(g)?.0;
    let state_data = extract_tensor_bytes(state)?.0;
    let (out_data, new_state_data) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_with_options_bytes(
            vk,
            &q_data,
            &k_data,
            &v_data,
            &beta_data,
            &g_data,
            &state_data,
            batch,
            heads,
            dk,
            dv,
            skip_state_readback,
        )?;
    let out = create_tensor_from_data(
        &out_data,
        &[batch, heads, dv],
        q_dtype,
    )?;
    let new_state = new_state_data
        .as_ref()
        .map(|sd| {
            create_tensor_from_data(sd, &state_dims, state_dtype)
        })
        .transpose()?;
    Ok((out, new_state))
}

/// Test-only candle wrapper for the `_bytes` recurrent dispatch — keeps the
/// candle-typed parity tests readable without reintroducing a candle-typed
/// pub fn on the kernel crate.
#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_recurrent_step_native_head_last_with_options_tensor(
    vk: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
    skip_state_readback: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let (batch, _seq, q_heads, dk) = q.dims4()?;
    let (_, _, heads, dv) = v.dims4()?;
    let q_dtype = q.dtype();
    let state_dtype = state.dtype();
    let state_dims = state.dims().to_vec();
    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let g_data = extract_tensor_bytes(g)?.0;
    let state_data = extract_tensor_bytes(state)?.0;
    let (out_data, new_state_data) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_native_head_last_with_options_bytes(
            vk,
            &q_data,
            &k_data,
            &v_data,
            &beta_data,
            &g_data,
            &state_data,
            batch,
            q_heads,
            heads,
            dk,
            dv,
            skip_state_readback,
        )?;
    let out = create_tensor_from_data(
        &out_data,
        &[batch, heads, dv],
        q_dtype,
    )?
    .unsqueeze(1)?;
    let new_state = new_state_data
        .as_ref()
        .map(|sd| {
            create_tensor_from_data(sd, &state_dims, state_dtype)
        })
        .transpose()?;
    Ok((out, new_state))
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
    let weight_buf = upload_tensor_f32_buffer(&vk, &weight)?;
    let x_bytes = extract_tensor_bytes(&x)?.0;
    let got_bytes = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bytes(
        &vk,
        &x_bytes,
        &weight_buf,
        1,
        hidden,
        out_dim,
        false,
    )
    .context("dispatch_linear_decode_cached_bytes")?;
    let got = create_tensor_from_data(
        &got_bytes,
        &[1, 1, out_dim],
        candle_core::DType::F32,
    )?;
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
    let weight_buf = upload_tensor_f32_buffer(&vk, &weight)?;
    let x_bytes = extract_tensor_bytes(&x)?.0;
    let got_bytes = kiln_vulkan_kernel::kernels::dispatch_linear_decode_cached_bytes(
        &vk,
        &x_bytes,
        &weight_buf,
        batch,
        hidden,
        out_dim,
        false,
    )
    .context("dispatch_linear_decode_cached_bytes batched")?;
    let got = create_tensor_from_data(
        &got_bytes,
        &[batch, 1, out_dim],
        candle_core::DType::F32,
    )?;
    assert_close(
        "linear decode batched",
        &got,
        &x.broadcast_matmul(&weight)?,
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
        let x_bytes = extract_tensor_bytes(&x)?.0;
        let weight_bytes = extract_tensor_bytes(&weight)?.0;
        let state_bytes = extract_tensor_bytes(&state)?.0;
        let (got_out_bytes, got_state_bytes) =
            kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill_bytes(
                &vk,
                &x_bytes,
                &weight_bytes,
                &state_bytes,
                batch,
                channels,
                seq_len,
                kernel_size,
            )
            .with_context(|| format!("dispatch_causal_conv1d_prefill_bytes seq_len={seq_len}"))?;
        let got_out = create_tensor_from_data(
            &got_out_bytes,
            &[batch, channels, seq_len],
            candle_core::DType::F32,
        )?;
        let got_state = create_tensor_from_data(
            &got_state_bytes,
            &[batch, channels, kernel_size - 1],
            candle_core::DType::F32,
        )?;

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
            &cpu_f32(exp_out.clone(), (batch, channels, seq_len))?,
            1e-5,
        )?;
        assert_close(
            &format!("causal conv1d prefill state seq_len={seq_len}"),
            &got_state,
            &cpu_f32(exp_state.clone(), (batch, channels, kernel_size - 1))?,
            1e-5,
        )?;

        let weight_buf = upload_tensor_f32_buffer(&vk, &weight)?;
        let x_data_bytes = extract_tensor_bytes(&x)?.0;
        let state_data_bytes = extract_tensor_bytes(&state)?.0;
        let (got_cached_out_bytes, got_cached_state_bytes) =
            kiln_vulkan_kernel::kernels::dispatch_causal_conv1d_prefill_cached_weight_bytes(
                &vk,
                &x_data_bytes,
                &weight_buf,
                &state_data_bytes,
                batch,
                channels,
                seq_len,
                kernel_size,
            )
            .with_context(|| {
                format!("dispatch_causal_conv1d_prefill_cached_weight_bytes seq_len={seq_len}")
            })?;
        let got_cached_out = create_tensor_from_data(
            &got_cached_out_bytes,
            &[batch, channels, seq_len],
            candle_core::DType::F32,
        )?;
        let got_cached_state = create_tensor_from_data(
            &got_cached_state_bytes,
            &[batch, channels, kernel_size - 1],
            candle_core::DType::F32,
        )?;
        assert_close(
            &format!("causal conv1d cached prefill out seq_len={seq_len}"),
            &got_cached_out,
            &cpu_f32(exp_out.clone(), (batch, channels, seq_len))?,
            1e-5,
        )?;
        assert_close(
            &format!("causal conv1d cached prefill state seq_len={seq_len}"),
            &got_cached_state,
            &cpu_f32(exp_state.clone(), (batch, channels, kernel_size - 1))?,
            1e-5,
        )?;
    }
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

    let (got_out, got_state) = {
        let (out, state) = dispatch_gdn_recurrent_step_with_options_tensor(
            &vk, &q, &k, &v, &beta, &g, &state, false,
        )
        .context("dispatch_gdn_recurrent_step")?;
        (out, state.context("dispatch_gdn_recurrent_step")?)
    };

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

    let (got_out, got_state) = {
        let (out, state) = dispatch_gdn_recurrent_step_with_options_tensor(
            &vk, &q, &k, &v, &beta, &g, &state, false,
        )
        .context("dispatch_gdn_recurrent_step f32")?;
        (out, state.context("dispatch_gdn_recurrent_step f32")?)
    };

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
fn gdn_recurrent_step_parallel_reduce_matches_f32_cpu_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, heads, dk, dv) = (2usize, 2usize, 64usize, 7usize);
    let q = cpu_f32(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 19.0) - 9.0) * 0.011)
            .collect(),
        (batch, heads, dk),
    )?;
    let k = cpu_f32(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 23.0) - 11.0) * -0.009)
            .collect(),
        (batch, heads, dk),
    )?;
    let v = cpu_f32(
        (0..batch * heads * dv)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.013)
            .collect(),
        (batch, heads, dv),
    )?;
    let beta = cpu_f32(
        (0..batch * heads)
            .map(|i| 0.25 + (i as f32) * 0.07)
            .collect(),
        (batch, heads),
    )?;
    let g = cpu_f32(
        (0..batch * heads)
            .map(|i| -0.04 - (i as f32) * 0.03)
            .collect(),
        (batch, heads),
    )?;
    let state = cpu_f32(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 29.0) - 14.0) * 0.004)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let (got_out, got_state) = {
        let (out, state) = dispatch_gdn_recurrent_step_with_options_tensor(
            &vk, &q, &k, &v, &beta, &g, &state, false,
        )
        .context("dispatch_gdn_recurrent_step parallel reduce f32")?;
        (out, state.context("dispatch_gdn_recurrent_step parallel reduce f32")?)
    };

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

    assert_close(
        "parallel recurrent out f32",
        &got_out,
        &cpu_f32(expected_out, (batch, heads, dv))?,
        5e-4,
    )?;
    assert_close(
        "parallel recurrent state f32",
        &got_state,
        &cpu_f32(expected_state, (batch, heads, dk, dv))?,
        1e-3,
    )?;
    Ok(())
}

#[test]
fn gdn_recurrent_step_native_head_last_matches_expanded_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, q_heads, gqa_ratio, dk, dv) = (2usize, 2usize, 3usize, 64usize, 5usize);
    let heads = q_heads * gqa_ratio;
    let q = cpu_f32(
        (0..batch * q_heads * dk)
            .map(|i| ((i as f32 % 19.0) - 9.0) * 0.011)
            .collect(),
        (batch, 1, q_heads, dk),
    )?;
    let k = cpu_f32(
        (0..batch * q_heads * dk)
            .map(|i| ((i as f32 % 23.0) - 11.0) * -0.009)
            .collect(),
        (batch, 1, q_heads, dk),
    )?;
    let v = cpu_f32(
        (0..batch * heads * dv)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.013)
            .collect(),
        (batch, 1, heads, dv),
    )?;
    let beta = cpu_f32(
        (0..batch * heads)
            .map(|i| 0.21 + (i as f32) * 0.031)
            .collect(),
        (batch, 1, heads),
    )?;
    let g = cpu_f32(
        (0..batch * heads)
            .map(|i| -0.03 - (i as f32) * 0.017)
            .collect(),
        (batch, 1, heads),
    )?;
    let state = cpu_f32(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 29.0) - 14.0) * 0.004)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let q_expanded = q
        .squeeze(1)?
        .unsqueeze(2)?
        .expand(&[batch, q_heads, gqa_ratio, dk])?
        .contiguous()?
        .reshape((batch, heads, dk))?;
    let k_expanded = k
        .squeeze(1)?
        .unsqueeze(2)?
        .expand(&[batch, q_heads, gqa_ratio, dk])?
        .contiguous()?
        .reshape((batch, heads, dk))?;
    let v_expanded = v.squeeze(1)?.contiguous()?;
    let beta_expanded = beta.squeeze(1)?.contiguous()?;
    let g_expanded = g.squeeze(1)?.contiguous()?;

    let (expected_out, expected_state) = {
        let (out, state) = dispatch_gdn_recurrent_step_with_options_tensor(
            &vk,
        &q_expanded,
        &k_expanded,
        &v_expanded,
        &beta_expanded,
        &g_expanded,
        &state, false,
        )
        .context("dispatch_gdn_recurrent_step expanded reference")?;
        (out, state.context("dispatch_gdn_recurrent_step expanded reference")?)
    };
    let (got_out, got_state) =
        dispatch_gdn_recurrent_step_native_head_last_with_options_tensor(
            &vk, &q, &k, &v, &beta, &g, &state, false,
        )
        .context("dispatch_gdn_recurrent_step_native_head_last_with_options")?;

    assert_close(
        "native-head recurrent out",
        &got_out,
        &expected_out.unsqueeze(1)?,
        5e-4,
    )?;
    assert_close(
        "native-head recurrent state",
        &got_state.context("native-head state readback")?,
        &expected_state,
        1e-3,
    )?;
    Ok(())
}

#[test]
fn gdn_recurrent_qk_norm_native_head_last_matches_split_path() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, q_heads, gqa_ratio, dk, dv) = (2usize, 2usize, 3usize, 64usize, 5usize);
    let heads = q_heads * gqa_ratio;
    let q = cpu_f32(
        (0..batch * q_heads * dk)
            .map(|i| ((i as f32 % 19.0) - 9.0) * 0.011)
            .collect(),
        (batch, 1, q_heads, dk),
    )?;
    let k = cpu_f32(
        (0..batch * q_heads * dk)
            .map(|i| ((i as f32 % 23.0) - 11.0) * -0.009)
            .collect(),
        (batch, 1, q_heads, dk),
    )?;
    let v = cpu_bf16(
        (0..batch * heads * dv)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.013)
            .collect(),
        (batch, 1, heads, dv),
    )?;
    let beta = cpu_bf16(
        (0..batch * heads)
            .map(|i| 0.21 + (i as f32) * 0.031)
            .collect(),
        (batch, 1, heads),
    )?;
    let g = cpu_bf16(
        (0..batch * heads)
            .map(|i| -0.03 - (i as f32) * 0.017)
            .collect(),
        (batch, 1, heads),
    )?;
    let state = cpu_bf16(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 29.0) - 14.0) * 0.004)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let q_sq = q
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_keepdim(candle_core::D::Minus1)?;
    let k_sq = k
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_keepdim(candle_core::D::Minus1)?;
    let q_norm = (q
        .to_dtype(DType::F32)?
        .broadcast_div(&(q_sq + 1e-6)?.sqrt()?)?
        * (1.0 / (dk as f64).sqrt()))?
    .to_dtype(DType::BF16)?;
    let k_norm = k
        .to_dtype(DType::F32)?
        .broadcast_div(&(k_sq + 1e-6)?.sqrt()?)?
        .to_dtype(DType::BF16)?;

    let (expected_out, expected_state) =
        dispatch_gdn_recurrent_step_native_head_last_with_options_tensor(
            &vk, &q_norm, &k_norm, &v, &beta, &g, &state, false,
        )
        .context("native-head split qk_norm recurrent")?;

    let q_data = extract_tensor_bytes(&q)?.0;
    let k_data = extract_tensor_bytes(&k)?.0;
    let v_data = extract_tensor_bytes(&v)?.0;
    let beta_data = extract_tensor_bytes(&beta)?.0;
    let g_data = extract_tensor_bytes(&g)?.0;
    let state_data = extract_tensor_bytes(&state)?.0;
    let (got_out_bytes, got_state_bytes) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_qk_norm_step_native_head_last_with_options_bytes(
            &vk,
            &q_data,
            &k_data,
            &v_data,
            &beta_data,
            &g_data,
            &state_data,
            batch,
            q_heads,
            heads,
            dk,
            dv,
            false,
        )
        .context("native-head fused qk_norm recurrent")?;
    let got_out = create_tensor_from_data(
        &got_out_bytes,
        &[batch, heads, dv],
        state.dtype(),
    )?
    .unsqueeze(1)?;
    let got_state = got_state_bytes
        .as_ref()
        .map(|sd| {
            create_tensor_from_data(
                sd,
                state.dims(),
                state.dtype(),
            )
        })
        .transpose()?;

    assert_close(
        "native-head qk-norm recurrent out",
        &got_out,
        &expected_out,
        1e-3,
    )?;
    assert_close(
        "native-head qk-norm recurrent state",
        &got_state.context("native-head qk-norm state readback")?,
        &expected_state.context("native-head split state readback")?,
        1e-2,
    )?;
    Ok(())
}

#[test]
fn gdn_recurrent_step_native_head_last_resident_state_matches_readback_path() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, q_heads, gqa_ratio, dk, dv) = (2usize, 2usize, 3usize, 64usize, 5usize);
    let heads = q_heads * gqa_ratio;
    let make_q = |scale: f32| -> Result<Tensor> {
        cpu_f32(
            (0..batch * q_heads * dk)
                .map(|i| ((i as f32 % 19.0) - 9.0) * scale)
                .collect(),
            (batch, 1, q_heads, dk),
        )
    };
    let make_k = |scale: f32| -> Result<Tensor> {
        cpu_f32(
            (0..batch * q_heads * dk)
                .map(|i| ((i as f32 % 23.0) - 11.0) * scale)
                .collect(),
            (batch, 1, q_heads, dk),
        )
    };
    let make_v = |scale: f32| -> Result<Tensor> {
        cpu_f32(
            (0..batch * heads * dv)
                .map(|i| ((i as f32 % 17.0) - 8.0) * scale)
                .collect(),
            (batch, 1, heads, dv),
        )
    };
    let make_beta = |base: f32| -> Result<Tensor> {
        cpu_f32(
            (0..batch * heads)
                .map(|i| base + (i as f32) * 0.031)
                .collect(),
            (batch, 1, heads),
        )
    };
    let make_g = |base: f32| -> Result<Tensor> {
        cpu_f32(
            (0..batch * heads)
                .map(|i| base - (i as f32) * 0.017)
                .collect(),
            (batch, 1, heads),
        )
    };
    let state0 = cpu_f32(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 29.0) - 14.0) * 0.004)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let q1 = make_q(0.011)?;
    let k1 = make_k(-0.009)?;
    let v1 = make_v(0.013)?;
    let beta1 = make_beta(0.21)?;
    let g1 = make_g(-0.03)?;
    let q2 = make_q(-0.007)?;
    let k2 = make_k(0.015)?;
    let v2 = make_v(-0.019)?;
    let beta2 = make_beta(0.17)?;
    let g2 = make_g(-0.02)?;

    let (expected_out1, expected_state1) =
        dispatch_gdn_recurrent_step_native_head_last_with_options_tensor(
            &vk, &q1, &k1, &v1, &beta1, &g1, &state0, false,
        )
        .context("native-head readback step 1")?;
    let expected_state1 = expected_state1.context("native-head readback state 1")?;
    let (expected_out2, expected_state2) =
        dispatch_gdn_recurrent_step_native_head_last_with_options_tensor(
            &vk,
            &q2,
            &k2,
            &v2,
            &beta2,
            &g2,
            &expected_state1,
            false,
        )
        .context("native-head readback step 2")?;
    let expected_state2 = expected_state2.context("native-head readback state 2")?;

    let (resident_out1, resident_state) = {
        let q_data_b = extract_tensor_bytes(&q1)?.0;
        let k_data_b = extract_tensor_bytes(&k1)?.0;
        let v_data_b = extract_tensor_bytes(&v1)?.0;
        let beta_data_b = extract_tensor_bytes(&beta1)?.0;
        let g_data_b = extract_tensor_bytes(&g1)?.0;
        let state_data_b = extract_tensor_bytes(&state0)?.0;
        let (b1, sl1, qh1, dk1) = q1.dims4()?;
        let (_, _, h1, dv1) = v1.dims4()?;
        let (out_b, st_buf) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_native_head_last_resident_state_bytes(
                &vk,
                &q_data_b, &k_data_b, &v_data_b, &beta_data_b, &g_data_b,
                Some(state_data_b.as_slice()),
                b1, sl1, qh1, h1, dk1, dv1,
                None,
            )
            .context("native-head resident step 1")?;
        let out_t = create_tensor_from_data(
            &out_b, &[b1, h1, dv1], q1.dtype(),
        )?
        .unsqueeze(1)?;
        (out_t, st_buf)
    };
    let (resident_out2, resident_state) = {
        let q_data_b = extract_tensor_bytes(&q2)?.0;
        let k_data_b = extract_tensor_bytes(&k2)?.0;
        let v_data_b = extract_tensor_bytes(&v2)?.0;
        let beta_data_b = extract_tensor_bytes(&beta2)?.0;
        let g_data_b = extract_tensor_bytes(&g2)?.0;
        let (b2, sl2, qh2, dk2) = q2.dims4()?;
        let (_, _, h2, dv2) = v2.dims4()?;
        let (out_b, st_buf) =
            kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_native_head_last_resident_state_bytes(
                &vk,
                &q_data_b, &k_data_b, &v_data_b, &beta_data_b, &g_data_b,
                None,
                b2, sl2, qh2, h2, dk2, dv2,
                Some(resident_state),
            )
            .context("native-head resident step 2")?;
        let out_t = create_tensor_from_data(
            &out_b, &[b2, h2, dv2], q2.dtype(),
        )?
        .unsqueeze(1)?;
        (out_t, st_buf)
    };
    let resident_state_data = kiln_vulkan_kernel::VulkanBuffer::read_back(
        vk.device(),
        vk.host_visible_mem_type(),
        vk.queue(),
        vk.queue_family_index(),
        &resident_state,
    )
    .context("read back resident native-head state")?;
    let resident_state = create_tensor_from_data(
        &resident_state_data,
        state0.dims(),
        state0.dtype(),
    )?;

    assert_close(
        "resident native-head out step1",
        &resident_out1,
        &expected_out1,
        5e-4,
    )?;
    assert_close(
        "resident native-head out step2",
        &resident_out2,
        &expected_out2,
        5e-4,
    )?;
    assert_close(
        "resident native-head state step2",
        &resident_state,
        &expected_state2,
        1e-3,
    )?;
    Ok(())
}

#[test]
fn gdn_recurrent_step_can_skip_state_readback() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, heads, dk, dv) = (2usize, 2usize, 8usize, 4usize);
    let q = cpu_f32(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.031)
            .collect(),
        (batch, heads, dk),
    )?;
    let k = cpu_f32(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 7.0) - 3.0) * 0.037)
            .collect(),
        (batch, heads, dk),
    )?;
    let v = cpu_f32(
        (0..batch * heads * dv)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.023)
            .collect(),
        (batch, heads, dv),
    )?;
    let beta = cpu_f32(vec![0.42, 0.57, 0.33, 0.61], (batch, heads))?;
    let g = cpu_f32(vec![-0.09, -0.14, -0.07, -0.19], (batch, heads))?;
    let state = cpu_f32(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.011)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let (expected_out, _expected_state) = {
        let (out, state) = dispatch_gdn_recurrent_step_with_options_tensor(
            &vk, &q, &k, &v, &beta, &g, &state, false,
        )
        .context("dispatch_gdn_recurrent_step reference")?;
        (out, state.context("dispatch_gdn_recurrent_step reference")?)
    };
    let (got_out, got_state) =
        dispatch_gdn_recurrent_step_with_options_tensor(
            &vk, &q, &k, &v, &beta, &g, &state, true,
        )
        .context("dispatch_gdn_recurrent_step skip state readback")?;

    assert!(
        got_state.is_none(),
        "skip-state-readback path should not materialize updated state"
    );
    assert_close("recurrent skip-readback out", &got_out, &expected_out, 1e-4)?;
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
        {
        let (out, state) = dispatch_gdn_recurrent_step_with_options_tensor(
                &vk, &q1, &k1, &v1, &beta1, &g1, &state, false,
        )
        .context("dispatch_gdn_recurrent_step reference step 1")?;
        (out, state.context("dispatch_gdn_recurrent_step reference step 1")?)
    };
    let (expected_out2, _expected_state2) =
        {
        let (out, state) = dispatch_gdn_recurrent_step_with_options_tensor(
                &vk,
            &q2,
            &k2,
            &v2,
            &beta2,
            &g2,
            &expected_state1, false,
        )
        .context("dispatch_gdn_recurrent_step reference step 2")?;
        (out, state.context("dispatch_gdn_recurrent_step reference step 2")?)
    };

    let (got_out1, resident_state) =
        {
            let q_data_b = extract_tensor_bytes(&q1)?.0;
            let k_data_b = extract_tensor_bytes(&k1)?.0;
            let v_data_b = extract_tensor_bytes(&v1)?.0;
            let beta_data_b = extract_tensor_bytes(&beta1)?.0;
            let g_data_b = extract_tensor_bytes(&g1)?.0;
            let state_data_b =
                Some(extract_tensor_bytes(&state)?.0);
            let q_dims_b = q1.dims();
            let (b_b, h_b, dk_b) = (q_dims_b[0], q_dims_b[1], q_dims_b[2]);
            let dv_b = v1.dims()[2];
            let q_dtype_b = q1.dtype();
            let (out_bytes, resident_buf) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_resident_state_bytes(
                    &vk,
                    &q_data_b, &k_data_b, &v_data_b, &beta_data_b, &g_data_b,
                    state_data_b.as_deref(),
                    b_b, h_b, dk_b, dv_b,
                    None,
                )
                .context("dispatch_gdn_recurrent_step_resident_state step 1")?;
            let out_t = create_tensor_from_data(
                &out_bytes, &[b_b, h_b, dv_b], q_dtype_b,
            )?;
            (out_t, resident_buf)
        };
    let (got_out2, _resident_state) =
        {
            let q_data_b = extract_tensor_bytes(&q2)?.0;
            let k_data_b = extract_tensor_bytes(&k2)?.0;
            let v_data_b = extract_tensor_bytes(&v2)?.0;
            let beta_data_b = extract_tensor_bytes(&beta2)?.0;
            let g_data_b = extract_tensor_bytes(&g2)?.0;
            let q_dims_b = q2.dims();
            let (b_b, h_b, dk_b) = (q_dims_b[0], q_dims_b[1], q_dims_b[2]);
            let dv_b = v2.dims()[2];
            let q_dtype_b = q2.dtype();
            let (out_bytes, resident_buf) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_resident_state_bytes(
                    &vk,
                    &q_data_b, &k_data_b, &v_data_b, &beta_data_b, &g_data_b,
                    None,
                    b_b, h_b, dk_b, dv_b,
                    Some(resident_state),
                )
                .context("dispatch_gdn_recurrent_step_resident_state step 2")?;
            let out_t = create_tensor_from_data(
                &out_bytes, &[b_b, h_b, dv_b], q_dtype_b,
            )?;
            (out_t, resident_buf)
        };

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
        dispatch_gdn_decode_gates_recurrent_rmsnorm_tensor(
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
        dispatch_gdn_decode_gates_recurrent_rmsnorm_tensor(
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
fn gdn_recurrent_resident_state_parallel_reduce_matches_two_step_reference() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, heads, dk, dv) = (1usize, 2usize, 64usize, 7usize);
    let q1 = cpu_f32(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 19.0) - 9.0) * 0.011)
            .collect(),
        (batch, heads, dk),
    )?;
    let k1 = cpu_f32(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 23.0) - 11.0) * -0.009)
            .collect(),
        (batch, heads, dk),
    )?;
    let v1 = cpu_f32(
        (0..batch * heads * dv)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.013)
            .collect(),
        (batch, heads, dv),
    )?;
    let beta1 = cpu_f32(
        (0..batch * heads)
            .map(|i| 0.25 + (i as f32) * 0.07)
            .collect(),
        (batch, heads),
    )?;
    let g1 = cpu_f32(
        (0..batch * heads)
            .map(|i| -0.04 - (i as f32) * 0.03)
            .collect(),
        (batch, heads),
    )?;
    let q2 = cpu_f32(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 29.0) - 14.0) * -0.007)
            .collect(),
        (batch, heads, dk),
    )?;
    let k2 = cpu_f32(
        (0..batch * heads * dk)
            .map(|i| ((i as f32 % 31.0) - 15.0) * 0.006)
            .collect(),
        (batch, heads, dk),
    )?;
    let v2 = cpu_f32(
        (0..batch * heads * dv)
            .map(|i| ((i as f32 % 13.0) - 6.0) * -0.015)
            .collect(),
        (batch, heads, dv),
    )?;
    let beta2 = cpu_f32(
        (0..batch * heads)
            .map(|i| 0.41 + (i as f32) * 0.05)
            .collect(),
        (batch, heads),
    )?;
    let g2 = cpu_f32(
        (0..batch * heads)
            .map(|i| -0.08 - (i as f32) * 0.02)
            .collect(),
        (batch, heads),
    )?;
    let state = cpu_f32(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 37.0) - 18.0) * 0.003)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let (expected_out1, expected_state1) =
        {
        let (out, state) = dispatch_gdn_recurrent_step_with_options_tensor(
                &vk, &q1, &k1, &v1, &beta1, &g1, &state, false,
        )
        .context("dispatch_gdn_recurrent_step parallel reference step 1")?;
        (out, state.context("dispatch_gdn_recurrent_step parallel reference step 1")?)
    };
    let (expected_out2, _expected_state2) =
        {
        let (out, state) = dispatch_gdn_recurrent_step_with_options_tensor(
                &vk,
            &q2,
            &k2,
            &v2,
            &beta2,
            &g2,
            &expected_state1, false,
        )
        .context("dispatch_gdn_recurrent_step parallel reference step 2")?;
        (out, state.context("dispatch_gdn_recurrent_step parallel reference step 2")?)
    };

    let (got_out1, resident_state) =
        {
            let q_data_b = extract_tensor_bytes(&q1)?.0;
            let k_data_b = extract_tensor_bytes(&k1)?.0;
            let v_data_b = extract_tensor_bytes(&v1)?.0;
            let beta_data_b = extract_tensor_bytes(&beta1)?.0;
            let g_data_b = extract_tensor_bytes(&g1)?.0;
            let state_data_b =
                Some(extract_tensor_bytes(&state)?.0);
            let q_dims_b = q1.dims();
            let (b_b, h_b, dk_b) = (q_dims_b[0], q_dims_b[1], q_dims_b[2]);
            let dv_b = v1.dims()[2];
            let q_dtype_b = q1.dtype();
            let (out_bytes, resident_buf) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_resident_state_bytes(
                    &vk,
                    &q_data_b, &k_data_b, &v_data_b, &beta_data_b, &g_data_b,
                    state_data_b.as_deref(),
                    b_b, h_b, dk_b, dv_b,
                    None,
                )
                .context("dispatch_gdn_recurrent_step_resident_state parallel step 1")?;
            let out_t = create_tensor_from_data(
                &out_bytes, &[b_b, h_b, dv_b], q_dtype_b,
            )?;
            (out_t, resident_buf)
        };
    let (got_out2, _resident_state) =
        {
            let q_data_b = extract_tensor_bytes(&q2)?.0;
            let k_data_b = extract_tensor_bytes(&k2)?.0;
            let v_data_b = extract_tensor_bytes(&v2)?.0;
            let beta_data_b = extract_tensor_bytes(&beta2)?.0;
            let g_data_b = extract_tensor_bytes(&g2)?.0;
            let q_dims_b = q2.dims();
            let (b_b, h_b, dk_b) = (q_dims_b[0], q_dims_b[1], q_dims_b[2]);
            let dv_b = v2.dims()[2];
            let q_dtype_b = q2.dtype();
            let (out_bytes, resident_buf) =
                kiln_vulkan_kernel::kernels::dispatch_gdn_recurrent_step_resident_state_bytes(
                    &vk,
                    &q_data_b, &k_data_b, &v_data_b, &beta_data_b, &g_data_b,
                    None,
                    b_b, h_b, dk_b, dv_b,
                    Some(resident_state),
                )
                .context("dispatch_gdn_recurrent_step_resident_state parallel step 2")?;
            let out_t = create_tensor_from_data(
                &out_bytes, &[b_b, h_b, dv_b], q_dtype_b,
            )?;
            (out_t, resident_buf)
        };

    assert_close(
        "resident parallel recurrent out step 1",
        &got_out1,
        &expected_out1,
        5e-4,
    )?;
    assert_close(
        "resident parallel recurrent out step 2",
        &got_out2,
        &expected_out2,
        5e-4,
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
        dispatch_gdn_decode_gates_recurrent_rmsnorm_tensor(
            &vk, &q1, &k1, &v1, &a1, &b1, &a_log, &dt_bias, &state, &z1, &weight, 1e-6, false,
        )
        .context("dispatch_gdn_decode_gates_recurrent_rmsnorm reference step 1")?;
    let (expected_out2, _expected_state2) =
        dispatch_gdn_decode_gates_recurrent_rmsnorm_tensor(
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
        dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state_tensor(
            &vk, &q1, &k1, &v1, &a1, &b1, &a_log, &dt_bias, &state, &z1, &weight, 1e-6, None,
        )
        .context("dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state step 1")?;
    let (got_out2, _resident_state) =
        dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state_tensor(
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

    let g_data_p = extract_tensor_bytes(&g)?.0;
    let v_data_p = extract_tensor_bytes(&v)?.0;
    let kkt_data_p = extract_tensor_bytes(&kkt)?.0;
    let qkt_data_p = extract_tensor_bytes(&qkt)?.0;
    let ks_entry_data_p = extract_tensor_bytes(&ks_entry)?.0;
    let q_s_data_p = extract_tensor_bytes(&q_s)?.0;
    let g_dims_p = g.dims();
    let (b_p, h_p, c_p) = (g_dims_p[0], g_dims_p[1], g_dims_p[2]);
    let dv_p = v.dims()[3];
    let (a_strict_bytes, b_mask_bytes, v_prime_bytes, q_s_scaled_bytes, decay_last_col_bytes, p_last_bytes) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_prep_bytes(
            &vk,
            &g_data_p, &v_data_p, &kkt_data_p, &qkt_data_p, &ks_entry_data_p, &q_s_data_p,
            b_p, h_p, c_p, dv_p,
        )
        .context("dispatch_gdn_chunk_prep_bytes")?;
    let cc_shape_p = [b_p, h_p, c_p, c_p];
    let cv_shape_p = [b_p, h_p, c_p, dv_p];
    let decay_shape_p = [b_p, h_p, c_p];
    let p_last_shape_p = [b_p, h_p];
    let a_strict = create_tensor_from_data(
        &a_strict_bytes, &cc_shape_p, candle_core::DType::BF16,
    )?;
    let b_mask = create_tensor_from_data(
        &b_mask_bytes, &cc_shape_p, candle_core::DType::BF16,
    )?;
    let v_prime = create_tensor_from_data(
        &v_prime_bytes, &cv_shape_p, candle_core::DType::BF16,
    )?;
    let q_s_scaled = create_tensor_from_data(
        &q_s_scaled_bytes, &cv_shape_p, candle_core::DType::BF16,
    )?;
    let decay_last_col = create_tensor_from_data(
        &decay_last_col_bytes, &decay_shape_p, candle_core::DType::BF16,
    )?;
    let p_last = create_tensor_from_data(
        &p_last_bytes, &p_last_shape_p, candle_core::DType::BF16,
    )?;

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

    let a_strict_data = extract_tensor_bytes(&a_strict)?.0;
    let b_mask_data = extract_tensor_bytes(&b_mask)?.0;
    let v_prime_data = extract_tensor_bytes(&v_prime)?.0;
    let q_s_scaled_data = extract_tensor_bytes(&q_s_scaled)?.0;
    let beta_data_b = extract_tensor_bytes(&beta)?.0;
    let decay_last_col_data = extract_tensor_bytes(&decay_last_col)?.0;
    let v_prime_dims = v_prime.dims();
    let (vp_b, vp_h, vp_c, vp_dv) =
        (v_prime_dims[0], v_prime_dims[1], v_prime_dims[2], v_prime_dims[3]);
    let (got_out_bytes_a, got_w_weighted_bytes) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_scan_bytes(
            &vk,
            &a_strict_data,
            &b_mask_data,
            &v_prime_data,
            &q_s_scaled_data,
            &beta_data_b,
            &decay_last_col_data,
            vp_b, vp_h, vp_c, vp_dv,
        )
        .context("dispatch_gdn_chunk_scan_bytes")?;
    let got_out = create_tensor_from_data(
        &got_out_bytes_a,
        &[vp_b, vp_h, vp_c, vp_dv],
        candle_core::DType::BF16,
    )?;
    let got_w_weighted = create_tensor_from_data(
        &got_w_weighted_bytes,
        &[vp_b, vp_h, vp_c, vp_dv],
        candle_core::DType::BF16,
    )?;

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

#[test]
fn gdn_full_chunk_forward_matches_split_vulkan_path() -> Result<()> {
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let (batch, heads, chunk, dk, dv) = (1usize, 1usize, 64usize, 5usize, 4usize);
    let g = cpu_bf16(
        (0..chunk)
            .map(|i| -0.015 - ((i % 7) as f32) * 0.0015)
            .collect(),
        (batch, heads, chunk),
    )?;
    let v = cpu_bf16(
        (0..batch * heads * chunk * dv)
            .map(|i| ((i as f32 % 17.0) - 8.0) * 0.011)
            .collect(),
        (batch, heads, chunk, dv),
    )?;
    let kkt = cpu_bf16(
        (0..batch * heads * chunk * chunk)
            .map(|i| ((i as f32 % 13.0) - 6.0) * 0.004)
            .collect(),
        (batch, heads, chunk, chunk),
    )?;
    let qkt = cpu_bf16(
        (0..batch * heads * chunk * chunk)
            .map(|i| ((i as f32 % 11.0) - 5.0) * 0.005)
            .collect(),
        (batch, heads, chunk, chunk),
    )?;
    let ks_entry = cpu_bf16(
        (0..batch * heads * chunk * dv)
            .map(|i| ((i as f32 % 19.0) - 9.0) * 0.007)
            .collect(),
        (batch, heads, chunk, dv),
    )?;
    let q_s = cpu_bf16(
        (0..batch * heads * chunk * dv)
            .map(|i| ((i as f32 % 23.0) - 11.0) * 0.006)
            .collect(),
        (batch, heads, chunk, dv),
    )?;
    let beta = cpu_bf16(
        (0..chunk).map(|i| 0.2 + ((i % 9) as f32) * 0.045).collect(),
        (batch, heads, chunk),
    )?;
    let k_t = cpu_bf16(
        (0..batch * heads * dk * chunk)
            .map(|i| ((i as f32 % 29.0) - 14.0) * 0.003)
            .collect(),
        (batch, heads, dk, chunk),
    )?;
    let state = cpu_bf16(
        (0..batch * heads * dk * dv)
            .map(|i| ((i as f32 % 31.0) - 15.0) * 0.004)
            .collect(),
        (batch, heads, dk, dv),
    )?;

    let g_data_p = extract_tensor_bytes(&g)?.0;
    let v_data_p = extract_tensor_bytes(&v)?.0;
    let kkt_data_p = extract_tensor_bytes(&kkt)?.0;
    let qkt_data_p = extract_tensor_bytes(&qkt)?.0;
    let ks_entry_data_p = extract_tensor_bytes(&ks_entry)?.0;
    let q_s_data_p = extract_tensor_bytes(&q_s)?.0;
    let g_dims_p = g.dims();
    let (b_p, h_p, c_p) = (g_dims_p[0], g_dims_p[1], g_dims_p[2]);
    let dv_p = v.dims()[3];
    let (a_strict_bytes, b_mask_bytes, v_prime_bytes, q_s_scaled_bytes, decay_last_col_bytes, p_last_bytes) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_prep_bytes(
            &vk,
            &g_data_p, &v_data_p, &kkt_data_p, &qkt_data_p, &ks_entry_data_p, &q_s_data_p,
            b_p, h_p, c_p, dv_p,
        )
        .context("dispatch_gdn_chunk_prep_bytes")?;
    let cc_shape_p = [b_p, h_p, c_p, c_p];
    let cv_shape_p = [b_p, h_p, c_p, dv_p];
    let decay_shape_p = [b_p, h_p, c_p];
    let p_last_shape_p = [b_p, h_p];
    let a_strict = create_tensor_from_data(
        &a_strict_bytes, &cc_shape_p, candle_core::DType::BF16,
    )?;
    let b_mask = create_tensor_from_data(
        &b_mask_bytes, &cc_shape_p, candle_core::DType::BF16,
    )?;
    let v_prime = create_tensor_from_data(
        &v_prime_bytes, &cv_shape_p, candle_core::DType::BF16,
    )?;
    let q_s_scaled = create_tensor_from_data(
        &q_s_scaled_bytes, &cv_shape_p, candle_core::DType::BF16,
    )?;
    let decay_last_col = create_tensor_from_data(
        &decay_last_col_bytes, &decay_shape_p, candle_core::DType::BF16,
    )?;
    let p_last = create_tensor_from_data(
        &p_last_bytes, &p_last_shape_p, candle_core::DType::BF16,
    )?;
    let a_strict_data_b = extract_tensor_bytes(&a_strict)?.0;
    let b_mask_data_b = extract_tensor_bytes(&b_mask)?.0;
    let v_prime_data_b = extract_tensor_bytes(&v_prime)?.0;
    let q_s_scaled_data_b = extract_tensor_bytes(&q_s_scaled)?.0;
    let beta_data_bb = extract_tensor_bytes(&beta)?.0;
    let decay_last_col_data_b = extract_tensor_bytes(&decay_last_col)?.0;
    let v_prime_dims_b = v_prime.dims();
    let (vp_b2, vp_h2, vp_c2, vp_dv2) = (
        v_prime_dims_b[0], v_prime_dims_b[1], v_prime_dims_b[2], v_prime_dims_b[3],
    );
    let (expected_out_bytes, w_weighted_bytes) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_chunk_scan_bytes(
            &vk,
            &a_strict_data_b,
            &b_mask_data_b,
            &v_prime_data_b,
            &q_s_scaled_data_b,
            &beta_data_bb,
            &decay_last_col_data_b,
            vp_b2, vp_h2, vp_c2, vp_dv2,
        )
        .context("dispatch_gdn_chunk_scan_bytes")?;
    let expected_out = create_tensor_from_data(
        &expected_out_bytes,
        &[vp_b2, vp_h2, vp_c2, vp_dv2],
        candle_core::DType::BF16,
    )?;
    let w_weighted = create_tensor_from_data(
        &w_weighted_bytes,
        &[vp_b2, vp_h2, vp_c2, vp_dv2],
        candle_core::DType::BF16,
    )?;

    let g_data_b = extract_tensor_bytes(&g)?.0;
    let v_data_b = extract_tensor_bytes(&v)?.0;
    let kkt_data_b = extract_tensor_bytes(&kkt)?.0;
    let qkt_data_b = extract_tensor_bytes(&qkt)?.0;
    let ks_entry_data_b = extract_tensor_bytes(&ks_entry)?.0;
    let q_s_data_b = extract_tensor_bytes(&q_s)?.0;
    let beta_data_b = extract_tensor_bytes(&beta)?.0;
    let k_t_data_b = extract_tensor_bytes(&k_t)?.0;
    let state_data_b = extract_tensor_bytes(&state)?.0;
    let state_dims = state.dims().as_ref().to_vec();
    let (got_out_bytes, got_state_bytes) =
        kiln_vulkan_kernel::kernels::dispatch_gdn_full_chunk_forward_bytes(
            &vk, &g_data_b, &v_data_b, &kkt_data_b, &qkt_data_b, &ks_entry_data_b,
            &q_s_data_b, &beta_data_b, &k_t_data_b, &state_data_b,
            batch, heads, chunk, dk, dv,
        )
        .context("dispatch_gdn_full_chunk_forward_bytes")?;
    let got_out = create_tensor_from_data(
        &got_out_bytes,
        &[batch, heads, chunk, dv],
        candle_core::DType::BF16,
    )?;
    let got_state = create_tensor_from_data(
        &got_state_bytes,
        &state_dims,
        candle_core::DType::BF16,
    )?;

    let p_last = tensor_data_f32(&p_last)?;
    let state_data = tensor_data_f32(&state)?;
    let k_t_data = tensor_data_f32(&k_t)?;
    let w_weighted_data = tensor_data_f32(&w_weighted)?;
    let mut expected_state = vec![0.0f32; batch * heads * dk * dv];
    for k_idx in 0..dk {
        for d in 0..dv {
            let mut delta = 0.0f32;
            for t in 0..chunk {
                delta += k_t_data[k_idx * chunk + t] * w_weighted_data[t * dv + d];
            }
            expected_state[k_idx * dv + d] = state_data[k_idx * dv + d] * p_last[0] + delta;
        }
    }
    let expected_state = cpu_bf16(expected_state, (batch, heads, dk, dv))?;

    assert_close("full chunk out", &got_out, &expected_out, 2e-2)?;
    assert_close("full chunk state", &got_state, &expected_state, 3e-2)?;
    Ok(())
}

/// CPU reference for SDPA forward at the [B, T, H, dh] layout that
/// `dispatch_sdpa_prefill_f32` consumes.  Used by the parity tests.
///
/// Performs the standard `softmax((Q @ K^T) / sqrt(dh)) @ V` against
/// candle CPU broadcast_matmul + softmax. Returns a tensor with shape
/// `[B, T, H, dh]` (token-major), F32.
fn cpu_sdpa_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    let (b, t, h, _dh) = q.dims4()?;
    // Token-major → head-major: [B, T, H, dh] → [B, H, T, dh]
    let q_h = q.transpose(1, 2)?.contiguous()?;
    let k_h = k.transpose(1, 2)?.contiguous()?;
    let v_h = v.transpose(1, 2)?.contiguous()?;
    // Scores: [B, H, T, T]
    let scores = q_h.broadcast_matmul(&k_h.transpose(2, 3)?.contiguous()?)?;
    let scores = (scores * (softmax_scale as f64))?;
    let scores = if causal {
        // Causal mask: scores[..., i, j] = -inf for j > i.
        let mut mask = vec![0.0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                if j > i {
                    mask[i * t + j] = f32::MIN;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (t, t), &Device::Cpu)?
            .reshape((1, 1, t, t))?
            .broadcast_as((b, h, t, t))?;
        scores.broadcast_add(&mask)?
    } else {
        scores
    };
    // Manual softmax (last dim) — candle-nn isn't a dep of this crate.
    let last_dim = scores.dims().len() - 1;
    let max_per_row = scores.max_keepdim(last_dim)?;
    let shifted = scores.broadcast_sub(&max_per_row)?;
    let exp_shifted = shifted.exp()?;
    let sum_exp = exp_shifted.sum_keepdim(last_dim)?;
    let probs = exp_shifted.broadcast_div(&sum_exp)?;
    let out_h = probs.broadcast_matmul(&v_h)?;
    // Back to token-major: [B, H, T, dh] → [B, T, H, dh]
    Ok(out_h.transpose(1, 2)?.contiguous()?)
}

#[test]
fn sgd_step_f32_matches_cpu_reference() -> Result<()> {
    use kiln_vulkan_kernel::{VulkanBuffer, kernels};
    let Some(vk) = maybe_vulkan() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    // Realistic LoRA-shape param: rank=8, hidden=64 → 512 F32.
    // Plus a small odd-length test (300) so the workgroup boundary
    // (multiples of 256) is exercised by the early-return path.
    for n in [512usize, 300usize, 1usize] {
        let param_data: Vec<f32> = (0..n)
            .map(|i| ((i as i32 - (n as i32 / 2)) as f32) * 0.01)
            .collect();
        let grad_data: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.005).collect();
        let lr: f32 = 0.013;

        // CPU reference: param -= lr * grad.
        let expected: Vec<f32> = param_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&p, &g)| p - lr * g)
            .collect();

        // Set up Vulkan buffers + upload + dispatch.
        let device = vk.device();
        let device_local_mt = vk.device_local_mem_type();
        let host_visible_mt = vk.host_visible_mem_type();
        let queue = vk.queue();
        let queue_family = vk.queue_family_index();
        let bytes = n * 4;

        let param_bytes: Vec<u8> = param_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let grad_bytes: Vec<u8> = grad_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let param_buf = VulkanBuffer::create_device_local(device, device_local_mt, bytes as u64)
            .context("alloc param buffer")?;
        let grad_buf = VulkanBuffer::create_device_local(device, device_local_mt, bytes as u64)
            .context("alloc grad buffer")?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            queue_family,
            &param_buf,
            &param_bytes,
        )
        .context("upload param")?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            queue_family,
            &grad_buf,
            &grad_bytes,
        )
        .context("upload grad")?;

        kernels::dispatch_sgd_step_f32(&vk, &param_buf, &grad_buf, n, lr)?;

        let updated =
            VulkanBuffer::read_back(device, host_visible_mt, queue, queue_family, &param_buf)
                .context("read back param")?;
        let updated_f32: Vec<f32> = updated
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        anyhow::ensure!(
            updated_f32.len() == n,
            "n={n}: read back {} elements, expected {n}",
            updated_f32.len()
        );
        for (i, (got, want)) in updated_f32.iter().zip(expected.iter()).enumerate() {
            let diff = (got - want).abs();
            anyhow::ensure!(
                diff < 1e-7,
                "n={n} idx={i}: got={got:.9} want={want:.9} diff={diff:e}"
            );
        }
    }
    Ok(())
}

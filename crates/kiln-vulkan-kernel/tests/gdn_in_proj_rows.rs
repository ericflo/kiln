use anyhow::Result;
use half::bf16;
use kiln_vulkan_kernel::{CommandBatch, VulkanBuffer, VulkanDevice, Workgroups, kernels, shaders};

fn cpu_linear(x: &[f32], w: &[bf16], batch: usize, hidden: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_dim];
    for row in 0..batch {
        for col in 0..out_dim {
            let mut sum = 0.0f32;
            for h in 0..hidden {
                sum += x[row * hidden + h] * w[h * out_dim + col].to_f32();
            }
            out[row * out_dim + col] = sum;
        }
    }
    out
}

fn append_projection(
    out: &mut [f32],
    offset: usize,
    projection: &[f32],
    batch: usize,
    dim: usize,
) {
    for row in 0..batch {
        let dst = offset + row * dim;
        let src = row * dim;
        out[dst..dst + dim].copy_from_slice(&projection[src..src + dim]);
    }
}

fn read_f32_buffer(dev: &VulkanDevice, buf: &VulkanBuffer) -> Result<Vec<f32>> {
    let bytes = VulkanBuffer::read_back(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        buf,
    )?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

fn silu(x: f32) -> f32 {
    if x >= 0.0 {
        x / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        x * e / (1.0 + e)
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_conv_split(
    qkv: &[f32],
    z: &[f32],
    a: &[f32],
    b: &[f32],
    conv_w: &[f32],
    conv_state: &[f32],
    batch: usize,
    qkv_dim: usize,
    qk_dim: usize,
    v_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let state_len = kernel_size - 1;
    let mut state = conv_state.to_vec();
    let mut q_out = vec![0.0f32; batch * qk_dim];
    let mut k_out = vec![0.0f32; batch * qk_dim];
    let mut v_out = vec![0.0f32; batch * v_dim];
    let mut z_out = vec![0.0f32; batch * z_dim];
    let mut a_out = vec![0.0f32; batch * a_dim];
    let mut b_out = vec![0.0f32; batch * b_dim];

    for row in 0..batch {
        for col in 0..qkv_dim {
            let x = qkv[row * qkv_dim + col];
            let state_base = (row * qkv_dim + col) * state_len;
            let mut sum = 0.0f32;
            for k in 0..kernel_size {
                let x_val = if k < state_len {
                    state[state_base + k]
                } else {
                    x
                };
                sum += x_val * conv_w[col * kernel_size + k];
            }
            if state_len > 0 {
                for s in 0..state_len.saturating_sub(1) {
                    state[state_base + s] = state[state_base + s + 1];
                }
                state[state_base + state_len - 1] = x;
            }

            let y = silu(sum);
            if col < qk_dim {
                q_out[row * qk_dim + col] = y;
            } else if col < 2 * qk_dim {
                k_out[row * qk_dim + (col - qk_dim)] = y;
            } else if col < 2 * qk_dim + v_dim {
                v_out[row * v_dim + (col - 2 * qk_dim)] = y;
            }
        }

        z_out[row * z_dim..(row + 1) * z_dim]
            .copy_from_slice(&z[row * z_dim..(row + 1) * z_dim]);
        a_out[row * a_dim..(row + 1) * a_dim]
            .copy_from_slice(&a[row * a_dim..(row + 1) * a_dim]);
        b_out[row * b_dim..(row + 1) * b_dim]
            .copy_from_slice(&b[row * b_dim..(row + 1) * b_dim]);
    }

    (q_out, k_out, v_out, z_out, a_out, b_out, state)
}

fn assert_close(label: &str, got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "{label} length mismatch");
    let max_abs = got
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs <= tol, "{label} max abs diff {max_abs}");
}

fn run_gdn_in_proj_case(shader: &'static str, row_group_size: usize, batch: usize) -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let hidden = 7usize;
    let qkv_dim = 13usize;
    let z_dim = 9usize;
    let a_dim = 5usize;
    let b_dim = 3usize;
    let total_out = qkv_dim + z_dim + a_dim + b_dim;
    let x: Vec<f32> = (0..batch * hidden)
        .map(|i| (((i * 17 + 5) % 97) as f32 - 48.0) * 0.015625)
        .collect();
    let qkv_w: Vec<bf16> = (0..hidden * qkv_dim)
        .map(|i| bf16::from_f32((((i * 19 + 3) % 109) as f32 - 54.0) * 0.0068359375))
        .collect();
    let z_w: Vec<bf16> = (0..hidden * z_dim)
        .map(|i| bf16::from_f32((((i * 23 + 7) % 113) as f32 - 56.0) * 0.005859375))
        .collect();
    let a_w: Vec<bf16> = (0..hidden * a_dim)
        .map(|i| bf16::from_f32((((i * 29 + 11) % 127) as f32 - 63.0) * 0.0048828125))
        .collect();
    let b_w: Vec<bf16> = (0..hidden * b_dim)
        .map(|i| bf16::from_f32((((i * 31 + 13) % 131) as f32 - 65.0) * 0.00439453125))
        .collect();

    let x_buf = kernels::upload_f32_buffer_from_slice(&dev, &x)?;
    let qkv_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &qkv_w)?;
    let z_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &z_w)?;
    let a_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &a_w)?;
    let b_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &b_w)?;
    let out_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * total_out * 4) as u64,
    )?;

    let dispatch_cols = qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + a_dim + b_dim;
    let mut commands = CommandBatch::new(&dev)?;
    commands.record_shader(
        shader,
        &[
            x_buf.handle(),
            qkv_buf.handle(),
            z_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
            out_buf.handle(),
        ],
        &[
            hidden as u32,
            qkv_dim as u32,
            z_dim as u32,
            a_dim as u32,
            b_dim as u32,
            total_out as u32,
            batch as u32,
        ],
        Workgroups::OneD((batch.div_ceil(row_group_size) * dispatch_cols.div_ceil(80)) as u32),
    )?;
    commands.submit_and_wait("gdn in-proj rows parity")?;

    let bytes = VulkanBuffer::read_back(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &out_buf,
    )?;
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

    let qkv = cpu_linear(&x, &qkv_w, batch, hidden, qkv_dim);
    let z = cpu_linear(&x, &z_w, batch, hidden, z_dim);
    let a = cpu_linear(&x, &a_w, batch, hidden, a_dim);
    let b = cpu_linear(&x, &b_w, batch, hidden, b_dim);
    let mut expected = vec![0.0f32; batch * total_out];
    append_projection(&mut expected, 0, &qkv, batch, qkv_dim);
    append_projection(&mut expected, batch * qkv_dim, &z, batch, z_dim);
    append_projection(
        &mut expected,
        batch * (qkv_dim + z_dim),
        &a,
        batch,
        a_dim,
    );
    append_projection(
        &mut expected,
        batch * (qkv_dim + z_dim + a_dim),
        &b,
        batch,
        b_dim,
    );

    let max_abs = got
        .iter()
        .zip(expected.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs < 1e-4, "max abs diff {max_abs}");
    Ok(())
}

#[test]
fn gdn_in_proj_rows4_conv_split_matches_cpu_with_tail_rows_and_odd_pairs() -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let batch = 17usize;
    let hidden = 7usize;
    let qk_dim = 5usize;
    let v_dim = 5usize;
    let qkv_dim = 2 * qk_dim + v_dim;
    let z_dim = 9usize;
    let a_dim = 5usize;
    let b_dim = 3usize;
    let total_out = qkv_dim + z_dim + a_dim + b_dim;
    let kernel_size = 4usize;
    let state_len = kernel_size - 1;

    let x: Vec<f32> = (0..batch * hidden)
        .map(|i| (((i * 17 + 5) % 97) as f32 - 48.0) * 0.015625)
        .collect();
    let qkv_w: Vec<bf16> = (0..hidden * qkv_dim)
        .map(|i| bf16::from_f32((((i * 19 + 3) % 109) as f32 - 54.0) * 0.0068359375))
        .collect();
    let z_w: Vec<bf16> = (0..hidden * z_dim)
        .map(|i| bf16::from_f32((((i * 23 + 7) % 113) as f32 - 56.0) * 0.005859375))
        .collect();
    let a_w: Vec<bf16> = (0..hidden * a_dim)
        .map(|i| bf16::from_f32((((i * 29 + 11) % 127) as f32 - 63.0) * 0.0048828125))
        .collect();
    let b_w: Vec<bf16> = (0..hidden * b_dim)
        .map(|i| bf16::from_f32((((i * 31 + 13) % 131) as f32 - 65.0) * 0.00439453125))
        .collect();
    let conv_w: Vec<f32> = (0..qkv_dim * kernel_size)
        .map(|i| (((i * 13 + 3) % 41) as f32 - 20.0) * 0.0125)
        .collect();
    let conv_state: Vec<f32> = (0..batch * qkv_dim * state_len)
        .map(|i| (((i * 7 + 2) % 53) as f32 - 26.0) * 0.01)
        .collect();

    let x_buf = kernels::upload_f32_buffer_from_slice(&dev, &x)?;
    let qkv_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &qkv_w)?;
    let z_weight_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &z_w)?;
    let a_weight_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &a_w)?;
    let b_weight_buf = kernels::upload_bf16_packed_buffer_from_slice(&dev, &b_w)?;
    let conv_weight_buf = kernels::upload_f32_buffer_from_slice(&dev, &conv_w)?;
    let conv_state_buf = kernels::upload_f32_buffer_from_slice(&dev, &conv_state)?;
    let q_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * qk_dim * 4) as u64,
    )?;
    let k_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * qk_dim * 4) as u64,
    )?;
    let v_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * v_dim * 4) as u64,
    )?;
    let z_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * z_dim * 4) as u64,
    )?;
    let a_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * a_dim * 4) as u64,
    )?;
    let b_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * b_dim * 4) as u64,
    )?;

    let dispatch_cols = qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + a_dim + b_dim;
    let mut commands = CommandBatch::new(&dev)?;
    commands.record_shader(
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W_CONV_SPLIT,
        &[
            x_buf.handle(),
            qkv_buf.handle(),
            z_weight_buf.handle(),
            a_weight_buf.handle(),
            b_weight_buf.handle(),
            conv_weight_buf.handle(),
            conv_state_buf.handle(),
            q_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
            z_buf.handle(),
            a_buf.handle(),
            b_buf.handle(),
        ],
        &[
            hidden as u32,
            qkv_dim as u32,
            z_dim as u32,
            a_dim as u32,
            b_dim as u32,
            total_out as u32,
            batch as u32,
            qk_dim as u32,
            v_dim as u32,
            kernel_size as u32,
        ],
        Workgroups::OneD((batch.div_ceil(4) * dispatch_cols.div_ceil(80)) as u32),
    )?;
    commands.submit_and_wait("gdn in-proj rows4 conv-split parity")?;

    let qkv = cpu_linear(&x, &qkv_w, batch, hidden, qkv_dim);
    let z = cpu_linear(&x, &z_w, batch, hidden, z_dim);
    let a = cpu_linear(&x, &a_w, batch, hidden, a_dim);
    let b = cpu_linear(&x, &b_w, batch, hidden, b_dim);
    let (q_exp, k_exp, v_exp, z_exp, a_exp, b_exp, state_exp) = cpu_conv_split(
        &qkv,
        &z,
        &a,
        &b,
        &conv_w,
        &conv_state,
        batch,
        qkv_dim,
        qk_dim,
        v_dim,
        z_dim,
        a_dim,
        b_dim,
        kernel_size,
    );

    assert_close("q", &read_f32_buffer(&dev, &q_buf)?, &q_exp, 1e-4);
    assert_close("k", &read_f32_buffer(&dev, &k_buf)?, &k_exp, 1e-4);
    assert_close("v", &read_f32_buffer(&dev, &v_buf)?, &v_exp, 1e-4);
    assert_close("z", &read_f32_buffer(&dev, &z_buf)?, &z_exp, 1e-4);
    assert_close("a", &read_f32_buffer(&dev, &a_buf)?, &a_exp, 1e-4);
    assert_close("b", &read_f32_buffer(&dev, &b_buf)?, &b_exp, 1e-4);
    assert_close(
        "conv_state",
        &read_f32_buffer(&dev, &conv_state_buf)?,
        &state_exp,
        1e-4,
    );
    Ok(())
}

#[test]
fn gdn_in_proj_rows2_matches_cpu_with_tail_rows_and_odd_pairs() -> Result<()> {
    run_gdn_in_proj_case(
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS2_BF16W,
        2,
        5,
    )
}

#[test]
fn gdn_in_proj_rows4_matches_cpu_with_tail_rows_and_odd_pairs() -> Result<()> {
    run_gdn_in_proj_case(
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W,
        4,
        17,
    )
}

#[test]
fn gdn_in_proj_rows8_matches_cpu_with_tail_rows_and_odd_pairs() -> Result<()> {
    run_gdn_in_proj_case(
        shaders::GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS8_BF16W,
        8,
        65,
    )
}

use anyhow::Result;
use half::bf16;
use kiln_vulkan_kernel::{CommandBatch, VulkanBuffer, VulkanDevice, Workgroups, shaders};

fn upload_bytes(dev: &VulkanDevice, bytes: &[u8]) -> Result<VulkanBuffer> {
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        bytes.len() as u64,
    )?;
    VulkanBuffer::upload_data(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &buf,
        bytes,
    )?;
    Ok(buf)
}

fn upload_f32(dev: &VulkanDevice, data: &[f32]) -> Result<VulkanBuffer> {
    upload_bytes(dev, bytemuck::cast_slice(data))
}

fn upload_packed_bf16(dev: &VulkanDevice, data: &[f32]) -> Result<(VulkanBuffer, Vec<f32>)> {
    let rounded: Vec<f32> = data.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let mut packed = Vec::with_capacity(data.len().div_ceil(2));
    for pair in data.chunks(2) {
        let lo = bf16::from_f32(pair[0]).to_bits() as u32;
        let hi = pair
            .get(1)
            .map(|&v| (bf16::from_f32(v).to_bits() as u32) << 16)
            .unwrap_or(0);
        packed.push(lo | hi);
    }
    Ok((upload_bytes(dev, bytemuck::cast_slice(&packed))?, rounded))
}

fn read_f32(dev: &VulkanDevice, buf: &VulkanBuffer) -> Result<Vec<f32>> {
    let bytes = VulkanBuffer::read_back(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        buf,
    )?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

fn cpu_linear(x: &[f32], w: &[f32], batch: usize, hidden: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * out_dim];
    for row in 0..batch {
        for col in 0..out_dim {
            let mut acc = 0.0f32;
            for h in 0..hidden {
                acc += x[row * hidden + h] * w[h * out_dim + col];
            }
            out[row * out_dim + col] = acc;
        }
    }
    out
}

fn assert_close(name: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{name} length");
    let max_abs = got
        .iter()
        .zip(expected)
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    assert!(max_abs <= 1e-3, "{name} max abs diff {max_abs}");
}

fn run_case(shader: &'static str, batch: usize) -> Result<()> {
    let dev = VulkanDevice::new()?;
    let hidden = 5usize;
    let num_heads = 2usize;
    let num_kv_heads = 1usize;
    let head_dim = 3usize;
    let q_dim = num_heads * 2 * head_dim;
    let q_out = num_heads * head_dim;
    let k_dim = num_kv_heads * head_dim;
    let v_dim = k_dim;
    let total_out = q_dim + k_dim + v_dim;

    let x: Vec<f32> = (0..batch * hidden)
        .map(|i| ((i as f32) * 0.17).sin() * 0.5)
        .collect();
    let q_w: Vec<f32> = (0..hidden * q_dim)
        .map(|i| ((i as f32) * 0.11).cos() * 0.25)
        .collect();
    let k_w: Vec<f32> = (0..hidden * k_dim)
        .map(|i| ((i as f32) * 0.19).sin() * 0.2)
        .collect();
    let v_w: Vec<f32> = (0..hidden * v_dim)
        .map(|i| ((i as f32) * 0.23).cos() * 0.15)
        .collect();

    let x_buf = upload_f32(&dev, &x)?;
    let (q_w_buf, q_w_rounded) = upload_packed_bf16(&dev, &q_w)?;
    let (k_w_buf, k_w_rounded) = upload_packed_bf16(&dev, &k_w)?;
    let (v_w_buf, v_w_rounded) = upload_packed_bf16(&dev, &v_w)?;
    let q_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * q_out * 4) as u64,
    )?;
    let gate_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * q_out * 4) as u64,
    )?;
    let k_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * k_dim * 4) as u64,
    )?;
    let v_buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (batch * v_dim * 4) as u64,
    )?;

    let row_groups = if shader == shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS4_BF16W {
        batch.div_ceil(4)
    } else {
        batch
    };
    let mut commands = CommandBatch::new(&dev)?;
    commands.record_shader(
        shader,
        &[
            x_buf.handle(),
            q_w_buf.handle(),
            k_w_buf.handle(),
            v_w_buf.handle(),
            q_buf.handle(),
            gate_buf.handle(),
            k_buf.handle(),
            v_buf.handle(),
        ],
        &[
            hidden as u32,
            q_dim as u32,
            k_dim as u32,
            v_dim as u32,
            total_out as u32,
            batch as u32,
            head_dim as u32,
        ],
        Workgroups::OneD((row_groups * total_out.div_ceil(16)) as u32),
    )?;
    commands.submit_and_wait("full_attn_qkv_gate_split_parity")?;

    let combined_q = cpu_linear(&x, &q_w_rounded, batch, hidden, q_dim);
    let expected_k = cpu_linear(&x, &k_w_rounded, batch, hidden, k_dim);
    let expected_v = cpu_linear(&x, &v_w_rounded, batch, hidden, v_dim);
    let mut expected_q = vec![0.0f32; batch * q_out];
    let mut expected_gate = vec![0.0f32; batch * q_out];
    for row in 0..batch {
        for head in 0..num_heads {
            for dim in 0..head_dim {
                let src_base = row * q_dim + head * 2 * head_dim;
                let dst = row * q_out + head * head_dim + dim;
                expected_q[dst] = combined_q[src_base + dim];
                expected_gate[dst] = combined_q[src_base + head_dim + dim];
            }
        }
    }

    assert_close("q", &read_f32(&dev, &q_buf)?, &expected_q);
    assert_close("gate", &read_f32(&dev, &gate_buf)?, &expected_gate);
    assert_close("k", &read_f32(&dev, &k_buf)?, &expected_k);
    assert_close("v", &read_f32(&dev, &v_buf)?, &expected_v);
    Ok(())
}

#[test]
fn direct_full_attn_qkv_gate_split_rows1_matches_cpu() -> Result<()> {
    if !VulkanDevice::probe() {
        return Ok(());
    }
    run_case(shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_BF16W, 3)
}

#[test]
fn direct_full_attn_qkv_gate_split_rows4_matches_cpu() -> Result<()> {
    if !VulkanDevice::probe() {
        return Ok(());
    }
    run_case(shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS4_BF16W, 5)
}

#[test]
fn direct_full_attn_qkv_gate_split_rows4_b1_matches_cpu() -> Result<()> {
    if !VulkanDevice::probe() {
        return Ok(());
    }
    run_case(shaders::FULL_ATTN_QKV_GATE_SPLIT_BATCHED_ROWS4_BF16W, 1)
}

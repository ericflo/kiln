use anyhow::Result;
use kiln_vulkan_kernel::{CommandBatch, VulkanBuffer, VulkanDevice, Workgroups, shaders};

fn upload_f32(dev: &VulkanDevice, data: &[f32]) -> Result<VulkanBuffer> {
    let bytes = bytemuck::cast_slice(data);
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

fn cpu_l2_expand(
    x: &[f32],
    rows_in: usize,
    hidden: usize,
    eps: f32,
    scale: f32,
    gqa_ratio: usize,
) -> Vec<f32> {
    let rows_out = rows_in * gqa_ratio;
    let mut out = vec![0.0f32; rows_out * hidden];
    for r_in in 0..rows_in {
        let base_in = r_in * hidden;
        let sq: f32 = x[base_in..base_in + hidden].iter().map(|v| v * v).sum();
        let coef = scale / (sq + eps).sqrt();
        for copy in 0..gqa_ratio {
            let base_out = (r_in * gqa_ratio + copy) * hidden;
            for i in 0..hidden {
                out[base_out + i] = x[base_in + i] * coef;
            }
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
    assert!(max_abs <= 1e-5, "{name} max abs diff {max_abs}");
}

#[test]
fn fused_l2_norm_qk_per_row_matches_cpu() -> Result<()> {
    if !VulkanDevice::probe() {
        return Ok(());
    }
    let dev = VulkanDevice::new()?;
    let rows_in = 6usize;
    let hidden = 128usize;
    let gqa_ratio = 2usize;
    let rows_out = rows_in * gqa_ratio;
    let eps = 1e-6f32;
    let q_scale = 1.0f32 / (hidden as f32).sqrt();
    let k_scale = 1.0f32;

    let q: Vec<f32> = (0..rows_in * hidden)
        .map(|i| ((i % 41) as f32 - 20.0) * 0.017)
        .collect();
    let k: Vec<f32> = (0..rows_in * hidden)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.013)
        .collect();
    let q_buf = upload_f32(&dev, &q)?;
    let k_buf = upload_f32(&dev, &k)?;
    let q_out = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (rows_out * hidden * 4) as u64,
    )?;
    let k_out = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (rows_out * hidden * 4) as u64,
    )?;

    let mut batch = CommandBatch::new(&dev)?;
    batch.record_shader(
        shaders::L2_NORM_QK_PER_ROW,
        &[
            q_buf.handle(),
            k_buf.handle(),
            q_out.handle(),
            k_out.handle(),
        ],
        &[
            rows_in as u32,
            hidden as u32,
            eps.to_bits(),
            q_scale.to_bits(),
            k_scale.to_bits(),
            gqa_ratio as u32,
        ],
        Workgroups::OneD(rows_out as u32),
    )?;
    batch.submit_and_wait("l2_norm_qk_per_row_parity")?;

    let q_expected = cpu_l2_expand(&q, rows_in, hidden, eps, q_scale, gqa_ratio);
    let k_expected = cpu_l2_expand(&k, rows_in, hidden, eps, k_scale, gqa_ratio);
    assert_close("q", &read_f32(&dev, &q_out)?, &q_expected);
    assert_close("k", &read_f32(&dev, &k_out)?, &k_expected);
    Ok(())
}

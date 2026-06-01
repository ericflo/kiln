use anyhow::Result;
use kiln_vulkan_kernel::{CommandBatch, VulkanBuffer, VulkanDevice, Workgroups, kernels, shaders};

#[test]
fn rope_tables_from_seq_lens_matches_cpu() -> Result<()> {
    let Ok(dev) = VulkanDevice::new() else {
        eprintln!("skipping: Vulkan device unavailable");
        return Ok(());
    };

    let seq_lens = [4u32, 18, 1025, 32768];
    let inv_freq = [1.0f32, 0.1, 0.01, 0.001];
    let rows = seq_lens.len();
    let half_rotary = inv_freq.len();
    let out_len = rows * half_rotary;
    let seq_bytes = bytemuck::cast_slice(&seq_lens);
    let out_bytes = (out_len * 4) as u64;

    let seq_buf = VulkanBuffer::create_host_visible(
        dev.device(),
        dev.host_visible_mem_type(),
        seq_bytes.len() as u64,
    )?;
    seq_buf.write_mapped(seq_bytes)?;
    let inv_buf = kernels::upload_f32_buffer_from_slice(&dev, &inv_freq)?;
    let cos_buf =
        VulkanBuffer::create_host_visible(dev.device(), dev.host_visible_mem_type(), out_bytes)?;
    let sin_buf =
        VulkanBuffer::create_host_visible(dev.device(), dev.host_visible_mem_type(), out_bytes)?;

    let mut batch = CommandBatch::new(&dev)?;
    batch.record_shader(
        shaders::VK_ROPE_TABLES_FROM_SEQ_LENS_F32,
        &[
            seq_buf.handle(),
            inv_buf.handle(),
            cos_buf.handle(),
            sin_buf.handle(),
        ],
        &[rows as u32, half_rotary as u32],
        Workgroups::OneD(out_len.div_ceil(256) as u32),
    )?;
    batch.submit_and_wait("test rope table fill")?;

    let cos_bytes = cos_buf.read_mapped(out_bytes as usize)?;
    let sin_bytes = sin_buf.read_mapped(out_bytes as usize)?;
    let cos: &[f32] = bytemuck::cast_slice(&cos_bytes);
    let sin: &[f32] = bytemuck::cast_slice(&sin_bytes);
    for (row, &seq_len) in seq_lens.iter().enumerate() {
        let position = seq_len.saturating_sub(1) as f32;
        for (pair, &inv) in inv_freq.iter().enumerate() {
            let idx = row * half_rotary + pair;
            let freq = position * inv;
            let cos_diff = (cos[idx] - freq.cos()).abs();
            let sin_diff = (sin[idx] - freq.sin()).abs();
            assert!(cos_diff < 1e-3, "cos row={row} pair={pair} diff={cos_diff:e}");
            assert!(sin_diff < 1e-3, "sin row={row} pair={pair} diff={sin_diff:e}");
        }
    }

    Ok(())
}

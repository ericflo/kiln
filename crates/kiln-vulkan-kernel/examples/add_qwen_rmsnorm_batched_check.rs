//! Verification probe for batched fused residual add + Qwen RMSNorm.

use anyhow::{ensure, Context, Result};
use kiln_vulkan_kernel::resident::dispatch_add_qwen_rmsnorm_batched_resident;
use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice};

fn upload_f32_buf(dev: &VulkanDevice, data: &[f32]) -> Result<VulkanBuffer> {
    let bytes = bytemuck::cast_slice(data);
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        bytes.len().max(4) as u64,
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

fn alloc_f32_buf(dev: &VulkanDevice, len: usize) -> Result<VulkanBuffer> {
    VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        (len.max(1) * 4) as u64,
    )
}

fn read_back_f32(dev: &VulkanDevice, buf: &VulkanBuffer) -> Result<Vec<f32>> {
    let bytes = VulkanBuffer::read_back(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        buf,
    )?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

fn main() -> Result<()> {
    let dev = VulkanDevice::new().context("create Vulkan device")?;
    let rows = 4usize;
    let hidden = 17usize;
    let eps = 1e-6f32;

    let a: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.03125)
        .collect();
    let b: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.015625)
        .collect();
    let weight: Vec<f32> = (0..hidden).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();

    let mut expected_sum = vec![0.0f32; rows * hidden];
    let mut expected_out = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let base = r * hidden;
        let mut sq_sum = 0.0f32;
        for c in 0..hidden {
            let v = a[base + c] + b[base + c];
            expected_sum[base + c] = v;
            sq_sum += v * v;
        }
        let rms_inv = (sq_sum / hidden as f32 + eps).sqrt().recip();
        for c in 0..hidden {
            expected_out[base + c] = (1.0 + weight[c]) * expected_sum[base + c] * rms_inv;
        }
    }

    let a_buf = upload_f32_buf(&dev, &a)?;
    let b_buf = upload_f32_buf(&dev, &b)?;
    let weight_buf = upload_f32_buf(&dev, &weight)?;
    let sum_buf = alloc_f32_buf(&dev, expected_sum.len())?;
    let out_buf = alloc_f32_buf(&dev, expected_out.len())?;

    dispatch_add_qwen_rmsnorm_batched_resident(
        &dev,
        &a_buf,
        &b_buf,
        &weight_buf,
        &sum_buf,
        &out_buf,
        rows,
        hidden,
        eps,
    )?;

    let got_sum = read_back_f32(&dev, &sum_buf)?;
    let got_out = read_back_f32(&dev, &out_buf)?;
    ensure!(got_sum == expected_sum, "sum mismatch");
    for (i, (&g, &e)) in got_out.iter().zip(expected_out.iter()).enumerate() {
        ensure!(
            (g - e).abs() <= 1.0e-5,
            "idx {i}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }

    println!("add_qwen_rmsnorm_batched_check: OK rows={rows} hidden={hidden}");
    Ok(())
}

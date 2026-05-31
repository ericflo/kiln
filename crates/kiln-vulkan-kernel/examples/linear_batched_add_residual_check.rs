//! Verification probe for batched BF16 linear + residual fusion.

use anyhow::{ensure, Context, Result};
use half::bf16;
use kiln_vulkan_kernel::resident::dispatch_linear_decode_batched_bf16w_add_residual_resident;
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

fn upload_u32_buf(dev: &VulkanDevice, data: &[u32]) -> Result<VulkanBuffer> {
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

fn pack_bf16_pairs(values: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let rounded: Vec<f32> = values.iter().map(|&v| bf16::from_f32(v).to_f32()).collect();
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));
    for pair in rounded.chunks(2) {
        let lo = bf16::from_f32(pair[0]).to_bits() as u32;
        let hi = if pair.len() > 1 {
            (bf16::from_f32(pair[1]).to_bits() as u32) << 16
        } else {
            0
        };
        packed.push(lo | hi);
    }
    (packed, rounded)
}

fn main() -> Result<()> {
    let dev = VulkanDevice::new().context("create Vulkan device")?;
    let batch = 3usize;
    let hidden = 9usize;
    let out_dim = 11usize;

    let x: Vec<f32> = (0..batch * hidden)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.03125)
        .collect();
    let w_f32: Vec<f32> = (0..hidden * out_dim)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.015625)
        .collect();
    let residual: Vec<f32> = (0..batch * out_dim)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.0625)
        .collect();
    let (w_packed, w_rounded) = pack_bf16_pairs(&w_f32);

    let mut expected = vec![0.0f32; batch * out_dim];
    for b in 0..batch {
        for c in 0..out_dim {
            let mut acc = residual[b * out_dim + c];
            for h in 0..hidden {
                acc += x[b * hidden + h] * w_rounded[h * out_dim + c];
            }
            expected[b * out_dim + c] = acc;
        }
    }

    let x_buf = upload_f32_buf(&dev, &x)?;
    let w_buf = upload_u32_buf(&dev, &w_packed)?;
    let residual_buf = upload_f32_buf(&dev, &residual)?;
    let out_buf = alloc_f32_buf(&dev, expected.len())?;

    dispatch_linear_decode_batched_bf16w_add_residual_resident(
        &dev,
        &x_buf,
        &w_buf,
        &residual_buf,
        &out_buf,
        batch,
        hidden,
        out_dim,
    )?;

    let got = read_back_f32(&dev, &out_buf)?;
    ensure!(got.len() == expected.len(), "unexpected output length");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        ensure!(
            (g - e).abs() <= 1.0e-6,
            "idx {i}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }

    println!(
        "linear_batched_add_residual_check: OK batch={batch} hidden={hidden} out_dim={out_dim}"
    );
    Ok(())
}

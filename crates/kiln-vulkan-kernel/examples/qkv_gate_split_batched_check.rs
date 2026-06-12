//! Verification probe for the batched full-attention QKV/gate split.

use anyhow::{Context, Result, ensure};
use kiln_vulkan_kernel::resident::dispatch_qkv_gate_split_batched_resident;
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
    let batch = 3usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 5usize;
    let full_heads_total = num_heads * head_dim;
    let kv_heads_total = num_kv_heads * head_dim;
    let combined_stride = 2 * full_heads_total + 2 * kv_heads_total;

    let mut combined = vec![0.0f32; batch * combined_stride];
    let mut expected_q = vec![0.0f32; batch * full_heads_total];
    let mut expected_gate = vec![0.0f32; batch * full_heads_total];
    let mut expected_k = vec![0.0f32; batch * kv_heads_total];
    let mut expected_v = vec![0.0f32; batch * kv_heads_total];

    for row in 0..batch {
        let row_base = row * combined_stride;
        for h in 0..num_heads {
            for d in 0..head_dim {
                let flat = h * head_dim + d;
                let q = 1000.0 + (row * 100 + flat) as f32;
                let gate = 2000.0 + (row * 100 + flat) as f32;
                combined[row_base + h * 2 * head_dim + d] = q;
                combined[row_base + h * 2 * head_dim + head_dim + d] = gate;
                expected_q[row * full_heads_total + flat] = q;
                expected_gate[row * full_heads_total + flat] = gate;
            }
        }
        for h in 0..num_kv_heads {
            for d in 0..head_dim {
                let flat = h * head_dim + d;
                let k = 3000.0 + (row * 100 + flat) as f32;
                let v = 4000.0 + (row * 100 + flat) as f32;
                combined[row_base + 2 * full_heads_total + flat] = k;
                combined[row_base + 2 * full_heads_total + kv_heads_total + flat] = v;
                expected_k[row * kv_heads_total + flat] = k;
                expected_v[row * kv_heads_total + flat] = v;
            }
        }
    }

    let combined_buf = upload_f32_buf(&dev, &combined)?;
    let q_buf = alloc_f32_buf(&dev, expected_q.len())?;
    let gate_buf = alloc_f32_buf(&dev, expected_gate.len())?;
    let k_buf = alloc_f32_buf(&dev, expected_k.len())?;
    let v_buf = alloc_f32_buf(&dev, expected_v.len())?;

    dispatch_qkv_gate_split_batched_resident(
        &dev,
        &combined_buf,
        &q_buf,
        &gate_buf,
        &k_buf,
        &v_buf,
        batch,
        num_heads,
        num_kv_heads,
        head_dim,
    )?;

    ensure!(read_back_f32(&dev, &q_buf)? == expected_q, "q mismatch");
    ensure!(
        read_back_f32(&dev, &gate_buf)? == expected_gate,
        "gate mismatch"
    );
    ensure!(read_back_f32(&dev, &k_buf)? == expected_k, "k mismatch");
    ensure!(read_back_f32(&dev, &v_buf)? == expected_v, "v mismatch");

    println!(
        "qkv_gate_split_batched_check: OK batch={batch} num_heads={num_heads} num_kv_heads={num_kv_heads} head_dim={head_dim}"
    );
    Ok(())
}

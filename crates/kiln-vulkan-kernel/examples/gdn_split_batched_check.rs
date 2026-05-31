//! Verification probe for batched GDN split kernels.

use anyhow::{ensure, Context, Result};
use kiln_vulkan_kernel::resident::{
    dispatch_gdn_in_proj_split_batched_resident, dispatch_gdn_qkv_split_batched_resident,
};
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

fn check_in_proj_split(dev: &VulkanDevice) -> Result<()> {
    let batch = 3usize;
    let qkv_dim = 7usize;
    let z_dim = 5usize;
    let a_dim = 3usize;
    let b_dim = 2usize;

    let qkv: Vec<f32> = (0..batch * qkv_dim).map(|i| 100.0 + i as f32).collect();
    let z: Vec<f32> = (0..batch * z_dim).map(|i| 200.0 + i as f32).collect();
    let a: Vec<f32> = (0..batch * a_dim).map(|i| 300.0 + i as f32).collect();
    let b: Vec<f32> = (0..batch * b_dim).map(|i| 400.0 + i as f32).collect();
    let mut combined = Vec::with_capacity(qkv.len() + z.len() + a.len() + b.len());
    combined.extend_from_slice(&qkv);
    combined.extend_from_slice(&z);
    combined.extend_from_slice(&a);
    combined.extend_from_slice(&b);

    let combined_buf = upload_f32_buf(dev, &combined)?;
    let qkv_buf = alloc_f32_buf(dev, qkv.len())?;
    let z_buf = alloc_f32_buf(dev, z.len())?;
    let a_buf = alloc_f32_buf(dev, a.len())?;
    let b_buf = alloc_f32_buf(dev, b.len())?;

    dispatch_gdn_in_proj_split_batched_resident(
        dev,
        &combined_buf,
        &qkv_buf,
        &z_buf,
        &a_buf,
        &b_buf,
        batch,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
    )?;

    ensure!(read_back_f32(dev, &qkv_buf)? == qkv, "qkv split mismatch");
    ensure!(read_back_f32(dev, &z_buf)? == z, "z split mismatch");
    ensure!(read_back_f32(dev, &a_buf)? == a, "a split mismatch");
    ensure!(read_back_f32(dev, &b_buf)? == b, "b split mismatch");
    Ok(())
}

fn check_qkv_split(dev: &VulkanDevice) -> Result<()> {
    let batch = 3usize;
    let qk_dim = 6usize;
    let v_dim = 4usize;
    let per_row = 2 * qk_dim + v_dim;
    let mut mixed = vec![0.0f32; batch * per_row];
    let mut q = vec![0.0f32; batch * qk_dim];
    let mut k = vec![0.0f32; batch * qk_dim];
    let mut v = vec![0.0f32; batch * v_dim];

    for row in 0..batch {
        for i in 0..qk_dim {
            let qv = 1000.0 + (row * 100 + i) as f32;
            let kv = 2000.0 + (row * 100 + i) as f32;
            mixed[row * per_row + i] = qv;
            mixed[row * per_row + qk_dim + i] = kv;
            q[row * qk_dim + i] = qv;
            k[row * qk_dim + i] = kv;
        }
        for i in 0..v_dim {
            let vv = 3000.0 + (row * 100 + i) as f32;
            mixed[row * per_row + 2 * qk_dim + i] = vv;
            v[row * v_dim + i] = vv;
        }
    }

    let mixed_buf = upload_f32_buf(dev, &mixed)?;
    let q_buf = alloc_f32_buf(dev, q.len())?;
    let k_buf = alloc_f32_buf(dev, k.len())?;
    let v_buf = alloc_f32_buf(dev, v.len())?;

    dispatch_gdn_qkv_split_batched_resident(
        dev, &mixed_buf, &q_buf, &k_buf, &v_buf, batch, qk_dim, v_dim,
    )?;

    ensure!(read_back_f32(dev, &q_buf)? == q, "q split mismatch");
    ensure!(read_back_f32(dev, &k_buf)? == k, "k split mismatch");
    ensure!(read_back_f32(dev, &v_buf)? == v, "v split mismatch");
    Ok(())
}

fn main() -> Result<()> {
    let dev = VulkanDevice::new().context("create Vulkan device")?;
    check_in_proj_split(&dev)?;
    check_qkv_split(&dev)?;
    println!("gdn_split_batched_check: OK");
    Ok(())
}

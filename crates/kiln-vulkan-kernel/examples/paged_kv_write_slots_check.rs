//! Verification probe for the batched resident paged K/V slot write.

use anyhow::{Context, Result, ensure};
use kiln_vulkan_kernel::resident::dispatch_paged_kv_write_slots_resident;
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
    let num_kv_heads = 2usize;
    let head_dim = 4usize;
    let total_slots = 8usize;
    let elements_per_slot = num_kv_heads * head_dim;
    let pool_elems = total_slots * elements_per_slot;
    let slots = [5u32, 1, 6];

    let k_in: Vec<f32> = (0..batch * elements_per_slot)
        .map(|i| 1000.0 + i as f32)
        .collect();
    let v_in: Vec<f32> = (0..batch * elements_per_slot)
        .map(|i| 2000.0 + i as f32)
        .collect();
    let zero_pool = vec![0.0f32; pool_elems];

    let k_in_buf = upload_f32_buf(&dev, &k_in)?;
    let v_in_buf = upload_f32_buf(&dev, &v_in)?;
    let slots_buf = upload_u32_buf(&dev, &slots)?;
    let k_pool = upload_f32_buf(&dev, &zero_pool)?;
    let v_pool = upload_f32_buf(&dev, &zero_pool)?;

    dispatch_paged_kv_write_slots_resident(
        &dev,
        &k_in_buf,
        &v_in_buf,
        &slots_buf,
        &k_pool,
        &v_pool,
        batch,
        num_kv_heads,
        head_dim,
        total_slots,
    )?;

    let k_got = read_back_f32(&dev, &k_pool)?;
    let v_got = read_back_f32(&dev, &v_pool)?;

    for row in 0..batch {
        let slot = slots[row] as usize;
        let src = row * elements_per_slot;
        let dst = slot * elements_per_slot;
        ensure!(
            &k_got[dst..dst + elements_per_slot] == &k_in[src..src + elements_per_slot],
            "k row {row} was not written to slot {slot}"
        );
        ensure!(
            &v_got[dst..dst + elements_per_slot] == &v_in[src..src + elements_per_slot],
            "v row {row} was not written to slot {slot}"
        );
    }

    for slot in 0..total_slots {
        if slots.contains(&(slot as u32)) {
            continue;
        }
        let off = slot * elements_per_slot;
        ensure!(
            k_got[off..off + elements_per_slot]
                .iter()
                .all(|&v| v == 0.0),
            "k slot {slot} changed unexpectedly"
        );
        ensure!(
            v_got[off..off + elements_per_slot]
                .iter()
                .all(|&v| v == 0.0),
            "v slot {slot} changed unexpectedly"
        );
    }

    println!(
        "paged_kv_write_slots_check: OK batch={batch} elements_per_slot={elements_per_slot} slots={slots:?}"
    );
    Ok(())
}

//! Verification probe for batched buffer uploads with destination offsets.

use anyhow::{ensure, Context, Result};
use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice};

fn main() -> Result<()> {
    let dev = VulkanDevice::new().context("create Vulkan device")?;
    let dst = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        48,
    )?;

    let a = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let b = [21u8, 22, 23, 24, 25];
    let c = [41u8, 42, 43, 44, 45, 46, 47, 48];
    VulkanBuffer::upload_data_at_offset_batch(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &[(&dst, 0, &a), (&dst, 16, &b), (&dst, 40, &c)],
    )?;

    let got = VulkanBuffer::read_back(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &dst,
    )?;
    let mut expected = vec![0u8; 48];
    expected[0..8].copy_from_slice(&a);
    expected[16..21].copy_from_slice(&b);
    expected[40..48].copy_from_slice(&c);

    ensure!(
        got == expected,
        "offset batch upload mismatch: got {:?}, expected {:?}",
        got,
        expected
    );
    println!("buffer_offset_batch_check: OK bytes={}", got.len());
    Ok(())
}

//! Verification probe for device-local row/batch buffer copies.

use anyhow::{Context, Result, ensure};
use kiln_vulkan_kernel::kernels::{
    copy_device_buffer_rows_to_batch, split_device_buffer_batch_rows,
};
use kiln_vulkan_kernel::{VulkanBuffer, VulkanDevice};
use std::sync::Arc;

fn upload_row(dev: &VulkanDevice, data: &[u8]) -> Result<Arc<VulkanBuffer>> {
    let buf = VulkanBuffer::create_device_local(
        dev.device(),
        dev.device_local_mem_type(),
        data.len() as u64,
    )?;
    VulkanBuffer::upload_data(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        &buf,
        data,
    )?;
    Ok(Arc::new(buf))
}

fn read(dev: &VulkanDevice, buf: &VulkanBuffer) -> Result<Vec<u8>> {
    VulkanBuffer::read_back(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        buf,
    )
}

fn main() -> Result<()> {
    let dev = VulkanDevice::new().context("create Vulkan device")?;
    let rows: Vec<Vec<u8>> = (0..3)
        .map(|row| (0..32).map(|col| (row * 41 + col) as u8).collect())
        .collect();
    let row_buffers: Vec<Arc<VulkanBuffer>> = rows
        .iter()
        .map(|row| upload_row(&dev, row))
        .collect::<Result<Vec<_>>>()?;

    let batch = copy_device_buffer_rows_to_batch(&dev, &row_buffers)?;
    let got_batch = read(&dev, &batch)?;
    let expected_batch: Vec<u8> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    ensure!(
        got_batch == expected_batch,
        "row-to-batch mismatch: got {:?}, expected {:?}",
        got_batch,
        expected_batch
    );

    let split = split_device_buffer_batch_rows(&dev, &batch, rows.len())?;
    ensure!(
        split.len() == rows.len(),
        "split row count mismatch: got {}, expected {}",
        split.len(),
        rows.len()
    );
    for (idx, row_buf) in split.iter().enumerate() {
        let got = read(&dev, row_buf)?;
        ensure!(
            got == rows[idx],
            "split row {idx} mismatch: got {:?}, expected {:?}",
            got,
            rows[idx]
        );
    }

    println!(
        "device_buffer_rows_check: OK rows={} row_bytes={} batch_bytes={}",
        rows.len(),
        rows[0].len(),
        got_batch.len()
    );
    Ok(())
}

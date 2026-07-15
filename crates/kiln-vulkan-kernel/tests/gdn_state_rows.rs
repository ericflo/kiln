use anyhow::Result;
use kiln_vulkan_kernel::{VulkanBuffer, buffer_pool, kernels};

mod support;

const SUITE: &str = "gdn_state_rows";

#[test]
fn gdn_state_batch_copies_recycle_every_output_buffer() -> Result<()> {
    let Some(dev) = support::vulkan_device(SUITE) else {
        return Ok(());
    };
    let device_handle = dev.device().handle();
    let before = buffer_pool::pool_stats_for_device(device_handle);

    let mut source_rows = Vec::new();
    for values in [[11_u32, 12], [21_u32, 22]] {
        let row = buffer_pool::pool_alloc_device_local(&dev, 8)?;
        VulkanBuffer::upload_data(
            dev.device(),
            dev.host_visible_mem_type(),
            dev.queue(),
            dev.queue_family_index(),
            row.as_ref(),
            bytemuck::cast_slice(&values),
        )?;
        source_rows.push(row);
    }
    assert_eq!(
        buffer_pool::pool_stats_for_device(device_handle).borrowed_buffer_count(),
        before.borrowed_buffer_count() + 2,
        "source state rows must borrow two recycler buffers"
    );

    let batch = kernels::copy_device_buffer_rows_to_batch(&dev, &source_rows, 8)?;
    assert_eq!(
        buffer_pool::pool_stats_for_device(device_handle).borrowed_buffer_count(),
        before.borrowed_buffer_count() + 3,
        "assembled state batch must borrow one recycler buffer"
    );

    let split = kernels::split_device_buffer_batch_rows(&dev, batch.as_ref(), 2, 8)?;
    assert_eq!(
        buffer_pool::pool_stats_for_device(device_handle).borrowed_buffer_count(),
        before.borrowed_buffer_count() + 5,
        "scattered state rows must borrow one recycler buffer per row"
    );
    for (row, expected) in split.iter().zip([[11_u32, 12], [21_u32, 22]]) {
        let bytes = VulkanBuffer::read_back(
            dev.device(),
            dev.host_visible_mem_type(),
            dev.queue(),
            dev.queue_family_index(),
            row.as_ref(),
        )?;
        assert_eq!(&bytes[..8], bytemuck::cast_slice(&expected));
    }

    drop(split);
    drop(batch);
    drop(source_rows);
    assert_eq!(
        buffer_pool::pool_stats_for_device(device_handle).borrowed_buffer_count(),
        before.borrowed_buffer_count(),
        "dropping assembled and scattered state buffers must return every borrow"
    );
    Ok(())
}

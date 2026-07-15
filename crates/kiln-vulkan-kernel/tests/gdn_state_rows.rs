use anyhow::Result;
use kiln_vulkan_kernel::{VulkanBuffer, buffer_pool, kernels};
use std::sync::Mutex;

mod support;

const SUITE: &str = "gdn_state_rows";
static TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn gdn_state_batch_copies_recycle_every_output_buffer() -> Result<()> {
    let _test_guard = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
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

#[test]
fn gdn_state_batch_scatter_reuses_existing_row_buffers() -> Result<()> {
    let _test_guard = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let Some(dev) = support::vulkan_device(SUITE) else {
        return Ok(());
    };
    let device_handle = dev.device().handle();
    let before = buffer_pool::pool_stats_for_device(device_handle);

    let mut source_rows = Vec::new();
    for values in [[31_u32, 32], [41_u32, 42]] {
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
    let batch = kernels::copy_device_buffer_rows_to_batch(&dev, &source_rows, 8)?;
    let destination_rows = vec![
        buffer_pool::pool_alloc_device_local(&dev, 8)?,
        buffer_pool::pool_alloc_device_local(&dev, 8)?,
    ];
    let before_scatter = buffer_pool::pool_stats_for_device(device_handle);

    kernels::copy_device_buffer_batch_to_rows(&dev, batch.as_ref(), &destination_rows, 8)?;
    assert_eq!(
        buffer_pool::pool_stats_for_device(device_handle).borrowed_buffer_count(),
        before_scatter.borrowed_buffer_count(),
        "in-place scatter must not acquire or replace a state-row buffer"
    );
    for (row, expected) in destination_rows.iter().zip([[31_u32, 32], [41_u32, 42]]) {
        let bytes = VulkanBuffer::read_back(
            dev.device(),
            dev.host_visible_mem_type(),
            dev.queue(),
            dev.queue_family_index(),
            row.as_ref(),
        )?;
        assert_eq!(&bytes[..8], bytemuck::cast_slice(&expected));
    }

    drop(destination_rows);
    drop(batch);
    drop(source_rows);
    assert_eq!(
        buffer_pool::pool_stats_for_device(device_handle).borrowed_buffer_count(),
        before.borrowed_buffer_count()
    );
    Ok(())
}

#[test]
fn gdn_state_batch_refresh_reuses_existing_batch_buffer() -> Result<()> {
    let _test_guard = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let Some(dev) = support::vulkan_device(SUITE) else {
        return Ok(());
    };
    let device_handle = dev.device().handle();

    let mut source_rows = Vec::new();
    for values in [[51_u32, 52], [61_u32, 62]] {
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
    let batch = buffer_pool::pool_alloc_device_local(&dev, 24)?;
    let before_pool = buffer_pool::pool_stats_for_device(device_handle);
    let before_allocations = kiln_vulkan_kernel::buffer::allocation_stats();

    kernels::copy_device_buffer_rows_to_existing_batch(&dev, &source_rows, batch.as_ref(), 8)?;

    assert_eq!(
        buffer_pool::pool_stats_for_device(device_handle),
        before_pool,
        "in-place batch refresh must not alter recycler accounting"
    );
    assert_eq!(
        kiln_vulkan_kernel::buffer::allocation_stats(),
        before_allocations,
        "in-place batch refresh must not allocate or free raw Vulkan buffers"
    );
    let bytes = VulkanBuffer::read_back(
        dev.device(),
        dev.host_visible_mem_type(),
        dev.queue(),
        dev.queue_family_index(),
        batch.as_ref(),
    )?;
    assert_eq!(
        &bytes[..16],
        bytemuck::cast_slice(&[51_u32, 52, 61_u32, 62])
    );

    let undersized_batch =
        VulkanBuffer::create_device_local(dev.device(), dev.device_local_mem_type(), 8)?;
    assert!(
        kernels::copy_device_buffer_rows_to_existing_batch(
            &dev,
            &source_rows,
            &undersized_batch,
            8,
        )
        .is_err(),
        "refresh must reject a logical payload larger than the retained capacity"
    );
    Ok(())
}

//! Test: fill buffer with vkCmdFillBuffer, read back. No shader involved.

use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::ptr::write_bytes;

fn main() -> Result<()> {
    let entry = unsafe { ash::Entry::load() }?;
    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"test\0").unwrap())
        .engine_name(CStr::from_bytes_with_nul(b"test\0").unwrap())
        .api_version(vk::make_api_version(0, 1, 2, 0))
        .build();
    let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info).build();
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    let phys_devices = unsafe { instance.enumerate_physical_devices()? };
    let phys = phys_devices[0];
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };

    let device_local_mt = (0..mem_props.memory_types.len())
        .find(|&i| mem_props.memory_types[i].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
        .context("no device local")?;
    let host_visible_mt = (0..mem_props.memory_types.len())
        .find(|&i| mem_props.memory_types[i].property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT))
        .context("no host visible")?;

    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(phys) };
    let qfi = queue_families.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)).context("no compute")? as u32;

    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(qfi).queue_priorities(&[1.0]).build();
    let device_info = vk::DeviceCreateInfo::builder().queue_create_infos(&[queue_info]).build();
    let device = unsafe { instance.create_device(phys, &device_info, None)? };
    let queue = unsafe { device.get_device_queue(qfi, 0) };

    // Create buffer with TRANSFER_SRC | TRANSFER_DST
    let buf_size: u64 = 16;
    let bci = vk::BufferCreateInfo::builder()
        .size(buf_size)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .build();
    let buf = unsafe { device.create_buffer(&bci, None)? };
    let reqs = unsafe { device.get_buffer_memory_requirements(buf) };
    let mai = vk::MemoryAllocateInfo::builder()
        .allocation_size(reqs.size).memory_type_index(device_local_mt as u32).build();
    let mem = unsafe { device.allocate_memory(&mai, None)? };
    unsafe { device.bind_buffer_memory(buf, mem, 0)? };

    // Staging buffer
    let sbci = vk::BufferCreateInfo::builder()
        .size(buf_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .build();
    let sbuf = unsafe { device.create_buffer(&sbci, None)? };
    let sreqs = unsafe { device.get_buffer_memory_requirements(sbuf) };
    let smai = vk::MemoryAllocateInfo::builder()
        .allocation_size(sreqs.size).memory_type_index(host_visible_mt as u32).build();
    let smem = unsafe { device.allocate_memory(&smai, None)? };
    unsafe { device.bind_buffer_memory(sbuf, smem, 0)? };

    // Command pool
    let mut pool_info: MaybeUninit<vk::CommandPoolCreateInfo> = MaybeUninit::uninit();
    unsafe { write_bytes(pool_info.as_mut_ptr(), 0, 1); }
    unsafe {
        let p = pool_info.as_mut_ptr();
        (*p).s_type = vk::StructureType::COMMAND_POOL_CREATE_INFO;
        (*p).queue_family_index = qfi;
        (*p).flags = vk::CommandPoolCreateFlags::empty();
    }
    let cmd_pool = unsafe { device.create_command_pool(&pool_info.assume_init(), None)? };

    // Allocate command buffer
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(cmd_pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1).build();
    let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
    let cmd = command_buffers[0];

    // Begin
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).build();
    unsafe { device.begin_command_buffer(cmd, &begin_info)? };

    // Fill buffer with 0x3F000000 (which is 1.0 as f32, but we fill byte-by-byte so each byte is 0x00, 0x00, 0x00, 0x3F repeating)
    // Actually vkCmdFillBuffer fills with 8-bit pattern repeated. Let's fill with 0x42 = 'B'
    unsafe {
        device.cmd_fill_buffer(cmd, buf, 0, buf_size, 0x42);

        // Copy to staging
        let copy = vk::BufferCopy::builder().size(buf_size).build();
        device.cmd_copy_buffer(cmd, buf, sbuf, &[copy]);

        device.end_command_buffer(cmd)?;
    }

    // Submit
    let submit_info = vk::SubmitInfo::builder().command_buffers(&[cmd]).build();
    unsafe {
        device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
    }

    // Map and read
    let ptr = unsafe { device.map_memory(smem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? };
    let data: Vec<u8> = unsafe { std::slice::from_raw_parts(ptr as *const u8, 16).to_vec() };
    println!("raw bytes = {:?}", data);
    println!("all 0x42 = {}", data.iter().all(|&b| b == 0x42));

    Ok(())
}

//! Test: use ash for instance/device, then test all allocate calls via raw fn to confirm pattern.

use anyhow::{Context, Result};
use ash::vk;
use std::mem::MaybeUninit;

fn main() -> Result<()> {
    let entry = unsafe { ash::Entry::load() }.map_err(|e| anyhow::anyhow!("entry load failed: {}", e))?;
    
    let app_info = vk::ApplicationInfo::builder()
        .application_name(std::ffi::CStr::from_bytes_with_nul(b"test\0").unwrap())
        .api_version(vk::make_api_version(0, 1, 2, 0))
        .build();
    
    let inst_info = vk::InstanceCreateInfo::builder()
        .application_info(&app_info)
        .build();
    
    let instance = unsafe {
        entry.create_instance(&inst_info, None)
            .context("create_instance failed")?
    };
    println!("instance created");
    
    let phys_devices = unsafe {
        instance.enumerate_physical_devices()
            .context("enumerate physical devices failed")?
    };
    println!("physical devices: {}", phys_devices.len());
    
    let q_priority = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(0)
        .queue_priorities(&q_priority)
        .build();
    
    let dev_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&[queue_info])
        .build();
    
    let device = unsafe {
        instance.create_device(phys_devices[0], &dev_info, None)
            .context("create_device failed")?
    };
    println!("device created");
    
    // Test 1: allocate_command_buffers via ash wrapper (should crash)
    println!("\n--- Test 1: ash wrapper ---");
    
    // Create pool via ash
    let pool_info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(0)
        .flags(vk::CommandPoolCreateFlags::empty())
        .build();
    
    let pool = unsafe {
        device.create_command_pool(&pool_info, None)
            .context("create_command_pool failed")?
    };
    println!("pool created via ash wrapper");
    
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1)
        .build();
    
    println!("calling ash allocate_command_buffers...");
    let cmd_buffers = unsafe {
        device.allocate_command_buffers(&alloc_info)
            .context("allocate_command_buffers failed")?
    };
    println!("allocate_command_buffers succeeded via ash wrapper! got {} buffers", cmd_buffers.len());
    
    // Cleanup
    unsafe {
        device.free_command_buffers(pool, &cmd_buffers);
        device.destroy_command_pool(pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
    
    println!("ALL TESTS PASSED!");
    Ok(())
}

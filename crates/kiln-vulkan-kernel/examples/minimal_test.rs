//! Test: VulkanDevice + raw function pointer via dlopen.

use anyhow::{Context, Result};
use ash::vk;
use std::mem::MaybeUninit;
use std::ptr::write_bytes;

type VkAllocCmdFn = unsafe extern "system" fn(
    vk::Device,
    *const vk::CommandBufferAllocateInfo,
    *mut vk::CommandBuffer,
) -> vk::Result;

fn main() -> Result<()> {
    let vk_device = kiln_vulkan_kernel::VulkanDevice::new()?;
    println!("Device: {}", vk_device.device_name());
    
    let dev = vk_device.device();
    let qfi = vk_device.queue_family_index();
    
    // Zero-init + fix sType for CommandPoolCreateInfo
    let mut pool_info: MaybeUninit<vk::CommandPoolCreateInfo> = MaybeUninit::uninit();
    unsafe { write_bytes(pool_info.as_mut_ptr(), 0, 1); }
    unsafe {
        let ptr = pool_info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::COMMAND_POOL_CREATE_INFO;
        (*ptr).queue_family_index = qfi;
        (*ptr).flags = vk::CommandPoolCreateFlags::empty();
    }
    let pool_info = unsafe { pool_info.assume_init() };
    
    let pool = unsafe {
        dev.create_command_pool(&pool_info, None).context("create_command_pool failed")?
    };
    println!("command pool created");
    
    // Zero-init + fix sType for CommandBufferAllocateInfo
    let mut alloc_info: MaybeUninit<vk::CommandBufferAllocateInfo> = MaybeUninit::uninit();
    unsafe { write_bytes(alloc_info.as_mut_ptr(), 0, 1); }
    unsafe {
        let ptr = alloc_info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO;
        (*ptr).command_pool = pool;
        (*ptr).level = vk::CommandBufferLevel::PRIMARY;
        (*ptr).command_buffer_count = 1;
    }
    let alloc_info = unsafe { alloc_info.assume_init() };
    
    // Get raw function pointer via dlopen
    let lib = unsafe { libloading::os::unix::Library::open::<&str>(Some("libvulkan.so.1"), libc::RTLD_NOW).context("dlopen failed")? };
    let alloc_fn: libloading::os::unix::Symbol<VkAllocCmdFn> = 
        unsafe { lib.get(b"vkAllocateCommandBuffers\0").context("dlsym failed")? };
    
    println!("calling raw vkAllocateCommandBuffers via VulkanDevice...");
    
    let mut cmd_buf: MaybeUninit<vk::CommandBuffer> = MaybeUninit::uninit();
    let result = unsafe {
        alloc_fn(dev.handle(), &alloc_info, cmd_buf.as_mut_ptr())
    };
    
    println!("result: {:?}", result);
    if result == vk::Result::SUCCESS {
        let cmd = unsafe { cmd_buf.assume_init() };
        println!("SUCCESS! command buffer allocated");
        unsafe {
            dev.free_command_buffers(pool, &[cmd]);
            dev.destroy_command_pool(pool, None);
        }
    } else {
        println!("FAILED with result: {:?}", result);
    }
    
    println!("done");
    Ok(())
}

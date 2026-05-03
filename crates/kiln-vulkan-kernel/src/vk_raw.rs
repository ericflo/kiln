//! Raw Vulkan function calls to work around ash 0.37 wrapper bugs.
//!
//! ash 0.37's `allocate_command_buffers` wrapper corrupts struct fields
//! before passing them to the Vulkan driver. We bypass it by calling the
//! raw function pointer directly.
//!
//! For all other operations (create/destroy command pool, free command buffers),
//! we use ash's own wrappers which work correctly.

use ash::vk;
use std::sync::OnceLock;

type VkAllocCmdFn = unsafe extern "system" fn(
    vk::Device,
    *const vk::CommandBufferAllocateInfo,
    *mut vk::CommandBuffer,
) -> vk::Result;

static ALLOC_CMD_FN: OnceLock<VkAllocCmdFn> = OnceLock::new();
static _VULKAN_LIB: OnceLock<libloading::os::unix::Library> = OnceLock::new();

fn load_vk_funcs() {
    _VULKAN_LIB.set(unsafe {
        libloading::os::unix::Library::open::<&str>(
            Some("libvulkan.so.1"),
            libc::RTLD_NOW,
        ).expect("failed to open libvulkan.so.1")
    }).ok();
    
    unsafe {
        let lib = _VULKAN_LIB.get().expect("library not loaded");
        let alloc_fn: libloading::os::unix::Symbol<VkAllocCmdFn> =
            lib.get(b"vkAllocateCommandBuffers\0").expect("vkAllocateCommandBuffers not found");
        ALLOC_CMD_FN.set(*alloc_fn).ok();
    }
}

/// Allocate command buffers via raw Vulkan function pointer (bypasses ash 0.37 wrapper bug).
///
/// This is the ONLY operation that needs raw calls. ash 0.37's wrapper for
/// `allocate_command_buffers` corrupts the `CommandBufferAllocateInfo` struct
/// before passing it to the driver, causing segfaults.
pub fn allocate_command_buffers(
    device: vk::Device,
    alloc_info: &vk::CommandBufferAllocateInfo,
    count: u32,
) -> Result<Vec<vk::CommandBuffer>, vk::Result> {
    if ALLOC_CMD_FN.get().is_none() {
        load_vk_funcs();
    }
    let fn_ptr = ALLOC_CMD_FN.get().expect("failed to load vkAllocateCommandBuffers");
    
    let mut cmd_bufs = vec![vk::CommandBuffer::null(); count as usize];
    let result = unsafe {
        fn_ptr(device, alloc_info, cmd_bufs.as_mut_ptr())
    };
    
    if result == vk::Result::SUCCESS {
        Ok(cmd_bufs)
    } else {
        Err(result)
    }
}

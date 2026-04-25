//! Check struct sizes.

use ash::vk;

fn main() {
    println!("ash version: {}", env!("CARGO_PKG_VERSION"));
    println!("CommandBufferAllocateInfo size: {} bytes", std::mem::size_of::<vk::CommandBufferAllocateInfo>());
    println!("CommandBufferAllocateInfo align: {} bytes", std::mem::align_of::<vk::CommandBufferAllocateInfo>());
    println!("CommandPool size: {} bytes", std::mem::size_of::<vk::CommandPool>());
    println!("CommandPool align: {} bytes", std::mem::align_of::<vk::CommandPool>());
    println!("CommandBuffer size: {} bytes", std::mem::size_of::<vk::CommandBuffer>());
    println!("CommandBuffer align: {} bytes", std::mem::align_of::<vk::CommandBuffer>());
    println!("StructureType size: {} bytes", std::mem::size_of::<vk::StructureType>());
    println!("CommandBufferLevel size: {} bytes", std::mem::size_of::<vk::CommandBufferLevel>());
    println!("ptr size: {} bytes", std::mem::size_of::<*const std::ffi::c_void>());
}

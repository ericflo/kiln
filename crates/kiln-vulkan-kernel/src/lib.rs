//! Vulkan compute kernels for Kiln.
//!
//! Provides Vulkan device management, buffer allocation, and kernel dispatch
//! functions for FlashAttention-2, Gated DeltaNet, and supporting operations.
//!
//! candle-core has no native Vulkan device, so this crate manages its own
//! Vulkan device and copies tensor data through the CPU path at kernel boundaries.

pub mod buffer;
pub mod device;
pub mod kernels;
pub mod pipeline;
pub mod vk_raw;

pub use buffer::VulkanBuffer;
pub use device::VulkanDevice;
pub use pipeline::ShaderPipeline;

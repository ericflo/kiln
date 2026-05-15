//! Vulkan compute kernels for Kiln.
//!
//! Provides Vulkan device management, buffer allocation, and kernel dispatch
//! functions for FlashAttention-2, Gated DeltaNet, and supporting operations.
//!
//! candle-core has no native Vulkan device, so this crate manages its own
//! Vulkan device and copies tensor data through the CPU path at kernel boundaries.
//!
//! The `vk_tensor` and `vk_autograd` modules host the vk-native training
//! stack: a GPU-resident `VkTensor` type and its own eager autograd tape,
//! used in training to keep every forward intermediate and gradient on the
//! device instead of materializing them as candle CPU storage.

pub mod buffer;
pub mod buffer_pool;
pub mod decode_resident_pool;
pub mod device;
pub mod kernels;
pub mod pipeline;
pub mod vk_autograd;
pub mod vk_ops;
pub mod vk_raw;
pub mod vk_tensor;

pub use buffer::VulkanBuffer;
pub use decode_resident_pool::DecodeResidentPool;
pub use device::VulkanDevice;
pub use pipeline::ShaderPipeline;
pub use vk_autograd::{VkGradStore, vk_backward};
pub use vk_tensor::{VkBackwardOp, VkDType, VkTensor, VkTensorInner, next_op_id};

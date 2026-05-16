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
pub mod cmd_batch;
pub mod decode_resident_pool;
pub mod device;
pub mod kernels;
pub mod pipeline;
pub mod resident;
pub mod vk_autograd;
pub mod vk_ops;
pub mod vk_paged_kv_cache;
pub mod vk_raw;
pub mod vk_tensor;

pub use buffer::VulkanBuffer;
pub use cmd_batch::{CommandBatch, KernelPlan, Workgroups};
pub use decode_resident_pool::DecodeResidentPool;
pub use device::VulkanDevice;
pub use pipeline::ShaderPipeline;
pub use vk_paged_kv_cache::VkPagedKvCache;

/// Public shader source paths for callers (in this crate or downstream)
/// that want to record dispatches into a [`CommandBatch`] directly
/// instead of going through the per-dispatch `dispatch_*_resident`
/// wrappers. Each constant is the absolute path under this crate's
/// `csrc/shaders/` directory, baked in at compile time.
pub mod shaders {
    macro_rules! shader_path {
        ($name:literal) => {
            concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/", $name, ".comp")
        };
    }
    pub const QWEN_RMSNORM_FORWARD: &str = shader_path!("qwen_rmsnorm_forward");
    pub const QWEN_RMSNORM_QK_COMBINED: &str = shader_path!("qwen_rmsnorm_qk_combined");
    pub const FULL_ATTN_QKV_DECODE_BF16W: &str = shader_path!("full_attn_qkv_decode_bf16w");
    pub const FULL_ATTN_QKV_DECODE_BF16W_WIDE: &str =
        shader_path!("full_attn_qkv_decode_bf16w_wide");
    pub const QKV_GATE_SPLIT: &str = shader_path!("qkv_gate_split");
    pub const VK_ROPE_F32: &str = shader_path!("vk_rope_f32");
    pub const PAGED_KV_WRITE_SLOT: &str = shader_path!("paged_kv_write_slot");
    pub const PAGED_ATTN_DECODE_BATCH_PAGED: &str = shader_path!("paged_attn_decode_batch_paged");
    pub const PAGED_ATTN_DECODE_BATCH_PAGED_SPLITK: &str =
        shader_path!("paged_attn_decode_batch_paged_splitk");
    pub const PAGED_ATTN_DECODE_BATCH_PAGED_SPLITK_REDUCE: &str =
        shader_path!("paged_attn_decode_batch_paged_splitk_reduce");
    pub const VK_MUL_SIGMOID_GATE_F32: &str = shader_path!("vk_mul_sigmoid_gate_f32");
    pub const LINEAR_DECODE_BF16W: &str = shader_path!("linear_decode_bf16w");
    pub const LINEAR_DECODE_BF16W_WIDE: &str = shader_path!("linear_decode_bf16w_wide");
    pub const ADD: &str = shader_path!("add");
    pub const ADD_QWEN_RMSNORM: &str = shader_path!("add_qwen_rmsnorm");
    pub const MLP_GATE_UP_DECODE_BF16W: &str = shader_path!("mlp_gate_up_decode_bf16w");
    pub const GDN_IN_PROJ_SPLIT: &str = shader_path!("gdn_in_proj_split");
    pub const GDN_QKV_SPLIT: &str = shader_path!("gdn_qkv_split");
    pub const GDN_IN_PROJ_DECODE_BF16W: &str = shader_path!("gdn_in_proj_decode_bf16w");
    pub const GDN_DECODE_GATES_RECURRENT_RMSNORM: &str =
        shader_path!("gdn_decode_gates_recurrent_rmsnorm");
    pub const CAUSAL_CONV1D: &str = shader_path!("causal_conv1d");
    pub const CAUSAL_CONV1D_STATE_ADVANCE: &str = shader_path!("causal_conv1d_state_advance");
}
pub use vk_autograd::{VkGradStore, vk_backward};
pub use vk_tensor::{VkBackwardOp, VkDType, VkTensor, VkTensorInner, next_op_id};

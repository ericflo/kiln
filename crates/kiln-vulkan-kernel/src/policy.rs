use anyhow::{Result, ensure};
use std::sync::OnceLock;

/// Vulkan compute and memory limits that can affect whether a shader route is
/// legal on the selected physical device.
///
/// Deliberately absent: vendor IDs, device IDs, driver names, and marketing
/// names. Those are useful receipt metadata, but they are not capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanComputeCapabilities {
    pub api_version: u32,
    pub max_compute_work_group_count: [u32; 3],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size: [u32; 3],
    pub max_compute_shared_memory_size: u32,
    pub max_push_constants_size: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_storage_buffer_range: u64,
    pub supports_compute_subgroup_basic_arithmetic: bool,
    pub has_coherent_device_local_host_visible_memory: bool,
    pub host_visible_staging_is_cached: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanShaderRequirements {
    pub local_size: [u32; 3],
    pub shared_memory_bytes: u32,
    pub storage_buffer_bindings: u32,
    pub push_constant_bytes: u32,
}

impl VulkanComputeCapabilities {
    /// Whether a fixed-workgroup compute shader fits the selected device's
    /// advertised core limits. Workload-dependent dispatch-grid and storage
    /// range checks remain at the individual dispatch site.
    pub fn supports_shader(self, requirements: VulkanShaderRequirements) -> bool {
        let [x, y, z] = requirements.local_size;
        let invocations = u64::from(x) * u64::from(y) * u64::from(z);

        x > 0
            && y > 0
            && z > 0
            && x <= self.max_compute_work_group_size[0]
            && y <= self.max_compute_work_group_size[1]
            && z <= self.max_compute_work_group_size[2]
            && invocations <= u64::from(self.max_compute_work_group_invocations)
            && requirements.shared_memory_bytes <= self.max_compute_shared_memory_size
            && requirements.push_constant_bytes <= self.max_push_constants_size
            && requirements.storage_buffer_bindings <= self.max_per_stage_descriptor_storage_buffers
            && requirements.storage_buffer_bindings <= self.max_descriptor_set_storage_buffers
    }

    pub fn supports_full_pipeline_prewarm(self) -> bool {
        self.supports_shader(VulkanShaderRequirements {
            local_size: [256, 1, 1],
            shared_memory_bytes: 32 * 1024,
            storage_buffer_bindings: 13,
            push_constant_bytes: 40,
        }) && self.supports_shader(VulkanShaderRequirements {
            local_size: [80, 3, 1],
            shared_memory_bytes: 15 * 1024,
            storage_buffer_bindings: 6,
            push_constant_bytes: 28,
        }) && self.supports_shader(VulkanShaderRequirements {
            local_size: [16, 16, 1],
            shared_memory_bytes: 8 * 1024,
            storage_buffer_bindings: 8,
            push_constant_bytes: 28,
        })
    }
}

/// Immutable kernel selection derived from the selected Vulkan device's
/// advertised capabilities.
///
/// Product selection must never depend on device names, vendor IDs, or
/// qualification-machine identities. Workload crossovers are shape-based;
/// route availability is limit-based.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanKernelPolicy {
    pub flash_attention_row_tile: usize,
    pub flash_attention_row_work_budget: usize,
    pub bf16_weight_row_tile: usize,
    pub elementwise_tile_elements: usize,
    pub exp_tile_elements: usize,
    pub gdn_enabled: bool,
    pub gdn_prefill_in_proj_enabled: bool,
    pub gdn_gates_enabled: bool,
    pub gdn_gated_rms_norm_enabled: bool,
    pub gdn_full_chunk_forward_enabled: bool,
    pub fused_conv1d_update_enabled: bool,
    pub fused_conv1d_prefill_enabled: bool,
    pub conv1d_prefill_single_submit_enabled: bool,
    pub gdn_forward_sub_enabled: bool,
    pub gdn_decode_fused_enabled: bool,
    pub gdn_recurrent_unexpanded_qk_enabled: bool,
    pub gdn_recurrent_qk_norm_unexpanded_enabled: bool,
    pub linear_decode_enabled: bool,
    pub linear_argmax_batch_enabled: bool,
    pub full_attn_qkv_enabled: bool,
    pub paged_attn_decode_batch_enabled: bool,
    pub mlp_decode_enabled: bool,
    pub mlp_gate_up_enabled: bool,
    pub mlp_bf16_gate_up_f32_down_enabled: bool,
    pub bf16_packed_linear_weights_enabled: bool,
    pub bf16_packed_gdn_in_proj_weights_enabled: bool,
    pub bf16_packed_full_attn_qkv_weights_enabled: bool,
    pub bf16_packed_mlp_decode_weights_enabled: bool,
    pub recurrent_state_residency_enabled: bool,
    pub prefill_recurrent_state_residency_enabled: bool,
    pub resident_decode_enabled: bool,
    pub bridged_rmsnorm_forward_enabled: bool,
    pub skip_final_gdn_state_readback_enabled: bool,
    pub flash_attn_prefill_enabled: bool,
    pub paged_decode_gpu_gather_enabled: bool,
    pub gdn_chunkwise_forward_enabled: bool,
    pub gdn_chunkwise_single_submit_enabled: bool,
    pub gdn_chunkwise_fallback_enabled: bool,
    pub gdn_decode_fused_resident_state_enabled: bool,
    pub linear_max_flop_per_dispatch: u64,
    pub mlp_bf16_gate_up_rows4: bool,
    pub mlp_f32_down_rows4: bool,
    pub mlp_bf16_down_rows4: bool,
    pub mlp_bf16_rows8: bool,
    pub mlp_bf16_rows8_min_batch: usize,
    pub mlp_bf16_gate_up_rows4_min_batch: usize,
    pub mlp_bf16_down_rows4_min_batch: usize,
    pub mlp_f32_down_rows4_min_batch: usize,
    pub linear_decode_bf16w_rows4: bool,
    pub linear_decode_bf16w_rows8: bool,
    pub linear_bf16_rows4_min_batch: usize,
    pub linear_bf16_rows8_min_batch: usize,
    pub gdn_in_proj_rows4_min_batch: usize,
    pub gdn_in_proj_rows8_min_batch: usize,
    pub full_attn_qkv_bf16w_rows4: bool,
    pub full_attn_qkv_bf16w_rows8: bool,
    pub full_attn_qkv_bf16_rows4_min_batch: usize,
    pub full_attn_qkv_bf16_rows8_min_batch: usize,
    pub paged_attn_single_submit: bool,
    pub qwen_rmsnorm_single_submit: bool,
    pub gdn_gates_single_submit: bool,
    pub gdn_gated_norm_single_submit: bool,
    pub mlp_gate_up_single_submit: bool,
    pub causal_conv1d_single_submit: bool,
    pub mlp_chained_dispatch: bool,
    pub mlp_chained_transfer_submit: bool,
    pub gdn_decode_host_visible_state: bool,
    pub gdn_decode_fused_single_submit: bool,
    pub gdn_recurrent_host_visible_state: bool,
    pub gdn_recurrent_host_visible_batch_state: bool,
    pub gdn_recurrent_single_submit: bool,
    pub gdn_recurrent_parallel_reduce: bool,
    pub linear_decode_single_submit: bool,
    pub linear_decode_argmax_single_submit: bool,
    pub full_attn_qkv_single_submit: bool,
    pub gdn_in_proj_single_submit: bool,
    pub gdn_in_proj_batch_pair_qkv_z: bool,
    pub gdn_in_proj_batch_row_pair: bool,
    pub gdn_in_proj_batch_row_quad: bool,
    pub gdn_in_proj_batch_row_octet: bool,
    pub gdn_gates_batched_transfers: bool,
    pub gdn_gated_norm_batched_uploads: bool,
    pub gdn_chunk_batched_transfers: bool,
    pub paged_attn_batched_uploads: bool,
    pub prefill_row_pair_matmul: bool,
    pub gdn_qk_norm_recurrent_fusion: bool,
    pub gdn_in_proj_conv_split_fusion: bool,
    pub profile_mlp_kernel_stages: bool,
    pub profile_gdn_in_proj_kernel_stages: bool,
    pub profile_gdn_recurrent_kernel_stages: bool,
    pub profile_resident_decode_timing: bool,
}

pub const VULKAN_KERNEL_POLICY_SCHEMA_ID: &str = "kiln.vulkan-kernel-policy.v6";

impl VulkanKernelPolicy {
    /// Select every compatible production route using Vulkan limits and memory
    /// properties. This function is pure so synthetic devices and captured
    /// `vulkaninfo` limits can be tested without a GPU.
    pub fn from_capabilities(capabilities: VulkanComputeCapabilities) -> Self {
        let scalar_256 = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [256, 1, 1],
            shared_memory_bytes: 7 * 1024,
            storage_buffer_bindings: 11,
            push_constant_bytes: 40,
        });
        let matrix_256 = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [16, 16, 1],
            shared_memory_bytes: 2 * 1024,
            storage_buffer_bindings: 8,
            push_constant_bytes: 28,
        });
        let gdn_pair = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [80, 3, 1],
            shared_memory_bytes: 2 * 1024,
            storage_buffer_bindings: 6,
            push_constant_bytes: 28,
        });
        let flash_prefill = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [256, 1, 1],
            shared_memory_bytes: 1024,
            storage_buffer_bindings: 5,
            push_constant_bytes: 20,
        });
        let linear_rows4 = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [32, 4, 1],
            shared_memory_bytes: 2 * 1024,
            storage_buffer_bindings: 4,
            push_constant_bytes: 16,
        });
        let linear_rows8 = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [32, 4, 1],
            shared_memory_bytes: 4 * 1024,
            storage_buffer_bindings: 4,
            push_constant_bytes: 16,
        });
        let mlp_rows4 = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [64, 2, 1],
            shared_memory_bytes: 4 * 1024,
            storage_buffer_bindings: 4,
            push_constant_bytes: 12,
        });
        let mlp_rows8 = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [64, 2, 1],
            shared_memory_bytes: 8 * 1024,
            storage_buffer_bindings: 4,
            push_constant_bytes: 12,
        });
        let gdn_rows4 = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [80, 3, 1],
            shared_memory_bytes: 8 * 1024,
            storage_buffer_bindings: 6,
            push_constant_bytes: 28,
        });
        let full_attn_rows4 = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [16, 16, 1],
            shared_memory_bytes: 4 * 1024,
            storage_buffer_bindings: 8,
            push_constant_bytes: 28,
        });
        let full_attn_rows8 = capabilities.supports_shader(VulkanShaderRequirements {
            local_size: [16, 16, 1],
            shared_memory_bytes: 8 * 1024,
            storage_buffer_bindings: 8,
            push_constant_bytes: 28,
        });
        let chunkwise = scalar_256
            && capabilities.supports_shader(VulkanShaderRequirements {
                local_size: [64, 1, 1],
                shared_memory_bytes: 32 * 1024,
                storage_buffer_bindings: 4,
                push_constant_bytes: 16,
            });
        let resident_decode = scalar_256 && matrix_256 && gdn_pair;
        let unified_coherent_state = capabilities.has_coherent_device_local_host_visible_memory;

        let mut policy = Self::portable_fallback();
        policy.gdn_enabled = scalar_256 && matrix_256;
        policy.gdn_prefill_in_proj_enabled = policy.gdn_enabled;
        policy.gdn_gates_enabled = scalar_256;
        policy.gdn_gated_rms_norm_enabled = scalar_256;
        policy.fused_conv1d_prefill_enabled = scalar_256;
        policy.conv1d_prefill_single_submit_enabled = scalar_256;
        policy.gdn_recurrent_unexpanded_qk_enabled = scalar_256;
        policy.gdn_recurrent_qk_norm_unexpanded_enabled = scalar_256;
        policy.linear_decode_enabled = matrix_256;
        policy.linear_argmax_batch_enabled = scalar_256 && matrix_256;
        policy.full_attn_qkv_enabled = matrix_256;
        policy.paged_attn_decode_batch_enabled = scalar_256;
        policy.mlp_decode_enabled = matrix_256;
        policy.mlp_bf16_gate_up_f32_down_enabled = matrix_256;
        policy.bf16_packed_linear_weights_enabled = matrix_256;
        policy.bf16_packed_gdn_in_proj_weights_enabled = scalar_256 && matrix_256;
        policy.bf16_packed_full_attn_qkv_weights_enabled = matrix_256;
        policy.bf16_packed_mlp_decode_weights_enabled = matrix_256;
        policy.resident_decode_enabled = resident_decode;
        policy.skip_final_gdn_state_readback_enabled = resident_decode;
        policy.flash_attn_prefill_enabled = flash_prefill;
        policy.paged_decode_gpu_gather_enabled = scalar_256;
        policy.gdn_chunkwise_forward_enabled = chunkwise;
        policy.gdn_chunkwise_single_submit_enabled = chunkwise;
        policy.gdn_chunkwise_fallback_enabled = !chunkwise;
        policy.gdn_decode_fused_resident_state_enabled = resident_decode;
        policy.mlp_bf16_gate_up_rows4 = mlp_rows4;
        policy.mlp_f32_down_rows4 = mlp_rows4;
        policy.mlp_bf16_down_rows4 = mlp_rows4;
        policy.mlp_bf16_rows8 = mlp_rows8;
        policy.linear_decode_bf16w_rows4 = linear_rows4;
        policy.linear_decode_bf16w_rows8 = linear_rows8;
        policy.full_attn_qkv_bf16w_rows4 = full_attn_rows4;
        policy.full_attn_qkv_bf16w_rows8 = full_attn_rows8;
        policy.paged_attn_single_submit = scalar_256;
        policy.qwen_rmsnorm_single_submit = scalar_256;
        policy.gdn_gates_single_submit = scalar_256;
        policy.gdn_gated_norm_single_submit = scalar_256;
        policy.mlp_gate_up_single_submit = matrix_256;
        policy.causal_conv1d_single_submit = scalar_256;
        policy.mlp_chained_dispatch = matrix_256;
        policy.mlp_chained_transfer_submit = matrix_256;
        policy.gdn_recurrent_host_visible_state = unified_coherent_state;
        policy.gdn_recurrent_single_submit = scalar_256;
        policy.gdn_recurrent_parallel_reduce =
            capabilities.supports_shader(VulkanShaderRequirements {
                local_size: [32, 1, 1],
                shared_memory_bytes: 128,
                storage_buffer_bindings: 7,
                push_constant_bytes: 24,
            });
        policy.linear_decode_single_submit = matrix_256;
        policy.linear_decode_argmax_single_submit = scalar_256 && matrix_256;
        policy.full_attn_qkv_single_submit = matrix_256;
        policy.gdn_in_proj_single_submit = scalar_256 && matrix_256;
        policy.gdn_in_proj_batch_pair_qkv_z = gdn_pair;
        policy.gdn_in_proj_batch_row_pair = gdn_pair;
        policy.gdn_in_proj_batch_row_quad = gdn_rows4;
        policy.gdn_gates_batched_transfers = scalar_256;
        policy.gdn_gated_norm_batched_uploads = scalar_256;
        policy.gdn_chunk_batched_transfers = chunkwise;
        policy.paged_attn_batched_uploads = scalar_256;
        policy.prefill_row_pair_matmul = matrix_256;
        policy.gdn_qk_norm_recurrent_fusion = scalar_256;
        policy
    }

    /// Decline optional optimized Vulkan routes while retaining bounded-work
    /// limits and explicit reference fallbacks.
    pub const fn portable_fallback() -> Self {
        Self {
            flash_attention_row_tile: 2048,
            flash_attention_row_work_budget: 10_000_000,
            bf16_weight_row_tile: 128,
            elementwise_tile_elements: 1 << 20,
            exp_tile_elements: 65_536,
            gdn_enabled: false,
            gdn_prefill_in_proj_enabled: false,
            gdn_gates_enabled: false,
            gdn_gated_rms_norm_enabled: false,
            gdn_full_chunk_forward_enabled: false,
            fused_conv1d_update_enabled: false,
            fused_conv1d_prefill_enabled: false,
            conv1d_prefill_single_submit_enabled: false,
            gdn_forward_sub_enabled: false,
            gdn_decode_fused_enabled: false,
            gdn_recurrent_unexpanded_qk_enabled: false,
            gdn_recurrent_qk_norm_unexpanded_enabled: false,
            linear_decode_enabled: false,
            linear_argmax_batch_enabled: false,
            full_attn_qkv_enabled: false,
            paged_attn_decode_batch_enabled: false,
            mlp_decode_enabled: false,
            mlp_gate_up_enabled: false,
            mlp_bf16_gate_up_f32_down_enabled: false,
            bf16_packed_linear_weights_enabled: false,
            bf16_packed_gdn_in_proj_weights_enabled: false,
            bf16_packed_full_attn_qkv_weights_enabled: false,
            bf16_packed_mlp_decode_weights_enabled: false,
            recurrent_state_residency_enabled: false,
            prefill_recurrent_state_residency_enabled: false,
            resident_decode_enabled: false,
            bridged_rmsnorm_forward_enabled: false,
            skip_final_gdn_state_readback_enabled: false,
            flash_attn_prefill_enabled: false,
            paged_decode_gpu_gather_enabled: false,
            gdn_chunkwise_forward_enabled: false,
            gdn_chunkwise_single_submit_enabled: false,
            gdn_chunkwise_fallback_enabled: true,
            gdn_decode_fused_resident_state_enabled: false,
            linear_max_flop_per_dispatch: 20_000_000_000,
            mlp_bf16_gate_up_rows4: false,
            mlp_f32_down_rows4: false,
            mlp_bf16_down_rows4: false,
            mlp_bf16_rows8: false,
            mlp_bf16_rows8_min_batch: 256,
            mlp_bf16_gate_up_rows4_min_batch: 8,
            mlp_bf16_down_rows4_min_batch: 16,
            mlp_f32_down_rows4_min_batch: 8,
            linear_decode_bf16w_rows4: false,
            linear_decode_bf16w_rows8: false,
            linear_bf16_rows4_min_batch: 16,
            linear_bf16_rows8_min_batch: 64,
            gdn_in_proj_rows4_min_batch: 16,
            gdn_in_proj_rows8_min_batch: 64,
            full_attn_qkv_bf16w_rows4: false,
            full_attn_qkv_bf16w_rows8: false,
            full_attn_qkv_bf16_rows4_min_batch: 2,
            full_attn_qkv_bf16_rows8_min_batch: 64,
            paged_attn_single_submit: false,
            qwen_rmsnorm_single_submit: false,
            gdn_gates_single_submit: false,
            gdn_gated_norm_single_submit: false,
            mlp_gate_up_single_submit: false,
            causal_conv1d_single_submit: false,
            mlp_chained_dispatch: false,
            mlp_chained_transfer_submit: false,
            gdn_decode_host_visible_state: false,
            gdn_decode_fused_single_submit: false,
            gdn_recurrent_host_visible_state: false,
            gdn_recurrent_host_visible_batch_state: false,
            gdn_recurrent_single_submit: false,
            gdn_recurrent_parallel_reduce: false,
            linear_decode_single_submit: false,
            linear_decode_argmax_single_submit: false,
            full_attn_qkv_single_submit: false,
            gdn_in_proj_single_submit: false,
            gdn_in_proj_batch_pair_qkv_z: false,
            gdn_in_proj_batch_row_pair: false,
            gdn_in_proj_batch_row_quad: false,
            gdn_in_proj_batch_row_octet: false,
            gdn_gates_batched_transfers: false,
            gdn_gated_norm_batched_uploads: false,
            gdn_chunk_batched_transfers: false,
            paged_attn_batched_uploads: false,
            prefill_row_pair_matmul: false,
            gdn_qk_norm_recurrent_fusion: false,
            gdn_in_proj_conv_split_fusion: false,
            profile_mlp_kernel_stages: false,
            profile_gdn_in_proj_kernel_stages: false,
            profile_gdn_recurrent_kernel_stages: false,
            profile_resident_decode_timing: false,
        }
    }
}

impl Default for VulkanKernelPolicy {
    fn default() -> Self {
        Self::portable_fallback()
    }
}

pub const PORTABLE_VULKAN_KERNEL_POLICY: VulkanKernelPolicy =
    VulkanKernelPolicy::portable_fallback();

static SELECTED_VULKAN_KERNEL_POLICY: OnceLock<VulkanKernelPolicy> = OnceLock::new();

pub fn install_vulkan_kernel_policy(policy: VulkanKernelPolicy) -> Result<()> {
    if let Some(installed) = SELECTED_VULKAN_KERNEL_POLICY.get() {
        ensure!(
            *installed == policy,
            "Vulkan kernel policy is already installed as {installed:?}; refusing conflicting selected-device capabilities {policy:?}"
        );
        return Ok(());
    }
    match SELECTED_VULKAN_KERNEL_POLICY.set(policy) {
        Ok(()) => Ok(()),
        Err(policy) => install_vulkan_kernel_policy(policy),
    }
}

pub fn vulkan_kernel_policy() -> VulkanKernelPolicy {
    SELECTED_VULKAN_KERNEL_POLICY
        .get()
        .copied()
        .unwrap_or(PORTABLE_VULKAN_KERNEL_POLICY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capabilities() -> VulkanComputeCapabilities {
        VulkanComputeCapabilities {
            api_version: ash::vk::make_api_version(0, 1, 2, 0),
            max_compute_work_group_count: [65_535; 3],
            max_compute_work_group_invocations: 1024,
            max_compute_work_group_size: [1024, 1024, 64],
            max_compute_shared_memory_size: 64 * 1024,
            max_push_constants_size: 256,
            max_per_stage_descriptor_storage_buffers: 32,
            max_descriptor_set_storage_buffers: 32,
            max_storage_buffer_range: u32::MAX as u64,
            supports_compute_subgroup_basic_arithmetic: true,
            has_coherent_device_local_host_visible_memory: false,
            host_visible_staging_is_cached: true,
        }
    }

    #[test]
    fn workgroup_support_uses_limits_not_identity() {
        let mut caps = capabilities();
        let shader = VulkanShaderRequirements {
            local_size: [80, 3, 1],
            shared_memory_bytes: 15 * 1024,
            storage_buffer_bindings: 6,
            push_constant_bytes: 28,
        };
        assert!(caps.supports_shader(shader));

        caps.max_compute_work_group_invocations = 128;
        assert!(!caps.supports_shader(shader));
        caps.max_compute_work_group_invocations = 1024;
        caps.max_compute_shared_memory_size = 8 * 1024;
        assert!(!caps.supports_shader(shader));
        caps.max_compute_shared_memory_size = 64 * 1024;
        caps.max_per_stage_descriptor_storage_buffers = 4;
        assert!(!caps.supports_shader(shader));
    }

    #[test]
    fn standard_minimum_device_keeps_compatible_routes_only() {
        let mut caps = capabilities();
        caps.api_version = ash::vk::make_api_version(0, 1, 0, 0);
        caps.supports_compute_subgroup_basic_arithmetic = false;
        caps.max_compute_work_group_invocations = 128;
        caps.max_compute_work_group_size = [128, 128, 64];
        caps.max_compute_shared_memory_size = 16 * 1024;
        caps.max_per_stage_descriptor_storage_buffers = 4;
        caps.max_descriptor_set_storage_buffers = 4;

        let policy = VulkanKernelPolicy::from_capabilities(caps);
        assert!(!policy.flash_attn_prefill_enabled);
        assert!(policy.linear_decode_bf16w_rows4);
        assert!(policy.linear_decode_bf16w_rows8);
        assert!(policy.mlp_bf16_gate_up_rows4);
        assert!(policy.mlp_bf16_rows8);
        assert!(!policy.resident_decode_enabled);
        assert!(!policy.gdn_enabled);
        assert!(!policy.gdn_chunkwise_forward_enabled);
        assert!(policy.gdn_chunkwise_fallback_enabled);
    }

    #[test]
    fn shared_memory_limit_gates_chunkwise_without_disabling_decode() {
        let mut caps = capabilities();
        caps.max_compute_shared_memory_size = 16 * 1024;

        let policy = VulkanKernelPolicy::from_capabilities(caps);
        assert!(policy.resident_decode_enabled);
        assert!(!policy.gdn_chunkwise_forward_enabled);
        assert!(policy.gdn_chunkwise_fallback_enabled);
        assert!(!caps.supports_full_pipeline_prewarm());
    }

    #[test]
    fn memory_topology_controls_host_visible_recurrent_state() {
        let discrete = VulkanKernelPolicy::from_capabilities(capabilities());
        assert!(!discrete.gdn_recurrent_host_visible_state);

        let mut unified_caps = capabilities();
        unified_caps.has_coherent_device_local_host_visible_memory = true;
        let unified = VulkanKernelPolicy::from_capabilities(unified_caps);
        assert!(unified.gdn_recurrent_host_visible_state);

        unified_caps.has_coherent_device_local_host_visible_memory = false;
        let missing_required_type = VulkanKernelPolicy::from_capabilities(unified_caps);
        assert!(!missing_required_type.gdn_recurrent_host_visible_state);
    }

    #[test]
    fn captured_rtx_6000_ada_limits_select_compatible_routes() {
        // Core limits captured in the repository's
        // VP_VULKANINFO_NVIDIA_RTX_6000_Ada_Generation_550_127_8_0.json.
        // Memory topology stays discrete: ordinary host-visible staging memory
        // is not assumed to be device-local.
        let caps = VulkanComputeCapabilities {
            api_version: 4_206_869,
            max_compute_work_group_count: [2_147_483_647, 65_535, 65_535],
            max_compute_work_group_invocations: 1024,
            max_compute_work_group_size: [1024, 1024, 64],
            max_compute_shared_memory_size: 49_152,
            max_push_constants_size: 256,
            max_per_stage_descriptor_storage_buffers: 1_048_576,
            max_descriptor_set_storage_buffers: 1_048_576,
            max_storage_buffer_range: u32::MAX as u64,
            supports_compute_subgroup_basic_arithmetic: true,
            has_coherent_device_local_host_visible_memory: false,
            host_visible_staging_is_cached: true,
        };

        let policy = VulkanKernelPolicy::from_capabilities(caps);
        assert!(policy.resident_decode_enabled);
        assert!(policy.gdn_chunkwise_forward_enabled);
        assert!(policy.flash_attn_prefill_enabled);
        assert!(policy.mlp_bf16_rows8);
        assert!(!policy.gdn_recurrent_host_visible_state);
        assert!(caps.supports_full_pipeline_prewarm());
    }
}

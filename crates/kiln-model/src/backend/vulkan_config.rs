//! Vulkan model-route projection of the qualified execution policy.

use kiln_vulkan_kernel::kernels::QUALIFIED_VULKAN_KERNEL_POLICY;

#[derive(Debug, Clone, Copy)]
pub(super) struct VulkanRuntimeConfig {
    pub(super) gdn_enabled: bool,
    pub(super) gdn_prefill_in_proj_enabled: bool,
    pub(super) gdn_gates_enabled: bool,
    pub(super) gdn_gated_rms_norm_enabled: bool,
    pub(super) gdn_full_chunk_forward_enabled: bool,
    pub(super) fused_conv1d_update_enabled: bool,
    pub(super) fused_conv1d_prefill_enabled: bool,
    pub(super) conv1d_prefill_single_submit_enabled: bool,
    pub(super) gdn_forward_sub_enabled: bool,
    pub(super) gdn_decode_fused_enabled: bool,
    pub(super) gdn_recurrent_unexpanded_qk_enabled: bool,
    pub(super) gdn_recurrent_qk_norm_unexpanded_enabled: bool,
    pub(super) linear_decode_enabled: bool,
    pub(super) linear_argmax_batch_enabled: bool,
    pub(super) full_attn_qkv_enabled: bool,
    pub(super) paged_attn_decode_batch_enabled: bool,
    pub(super) mlp_decode_enabled: bool,
    pub(super) mlp_gate_up_enabled: bool,
    pub(super) mlp_bf16_gate_up_f32_down_enabled: bool,
    pub(super) bf16_packed_linear_weights_enabled: bool,
    pub(super) bf16_packed_gdn_in_proj_weights_enabled: bool,
    pub(super) bf16_packed_full_attn_qkv_weights_enabled: bool,
    pub(super) bf16_packed_mlp_decode_weights_enabled: bool,
    pub(super) recurrent_state_residency_enabled: bool,
    pub(super) prefill_recurrent_state_residency_enabled: bool,
    pub(super) resident_decode_enabled: bool,
}

impl VulkanRuntimeConfig {
    pub(super) fn qualified() -> Self {
        let policy = QUALIFIED_VULKAN_KERNEL_POLICY;

        Self {
            gdn_enabled: policy.gdn_enabled,
            gdn_prefill_in_proj_enabled: policy.gdn_prefill_in_proj_enabled,
            gdn_gates_enabled: policy.gdn_gates_enabled,
            gdn_gated_rms_norm_enabled: policy.gdn_gated_rms_norm_enabled,
            gdn_full_chunk_forward_enabled: policy.gdn_full_chunk_forward_enabled,
            fused_conv1d_update_enabled: policy.fused_conv1d_update_enabled,
            fused_conv1d_prefill_enabled: policy.fused_conv1d_prefill_enabled,
            conv1d_prefill_single_submit_enabled: policy.conv1d_prefill_single_submit_enabled,
            gdn_forward_sub_enabled: policy.gdn_forward_sub_enabled,
            gdn_decode_fused_enabled: policy.gdn_decode_fused_enabled,
            gdn_recurrent_unexpanded_qk_enabled: policy.gdn_recurrent_unexpanded_qk_enabled,
            gdn_recurrent_qk_norm_unexpanded_enabled: policy
                .gdn_recurrent_qk_norm_unexpanded_enabled,
            linear_decode_enabled: policy.linear_decode_enabled,
            linear_argmax_batch_enabled: policy.linear_argmax_batch_enabled,
            full_attn_qkv_enabled: policy.full_attn_qkv_enabled,
            paged_attn_decode_batch_enabled: policy.paged_attn_decode_batch_enabled,
            mlp_decode_enabled: policy.mlp_decode_enabled,
            mlp_gate_up_enabled: policy.mlp_gate_up_enabled,
            mlp_bf16_gate_up_f32_down_enabled: policy.mlp_bf16_gate_up_f32_down_enabled,
            bf16_packed_linear_weights_enabled: policy.bf16_packed_linear_weights_enabled,
            bf16_packed_gdn_in_proj_weights_enabled: policy.bf16_packed_gdn_in_proj_weights_enabled,
            bf16_packed_full_attn_qkv_weights_enabled: policy
                .bf16_packed_full_attn_qkv_weights_enabled,
            bf16_packed_mlp_decode_weights_enabled: policy.bf16_packed_mlp_decode_weights_enabled,
            recurrent_state_residency_enabled: policy.recurrent_state_residency_enabled,
            prefill_recurrent_state_residency_enabled: policy
                .prefill_recurrent_state_residency_enabled,
            resident_decode_enabled: policy.resident_decode_enabled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_model_routes_match_the_shared_vulkan_policy() {
        let config = VulkanRuntimeConfig::qualified();
        let policy = QUALIFIED_VULKAN_KERNEL_POLICY;

        assert_eq!(config.gdn_enabled, policy.gdn_enabled);
        assert_eq!(config.mlp_decode_enabled, policy.mlp_decode_enabled);
        assert_eq!(config.linear_decode_enabled, policy.linear_decode_enabled);
        assert_eq!(config.full_attn_qkv_enabled, policy.full_attn_qkv_enabled);
        assert_eq!(
            config.paged_attn_decode_batch_enabled,
            policy.paged_attn_decode_batch_enabled
        );
        assert_eq!(
            config.recurrent_state_residency_enabled,
            policy.recurrent_state_residency_enabled
        );
        assert_eq!(
            config.resident_decode_enabled,
            policy.resident_decode_enabled
        );
    }
}

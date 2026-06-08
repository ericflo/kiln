//! Vulkan runtime configuration and env-gated support policy.

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
    pub(super) weight_prewarm_enabled: bool,
    pub(super) recurrent_state_residency_enabled: bool,
    pub(super) resident_decode_enabled: bool,
}

impl VulkanRuntimeConfig {
    pub(super) fn from_env() -> Self {
        let gdn_enabled = std::env::var("KILN_DISABLE_GDN_KERNEL").is_err();
        let gdn_prefill_in_proj_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_VULKAN_GDN_PREFILL_IN_PROJ").is_err();
        let gdn_gates_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATES").is_err();
        let gdn_gated_rms_norm_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_FUSED_GDN_GATED_RMS_NORM").is_err();
        // The fused full-chunk shader is parity-covered, but default-on A070
        // latency regressed on Strix Halo. Keep it available for explicit
        // tuning without changing the production route.
        let gdn_full_chunk_forward_enabled =
            gdn_enabled && std::env::var("KILN_ENABLE_VULKAN_GDN_FULL_CHUNK_FORWARD").is_ok();
        // Conv1d prefill now wins on Strix Halo, while single-token update
        // still regresses decode latency. Keep update opt-in and leave a
        // prefill rollback for driver/model-specific follow-up.
        let fused_conv1d_update_enabled = gdn_enabled
            && (std::env::var("KILN_ENABLE_VULKAN_FUSED_CONV1D").is_ok()
                || std::env::var("KILN_ENABLE_VULKAN_FUSED_CONV1D_UPDATE").is_ok());
        let fused_conv1d_prefill_enabled =
            gdn_enabled && std::env::var("KILN_DISABLE_VULKAN_FUSED_CONV1D_PREFILL").is_err();
        let conv1d_prefill_single_submit_enabled = fused_conv1d_prefill_enabled
            && std::env::var("KILN_DISABLE_VULKAN_CONV1D_PREFILL_SINGLE_SUBMIT").is_err();
        // forward_sub is opt-in only (default off): solve_tri shared-memory
        // layout is not yet validated against CPU parity and may exceed
        // maxComputeSharedMemorySize on many GPUs.
        let gdn_forward_sub_enabled =
            gdn_enabled && std::env::var("KILN_ENABLE_VULKAN_GDN_FORWARD_SUB").is_ok();
        // The fused GDN decode path is validated, but for bs=1 it remains
        // run-to-run unstable on Strix Halo. Batch decode enables it by shape
        // in `gdn_decode_gates_recurrent_rmsnorm`; this env gates bs=1 only.
        let gdn_decode_fused_enabled =
            gdn_enabled && std::env::var("KILN_ENABLE_VULKAN_GDN_DECODE_FUSED").is_ok();
        let gdn_recurrent_unexpanded_qk_enabled = gdn_enabled
            && std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_UNEXPANDED_QK").is_err();
        let gdn_recurrent_qk_norm_unexpanded_enabled = gdn_recurrent_unexpanded_qk_enabled
            && std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_QK_NORM").is_err();
        let linear_decode_enabled = std::env::var("KILN_DISABLE_VULKAN_LINEAR_DECODE").is_err();
        let bf16_packed_linear_weights_enabled = linear_decode_enabled
            && std::env::var("KILN_DISABLE_VULKAN_BF16_PACKED_LINEAR_WEIGHTS").is_err();
        let bf16_packed_gdn_in_proj_weights_enabled = gdn_enabled
            && std::env::var("KILN_DISABLE_VULKAN_BF16_PACKED_GDN_IN_PROJ_WEIGHTS").is_err();
        let linear_argmax_batch_enabled =
            std::env::var("KILN_DISABLE_VULKAN_LINEAR_ARGMAX_BATCH").is_err();
        let full_attn_qkv_enabled = std::env::var("KILN_DISABLE_VULKAN_FULL_ATTN_QKV").is_err();
        let bf16_packed_full_attn_qkv_weights_enabled = full_attn_qkv_enabled
            && std::env::var("KILN_DISABLE_VULKAN_BF16_PACKED_FULL_ATTN_QKV_WEIGHTS").is_err();
        let paged_attn_decode_batch_enabled =
            std::env::var("KILN_DISABLE_VULKAN_PAGED_ATTN_DECODE_BATCH").is_err();
        // Full fused MLP decode is validated for single-token no-LoRA decode.
        // After descriptor-pool reuse and tiled projection kernels it is now
        // consistently faster than the split generic GEMV path on Strix Halo.
        let mlp_decode_enabled = std::env::var("KILN_DISABLE_VULKAN_MLP_DECODE").is_err();
        let bf16_packed_mlp_decode_weights_enabled = mlp_decode_enabled
            && std::env::var("KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS").is_err();
        let mlp_bf16_gate_up_f32_down_enabled = bf16_packed_mlp_decode_weights_enabled
            && std::env::var("KILN_DISABLE_VULKAN_MLP_BF16_GATE_UP_F32_DOWN").is_err();
        // The fused Vulkan MLP gate/up shader is validated, but on Strix Halo
        // it was slower than the generic cached GEMV path in short decode
        // benchmarks. Keep it opt-in until it is tiled/tuned.
        let mlp_gate_up_enabled = std::env::var("KILN_ENABLE_VULKAN_MLP_GATE_UP").is_ok();
        let weight_prewarm_enabled = std::env::var("KILN_DISABLE_VULKAN_WEIGHT_PREWARM").is_err();
        // Device-resident recurrent state is correct but regressed the live
        // Strix Halo batcher A/B in A129 because row/batch buffer copies cost
        // more than the saved readback/upload at the current batch shape.
        let recurrent_state_residency_enabled = gdn_enabled
            && std::env::var("KILN_ENABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE").is_ok()
            && std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE").is_err();
        // Default ON: every Vulkan build that brings up a logical device wants
        // to route decode through the resident path. Pool feasibility is
        // checked later at first use; if the device can't fit the ring, the
        // call site falls back transparently to the per-call kt Tensor path and
        // emits a one-time tracing::warn!, exactly the contract spelled out in
        // gate (b) of docs/vk_resident_decode_plan.md.
        let resident_decode_enabled =
            kiln_core::env_flag::env_flag("KILN_VULKAN_RESIDENT_DECODE", true);

        Self {
            gdn_enabled,
            gdn_prefill_in_proj_enabled,
            gdn_gates_enabled,
            gdn_gated_rms_norm_enabled,
            gdn_full_chunk_forward_enabled,
            fused_conv1d_update_enabled,
            fused_conv1d_prefill_enabled,
            conv1d_prefill_single_submit_enabled,
            gdn_forward_sub_enabled,
            gdn_decode_fused_enabled,
            gdn_recurrent_unexpanded_qk_enabled,
            gdn_recurrent_qk_norm_unexpanded_enabled,
            linear_decode_enabled,
            linear_argmax_batch_enabled,
            full_attn_qkv_enabled,
            paged_attn_decode_batch_enabled,
            mlp_decode_enabled,
            mlp_gate_up_enabled,
            mlp_bf16_gate_up_f32_down_enabled,
            bf16_packed_linear_weights_enabled,
            bf16_packed_gdn_in_proj_weights_enabled,
            bf16_packed_full_attn_qkv_weights_enabled,
            bf16_packed_mlp_decode_weights_enabled,
            weight_prewarm_enabled,
            recurrent_state_residency_enabled,
            resident_decode_enabled,
        }
    }
}

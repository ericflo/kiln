use anyhow::{Context, Result};
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

fn compile_shader_command(
    glsl_path: &str,
    spv_path: &std::path::Path,
) -> std::io::Result<std::process::Output> {
    let glslc = std::process::Command::new("glslc")
        .arg(glsl_path)
        .arg("-o")
        .arg(spv_path)
        .arg("-DFLOAT_TYPE=float")
        .arg("-DUSE_BFLOAT16=1")
        .arg("-DUSE_SUBGROUP_ADD=1")
        .arg("-DUSE_SUBGROUP_CLUSTERED=1")
        .output();
    if glslc.is_ok() {
        return glslc;
    }

    std::process::Command::new("glslangValidator")
        .arg("-V")
        .arg(glsl_path)
        .arg("-o")
        .arg(spv_path)
        .arg("-DFLOAT_TYPE=float")
        .arg("-DUSE_BFLOAT16=1")
        .arg("-DUSE_SUBGROUP_ADD=1")
        .arg("-DUSE_SUBGROUP_CLUSTERED=1")
        .output()
}

// Include the build-time generated SPIR-V modules
include!(concat!(env!("OUT_DIR"), "/vulkan_spirv.rs"));

/// Map shader base names to their embedded SPIR-V module constants.
const SHADER_SPIRVS: &[(&str, &[u8])] = &[
    ("full_attn_qkv_decode", SPIR_V_FULL_ATTN_QKV_DECODE),
    (
        "full_attn_qkv_decode_bf16w",
        SPIR_V_FULL_ATTN_QKV_DECODE_BF16W,
    ),
    ("gdn_gates", SPIR_V_GDN_GATES),
    (
        "gdn_decode_gates_recurrent_rmsnorm",
        SPIR_V_GDN_DECODE_GATES_RECURRENT_RMSNORM,
    ),
    ("gdn_in_proj_decode", SPIR_V_GDN_IN_PROJ_DECODE),
    ("gdn_in_proj_decode_bf16w", SPIR_V_GDN_IN_PROJ_DECODE_BF16W),
    (
        "gdn_in_proj_decode_batched",
        SPIR_V_GDN_IN_PROJ_DECODE_BATCHED,
    ),
    (
        "gdn_in_proj_decode_batched_bf16w",
        SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_BF16W,
    ),
    (
        "gdn_in_proj_decode_batched_pair_qkv_z",
        SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z,
    ),
    (
        "gdn_in_proj_decode_batched_pair_qkv_z_bf16w",
        SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_BF16W,
    ),
    (
        "gdn_in_proj_decode_batched_pair_qkv_z_rows2_bf16w",
        SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS2_BF16W,
    ),
    (
        "gdn_in_proj_decode_batched_pair_qkv_z_rows4_bf16w",
        SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W,
    ),
    ("gdn_gated_rms_norm", SPIR_V_GDN_GATED_RMS_NORM),
    ("causal_conv1d", SPIR_V_CAUSAL_CONV1D),
    ("causal_conv1d_offset", SPIR_V_CAUSAL_CONV1D_OFFSET),
    (
        "vk_causal_conv1d_pre_silu",
        SPIR_V_VK_CAUSAL_CONV1D_PRE_SILU,
    ),
    (
        "vk_causal_conv1d_pre_silu_offset",
        SPIR_V_VK_CAUSAL_CONV1D_PRE_SILU_OFFSET,
    ),
    ("vk_causal_conv1d_bwd", SPIR_V_VK_CAUSAL_CONV1D_BWD),
    (
        "vk_causal_conv1d_bwd_offset",
        SPIR_V_VK_CAUSAL_CONV1D_BWD_OFFSET,
    ),
    (
        "causal_conv1d_state_advance",
        SPIR_V_CAUSAL_CONV1D_STATE_ADVANCE,
    ),
    ("solve_tri", SPIR_V_SOLVE_TRI),
    ("gdn_recurrent_prefill", SPIR_V_GDN_RECURRENT_PREFILL),
    (
        "gdn_recurrent_step_parallel",
        SPIR_V_GDN_RECURRENT_STEP_PARALLEL,
    ),
    ("gdn_chunk_prep", SPIR_V_GDN_CHUNK_PREP),
    ("gdn_full_chunk_forward", SPIR_V_GDN_FULL_CHUNK_FORWARD),
    ("gdn_chunk_scan", SPIR_V_GDN_CHUNK_SCAN),
    ("linear_decode", SPIR_V_LINEAR_DECODE),
    ("linear_decode_bf16w", SPIR_V_LINEAR_DECODE_BF16W),
    ("linear_decode_batched", SPIR_V_LINEAR_DECODE_BATCHED),
    (
        "linear_decode_batched_bf16w",
        SPIR_V_LINEAR_DECODE_BATCHED_BF16W,
    ),
    ("vk_matmul_bf16w_fwd_rows", SPIR_V_VK_MATMUL_BF16W_FWD_ROWS),
    ("vk_matmul_bf16w_bwd_rows", SPIR_V_VK_MATMUL_BF16W_BWD_ROWS),
    (
        "linear_decode_batched_rows2",
        SPIR_V_LINEAR_DECODE_BATCHED_ROWS2,
    ),
    (
        "linear_decode_batched_rows4",
        SPIR_V_LINEAR_DECODE_BATCHED_ROWS4,
    ),
    (
        "linear_decode_batched_rows4_bf16w",
        SPIR_V_LINEAR_DECODE_BATCHED_ROWS4_BF16W,
    ),
    (
        "linear_decode_argmax_blocks",
        SPIR_V_LINEAR_DECODE_ARGMAX_BLOCKS,
    ),
    (
        "linear_decode_argmax_blocks_bf16w",
        SPIR_V_LINEAR_DECODE_ARGMAX_BLOCKS_BF16W,
    ),
    (
        "linear_decode_argmax_reduce",
        SPIR_V_LINEAR_DECODE_ARGMAX_REDUCE,
    ),
    (
        "linear_decode_argmax_batched_blocks",
        SPIR_V_LINEAR_DECODE_ARGMAX_BATCHED_BLOCKS,
    ),
    (
        "linear_decode_argmax_batched_blocks_bf16w",
        SPIR_V_LINEAR_DECODE_ARGMAX_BATCHED_BLOCKS_BF16W,
    ),
    (
        "linear_decode_argmax_batched_reduce",
        SPIR_V_LINEAR_DECODE_ARGMAX_BATCHED_REDUCE,
    ),
    (
        "apply_token_penalties",
        SPIR_V_APPLY_TOKEN_PENALTIES,
    ),
    ("topk_sample", SPIR_V_TOPK_SAMPLE),
    ("mlp_gate_up_decode", SPIR_V_MLP_GATE_UP_DECODE),
    ("mlp_gate_up_decode_bf16w", SPIR_V_MLP_GATE_UP_DECODE_BF16W),
    (
        "mlp_gate_up_decode_batched",
        SPIR_V_MLP_GATE_UP_DECODE_BATCHED,
    ),
    (
        "mlp_gate_up_decode_batched_bf16w",
        SPIR_V_MLP_GATE_UP_DECODE_BATCHED_BF16W,
    ),
    (
        "mlp_gate_up_decode_batched_rows4_bf16w",
        SPIR_V_MLP_GATE_UP_DECODE_BATCHED_ROWS4_BF16W,
    ),
    (
        "mlp_gate_up_decode_batched_rows2",
        SPIR_V_MLP_GATE_UP_DECODE_BATCHED_ROWS2,
    ),
    ("paged_attn_decode_batch", SPIR_V_PAGED_ATTN_DECODE_BATCH),
    ("flash_attn", SPIR_V_FLASH_ATTN),
    ("sdpa_prefill_f32", SPIR_V_SDPA_PREFILL_F32),
    ("vk_flash_sdpa_fwd_f32", SPIR_V_VK_FLASH_SDPA_FWD_F32),
    (
        "vk_flash_sdpa_fwd_f32_offset",
        SPIR_V_VK_FLASH_SDPA_FWD_F32_OFFSET,
    ),
    (
        "vk_flash_sdpa_decode_split_f32",
        SPIR_V_VK_FLASH_SDPA_DECODE_SPLIT_F32,
    ),
    ("vk_flash_sdpa_delta_f32", SPIR_V_VK_FLASH_SDPA_DELTA_F32),
    (
        "vk_flash_sdpa_delta_f32_offset",
        SPIR_V_VK_FLASH_SDPA_DELTA_F32_OFFSET,
    ),
    ("vk_flash_sdpa_bwd_dq_f32", SPIR_V_VK_FLASH_SDPA_BWD_DQ_F32),
    (
        "vk_flash_sdpa_bwd_dq_f32_offset",
        SPIR_V_VK_FLASH_SDPA_BWD_DQ_F32_OFFSET,
    ),
    (
        "vk_flash_sdpa_bwd_dkdv_f32",
        SPIR_V_VK_FLASH_SDPA_BWD_DKDV_F32,
    ),
    (
        "vk_flash_sdpa_bwd_dkdv_f32_offset",
        SPIR_V_VK_FLASH_SDPA_BWD_DKDV_F32_OFFSET,
    ),
    ("sgd_step_f32", SPIR_V_SGD_STEP_F32),
    ("sgd_step_bf16", SPIR_V_SGD_STEP_BF16),
    ("adamw_step_f32", SPIR_V_ADAMW_STEP_F32),
    ("adamw_step_bf16", SPIR_V_ADAMW_STEP_BF16),
    // vk-native training shaders (Phase A)
    (
        "vk_elementwise_binary_f32",
        SPIR_V_VK_ELEMENTWISE_BINARY_F32,
    ),
    (
        "vk_elementwise_binary_f32_offset",
        SPIR_V_VK_ELEMENTWISE_BINARY_F32_OFFSET,
    ),
    ("vk_fill_f32", SPIR_V_VK_FILL_F32),
    ("vk_fill_f32_offset", SPIR_V_VK_FILL_F32_OFFSET),
    ("vk_reduce_sum_f32", SPIR_V_VK_REDUCE_SUM_F32),
    ("vk_broadcast_scalar_f32", SPIR_V_VK_BROADCAST_SCALAR_F32),
    (
        "vk_broadcast_scalar_f32_offset",
        SPIR_V_VK_BROADCAST_SCALAR_F32_OFFSET,
    ),
    ("vk_cast_f32_to_bf16", SPIR_V_VK_CAST_F32_TO_BF16),
    ("vk_cast_bf16_to_f32", SPIR_V_VK_CAST_BF16_TO_F32),
    ("vk_transpose_2d_f32", SPIR_V_VK_TRANSPOSE_2D_F32),
    ("vk_transpose_2d_bf16", SPIR_V_VK_TRANSPOSE_2D_BF16),
    ("vk_matmul_f32", SPIR_V_VK_MATMUL_F32),
    ("vk_softmax_lastdim_f32", SPIR_V_VK_SOFTMAX_LASTDIM_F32),
    (
        "vk_softmax_lastdim_bwd_f32",
        SPIR_V_VK_SOFTMAX_LASTDIM_BWD_F32,
    ),
    ("vk_gdn_gates_bwd", SPIR_V_VK_GDN_GATES_BWD),
    (
        "vk_gdn_gated_rms_norm_bwd",
        SPIR_V_VK_GDN_GATED_RMS_NORM_BWD,
    ),
    ("vk_l2_norm_lastdim_f32", SPIR_V_VK_L2_NORM_LASTDIM_F32),
    (
        "vk_l2_norm_lastdim_bwd_f32",
        SPIR_V_VK_L2_NORM_LASTDIM_BWD_F32,
    ),
    ("vk_silu_f32", SPIR_V_VK_SILU_F32),
    ("vk_silu_f32_offset", SPIR_V_VK_SILU_F32_OFFSET),
    ("vk_silu_bwd_f32", SPIR_V_VK_SILU_BWD_F32),
    ("vk_silu_bwd_f32_offset", SPIR_V_VK_SILU_BWD_F32_OFFSET),
    ("vk_rope_f32", SPIR_V_VK_ROPE_F32),
    ("vk_rope_bwd_f32", SPIR_V_VK_ROPE_BWD_F32),
    ("vk_causal_mask_add_f32", SPIR_V_VK_CAUSAL_MASK_ADD_F32),
    ("vk_scale_inplace_f32", SPIR_V_VK_SCALE_INPLACE_F32),
    (
        "vk_scale_inplace_f32_offset",
        SPIR_V_VK_SCALE_INPLACE_F32_OFFSET,
    ),
    ("vk_matmul_batched_f32", SPIR_V_VK_MATMUL_BATCHED_F32),
    ("vk_transpose_3d_f32", SPIR_V_VK_TRANSPOSE_3D_F32),
    ("vk_permute_rh_to_hr_f32", SPIR_V_VK_PERMUTE_RH_TO_HR_F32),
    (
        "vk_permute_rh_to_hr_f32_offset",
        SPIR_V_VK_PERMUTE_RH_TO_HR_F32_OFFSET,
    ),
    ("vk_permute_hr_to_rh_f32", SPIR_V_VK_PERMUTE_HR_TO_RH_F32),
    (
        "vk_permute_hr_to_rh_f32_offset",
        SPIR_V_VK_PERMUTE_HR_TO_RH_F32_OFFSET,
    ),
    ("vk_repeat_kv_heads_f32", SPIR_V_VK_REPEAT_KV_HEADS_F32),
    (
        "vk_repeat_kv_heads_f32_offset",
        SPIR_V_VK_REPEAT_KV_HEADS_F32_OFFSET,
    ),
    ("vk_sum_kv_groups_f32", SPIR_V_VK_SUM_KV_GROUPS_F32),
    (
        "vk_sum_kv_groups_f32_offset",
        SPIR_V_VK_SUM_KV_GROUPS_F32_OFFSET,
    ),
    ("vk_embedding_lookup_f32", SPIR_V_VK_EMBEDDING_LOOKUP_F32),
    (
        "vk_embedding_lookup_bf16w_f32",
        SPIR_V_VK_EMBEDDING_LOOKUP_BF16W_F32,
    ),
    ("vk_flce_chunk_stats_f32", SPIR_V_VK_FLCE_CHUNK_STATS_F32),
    (
        "vk_flce_gather_correct_f32",
        SPIR_V_VK_FLCE_GATHER_CORRECT_F32,
    ),
    (
        "vk_flce_log_sum_exp_combine_f32",
        SPIR_V_VK_FLCE_LOG_SUM_EXP_COMBINE_F32,
    ),
    (
        "vk_flce_per_token_loss_f32",
        SPIR_V_VK_FLCE_PER_TOKEN_LOSS_F32,
    ),
    ("vk_flce_grad_chunk_f32", SPIR_V_VK_FLCE_GRAD_CHUNK_F32),
    ("vk_selected_logprob_f32", SPIR_V_VK_SELECTED_LOGPROB_F32),
    ("vk_grpo_per_token_f32", SPIR_V_VK_GRPO_PER_TOKEN_F32),
    ("vk_grpo_grad_chunk_f32", SPIR_V_VK_GRPO_GRAD_CHUNK_F32),
    ("vk_sigmoid_f32", SPIR_V_VK_SIGMOID_F32),
    ("vk_sigmoid_f32_offset", SPIR_V_VK_SIGMOID_F32_OFFSET),
    ("vk_sigmoid_bwd_f32", SPIR_V_VK_SIGMOID_BWD_F32),
    (
        "vk_sigmoid_bwd_f32_offset",
        SPIR_V_VK_SIGMOID_BWD_F32_OFFSET,
    ),
    ("vk_mul_sigmoid_gate_f32", SPIR_V_VK_MUL_SIGMOID_GATE_F32),
    (
        "vk_mul_sigmoid_gate_f32_offset",
        SPIR_V_VK_MUL_SIGMOID_GATE_F32_OFFSET,
    ),
    (
        "vk_mul_sigmoid_gate_bwd_f32",
        SPIR_V_VK_MUL_SIGMOID_GATE_BWD_F32,
    ),
    (
        "vk_mul_sigmoid_gate_bwd_f32_offset",
        SPIR_V_VK_MUL_SIGMOID_GATE_BWD_F32_OFFSET,
    ),
    ("vk_narrow_lastdim_f32", SPIR_V_VK_NARROW_LASTDIM_F32),
    (
        "vk_narrow_lastdim_f32_offset",
        SPIR_V_VK_NARROW_LASTDIM_F32_OFFSET,
    ),
    (
        "vk_narrow_lastdim_bwd_f32",
        SPIR_V_VK_NARROW_LASTDIM_BWD_F32,
    ),
    (
        "vk_narrow_lastdim_bwd_f32_offset",
        SPIR_V_VK_NARROW_LASTDIM_BWD_F32_OFFSET,
    ),
    ("vk_index_select_rows_f32", SPIR_V_VK_INDEX_SELECT_ROWS_F32),
    (
        "vk_index_select_rows_bwd_f32",
        SPIR_V_VK_INDEX_SELECT_ROWS_BWD_F32,
    ),
];

// Re-export the spirv_modules for use in SHADER_SPIRVS
use spirv_modules::*;

/// Shader pipeline cache and dispatcher.
pub struct ShaderPipeline {
    pipelines: HashMap<String, (vk::PipelineLayout, vk::Pipeline)>,
    #[allow(dead_code)]
    device: Arc<ash::Device>,
}

impl std::fmt::Debug for ShaderPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderPipeline")
            .field("pipeline_count", &self.pipelines.len())
            .finish()
    }
}

impl ShaderPipeline {
    pub fn new(device: &Arc<ash::Device>) -> Self {
        Self {
            pipelines: HashMap::new(),
            device: Arc::clone(device),
        }
    }

    /// Get embedded SPIR-V for a shader by base name.
    /// Falls back to runtime `glslc` compilation if the embedded module is empty
    /// (e.g., when glslc was not available at build time).
    pub fn compile_shader(glsl_path: &str) -> Result<Vec<u8>> {
        // Extract base name from path (e.g., "gdn_gates" from ".../gdn_gates.comp")
        let stem = std::path::Path::new(glsl_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("failed to extract shader name from path"))?;

        // Check embedded SPIR-V first
        for (name, spv) in SHADER_SPIRVS {
            if *name == stem && !spv.is_empty() {
                tracing::trace!(shader = %name, "using embedded SPIR-V");
                return Ok(spv.to_vec());
            }
        }

        // Fall back to runtime glslc compilation
        tracing::trace!(shader = %stem, "compiling shader at runtime via glslc");
        Self::compile_shader_runtime(glsl_path)
    }

    /// Runtime glslc compilation (fallback when embedded SPIR-V is unavailable).
    fn compile_shader_runtime(glsl_path: &str) -> Result<Vec<u8>> {
        // Write to temp dir, not source tree (source tree may be read-only in prod).
        let stem = std::path::Path::new(glsl_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("shader");
        let spv_path = std::env::temp_dir().join(format!("kiln_vulkan_{}.spv", stem));

        // Try to load a pre-compiled .spv from temp dir (cached from a prior run)
        if spv_path.exists() {
            return std::fs::read(&spv_path).context(format!(
                "failed to read cached SPIR-V: {}",
                spv_path.display()
            ));
        }

        let output = compile_shader_command(glsl_path, &spv_path)
            .context("failed to run glslc or glslangValidator")?;

        if !output.status.success() {
            anyhow::bail!(
                "shader compiler failed to compile '{}': {}",
                glsl_path,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        std::fs::read(&spv_path).context(format!(
            "failed to read compiled SPIR-V: {}",
            spv_path.display()
        ))
    }

    /// Create or retrieve a cached compute pipeline.
    pub fn get_or_create(
        &mut self,
        name: &str,
        spirv: &[u8],
        push_constant_size: u32,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
        if let Some((layout, pipeline)) = self.pipelines.get(name) {
            return Ok((*layout, *pipeline));
        }

        // Create shader module
        let spirv_words: &[u32] = bytemuck::cast_slice(spirv);
        let shader_module_info = vk::ShaderModuleCreateInfo::builder()
            .code(spirv_words)
            .build();

        let shader_module = unsafe {
            self.device
                .create_shader_module(&shader_module_info, None)
                .context(format!("failed to create shader module: {}", name))?
        };

        // Create pipeline layout
        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_constant_size)
            .build();
        let pcr = vec![push_constant_range];

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&pcr)
            .build();

        let layout = unsafe {
            self.device
                .create_pipeline_layout(&layout_info, None)
                .context(format!("failed to create pipeline layout: {}", name))?
        };

        // Create compute pipeline
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
            .build();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_info)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1)
            .build();

        let pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(errs, _)| {
                    if !errs.is_empty() {
                        anyhow::anyhow!("failed to create compute pipeline {}: {:?}", name, errs[0])
                    } else {
                        anyhow::anyhow!("failed to create compute pipeline {}", name)
                    }
                })?
        };
        let pipeline = pipelines[0];

        // Clean up shader module
        unsafe {
            self.device.destroy_shader_module(shader_module, None);
        }

        self.pipelines.insert(name.to_string(), (layout, pipeline));
        Ok((layout, pipeline))
    }

    /// Cleanup all pipelines.
    pub fn cleanup(&mut self) {
        for (_, (layout, pipeline)) in self.pipelines.drain() {
            unsafe {
                self.device.destroy_pipeline(pipeline, None);
                self.device.destroy_pipeline_layout(layout, None);
            }
        }
    }

    /// Get the number of cached pipelines.
    pub fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }
}

impl Drop for ShaderPipeline {
    fn drop(&mut self) {
        self.cleanup();
    }
}

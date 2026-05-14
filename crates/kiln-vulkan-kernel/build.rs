//! Build script for kiln-vulkan-kernel.
//!
//! Compiles GLSL compute shaders to SPIR-V at build time using `glslc`,
//! then embeds them into the binary via `include_bytes!` macros.
//!
//! This avoids the runtime `glslc` shellout, which is unsuitable for
//! production (source tree is read-only, glslc may not be on PATH, etc.)

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Shader files to compile. Format: (base_name, output_module_name).
const SHADERS: &[(&str, &str)] = &[
    ("full_attn_qkv_decode", "SPIR_V_FULL_ATTN_QKV_DECODE"),
    (
        "full_attn_qkv_decode_bf16w",
        "SPIR_V_FULL_ATTN_QKV_DECODE_BF16W",
    ),
    ("gdn_gates", "SPIR_V_GDN_GATES"),
    (
        "gdn_decode_gates_recurrent_rmsnorm",
        "SPIR_V_GDN_DECODE_GATES_RECURRENT_RMSNORM",
    ),
    ("gdn_in_proj_decode", "SPIR_V_GDN_IN_PROJ_DECODE"),
    (
        "gdn_in_proj_decode_bf16w",
        "SPIR_V_GDN_IN_PROJ_DECODE_BF16W",
    ),
    (
        "gdn_in_proj_decode_batched",
        "SPIR_V_GDN_IN_PROJ_DECODE_BATCHED",
    ),
    (
        "gdn_in_proj_decode_batched_bf16w",
        "SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_BF16W",
    ),
    (
        "gdn_in_proj_decode_batched_pair_qkv_z",
        "SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z",
    ),
    (
        "gdn_in_proj_decode_batched_pair_qkv_z_bf16w",
        "SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_BF16W",
    ),
    (
        "gdn_in_proj_decode_batched_pair_qkv_z_rows2_bf16w",
        "SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS2_BF16W",
    ),
    (
        "gdn_in_proj_decode_batched_pair_qkv_z_rows4_bf16w",
        "SPIR_V_GDN_IN_PROJ_DECODE_BATCHED_PAIR_QKV_Z_ROWS4_BF16W",
    ),
    ("gdn_gated_rms_norm", "SPIR_V_GDN_GATED_RMS_NORM"),
    ("causal_conv1d", "SPIR_V_CAUSAL_CONV1D"),
    ("causal_conv1d_offset", "SPIR_V_CAUSAL_CONV1D_OFFSET"),
    (
        "vk_causal_conv1d_pre_silu",
        "SPIR_V_VK_CAUSAL_CONV1D_PRE_SILU",
    ),
    (
        "vk_causal_conv1d_pre_silu_offset",
        "SPIR_V_VK_CAUSAL_CONV1D_PRE_SILU_OFFSET",
    ),
    ("vk_causal_conv1d_bwd", "SPIR_V_VK_CAUSAL_CONV1D_BWD"),
    (
        "vk_causal_conv1d_bwd_offset",
        "SPIR_V_VK_CAUSAL_CONV1D_BWD_OFFSET",
    ),
    (
        "causal_conv1d_state_advance",
        "SPIR_V_CAUSAL_CONV1D_STATE_ADVANCE",
    ),
    ("solve_tri", "SPIR_V_SOLVE_TRI"),
    ("gdn_recurrent_prefill", "SPIR_V_GDN_RECURRENT_PREFILL"),
    (
        "gdn_recurrent_step_parallel",
        "SPIR_V_GDN_RECURRENT_STEP_PARALLEL",
    ),
    ("gdn_chunk_prep", "SPIR_V_GDN_CHUNK_PREP"),
    ("gdn_full_chunk_forward", "SPIR_V_GDN_FULL_CHUNK_FORWARD"),
    ("gdn_chunk_scan", "SPIR_V_GDN_CHUNK_SCAN"),
    ("linear_decode", "SPIR_V_LINEAR_DECODE"),
    ("linear_decode_bf16w", "SPIR_V_LINEAR_DECODE_BF16W"),
    ("linear_decode_batched", "SPIR_V_LINEAR_DECODE_BATCHED"),
    (
        "linear_decode_batched_bf16w",
        "SPIR_V_LINEAR_DECODE_BATCHED_BF16W",
    ),
    (
        "linear_decode_batched_offset_bf16w",
        "SPIR_V_LINEAR_DECODE_BATCHED_OFFSET_BF16W",
    ),
    (
        "linear_decode_batched_transposed_bf16w",
        "SPIR_V_LINEAR_DECODE_BATCHED_TRANSPOSED_BF16W",
    ),
    (
        "vk_matmul_bf16w_fwd_rows",
        "SPIR_V_VK_MATMUL_BF16W_FWD_ROWS",
    ),
    (
        "vk_matmul_bf16w_bwd_rows",
        "SPIR_V_VK_MATMUL_BF16W_BWD_ROWS",
    ),
    ("qwen_rmsnorm_forward", "SPIR_V_QWEN_RMSNORM_FORWARD"),
    ("qwen_rmsnorm_backward", "SPIR_V_QWEN_RMSNORM_BACKWARD"),
    (
        "linear_decode_batched_rows2",
        "SPIR_V_LINEAR_DECODE_BATCHED_ROWS2",
    ),
    (
        "linear_decode_batched_rows4",
        "SPIR_V_LINEAR_DECODE_BATCHED_ROWS4",
    ),
    (
        "linear_decode_argmax_blocks",
        "SPIR_V_LINEAR_DECODE_ARGMAX_BLOCKS",
    ),
    (
        "linear_decode_argmax_blocks_bf16w",
        "SPIR_V_LINEAR_DECODE_ARGMAX_BLOCKS_BF16W",
    ),
    (
        "linear_decode_argmax_reduce",
        "SPIR_V_LINEAR_DECODE_ARGMAX_REDUCE",
    ),
    (
        "linear_decode_argmax_batched_blocks",
        "SPIR_V_LINEAR_DECODE_ARGMAX_BATCHED_BLOCKS",
    ),
    (
        "linear_decode_argmax_batched_blocks_bf16w",
        "SPIR_V_LINEAR_DECODE_ARGMAX_BATCHED_BLOCKS_BF16W",
    ),
    (
        "linear_decode_argmax_batched_reduce",
        "SPIR_V_LINEAR_DECODE_ARGMAX_BATCHED_REDUCE",
    ),
    (
        "apply_token_penalties",
        "SPIR_V_APPLY_TOKEN_PENALTIES",
    ),
    ("topk_sample", "SPIR_V_TOPK_SAMPLE"),
    ("mlp_gate_up_decode", "SPIR_V_MLP_GATE_UP_DECODE"),
    (
        "mlp_gate_up_decode_bf16w",
        "SPIR_V_MLP_GATE_UP_DECODE_BF16W",
    ),
    (
        "mlp_gate_up_decode_batched",
        "SPIR_V_MLP_GATE_UP_DECODE_BATCHED",
    ),
    (
        "mlp_gate_up_decode_batched_bf16w",
        "SPIR_V_MLP_GATE_UP_DECODE_BATCHED_BF16W",
    ),
    (
        "mlp_gate_up_decode_batched_rows4_bf16w",
        "SPIR_V_MLP_GATE_UP_DECODE_BATCHED_ROWS4_BF16W",
    ),
    (
        "mlp_gate_up_decode_batched_rows2",
        "SPIR_V_MLP_GATE_UP_DECODE_BATCHED_ROWS2",
    ),
    ("paged_attn_decode_batch", "SPIR_V_PAGED_ATTN_DECODE_BATCH"),
    ("flash_attn", "SPIR_V_FLASH_ATTN"),
    ("sdpa_prefill_f32", "SPIR_V_SDPA_PREFILL_F32"),
    ("vk_flash_sdpa_fwd_f32", "SPIR_V_VK_FLASH_SDPA_FWD_F32"),
    (
        "vk_flash_sdpa_fwd_f32_offset",
        "SPIR_V_VK_FLASH_SDPA_FWD_F32_OFFSET",
    ),
    (
        "vk_flash_sdpa_decode_split_f32",
        "SPIR_V_VK_FLASH_SDPA_DECODE_SPLIT_F32",
    ),
    ("vk_flash_sdpa_delta_f32", "SPIR_V_VK_FLASH_SDPA_DELTA_F32"),
    (
        "vk_flash_sdpa_delta_f32_offset",
        "SPIR_V_VK_FLASH_SDPA_DELTA_F32_OFFSET",
    ),
    (
        "vk_flash_sdpa_bwd_dq_f32",
        "SPIR_V_VK_FLASH_SDPA_BWD_DQ_F32",
    ),
    (
        "vk_flash_sdpa_bwd_dq_f32_offset",
        "SPIR_V_VK_FLASH_SDPA_BWD_DQ_F32_OFFSET",
    ),
    (
        "vk_flash_sdpa_bwd_dkdv_f32",
        "SPIR_V_VK_FLASH_SDPA_BWD_DKDV_F32",
    ),
    (
        "vk_flash_sdpa_bwd_dkdv_f32_offset",
        "SPIR_V_VK_FLASH_SDPA_BWD_DKDV_F32_OFFSET",
    ),
    ("sgd_step_f32", "SPIR_V_SGD_STEP_F32"),
    ("sgd_step_bf16", "SPIR_V_SGD_STEP_BF16"),
    ("adamw_step_f32", "SPIR_V_ADAMW_STEP_F32"),
    ("adamw_step_bf16", "SPIR_V_ADAMW_STEP_BF16"),
    // vk-native training shaders (Phase A)
    (
        "vk_elementwise_binary_f32",
        "SPIR_V_VK_ELEMENTWISE_BINARY_F32",
    ),
    (
        "vk_elementwise_binary_f32_offset",
        "SPIR_V_VK_ELEMENTWISE_BINARY_F32_OFFSET",
    ),
    ("vk_fill_f32", "SPIR_V_VK_FILL_F32"),
    ("vk_fill_f32_offset", "SPIR_V_VK_FILL_F32_OFFSET"),
    ("vk_reduce_sum_f32", "SPIR_V_VK_REDUCE_SUM_F32"),
    ("vk_broadcast_scalar_f32", "SPIR_V_VK_BROADCAST_SCALAR_F32"),
    (
        "vk_broadcast_scalar_f32_offset",
        "SPIR_V_VK_BROADCAST_SCALAR_F32_OFFSET",
    ),
    ("vk_cast_f32_to_bf16", "SPIR_V_VK_CAST_F32_TO_BF16"),
    ("vk_cast_bf16_to_f32", "SPIR_V_VK_CAST_BF16_TO_F32"),
    ("vk_transpose_2d_f32", "SPIR_V_VK_TRANSPOSE_2D_F32"),
    ("vk_transpose_2d_bf16", "SPIR_V_VK_TRANSPOSE_2D_BF16"),
    ("vk_matmul_f32", "SPIR_V_VK_MATMUL_F32"),
    ("vk_softmax_lastdim_f32", "SPIR_V_VK_SOFTMAX_LASTDIM_F32"),
    (
        "vk_softmax_lastdim_bwd_f32",
        "SPIR_V_VK_SOFTMAX_LASTDIM_BWD_F32",
    ),
    ("vk_gdn_gates_bwd", "SPIR_V_VK_GDN_GATES_BWD"),
    (
        "vk_gdn_gated_rms_norm_bwd",
        "SPIR_V_VK_GDN_GATED_RMS_NORM_BWD",
    ),
    ("vk_l2_norm_lastdim_f32", "SPIR_V_VK_L2_NORM_LASTDIM_F32"),
    (
        "vk_l2_norm_lastdim_bwd_f32",
        "SPIR_V_VK_L2_NORM_LASTDIM_BWD_F32",
    ),
    ("vk_silu_f32", "SPIR_V_VK_SILU_F32"),
    ("vk_silu_f32_offset", "SPIR_V_VK_SILU_F32_OFFSET"),
    ("vk_silu_bwd_f32", "SPIR_V_VK_SILU_BWD_F32"),
    ("vk_silu_bwd_f32_offset", "SPIR_V_VK_SILU_BWD_F32_OFFSET"),
    ("vk_rope_f32", "SPIR_V_VK_ROPE_F32"),
    ("vk_rope_bwd_f32", "SPIR_V_VK_ROPE_BWD_F32"),
    ("vk_causal_mask_add_f32", "SPIR_V_VK_CAUSAL_MASK_ADD_F32"),
    ("vk_scale_inplace_f32", "SPIR_V_VK_SCALE_INPLACE_F32"),
    (
        "vk_scale_inplace_f32_offset",
        "SPIR_V_VK_SCALE_INPLACE_F32_OFFSET",
    ),
    ("vk_matmul_batched_f32", "SPIR_V_VK_MATMUL_BATCHED_F32"),
    ("vk_transpose_3d_f32", "SPIR_V_VK_TRANSPOSE_3D_F32"),
    ("vk_permute_rh_to_hr_f32", "SPIR_V_VK_PERMUTE_RH_TO_HR_F32"),
    (
        "vk_permute_rh_to_hr_f32_offset",
        "SPIR_V_VK_PERMUTE_RH_TO_HR_F32_OFFSET",
    ),
    ("vk_permute_hr_to_rh_f32", "SPIR_V_VK_PERMUTE_HR_TO_RH_F32"),
    (
        "vk_permute_hr_to_rh_f32_offset",
        "SPIR_V_VK_PERMUTE_HR_TO_RH_F32_OFFSET",
    ),
    ("vk_repeat_kv_heads_f32", "SPIR_V_VK_REPEAT_KV_HEADS_F32"),
    (
        "vk_repeat_kv_heads_f32_offset",
        "SPIR_V_VK_REPEAT_KV_HEADS_F32_OFFSET",
    ),
    ("vk_sum_kv_groups_f32", "SPIR_V_VK_SUM_KV_GROUPS_F32"),
    (
        "vk_sum_kv_groups_f32_offset",
        "SPIR_V_VK_SUM_KV_GROUPS_F32_OFFSET",
    ),
    ("vk_embedding_lookup_f32", "SPIR_V_VK_EMBEDDING_LOOKUP_F32"),
    (
        "vk_embedding_lookup_bf16w_f32",
        "SPIR_V_VK_EMBEDDING_LOOKUP_BF16W_F32",
    ),
    ("vk_flce_chunk_stats_f32", "SPIR_V_VK_FLCE_CHUNK_STATS_F32"),
    (
        "vk_flce_gather_correct_f32",
        "SPIR_V_VK_FLCE_GATHER_CORRECT_F32",
    ),
    (
        "vk_flce_log_sum_exp_combine_f32",
        "SPIR_V_VK_FLCE_LOG_SUM_EXP_COMBINE_F32",
    ),
    (
        "vk_flce_per_token_loss_f32",
        "SPIR_V_VK_FLCE_PER_TOKEN_LOSS_F32",
    ),
    ("vk_flce_grad_chunk_f32", "SPIR_V_VK_FLCE_GRAD_CHUNK_F32"),
    ("vk_selected_logprob_f32", "SPIR_V_VK_SELECTED_LOGPROB_F32"),
    ("vk_grpo_per_token_f32", "SPIR_V_VK_GRPO_PER_TOKEN_F32"),
    ("vk_grpo_grad_chunk_f32", "SPIR_V_VK_GRPO_GRAD_CHUNK_F32"),
    ("vk_sigmoid_f32", "SPIR_V_VK_SIGMOID_F32"),
    ("vk_sigmoid_f32_offset", "SPIR_V_VK_SIGMOID_F32_OFFSET"),
    ("vk_sigmoid_bwd_f32", "SPIR_V_VK_SIGMOID_BWD_F32"),
    (
        "vk_sigmoid_bwd_f32_offset",
        "SPIR_V_VK_SIGMOID_BWD_F32_OFFSET",
    ),
    ("vk_mul_sigmoid_gate_f32", "SPIR_V_VK_MUL_SIGMOID_GATE_F32"),
    (
        "vk_mul_sigmoid_gate_f32_offset",
        "SPIR_V_VK_MUL_SIGMOID_GATE_F32_OFFSET",
    ),
    (
        "vk_mul_sigmoid_gate_bwd_f32",
        "SPIR_V_VK_MUL_SIGMOID_GATE_BWD_F32",
    ),
    (
        "vk_mul_sigmoid_gate_bwd_f32_offset",
        "SPIR_V_VK_MUL_SIGMOID_GATE_BWD_F32_OFFSET",
    ),
    ("vk_narrow_lastdim_f32", "SPIR_V_VK_NARROW_LASTDIM_F32"),
    (
        "vk_narrow_lastdim_f32_offset",
        "SPIR_V_VK_NARROW_LASTDIM_F32_OFFSET",
    ),
    (
        "vk_narrow_lastdim_bwd_f32",
        "SPIR_V_VK_NARROW_LASTDIM_BWD_F32",
    ),
    (
        "vk_narrow_lastdim_bwd_f32_offset",
        "SPIR_V_VK_NARROW_LASTDIM_BWD_F32_OFFSET",
    ),
    (
        "vk_index_select_rows_f32",
        "SPIR_V_VK_INDEX_SELECT_ROWS_F32",
    ),
    (
        "vk_index_select_rows_bwd_f32",
        "SPIR_V_VK_INDEX_SELECT_ROWS_BWD_F32",
    ),
];

fn compile_shader_command(
    glsl_path: &std::path::Path,
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

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc_dir = manifest_dir.join("csrc/shaders");

    // Ensure the output directory exists
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    fs::create_dir_all(&out_dir).unwrap();

    // Compile each shader
    for (name, _var_name) in SHADERS {
        let glsl_path = csrc_dir.join(format!("{}.comp", name));
        let spv_path = out_dir.join(format!("{}.spv", name));

        if !glsl_path.exists() {
            eprintln!(
                "cargo:warning=GLSL shader not found: {}",
                glsl_path.display()
            );
            continue;
        }

        let output = compile_shader_command(&glsl_path, &spv_path);

        match &output {
            Ok(output) if output.status.success() => {
                // silent on success — println! from build.rs is invisible to cargo
            }
            Ok(output) => {
                eprintln!(
                    "cargo:warning=glslc failed for {}: {} — shaders will compile at runtime",
                    name,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            Err(e) => {
                eprintln!(
                    "cargo:warning=glslc not found ({}) — Vulkan shaders will compile at runtime (or fail with a clear error)",
                    e
                );
            }
        }
    }

    // Generate the Rust code with include_bytes! macros
    let mut out = String::new();
    out.push_str("// Auto-generated by build.rs — do not edit\n");
    out.push_str("// Vulkan SPIR-V shader modules embedded at build time\n\n");
    out.push_str("#[rustfmt::skip]\n");
    out.push_str("pub mod spirv_modules {\n");
    for (name, var_name) in SHADERS {
        let spv_path = out_dir.join(format!("{}.spv", name));
        if spv_path.exists() {
            let spv_bytes = fs::read(&spv_path).unwrap_or_default();
            let len = spv_bytes.len();
            out.push_str(&format!(
                "    /// Embedded SPIR-V for {}.comp ({len} bytes)\n",
                name
            ));
            out.push_str(&format!(
                "    pub const {}: &[u8] = include_bytes!(\"{}\");\n\n",
                var_name,
                spv_path.display()
            ));
        } else {
            // Fallback: empty bytes (will trigger runtime compilation if needed)
            out.push_str(&format!(
                "    /// Shader {} not compiled at build time\n",
                name
            ));
            out.push_str(&format!("    pub const {}: &[u8] = &[];\n\n", var_name));
        }
    }
    out.push_str("}\n");

    // Write the generated file
    let generated_path = out_dir.join("vulkan_spirv.rs");
    let mut file = fs::File::create(&generated_path).unwrap();
    file.write_all(out.as_bytes()).unwrap();

    println!("cargo:rerun-if-changed=csrc/shaders/");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PATH");
}

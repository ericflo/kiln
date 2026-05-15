use crate::buffer::VulkanBuffer;
use crate::device::VulkanDevice;
use anyhow::{Context, Result};
use ash::vk;
use candle_core::{DType, Device, Tensor};
use half::bf16;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

fn env_truthy_for_profile(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn profile_vulkan_mlp_kernel_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy_for_profile("KILN_PROFILE_VULKAN_MLP_KERNEL_STAGES"))
}

fn mlp_bf16_gate_up_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_BF16_GATE_UP_ROWS4").is_err())
}

fn mlp_f32_down_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_F32_DOWN_ROWS4").is_err())
}

fn mlp_bf16_down_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_BF16_DOWN_ROWS4").is_err())
}

fn mlp_bf16_rows8_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_BF16_ROWS8").is_err())
}

fn linear_decode_bf16w_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_LINEAR_DECODE_BF16W_ROWS4").is_err()
    })
}

fn paged_attn_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_PAGED_ATTN_SINGLE_SUBMIT").is_err()
    })
}

fn qwen_rmsnorm_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_QWEN_RMSNORM_SINGLE_SUBMIT").is_err()
    })
}

fn full_attn_qkv_bf16w_rows4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_FULL_ATTN_QKV_BF16W_ROWS4").is_err()
    })
}

fn mlp_chained_dispatch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_CHAINED_DISPATCH").is_err())
}

fn mlp_chained_transfer_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_MLP_CHAINED_TRANSFER_SUBMIT").is_err())
}

fn profile_vulkan_gdn_in_proj_kernel_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy_for_profile("KILN_PROFILE_VULKAN_GDN_IN_PROJ_KERNEL_STAGES"))
}

fn profile_vulkan_gdn_recurrent_kernel_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| env_truthy_for_profile("KILN_PROFILE_VULKAN_GDN_RECURRENT_KERNEL_STAGES"))
}

#[allow(clippy::too_many_arguments)]
fn finish_vulkan_mlp_kernel_stage_profile(
    stage: &str,
    batch: usize,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
    gate_up_bf16_weights: bool,
    down_bf16_weights: bool,
    gate_up_rows2: bool,
    gate_up_rows4: bool,
    down_rows4: bool,
    down_rows2: bool,
    start: Option<Instant>,
) {
    let Some(start) = start else {
        return;
    };
    eprintln!(
        "kiln_profile_vulkan_mlp_kernel_stage stage={stage} batch={batch} hidden={hidden} intermediate={intermediate} out_dim={out_dim} bf16_weights={gate_up_bf16_weights} down_bf16_weights={down_bf16_weights} rows2={gate_up_rows2} gate_up_rows4={gate_up_rows4} down_rows4={down_rows4} down_rows2={down_rows2} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

#[allow(clippy::too_many_arguments)]
fn finish_vulkan_gdn_in_proj_kernel_stage_profile(
    stage: &str,
    batch: usize,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    packed_bf16_weights: bool,
    pair_qkv_z: bool,
    row_group_size: usize,
    single_submit: bool,
    start: Option<Instant>,
) {
    let Some(start) = start else {
        return;
    };
    eprintln!(
        "kiln_profile_vulkan_gdn_in_proj_kernel_stage stage={stage} batch={batch} hidden={hidden} qkv_dim={qkv_dim} z_dim={z_dim} a_dim={a_dim} b_dim={b_dim} packed_bf16_weights={packed_bf16_weights} pair_qkv_z={pair_qkv_z} row_group_size={row_group_size} single_submit={single_submit} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

#[allow(clippy::too_many_arguments)]
fn finish_vulkan_gdn_recurrent_kernel_stage_profile(
    stage: &str,
    batch: usize,
    heads: usize,
    dk: usize,
    dv: usize,
    parallel_reduce: bool,
    single_submit: bool,
    skip_state_readback: bool,
    start: Option<Instant>,
) {
    let Some(start) = start else {
        return;
    };
    eprintln!(
        "kiln_profile_vulkan_gdn_recurrent_kernel_stage stage={stage} batch={batch} heads={heads} dk={dk} dv={dv} parallel_reduce={parallel_reduce} single_submit={single_submit} skip_state_readback={skip_state_readback} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

fn gdn_decode_host_visible_state_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_ENABLE_VULKAN_GDN_HOST_VISIBLE_STATE").is_ok())
}

fn gdn_decode_fused_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_ENABLE_VULKAN_GDN_DECODE_FUSED_SINGLE_SUBMIT").is_ok())
}

fn gdn_recurrent_host_visible_state_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_STATE").is_err()
    })
}

fn gdn_recurrent_host_visible_batch_state_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_ENABLE_VULKAN_GDN_RECURRENT_HOST_VISIBLE_BATCH_STATE").is_ok()
    })
}

fn gdn_recurrent_use_host_visible_state(batch: usize) -> bool {
    gdn_recurrent_host_visible_state_enabled()
        && (batch == 1 || gdn_recurrent_host_visible_batch_state_enabled())
}

fn gdn_recurrent_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_SINGLE_SUBMIT").is_err())
}

fn gdn_recurrent_parallel_reduce_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_RECURRENT_PARALLEL_REDUCE").is_err())
}

fn use_gdn_recurrent_parallel_reduce(dk: usize, dv: usize) -> bool {
    dk >= 32 && dv > 0 && gdn_recurrent_parallel_reduce_enabled()
}

fn linear_decode_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_LINEAR_DECODE_SINGLE_SUBMIT").is_err())
}

fn linear_decode_argmax_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_LINEAR_ARGMAX_SINGLE_SUBMIT").is_err())
}

fn full_attn_qkv_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_FULL_ATTN_QKV_SINGLE_SUBMIT").is_err())
}

fn gdn_in_proj_single_submit_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_IN_PROJ_SINGLE_SUBMIT").is_err())
}

fn gdn_in_proj_batch_pair_qkv_z_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_PAIR_QKV_Z").is_err())
}

fn gdn_in_proj_batch_row_pair_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_PAIR").is_err())
}

fn gdn_in_proj_batch_row_quad_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_QUAD").is_err())
}

fn gdn_gates_batched_transfers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_GATES_BATCHED_TRANSFERS").is_err())
}

fn gdn_gated_norm_batched_uploads_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_DISABLE_VULKAN_GDN_GATED_NORM_BATCHED_UPLOADS").is_err()
    })
}

fn gdn_chunk_batched_transfers_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_GDN_CHUNK_BATCHED_TRANSFERS").is_err())
}

fn paged_attn_batched_uploads_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_PAGED_ATTN_BATCHED_UPLOADS").is_err())
}

fn prefill_row_pair_matmul_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_PREFILL_ROW_PAIR_MATMUL").is_err())
}

fn use_prefill_row_pair_matmul(batch: usize) -> bool {
    batch >= 8 && prefill_row_pair_matmul_enabled()
}

/// Pre-create the validated built-in compute pipelines on this Vulkan device.
///
/// SPIR-V bytecode is embedded at build time when `glslc` is available. This
/// function fills the per-device pipeline cache so the first live request does
/// not pay RADV pipeline creation latency on the decode path.
pub fn prewarm_builtin_pipelines(vk_device: &VulkanDevice) -> Result<()> {
    let shaders = [
        ("full_attn_qkv_decode", 5usize, 20u32),
        ("full_attn_qkv_decode_bf16w", 5usize, 20u32),
        ("full_attn_qkv_decode_batched", 5usize, 24u32),
        ("full_attn_qkv_decode_batched_bf16w", 5usize, 24u32),
        ("full_attn_qkv_decode_batched_rows4_bf16w", 5usize, 24u32),
        ("gdn_gates", 6usize, 8u32),
        ("gdn_decode_gates_recurrent_rmsnorm", 11, 20),
        ("gdn_in_proj_decode", 6, 24),
        ("gdn_in_proj_decode_bf16w", 6, 24),
        ("gdn_in_proj_decode_batched", 6, 28),
        ("gdn_in_proj_decode_batched_bf16w", 6, 28),
        ("gdn_in_proj_decode_batched_pair_qkv_z_rows2_bf16w", 6, 28),
        ("gdn_in_proj_decode_batched_pair_qkv_z_rows4_bf16w", 6, 28),
        ("gdn_gated_rms_norm", 4, 12),
        ("causal_conv1d", 4, 16),
        ("causal_conv1d_state_advance", 2, 16),
        ("gdn_recurrent_prefill", 7, 24),
        ("gdn_recurrent_step_parallel", 7, 24),
        ("gdn_recurrent_qk_norm_step", 7, 24),
        ("gdn_chunk_prep", 12, 16),
        ("gdn_chunk_scan", 8, 16),
        ("linear_decode", 3, 8),
        ("linear_decode_bf16w", 3, 8),
        ("linear_decode_batched", 3, 12),
        ("linear_decode_batched_bf16w", 3, 12),
        ("linear_decode_batched_rows2", 3, 12),
        ("linear_decode_batched_rows4", 3, 12),
        ("linear_decode_batched_rows4_bf16w", 3, 12),
        ("linear_decode_batched_rows8_bf16w", 3, 12),
        ("linear_decode_argmax_blocks", 4, 12),
        ("linear_decode_argmax_blocks_bf16w", 4, 12),
        ("linear_decode_argmax_reduce", 3, 4),
        ("linear_decode_argmax_batched_blocks", 4, 12),
        ("linear_decode_argmax_batched_blocks_bf16w", 4, 12),
        ("linear_decode_argmax_batched_reduce", 3, 4),
        ("mlp_gate_up_decode", 4, 8),
        ("mlp_gate_up_decode_bf16w", 4, 8),
        ("mlp_gate_up_decode_batched", 4, 12),
        ("mlp_gate_up_decode_batched_bf16w", 4, 12),
        ("mlp_gate_up_decode_batched_rows4_bf16w", 4, 12),
        ("mlp_gate_up_decode_batched_rows8_bf16w", 4, 12),
        ("mlp_gate_up_decode_batched_rows2", 4, 12),
        ("paged_attn_decode_batch", 5, 20),
        ("paged_attn_decode_batch_paged", 6, 24),
    ];

    for (shader_name, total_bindings, push_bytes) in shaders {
        let glsl_path = format!(
            "{}/csrc/shaders/{}.comp",
            env!("CARGO_MANIFEST_DIR"),
            shader_name
        );
        let spirv = crate::pipeline::ShaderPipeline::compile_shader(&glsl_path)
            .with_context(|| format!("compile Vulkan shader {shader_name}"))?;
        vk_device
            .get_or_create_compute_pipeline(&spirv, total_bindings, push_bytes)
            .with_context(|| format!("create Vulkan pipeline {shader_name}"))?;
    }

    Ok(())
}

/// Dispatch a Vulkan compute kernel.
///
/// Manages the full lifecycle: create buffers, upload inputs, dispatch, read back output.
pub fn dispatch_kernel(
    vk_device: &VulkanDevice,
    spirv: &[u8],
    push_constants: &[u32],
    workgroup_count: (u32, u32, u32),
    input_tensors: &[&Tensor],
    output_shape: &[usize],
    output_dtype: DType,
) -> Result<Tensor> {
    // Per-axis dispatch grid limit. Use the actual device limit
    // (typically ≈ 2^31 - 1 on AMD/Strix Halo) rather than the
    // Vulkan spec minimum (65535), so we don't bail on legitimate
    // dispatches that the hardware can handle.
    let limit_x = vk_device.max_compute_work_group_count(0);
    let limit_y = vk_device.max_compute_work_group_count(1);
    let limit_z = vk_device.max_compute_work_group_count(2);
    anyhow::ensure!(
        workgroup_count.0 <= limit_x
            && workgroup_count.1 <= limit_y
            && workgroup_count.2 <= limit_z,
        "dispatch_kernel: workgroup_count {:?} exceeds device per-axis \
         limits ({}, {}, {})",
        workgroup_count,
        limit_x,
        limit_y,
        limit_z
    );
    let device = vk_device.device();
    let queue = vk_device.queue();
    let queue_family_index = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();
    // --- Extract input data (flatten to f32) ---
    let mut input_data: Vec<Vec<u8>> = Vec::with_capacity(input_tensors.len());
    for tensor in input_tensors {
        let (data, _) = extract_tensor_bytes(tensor)?;
        input_data.push(data);
    }

    // --- Create output buffer ---
    let elem_count: usize = output_shape.iter().product();
    let elem_size = match output_dtype {
        DType::F32 => 4,
        DType::BF16 | DType::F16 => 2,
        DType::F64 => 8,
        _ => 4,
    };
    let output_size = (elem_count * elem_size) as u64;
    let output_buffer = VulkanBuffer::create_device_local(device, device_local_mt, output_size)
        .context("failed to create output buffer")?;

    // --- Create input buffers + upload ---
    let mut input_buffers: Vec<VulkanBuffer> = Vec::with_capacity(input_data.len());
    for data in &input_data {
        let buf = VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)
            .context("failed to create input buffer")?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            queue_family_index,
            &buf,
            data,
        )
        .context("failed to upload input data")?;
        input_buffers.push(buf);
    }

    // --- Build combined binding list (inputs first, then output) ---
    let total_bindings = input_buffers.len() + 1;
    tracing::trace!(
        total_bindings,
        inputs = input_tensors.len(),
        "Vulkan dispatch start"
    );
    let mut all_handles: Vec<vk::Buffer> = Vec::with_capacity(total_bindings);
    for buf in &input_buffers {
        all_handles.push(buf.handle());
    }
    all_handles.push(output_buffer.handle());

    // --- Shader module ---
    let spirv_words: &[u32] = bytemuck::cast_slice(spirv);
    let shader_module_info = vk::ShaderModuleCreateInfo::builder()
        .code(spirv_words)
        .build();
    let shader_module = unsafe {
        device
            .create_shader_module(&shader_module_info, None)
            .context("failed to create shader module")?
    };

    // --- Descriptor set layout (STORAGE_BUFFER for all bindings) ---
    let desc_bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..total_bindings as u32)
        .map(|i| {
            vk::DescriptorSetLayoutBinding::builder()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .build()
        })
        .collect();

    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&desc_bindings)
        .build();
    let set_layout = unsafe {
        device
            .create_descriptor_set_layout(&set_layout_info, None)
            .context("failed to create descriptor set layout")?
    };

    // --- Pipeline layout ---
    let push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size((push_constants.len() * 4) as u32)
        .build();
    let pcr = vec![push_constant_range];
    let set_layouts = vec![set_layout];

    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&pcr)
        .build();
    let layout = unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .context("failed to create pipeline layout")?
    };

    // --- Compute pipeline ---
    let stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
        .build();

    let pipeline_info = vk::ComputePipelineCreateInfo::builder()
        .stage(stage_info)
        .layout(layout)
        .build();

    let pipelines = unsafe {
        device
            .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .map_err(|(errs, _)| {
                if !errs.is_empty() {
                    anyhow::anyhow!("failed to create compute pipeline: {:?}", errs[0])
                } else {
                    anyhow::anyhow!("failed to create compute pipeline")
                }
            })?
    };
    let pipeline = pipelines[0];

    // --- Descriptor pool + set (STORAGE_BUFFER) ---
    let pool_size = vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(total_bindings as u32)
        .build();
    let pool_sizes = vec![pool_size];

    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .max_sets(1)
        .pool_sizes(&pool_sizes)
        .build();
    let pool = unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .context("failed to create descriptor pool")?
    };

    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&set_layouts)
        .build();
    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(&alloc_info)
            .context("failed to allocate descriptor sets")?
    };
    let descriptor_set = descriptor_sets[0];

    // --- Descriptor writes using STORAGE_BUFFER (no buffer views needed) ---
    {
        // Build DescriptorBufferInfo entries (must outlive WriteDescriptorSet)
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&buf_handle| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(buf_handle)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();

        let descriptor_write_infos: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, bui)| make_write_descriptor_set_buf(descriptor_set, i as u32, bui))
            .collect();

        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    // --- Command buffer + dispatch ---
    let cmd_pool_info = make_cmd_pool_info(queue_family_index);
    let cmd_pool = unsafe {
        device
            .create_command_pool(&cmd_pool_info, None)
            .context("failed to create command pool")?
    };

    let alloc_info = make_cmd_alloc_info(cmd_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
        .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    let begin_info = make_cmd_begin_info();
    unsafe {
        device
            .begin_command_buffer(cmd, &begin_info)
            .context("failed to begin command buffer")?;

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);

        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );

        let push_const_bytes = bytemuck::cast_slice(push_constants);
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_const_bytes,
        );

        device.cmd_dispatch(cmd, workgroup_count.0, workgroup_count.1, workgroup_count.2);

        // Memory barrier: flush compute writes so readback sees them
        let barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
    }

    // --- Submit + wait ---
    let cmds = vec![cmd];
    let submit_info = make_submit_info(&cmds);
    unsafe {
        device
            .queue_submit(queue, &[submit_info], vk::Fence::null())
            .context("failed to submit compute dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for queue")?;
    }

    // --- Read back output ---
    let output_data = VulkanBuffer::read_back(
        device,
        host_visible_mt,
        queue,
        queue_family_index,
        &output_buffer,
    )
    .context("failed to read back output")?;

    // --- Cleanup (input_buffers and output_buffer dropped here) ---
    drop(input_buffers);
    drop(output_buffer);

    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_shader_module(shader_module, None);
        device.free_command_buffers(cmd_pool, &command_buffers);
        device.destroy_command_pool(cmd_pool, None);
    }
    tracing::trace!("Vulkan dispatch complete");

    // --- Create output tensor ---
    create_tensor_from_data(&output_data, output_shape, output_dtype)
        .context("failed to create output tensor")
}

/// Create a zero-init CommandPoolCreateInfo with fixed sType.
fn make_cmd_pool_info(queue_family_index: u32) -> vk::CommandPoolCreateInfo {
    use std::mem::MaybeUninit;
    use std::ptr::write_bytes;
    let mut info: MaybeUninit<vk::CommandPoolCreateInfo> = MaybeUninit::uninit();
    unsafe {
        write_bytes(info.as_mut_ptr(), 0, 1);
    }
    unsafe {
        let ptr = info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::COMMAND_POOL_CREATE_INFO;
        (*ptr).queue_family_index = queue_family_index;
        (*ptr).flags = vk::CommandPoolCreateFlags::TRANSIENT;
    }
    unsafe { info.assume_init() }
}

/// Create a zero-init CommandBufferAllocateInfo with fixed sType.
fn make_cmd_alloc_info(pool: vk::CommandPool) -> vk::CommandBufferAllocateInfo {
    use std::mem::MaybeUninit;
    use std::ptr::write_bytes;
    let mut info: MaybeUninit<vk::CommandBufferAllocateInfo> = MaybeUninit::uninit();
    unsafe {
        write_bytes(info.as_mut_ptr(), 0, 1);
    }
    unsafe {
        let ptr = info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO;
        (*ptr).command_pool = pool;
        (*ptr).level = vk::CommandBufferLevel::PRIMARY;
        (*ptr).command_buffer_count = 1;
    }
    unsafe { info.assume_init() }
}

/// Create a zero-init CommandBufferBeginInfo with fixed sType.
fn make_cmd_begin_info() -> vk::CommandBufferBeginInfo {
    use std::mem::MaybeUninit;
    use std::ptr::write_bytes;
    let mut info: MaybeUninit<vk::CommandBufferBeginInfo> = MaybeUninit::uninit();
    unsafe {
        write_bytes(info.as_mut_ptr(), 0, 1);
    }
    unsafe {
        let ptr = info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::COMMAND_BUFFER_BEGIN_INFO;
        (*ptr).flags = vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;
        (*ptr).p_inheritance_info = std::ptr::null();
    }
    unsafe { info.assume_init() }
}

/// Create a zero-init SubmitInfo with fixed sType.
fn make_submit_info(cmds: &[vk::CommandBuffer]) -> vk::SubmitInfo {
    use std::mem::MaybeUninit;
    use std::ptr::write_bytes;
    let mut info: MaybeUninit<vk::SubmitInfo> = MaybeUninit::uninit();
    unsafe {
        write_bytes(info.as_mut_ptr(), 0, 1);
    }
    unsafe {
        let ptr = info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::SUBMIT_INFO;
        (*ptr).wait_semaphore_count = 0;
        (*ptr).p_wait_semaphores = std::ptr::null();
        (*ptr).p_wait_dst_stage_mask = std::ptr::null();
        (*ptr).command_buffer_count = cmds.len() as u32;
        (*ptr).p_command_buffers = cmds.as_ptr();
        (*ptr).signal_semaphore_count = 0;
        (*ptr).p_signal_semaphores = std::ptr::null();
    }
    unsafe { info.assume_init() }
}

fn upload_buffers_with_command_pool(
    device: &Arc<ash::Device>,
    host_mem_type: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    uploads: &[(&VulkanBuffer, &[u8])],
) -> Result<()> {
    if uploads.is_empty() {
        return Ok(());
    }

    let mut staging = Vec::with_capacity(uploads.len());
    for (_, data) in uploads {
        let stage = VulkanBuffer::create_host_visible(device, host_mem_type, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &stage, data)?;
        staging.push(stage);
    }

    let alloc_info = make_cmd_alloc_info(command_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
        .context("failed to allocate batched transfer command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin batched transfer command buffer")?;
        for ((dst, data), stage) in uploads.iter().zip(staging.iter()) {
            device.cmd_copy_buffer(
                cmd,
                stage.handle(),
                dst.handle(),
                &[vk::BufferCopy::builder().size(data.len() as u64).build()],
            );
        }
        device
            .end_command_buffer(cmd)
            .context("failed to end batched transfer command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit batched transfer")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for batched transfer")?;
        device.free_command_buffers(command_pool, &command_buffers);
    }

    Ok(())
}

fn read_back_buffers_with_command_pool(
    device: &Arc<ash::Device>,
    host_mem_type: u32,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    buffers: &[&VulkanBuffer],
) -> Result<Vec<Vec<u8>>> {
    if buffers.is_empty() {
        return Ok(Vec::new());
    }

    let mut staging = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        staging.push(VulkanBuffer::create_host_visible(
            device,
            host_mem_type,
            buffer.size(),
        )?);
    }

    let alloc_info = make_cmd_alloc_info(command_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
        .context("failed to allocate batched readback command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin batched readback command buffer")?;
        for (src, stage) in buffers.iter().zip(staging.iter()) {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                stage.handle(),
                &[vk::BufferCopy::builder().size(src.size()).build()],
            );
        }
        device
            .end_command_buffer(cmd)
            .context("failed to end batched readback command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit batched readback")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for batched readback")?;
        device.free_command_buffers(command_pool, &command_buffers);
    }

    staging
        .iter()
        .map(|stage| VulkanBuffer::read_host_visible(device, stage))
        .collect()
}

/// Create a zero-init WriteDescriptorSet for STORAGE_BUFFER with fixed sType.
fn make_write_descriptor_set_buf(
    dst_set: vk::DescriptorSet,
    dst_binding: u32,
    bui: &vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet {
    use std::mem::MaybeUninit;
    use std::ptr::write_bytes;

    let mut info: MaybeUninit<vk::WriteDescriptorSet> = MaybeUninit::uninit();
    unsafe {
        write_bytes(info.as_mut_ptr(), 0, 1);
    }
    unsafe {
        let ptr = info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::WRITE_DESCRIPTOR_SET;
        (*ptr).dst_set = dst_set;
        (*ptr).dst_binding = dst_binding;
        (*ptr).descriptor_count = 1;
        (*ptr).descriptor_type = vk::DescriptorType::STORAGE_BUFFER;
        (*ptr).p_image_info = std::ptr::null();
        (*ptr).p_buffer_info = bui as *const _;
        (*ptr).p_texel_buffer_view = std::ptr::null();
    }
    unsafe { info.assume_init() }
}

/// Create a zero-init MemoryBarrier with fixed sType.
fn make_memory_barrier(src: vk::AccessFlags, dst: vk::AccessFlags) -> vk::MemoryBarrier {
    use std::mem::MaybeUninit;
    use std::ptr::write_bytes;
    let mut info: MaybeUninit<vk::MemoryBarrier> = MaybeUninit::uninit();
    unsafe {
        write_bytes(info.as_mut_ptr(), 0, 1);
    }
    unsafe {
        let ptr = info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::MEMORY_BARRIER;
        (*ptr).src_access_mask = src;
        (*ptr).dst_access_mask = dst;
    }
    unsafe { info.assume_init() }
}

/// Extract raw f32 bytes from a candle-core Tensor.
pub fn extract_tensor_bytes(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let flat = tensor.flatten_all().context("failed to flatten tensor")?;
    let f32_data = flat
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .context("failed to extract f32 data")?;
    Ok((bytemuck::cast_slice(&f32_data).to_vec(), shape))
}

/// Extract raw bf16 weights packed two values per u32 in row-major order.
///
/// Shaders expand each 16-bit lane with `uintBitsToFloat(bits << 16)`, which
/// preserves the exact bf16 value without requiring native shader bf16 support.
/// Public re-export for kiln-model's residency registry — same impl
/// as the private `extract_tensor_packed_bf16_bytes`. Used by the
/// `register_resident_activation` BF16 path to upload bytes in the
/// layout every Vulkan kernel's `load_weight` helper expects.
pub fn extract_tensor_packed_bf16_bytes_pub(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    extract_tensor_packed_bf16_bytes(tensor)
}

fn extract_tensor_packed_bf16_bytes(tensor: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    anyhow::ensure!(
        tensor.dtype() == DType::BF16,
        "packed bf16 upload requires BF16 tensor, got {:?}",
        tensor.dtype()
    );
    let shape: Vec<usize> = tensor.shape().dims().to_vec();
    let flat = tensor.flatten_all().context("failed to flatten tensor")?;
    let bf16_data = flat
        .to_vec1::<bf16>()
        .context("failed to extract bf16 data")?;
    let mut packed = Vec::with_capacity(bf16_data.len().div_ceil(2));
    for pair in bf16_data.chunks(2) {
        let lo = pair[0].to_bits() as u32;
        let hi = pair.get(1).map(|v| v.to_bits() as u32).unwrap_or(0);
        packed.push(lo | (hi << 16));
    }
    Ok((bytemuck::cast_slice(&packed).to_vec(), shape))
}

/// Create a candle-core Tensor from raw bytes.
pub fn create_tensor_from_data(data: &[u8], shape: &[usize], dtype: DType) -> Result<Tensor> {
    let f32_data: &[f32] = bytemuck::cast_slice(data);
    let tensor =
        Tensor::from_vec(f32_data.to_vec(), f32_data.len(), &Device::Cpu)?.reshape(shape)?;

    if dtype == DType::BF16 {
        Ok(tensor.to_dtype(DType::BF16)?)
    } else {
        Ok(tensor)
    }
}

/// Decode a registry-resident `VulkanBuffer` back into a candle CPU
/// Tensor of the requested `shape` and `dtype`.
///
/// Inverse of the encoding choices in
/// `vulkan::register_resident_activation`: BF16 entries are stored as
/// packed bf16 (two bf16 lanes per u32 word, `(hi << 16) | lo`), F32
/// entries are stored as raw f32 bytes. The decoder bit-expands each
/// bf16 lane back to f32 then casts to the target dtype via candle so
/// we don't need a hard dependency on the `half` crate at this layer.
///
/// Used by `VulkanLoraOp::bwd` to read LoRA `A` and `B` weights
/// straight from the registry instead of candle CPU storage —
/// closes the candle-storage staleness gap that the lazy
/// `sync_to_candle` flow opens.
pub fn buffer_to_tensor(
    vk_device: &VulkanDevice,
    buffer: &VulkanBuffer,
    shape: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let bytes = VulkanBuffer::read_back(
        vk_device.device(),
        vk_device.host_visible_mem_type(),
        vk_device.queue(),
        vk_device.queue_family_index(),
        buffer,
    )
    .context("buffer_to_tensor: VulkanBuffer::read_back")?;
    if dtype == DType::BF16 {
        anyhow::ensure!(
            bytes.len() % 2 == 0,
            "buffer_to_tensor BF16: buffer byte count {} is not a multiple of 2",
            bytes.len()
        );
        let elem_count: usize = shape.iter().product();
        let stored = bytes.len() / 2;
        anyhow::ensure!(
            stored >= elem_count,
            "buffer_to_tensor BF16: buffer holds {} bf16 elements, expected at least {} \
             for shape {:?}",
            stored,
            elem_count,
            shape,
        );
        let mut f32_data = Vec::with_capacity(elem_count);
        for i in 0..elem_count {
            let lo = bytes[i * 2] as u32;
            let hi = bytes[i * 2 + 1] as u32;
            let bf16_bits = (hi << 8) | lo;
            f32_data.push(f32::from_bits(bf16_bits << 16));
        }
        Ok(Tensor::from_vec(f32_data, shape, &Device::Cpu)?.to_dtype(DType::BF16)?)
    } else {
        create_tensor_from_data(&bytes, shape, dtype)
    }
}

/// Upload a Candle tensor as contiguous f32 values into a device-local Vulkan buffer.
///
/// This is used by model-level caches for immutable weights so repeated decode
/// steps do not re-upload multi-megabyte projection matrices.
pub fn upload_tensor_f32_buffer(vk_device: &VulkanDevice, tensor: &Tensor) -> Result<VulkanBuffer> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();
    let tensor_f32;
    let tensor = if tensor.dtype() == DType::F32 {
        tensor
    } else {
        tensor_f32 = tensor
            .to_dtype(DType::F32)
            .context("failed to convert cached tensor to f32 for Vulkan upload")?;
        &tensor_f32
    };
    let data = extract_tensor_bytes(tensor)?.0;

    let buffer = VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)
        .context("failed to create cached tensor buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &buffer,
            &data,
        )
        .context("failed to upload cached tensor buffer")?;
    }
    Ok(buffer)
}

/// Upload a BF16 Candle tensor as packed immutable weights into a Vulkan buffer.
///
/// The resulting buffer stores two BF16 values per u32, matching the
/// `*_bf16w.comp` shader variants.
pub fn upload_tensor_bf16_packed_buffer(
    vk_device: &VulkanDevice,
    tensor: &Tensor,
) -> Result<VulkanBuffer> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();
    let data = extract_tensor_packed_bf16_bytes(tensor)?.0;

    let buffer = VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)
        .context("failed to create cached packed bf16 tensor buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &buffer,
            &data,
        )
        .context("failed to upload cached packed bf16 tensor buffer")?;
    }
    Ok(buffer)
}

/// Dispatch the fused single-token GDN input projection kernel with cached weights.
///
/// `x` is `[batch, 1, hidden]` and each weight is already transposed as
/// `[hidden, out_dim]`. The returned tensors are f32 CPU tensors with shapes
/// `[batch, 1, qkv_dim]`, `[batch, 1, z_dim]`, `[batch, 1, a_dim]`, and
/// `[batch, 1, b_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_in_proj_decode_cached(
    vk_device: &VulkanDevice,
    x: &Tensor,
    qkv_weight_t: &VulkanBuffer,
    z_weight_t: &VulkanBuffer,
    a_weight_t: &VulkanBuffer,
    b_weight_t: &VulkanBuffer,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    dispatch_gdn_in_proj_decode_cached_impl(
        vk_device,
        x,
        qkv_weight_t,
        z_weight_t,
        a_weight_t,
        b_weight_t,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_in_proj_decode_cached_bf16_weights(
    vk_device: &VulkanDevice,
    x: &Tensor,
    qkv_weight_t: &VulkanBuffer,
    z_weight_t: &VulkanBuffer,
    a_weight_t: &VulkanBuffer,
    b_weight_t: &VulkanBuffer,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    dispatch_gdn_in_proj_decode_cached_impl(
        vk_device,
        x,
        qkv_weight_t,
        z_weight_t,
        a_weight_t,
        b_weight_t,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_in_proj_decode_cached_impl(
    vk_device: &VulkanDevice,
    x: &Tensor,
    qkv_weight_t: &VulkanBuffer,
    z_weight_t: &VulkanBuffer,
    a_weight_t: &VulkanBuffer,
    b_weight_t: &VulkanBuffer,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    packed_bf16_weights: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_dims = x.dims();
    anyhow::ensure!(
        x_dims.len() == 3 && x_dims[1] == 1 && x_dims[2] == hidden,
        "gdn_in_proj_decode: x shape {:?} does not match [batch, 1, {hidden}]",
        x_dims
    );
    let batch = x_dims[0];
    let profile_stages = profile_vulkan_gdn_in_proj_kernel_stages_enabled();
    let total_start = profile_stages.then(Instant::now);
    let total_out = qkv_dim + z_dim + a_dim + b_dim;
    let pair_qkv_z = batch > 1 && gdn_in_proj_batch_pair_qkv_z_enabled();
    let row_grouping =
        packed_bf16_weights && pair_qkv_z && batch >= 3 && gdn_in_proj_batch_row_pair_enabled();
    let row_group_size = if row_grouping && batch >= 8 && gdn_in_proj_batch_row_quad_enabled() {
        4usize
    } else if row_grouping {
        2usize
    } else {
        1usize
    };
    let dispatch_cols = if pair_qkv_z {
        qkv_dim.div_ceil(2) + z_dim.div_ceil(2) + a_dim + b_dim
    } else {
        total_out
    };
    let single_submit = gdn_in_proj_single_submit_enabled();
    let stage_start = profile_stages.then(Instant::now);
    let x_data = extract_tensor_bytes(x)?.0;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "extract_x",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "gdn_in_proj_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );

    let glsl_path = if batch == 1 {
        if packed_bf16_weights {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode.comp"
            )
        }
    } else {
        if packed_bf16_weights {
            if row_group_size == 4 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_rows4_bf16w.comp"
                )
            } else if row_group_size == 2 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_rows2_bf16w.comp"
                )
            } else if pair_qkv_z {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z_bf16w.comp"
                )
            } else {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/gdn_in_proj_decode_batched_bf16w.comp"
                )
            }
        } else if pair_qkv_z {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_batched_pair_qkv_z.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/gdn_in_proj_decode_batched.comp"
            )
        }
    };
    let stage_start = profile_stages.then(Instant::now);
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "shader",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );
    let mut push_constants = vec![
        hidden as u32,
        qkv_dim as u32,
        z_dim as u32,
        a_dim as u32,
        b_dim as u32,
        total_out as u32,
    ];
    if batch > 1 {
        push_constants.push(batch as u32);
    }
    if single_submit {
        return dispatch_gdn_in_proj_decode_cached_single_submit(
            vk_device,
            qkv_weight_t,
            z_weight_t,
            a_weight_t,
            b_weight_t,
            batch,
            hidden,
            qkv_dim,
            z_dim,
            a_dim,
            b_dim,
            total_out,
            dispatch_cols,
            row_group_size,
            packed_bf16_weights,
            pair_qkv_z,
            profile_stages,
            total_start,
            &spirv,
            &push_constants,
            &x_data,
        );
    }

    let stage_start = profile_stages.then(Instant::now);
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create gdn_in_proj x buffer")?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_x_buffer",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );
    {
        let stage_start = profile_stages.then(Instant::now);
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload gdn_in_proj x buffer")?;
        finish_vulkan_gdn_in_proj_kernel_stage_profile(
            "upload_x",
            batch,
            hidden,
            qkv_dim,
            z_dim,
            a_dim,
            b_dim,
            packed_bf16_weights,
            pair_qkv_z,
            row_group_size,
            single_submit,
            stage_start,
        );
    }

    let stage_start = profile_stages.then(Instant::now);
    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * total_out * 4) as u64)
            .context("failed to create gdn_in_proj output buffer")?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_out_buffer",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );
    let all_handles = vec![
        x_buf.handle(),
        qkv_weight_t.handle(),
        z_weight_t.handle(),
        a_weight_t.handle(),
        b_weight_t.handle(),
        out_buf.handle(),
    ];

    let stage_start = profile_stages.then(Instant::now);
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        if batch == 1 {
            total_out.div_ceil(16) as u32
        } else if row_group_size > 1 {
            (batch.div_ceil(row_group_size) * dispatch_cols.div_ceil(80)) as u32
        } else {
            (batch * dispatch_cols.div_ceil(80)) as u32
        },
    )
    .context("gdn_in_proj_decode kernel failed")?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "dispatch",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );

    let out_data = {
        let stage_start = profile_stages.then(Instant::now);
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back gdn_in_proj output")?;
        finish_vulkan_gdn_in_proj_kernel_stage_profile(
            "readback",
            batch,
            hidden,
            qkv_dim,
            z_dim,
            a_dim,
            b_dim,
            packed_bf16_weights,
            pair_qkv_z,
            row_group_size,
            single_submit,
            stage_start,
        );
        out_data
    };

    let stage_start = profile_stages.then(Instant::now);
    let out = create_gdn_in_proj_tensors_from_data(&out_data, batch, qkv_dim, z_dim, a_dim, b_dim);
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_tensors",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        stage_start,
    );
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "total",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        single_submit,
        total_start,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_in_proj_decode_cached_single_submit(
    vk_device: &VulkanDevice,
    qkv_weight_t: &VulkanBuffer,
    z_weight_t: &VulkanBuffer,
    a_weight_t: &VulkanBuffer,
    b_weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
    total_out: usize,
    dispatch_cols: usize,
    row_group_size: usize,
    packed_bf16_weights: bool,
    pair_qkv_z: bool,
    profile_stages: bool,
    total_start: Option<Instant>,
    spirv: &[u8],
    push_constants: &[u32],
    x_data: &[u8],
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let stage_start = profile_stages.then(Instant::now);
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create gdn_in_proj x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create gdn_in_proj x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, x_data)?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_x_stage_write",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );

    let out_size = (batch * total_out * 4) as u64;
    let stage_start = profile_stages.then(Instant::now);
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create gdn_in_proj output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create gdn_in_proj output staging buffer")?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_out_buffers",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );

    let stage_start = profile_stages.then(Instant::now);
    let all_handles = vec![
        x_buf.handle(),
        qkv_weight_t.handle(),
        z_weight_t.handle(),
        a_weight_t.handle(),
        b_weight_t.handle(),
        out_buf.handle(),
    ];
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate gdn_in_proj descriptor set")?[0]
    };
    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "pipeline_descriptor_setup",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    let stage_start = profile_stages.then(Instant::now);
    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::builder().size(x_data.len() as u64).build()],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(
            cmd,
            if batch == 1 {
                total_out.div_ceil(16) as u32
            } else if row_group_size > 1 {
                (batch.div_ceil(row_group_size) * dispatch_cols.div_ceil(80)) as u32
            } else {
                (batch * dispatch_cols.div_ceil(80)) as u32
            },
            1,
            1,
        );
        let output_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[output_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size(out_size).build()],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn_in_proj single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn_in_proj single-submit dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "record_submit_wait",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );

    let stage_start = profile_stages.then(Instant::now);
    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "read_host_visible",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );
    let stage_start = profile_stages.then(Instant::now);
    let out = create_gdn_in_proj_tensors_from_data(&out_data, batch, qkv_dim, z_dim, a_dim, b_dim);
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "create_tensors",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        stage_start,
    );
    finish_vulkan_gdn_in_proj_kernel_stage_profile(
        "total",
        batch,
        hidden,
        qkv_dim,
        z_dim,
        a_dim,
        b_dim,
        packed_bf16_weights,
        pair_qkv_z,
        row_group_size,
        true,
        total_start,
    );
    out
}

fn create_gdn_in_proj_tensors_from_data(
    out_data: &[u8],
    batch: usize,
    qkv_dim: usize,
    z_dim: usize,
    a_dim: usize,
    b_dim: usize,
) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let mut offset = 0usize;
    let mut take = |len: usize, shape: &[usize]| -> Result<Tensor> {
        let byte_len = batch * len * 4;
        let end = offset + byte_len;
        anyhow::ensure!(
            end <= out_data.len(),
            "gdn_in_proj_decode output slice exceeds readback buffer"
        );
        let tensor = create_tensor_from_data(&out_data[offset..end], shape, DType::F32)?;
        offset = end;
        Ok(tensor)
    };

    let qkv = take(qkv_dim, &[batch, 1, qkv_dim])?;
    let z = take(z_dim, &[batch, 1, z_dim])?;
    let a = take(a_dim, &[batch, 1, a_dim])?;
    let b = take(b_dim, &[batch, 1, b_dim])?;
    Ok((qkv, z, a, b))
}

/// Dispatch a cached single-token linear projection.
///
/// `x` is `[batch, 1, hidden]` and `weight_t` is `[hidden, out_dim]`. The
/// returned tensor is an f32 CPU tensor shaped `[batch, 1, out_dim]`.
pub fn dispatch_linear_decode_cached(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<Tensor> {
    dispatch_linear_decode_cached_impl(vk_device, x, weight_t, batch, hidden, out_dim, false)
}

pub fn dispatch_linear_decode_cached_bf16_weights(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<Tensor> {
    dispatch_linear_decode_cached_impl(vk_device, x, weight_t, batch, hidden, out_dim, true)
}

/// Variant of [`dispatch_linear_decode_cached_bf16_weights`] that takes a
/// SLICE of a larger weight buffer.
///
/// `weight_buffer` holds a row-major bf16-packed `[hidden, full_out_dim]`
/// matrix. This function dispatches the matmul against the column slice
/// `weight_buffer[:, weight_offset .. weight_offset + out_dim]` without
/// requiring a fresh upload of the slice — the same buffer can be reused
/// across many chunked dispatches.
///
/// `weight_offset` is in bf16 elements (i.e., the column index in the
/// original matrix). Output shape is `[batch, 1, out_dim]`.
///
/// Used by the FLCE chunked-head loop and by VulkanLinearOp's backward
/// path so a once-uploaded weight buffer can serve many chunked /
/// transposed dispatches.
pub fn dispatch_linear_decode_cached_bf16_weights_offset(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_buffer: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    weight_offset: usize,
    full_out_dim: usize,
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_data = extract_tensor_bytes(x)?.0;
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "linear_decode_offset: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );
    anyhow::ensure!(
        weight_offset + out_dim <= full_out_dim,
        "weight_offset({}) + out_dim({}) overflows full_out_dim({})",
        weight_offset,
        out_dim,
        full_out_dim,
    );

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode_offset x buffer")?;
    let out_size = (batch * out_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create linear_decode_offset output buffer")?;

    let all_handles = vec![x_buf.handle(), weight_buffer.handle(), out_buf.handle()];
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_batched_offset_bf16w.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [
        hidden as u32,
        out_dim as u32,
        batch as u32,
        weight_offset as u32,
        full_out_dim as u32,
    ];
    let workgroups = (batch * out_dim.div_ceil(32)) as u32;

    let out_data = if linear_decode_single_submit_enabled() {
        run_compute_pipeline_with_transfer_readback(
            vk_device,
            &x_buf,
            &x_data,
            &out_buf,
            out_size,
            &spirv,
            &all_handles,
            &push_constants,
            workgroups,
        )
        .context("linear_decode_batched_offset_bf16w single-submit kernel failed")?
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                &x_data,
            )
            .context("failed to upload linear_decode_offset x buffer")?;
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            workgroups,
        )
        .context("linear_decode_batched_offset_bf16w kernel failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back linear_decode_offset output")?
    };
    create_tensor_from_data(&out_data, &[batch, 1, out_dim], DType::F32)
}

/// Qwen3.5-style RMSNorm forward: `(1 + weight) * x * rsqrt(mean(x^2) + eps)`.
///
/// `x` is `[..., hidden]` F32; `weight` is `[hidden]` F32. Returns a
/// freshly-allocated F32 tensor with the same shape as `x`. The kernel
/// tiles one row per workgroup; the leading dims of `x` are flattened
/// to the row count.
///
/// Used by the Vulkan training path to replace the candle CPU
/// `rms_norm_fallback` per-layer cost (allocates `x_f32`, `variance`,
/// `rms_inv`, `normed`, `w_plus_one`, then casts back). At T=2500 with
/// ~64 RMSNorm calls per forward this was a substantial CPU contributor.
pub fn dispatch_qwen_rmsnorm_forward(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    anyhow::ensure!(
        x.dtype() == DType::F32,
        "qwen_rmsnorm_forward: x must be F32, got {:?}",
        x.dtype()
    );
    anyhow::ensure!(
        weight.dtype() == DType::F32,
        "qwen_rmsnorm_forward: weight must be F32, got {:?}",
        weight.dtype()
    );
    let dims = x.shape().dims().to_vec();
    let hidden = *dims.last().context("qwen_rmsnorm_forward: x has no dims")?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    anyhow::ensure!(
        weight.dims() == [hidden],
        "qwen_rmsnorm_forward: weight shape {:?} does not match hidden {}",
        weight.dims(),
        hidden,
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_data = extract_tensor_bytes(x)?.0;
    anyhow::ensure!(
        x_data.len() == rows * hidden * 4,
        "qwen_rmsnorm_forward: x buffer has {} bytes, expected {}",
        x_data.len(),
        rows * hidden * 4
    );
    let weight_data = extract_tensor_bytes(weight)?.0;
    anyhow::ensure!(
        weight_data.len() == hidden * 4,
        "qwen_rmsnorm_forward: weight buffer has {} bytes, expected {}",
        weight_data.len(),
        hidden * 4
    );

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("qwen_rmsnorm_forward: create x buffer")?;
    let weight_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)
            .context("qwen_rmsnorm_forward: create weight buffer")?;
    let out_size = x_data.len() as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("qwen_rmsnorm_forward: create out buffer")?;

    let all_handles = vec![x_buf.handle(), weight_buf.handle(), out_buf.handle()];
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_forward.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    // Push constants: rows, hidden, eps. eps is f32 transmuted to u32 bits.
    let push_constants: [u32; 3] = [rows as u32, hidden as u32, eps.to_bits()];
    let workgroups = rows as u32;

    let out_data = if qwen_rmsnorm_single_submit_enabled() {
        run_compute_pipeline_with_transfers_readback(
            vk_device,
            &[(&x_buf, &x_data), (&weight_buf, &weight_data)],
            &out_buf,
            out_size,
            &spirv,
            &all_handles,
            &push_constants,
            workgroups,
        )
        .context("qwen_rmsnorm_forward: single-submit dispatch")?
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                &x_data,
            )
            .context("qwen_rmsnorm_forward: upload x")?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &weight_buf,
                &weight_data,
            )
            .context("qwen_rmsnorm_forward: upload weight")?;
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            workgroups,
        )
        .context("qwen_rmsnorm_forward: kernel dispatch")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("qwen_rmsnorm_forward: read back out")?
    };
    create_tensor_from_data(&out_data, &dims, DType::F32)
}

/// Qwen3.5-style RMSNorm backward.
///
/// Given the forward inputs (`x`, `weight`, `eps`) and the gradient of
/// the loss w.r.t. the forward output (`grad_y`), returns `dL/dx` with
/// the same shape as `x`. `dL/dw` is intentionally NOT computed — the
/// Qwen3.5 base RMSNorm weights are frozen during LoRA training.
///
/// All tensors are F32 row-major. Used by the Vulkan training path
/// (RmsNormCustomOp1) to backprop without materializing the chain of
/// candle intermediates that would otherwise dominate the per-layer
/// backward cost on long-context training.
pub fn dispatch_qwen_rmsnorm_backward(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight: &Tensor,
    grad_y: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    anyhow::ensure!(
        x.dtype() == DType::F32,
        "qwen_rmsnorm_backward: x must be F32, got {:?}",
        x.dtype()
    );
    anyhow::ensure!(
        weight.dtype() == DType::F32,
        "qwen_rmsnorm_backward: weight must be F32, got {:?}",
        weight.dtype()
    );
    anyhow::ensure!(
        grad_y.dtype() == DType::F32,
        "qwen_rmsnorm_backward: grad_y must be F32, got {:?}",
        grad_y.dtype()
    );
    anyhow::ensure!(
        x.dims() == grad_y.dims(),
        "qwen_rmsnorm_backward: x dims {:?} != grad_y dims {:?}",
        x.dims(),
        grad_y.dims()
    );

    let dims = x.shape().dims().to_vec();
    let hidden = *dims
        .last()
        .context("qwen_rmsnorm_backward: x has no dims")?;
    let rows: usize = dims[..dims.len() - 1].iter().product();
    anyhow::ensure!(
        weight.dims() == [hidden],
        "qwen_rmsnorm_backward: weight shape {:?} does not match hidden {}",
        weight.dims(),
        hidden,
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_data = extract_tensor_bytes(x)?.0;
    let weight_data = extract_tensor_bytes(weight)?.0;
    let grad_y_data = extract_tensor_bytes(grad_y)?.0;
    let out_len = x_data.len();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("qwen_rmsnorm_backward: create x buffer")?;
    let weight_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)
            .context("qwen_rmsnorm_backward: create weight buffer")?;
    let grad_y_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, grad_y_data.len() as u64)
            .context("qwen_rmsnorm_backward: create grad_y buffer")?;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_len as u64)
        .context("qwen_rmsnorm_backward: create out buffer")?;

    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("qwen_rmsnorm_backward: upload x")?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &weight_buf,
            &weight_data,
        )
        .context("qwen_rmsnorm_backward: upload weight")?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &grad_y_buf,
            &grad_y_data,
        )
        .context("qwen_rmsnorm_backward: upload grad_y")?;
    }

    let all_handles = vec![
        x_buf.handle(),
        weight_buf.handle(),
        grad_y_buf.handle(),
        out_buf.handle(),
    ];
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/qwen_rmsnorm_backward.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 3] = [rows as u32, hidden as u32, eps.to_bits()];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        rows as u32,
    )
    .context("qwen_rmsnorm_backward: kernel dispatch")?;

    let out_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("qwen_rmsnorm_backward: read back grad_x")?
    };
    create_tensor_from_data(&out_data, &dims, DType::F32)
}

/// Matmul against the TRANSPOSE of a bf16-packed weight buffer.
///
/// `weight_buffer` holds a row-major bf16-packed matrix of shape
/// `[forward_k, forward_n]` (the same buffer the forward kernel
/// dispatches against). This function dispatches:
///
/// `out[batch, n_dim] = x[batch, k_dim] * W.T`
///
/// where `W.T` has shape `[forward_n, forward_k]` and `k_dim = forward_n`,
/// `n_dim = forward_k`. Used by VulkanLinearOp::bwd to compute
/// `dx = dy @ weight_t.T` against the same buffer the forward dispatch
/// uploaded — no separate upload of the transposed weight needed.
///
/// Output shape is `[batch, 1, n_dim]`.
pub fn dispatch_linear_decode_cached_bf16_weights_transposed(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_buffer: &VulkanBuffer,
    batch: usize,
    k_dim: usize,
    n_dim: usize,
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_data = extract_tensor_bytes(x)?.0;
    anyhow::ensure!(
        x_data.len() == batch * k_dim * 4,
        "linear_decode_transposed: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * k_dim * 4
    );

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode_transposed x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload linear_decode_transposed x buffer")?;
    }

    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * n_dim * 4) as u64)
            .context("failed to create linear_decode_transposed output buffer")?;

    let all_handles = vec![x_buf.handle(), weight_buffer.handle(), out_buf.handle()];
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_batched_transposed_bf16w.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 3] = [k_dim as u32, n_dim as u32, batch as u32];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        (batch * n_dim.div_ceil(32)) as u32,
    )
    .context("linear_decode_batched_transposed_bf16w kernel failed")?;

    let out_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back linear_decode_transposed output")?
    };
    create_tensor_from_data(&out_data, &[batch, 1, n_dim], DType::F32)
}

fn dispatch_linear_decode_cached_impl(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    packed_bf16_weights: bool,
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_data = extract_tensor_bytes(x)?.0;
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "linear_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );
    if linear_decode_single_submit_enabled() {
        return dispatch_linear_decode_cached_single_submit(
            vk_device,
            weight_t,
            batch,
            hidden,
            out_dim,
            &x_data,
            packed_bf16_weights,
        );
    }

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload linear_decode x buffer")?;
    }

    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * out_dim * 4) as u64)
            .context("failed to create linear_decode output buffer")?;

    let all_handles = vec![x_buf.handle(), weight_t.handle(), out_buf.handle()];
    if batch == 1 {
        let glsl_path = if packed_bf16_weights {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode.comp"
            )
        };
        let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
        let push_constants: [u32; 2] = [hidden as u32, out_dim as u32];
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            out_dim.div_ceil(16) as u32,
        )
        .context("linear_decode kernel failed")?;
    } else {
        let rows4 = packed_bf16_weights && batch >= 32 && linear_decode_bf16w_rows4_enabled();
        let glsl_path = if packed_bf16_weights {
            if rows4 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_rows4_bf16w.comp"
                )
            } else {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_bf16w.comp"
                )
            }
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched.comp"
            )
        };
        let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
        let push_constants: [u32; 3] = [hidden as u32, out_dim as u32, batch as u32];
        let workgroups = if rows4 {
            (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32
        } else {
            (batch * out_dim.div_ceil(32)) as u32
        };
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            workgroups,
        )
        .context("linear_decode_batched kernel failed")?;
    }

    let out_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back linear_decode output")?
    };
    create_tensor_from_data(&out_data, &[batch, 1, out_dim], DType::F32)
}

fn dispatch_linear_decode_cached_single_submit(
    vk_device: &VulkanDevice,
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    x_data: &[u8],
    packed_bf16_weights: bool,
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create linear_decode x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, x_data)?;

    let out_size = (batch * out_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create linear_decode output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create linear_decode output staging buffer")?;

    let (spirv, push_constants, workgroup_count): (Vec<u8>, Vec<u32>, u32) = if batch == 1 {
        let glsl_path = if packed_bf16_weights {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode.comp"
            )
        };
        (
            crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?,
            vec![hidden as u32, out_dim as u32],
            out_dim.div_ceil(16) as u32,
        )
    } else {
        let rows4 = packed_bf16_weights && batch >= 32 && linear_decode_bf16w_rows4_enabled();
        let glsl_path = if packed_bf16_weights {
            if rows4 {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_rows4_bf16w.comp"
                )
            } else {
                concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/csrc/shaders/linear_decode_batched_bf16w.comp"
                )
            }
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched.comp"
            )
        };
        let workgroups = if rows4 {
            (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32
        } else {
            (batch * out_dim.div_ceil(32)) as u32
        };
        (
            crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?,
            vec![hidden as u32, out_dim as u32, batch as u32],
            workgroups,
        )
    };

    let all_handles = vec![x_buf.handle(), weight_t.handle(), out_buf.handle()];
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        &spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate descriptor sets")?[0]
    };
    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::builder().size(x_data.len() as u64).build()],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);
        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size(out_size).build()],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit linear_decode single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for linear_decode single-submit dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    create_tensor_from_data(&out_data, &[batch, 1, out_dim], DType::F32)
}

/// Dispatch a single-token transposed linear projection and return argmax.
///
/// This is intended for greedy LM-head decode: the full vocab logits stay on
/// the Vulkan device and only the winning token id is read back.
pub fn dispatch_linear_decode_argmax_cached(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    hidden: usize,
    out_dim: usize,
) -> Result<u32> {
    dispatch_linear_decode_argmax_cached_impl(vk_device, x, weight_t, hidden, out_dim, false)
}

pub fn dispatch_linear_decode_argmax_cached_bf16_weights(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    hidden: usize,
    out_dim: usize,
) -> Result<u32> {
    dispatch_linear_decode_argmax_cached_impl(vk_device, x, weight_t, hidden, out_dim, true)
}

fn dispatch_linear_decode_argmax_cached_impl(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    hidden: usize,
    out_dim: usize,
    packed_bf16_weights: bool,
) -> Result<u32> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(out_dim > 0, "linear argmax: out_dim must be nonzero");
    let x_data = extract_tensor_bytes(x)?.0;
    anyhow::ensure!(
        x_data.len() == hidden * 4,
        "linear argmax: x buffer has {} bytes, expected {}",
        x_data.len(),
        hidden * 4
    );
    if linear_decode_argmax_single_submit_enabled() {
        return dispatch_linear_decode_argmax_cached_single_submit(
            vk_device,
            weight_t,
            hidden,
            out_dim,
            &x_data,
            packed_bf16_weights,
        );
    }

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear argmax x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload linear argmax x buffer")?;
    }

    let block_count = out_dim.div_ceil(16);
    let block_score_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (block_count * 4) as u64)
            .context("failed to create linear argmax block score buffer")?;
    let block_index_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (block_count * 4) as u64)
            .context("failed to create linear argmax block index buffer")?;
    let out_index_buf = VulkanBuffer::create_device_local(device, device_local_mt, 4)
        .context("failed to create linear argmax output index buffer")?;

    let blocks_glsl = if packed_bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_blocks_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_blocks.comp"
        )
    };
    let blocks_spirv = crate::pipeline::ShaderPipeline::compile_shader(blocks_glsl)?;
    let block_push: [u32; 3] = [hidden as u32, out_dim as u32, block_count as u32];
    let block_handles = vec![
        x_buf.handle(),
        weight_t.handle(),
        block_score_buf.handle(),
        block_index_buf.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &blocks_spirv,
        &block_handles,
        block_handles.len(),
        &block_push,
        block_count as u32,
    )
    .context("linear_decode_argmax block kernel failed")?;

    let reduce_glsl = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_argmax_reduce.comp"
    );
    let reduce_spirv = crate::pipeline::ShaderPipeline::compile_shader(reduce_glsl)?;
    let reduce_push: [u32; 1] = [block_count as u32];
    let reduce_handles = vec![
        block_score_buf.handle(),
        block_index_buf.handle(),
        out_index_buf.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &reduce_spirv,
        &reduce_handles,
        reduce_handles.len(),
        &reduce_push,
        1,
    )
    .context("linear_decode_argmax reduce kernel failed")?;

    let out_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_index_buf,
        )
        .context("failed to read back linear argmax output index")?
    };
    let indices: &[u32] = bytemuck::cast_slice(&out_data);
    indices
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("linear argmax readback was empty"))
}

fn dispatch_linear_decode_argmax_cached_single_submit(
    vk_device: &VulkanDevice,
    weight_t: &VulkanBuffer,
    hidden: usize,
    out_dim: usize,
    x_data: &[u8],
    packed_bf16_weights: bool,
) -> Result<u32> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear argmax x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create linear argmax x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, x_data)?;

    let block_count = out_dim.div_ceil(16);
    let block_score_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (block_count * 4) as u64)
            .context("failed to create linear argmax block score buffer")?;
    let block_index_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (block_count * 4) as u64)
            .context("failed to create linear argmax block index buffer")?;
    let out_index_buf = VulkanBuffer::create_device_local(device, device_local_mt, 4)
        .context("failed to create linear argmax output index buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, 4)
        .context("failed to create linear argmax output staging buffer")?;

    let blocks_glsl = if packed_bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_blocks_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_blocks.comp"
        )
    };
    let blocks_spirv = crate::pipeline::ShaderPipeline::compile_shader(blocks_glsl)?;
    let block_push: [u32; 3] = [hidden as u32, out_dim as u32, block_count as u32];
    let block_handles = vec![
        x_buf.handle(),
        weight_t.handle(),
        block_score_buf.handle(),
        block_index_buf.handle(),
    ];
    let (block_set_layout, block_layout, block_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            &blocks_spirv,
            block_handles.len(),
            (block_push.len() * 4) as u32,
        )?;

    let reduce_glsl = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_argmax_reduce.comp"
    );
    let reduce_spirv = crate::pipeline::ShaderPipeline::compile_shader(reduce_glsl)?;
    let reduce_push: [u32; 1] = [block_count as u32];
    let reduce_handles = vec![
        block_score_buf.handle(),
        block_index_buf.handle(),
        out_index_buf.handle(),
    ];
    let (reduce_set_layout, reduce_layout, reduce_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            &reduce_spirv,
            reduce_handles.len(),
            (reduce_push.len() * 4) as u32,
        )?;

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let block_set_layouts = vec![block_set_layout];
    let block_descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&block_set_layouts)
                    .build(),
            )
            .context("failed to allocate linear argmax block descriptor set")?[0]
    };
    let block_buf_infos: Vec<vk::DescriptorBufferInfo> = block_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let block_descriptor_writes: Vec<vk::WriteDescriptorSet> = block_buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(block_descriptor_set, i as u32, info))
        .collect();

    let reduce_set_layouts = vec![reduce_set_layout];
    let reduce_descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&reduce_set_layouts)
                    .build(),
            )
            .context("failed to allocate linear argmax reduce descriptor set")?[0]
    };
    let reduce_buf_infos: Vec<vk::DescriptorBufferInfo> = reduce_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let reduce_descriptor_writes: Vec<vk::WriteDescriptorSet> = reduce_buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(reduce_descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&block_descriptor_writes, &[]);
        device.update_descriptor_sets(&reduce_descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::builder().size(x_data.len() as u64).build()],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, block_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            block_layout,
            0,
            &[block_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            block_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&block_push),
        );
        device.cmd_dispatch(cmd, block_count as u32, 1, 1);

        let block_barrier =
            make_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[block_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, reduce_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            reduce_layout,
            0,
            &[reduce_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            reduce_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&reduce_push),
        );
        device.cmd_dispatch(cmd, 1, 1, 1);

        let output_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[output_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_index_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size(4).build()],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit linear argmax single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for linear argmax single-submit dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    let indices: &[u32] = bytemuck::cast_slice(&out_data);
    indices
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("linear argmax readback was empty"))
}

/// Single-token transposed linear projection + full Qwen3.5 stochastic
/// sampling, fully fused on the Vulkan device. **Returns only the 4-byte
/// sampled token id — the full-vocab logits never leave GPU memory.**
///
/// Pipeline (all on-device):
/// 1. lm_head matmul → `logits` device buffer of size `[out_dim]`.
/// 2. (Optional) `apply_token_penalties` scatter applies repetition,
///    presence, and frequency penalties at history token indices.
/// 3. `topk_sample` fused kernel does temperature + top-k + softmax +
///    min-p + top-p + seeded categorical sample. Writes 1 u32 token.
/// 4. Read back 4 bytes.
///
/// This is the Vulkan equivalent of CUDA/Metal's on-device sampling
/// path. Replaces the legacy "linear_decode + full vocab readback +
/// host sampler" flow for non-greedy decode steps.
///
/// `top_k` must be ≤ `TOPK_SAMPLE_KERNEL_K_MAX` (= 64). Callers should
/// fall back to the legacy host path for larger top_k requests.
pub const TOPK_SAMPLE_KERNEL_K_MAX: u32 = 64;

#[allow(clippy::too_many_arguments)]
pub fn dispatch_linear_decode_sample(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    packed_bf16_weights: bool,
    hidden: usize,
    out_dim: usize,
    history_indices: &[u32],
    history_counts: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
    frequency_penalty: f32,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    seed: u64,
) -> Result<u32> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(out_dim > 0, "linear_decode_sample: out_dim must be nonzero");
    anyhow::ensure!(
        top_k > 0 && top_k <= TOPK_SAMPLE_KERNEL_K_MAX,
        "linear_decode_sample: top_k {top_k} out of range (1..={})",
        TOPK_SAMPLE_KERNEL_K_MAX
    );
    anyhow::ensure!(
        history_indices.len() == history_counts.len(),
        "linear_decode_sample: history indices/counts length mismatch ({} vs {})",
        history_indices.len(),
        history_counts.len()
    );
    let x_data = extract_tensor_bytes(x)?.0;
    anyhow::ensure!(
        x_data.len() == hidden * 4,
        "linear_decode_sample: x buffer has {} bytes, expected {}",
        x_data.len(),
        hidden * 4
    );

    // ---- Allocate the device-local buffers ----
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create linear_decode_sample x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload linear_decode_sample x buffer")?;
    }

    // Logits buffer is `[out_dim]` f32. Stays on device for the entire
    // pipeline — never copied back to host.
    let logits_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (out_dim * 4) as u64)
            .context("failed to create linear_decode_sample logits buffer")?;

    // ---- Step 1: lm_head matmul ----
    let lm_glsl = if packed_bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode.comp"
        )
    };
    let lm_spirv = crate::pipeline::ShaderPipeline::compile_shader(lm_glsl)?;
    let lm_push: [u32; 2] = [hidden as u32, out_dim as u32];
    let lm_handles = vec![x_buf.handle(), weight_t.handle(), logits_buf.handle()];
    run_compute_pipeline(
        vk_device,
        &lm_spirv,
        &lm_handles,
        lm_handles.len(),
        &lm_push,
        out_dim.div_ceil(16) as u32,
    )
    .context("linear_decode_sample: lm_head dispatch failed")?;

    // ---- Step 2: (optional) apply_token_penalties scatter ----
    let penalties_active = !history_indices.is_empty()
        && ((repetition_penalty.is_finite() && (repetition_penalty - 1.0).abs() > f32::EPSILON)
            || (presence_penalty.is_finite() && presence_penalty != 0.0)
            || (frequency_penalty.is_finite() && frequency_penalty != 0.0));
    let _history_idx_buf;
    let _history_cnt_buf;
    if penalties_active {
        let n_unique = history_indices.len() as u32;
        let idx_buf = VulkanBuffer::create_device_local(
            device,
            device_local_mt,
            (history_indices.len() * 4) as u64,
        )
        .context("failed to create penalty history-index buffer")?;
        let cnt_buf = VulkanBuffer::create_device_local(
            device,
            device_local_mt,
            (history_counts.len() * 4) as u64,
        )
        .context("failed to create penalty history-count buffer")?;
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &idx_buf,
                bytemuck::cast_slice(history_indices),
            )
            .context("failed to upload penalty history-index buffer")?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &cnt_buf,
                bytemuck::cast_slice(history_counts),
            )
            .context("failed to upload penalty history-count buffer")?;
        }

        let pen_glsl = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/apply_token_penalties.comp"
        );
        let pen_spirv = crate::pipeline::ShaderPipeline::compile_shader(pen_glsl)?;
        // Push constants: u32 n_unique, u32 vocab_size, f32 rep, f32 presence, f32 frequency.
        let pen_push: [u32; 5] = [
            n_unique,
            out_dim as u32,
            repetition_penalty.to_bits(),
            presence_penalty.to_bits(),
            frequency_penalty.to_bits(),
        ];
        let pen_handles = vec![logits_buf.handle(), idx_buf.handle(), cnt_buf.handle()];
        run_compute_pipeline(
            vk_device,
            &pen_spirv,
            &pen_handles,
            pen_handles.len(),
            &pen_push,
            n_unique.div_ceil(64),
        )
        .context("linear_decode_sample: apply_token_penalties dispatch failed")?;
        // Keep buffers alive until the queue idles (run_compute_pipeline waits inside).
        _history_idx_buf = Some(idx_buf);
        _history_cnt_buf = Some(cnt_buf);
    } else {
        _history_idx_buf = None;
        _history_cnt_buf = None;
    }

    // ---- Step 3: fused topk_sample → 4-byte token ----
    let out_token_buf = VulkanBuffer::create_device_local(device, device_local_mt, 4)
        .context("failed to create linear_decode_sample out-token buffer")?;
    let sample_glsl = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/topk_sample.comp"
    );
    let sample_spirv = crate::pipeline::ShaderPipeline::compile_shader(sample_glsl)?;
    let seed_lo = (seed & 0xFFFF_FFFF) as u32;
    let seed_hi = (seed >> 32) as u32;
    // Push constants: u32 vocab_size, u32 top_k, f32 temperature, f32 top_p, f32 min_p, u32 seed_lo, u32 seed_hi
    let sample_push: [u32; 7] = [
        out_dim as u32,
        top_k,
        temperature.to_bits(),
        top_p.to_bits(),
        min_p.to_bits(),
        seed_lo,
        seed_hi,
    ];
    let sample_handles = vec![logits_buf.handle(), out_token_buf.handle()];
    // The fused topk_sample shader is a SINGLE workgroup pass (its own
    // tree reduction is inside the shader). Always dispatch x=1.
    run_compute_pipeline(
        vk_device,
        &sample_spirv,
        &sample_handles,
        sample_handles.len(),
        &sample_push,
        1,
    )
    .context("linear_decode_sample: topk_sample dispatch failed")?;

    // ---- Step 4: read back the 4-byte token ----
    let out_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_token_buf,
        )
        .context("failed to read back linear_decode_sample token")?
    };
    let tokens: &[u32] = bytemuck::cast_slice(&out_data);
    tokens
        .first()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("linear_decode_sample readback was empty"))
}

/// Dispatch a batched single-token transposed linear projection and return one
/// argmax token per batch row.
///
/// This is intended for greedy batched LM-head decode. It keeps the vocab
/// logits on the Vulkan device and reads back only `[batch]` token ids.
pub fn dispatch_linear_decode_argmax_batched_cached(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<Vec<u32>> {
    dispatch_linear_decode_argmax_batched_cached_impl(
        vk_device, x, weight_t, batch, hidden, out_dim, false,
    )
}

pub fn dispatch_linear_decode_argmax_batched_cached_bf16_weights(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
) -> Result<Vec<u32>> {
    dispatch_linear_decode_argmax_batched_cached_impl(
        vk_device, x, weight_t, batch, hidden, out_dim, true,
    )
}

fn dispatch_linear_decode_argmax_batched_cached_impl(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    out_dim: usize,
    packed_bf16_weights: bool,
) -> Result<Vec<u32>> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(batch > 0, "batched linear argmax: batch must be nonzero");
    anyhow::ensure!(
        out_dim > 0,
        "batched linear argmax: out_dim must be nonzero"
    );
    let x_data = extract_tensor_bytes(x)?.0;
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "batched linear argmax: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create batched linear argmax x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create batched linear argmax x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, &x_data)?;

    let block_count = out_dim.div_ceil(64);
    let total_blocks = batch * block_count;
    let block_score_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (total_blocks * 4) as u64)
            .context("failed to create batched linear argmax block score buffer")?;
    let block_index_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (total_blocks * 4) as u64)
            .context("failed to create batched linear argmax block index buffer")?;
    let out_index_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * 4) as u64)
            .context("failed to create batched linear argmax output index buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, (batch * 4) as u64)
        .context("failed to create batched linear argmax output staging buffer")?;

    let blocks_glsl = if packed_bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_batched_blocks_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_argmax_batched_blocks.comp"
        )
    };
    let blocks_spirv = crate::pipeline::ShaderPipeline::compile_shader(blocks_glsl)?;
    let block_push: [u32; 3] = [hidden as u32, out_dim as u32, block_count as u32];
    let block_handles = vec![
        x_buf.handle(),
        weight_t.handle(),
        block_score_buf.handle(),
        block_index_buf.handle(),
    ];
    let (block_set_layout, block_layout, block_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            &blocks_spirv,
            block_handles.len(),
            (block_push.len() * 4) as u32,
        )?;

    let reduce_glsl = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/linear_decode_argmax_batched_reduce.comp"
    );
    let reduce_spirv = crate::pipeline::ShaderPipeline::compile_shader(reduce_glsl)?;
    let reduce_push: [u32; 1] = [block_count as u32];
    let reduce_handles = vec![
        block_score_buf.handle(),
        block_index_buf.handle(),
        out_index_buf.handle(),
    ];
    let (reduce_set_layout, reduce_layout, reduce_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            &reduce_spirv,
            reduce_handles.len(),
            (reduce_push.len() * 4) as u32,
        )?;

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let block_set_layouts = vec![block_set_layout];
    let block_descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&block_set_layouts)
                    .build(),
            )
            .context("failed to allocate batched linear argmax block descriptor set")?[0]
    };
    let block_buf_infos: Vec<vk::DescriptorBufferInfo> = block_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let block_descriptor_writes: Vec<vk::WriteDescriptorSet> = block_buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(block_descriptor_set, i as u32, info))
        .collect();

    let reduce_set_layouts = vec![reduce_set_layout];
    let reduce_descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&reduce_set_layouts)
                    .build(),
            )
            .context("failed to allocate batched linear argmax reduce descriptor set")?[0]
    };
    let reduce_buf_infos: Vec<vk::DescriptorBufferInfo> = reduce_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let reduce_descriptor_writes: Vec<vk::WriteDescriptorSet> = reduce_buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(reduce_descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&block_descriptor_writes, &[]);
        device.update_descriptor_sets(&reduce_descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::builder().size(x_data.len() as u64).build()],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, block_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            block_layout,
            0,
            &[block_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            block_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&block_push),
        );
        device.cmd_dispatch(cmd, total_blocks as u32, 1, 1);

        let block_barrier =
            make_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[block_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, reduce_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            reduce_layout,
            0,
            &[reduce_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            reduce_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&reduce_push),
        );
        device.cmd_dispatch(cmd, batch as u32, 1, 1);

        let output_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[output_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_index_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size((batch * 4) as u64).build()],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit batched linear argmax dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for batched linear argmax dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    let indices: &[u32] = bytemuck::cast_slice(&out_data);
    anyhow::ensure!(
        indices.len() >= batch,
        "batched linear argmax readback returned {} indices, expected {batch}",
        indices.len()
    );
    Ok(indices[..batch].to_vec())
}

/// Dispatch fused single-token full-attention Q/K/V projections.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_attn_qkv_decode_cached(
    vk_device: &VulkanDevice,
    x: &Tensor,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    dispatch_full_attn_qkv_decode_cached_impl(
        vk_device, x, q_weight_t, k_weight_t, v_weight_t, hidden, q_dim, k_dim, v_dim, false,
    )
}

/// Dispatch fused single-token full-attention Q/K/V projections with packed
/// BF16 immutable weights.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_attn_qkv_decode_cached_bf16_weights(
    vk_device: &VulkanDevice,
    x: &Tensor,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    dispatch_full_attn_qkv_decode_cached_impl(
        vk_device, x, q_weight_t, k_weight_t, v_weight_t, hidden, q_dim, k_dim, v_dim, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_full_attn_qkv_decode_cached_impl(
    vk_device: &VulkanDevice,
    x: &Tensor,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    bf16_weights: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_data = extract_tensor_bytes(x)?.0;
    anyhow::ensure!(
        x_data.len() == hidden * 4,
        "full_attn_qkv_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        hidden * 4
    );

    let total_out = q_dim + k_dim + v_dim;
    let glsl_path = if bf16_weights {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/full_attn_qkv_decode_bf16w.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/full_attn_qkv_decode.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [
        hidden as u32,
        q_dim as u32,
        k_dim as u32,
        v_dim as u32,
        total_out as u32,
    ];
    if full_attn_qkv_single_submit_enabled() {
        return dispatch_full_attn_qkv_decode_cached_single_submit(
            vk_device,
            q_weight_t,
            k_weight_t,
            v_weight_t,
            q_dim,
            k_dim,
            v_dim,
            total_out,
            &spirv,
            &push_constants,
            &x_data,
        );
    }

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create full_attn_qkv_decode x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload full_attn_qkv_decode x buffer")?;
    }

    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (total_out * 4) as u64)
            .context("failed to create full_attn_qkv_decode output buffer")?;
    let all_handles = vec![
        x_buf.handle(),
        q_weight_t.handle(),
        k_weight_t.handle(),
        v_weight_t.handle(),
        out_buf.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        total_out.div_ceil(16) as u32,
    )
    .context("full_attn_qkv_decode kernel failed")?;

    let out_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back full_attn_qkv_decode output")?
    };

    create_full_attn_qkv_tensors_from_data(&out_data, q_dim, k_dim, v_dim)
}

#[allow(clippy::too_many_arguments)]
fn dispatch_full_attn_qkv_decode_cached_single_submit(
    vk_device: &VulkanDevice,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    total_out: usize,
    spirv: &[u8],
    push_constants: &[u32],
    x_data: &[u8],
) -> Result<(Tensor, Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create full_attn_qkv_decode x buffer")?;
    let x_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, x_data.len() as u64)
        .context("failed to create full_attn_qkv_decode x staging buffer")?;
    VulkanBuffer::write_host_visible(device, &x_stage, x_data)?;

    let out_size = (total_out * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create full_attn_qkv_decode output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create full_attn_qkv_decode output staging buffer")?;

    let all_handles = vec![
        x_buf.handle(),
        q_weight_t.handle(),
        k_weight_t.handle(),
        v_weight_t.handle(),
        out_buf.handle(),
    ];
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate full_attn_qkv_decode descriptor set")?[0]
    };
    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            x_stage.handle(),
            x_buf.handle(),
            &[vk::BufferCopy::builder().size(x_data.len() as u64).build()],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, total_out.div_ceil(16) as u32, 1, 1);
        let output_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[output_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size(out_size).build()],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit full_attn_qkv_decode single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for full_attn_qkv_decode single-submit dispatch")?;
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    create_full_attn_qkv_tensors_from_data(&out_data, q_dim, k_dim, v_dim)
}

fn create_full_attn_qkv_tensors_from_data(
    out_data: &[u8],
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut offset = 0usize;
    let mut take = |len: usize, shape: &[usize]| -> Result<Tensor> {
        let byte_len = len * 4;
        let end = offset + byte_len;
        anyhow::ensure!(
            end <= out_data.len(),
            "full_attn_qkv_decode output slice exceeds readback buffer"
        );
        let tensor = create_tensor_from_data(&out_data[offset..end], shape, DType::F32)?;
        offset = end;
        Ok(tensor)
    };
    let q = take(q_dim, &[1, 1, q_dim])?;
    let k = take(k_dim, &[1, 1, k_dim])?;
    let v = take(v_dim, &[1, 1, v_dim])?;
    Ok((q, k, v))
}

/// Batched variant of [`dispatch_full_attn_qkv_decode_cached`] — fused single-token
/// Q/K/V projections across an arbitrary leading batch dim. `x` must be
/// `[batch, 1, hidden]`; outputs are `[batch, 1, q_dim]`, `[batch, 1, k_dim]`,
/// `[batch, 1, v_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_attn_qkv_decode_cached_batched(
    vk_device: &VulkanDevice,
    x: &Tensor,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    dispatch_full_attn_qkv_decode_cached_batched_impl(
        vk_device,
        x,
        q_weight_t,
        k_weight_t,
        v_weight_t,
        batch,
        hidden,
        q_dim,
        k_dim,
        v_dim,
        false,
    )
}

/// BF16-weight batched variant of [`dispatch_full_attn_qkv_decode_cached_batched`].
#[allow(clippy::too_many_arguments)]
pub fn dispatch_full_attn_qkv_decode_cached_batched_bf16_weights(
    vk_device: &VulkanDevice,
    x: &Tensor,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    dispatch_full_attn_qkv_decode_cached_batched_impl(
        vk_device,
        x,
        q_weight_t,
        k_weight_t,
        v_weight_t,
        batch,
        hidden,
        q_dim,
        k_dim,
        v_dim,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_full_attn_qkv_decode_cached_batched_impl(
    vk_device: &VulkanDevice,
    x: &Tensor,
    q_weight_t: &VulkanBuffer,
    k_weight_t: &VulkanBuffer,
    v_weight_t: &VulkanBuffer,
    batch: usize,
    hidden: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
    bf16_weights: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(batch > 0, "full_attn_qkv_decode_batched: batch must be > 0");

    let x_data = extract_tensor_bytes(x)?.0;
    let expected_x_bytes = batch
        .checked_mul(hidden)
        .and_then(|n| n.checked_mul(4))
        .context("full_attn_qkv_decode_batched: x byte count overflow")?;
    anyhow::ensure!(
        x_data.len() == expected_x_bytes,
        "full_attn_qkv_decode_batched: x buffer has {} bytes, expected {} (batch={}, hidden={})",
        x_data.len(),
        expected_x_bytes,
        batch,
        hidden
    );

    let total_out = q_dim
        .checked_add(k_dim)
        .and_then(|n| n.checked_add(v_dim))
        .context("full_attn_qkv_decode_batched: total_out overflow")?;
    anyhow::ensure!(total_out > 0, "full_attn_qkv_decode_batched: total_out is zero");
    let full_attn_qkv_rows4 = bf16_weights && batch >= 16 && full_attn_qkv_bf16w_rows4_enabled();
    let glsl_path = if bf16_weights {
        if full_attn_qkv_rows4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/full_attn_qkv_decode_batched_rows4_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/full_attn_qkv_decode_batched_bf16w.comp"
            )
        }
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/full_attn_qkv_decode_batched.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 6] = [
        hidden as u32,
        q_dim as u32,
        k_dim as u32,
        v_dim as u32,
        total_out as u32,
        batch as u32,
    ];

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create full_attn_qkv_decode_batched x buffer")?;

    let out_bytes = batch
        .checked_mul(total_out)
        .and_then(|n| n.checked_mul(4))
        .context("full_attn_qkv_decode_batched: output byte count overflow")?;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_bytes as u64)
        .context("failed to create full_attn_qkv_decode_batched output buffer")?;

    let all_handles = vec![
        x_buf.handle(),
        q_weight_t.handle(),
        k_weight_t.handle(),
        v_weight_t.handle(),
        out_buf.handle(),
    ];
    let col_groups = total_out.div_ceil(16);
    let row_groups = if full_attn_qkv_rows4 {
        batch.div_ceil(4)
    } else {
        batch
    };
    let total_groups = row_groups
        .checked_mul(col_groups)
        .context("full_attn_qkv_decode_batched: workgroup count overflow")?;
    let single_submit = full_attn_qkv_single_submit_enabled();
    let out_data = if single_submit {
        run_compute_pipeline_with_transfer_readback(
            vk_device,
            &x_buf,
            &x_data,
            &out_buf,
            out_bytes as u64,
            &spirv,
            &all_handles,
            &push_constants,
            total_groups as u32,
        )
        .context("full_attn_qkv_decode_batched single-submit kernel failed")?
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &x_buf,
                &x_data,
            )
            .context("failed to upload full_attn_qkv_decode_batched x buffer")?;
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            total_groups as u32,
        )
        .context("full_attn_qkv_decode_batched kernel failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back full_attn_qkv_decode_batched output")?
    };

    split_batched_qkv_output(&out_data, batch, q_dim, k_dim, v_dim)
}

/// Split the contiguous batched `[batch, total_out]` readback buffer into
/// three `[batch, 1, *_dim]` candle tensors. The shader writes rows in
/// `(q | k | v)` order per batch element, so we copy row-by-row into three
/// per-dim accumulators.
fn split_batched_qkv_output(
    out_data: &[u8],
    batch: usize,
    q_dim: usize,
    k_dim: usize,
    v_dim: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let total_out = q_dim + k_dim + v_dim;
    let expected_bytes = batch
        .checked_mul(total_out)
        .and_then(|n| n.checked_mul(4))
        .context("split_batched_qkv_output: byte count overflow")?;
    anyhow::ensure!(
        out_data.len() >= expected_bytes,
        "split_batched_qkv_output: readback has {} bytes, expected {}",
        out_data.len(),
        expected_bytes
    );
    let out_f32: &[f32] = bytemuck::cast_slice(&out_data[..expected_bytes]);

    let mut q_buf = Vec::with_capacity(batch * q_dim);
    let mut k_buf = Vec::with_capacity(batch * k_dim);
    let mut v_buf = Vec::with_capacity(batch * v_dim);
    for row in 0..batch {
        let base = row * total_out;
        q_buf.extend_from_slice(&out_f32[base..base + q_dim]);
        k_buf.extend_from_slice(&out_f32[base + q_dim..base + q_dim + k_dim]);
        v_buf.extend_from_slice(&out_f32[base + q_dim + k_dim..base + total_out]);
    }

    let q = Tensor::from_vec(q_buf, batch * q_dim, &Device::Cpu)?.reshape((batch, 1, q_dim))?;
    let k = Tensor::from_vec(k_buf, batch * k_dim, &Device::Cpu)?.reshape((batch, 1, k_dim))?;
    let v = Tensor::from_vec(v_buf, batch * v_dim, &Device::Cpu)?.reshape((batch, 1, v_dim))?;
    Ok((q, k, v))
}

/// Dispatch batched paged decode attention over compacted K/V windows.
///
/// `q` is `[batch, 1, num_heads, head_dim]`, `k` and `v` are compact
/// `[batch, max_seqlen, num_kv_heads, head_dim]`, and `seq_lens` gives the
/// active prefix length for each row. Output is `[batch, 1, num_heads,
/// head_dim]`.
pub fn dispatch_paged_attn_decode_batch_f32(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seq_lens: &[u32],
    softmax_scale: f32,
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let (batch, q_len, num_heads, head_dim) = q.dims4()?;
    let (k_batch, max_seqlen, num_kv_heads, k_head_dim) = k.dims4()?;
    let v_dims = v.dims4()?;
    anyhow::ensure!(
        q_len == 1,
        "paged_attn_decode_batch requires q_len=1, got {q_len}"
    );
    anyhow::ensure!(
        k_batch == batch && v_dims == (batch, max_seqlen, num_kv_heads, head_dim),
        "paged_attn_decode_batch K/V shape mismatch: k={:?} v={:?} q_batch={batch}",
        k.dims(),
        v.dims()
    );
    anyhow::ensure!(
        k_head_dim == head_dim && head_dim <= 256,
        "paged_attn_decode_batch supports head_dim <= 256 with matching K dim"
    );
    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "paged_attn_decode_batch requires integer GQA ratio"
    );
    anyhow::ensure!(
        seq_lens.len() == batch,
        "paged_attn_decode_batch seq_lens length {} != batch {batch}",
        seq_lens.len()
    );
    for &len in seq_lens {
        anyhow::ensure!(
            len > 0 && len as usize <= max_seqlen,
            "paged_attn_decode_batch invalid row seq_len {len} for max_seqlen {max_seqlen}"
        );
    }

    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let seq_data = bytemuck::cast_slice(seq_lens).to_vec();

    let make_input = |data: &[u8], label: &str| -> Result<VulkanBuffer> {
        let buf = VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)
            .with_context(|| format!("failed to create paged_attn_decode_batch {label} buffer"))?;
        Ok(buf)
    };
    let q_buf = make_input(&q_data, "q")?;
    let k_buf = make_input(&k_data, "k")?;
    let v_buf = make_input(&v_data, "v")?;
    let seq_buf = make_input(&seq_data, "seq_lens")?;

    let out_size = (batch * num_heads * head_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create paged_attn_decode_batch output buffer")?;
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants = [
        max_seqlen as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        softmax_scale.to_bits(),
    ];
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        seq_buf.handle(),
        out_buf.handle(),
    ];
    let out_data = if paged_attn_single_submit_enabled() {
        run_compute_pipeline_with_transfers_readback(
            vk_device,
            &[
                (&q_buf, &q_data),
                (&k_buf, &k_data),
                (&v_buf, &v_data),
                (&seq_buf, &seq_data),
            ],
            &out_buf,
            out_size,
            &spirv,
            &all_handles,
            &push_constants,
            (batch * num_heads) as u32,
        )
        .context("paged_attn_decode_batch single-submit kernel failed")?
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            if paged_attn_batched_uploads_enabled() {
                upload_buffers_with_command_pool(
                    device,
                    host_visible_mt,
                    queue,
                    *command_pool,
                    &[
                        (&q_buf, &q_data),
                        (&k_buf, &k_data),
                        (&v_buf, &v_data),
                        (&seq_buf, &seq_data),
                    ],
                )
                .context("failed to upload paged_attn_decode_batch inputs")?;
            } else {
                for (buf, data, label) in [
                    (&q_buf, &q_data, "q"),
                    (&k_buf, &k_data, "k"),
                    (&v_buf, &v_data, "v"),
                    (&seq_buf, &seq_data, "seq_lens"),
                ] {
                    VulkanBuffer::upload_data_with_command_pool(
                        device,
                        host_visible_mt,
                        queue,
                        *command_pool,
                        buf,
                        data,
                    )
                    .with_context(|| {
                        format!("failed to upload paged_attn_decode_batch {label} buffer")
                    })?;
                }
            }
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            (batch * num_heads) as u32,
        )
        .context("paged_attn_decode_batch kernel failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back paged_attn_decode_batch output")?
    };
    create_tensor_from_data(&out_data, &[batch, 1usize, num_heads, head_dim], DType::F32)
}

/// Paged-pool variant of [`dispatch_paged_attn_decode_batch_f32`].
///
/// Skips the host-side block_table → compacted K/V gather and uploads the
/// raw K/V pool plus the block table; the shader walks the per-row block
/// indices inline. Use when the compacted view would dwarf the pool itself
/// (i.e., when `batch * max_seqlen > total_slots`), which is the typical
/// shape for multi-batch decode at non-trivial context lengths.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_paged_attn_decode_batch_paged_f32(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k_pool: &Tensor,
    v_pool: &Tensor,
    block_table_u32: &[u32],
    seq_lens: &[u32],
    batch: usize,
    max_blocks_per_seq: usize,
    page_block_size: usize,
    softmax_scale: f32,
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    anyhow::ensure!(
        batch > 0 && max_blocks_per_seq > 0 && page_block_size > 0,
        "paged_attn_decode_batch_paged: batch/max_blocks_per_seq/page_block_size must be > 0"
    );
    anyhow::ensure!(
        seq_lens.len() == batch,
        "paged_attn_decode_batch_paged: seq_lens length {} != batch {batch}",
        seq_lens.len()
    );
    anyhow::ensure!(
        block_table_u32.len() == batch * max_blocks_per_seq,
        "paged_attn_decode_batch_paged: block_table length {} != batch*max_blocks_per_seq {}",
        block_table_u32.len(),
        batch * max_blocks_per_seq
    );

    let (q_batch, q_len, num_heads, head_dim) = q.dims4()?;
    let (total_slots, num_kv_heads, k_head_dim) = k_pool.dims3()?;
    let v_dims = v_pool.dims3()?;
    anyhow::ensure!(
        q_batch == batch && q_len == 1,
        "paged_attn_decode_batch_paged: q shape {:?} mismatch (expected [{batch}, 1, *, *])",
        q.dims()
    );
    anyhow::ensure!(
        k_head_dim == head_dim && v_dims == (total_slots, num_kv_heads, head_dim),
        "paged_attn_decode_batch_paged: K/V shape mismatch k={:?} v={:?}",
        k_pool.dims(),
        v_pool.dims()
    );
    anyhow::ensure!(
        num_heads % num_kv_heads == 0,
        "paged_attn_decode_batch_paged: requires integer GQA ratio"
    );
    for &len in seq_lens {
        anyhow::ensure!(
            len > 0,
            "paged_attn_decode_batch_paged: zero-length seq_len not supported"
        );
    }

    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k_pool)?.0;
    let v_data = extract_tensor_bytes(v_pool)?.0;
    let bt_bytes: Vec<u8> = bytemuck::cast_slice(block_table_u32).to_vec();
    let seq_bytes: Vec<u8> = bytemuck::cast_slice(seq_lens).to_vec();

    let make_input = |data: &[u8], label: &str| -> Result<VulkanBuffer> {
        VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)
            .with_context(|| format!("failed to create paged_attn_decode_batch_paged {label} buffer"))
    };
    let q_buf = make_input(&q_data, "q")?;
    let k_buf = make_input(&k_data, "k_pool")?;
    let v_buf = make_input(&v_data, "v_pool")?;
    let bt_buf = make_input(&bt_bytes, "block_table")?;
    let seq_buf = make_input(&seq_bytes, "seq_lens")?;

    let out_size = (batch * num_heads * head_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create paged_attn_decode_batch_paged output buffer")?;
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/paged_attn_decode_batch_paged.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants = [
        max_blocks_per_seq as u32,
        page_block_size as u32,
        num_heads as u32,
        num_kv_heads as u32,
        head_dim as u32,
        softmax_scale.to_bits(),
    ];
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        bt_buf.handle(),
        seq_buf.handle(),
        out_buf.handle(),
    ];
    let out_data = if paged_attn_single_submit_enabled() {
        run_compute_pipeline_with_transfers_readback(
            vk_device,
            &[
                (&q_buf, &q_data),
                (&k_buf, &k_data),
                (&v_buf, &v_data),
                (&bt_buf, &bt_bytes),
                (&seq_buf, &seq_bytes),
            ],
            &out_buf,
            out_size,
            &spirv,
            &all_handles,
            &push_constants,
            (batch * num_heads) as u32,
        )
        .context("paged_attn_decode_batch_paged single-submit kernel failed")?
    } else {
        {
            let command_pool = vk_device.transient_command_pool()?;
            if paged_attn_batched_uploads_enabled() {
                upload_buffers_with_command_pool(
                    device,
                    host_visible_mt,
                    queue,
                    *command_pool,
                    &[
                        (&q_buf, &q_data),
                        (&k_buf, &k_data),
                        (&v_buf, &v_data),
                        (&bt_buf, &bt_bytes),
                        (&seq_buf, &seq_bytes),
                    ],
                )
                .context("failed to upload paged_attn_decode_batch_paged inputs")?;
            } else {
                for (buf, data, label) in [
                    (&q_buf, &q_data, "q"),
                    (&k_buf, &k_data, "k_pool"),
                    (&v_buf, &v_data, "v_pool"),
                    (&bt_buf, &bt_bytes, "block_table"),
                    (&seq_buf, &seq_bytes, "seq_lens"),
                ] {
                    VulkanBuffer::upload_data_with_command_pool(
                        device,
                        host_visible_mt,
                        queue,
                        *command_pool,
                        buf,
                        data,
                    )
                    .with_context(|| {
                        format!("failed to upload paged_attn_decode_batch_paged {label} buffer")
                    })?;
                }
            }
        }
        run_compute_pipeline(
            vk_device,
            &spirv,
            &all_handles,
            all_handles.len(),
            &push_constants,
            (batch * num_heads) as u32,
        )
        .context("paged_attn_decode_batch_paged kernel failed")?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back paged_attn_decode_batch_paged output")?
    };
    create_tensor_from_data(&out_data, &[batch, 1usize, num_heads, head_dim], DType::F32)
}

/// Dispatch a cached fused single-token SwiGLU gate/up projection.
///
/// `x` is `[batch, 1, hidden]`; both weights are `[hidden, intermediate]`. The
/// returned tensor is f32 CPU `[batch, 1, intermediate]` after `silu(gate) * up`.
pub fn dispatch_mlp_gate_up_decode_cached(
    vk_device: &VulkanDevice,
    x: &Tensor,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    hidden: usize,
    intermediate: usize,
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_dims = x.dims();
    anyhow::ensure!(
        x_dims.len() == 3 && x_dims[1] == 1 && x_dims[2] == hidden,
        "mlp_gate_up_decode: x shape {:?} does not match [batch, 1, {hidden}]",
        x_dims
    );
    let batch = x_dims[0];
    let x_data = extract_tensor_bytes(x)?.0;
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "mlp_gate_up_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create mlp_gate_up_decode x buffer")?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload mlp_gate_up_decode x buffer")?;
    }

    let out_buf = VulkanBuffer::create_device_local(
        device,
        device_local_mt,
        (batch * intermediate * 4) as u64,
    )
    .context("failed to create mlp_gate_up_decode output buffer")?;

    let use_rows2 = use_prefill_row_pair_matmul(batch);
    let glsl_path = if batch == 1 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode.comp"
        )
    } else if use_rows2 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched_rows2.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let mut push_constants = vec![hidden as u32, intermediate as u32];
    if batch > 1 {
        push_constants.push(batch as u32);
    }
    let all_handles = vec![
        x_buf.handle(),
        gate_weight_t.handle(),
        up_weight_t.handle(),
        out_buf.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        if batch == 1 {
            intermediate.div_ceil(64) as u32
        } else if use_rows2 {
            (batch.div_ceil(2) * intermediate.div_ceil(64)) as u32
        } else {
            (batch * intermediate.div_ceil(128)) as u32
        },
    )
    .context("mlp_gate_up_decode kernel failed")?;

    let out_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back mlp_gate_up_decode output")?
    };
    create_tensor_from_data(&out_data, &[batch, 1, intermediate], DType::F32)
}

/// Dispatch single-token SwiGLU MLP with the hidden activation kept on Vulkan.
///
/// This runs two kernels:
/// 1. `hidden = silu(x @ gate_t) * (x @ up_t)`
/// 2. `out = hidden @ down_t`
///
/// Only the final `[batch, 1, out_dim]` tensor is read back to CPU.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlp_decode_cached(
    vk_device: &VulkanDevice,
    x: &Tensor,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<Tensor> {
    dispatch_mlp_decode_cached_impl(
        vk_device,
        x,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        hidden,
        intermediate,
        out_dim,
        false,
        false,
    )
}

/// Dispatch single-token SwiGLU MLP with packed BF16 immutable weights.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlp_decode_cached_bf16_weights(
    vk_device: &VulkanDevice,
    x: &Tensor,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<Tensor> {
    dispatch_mlp_decode_cached_impl(
        vk_device,
        x,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        hidden,
        intermediate,
        out_dim,
        true,
        true,
    )
}

/// Dispatch single-token SwiGLU MLP with packed BF16 gate/up weights and an
/// F32 down-projection weight.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlp_decode_cached_bf16_gate_up_f32_down(
    vk_device: &VulkanDevice,
    x: &Tensor,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
) -> Result<Tensor> {
    dispatch_mlp_decode_cached_impl(
        vk_device,
        x,
        gate_weight_t,
        up_weight_t,
        down_weight_t,
        hidden,
        intermediate,
        out_dim,
        true,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_mlp_decode_cached_impl(
    vk_device: &VulkanDevice,
    x: &Tensor,
    gate_weight_t: &VulkanBuffer,
    up_weight_t: &VulkanBuffer,
    down_weight_t: &VulkanBuffer,
    hidden: usize,
    intermediate: usize,
    out_dim: usize,
    gate_up_bf16_weights: bool,
    down_bf16_weights: bool,
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let x_dims = x.dims();
    anyhow::ensure!(
        x_dims.len() == 3 && x_dims[1] == 1 && x_dims[2] == hidden,
        "mlp_decode: x shape {:?} does not match [batch, 1, {hidden}]",
        x_dims
    );
    let batch = x_dims[0];
    let profile_stages = profile_vulkan_mlp_kernel_stages_enabled();
    let gate_up_rows2 = !gate_up_bf16_weights && use_prefill_row_pair_matmul(batch);
    // For the all-bf16 MLP, the rows4 / rows8 amortization only beats the
    // per-batch-row bf16w kernel once we have enough rows to keep the SMs
    // full: rows4 cuts workgroup count by 4×, rows8 by 8×. On NVIDIA RTX
    // 6000 Ada the empirical crossover is batch ≈ 32 (rows4) and batch ≈ 64
    // (rows8) — at smaller batches the unbatched bf16w kernel wins because
    // it puts batch×col_groups workgroups on the GPU. The f32-down rows4
    // path keeps its older batch≥8 threshold because reading 4 B/weight
    // makes weight-read reuse pay off sooner. See decode_microbench output
    // in PR description for the empirical curve.
    let rows8_path = gate_up_bf16_weights
        && down_bf16_weights
        && batch >= 64
        && mlp_bf16_rows8_enabled();
    let down_bf16_rows4 = down_bf16_weights
        && gate_up_bf16_weights
        && batch >= 32
        && !rows8_path
        && mlp_bf16_down_rows4_enabled();
    // gate_up rows4 reuses weights across 4 rows. For the bf16/f32-down case
    // it has always paid off at batch≥8 (existing behavior); for the
    // all-bf16 case the win only materializes once linear-down also shifts
    // off the per-batch-row kernel, so pair it with `down_bf16_rows4`.
    let gate_up_rows4 = gate_up_bf16_weights
        && batch >= 8
        && !rows8_path
        && (down_bf16_rows4 || !down_bf16_weights)
        && mlp_bf16_gate_up_rows4_enabled();
    let down_rows4 =
        gate_up_bf16_weights && !down_bf16_weights && batch >= 8 && mlp_f32_down_rows4_enabled();
    let down_rows2 = !down_bf16_weights && !down_rows4 && use_prefill_row_pair_matmul(batch);
    let chained_dispatch = mlp_chained_dispatch_enabled();
    let chained_transfer_submit = chained_dispatch && mlp_chained_transfer_submit_enabled();
    let total_start = profile_stages.then(Instant::now);
    let stage_start = profile_stages.then(Instant::now);
    let x_data = extract_tensor_bytes(x)?.0;
    finish_vulkan_mlp_kernel_stage_profile(
        "extract_x",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );
    anyhow::ensure!(
        x_data.len() == batch * hidden * 4,
        "mlp_decode: x buffer has {} bytes, expected {}",
        x_data.len(),
        batch * hidden * 4
    );
    let stage_start = profile_stages.then(Instant::now);
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)
        .context("failed to create mlp_decode x buffer")?;
    finish_vulkan_mlp_kernel_stage_profile(
        "create_x_buffer",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );
    if !chained_transfer_submit {
        let stage_start = profile_stages.then(Instant::now);
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )
        .context("failed to upload mlp_decode x buffer")?;
        finish_vulkan_mlp_kernel_stage_profile(
            "upload_x",
            batch,
            hidden,
            intermediate,
            out_dim,
            gate_up_bf16_weights,
            down_bf16_weights,
            gate_up_rows2,
            gate_up_rows4,
            down_rows4,
            down_rows2,
            stage_start,
        );
    }

    let stage_start = profile_stages.then(Instant::now);
    let hidden_buf = VulkanBuffer::create_device_local(
        device,
        device_local_mt,
        (batch * intermediate * 4) as u64,
    )
    .context("failed to create mlp_decode hidden buffer")?;
    let out_size = (batch * out_dim * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create mlp_decode output buffer")?;
    finish_vulkan_mlp_kernel_stage_profile(
        "create_work_buffers",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );

    let gate_up_glsl = if gate_up_bf16_weights {
        if batch == 1 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_bf16w.comp"
            )
        } else if rows8_path {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_batched_rows8_bf16w.comp"
            )
        } else if gate_up_rows4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_batched_rows4_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/mlp_gate_up_decode_batched_bf16w.comp"
            )
        }
    } else if batch == 1 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode.comp"
        )
    } else if gate_up_rows2 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched_rows2.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/mlp_gate_up_decode_batched.comp"
        )
    };
    let stage_start = profile_stages.then(Instant::now);
    let gate_up_spirv = crate::pipeline::ShaderPipeline::compile_shader(gate_up_glsl)?;
    finish_vulkan_mlp_kernel_stage_profile(
        "gate_up_shader",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );
    let mut gate_up_push = vec![hidden as u32, intermediate as u32];
    if batch > 1 {
        gate_up_push.push(batch as u32);
    }
    let gate_up_handles = vec![
        x_buf.handle(),
        gate_weight_t.handle(),
        up_weight_t.handle(),
        hidden_buf.handle(),
    ];
    let gate_up_workgroups = if batch == 1 {
        intermediate.div_ceil(64) as u32
    } else if rows8_path {
        (batch.div_ceil(8) * intermediate.div_ceil(64)) as u32
    } else if gate_up_rows4 {
        (batch.div_ceil(4) * intermediate.div_ceil(64)) as u32
    } else if gate_up_rows2 {
        (batch.div_ceil(2) * intermediate.div_ceil(64)) as u32
    } else {
        (batch * intermediate.div_ceil(128)) as u32
    };

    let linear_glsl = if down_bf16_weights {
        if batch == 1 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_bf16w.comp"
            )
        } else if rows8_path {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched_rows8_bf16w.comp"
            )
        } else if down_bf16_rows4 {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched_rows4_bf16w.comp"
            )
        } else {
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/csrc/shaders/linear_decode_batched_bf16w.comp"
            )
        }
    } else if batch == 1 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode.comp"
        )
    } else if down_rows4 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_batched_rows4.comp"
        )
    } else if down_rows2 {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_batched_rows2.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/linear_decode_batched.comp"
        )
    };
    let stage_start = profile_stages.then(Instant::now);
    let linear_spirv = crate::pipeline::ShaderPipeline::compile_shader(linear_glsl)?;
    finish_vulkan_mlp_kernel_stage_profile(
        "down_shader",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );
    let mut linear_push = vec![intermediate as u32, out_dim as u32];
    if batch > 1 {
        linear_push.push(batch as u32);
    }
    let linear_handles = vec![
        hidden_buf.handle(),
        down_weight_t.handle(),
        out_buf.handle(),
    ];
    let linear_workgroups = if batch == 1 {
        out_dim.div_ceil(16) as u32
    } else if rows8_path {
        (batch.div_ceil(8) * out_dim.div_ceil(32)) as u32
    } else if down_rows4 || down_bf16_rows4 {
        (batch.div_ceil(4) * out_dim.div_ceil(32)) as u32
    } else if down_rows2 {
        (batch.div_ceil(2) * out_dim.div_ceil(32)) as u32
    } else {
        (batch * out_dim.div_ceil(32)) as u32
    };

    let out_data = if chained_dispatch {
        if chained_transfer_submit {
            let stage_start = profile_stages.then(Instant::now);
            let out_data = run_two_stage_compute_pipeline_with_transfer_readback(
                vk_device,
                &x_buf,
                &x_data,
                &out_buf,
                out_size,
                &gate_up_spirv,
                &gate_up_handles,
                &gate_up_push,
                gate_up_workgroups,
                &linear_spirv,
                &linear_handles,
                &linear_push,
                linear_workgroups,
            )
            .context("mlp_decode chained transfer + gate/up + down kernels failed")?;
            finish_vulkan_mlp_kernel_stage_profile(
                "chained_transfer_dispatch_readback",
                batch,
                hidden,
                intermediate,
                out_dim,
                gate_up_bf16_weights,
                down_bf16_weights,
                gate_up_rows2,
                gate_up_rows4,
                down_rows4,
                down_rows2,
                stage_start,
            );
            out_data
        } else {
            let stage_start = profile_stages.then(Instant::now);
            run_two_stage_compute_pipeline(
                vk_device,
                &gate_up_spirv,
                &gate_up_handles,
                &gate_up_push,
                gate_up_workgroups,
                &linear_spirv,
                &linear_handles,
                &linear_push,
                linear_workgroups,
            )
            .context("mlp_decode chained gate/up + down kernels failed")?;
            finish_vulkan_mlp_kernel_stage_profile(
                "chained_dispatch",
                batch,
                hidden,
                intermediate,
                out_dim,
                gate_up_bf16_weights,
                down_bf16_weights,
                gate_up_rows2,
                gate_up_rows4,
                down_rows4,
                down_rows2,
                stage_start,
            );
            let stage_start = profile_stages.then(Instant::now);
            let command_pool = vk_device.transient_command_pool()?;
            let out_data = VulkanBuffer::read_back_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &out_buf,
            )
            .context("failed to read back mlp_decode output")?;
            finish_vulkan_mlp_kernel_stage_profile(
                "readback",
                batch,
                hidden,
                intermediate,
                out_dim,
                gate_up_bf16_weights,
                down_bf16_weights,
                gate_up_rows2,
                gate_up_rows4,
                down_rows4,
                down_rows2,
                stage_start,
            );
            out_data
        }
    } else {
        let stage_start = profile_stages.then(Instant::now);
        run_compute_pipeline(
            vk_device,
            &gate_up_spirv,
            &gate_up_handles,
            gate_up_handles.len(),
            &gate_up_push,
            gate_up_workgroups,
        )
        .context("mlp_decode gate/up kernel failed")?;
        finish_vulkan_mlp_kernel_stage_profile(
            "gate_up_dispatch",
            batch,
            hidden,
            intermediate,
            out_dim,
            gate_up_bf16_weights,
            down_bf16_weights,
            gate_up_rows2,
            gate_up_rows4,
            down_rows4,
            down_rows2,
            stage_start,
        );
        let stage_start = profile_stages.then(Instant::now);
        run_compute_pipeline(
            vk_device,
            &linear_spirv,
            &linear_handles,
            linear_handles.len(),
            &linear_push,
            linear_workgroups,
        )
        .context("mlp_decode down kernel failed")?;
        finish_vulkan_mlp_kernel_stage_profile(
            "down_dispatch",
            batch,
            hidden,
            intermediate,
            out_dim,
            gate_up_bf16_weights,
            down_bf16_weights,
            gate_up_rows2,
            gate_up_rows4,
            down_rows4,
            down_rows2,
            stage_start,
        );
        let stage_start = profile_stages.then(Instant::now);
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back mlp_decode output")?;
        finish_vulkan_mlp_kernel_stage_profile(
            "readback",
            batch,
            hidden,
            intermediate,
            out_dim,
            gate_up_bf16_weights,
            down_bf16_weights,
            gate_up_rows2,
            gate_up_rows4,
            down_rows4,
            down_rows2,
            stage_start,
        );
        out_data
    };
    let stage_start = profile_stages.then(Instant::now);
    let out = create_tensor_from_data(&out_data, &[batch, 1, out_dim], DType::F32);
    finish_vulkan_mlp_kernel_stage_profile(
        "create_tensor",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        stage_start,
    );
    finish_vulkan_mlp_kernel_stage_profile(
        "total",
        batch,
        hidden,
        intermediate,
        out_dim,
        gate_up_bf16_weights,
        down_bf16_weights,
        gate_up_rows2,
        gate_up_rows4,
        down_rows4,
        down_rows2,
        total_start,
    );
    out
}

/// Dispatch fused single-token GDN gates + recurrent update + gated RMSNorm.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_decode_gates_recurrent_rmsnorm(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
    skip_state_readback: bool,
) -> Result<(Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let (batch, _, nv, dk) = q.dims4()?;
    let (v_batch, _, v_nv, dv) = v.dims4()?;
    anyhow::ensure!(
        v_batch == batch,
        "gdn_decode fused: v batch {v_batch} != q batch {batch}"
    );
    anyhow::ensure!(
        v_nv == nv,
        "gdn_decode fused: v heads {v_nv} != q heads {nv}"
    );
    anyhow::ensure!(
        dv <= 256,
        "gdn_decode fused: dv {dv} exceeds shader local capacity 256"
    );

    let input_tensors = [q, k, v, a, b, a_log, dt_bias, state, z, weight];
    let mut input_data = Vec::with_capacity(input_tensors.len());
    for tensor in &input_tensors {
        input_data.push(extract_tensor_bytes(tensor)?.0);
    }

    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_decode_gates_recurrent_rmsnorm.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [nv as u32, dk as u32, dv as u32, eps.to_bits(), batch as u32];

    if gdn_decode_fused_single_submit_enabled() {
        return dispatch_gdn_decode_gates_recurrent_rmsnorm_single_submit(
            vk_device,
            q,
            state,
            &input_data,
            &spirv,
            push_constants,
            batch,
            nv,
            dv,
            skip_state_readback,
        );
    }

    let use_host_visible_state = gdn_decode_host_visible_state_enabled();
    let mut buffers = Vec::with_capacity(input_data.len());
    for (idx, data) in input_data.iter().enumerate() {
        let buffer = if use_host_visible_state && idx == 7 {
            let buffer =
                VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
            VulkanBuffer::write_host_visible(device, &buffer, data)?;
            buffer
        } else {
            let buffer =
                VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
            let command_pool = vk_device.transient_command_pool()?;
            VulkanBuffer::upload_data_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &buffer,
                data,
            )?;
            buffer
        };
        buffers.push(buffer);
    }

    let out_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, (batch * nv * dv * 4) as u64)
            .context("failed to create gdn_decode fused output buffer")?;

    let mut all_handles: Vec<vk::Buffer> = buffers.iter().map(|buf| buf.handle()).collect();
    all_handles.push(out_buf.handle());

    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        (batch * nv) as u32,
    )
    .context("gdn_decode_gates_recurrent_rmsnorm kernel failed")?;

    let (out_data, state_data) = {
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )
        .context("failed to read back gdn_decode fused output")?;
        let state_data = if skip_state_readback {
            None
        } else if use_host_visible_state {
            Some(VulkanBuffer::read_host_visible(device, &buffers[7]))
        } else {
            Some(VulkanBuffer::read_back_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &buffers[7],
            ))
        }
        .transpose()
        .context("failed to read back gdn_decode fused state")?;
        (out_data, state_data)
    };

    let out = create_tensor_from_data(&out_data, &[batch, 1, nv, dv], q.dtype())?;
    let new_state = if let Some(state_data) = state_data {
        create_tensor_from_data(&state_data, state.dims().as_ref(), state.dtype())?
    } else {
        state.clone()
    };
    Ok((out, new_state))
}

/// Dispatch fused GDN decode while keeping recurrent state device-resident.
///
/// The first call uploads `state` into a device-local buffer. Later calls pass
/// the returned buffer back and avoid the full recurrent-state readback/upload
/// pair; only the small normalized output is copied to the CPU.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_decode_gates_recurrent_rmsnorm_resident_state(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
    resident_state: Option<Arc<VulkanBuffer>>,
) -> Result<(Tensor, Arc<VulkanBuffer>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let (batch, _, nv, dk) = q.dims4()?;
    let (v_batch, _, v_nv, dv) = v.dims4()?;
    anyhow::ensure!(
        v_batch == batch,
        "gdn_decode fused resident: v batch {v_batch} != q batch {batch}"
    );
    anyhow::ensure!(
        v_nv == nv,
        "gdn_decode fused resident: v heads {v_nv} != q heads {nv}"
    );
    anyhow::ensure!(
        dv <= 256,
        "gdn_decode fused resident: dv {dv} exceeds shader local capacity 256"
    );

    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let a_data = extract_tensor_bytes(a)?.0;
    let b_data = extract_tensor_bytes(b)?.0;
    let a_log_data = extract_tensor_bytes(a_log)?.0;
    let dt_bias_data = extract_tensor_bytes(dt_bias)?.0;
    let state_data = if resident_state.is_none() {
        Some(extract_tensor_bytes(state)?.0)
    } else {
        None
    };
    let z_data = extract_tensor_bytes(z)?.0;
    let weight_data = extract_tensor_bytes(weight)?.0;

    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_decode_gates_recurrent_rmsnorm.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let push_constants: [u32; 5] = [nv as u32, dk as u32, dv as u32, eps.to_bits(), batch as u32];

    let make_device_and_staging = |data: &[u8]| -> Result<(VulkanBuffer, VulkanBuffer)> {
        let device_buf =
            VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Ok((device_buf, staging))
    };

    let (q_buf, q_stage) = make_device_and_staging(&q_data)?;
    let (k_buf, k_stage) = make_device_and_staging(&k_data)?;
    let (v_buf, v_stage) = make_device_and_staging(&v_data)?;
    let (a_buf, a_stage) = make_device_and_staging(&a_data)?;
    let (b_buf, b_stage) = make_device_and_staging(&b_data)?;
    let (a_log_buf, a_log_stage) = make_device_and_staging(&a_log_data)?;
    let (dt_bias_buf, dt_bias_stage) = make_device_and_staging(&dt_bias_data)?;
    let (z_buf, z_stage) = make_device_and_staging(&z_data)?;
    let (weight_buf, weight_stage) = make_device_and_staging(&weight_data)?;

    let state_buf = match resident_state {
        Some(buffer) => buffer,
        None => {
            let data = state_data
                .as_ref()
                .expect("state data exists when resident state is absent");
            Arc::new(VulkanBuffer::create_device_local(
                device,
                device_local_mt,
                data.len() as u64,
            )?)
        }
    };
    let state_stage = if let Some(data) = &state_data {
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Some(staging)
    } else {
        None
    };

    let out_size = (batch * nv * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create gdn_decode fused resident output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create gdn_decode fused resident output staging buffer")?;

    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        a_buf.handle(),
        b_buf.handle(),
        a_log_buf.handle(),
        dt_bias_buf.handle(),
        state_buf.handle(),
        z_buf.handle(),
        weight_buf.handle(),
        out_buf.handle(),
    ];
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        &spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate gdn_decode fused resident descriptor set")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (src, dst, size) in [
            (&q_stage, &q_buf, q_data.len() as u64),
            (&k_stage, &k_buf, k_data.len() as u64),
            (&v_stage, &v_buf, v_data.len() as u64),
            (&a_stage, &a_buf, a_data.len() as u64),
            (&b_stage, &b_buf, b_data.len() as u64),
            (&a_log_stage, &a_log_buf, a_log_data.len() as u64),
            (&dt_bias_stage, &dt_bias_buf, dt_bias_data.len() as u64),
            (&z_stage, &z_buf, z_data.len() as u64),
            (&weight_stage, &weight_buf, weight_data.len() as u64),
        ] {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                dst.handle(),
                &[vk::BufferCopy::builder().size(size).build()],
            );
        }
        if let (Some(state_stage), Some(state_data)) = (&state_stage, &state_data) {
            device.cmd_copy_buffer(
                cmd,
                state_stage.handle(),
                state_buf.handle(),
                &[vk::BufferCopy::builder()
                    .size(state_data.len() as u64)
                    .build()],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, (batch * nv) as u32, 1, 1);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size(out_size).build()],
        );

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn_decode fused resident dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn_decode fused resident dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)
        .context("failed to read back gdn_decode fused resident output")?;
    let out = create_tensor_from_data(&out_data, &[batch, 1, nv, dv], q.dtype())?;
    Ok((out, state_buf))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_decode_gates_recurrent_rmsnorm_single_submit(
    vk_device: &VulkanDevice,
    q: &Tensor,
    state: &Tensor,
    input_data: &[Vec<u8>],
    spirv: &[u8],
    push_constants: [u32; 5],
    batch: usize,
    nv: usize,
    dv: usize,
    skip_state_readback: bool,
) -> Result<(Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();
    anyhow::ensure!(
        input_data.len() == 10,
        "gdn_decode fused single-submit expects 10 inputs, got {}",
        input_data.len()
    );

    let use_host_visible_state = gdn_decode_host_visible_state_enabled();
    let mut buffers = Vec::with_capacity(input_data.len());
    let mut staging = Vec::with_capacity(input_data.len());
    for (idx, data) in input_data.iter().enumerate() {
        if use_host_visible_state && idx == 7 {
            let buffer =
                VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
            VulkanBuffer::write_host_visible(device, &buffer, data)?;
            buffers.push(buffer);
            staging.push(None);
        } else {
            let buffer =
                VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
            let stage =
                VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
            VulkanBuffer::write_host_visible(device, &stage, data)?;
            buffers.push(buffer);
            staging.push(Some(stage));
        }
    }

    let out_size = (batch * nv * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)
        .context("failed to create gdn_decode fused output buffer")?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)
        .context("failed to create gdn_decode fused output staging buffer")?;

    let mut all_handles: Vec<vk::Buffer> = buffers.iter().map(|buf| buf.handle()).collect();
    all_handles.push(out_buf.handle());
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (idx, stage) in staging.iter().enumerate() {
            let Some(stage) = stage else {
                continue;
            };
            device.cmd_copy_buffer(
                cmd,
                stage.handle(),
                buffers[idx].handle(),
                &[vk::BufferCopy::builder()
                    .size(input_data[idx].len() as u64)
                    .build()],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, (batch * nv) as u32, 1, 1);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );

        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size(out_size).build()],
        );
        if !use_host_visible_state && !skip_state_readback {
            let state_stage = staging[7]
                .as_ref()
                .expect("state staging exists when state is device-local");
            device.cmd_copy_buffer(
                cmd,
                buffers[7].handle(),
                state_stage.handle(),
                &[vk::BufferCopy::builder()
                    .size(input_data[7].len() as u64)
                    .build()],
            );
        }

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn_decode fused single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn_decode fused single-submit dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)
        .context("failed to read back gdn_decode fused output")?;
    let state_data = if skip_state_readback {
        None
    } else if use_host_visible_state {
        Some(VulkanBuffer::read_host_visible(device, &buffers[7]))
    } else {
        Some(VulkanBuffer::read_host_visible(
            device,
            staging[7]
                .as_ref()
                .expect("state staging exists when state is device-local"),
        ))
    }
    .transpose()
    .context("failed to read back gdn_decode fused state")?;

    let out = create_tensor_from_data(&out_data, &[batch, 1, nv, dv], q.dtype())?;
    let new_state = if let Some(state_data) = state_data {
        create_tensor_from_data(&state_data, state.dims().as_ref(), state.dtype())?
    } else {
        state.clone()
    };
    Ok((out, new_state))
}

// ---------------------------------------------------------------------------
// Specialized dispatch functions for GDN kernels
// ---------------------------------------------------------------------------

/// Dispatch GDN gates kernel.
///
/// beta = sigmoid(b)
/// g    = -exp(A_log) * softplus(a + dt_bias)
///
/// Inputs:  a[B,T,nv], b[B,T,nv], A_log[nv], dt_bias[nv]
/// Outputs: beta[B,T,nv], g[B,T,nv]
pub fn dispatch_gdn_gates(
    vk_device: &VulkanDevice,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    out_shape: &[usize],
) -> Result<(Tensor, Tensor)> {
    let nv = a_log.elem_count();
    anyhow::ensure!(
        dt_bias.elem_count() == nv,
        "gdn_gates: dt_bias has {} elements, expected {}",
        dt_bias.elem_count(),
        nv
    );
    let a_log_buf = upload_tensor_f32_buffer(vk_device, a_log)?;
    let dt_bias_buf = upload_tensor_f32_buffer(vk_device, dt_bias)?;
    dispatch_gdn_gates_cached(vk_device, a, b, &a_log_buf, &dt_bias_buf, nv, out_shape)
}

/// Dispatch GDN gates kernel with cached immutable A_log and dt_bias buffers.
pub fn dispatch_gdn_gates_cached(
    vk_device: &VulkanDevice,
    a: &Tensor,
    b: &Tensor,
    a_log: &VulkanBuffer,
    dt_bias: &VulkanBuffer,
    nv: usize,
    out_shape: &[usize],
) -> Result<(Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let a_data = extract_tensor_bytes(a)?.0;
    let b_data = extract_tensor_bytes(b)?.0;

    // Compile shader
    let glsl_path = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/gdn_gates.comp");
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Create input buffers + upload
    let a_buf = VulkanBuffer::create_device_local(device, device_local_mt, a_data.len() as u64)?;
    let b_buf = VulkanBuffer::create_device_local(device, device_local_mt, b_data.len() as u64)?;
    if gdn_gates_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        upload_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[(&a_buf, &a_data), (&b_buf, &b_data)],
        )?;
    } else {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &a_buf,
            &a_data,
        )?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &b_buf,
            &b_data,
        )?;
    }

    // Create output buffers
    let elem_count: usize = out_shape.iter().product();
    let output_size = (elem_count * 4) as u64; // f32
    let beta_buf = VulkanBuffer::create_device_local(device, device_local_mt, output_size)?;
    let g_buf = VulkanBuffer::create_device_local(device, device_local_mt, output_size)?;

    // Push constants: total elements, nv
    let push_constants: [u32; 2] = [elem_count as u32, nv as u32];

    // Workgroup count
    let workgroup_count = elem_count.div_ceil(256) as u32;

    // Build descriptor bindings: a=0, b=1, a_log=2, dt_bias=3, beta_out=4, g_out=5
    let all_handles = vec![
        a_buf.handle(),
        b_buf.handle(),
        a_log.handle(),
        dt_bias.handle(),
        beta_buf.handle(),
        g_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back both outputs
    let (beta_data, g_data) = if gdn_gates_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        let mut data = read_back_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[&beta_buf, &g_buf],
        )?;
        anyhow::ensure!(
            data.len() == 2,
            "gdn_gates batched readback returned wrong count"
        );
        (data.remove(0), data.remove(0))
    } else {
        let command_pool = vk_device.transient_command_pool()?;
        let beta_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &beta_buf,
        )?;
        let g_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &g_buf,
        )?;
        (beta_data, g_data)
    };

    // Cleanup
    drop(a_buf);
    drop(b_buf);
    drop(beta_buf);
    drop(g_buf);

    let output_dtype = a.dtype();
    let beta_tensor = create_tensor_from_data(&beta_data, out_shape, output_dtype)?;
    let g_tensor = create_tensor_from_data(&g_data, out_shape, output_dtype)?;
    Ok((beta_tensor, g_tensor))
}

/// Dispatch GDN gated RMSNorm kernel.
///
/// out = rms_norm(x, weight, eps) * silu(z)
///
/// Inputs: x[...hidden], z[...hidden], weight[hidden]
/// Output: out[...hidden]
pub fn dispatch_gdn_gated_rms_norm(
    vk_device: &VulkanDevice,
    x: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f32,
    out_shape: &[usize],
) -> Result<Tensor> {
    let hidden = weight.elem_count();
    let weight_buf = upload_tensor_f32_buffer(vk_device, weight)?;
    dispatch_gdn_gated_rms_norm_cached(vk_device, x, z, &weight_buf, hidden, eps, out_shape)
}

/// Dispatch GDN gated RMSNorm kernel with cached immutable norm weight.
pub fn dispatch_gdn_gated_rms_norm_cached(
    vk_device: &VulkanDevice,
    x: &Tensor,
    z: &Tensor,
    weight: &VulkanBuffer,
    hidden: usize,
    eps: f32,
    out_shape: &[usize],
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let x_data = extract_tensor_bytes(x)?.0;
    let z_data = extract_tensor_bytes(z)?.0;

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_gated_rms_norm.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Create input buffers + upload
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)?;
    let z_buf = VulkanBuffer::create_device_local(device, device_local_mt, z_data.len() as u64)?;
    if gdn_gated_norm_batched_uploads_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        upload_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[(&x_buf, &x_data), (&z_buf, &z_data)],
        )?;
    } else {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &z_buf,
            &z_data,
        )?;
    }

    // Create output buffer
    let elem_count: usize = out_shape.iter().product();
    let output_size = (elem_count * 4) as u64; // f32
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, output_size)?;

    // Push constants: rows, hidden, eps
    let rows = elem_count / hidden;
    let push_constants: [u32; 3] = [rows as u32, hidden as u32, eps.to_bits()];

    // Workgroup count: one group per row
    let workgroup_count = rows as u32;

    // Build descriptor bindings: x=0, z=1, weight=2, out=3
    let all_handles = vec![
        x_buf.handle(),
        z_buf.handle(),
        weight.handle(),
        out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back output
    let output_data = {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )?
    };

    // Cleanup
    drop(x_buf);
    drop(z_buf);
    drop(out_buf);

    create_tensor_from_data(&output_data, out_shape, x.dtype())
        .context("failed to create gdn_gated_rms_norm output tensor")
}

/// Dispatch causal_conv1d update kernel (single-token decode path).
///
/// Depthwise conv1d with kernel_size=4, silu-fused.
/// `x`: `[B, C, 1]` bf16. `weight`: `[C, K]` bf16. `conv_state`: `[B, C, K-1]` f32.
/// Returns `out: [B, C, 1]` f32 and updates `conv_state` in-place.
///
/// Two-dispatch approach to avoid data races on conv_state:
/// 1. `causal_conv1d.comp` — computes output only (no state writes)
/// 2. `causal_conv1d_state_advance.comp` — advances state per (b, c) pair
pub fn dispatch_causal_conv1d_update(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
) -> Result<(Tensor, Tensor)> {
    if kernel_size != 4 {
        anyhow::bail!("causal_conv1d: only kernel_size=4 supported");
    }

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let x_data = extract_tensor_bytes(x)?.0;
    let weight_data = extract_tensor_bytes(weight)?.0;
    let state_data = extract_tensor_bytes(conv_state)?.0;

    // Parse shape [B, C, T]
    let dims = x.dims();
    let (batch, channels, seq_len) = (dims[0], dims[1], dims[2]);

    // Create input buffers + upload
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)?;
    let weight_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)?;
    let state_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &weight_buf,
            &weight_data,
        )?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &state_buf,
            &state_data,
        )?;
    }

    // Create output buffer (f32)
    let out_size = (batch * channels * seq_len * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // ---- Dispatch 1: causal_conv1d.comp (output only, no state writes) ----
    let glsl_output = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d.comp"
    );
    let spirv_output = crate::pipeline::ShaderPipeline::compile_shader(glsl_output)?;

    // Bindings for output shader: x=0, weight=1, conv_state=2, out=3
    let output_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        weight_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let output_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let total = batch * channels * seq_len;
    let output_wg = ((total + 255) / 256) as u32;

    run_compute_pipeline(
        vk_device,
        &spirv_output,
        &output_handles,
        output_handles.len(),
        &output_push,
        output_wg,
    )?;

    // ---- Dispatch 2: causal_conv1d_state_advance.comp (state update only) ----
    // Each workgroup handles one (b, c) pair: batch * channels workgroups
    let glsl_state = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d_state_advance.comp"
    );
    let spirv_state = crate::pipeline::ShaderPipeline::compile_shader(glsl_state)?;

    // Bindings for state shader: x=0, conv_state=1
    let state_handles: Vec<vk::Buffer> = vec![x_buf.handle(), state_buf.handle()];
    let state_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let state_wg = (batch * channels) as u32;

    run_compute_pipeline(
        vk_device,
        &spirv_state,
        &state_handles,
        2,
        &state_push,
        state_wg,
    )?;

    // Read back both output and updated state
    let (out_data, state_data) = {
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )?;
        let state_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &state_buf,
        )?;
        (out_data, state_data)
    };

    // Cleanup
    drop(x_buf);
    drop(weight_buf);
    drop(state_buf);
    drop(out_buf);

    // Create output tensors
    let out_shape = x.dims().as_ref().to_vec();
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, DType::F32)?;
    let state_tensor =
        create_tensor_from_data(&state_data, conv_state.dims().as_ref(), DType::F32)?;
    Ok((out_tensor, state_tensor))
}

/// Dispatch causal_conv1d prefill kernel (multi-token path).
///
/// Depthwise conv1d with kernel_size=4, silu-fused.
/// `x`: `[B, C, T]` bf16. `weight`: `[C, K]` bf16. `conv_state`: `[B, C, K-1]` f32.
/// Returns `out: [B, C, T]` f32 and updates `conv_state` in-place.
///
/// Two-dispatch approach to avoid data races on conv_state:
/// 1. `causal_conv1d.comp` — computes output only (no state writes)
/// 2. `causal_conv1d_state_advance.comp` — advances state per (b, c) pair
pub fn dispatch_causal_conv1d_prefill(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
) -> Result<(Tensor, Tensor)> {
    if kernel_size != 4 {
        anyhow::bail!("causal_conv1d: only kernel_size=4 supported");
    }

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let x_data = extract_tensor_bytes(x)?.0;
    let weight_data = extract_tensor_bytes(weight)?.0;
    let state_data = extract_tensor_bytes(conv_state)?.0;

    // Parse shape [B, C, T]
    let dims = x.dims();
    let (batch, channels, seq_len) = (dims[0], dims[1], dims[2]);

    // Create input buffers + upload
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)?;
    let weight_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)?;
    let state_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &x_buf,
            &x_data,
        )?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &weight_buf,
            &weight_data,
        )?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &state_buf,
            &state_data,
        )?;
    }

    // Create output buffer (f32)
    let out_size = (batch * channels * seq_len * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // ---- Dispatch 1: causal_conv1d.comp (output only, no state writes) ----
    let glsl_output = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d.comp"
    );
    let spirv_output = crate::pipeline::ShaderPipeline::compile_shader(glsl_output)?;

    // Bindings for output shader: x=0, weight=1, conv_state=2, out=3
    let output_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        weight_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let output_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let total = batch * channels * seq_len;
    let output_wg = ((total + 255) / 256) as u32;

    run_compute_pipeline(
        vk_device,
        &spirv_output,
        &output_handles,
        output_handles.len(),
        &output_push,
        output_wg,
    )?;

    // ---- Dispatch 2: causal_conv1d_state_advance.comp (state update only) ----
    // Each workgroup handles one (b, c) pair: batch * channels workgroups
    let glsl_state = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d_state_advance.comp"
    );
    let spirv_state = crate::pipeline::ShaderPipeline::compile_shader(glsl_state)?;

    // Bindings for state shader: x=0, conv_state=1
    let state_handles: Vec<vk::Buffer> = vec![x_buf.handle(), state_buf.handle()];
    let state_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let state_wg = (batch * channels) as u32;

    run_compute_pipeline(
        vk_device,
        &spirv_state,
        &state_handles,
        2,
        &state_push,
        state_wg,
    )?;

    // Read back both output and updated state
    let (out_data, state_data) = {
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )?;
        let state_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &state_buf,
        )?;
        (out_data, state_data)
    };

    // Cleanup
    drop(x_buf);
    drop(weight_buf);
    drop(state_buf);
    drop(out_buf);

    // Create output tensors
    let out_shape = x.dims().as_ref().to_vec();
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, DType::F32)?;
    let state_tensor =
        create_tensor_from_data(&state_data, conv_state.dims().as_ref(), DType::F32)?;
    Ok((out_tensor, state_tensor))
}

/// Dispatch causal_conv1d prefill with an immutable cached f32 weight buffer.
///
/// This keeps the old tensor-weight entry point available as a rollback path,
/// while avoiding one per-layer weight upload and folding the two uploads, two
/// compute dispatches, and two readbacks into one command buffer/queue submit.
pub fn dispatch_causal_conv1d_prefill_cached_weight(
    vk_device: &VulkanDevice,
    x: &Tensor,
    weight_buf: &VulkanBuffer,
    conv_state: &Tensor,
    kernel_size: usize,
) -> Result<(Tensor, Tensor)> {
    if kernel_size != 4 {
        anyhow::bail!("causal_conv1d: only kernel_size=4 supported");
    }

    let device = vk_device.device();
    let device_local_mt = vk_device.device_local_mem_type();

    let x_data = extract_tensor_bytes(x)?.0;
    let state_data = extract_tensor_bytes(conv_state)?.0;

    let dims = x.dims();
    let (batch, channels, seq_len) = (dims[0], dims[1], dims[2]);

    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)?;
    let state_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    let out_size = (batch * channels * seq_len * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    let glsl_output = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d.comp"
    );
    let spirv_output = crate::pipeline::ShaderPipeline::compile_shader(glsl_output)?;
    let output_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        weight_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let output_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let output_wg = ((batch * channels * seq_len).div_ceil(256)) as u32;

    let glsl_state = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d_state_advance.comp"
    );
    let spirv_state = crate::pipeline::ShaderPipeline::compile_shader(glsl_state)?;
    let state_handles: Vec<vk::Buffer> = vec![x_buf.handle(), state_buf.handle()];
    let state_push: [u32; 4] = [
        batch as u32,
        channels as u32,
        seq_len as u32,
        kernel_size as u32,
    ];
    let state_wg = (batch * channels) as u32;

    let readbacks = run_two_stage_compute_pipeline_with_transfers(
        vk_device,
        &[(&x_buf, &x_data), (&state_buf, &state_data)],
        &[&out_buf, &state_buf],
        &spirv_output,
        &output_handles,
        &output_push,
        output_wg,
        &spirv_state,
        &state_handles,
        &state_push,
        state_wg,
    )
    .context("causal_conv1d prefill cached-weight single-submit failed")?;
    anyhow::ensure!(
        readbacks.len() == 2,
        "causal_conv1d prefill expected 2 readbacks, got {}",
        readbacks.len()
    );
    let out_data = &readbacks[0];
    let state_data = &readbacks[1];

    let out_shape = x.dims().as_ref().to_vec();
    let out_tensor = create_tensor_from_data(out_data, &out_shape, DType::F32)?;
    let state_tensor = create_tensor_from_data(state_data, conv_state.dims().as_ref(), DType::F32)?;
    Ok((out_tensor, state_tensor))
}

// ---------------------------------------------------------------------------
// Common pipeline build + dispatch helper to reduce code duplication
// ---------------------------------------------------------------------------

/// Dispatch a cached Vulkan compute pipeline and wait for completion.
///
/// This helper is used by causal_conv1d (two-dispatch path) and gdn
/// kernels. Pipeline state is cached on `VulkanDevice`; descriptor sets are
/// allocated from a reusable transient pool and command buffers remain
/// per-dispatch because they depend on live buffers.
pub fn run_compute_pipeline(
    vk_device: &VulkanDevice,
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    total_bindings: usize,
    push_constants: &[u32],
    workgroup_count: u32,
) -> Result<()> {
    // Use the actual device per-axis limit rather than the Vulkan
    // spec minimum (65535). Real devices typically support much
    // more (AMD/Strix Halo ≈ 2^31 - 1).
    let limit_x = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit_x,
        "run_compute_pipeline: workgroup_count={workgroup_count} \
         exceeds device per-axis limit {limit_x}; caller should \
         split into a multi-axis dispatch via dispatch_kernel"
    );
    let device = vk_device.device();
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        total_bindings,
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];

    anyhow::ensure!(
        total_bindings <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {total_bindings}"
    );
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    // Descriptor writes
    {
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let descriptor_write_infos: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, bui)| make_write_descriptor_set_buf(descriptor_set, i as u32, bui))
            .collect();
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    // Command buffer + dispatch
    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_compute_pipeline")?;

    // Cleanup
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    Ok(())
}

/// 3D-grid variant of `run_compute_pipeline`. Same caching/descriptor
/// machinery, but dispatches `(x, y, z)` workgroups for shaders that
/// use 2D (transpose) or 3D workgroup layouts.
pub fn run_compute_pipeline_3d(
    vk_device: &VulkanDevice,
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    total_bindings: usize,
    push_constants: &[u32],
    workgroup_count: (u32, u32, u32),
) -> Result<()> {
    let (wx, wy, wz) = workgroup_count;
    let limit_x = vk_device.max_compute_work_group_count(0);
    let limit_y = vk_device.max_compute_work_group_count(1);
    let limit_z = vk_device.max_compute_work_group_count(2);
    anyhow::ensure!(
        wx <= limit_x && wy <= limit_y && wz <= limit_z,
        "run_compute_pipeline_3d: workgroups ({wx},{wy},{wz}) exceed device limits \
         ({limit_x},{limit_y},{limit_z})"
    );
    let device = vk_device.device();
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        total_bindings,
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    anyhow::ensure!(
        total_bindings <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {total_bindings}"
    );
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate descriptor sets")?[0]
    };
    {
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let descriptor_write_infos: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, bui)| make_write_descriptor_set_buf(descriptor_set, i as u32, bui))
            .collect();
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }
    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];
    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, wx, wy, wz);
        let barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
    }
    vk_device.submit_and_wait(cmd, "run_compute_pipeline_3d")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }
    Ok(())
}

/// Single-submit upload + dispatch + readback. Sequences a host-to-device
/// copy of `upload_data` into `upload_dst`, runs one compute kernel, then
/// copies `readback_size` bytes from `readback_src` into a host-visible
/// staging buffer — all in one command buffer and one queue submit. Saves
/// the two extra `vkQueueSubmit` + fence-wait round trips the
/// `extract → upload → dispatch → readback` decode kernels otherwise pay
/// per call (≈ 600 µs on NVIDIA Vulkan).
#[allow(clippy::too_many_arguments)]
fn run_compute_pipeline_with_transfer_readback(
    vk_device: &VulkanDevice,
    upload_dst: &VulkanBuffer,
    upload_data: &[u8],
    readback_src: &VulkanBuffer,
    readback_size: u64,
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    push_constants: &[u32],
    workgroup_count: u32,
) -> Result<Vec<u8>> {
    let limit_x = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit_x,
        "run_compute_pipeline_with_transfer_readback: workgroup_count={workgroup_count} \
         exceeds device per-axis limit {limit_x}"
    );
    let device = vk_device.device();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let upload_stage =
        VulkanBuffer::create_host_visible(device, host_visible_mt, upload_data.len() as u64)
            .context("failed to create transfer-readback upload staging buffer")?;
    VulkanBuffer::write_host_visible(device, &upload_stage, upload_data)?;
    let readback_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, readback_size)
        .context("failed to create transfer-readback readback staging buffer")?;

    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    anyhow::ensure!(
        all_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        all_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [set_layout];
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate transfer-readback descriptor set")?[0]
    };

    {
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
            .collect();
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate transfer-readback command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin transfer-readback command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            upload_stage.handle(),
            upload_dst.handle(),
            &[vk::BufferCopy::builder()
                .size(upload_data.len() as u64)
                .build()],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[readback_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            readback_src.handle(),
            readback_stage.handle(),
            &[vk::BufferCopy::builder().size(readback_size).build()],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end transfer-readback command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_compute_pipeline_with_transfer_readback")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transfer-readback descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    VulkanBuffer::read_host_visible(device, &readback_stage)
        .context("failed to read transfer-readback output")
}

/// Single-submit multi-upload + dispatch + readback. Variant of
/// `run_compute_pipeline_with_transfer_readback` for kernels that take
/// several disjoint input buffers (e.g. paged_attn_decode_batch's
/// Q/K/V/seq_lens uploads). Schedules all host→device copies, the compute
/// dispatch, and a single device→host readback into one command buffer.
#[allow(clippy::too_many_arguments)]
fn run_compute_pipeline_with_transfers_readback(
    vk_device: &VulkanDevice,
    uploads: &[(&VulkanBuffer, &[u8])],
    readback_src: &VulkanBuffer,
    readback_size: u64,
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    push_constants: &[u32],
    workgroup_count: u32,
) -> Result<Vec<u8>> {
    let limit_x = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit_x,
        "run_compute_pipeline_with_transfers_readback: workgroup_count={workgroup_count} \
         exceeds device per-axis limit {limit_x}"
    );
    let device = vk_device.device();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let mut upload_stages = Vec::with_capacity(uploads.len());
    for (_, data) in uploads {
        let stage = VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)
            .context("failed to create transfers-readback upload staging buffer")?;
        VulkanBuffer::write_host_visible(device, &stage, data)?;
        upload_stages.push(stage);
    }
    let readback_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, readback_size)
        .context("failed to create transfers-readback readback staging buffer")?;

    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    anyhow::ensure!(
        all_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        all_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [set_layout];
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate transfers-readback descriptor set")?[0]
    };

    {
        let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buf_infos
            .iter()
            .enumerate()
            .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
            .collect();
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate transfers-readback command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin transfers-readback command buffer")?;
        for ((dst, data), stage) in uploads.iter().zip(upload_stages.iter()) {
            device.cmd_copy_buffer(
                cmd,
                stage.handle(),
                dst.handle(),
                &[vk::BufferCopy::builder().size(data.len() as u64).build()],
            );
        }
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[readback_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            readback_src.handle(),
            readback_stage.handle(),
            &[vk::BufferCopy::builder().size(readback_size).build()],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end transfers-readback command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_compute_pipeline_with_transfers_readback")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transfers-readback descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    VulkanBuffer::read_host_visible(device, &readback_stage)
        .context("failed to read transfers-readback output")
}

#[allow(clippy::too_many_arguments)]
fn run_two_stage_compute_pipeline(
    vk_device: &VulkanDevice,
    first_spirv: &[u8],
    first_handles: &[vk::Buffer],
    first_push_constants: &[u32],
    first_workgroup_count: u32,
    second_spirv: &[u8],
    second_handles: &[vk::Buffer],
    second_push_constants: &[u32],
    second_workgroup_count: u32,
) -> Result<()> {
    let device = vk_device.device();
    let (first_set_layout, first_layout, first_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            first_spirv,
            first_handles.len(),
            (first_push_constants.len() * 4) as u32,
        )?;
    let (second_set_layout, second_layout, second_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            second_spirv,
            second_handles.len(),
            (second_push_constants.len() * 4) as u32,
        )?;
    anyhow::ensure!(
        first_handles.len() + second_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        first_handles.len() + second_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [first_set_layout, second_set_layout];
    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate two-stage descriptor sets")?
    };
    let first_descriptor_set = descriptor_sets[0];
    let second_descriptor_set = descriptor_sets[1];

    {
        let first_buf_infos: Vec<vk::DescriptorBufferInfo> = first_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let second_buf_infos: Vec<vk::DescriptorBufferInfo> = second_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let mut descriptor_write_infos: Vec<vk::WriteDescriptorSet> =
            Vec::with_capacity(first_buf_infos.len() + second_buf_infos.len());
        descriptor_write_infos.extend(
            first_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(first_descriptor_set, i as u32, info)
            }),
        );
        descriptor_write_infos.extend(
            second_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(second_descriptor_set, i as u32, info)
            }),
        );
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate two-stage command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin two-stage command buffer")?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, first_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            first_layout,
            0,
            &[first_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            first_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(first_push_constants),
        );
        device.cmd_dispatch(cmd, first_workgroup_count, 1, 1);

        let first_to_second_barrier =
            make_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[first_to_second_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, second_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            second_layout,
            0,
            &[second_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            second_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(second_push_constants),
        );
        device.cmd_dispatch(cmd, second_workgroup_count, 1, 1);

        let second_to_readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[second_to_readback_barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end two-stage command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_two_stage_compute_pipeline")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset two-stage transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_two_stage_compute_pipeline_with_transfer_readback(
    vk_device: &VulkanDevice,
    upload_dst: &VulkanBuffer,
    upload_data: &[u8],
    readback_src: &VulkanBuffer,
    readback_size: u64,
    first_spirv: &[u8],
    first_handles: &[vk::Buffer],
    first_push_constants: &[u32],
    first_workgroup_count: u32,
    second_spirv: &[u8],
    second_handles: &[vk::Buffer],
    second_push_constants: &[u32],
    second_workgroup_count: u32,
) -> Result<Vec<u8>> {
    let device = vk_device.device();
    let host_visible_mt = vk_device.host_visible_mem_type();
    let upload_stage =
        VulkanBuffer::create_host_visible(device, host_visible_mt, upload_data.len() as u64)
            .context("failed to create two-stage upload staging buffer")?;
    VulkanBuffer::write_host_visible(device, &upload_stage, upload_data)?;
    let readback_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, readback_size)
        .context("failed to create two-stage readback staging buffer")?;

    let (first_set_layout, first_layout, first_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            first_spirv,
            first_handles.len(),
            (first_push_constants.len() * 4) as u32,
        )?;
    let (second_set_layout, second_layout, second_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            second_spirv,
            second_handles.len(),
            (second_push_constants.len() * 4) as u32,
        )?;
    anyhow::ensure!(
        first_handles.len() + second_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        first_handles.len() + second_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [first_set_layout, second_set_layout];
    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate transfer two-stage descriptor sets")?
    };
    let first_descriptor_set = descriptor_sets[0];
    let second_descriptor_set = descriptor_sets[1];

    {
        let first_buf_infos: Vec<vk::DescriptorBufferInfo> = first_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let second_buf_infos: Vec<vk::DescriptorBufferInfo> = second_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let mut descriptor_write_infos: Vec<vk::WriteDescriptorSet> =
            Vec::with_capacity(first_buf_infos.len() + second_buf_infos.len());
        descriptor_write_infos.extend(
            first_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(first_descriptor_set, i as u32, info)
            }),
        );
        descriptor_write_infos.extend(
            second_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(second_descriptor_set, i as u32, info)
            }),
        );
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate transfer two-stage command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin transfer two-stage command buffer")?;
        device.cmd_copy_buffer(
            cmd,
            upload_stage.handle(),
            upload_dst.handle(),
            &[vk::BufferCopy::builder()
                .size(upload_data.len() as u64)
                .build()],
        );
        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, first_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            first_layout,
            0,
            &[first_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            first_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(first_push_constants),
        );
        device.cmd_dispatch(cmd, first_workgroup_count, 1, 1);

        let first_to_second_barrier =
            make_memory_barrier(vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::SHADER_READ);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[first_to_second_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, second_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            second_layout,
            0,
            &[second_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            second_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(second_push_constants),
        );
        device.cmd_dispatch(cmd, second_workgroup_count, 1, 1);

        let second_to_readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[second_to_readback_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            readback_src.handle(),
            readback_stage.handle(),
            &[vk::BufferCopy::builder().size(readback_size).build()],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end transfer two-stage command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_two_stage_compute_pipeline_with_transfer_readback")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transfer two-stage transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    VulkanBuffer::read_host_visible(device, &readback_stage)
        .context("failed to read transfer two-stage output")
}

#[allow(clippy::too_many_arguments)]
fn run_two_stage_compute_pipeline_with_transfers(
    vk_device: &VulkanDevice,
    uploads: &[(&VulkanBuffer, &[u8])],
    readbacks: &[&VulkanBuffer],
    first_spirv: &[u8],
    first_handles: &[vk::Buffer],
    first_push_constants: &[u32],
    first_workgroup_count: u32,
    second_spirv: &[u8],
    second_handles: &[vk::Buffer],
    second_push_constants: &[u32],
    second_workgroup_count: u32,
) -> Result<Vec<Vec<u8>>> {
    let device = vk_device.device();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let mut upload_stages = Vec::with_capacity(uploads.len());
    for (_, data) in uploads {
        let stage = VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)
            .context("failed to create two-stage upload staging buffer")?;
        VulkanBuffer::write_host_visible(device, &stage, data)?;
        upload_stages.push(stage);
    }

    let mut readback_stages = Vec::with_capacity(readbacks.len());
    for buffer in readbacks {
        readback_stages.push(
            VulkanBuffer::create_host_visible(device, host_visible_mt, buffer.size())
                .context("failed to create two-stage readback staging buffer")?,
        );
    }

    let (first_set_layout, first_layout, first_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            first_spirv,
            first_handles.len(),
            (first_push_constants.len() * 4) as u32,
        )?;
    let (second_set_layout, second_layout, second_pipeline) = vk_device
        .get_or_create_compute_pipeline(
            second_spirv,
            second_handles.len(),
            (second_push_constants.len() * 4) as u32,
        )?;
    anyhow::ensure!(
        first_handles.len() + second_handles.len() <= 64,
        "Vulkan transient descriptor pool only supports up to 64 bindings, got {}",
        first_handles.len() + second_handles.len()
    );

    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let set_layouts = [first_set_layout, second_set_layout];
    let descriptor_sets = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate transfer two-stage descriptor sets")?
    };
    let first_descriptor_set = descriptor_sets[0];
    let second_descriptor_set = descriptor_sets[1];

    {
        let first_buf_infos: Vec<vk::DescriptorBufferInfo> = first_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let second_buf_infos: Vec<vk::DescriptorBufferInfo> = second_handles
            .iter()
            .map(|&h| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(h)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
                    .build()
            })
            .collect();
        let mut descriptor_write_infos: Vec<vk::WriteDescriptorSet> =
            Vec::with_capacity(first_buf_infos.len() + second_buf_infos.len());
        descriptor_write_infos.extend(
            first_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(first_descriptor_set, i as u32, info)
            }),
        );
        descriptor_write_infos.extend(
            second_buf_infos.iter().enumerate().map(|(i, info)| {
                make_write_descriptor_set_buf(second_descriptor_set, i as u32, info)
            }),
        );
        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate transfer two-stage command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin transfer two-stage command buffer")?;

        for ((dst, data), stage) in uploads.iter().zip(upload_stages.iter()) {
            device.cmd_copy_buffer(
                cmd,
                stage.handle(),
                dst.handle(),
                &[vk::BufferCopy::builder().size(data.len() as u64).build()],
            );
        }
        if !uploads.is_empty() {
            let upload_barrier = make_memory_barrier(
                vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[upload_barrier],
                &[],
                &[],
            );
        }

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, first_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            first_layout,
            0,
            &[first_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            first_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(first_push_constants),
        );
        device.cmd_dispatch(cmd, first_workgroup_count, 1, 1);

        let first_to_second_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[first_to_second_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, second_pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            second_layout,
            0,
            &[second_descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            second_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(second_push_constants),
        );
        device.cmd_dispatch(cmd, second_workgroup_count, 1, 1);

        let readback_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[readback_barrier],
            &[],
            &[],
        );

        for (src, stage) in readbacks.iter().zip(readback_stages.iter()) {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                stage.handle(),
                &[vk::BufferCopy::builder().size(src.size()).build()],
            );
        }

        device
            .end_command_buffer(cmd)
            .context("failed to end transfer two-stage command buffer")?;
    }

    vk_device.submit_and_wait(cmd, "run_two_stage_compute_pipeline_with_transfers")?;
    unsafe {
        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transfer two-stage transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    readback_stages
        .iter()
        .map(|stage| VulkanBuffer::read_host_visible(device, stage))
        .collect::<Result<Vec<_>>>()
        .context("failed to read transfer two-stage outputs")
}

// ---------------------------------------------------------------------------
// GDN forward substitution (triangular solve) kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN forward substitution kernel.
///
/// Computes W = (I + A_strict)^{-1} (beta * V_prime)
/// A_strict: [B,H,C,C] lower-triangular, V_prime: [B,H,C,dv], beta: [B,H,C]
/// Output: W: [B,H,C,dv]
pub fn dispatch_gdn_forward_substitution(
    vk_device: &VulkanDevice,
    a_strict: &Tensor,
    v_prime: &Tensor,
    beta: &Tensor,
) -> Result<Tensor> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let a_strict_data = extract_tensor_bytes(a_strict)?.0;
    let v_prime_data = extract_tensor_bytes(v_prime)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;

    // Compile shader
    let glsl_path = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/solve_tri.comp");
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Parse output shape [B, H, C, dv]
    let dims = v_prime.dims();
    let (batch, heads, chunk, dv) = (dims[0], dims[1], dims[2], dims[3]);

    // Create input buffers + upload
    let a_strict_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, a_strict_data.len() as u64)?;
    VulkanBuffer::upload_data(
        device,
        host_visible_mt,
        queue,
        qfi,
        &a_strict_buf,
        &a_strict_data,
    )?;

    let v_prime_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, v_prime_data.len() as u64)?;
    VulkanBuffer::upload_data(
        device,
        host_visible_mt,
        queue,
        qfi,
        &v_prime_buf,
        &v_prime_data,
    )?;

    let beta_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, beta_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &beta_buf, &beta_data)?;

    // Create output buffer (f32)
    let out_size = (batch * heads * chunk * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, heads, chunk, dv
    let push_constants: [u32; 4] = [batch as u32, heads as u32, chunk as u32, dv as u32];

    // Workgroup count: total elements / 256
    let total = batch * heads * chunk * dv;
    let workgroup_count = ((total + 255) / 256) as u32;

    // Bindings: A_strict=0, V_prime=1, beta=2, out=3
    let all_handles = vec![
        a_strict_buf.handle(),
        v_prime_buf.handle(),
        beta_buf.handle(),
        out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back output
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;

    // Cleanup
    drop(a_strict_buf);
    drop(v_prime_buf);
    drop(beta_buf);
    drop(out_buf);

    let out_shape = vec![batch, heads, chunk, dv];
    create_tensor_from_data(&out_data, &out_shape, DType::BF16)
        .context("failed to create gdn_forward_substitution output tensor")
}

// ---------------------------------------------------------------------------
// GDN recurrent step kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN recurrent step kernel.
///
/// Recurrent state update for GDN.
/// Q: [B,H,dk], K: [B,H,dk], V: [B,H,dv], beta: [B,H], g: [B,H]
/// State: [B,H,dk,dv] (in/out), Output: [B,H,dv]
pub fn dispatch_gdn_recurrent_step(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (out, state) =
        dispatch_gdn_recurrent_step_with_options(vk_device, q, k, v, beta, g, state, false)?;
    let state = state.context("gdn recurrent state was unexpectedly skipped")?;
    Ok((out, state))
}

pub fn copy_gdn_recurrent_state_rows_to_batch(
    vk_device: &VulkanDevice,
    rows: &[Arc<VulkanBuffer>],
) -> Result<Arc<VulkanBuffer>> {
    anyhow::ensure!(
        !rows.is_empty(),
        "copy_gdn_recurrent_state_rows_to_batch requires at least one row"
    );
    let row_size = rows[0].size();
    anyhow::ensure!(
        row_size > 0,
        "copy_gdn_recurrent_state_rows_to_batch row size must be non-zero"
    );
    for (idx, row) in rows.iter().enumerate() {
        anyhow::ensure!(
            row.size() == row_size,
            "copy_gdn_recurrent_state_rows_to_batch row {idx} size {} != row 0 size {row_size}",
            row.size()
        );
    }

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let batch_buf = Arc::new(VulkanBuffer::create_device_local(
        device,
        device_local_mt,
        row_size * rows.len() as u64,
    )?);

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate recurrent row-to-batch copy command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin recurrent row-to-batch copy command buffer")?;
        for (row_idx, row) in rows.iter().enumerate() {
            device.cmd_copy_buffer(
                cmd,
                row.handle(),
                batch_buf.handle(),
                &[vk::BufferCopy::builder()
                    .src_offset(0)
                    .dst_offset(row_size * row_idx as u64)
                    .size(row_size)
                    .build()],
            );
        }
        let copy_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[copy_barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end recurrent row-to-batch copy command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit recurrent row-to-batch copy")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for recurrent row-to-batch copy")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    Ok(batch_buf)
}

pub fn split_gdn_recurrent_state_batch_rows(
    vk_device: &VulkanDevice,
    batch_buffer: &VulkanBuffer,
    batch: usize,
) -> Result<Vec<Arc<VulkanBuffer>>> {
    anyhow::ensure!(
        batch > 0,
        "split_gdn_recurrent_state_batch_rows requires a non-zero batch"
    );
    anyhow::ensure!(
        batch_buffer.size() % batch as u64 == 0,
        "split_gdn_recurrent_state_batch_rows buffer size {} is not divisible by batch {batch}",
        batch_buffer.size()
    );
    let row_size = batch_buffer.size() / batch as u64;
    anyhow::ensure!(
        row_size > 0,
        "split_gdn_recurrent_state_batch_rows row size must be non-zero"
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let mut rows = Vec::with_capacity(batch);
    for _ in 0..batch {
        rows.push(Arc::new(VulkanBuffer::create_device_local(
            device,
            device_local_mt,
            row_size,
        )?));
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate recurrent batch-to-row copy command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin recurrent batch-to-row copy command buffer")?;
        let pre_copy_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[pre_copy_barrier],
            &[],
            &[],
        );
        for (row_idx, row) in rows.iter().enumerate() {
            device.cmd_copy_buffer(
                cmd,
                batch_buffer.handle(),
                row.handle(),
                &[vk::BufferCopy::builder()
                    .src_offset(row_size * row_idx as u64)
                    .dst_offset(0)
                    .size(row_size)
                    .build()],
            );
        }
        let post_copy_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[post_copy_barrier],
            &[],
            &[],
        );
        device
            .end_command_buffer(cmd)
            .context("failed to end recurrent batch-to-row copy command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit recurrent batch-to-row copy")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for recurrent batch-to-row copy")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    Ok(rows)
}

pub fn dispatch_gdn_recurrent_step_with_options(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
    skip_state_readback: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let profile_kernel_stages = profile_vulkan_gdn_recurrent_kernel_stages_enabled();
    let stage_profile = profile_kernel_stages.then(Instant::now);
    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let g_data = extract_tensor_bytes(g)?.0;

    // Parse shape [B, H, dk/dv].
    let dims = q.dims();
    let (batch, heads, dk) = (dims[0], dims[1], dims[2]);
    let dims_v = v.dims();
    let dv = dims_v[2];

    let single_submit = gdn_recurrent_single_submit_enabled();
    let parallel_reduce = single_submit && use_gdn_recurrent_parallel_reduce(dk, dv);
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "extract_inputs",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        single_submit,
        skip_state_readback,
        stage_profile,
    );
    let stage_profile = profile_kernel_stages.then(Instant::now);
    let state_data = extract_tensor_bytes(state)?.0;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "extract_state",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        single_submit,
        skip_state_readback,
        stage_profile,
    );
    let stage_profile = profile_kernel_stages.then(Instant::now);
    let glsl_path = if parallel_reduce {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_step_parallel.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_prefill.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "compile_shader",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        single_submit,
        skip_state_readback,
        stage_profile,
    );

    if single_submit {
        return dispatch_gdn_recurrent_step_single_submit(
            vk_device,
            q,
            state,
            &q_data,
            &k_data,
            &v_data,
            &beta_data,
            &g_data,
            &state_data,
            &spirv,
            batch,
            heads,
            heads,
            dk,
            dv,
            parallel_reduce,
            skip_state_readback,
            profile_kernel_stages,
            q.dtype(),
            None,
        );
    }

    // Create input buffers + upload.
    let make_input_buf = |data: &[u8]| -> Result<VulkanBuffer> {
        let buf = VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &buf,
            data,
        )?;
        Ok(buf)
    };
    let q_buf = make_input_buf(&q_data)?;
    let k_buf = make_input_buf(&k_data)?;
    let v_buf = make_input_buf(&v_data)?;
    let beta_buf = make_input_buf(&beta_data)?;
    let g_buf = make_input_buf(&g_data)?;
    // State is mutable — upload, dispatch, read back. On Strix Halo, direct
    // host-visible state is faster for batch 1, while batch >1 benefits from
    // device-local state plus explicit staging copies.
    let host_visible_state = gdn_recurrent_use_host_visible_state(batch);
    let state_buf = if host_visible_state {
        VulkanBuffer::create_host_visible(device, host_visible_mt, state_data.len() as u64)?
    } else {
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?
    };
    if host_visible_state {
        VulkanBuffer::write_host_visible(device, &state_buf, &state_data)?;
    } else {
        let command_pool = vk_device.transient_command_pool()?;
        VulkanBuffer::upload_data_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &state_buf,
            &state_data,
        )?;
    }

    // Create output buffer (f32 shader output, converted to bf16 below).
    let out_size = (batch * heads * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, value heads, seq_len, dk, dv, q/k heads. seq_len
    // is always 1 for this single-token kernel.
    let push_constants: [u32; 6] = [
        batch as u32,
        heads as u32,
        1,
        dk as u32,
        dv as u32,
        heads as u32,
    ];

    // Workgroup count: total elements / 256
    let total = batch * heads * dv;
    let workgroup_count = total.div_ceil(256) as u32;

    // Bindings: Q=0, K=1, V=2, beta=3, g=4, state=5, out=6
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        beta_buf.handle(),
        g_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back output and updated state
    let (out_data, state_data) = {
        let command_pool = vk_device.transient_command_pool()?;
        let out_data = VulkanBuffer::read_back_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &out_buf,
        )?;
        let state_data = if skip_state_readback {
            None
        } else if host_visible_state {
            Some(VulkanBuffer::read_host_visible(device, &state_buf)?)
        } else {
            Some(VulkanBuffer::read_back_with_command_pool(
                device,
                host_visible_mt,
                queue,
                *command_pool,
                &state_buf,
            )?)
        };
        (out_data, state_data)
    };

    // Cleanup
    drop(q_buf);
    drop(k_buf);
    drop(v_buf);
    drop(beta_buf);
    drop(g_buf);
    drop(state_buf);
    drop(out_buf);

    let out_shape = vec![batch, heads, dv];
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, q.dtype())?;
    let state_tensor = state_data
        .as_ref()
        .map(|state_data| create_tensor_from_data(state_data, state.dims().as_ref(), state.dtype()))
        .transpose()?;
    Ok((out_tensor, state_tensor))
}

/// Dispatch a single-token recurrent step while keeping `state` resident.
///
/// The first call uploads the CPU state into a device-local Vulkan buffer and
/// returns it. Later calls can pass that buffer back and avoid the full state
/// upload/readback pair; only the small recurrent output is copied to the CPU.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_step_resident_state(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
    resident_state: Option<Arc<VulkanBuffer>>,
) -> Result<(Tensor, Arc<VulkanBuffer>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let g_data = extract_tensor_bytes(g)?.0;
    let state_data = if resident_state.is_none() {
        Some(extract_tensor_bytes(state)?.0)
    } else {
        None
    };

    let dims = q.dims();
    let (batch, heads, dk) = (dims[0], dims[1], dims[2]);
    let dims_v = v.dims();
    let dv = dims_v[2];

    let parallel_reduce = use_gdn_recurrent_parallel_reduce(dk, dv);
    let glsl_path = if parallel_reduce {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_step_parallel.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_prefill.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    let make_device_and_staging = |data: &[u8]| -> Result<(VulkanBuffer, VulkanBuffer)> {
        let device_buf =
            VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Ok((device_buf, staging))
    };

    let (q_buf, q_stage) = make_device_and_staging(&q_data)?;
    let (k_buf, k_stage) = make_device_and_staging(&k_data)?;
    let (v_buf, v_stage) = make_device_and_staging(&v_data)?;
    let (beta_buf, beta_stage) = make_device_and_staging(&beta_data)?;
    let (g_buf, g_stage) = make_device_and_staging(&g_data)?;

    let state_buf = match resident_state {
        Some(buffer) => buffer,
        None => {
            let data = state_data
                .as_ref()
                .expect("state data exists when resident state is absent");
            Arc::new(VulkanBuffer::create_device_local(
                device,
                device_local_mt,
                data.len() as u64,
            )?)
        }
    };
    let state_stage = if let Some(data) = &state_data {
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Some(staging)
    } else {
        None
    };

    let out_size = (batch * heads * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)?;

    let push_constants: [u32; 6] = [
        batch as u32,
        heads as u32,
        1,
        dk as u32,
        dv as u32,
        heads as u32,
    ];
    let total = batch * heads * dv;
    let workgroup_count = total.div_ceil(256) as u32;
    let dispatch_counts = if parallel_reduce {
        (batch as u32, heads as u32, dv as u32)
    } else {
        (workgroup_count, 1, 1)
    };
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        beta_buf.handle(),
        g_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];

    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        &spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (src, dst, size) in [
            (&q_stage, &q_buf, q_data.len() as u64),
            (&k_stage, &k_buf, k_data.len() as u64),
            (&v_stage, &v_buf, v_data.len() as u64),
            (&beta_stage, &beta_buf, beta_data.len() as u64),
            (&g_stage, &g_buf, g_data.len() as u64),
        ] {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                dst.handle(),
                &[vk::BufferCopy::builder().size(size).build()],
            );
        }
        if let (Some(state_stage), Some(state_data)) = (&state_stage, &state_data) {
            device.cmd_copy_buffer(
                cmd,
                state_stage.handle(),
                state_buf.handle(),
                &[vk::BufferCopy::builder()
                    .size(state_data.len() as u64)
                    .build()],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, dispatch_counts.0, dispatch_counts.1, dispatch_counts.2);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size(out_size).build()],
        );

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn recurrent resident-state dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn recurrent resident-state dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    let out_shape = vec![batch, heads, dv];
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, q.dtype())?;
    Ok((out_tensor, state_buf))
}

/// Dispatch a native-head single-token recurrent step while keeping `state`
/// resident. `q`/`k` are `[batch, 1, q_heads, dk]`; value-side tensors and
/// state use `heads`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_step_native_head_last_resident_state(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
    resident_state: Option<Arc<VulkanBuffer>>,
) -> Result<(Tensor, Arc<VulkanBuffer>)> {
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (k_batch, k_seq_len, k_heads, k_dk) = k.dims4()?;
    let (v_batch, v_seq_len, heads, dv) = v.dims4()?;
    let (beta_batch, beta_seq_len, beta_heads) = beta.dims3()?;
    let (g_batch, g_seq_len, g_heads) = g.dims3()?;
    let (state_batch, state_heads, state_dk, state_dv) = state.dims4()?;

    anyhow::ensure!(
        seq_len == 1,
        "native-head resident recurrent expects seq_len=1"
    );
    anyhow::ensure!(
        (k_batch, k_seq_len, k_heads, k_dk) == (batch, seq_len, q_heads, dk),
        "native-head resident recurrent k shape mismatch"
    );
    anyhow::ensure!(
        (v_batch, v_seq_len) == (batch, seq_len),
        "native-head resident recurrent v batch/seq mismatch"
    );
    anyhow::ensure!(
        (beta_batch, beta_seq_len, beta_heads) == (batch, seq_len, heads),
        "native-head resident recurrent beta shape mismatch"
    );
    anyhow::ensure!(
        (g_batch, g_seq_len, g_heads) == (batch, seq_len, heads),
        "native-head resident recurrent g shape mismatch"
    );
    anyhow::ensure!(
        (state_batch, state_heads, state_dk, state_dv) == (batch, heads, dk, dv),
        "native-head resident recurrent state shape mismatch"
    );
    anyhow::ensure!(
        q_heads > 0,
        "native-head resident recurrent q_heads must be positive"
    );
    anyhow::ensure!(
        heads % q_heads == 0,
        "native-head resident recurrent heads {heads} must be divisible by q_heads {q_heads}"
    );

    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let g_data = extract_tensor_bytes(g)?.0;
    let state_data = if resident_state.is_none() {
        Some(extract_tensor_bytes(state)?.0)
    } else {
        None
    };

    let parallel_reduce = use_gdn_recurrent_parallel_reduce(dk, dv);
    let glsl_path = if parallel_reduce {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_step_parallel.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_prefill.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    let make_device_and_staging = |data: &[u8]| -> Result<(VulkanBuffer, VulkanBuffer)> {
        let device_buf =
            VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Ok((device_buf, staging))
    };

    let (q_buf, q_stage) = make_device_and_staging(&q_data)?;
    let (k_buf, k_stage) = make_device_and_staging(&k_data)?;
    let (v_buf, v_stage) = make_device_and_staging(&v_data)?;
    let (beta_buf, beta_stage) = make_device_and_staging(&beta_data)?;
    let (g_buf, g_stage) = make_device_and_staging(&g_data)?;

    let state_buf = match resident_state {
        Some(buffer) => buffer,
        None => {
            let data = state_data
                .as_ref()
                .expect("state data exists when resident state is absent");
            Arc::new(VulkanBuffer::create_device_local(
                device,
                device_local_mt,
                data.len() as u64,
            )?)
        }
    };
    let state_stage = if let Some(data) = &state_data {
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Some(staging)
    } else {
        None
    };

    let out_size = (batch * heads * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)?;

    let push_constants: [u32; 6] = [
        batch as u32,
        heads as u32,
        1,
        dk as u32,
        dv as u32,
        q_heads as u32,
    ];
    let total = batch * heads * dv;
    let workgroup_count = total.div_ceil(256) as u32;
    let dispatch_counts = if parallel_reduce {
        (batch as u32, heads as u32, dv as u32)
    } else {
        (workgroup_count, 1, 1)
    };
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        beta_buf.handle(),
        g_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];

    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        &spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }

    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (src, dst, size) in [
            (&q_stage, &q_buf, q_data.len() as u64),
            (&k_stage, &k_buf, k_data.len() as u64),
            (&v_stage, &v_buf, v_data.len() as u64),
            (&beta_stage, &beta_buf, beta_data.len() as u64),
            (&g_stage, &g_buf, g_data.len() as u64),
        ] {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                dst.handle(),
                &[vk::BufferCopy::builder().size(size).build()],
            );
        }
        if let (Some(state_stage), Some(state_data)) = (&state_stage, &state_data) {
            device.cmd_copy_buffer(
                cmd,
                state_stage.handle(),
                state_buf.handle(),
                &[vk::BufferCopy::builder()
                    .size(state_data.len() as u64)
                    .build()],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, dispatch_counts.0, dispatch_counts.1, dispatch_counts.2);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );
        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size(out_size).build()],
        );

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit native-head gdn recurrent resident-state dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for native-head gdn recurrent resident-state dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }

    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    let out_shape = vec![batch, heads, dv];
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, q.dtype())?;
    Ok((out_tensor.unsqueeze(1)?, state_buf))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_gdn_recurrent_step_single_submit(
    vk_device: &VulkanDevice,
    _q: &Tensor,
    state: &Tensor,
    q_data: &[u8],
    k_data: &[u8],
    v_data: &[u8],
    beta_data: &[u8],
    g_data: &[u8],
    state_data: &[u8],
    spirv: &[u8],
    batch: usize,
    heads: usize,
    q_heads: usize,
    dk: usize,
    dv: usize,
    parallel_reduce: bool,
    skip_state_readback: bool,
    profile_kernel_stages: bool,
    output_dtype: DType,
    dispatch_counts_override: Option<(u32, u32, u32)>,
) -> Result<(Tensor, Option<Tensor>)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    let make_device_and_staging = |data: &[u8]| -> Result<(VulkanBuffer, VulkanBuffer)> {
        let device_buf =
            VulkanBuffer::create_device_local(device, device_local_mt, data.len() as u64)?;
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, data)?;
        Ok((device_buf, staging))
    };

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let (q_buf, q_stage) = make_device_and_staging(q_data)?;
    let (k_buf, k_stage) = make_device_and_staging(k_data)?;
    let (v_buf, v_stage) = make_device_and_staging(v_data)?;
    let (beta_buf, beta_stage) = make_device_and_staging(beta_data)?;
    let (g_buf, g_stage) = make_device_and_staging(g_data)?;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "make_input_staging",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let host_visible_state = gdn_recurrent_use_host_visible_state(batch);
    let state_buf = if host_visible_state {
        let buf =
            VulkanBuffer::create_host_visible(device, host_visible_mt, state_data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &buf, state_data)?;
        buf
    } else {
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?
    };
    let state_stage = if host_visible_state {
        None
    } else {
        let staging =
            VulkanBuffer::create_host_visible(device, host_visible_mt, state_data.len() as u64)?;
        VulkanBuffer::write_host_visible(device, &staging, state_data)?;
        Some(staging)
    };
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "make_state_staging",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let out_size = (batch * heads * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;
    let out_stage = VulkanBuffer::create_host_visible(device, host_visible_mt, out_size)?;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "create_output_buffers",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let push_constants: [u32; 6] = [
        batch as u32,
        heads as u32,
        1,
        dk as u32,
        dv as u32,
        q_heads as u32,
    ];
    let total = batch * heads * dv;
    let workgroup_count = total.div_ceil(256) as u32;
    let dispatch_counts = if parallel_reduce {
        (batch as u32, heads as u32, dv as u32)
    } else {
        (workgroup_count, 1, 1)
    };
    let dispatch_counts = dispatch_counts_override.unwrap_or(dispatch_counts);
    let all_handles = vec![
        q_buf.handle(),
        k_buf.handle(),
        v_buf.handle(),
        beta_buf.handle(),
        g_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let (set_layout, layout, pipeline) = vk_device.get_or_create_compute_pipeline(
        spirv,
        all_handles.len(),
        (push_constants.len() * 4) as u32,
    )?;
    let set_layouts = vec![set_layout];
    let descriptor_pool = vk_device.transient_descriptor_pool()?;
    let descriptor_set = unsafe {
        device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .descriptor_pool(*descriptor_pool)
                    .set_layouts(&set_layouts)
                    .build(),
            )
            .context("failed to allocate descriptor sets")?[0]
    };

    let buf_infos: Vec<vk::DescriptorBufferInfo> = all_handles
        .iter()
        .map(|&h| {
            vk::DescriptorBufferInfo::builder()
                .buffer(h)
                .offset(0)
                .range(vk::WHOLE_SIZE)
                .build()
        })
        .collect();
    let descriptor_writes: Vec<vk::WriteDescriptorSet> = buf_infos
        .iter()
        .enumerate()
        .map(|(i, info)| make_write_descriptor_set_buf(descriptor_set, i as u32, info))
        .collect();
    unsafe {
        device.update_descriptor_sets(&descriptor_writes, &[]);
    }
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "pipeline_descriptor_setup",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let cmd_pool = vk_device.transient_command_pool()?;
    let cmd_alloc_info = make_cmd_alloc_info(*cmd_pool);
    let command_buffers =
        crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
            .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device
            .begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;

        for (src, dst, size) in [
            (&q_stage, &q_buf, q_data.len() as u64),
            (&k_stage, &k_buf, k_data.len() as u64),
            (&v_stage, &v_buf, v_data.len() as u64),
            (&beta_stage, &beta_buf, beta_data.len() as u64),
            (&g_stage, &g_buf, g_data.len() as u64),
        ] {
            device.cmd_copy_buffer(
                cmd,
                src.handle(),
                dst.handle(),
                &[vk::BufferCopy::builder().size(size).build()],
            );
        }
        if let Some(state_stage) = &state_stage {
            device.cmd_copy_buffer(
                cmd,
                state_stage.handle(),
                state_buf.handle(),
                &[vk::BufferCopy::builder()
                    .size(state_data.len() as u64)
                    .build()],
            );
        }

        let upload_barrier = make_memory_barrier(
            vk::AccessFlags::TRANSFER_WRITE | vk::AccessFlags::HOST_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[upload_barrier],
            &[],
            &[],
        );

        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, dispatch_counts.0, dispatch_counts.1, dispatch_counts.2);

        let compute_barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ,
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[compute_barrier],
            &[],
            &[],
        );

        device.cmd_copy_buffer(
            cmd,
            out_buf.handle(),
            out_stage.handle(),
            &[vk::BufferCopy::builder().size(out_size).build()],
        );
        if !skip_state_readback && let Some(state_stage) = &state_stage {
            device.cmd_copy_buffer(
                cmd,
                state_buf.handle(),
                state_stage.handle(),
                &[vk::BufferCopy::builder()
                    .size(state_data.len() as u64)
                    .build()],
            );
        }

        device
            .end_command_buffer(cmd)
            .context("failed to end command buffer")?;
        device
            .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit gdn recurrent single-submit dispatch")?;
        device
            .queue_wait_idle(queue)
            .context("failed to wait for gdn recurrent single-submit dispatch")?;

        device
            .reset_descriptor_pool(*descriptor_pool, vk::DescriptorPoolResetFlags::empty())
            .context("failed to reset transient descriptor pool")?;
        device.free_command_buffers(*cmd_pool, &command_buffers);
    }
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "record_submit_wait",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let out_data = VulkanBuffer::read_host_visible(device, &out_stage)?;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "read_output",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );
    let stage_profile = profile_kernel_stages.then(Instant::now);
    let state_data = if skip_state_readback {
        None
    } else if host_visible_state {
        Some(VulkanBuffer::read_host_visible(device, &state_buf)?)
    } else {
        Some(VulkanBuffer::read_host_visible(
            device,
            state_stage
                .as_ref()
                .expect("state staging exists when state is device-local"),
        )?)
    };
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "read_state",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );

    let stage_profile = profile_kernel_stages.then(Instant::now);
    let out_shape = vec![batch, heads, dv];
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, output_dtype)?;
    let state_tensor = state_data
        .as_ref()
        .map(|state_data| create_tensor_from_data(state_data, state.dims().as_ref(), state.dtype()))
        .transpose()?;
    finish_vulkan_gdn_recurrent_kernel_stage_profile(
        "create_tensors",
        batch,
        heads,
        dk,
        dv,
        parallel_reduce,
        true,
        skip_state_readback,
        stage_profile,
    );
    Ok((out_tensor, state_tensor))
}

/// Dispatch a single-token recurrent step with unexpanded GQA Q/K heads.
///
/// `q` and `k` are `[batch, 1, q_heads, dk]`; `v`, `beta`, and `g` use value
/// heads (`[batch, 1, heads, ...]`). The shader maps each value head to its
/// source Q/K head with `h / (heads / q_heads)`, matching the regular GQA
/// expansion used by the portable path without materializing the repeated Q/K
/// tensors on the host.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_step_native_head_last_with_options(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
    skip_state_readback: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (k_batch, k_seq_len, k_heads, k_dk) = k.dims4()?;
    let (v_batch, v_seq_len, heads, dv) = v.dims4()?;
    let (beta_batch, beta_seq_len, beta_heads) = beta.dims3()?;
    let (g_batch, g_seq_len, g_heads) = g.dims3()?;
    let (state_batch, state_heads, state_dk, state_dv) = state.dims4()?;

    anyhow::ensure!(seq_len == 1, "native-head recurrent expects seq_len=1");
    anyhow::ensure!(
        (k_batch, k_seq_len, k_heads, k_dk) == (batch, seq_len, q_heads, dk),
        "native-head recurrent k shape mismatch"
    );
    anyhow::ensure!(
        (v_batch, v_seq_len) == (batch, seq_len),
        "native-head recurrent v batch/seq mismatch"
    );
    anyhow::ensure!(
        (beta_batch, beta_seq_len, beta_heads) == (batch, seq_len, heads),
        "native-head recurrent beta shape mismatch"
    );
    anyhow::ensure!(
        (g_batch, g_seq_len, g_heads) == (batch, seq_len, heads),
        "native-head recurrent g shape mismatch"
    );
    anyhow::ensure!(
        (state_batch, state_heads, state_dk, state_dv) == (batch, heads, dk, dv),
        "native-head recurrent state shape mismatch"
    );
    anyhow::ensure!(
        q_heads > 0,
        "native-head recurrent q_heads must be positive"
    );
    anyhow::ensure!(
        heads % q_heads == 0,
        "native-head recurrent heads {heads} must be divisible by q_heads {q_heads}"
    );

    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let g_data = extract_tensor_bytes(g)?.0;
    let state_data = extract_tensor_bytes(state)?.0;

    let parallel_reduce = use_gdn_recurrent_parallel_reduce(dk, dv);
    let glsl_path = if parallel_reduce {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_step_parallel.comp"
        )
    } else {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/csrc/shaders/gdn_recurrent_prefill.comp"
        )
    };
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    let (out, state) = dispatch_gdn_recurrent_step_single_submit(
        vk_device,
        q,
        state,
        &q_data,
        &k_data,
        &v_data,
        &beta_data,
        &g_data,
        &state_data,
        &spirv,
        batch,
        heads,
        q_heads,
        dk,
        dv,
        parallel_reduce,
        skip_state_readback,
        false,
        q.dtype(),
        None,
    )?;
    Ok((out.unsqueeze(1)?, state))
}

/// Dispatch a single-token recurrent step with unexpanded raw GQA Q/K heads,
/// folding the split path's Q/K L2 normalization into the recurrent shader.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gdn_recurrent_qk_norm_step_native_head_last_with_options(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
    skip_state_readback: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    let (batch, seq_len, q_heads, dk) = q.dims4()?;
    let (k_batch, k_seq_len, k_heads, k_dk) = k.dims4()?;
    let (v_batch, v_seq_len, heads, dv) = v.dims4()?;
    let (beta_batch, beta_seq_len, beta_heads) = beta.dims3()?;
    let (g_batch, g_seq_len, g_heads) = g.dims3()?;
    let (state_batch, state_heads, state_dk, state_dv) = state.dims4()?;

    anyhow::ensure!(
        seq_len == 1,
        "native-head qk-norm recurrent expects seq_len=1"
    );
    anyhow::ensure!(
        (k_batch, k_seq_len, k_heads, k_dk) == (batch, seq_len, q_heads, dk),
        "native-head qk-norm recurrent k shape mismatch"
    );
    anyhow::ensure!(
        (v_batch, v_seq_len) == (batch, seq_len),
        "native-head qk-norm recurrent v batch/seq mismatch"
    );
    anyhow::ensure!(
        (beta_batch, beta_seq_len, beta_heads) == (batch, seq_len, heads),
        "native-head qk-norm recurrent beta shape mismatch"
    );
    anyhow::ensure!(
        (g_batch, g_seq_len, g_heads) == (batch, seq_len, heads),
        "native-head qk-norm recurrent g shape mismatch"
    );
    anyhow::ensure!(
        (state_batch, state_heads, state_dk, state_dv) == (batch, heads, dk, dv),
        "native-head qk-norm recurrent state shape mismatch"
    );
    anyhow::ensure!(
        q_heads > 0,
        "native-head qk-norm recurrent q_heads must be positive"
    );
    anyhow::ensure!(
        heads % q_heads == 0,
        "native-head qk-norm recurrent heads {heads} must be divisible by q_heads {q_heads}"
    );
    anyhow::ensure!(
        dk <= 256 && dv <= 256,
        "native-head qk-norm recurrent supports dk/dv <= 256, got dk={dk} dv={dv}"
    );

    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let g_data = extract_tensor_bytes(g)?.0;
    let state_data = extract_tensor_bytes(state)?.0;

    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_recurrent_qk_norm_step.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;
    let (out, state) = dispatch_gdn_recurrent_step_single_submit(
        vk_device,
        q,
        state,
        &q_data,
        &k_data,
        &v_data,
        &beta_data,
        &g_data,
        &state_data,
        &spirv,
        batch,
        heads,
        q_heads,
        dk,
        dv,
        false,
        skip_state_readback,
        false,
        state.dtype(),
        Some((batch as u32, heads as u32, 1)),
    )?;
    Ok((out.unsqueeze(1)?, state))
}

// ---------------------------------------------------------------------------
// GDN chunk prep kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN chunk prep kernel.
///
/// Computes: a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last
/// Input: g[B,H,C], v[B,H,C,dv], kkt[B,H,C,C], qkt[B,H,C,C],
///         ks_entry[B,H,C,dv], q_s[B,H,C,dv]
pub fn dispatch_gdn_chunk_prep(
    vk_device: &VulkanDevice,
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let g_data = extract_tensor_bytes(g)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let kkt_data = extract_tensor_bytes(kkt)?.0;
    let qkt_data = extract_tensor_bytes(qkt)?.0;
    let ks_entry_data = extract_tensor_bytes(ks_entry)?.0;
    let q_s_data = extract_tensor_bytes(q_s)?.0;

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_chunk_prep.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Parse shapes
    let dims_g = g.dims();
    let (batch, heads, chunk) = (dims_g[0], dims_g[1], dims_g[2]);
    let dims_v = v.dims();
    let dv = dims_v[3];

    // Create input buffers + upload
    let g_buf = VulkanBuffer::create_device_local(device, device_local_mt, g_data.len() as u64)?;
    let v_buf = VulkanBuffer::create_device_local(device, device_local_mt, v_data.len() as u64)?;
    let kkt_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, kkt_data.len() as u64)?;
    let qkt_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, qkt_data.len() as u64)?;
    let ks_entry_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, ks_entry_data.len() as u64)?;
    let q_s_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, q_s_data.len() as u64)?;
    if gdn_chunk_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        upload_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[
                (&g_buf, &g_data),
                (&v_buf, &v_data),
                (&kkt_buf, &kkt_data),
                (&qkt_buf, &qkt_data),
                (&ks_entry_buf, &ks_entry_data),
                (&q_s_buf, &q_s_data),
            ],
        )
        .context("failed to upload gdn_chunk_prep inputs")?;
    } else {
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &g_buf, &g_data)?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &v_buf, &v_data)?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &kkt_buf, &kkt_data)?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &qkt_buf, &qkt_data)?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &ks_entry_buf,
            &ks_entry_data,
        )?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &q_s_buf, &q_s_data)?;
    }

    // Create output buffers (f32 shader outputs, converted to bf16 below).
    let cc_size = (batch * heads * chunk * chunk * 4) as u64;
    let cv_size = (batch * heads * chunk * dv * 4) as u64;
    let decay_size = (batch * heads * chunk * 4) as u64;
    let p_last_size = (batch * heads * 4) as u64;
    let a_strict_buf = VulkanBuffer::create_device_local(device, device_local_mt, cc_size)?;
    let b_mask_buf = VulkanBuffer::create_device_local(device, device_local_mt, cc_size)?;
    let v_prime_buf = VulkanBuffer::create_device_local(device, device_local_mt, cv_size)?;
    let q_s_scaled_buf = VulkanBuffer::create_device_local(device, device_local_mt, cv_size)?;
    let decay_last_col_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, decay_size)?;
    let p_last_buf = VulkanBuffer::create_device_local(device, device_local_mt, p_last_size)?;

    // Push constants: batch, heads, chunk, dv
    let push_constants: [u32; 4] = [batch as u32, heads as u32, chunk as u32, dv as u32];

    // Workgroup count: total elements / 256
    let total = batch * heads * (chunk * chunk + chunk * dv + chunk + 1);
    let workgroup_count = ((total + 255) / 256) as u32;

    // Bindings: g=0, v=1, kkt=2, qkt=3, ks_entry=4, q_s=5,
    //           a_strict=6, b_mask=7, v_prime=8, q_s_scaled=9, decay_last_col=10, p_last=11
    let all_handles = vec![
        g_buf.handle(),
        v_buf.handle(),
        kkt_buf.handle(),
        qkt_buf.handle(),
        ks_entry_buf.handle(),
        q_s_buf.handle(),
        a_strict_buf.handle(),
        b_mask_buf.handle(),
        v_prime_buf.handle(),
        q_s_scaled_buf.handle(),
        decay_last_col_buf.handle(),
        p_last_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back all outputs
    let (
        a_strict_data,
        b_mask_data,
        v_prime_data,
        q_s_scaled_data,
        decay_last_col_data,
        p_last_data,
    ) = if gdn_chunk_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        let mut data = read_back_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[
                &a_strict_buf,
                &b_mask_buf,
                &v_prime_buf,
                &q_s_scaled_buf,
                &decay_last_col_buf,
                &p_last_buf,
            ],
        )
        .context("failed to read back gdn_chunk_prep outputs")?;
        anyhow::ensure!(
            data.len() == 6,
            "gdn_chunk_prep batched readback returned wrong count"
        );
        (
            data.remove(0),
            data.remove(0),
            data.remove(0),
            data.remove(0),
            data.remove(0),
            data.remove(0),
        )
    } else {
        (
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &a_strict_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &b_mask_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &v_prime_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &q_s_scaled_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &decay_last_col_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &p_last_buf)?,
        )
    };

    // Cleanup
    drop(g_buf);
    drop(v_buf);
    drop(kkt_buf);
    drop(qkt_buf);
    drop(ks_entry_buf);
    drop(q_s_buf);
    drop(a_strict_buf);
    drop(b_mask_buf);
    drop(v_prime_buf);
    drop(q_s_scaled_buf);
    drop(decay_last_col_buf);
    drop(p_last_buf);

    let cc_shape = vec![batch, heads, chunk, chunk];
    let cv_shape = vec![batch, heads, chunk, dv];
    let decay_shape = vec![batch, heads, chunk];
    let p_last_shape = vec![batch, heads];

    let a_strict_tensor = create_tensor_from_data(&a_strict_data, &cc_shape, DType::BF16)?;
    let b_mask_tensor = create_tensor_from_data(&b_mask_data, &cc_shape, DType::BF16)?;
    let v_prime_tensor = create_tensor_from_data(&v_prime_data, &cv_shape, DType::BF16)?;
    let q_s_scaled_tensor = create_tensor_from_data(&q_s_scaled_data, &cv_shape, DType::BF16)?;
    let decay_last_col_tensor =
        create_tensor_from_data(&decay_last_col_data, &decay_shape, DType::BF16)?;
    let p_last_tensor = create_tensor_from_data(&p_last_data, &p_last_shape, DType::BF16)?;

    Ok((
        a_strict_tensor,
        b_mask_tensor,
        v_prime_tensor,
        q_s_scaled_tensor,
        decay_last_col_tensor,
        p_last_tensor,
    ))
}

// ---------------------------------------------------------------------------
// GDN full chunk forward kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN full chunk forward kernel (fused prep + scan).
///
/// Input: g[B,H,C], v[B,H,C,dv], kkt[B,H,C,C], qkt[B,H,C,C],
///         ks_entry[B,H,C,dv], q_s[B,H,C,dv], beta[B,H,C], k_t[B,H,dk,C]
/// State: [B,H,dk,dv] (in/out)
/// Output: [B,H,C,dv]
pub fn dispatch_gdn_full_chunk_forward(
    vk_device: &VulkanDevice,
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
    beta: &Tensor,
    k_t: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let g_data = extract_tensor_bytes(g)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let kkt_data = extract_tensor_bytes(kkt)?.0;
    let qkt_data = extract_tensor_bytes(qkt)?.0;
    let ks_entry_data = extract_tensor_bytes(ks_entry)?.0;
    let q_s_data = extract_tensor_bytes(q_s)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let k_t_data = extract_tensor_bytes(k_t)?.0;
    let state_data = extract_tensor_bytes(state)?.0;

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_full_chunk_forward.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Parse shapes
    let dims_g = g.dims();
    let (batch, heads, chunk) = (dims_g[0], dims_g[1], dims_g[2]);
    let dims_v = v.dims();
    let dv = dims_v[3];
    let dims_kt = k_t.dims();
    let dk = dims_kt[2];
    anyhow::ensure!(
        chunk == 64 && dv <= 128,
        "gdn_full_chunk_forward supports chunk=64 and dv<=128, got chunk={chunk} dv={dv}"
    );

    // Create input buffers + upload
    let g_buf = VulkanBuffer::create_device_local(device, device_local_mt, g_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &g_buf, &g_data)?;

    let v_buf = VulkanBuffer::create_device_local(device, device_local_mt, v_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &v_buf, &v_data)?;

    let kkt_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, kkt_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &kkt_buf, &kkt_data)?;

    let qkt_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, qkt_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &qkt_buf, &qkt_data)?;

    let ks_entry_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, ks_entry_data.len() as u64)?;
    VulkanBuffer::upload_data(
        device,
        host_visible_mt,
        queue,
        qfi,
        &ks_entry_buf,
        &ks_entry_data,
    )?;

    let q_s_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, q_s_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &q_s_buf, &q_s_data)?;

    let beta_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, beta_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &beta_buf, &beta_data)?;

    let k_t_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, k_t_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &k_t_buf, &k_t_data)?;

    // State is mutable — upload, dispatch, read back
    let state_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &state_buf, &state_data)?;

    // Create output buffer (f32)
    let out_size = (batch * heads * chunk * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, heads, chunk, dk, dv
    let push_constants: [u32; 5] = [
        batch as u32,
        heads as u32,
        chunk as u32,
        dk as u32,
        dv as u32,
    ];

    // One workgroup owns one (batch, head) chunk. Threads within the workgroup
    // cooperate over the fixed 64-token chunk and dv lanes.
    let workgroup_count = (batch * heads) as u32;

    // Bindings: g=0, v=1, kkt=2, qkt=3, ks_entry=4, q_s=5, beta=6, k_t=7, state=8, out=9
    let all_handles = vec![
        g_buf.handle(),
        v_buf.handle(),
        kkt_buf.handle(),
        qkt_buf.handle(),
        ks_entry_buf.handle(),
        q_s_buf.handle(),
        beta_buf.handle(),
        k_t_buf.handle(),
        state_buf.handle(),
        out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back output and updated state
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;
    let state_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &state_buf)?;

    // Cleanup
    drop(g_buf);
    drop(v_buf);
    drop(kkt_buf);
    drop(qkt_buf);
    drop(ks_entry_buf);
    drop(q_s_buf);
    drop(beta_buf);
    drop(k_t_buf);
    drop(state_buf);
    drop(out_buf);

    let out_shape = vec![batch, heads, chunk, dv];
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, DType::BF16)?;
    let state_tensor = create_tensor_from_data(&state_data, state.dims().as_ref(), DType::BF16)?;
    Ok((out_tensor, state_tensor))
}

// ---------------------------------------------------------------------------
// GDN chunk scan kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN chunk scan kernel.
///
/// Performs the scan operation for chunkwise recurrence:
///   1. forward-substitution for W[t]
///   2. intra = B_mask @ W
///   3. out = q_s_scaled + intra
///
/// Input: a_strict[B,H,C,C], b_mask[B,H,C,C], v_prime[B,H,C,dv],
///         q_s_scaled[B,H,C,dv], beta[B,H,C], decay_last_col[B,H,C]
/// Output: out[B,H,C,dv], p_out[B,H,C,dv]
pub fn dispatch_gdn_chunk_scan(
    vk_device: &VulkanDevice,
    a_strict: &Tensor,
    b_mask: &Tensor,
    v_prime: &Tensor,
    q_s_scaled: &Tensor,
    beta: &Tensor,
    decay_last_col: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let a_strict_data = extract_tensor_bytes(a_strict)?.0;
    let b_mask_data = extract_tensor_bytes(b_mask)?.0;
    let v_prime_data = extract_tensor_bytes(v_prime)?.0;
    let q_s_scaled_data = extract_tensor_bytes(q_s_scaled)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let decay_last_col_data = extract_tensor_bytes(decay_last_col)?.0;

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_chunk_scan.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Parse shapes
    let dims = v_prime.dims();
    let (batch, heads, chunk, dv) = (dims[0], dims[1], dims[2], dims[3]);

    // Create input buffers + upload
    let a_strict_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, a_strict_data.len() as u64)?;
    let b_mask_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, b_mask_data.len() as u64)?;
    let v_prime_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, v_prime_data.len() as u64)?;
    let q_s_scaled_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, q_s_scaled_data.len() as u64)?;
    let beta_buf =
        VulkanBuffer::create_device_local(device, device_local_mt, beta_data.len() as u64)?;
    let decay_last_col_buf = VulkanBuffer::create_device_local(
        device,
        device_local_mt,
        decay_last_col_data.len() as u64,
    )?;
    if gdn_chunk_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        upload_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[
                (&a_strict_buf, &a_strict_data),
                (&b_mask_buf, &b_mask_data),
                (&v_prime_buf, &v_prime_data),
                (&q_s_scaled_buf, &q_s_scaled_data),
                (&beta_buf, &beta_data),
                (&decay_last_col_buf, &decay_last_col_data),
            ],
        )
        .context("failed to upload gdn_chunk_scan inputs")?;
    } else {
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &a_strict_buf,
            &a_strict_data,
        )?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &b_mask_buf,
            &b_mask_data,
        )?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &v_prime_buf,
            &v_prime_data,
        )?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &q_s_scaled_buf,
            &q_s_scaled_data,
        )?;
        VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &beta_buf, &beta_data)?;
        VulkanBuffer::upload_data(
            device,
            host_visible_mt,
            queue,
            qfi,
            &decay_last_col_buf,
            &decay_last_col_data,
        )?;
    }

    // Create output buffers (f32)
    let out_size = (batch * heads * chunk * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;
    let p_out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, heads, chunk, dv
    let push_constants: [u32; 4] = [batch as u32, heads as u32, chunk as u32, dv as u32];

    // Workgroup count: total elements / 256
    let total = batch * heads * chunk * dv;
    let workgroup_count = ((total + 255) / 256) as u32;

    // Bindings: a_strict=0, b_mask=1, v_prime=2, q_s_scaled=3, beta=4, decay_last_col=5, out=6, p_out=7
    let all_handles = vec![
        a_strict_buf.handle(),
        b_mask_buf.handle(),
        v_prime_buf.handle(),
        q_s_scaled_buf.handle(),
        beta_buf.handle(),
        decay_last_col_buf.handle(),
        out_buf.handle(),
        p_out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // Build pipeline
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        total_bindings,
        &push_constants,
        workgroup_count,
    )?;

    // Read back outputs
    let (out_data, p_out_data) = if gdn_chunk_batched_transfers_enabled() {
        let command_pool = vk_device.transient_command_pool()?;
        let mut data = read_back_buffers_with_command_pool(
            device,
            host_visible_mt,
            queue,
            *command_pool,
            &[&out_buf, &p_out_buf],
        )
        .context("failed to read back gdn_chunk_scan outputs")?;
        anyhow::ensure!(
            data.len() == 2,
            "gdn_chunk_scan batched readback returned wrong count"
        );
        (data.remove(0), data.remove(0))
    } else {
        (
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?,
            VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &p_out_buf)?,
        )
    };

    // Cleanup
    drop(a_strict_buf);
    drop(b_mask_buf);
    drop(v_prime_buf);
    drop(q_s_scaled_buf);
    drop(beta_buf);
    drop(decay_last_col_buf);
    drop(out_buf);
    drop(p_out_buf);

    let out_shape = vec![batch, heads, chunk, dv];
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, DType::BF16)?;
    let p_out_tensor = create_tensor_from_data(&p_out_data, &out_shape, DType::BF16)?;
    Ok((out_tensor, p_out_tensor))
}

/// Scaled dot-product attention forward (prefill), online softmax.
///
/// Inputs `q`, `k`, `v` are F32 row-major `[batch, seq_len, num_heads,
/// head_dim]`. Returns the SDPA output with the same shape and dtype.
///
/// Replaces the buggy `flash_attn.comp` placeholder. The shader runs
/// one workgroup per `(batch, head, q_row)` with 128 threads doing a
/// parallel head_dim reduction per K row, plus the standard online
/// softmax recurrence. No scratch / LSE buffers are written; this is
/// the forward-only path used by training prefill.
///
/// Constraints: `head_dim` must be ≤ 128 (the workgroup size). For
/// Qwen3.5-4B head_dim=128 this is exact; smaller head_dim wastes
/// some threads but produces correct output.
pub fn dispatch_sdpa_prefill_f32(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    anyhow::ensure!(
        q.dtype() == DType::F32 && k.dtype() == DType::F32 && v.dtype() == DType::F32,
        "sdpa_prefill_f32: q/k/v must all be F32, got {:?}/{:?}/{:?}",
        q.dtype(),
        k.dtype(),
        v.dtype()
    );
    let (batch, seq_len, num_heads, head_dim) = q.dims4().context("sdpa_prefill_f32: q dims4")?;
    anyhow::ensure!(
        k.dims() == q.dims() && v.dims() == q.dims(),
        "sdpa_prefill_f32: q/k/v must have identical shape, got {:?}/{:?}/{:?}",
        q.dims(),
        k.dims(),
        v.dims()
    );
    anyhow::ensure!(
        head_dim <= 128,
        "sdpa_prefill_f32: head_dim {head_dim} > 128 (workgroup size limit)"
    );
    // Vulkan spec only guarantees `maxComputeWorkGroupCount[i] >= 65535`
    // per axis. The dispatch grid is (seq_len, num_heads, batch); if any
    // axis would exceed that, surface a clear error rather than letting
    // vkCmdDispatch silently drop work or fail with an opaque
    // VK_ERROR_OUT_OF_DEVICE_MEMORY. Use the actual device limit
    // (typically much higher than the spec minimum on AMD/Strix Halo).
    let limit_x = vk_device.max_compute_work_group_count(0) as usize;
    let limit_y = vk_device.max_compute_work_group_count(1) as usize;
    let limit_z = vk_device.max_compute_work_group_count(2) as usize;
    anyhow::ensure!(
        seq_len <= limit_x && num_heads <= limit_y && batch <= limit_z,
        "sdpa_prefill_f32: dispatch grid (seq_len={seq_len}, num_heads={num_heads}, \
         batch={batch}) exceeds device per-axis limits ({limit_x}, {limit_y}, {limit_z})"
    );

    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/sdpa_prefill_f32.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("sdpa_prefill_f32: shader compile/load")?;
    let push_constants: [u32; 6] = [
        batch as u32,
        seq_len as u32,
        num_heads as u32,
        head_dim as u32,
        softmax_scale.to_bits(),
        causal as u32,
    ];
    // Workgroup grid: (q_row, head, batch). Matches gl_WorkGroupID
    // assignments in the shader.
    let workgroup_count = (seq_len as u32, num_heads as u32, batch as u32);
    let output_shape = vec![batch, seq_len, num_heads, head_dim];
    dispatch_kernel(
        vk_device,
        &spirv,
        &push_constants,
        workgroup_count,
        &[q, k, v],
        &output_shape,
        DType::F32,
    )
    .context("sdpa_prefill_f32: dispatch_kernel")
}

/// Vulkan SGD parameter update step: `param -= lr * grad`, in-place
/// against an existing `VulkanBuffer` (the parameter buffer) using
/// the gradient as a read-only second buffer.
///
/// Phase 4.2 of the residency plan. Used by the trainer once
/// `TrainableLoraParams` have been migrated to registry-resident
/// `VulkanBuffer`s in Phase 4.1; until then, the existing CPU SGD
/// step in `kiln-train::trainer::sgd_step` continues to run.
///
/// Both buffers are flat F32 of length `n_elements`. The dispatch
/// allocates one workgroup per 256 elements; per-step compute is
/// trivially small (3n F32 reads/writes) so no chunking is required
/// even for the largest LoRA Vars (rank=64, hidden=2560 = 164K F32 =
/// 640 KB).
/// BF16 variant of `dispatch_sgd_step_f32`. Both buffers hold
/// packed BF16 (2 bf16 elements per u32) — same layout as the
/// `extract_tensor_packed_bf16_bytes_pub` encoding the residency
/// registry uses for BF16 tensors. One thread per u32 word; each
/// thread updates both lanes via bf16↔f32 bit-expansion.
///
/// Used by the trainer to run SGD on registry-resident LoRA Vars
/// (which are BF16 by convention) without the candle CPU
/// var.set + update_resident_activation re-upload.
pub fn dispatch_sgd_step_bf16(
    vk_device: &VulkanDevice,
    param_buffer: &VulkanBuffer,
    grad_buffer: &VulkanBuffer,
    n_elements: usize,
    lr: f32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "sgd_step_bf16: n_elements must be > 0");
    let num_words = n_elements.div_ceil(2);
    let workgroup_count = num_words.div_ceil(256) as u32;
    let limit = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit,
        "sgd_step_bf16: n_elements={n_elements} → {workgroup_count} workgroups \
         (>{limit} device per-axis limit)"
    );
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/sgd_step_bf16.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("sgd_step_bf16: shader compile/load")?;
    let push_constants: [u32; 2] = [n_elements as u32, lr.to_bits()];
    let all_handles = vec![param_buffer.handle(), grad_buffer.handle()];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        workgroup_count,
    )
    .context("sgd_step_bf16: kernel dispatch")
}

pub fn dispatch_sgd_step_f32(
    vk_device: &VulkanDevice,
    param_buffer: &VulkanBuffer,
    grad_buffer: &VulkanBuffer,
    n_elements: usize,
    lr: f32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "sgd_step_f32: n_elements must be > 0");
    // Vulkan only guarantees `maxComputeWorkGroupCount[i] >= 65535`.
    // The dispatch is `n_elements.div_ceil(256)` workgroups on axis x;
    // Use the actual device limit rather than the spec minimum.
    let limit = vk_device.max_compute_work_group_count(0) as usize;
    anyhow::ensure!(
        n_elements.div_ceil(256) <= limit,
        "sgd_step_f32: n_elements={n_elements} would dispatch \
         {} workgroups (>{limit} device per-axis limit)",
        n_elements.div_ceil(256)
    );
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/sgd_step_f32.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("sgd_step_f32: shader compile/load")?;
    let push_constants: [u32; 2] = [n_elements as u32, lr.to_bits()];
    let all_handles = vec![param_buffer.handle(), grad_buffer.handle()];
    let workgroup_count = n_elements.div_ceil(256) as u32;
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        workgroup_count,
    )
    .context("sgd_step_f32: kernel dispatch")
}

/// AdamW (decoupled weight decay) for registry-resident BF16 buffers.
///
/// Updates `param`, `m`, and `v` in place. All four buffers (param,
/// grad, m, v) hold packed BF16 (2 bf16 per u32 word) in the
/// `extract_tensor_packed_bf16_bytes_pub` encoding, and must share
/// the same element count `n_elements`. The step counter is 1-indexed
/// (so the first call after `m=v=0` passes `step=1`); host-side this
/// helper computes `bias_correction{1,2} = 1 - beta^step` and ships
/// them via push constants so the shader doesn't need a pow call.
///
/// One thread per u32 word (i.e. two BF16 lanes), 256 threads per
/// workgroup. Per-step cost is ~8n BF16 reads/writes — bandwidth-bound,
/// trivially small even for the largest LoRA Vars.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_adamw_step_bf16(
    vk_device: &VulkanDevice,
    param_buffer: &VulkanBuffer,
    grad_buffer: &VulkanBuffer,
    first_moment_buffer: &VulkanBuffer,
    second_moment_buffer: &VulkanBuffer,
    n_elements: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "adamw_step_bf16: n_elements must be > 0");
    anyhow::ensure!(step >= 1, "adamw_step_bf16: step must be 1-indexed (>=1)");
    let num_words = n_elements.div_ceil(2);
    let workgroup_count = num_words.div_ceil(256) as u32;
    let limit = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit,
        "adamw_step_bf16: n_elements={n_elements} → {workgroup_count} workgroups \
         (>{limit} device per-axis limit)"
    );
    let bc1 = 1.0_f32 - beta1.powi(step as i32);
    let bc2 = 1.0_f32 - beta2.powi(step as i32);
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/adamw_step_bf16.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("adamw_step_bf16: shader compile/load")?;
    let push_constants: [u32; 9] = [
        n_elements as u32,
        step,
        lr.to_bits(),
        beta1.to_bits(),
        beta2.to_bits(),
        eps.to_bits(),
        weight_decay.to_bits(),
        bc1.to_bits(),
        bc2.to_bits(),
    ];
    let all_handles = vec![
        param_buffer.handle(),
        grad_buffer.handle(),
        first_moment_buffer.handle(),
        second_moment_buffer.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        workgroup_count,
    )
    .context("adamw_step_bf16: kernel dispatch")
}

/// F32 variant of `dispatch_adamw_step_bf16`. Kept for parity with
/// `dispatch_sgd_step_f32`; currently LoRA Vars default to BF16 so
/// this path is exercised mainly by tests.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_adamw_step_f32(
    vk_device: &VulkanDevice,
    param_buffer: &VulkanBuffer,
    grad_buffer: &VulkanBuffer,
    first_moment_buffer: &VulkanBuffer,
    second_moment_buffer: &VulkanBuffer,
    n_elements: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<()> {
    anyhow::ensure!(n_elements > 0, "adamw_step_f32: n_elements must be > 0");
    anyhow::ensure!(step >= 1, "adamw_step_f32: step must be 1-indexed (>=1)");
    let workgroup_count = n_elements.div_ceil(256) as u32;
    let limit = vk_device.max_compute_work_group_count(0);
    anyhow::ensure!(
        workgroup_count <= limit,
        "adamw_step_f32: n_elements={n_elements} → {workgroup_count} workgroups \
         (>{limit} device per-axis limit)"
    );
    let bc1 = 1.0_f32 - beta1.powi(step as i32);
    let bc2 = 1.0_f32 - beta2.powi(step as i32);
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/adamw_step_f32.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)
        .context("adamw_step_f32: shader compile/load")?;
    let push_constants: [u32; 9] = [
        n_elements as u32,
        step,
        lr.to_bits(),
        beta1.to_bits(),
        beta2.to_bits(),
        eps.to_bits(),
        weight_decay.to_bits(),
        bc1.to_bits(),
        bc2.to_bits(),
    ];
    let all_handles = vec![
        param_buffer.handle(),
        grad_buffer.handle(),
        first_moment_buffer.handle(),
        second_moment_buffer.handle(),
    ];
    run_compute_pipeline(
        vk_device,
        &spirv,
        &all_handles,
        all_handles.len(),
        &push_constants,
        workgroup_count,
    )
    .context("adamw_step_f32: kernel dispatch")
}

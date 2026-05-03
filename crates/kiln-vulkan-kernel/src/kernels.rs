use anyhow::{Context, Result};
use ash::vk;
use candle_core::{DType, Device, Tensor};
use crate::device::VulkanDevice;
use crate::buffer::VulkanBuffer;

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
        VulkanBuffer::upload_data(device, host_visible_mt, queue, queue_family_index, &buf, data)
            .context("failed to upload input data")?;
        input_buffers.push(buf);
    }

    // --- Build combined binding list (inputs first, then output) ---
    let total_bindings = input_buffers.len() + 1;
    tracing::trace!(total_bindings, inputs = input_tensors.len(), "Vulkan dispatch start");
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
        device.create_shader_module(&shader_module_info, None)
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
        device.create_descriptor_set_layout(&set_layout_info, None)
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
        device.create_pipeline_layout(&layout_info, None)
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
        device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
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
        device.create_descriptor_pool(&pool_info, None)
            .context("failed to create descriptor pool")?
    };

    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool)
        .set_layouts(&set_layouts)
        .build();
    let descriptor_sets = unsafe {
        device.allocate_descriptor_sets(&alloc_info)
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
            .map(|(i, bui)| {
                make_write_descriptor_set_buf(descriptor_set, i as u32, bui)
            })
            .collect();

        unsafe {
            device.update_descriptor_sets(&descriptor_write_infos, &[]);
        }
    }

    // --- Command buffer + dispatch ---
    let cmd_pool_info = make_cmd_pool_info(queue_family_index);
    let cmd_pool = unsafe {
        device.create_command_pool(&cmd_pool_info, None)
            .context("failed to create command pool")?
    };

    let alloc_info = make_cmd_alloc_info(cmd_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
        .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    let begin_info = make_cmd_begin_info();
    unsafe {
        device.begin_command_buffer(cmd, &begin_info)
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

        device.end_command_buffer(cmd)
            .context("failed to end command buffer")?;
    }

    // --- Submit + wait ---
    let cmds = vec![cmd];
    let submit_info = make_submit_info(&cmds);
    unsafe {
        device.queue_submit(queue, &[submit_info], vk::Fence::null())
            .context("failed to submit compute dispatch")?;
        device.queue_wait_idle(queue)
            .context("failed to wait for queue")?;
    }

    // --- Read back output ---
    let output_data = VulkanBuffer::read_back(
        device,
        host_visible_mt,
        queue,
        queue_family_index,
        &output_buffer,
    ).context("failed to read back output")?;

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
    unsafe { write_bytes(info.as_mut_ptr(), 0, 1); }
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
    unsafe { write_bytes(info.as_mut_ptr(), 0, 1); }
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
    unsafe { write_bytes(info.as_mut_ptr(), 0, 1); }
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
    unsafe { write_bytes(info.as_mut_ptr(), 0, 1); }
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

/// Create a zero-init WriteDescriptorSet for STORAGE_BUFFER with fixed sType.
fn make_write_descriptor_set_buf(
    dst_set: vk::DescriptorSet,
    dst_binding: u32,
    bui: &vk::DescriptorBufferInfo,
) -> vk::WriteDescriptorSet {
    use std::mem::MaybeUninit;
    use std::ptr::write_bytes;

    let mut info: MaybeUninit<vk::WriteDescriptorSet> = MaybeUninit::uninit();
    unsafe { write_bytes(info.as_mut_ptr(), 0, 1); }
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
    unsafe { write_bytes(info.as_mut_ptr(), 0, 1); }
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
    let f32_data = flat.to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .context("failed to extract f32 data")?;
    Ok((bytemuck::cast_slice(&f32_data).to_vec(), shape))
}

/// Create a candle-core Tensor from raw bytes.
pub fn create_tensor_from_data(
    data: &[u8],
    shape: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let f32_data: &[f32] = bytemuck::cast_slice(data);
    let tensor = Tensor::from_vec(f32_data.to_vec(), f32_data.len(), &Device::Cpu)?
        .reshape(shape)?;

    if dtype == DType::BF16 {
        Ok(tensor.to_dtype(DType::BF16)?)
    } else {
        Ok(tensor)
    }
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
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let a_data = extract_tensor_bytes(a)?.0;
    let b_data = extract_tensor_bytes(b)?.0;
    let a_log_data = extract_tensor_bytes(a_log)?.0;
    let dt_bias_data = extract_tensor_bytes(dt_bias)?.0;

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_gates.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Create input buffers + upload
    let a_buf = VulkanBuffer::create_device_local(device, device_local_mt, a_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &a_buf, &a_data)?;

    let b_buf = VulkanBuffer::create_device_local(device, device_local_mt, b_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &b_buf, &b_data)?;

    let a_log_buf = VulkanBuffer::create_device_local(device, device_local_mt, a_log_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &a_log_buf, &a_log_data)?;

    let dt_bias_buf = VulkanBuffer::create_device_local(device, device_local_mt, dt_bias_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &dt_bias_buf, &dt_bias_data)?;

    // Create output buffers
    let elem_count: usize = out_shape.iter().product();
    let output_size = (elem_count * 4) as u64; // f32
    let beta_buf = VulkanBuffer::create_device_local(device, device_local_mt, output_size)?;
    let g_buf = VulkanBuffer::create_device_local(device, device_local_mt, output_size)?;

    // Push constants: total elements, nv
    let nv = a_log_data.len() / 4; // nv in floats
    let push_constants: [u32; 2] = [elem_count as u32, nv as u32];

    // Workgroup count
    let workgroup_count = ((elem_count + 255) / 256) as u32;

    // Build descriptor bindings: a=0, b=1, a_log=2, dt_bias=3, beta_out=4, g_out=5
    let all_handles = vec![
        a_buf.handle(),
        b_buf.handle(),
        a_log_buf.handle(),
        dt_bias_buf.handle(),
        beta_buf.handle(),
        g_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // --- Build pipeline ---
    let spirv_words: &[u32] = bytemuck::cast_slice(&spirv);
    let shader_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(spirv_words).build(),
            None,
        ).context("failed to create shader module")?
    };

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
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_bindings).build(),
            None,
        ).context("failed to create descriptor set layout")?
    };

    let push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size((push_constants.len() * 4) as u32)
        .build();
    let set_layouts = vec![set_layout];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&[push_constant_range])
                .build(),
            None,
        ).context("failed to create pipeline layout")?
    };

    let stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
        .build();
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::builder().stage(stage_info).layout(layout).build()],
            None,
        ).map_err(|(errs, _)| {
            if !errs.is_empty() {
                anyhow::anyhow!("failed to create compute pipeline: {:?}", errs[0])
            } else {
                anyhow::anyhow!("failed to create compute pipeline")
            }
        })?[0]
    };

    // Descriptor pool + set
    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&[vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(total_bindings as u32)
                    .build()])
                .build(),
            None,
        ).context("failed to create descriptor pool")?
    };
    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts)
                .build(),
        ).context("failed to allocate descriptor sets")?[0]
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
    let cmd_pool = unsafe {
        device.create_command_pool(&make_cmd_pool_info(qfi), None)
            .context("failed to create command pool")?
    };
    let cmd_alloc_info = make_cmd_alloc_info(cmd_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
        .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device.begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, layout, 0,
            &[descriptor_set], &[],
        );
        device.cmd_push_constants(
            cmd, layout, vk::ShaderStageFlags::COMPUTE, 0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(), &[barrier], &[], &[],
        );
        device.end_command_buffer(cmd).context("failed to end command buffer")?;
    }

    unsafe {
        device.queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit")?;
        device.queue_wait_idle(queue).context("failed to wait for queue")?;
    }

    // Read back both outputs
    let beta_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &beta_buf)?;
    let g_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &g_buf)?;

    // Cleanup
    drop(a_buf); drop(b_buf); drop(a_log_buf); drop(dt_bias_buf);
    drop(beta_buf); drop(g_buf);
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_shader_module(shader_module, None);
        device.free_command_buffers(cmd_pool, &command_buffers);
        device.destroy_command_pool(cmd_pool, None);
    }

    let beta_tensor = create_tensor_from_data(&beta_data, out_shape, DType::BF16)?;
    let g_tensor = create_tensor_from_data(&g_data, out_shape, DType::BF16)?;
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
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let x_data = extract_tensor_bytes(x)?.0;
    let z_data = extract_tensor_bytes(z)?.0;
    let weight_data = extract_tensor_bytes(weight)?.0;

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_gated_rms_norm.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Create input buffers + upload
    let x_buf = VulkanBuffer::create_device_local(device, device_local_mt, x_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &x_buf, &x_data)?;

    let z_buf = VulkanBuffer::create_device_local(device, device_local_mt, z_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &z_buf, &z_data)?;

    let weight_buf = VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &weight_buf, &weight_data)?;

    // Create output buffer
    let elem_count: usize = out_shape.iter().product();
    let output_size = (elem_count * 4) as u64; // f32
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, output_size)?;

    // Push constants: rows, hidden, eps
    let hidden = weight_data.len() / 4; // hidden in floats
    let rows = elem_count / hidden;
    let push_constants: [u32; 3] = [rows as u32, hidden as u32, eps.to_bits()];

    // Workgroup count: one group per row
    let workgroup_count = rows as u32;

    // Build descriptor bindings: x=0, z=1, weight=2, out=3
    let all_handles = vec![
        x_buf.handle(),
        z_buf.handle(),
        weight_buf.handle(),
        out_buf.handle(),
    ];
    let total_bindings = all_handles.len();

    // --- Build pipeline ---
    let spirv_words: &[u32] = bytemuck::cast_slice(&spirv);
    let shader_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(spirv_words).build(),
            None,
        ).context("failed to create shader module")?
    };

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
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_bindings).build(),
            None,
        ).context("failed to create descriptor set layout")?
    };

    let push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size((push_constants.len() * 4) as u32)
        .build();
    let set_layouts = vec![set_layout];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&[push_constant_range])
                .build(),
            None,
        ).context("failed to create pipeline layout")?
    };

    let stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
        .build();
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::builder().stage(stage_info).layout(layout).build()],
            None,
        ).map_err(|(errs, _)| {
            if !errs.is_empty() {
                anyhow::anyhow!("failed to create compute pipeline: {:?}", errs[0])
            } else {
                anyhow::anyhow!("failed to create compute pipeline")
            }
        })?[0]
    };

    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&[vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(total_bindings as u32)
                    .build()])
                .build(),
            None,
        ).context("failed to create descriptor pool")?
    };
    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts)
                .build(),
        ).context("failed to allocate descriptor sets")?[0]
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
    let cmd_pool = unsafe {
        device.create_command_pool(&make_cmd_pool_info(qfi), None)
            .context("failed to create command pool")?
    };
    let cmd_alloc_info = make_cmd_alloc_info(cmd_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
        .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device.begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, layout, 0,
            &[descriptor_set], &[],
        );
        device.cmd_push_constants(
            cmd, layout, vk::ShaderStageFlags::COMPUTE, 0,
            bytemuck::cast_slice(&push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(), &[barrier], &[], &[],
        );
        device.end_command_buffer(cmd).context("failed to end command buffer")?;
    }

    unsafe {
        device.queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit")?;
        device.queue_wait_idle(queue).context("failed to wait for queue")?;
    }

    // Read back output
    let output_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;

    // Cleanup
    drop(x_buf); drop(z_buf); drop(weight_buf); drop(out_buf);
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_shader_module(shader_module, None);
        device.free_command_buffers(cmd_pool, &command_buffers);
        device.destroy_command_pool(cmd_pool, None);
    }

    create_tensor_from_data(&output_data, out_shape, DType::BF16)
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
    let qfi = vk_device.queue_family_index();
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
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &x_buf, &x_data)?;

    let weight_buf = VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &weight_buf, &weight_data)?;

    // conv_state is mutable — upload, dispatch, read back
    let state_buf = VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &state_buf, &state_data)?;

    // Create output buffer (f32)
    let out_size = (batch * channels * seq_len * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // ---- Dispatch 1: causal_conv1d.comp (output only, no state writes) ----
    let glsl_output = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d.comp"
    );
    let spirv_output = crate::pipeline::ShaderPipeline::compile_shader(glsl_output)?;

    // Bindings for output shader: x=0, weight=1, out=2
    let output_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        weight_buf.handle(),
        out_buf.handle(),
    ];
    let output_push: [u32; 4] = [batch as u32, channels as u32, seq_len as u32, kernel_size as u32];
    let total = batch * channels * seq_len;
    let output_wg = ((total + 255) / 256) as u32;

    run_compute_pipeline(
        device, queue, qfi, &spirv_output,
        &output_handles, 3,
        &output_push, output_wg,
    )?;

    // ---- Dispatch 2: causal_conv1d_state_advance.comp (state update only) ----
    // Each workgroup handles one (b, c) pair: batch * channels workgroups
    let glsl_state = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d_state_advance.comp"
    );
    let spirv_state = crate::pipeline::ShaderPipeline::compile_shader(glsl_state)?;

    // Bindings for state shader: x=0, conv_state=1
    let state_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        state_buf.handle(),
    ];
    let state_push: [u32; 4] = [batch as u32, channels as u32, seq_len as u32, kernel_size as u32];
    let state_wg = (batch * channels) as u32;

    run_compute_pipeline(
        device, queue, qfi, &spirv_state,
        &state_handles, 2,
        &state_push, state_wg,
    )?;

    // Read back both output and updated state
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;
    let state_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &state_buf)?;

    // Cleanup
    drop(x_buf); drop(weight_buf); drop(state_buf); drop(out_buf);

    // Create output tensors
    let out_shape = x.dims().as_ref().to_vec();
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, DType::F32)?;
    let state_tensor = create_tensor_from_data(&state_data, conv_state.dims().as_ref(), DType::F32)?;
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
    let qfi = vk_device.queue_family_index();
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
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &x_buf, &x_data)?;

    let weight_buf = VulkanBuffer::create_device_local(device, device_local_mt, weight_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &weight_buf, &weight_data)?;

    // conv_state is mutable — upload, dispatch, read back
    let state_buf = VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &state_buf, &state_data)?;

    // Create output buffer (f32)
    let out_size = (batch * channels * seq_len * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // ---- Dispatch 1: causal_conv1d.comp (output only, no state writes) ----
    let glsl_output = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d.comp"
    );
    let spirv_output = crate::pipeline::ShaderPipeline::compile_shader(glsl_output)?;

    // Bindings for output shader: x=0, weight=1, out=2
    let output_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        weight_buf.handle(),
        out_buf.handle(),
    ];
    let output_push: [u32; 4] = [batch as u32, channels as u32, seq_len as u32, kernel_size as u32];
    let total = batch * channels * seq_len;
    let output_wg = ((total + 255) / 256) as u32;

    run_compute_pipeline(
        device, queue, qfi, &spirv_output,
        &output_handles, 3,
        &output_push, output_wg,
    )?;

    // ---- Dispatch 2: causal_conv1d_state_advance.comp (state update only) ----
    // Each workgroup handles one (b, c) pair: batch * channels workgroups
    let glsl_state = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/causal_conv1d_state_advance.comp"
    );
    let spirv_state = crate::pipeline::ShaderPipeline::compile_shader(glsl_state)?;

    // Bindings for state shader: x=0, conv_state=1
    let state_handles: Vec<vk::Buffer> = vec![
        x_buf.handle(),
        state_buf.handle(),
    ];
    let state_push: [u32; 4] = [batch as u32, channels as u32, seq_len as u32, kernel_size as u32];
    let state_wg = (batch * channels) as u32;

    run_compute_pipeline(
        device, queue, qfi, &spirv_state,
        &state_handles, 2,
        &state_push, state_wg,
    )?;

    // Read back both output and updated state
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;
    let state_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &state_buf)?;

    // Cleanup
    drop(x_buf); drop(weight_buf); drop(state_buf); drop(out_buf);

    // Create output tensors
    let out_shape = x.dims().as_ref().to_vec();
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, DType::F32)?;
    let state_tensor = create_tensor_from_data(&state_data, conv_state.dims().as_ref(), DType::F32)?;
    Ok((out_tensor, state_tensor))
}



// ---------------------------------------------------------------------------
// Common pipeline build + dispatch helper to reduce code duplication
// ---------------------------------------------------------------------------

/// Build a Vulkan compute pipeline, dispatch it, wait for completion,
/// and clean up all resources.
///
/// This helper is used by causal_conv1d (two-dispatch path) and gdn
/// kernels. All Vulkan resources (shader module, pipeline layout, compute
/// pipeline, descriptor pool, command pool) are created and destroyed
/// within this function to keep individual dispatch functions focused.
pub fn run_compute_pipeline(
    device: &ash::Device,
    queue: vk::Queue,
    qfi: u32,
    spirv: &[u8],
    all_handles: &[vk::Buffer],
    total_bindings: usize,
    push_constants: &[u32],
    workgroup_count: u32,
) -> Result<()> {
    let spirv_words: &[u32] = bytemuck::cast_slice(spirv);
    let shader_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::builder().code(spirv_words).build(),
            None,
        ).context("failed to create shader module")?
    };

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
    let set_layout = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_bindings).build(),
            None,
        ).context("failed to create descriptor set layout")?
    };

    let push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size((push_constants.len() * 4) as u32)
        .build();
    let set_layouts = vec![set_layout];
    let layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&[push_constant_range])
                .build(),
            None,
        ).context("failed to create pipeline layout")?
    };

    let stage_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap())
        .build();
    let pipeline = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[vk::ComputePipelineCreateInfo::builder().stage(stage_info).layout(layout).build()],
            None,
        ).map_err(|(errs, _)| {
            if !errs.is_empty() {
                anyhow::anyhow!("failed to create compute pipeline: {:?}", errs[0])
            } else {
                anyhow::anyhow!("failed to create compute pipeline")
            }
        })?[0]
    };

    let pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .max_sets(1)
                .pool_sizes(&[vk::DescriptorPoolSize::builder()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(total_bindings as u32)
                    .build()])
                .build(),
            None,
        ).context("failed to create descriptor pool")?
    };
    let descriptor_set = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(pool)
                .set_layouts(&set_layouts)
                .build(),
        ).context("failed to allocate descriptor sets")?[0]
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
    let cmd_pool = unsafe {
        device.create_command_pool(&make_cmd_pool_info(qfi), None)
            .context("failed to create command pool")?
    };
    let cmd_alloc_info = make_cmd_alloc_info(cmd_pool);
    let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &cmd_alloc_info, 1)
        .context("failed to allocate command buffer")?;
    let cmd = command_buffers[0];

    unsafe {
        device.begin_command_buffer(cmd, &make_cmd_begin_info())
            .context("failed to begin command buffer")?;
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            cmd, vk::PipelineBindPoint::COMPUTE, layout, 0,
            &[descriptor_set], &[],
        );
        device.cmd_push_constants(
            cmd, layout, vk::ShaderStageFlags::COMPUTE, 0,
            bytemuck::cast_slice(push_constants),
        );
        device.cmd_dispatch(cmd, workgroup_count, 1, 1);

        let barrier = make_memory_barrier(
            vk::AccessFlags::SHADER_WRITE, vk::AccessFlags::TRANSFER_READ,
        );
        device.cmd_pipeline_barrier(
            cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(), &[barrier], &[], &[],
        );
        device.end_command_buffer(cmd).context("failed to end command buffer")?;
    }

    unsafe {
        device.queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
            .context("failed to submit")?;
        device.queue_wait_idle(queue).context("failed to wait for queue")?;
    }

    // Cleanup
    unsafe {
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_shader_module(shader_module, None);
        device.free_command_buffers(cmd_pool, &command_buffers);
        device.destroy_command_pool(cmd_pool, None);
    }

    Ok(())
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
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/solve_tri.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Parse output shape [B, H, C, dv]
    let dims = v_prime.dims();
    let (batch, heads, chunk, dv) = (dims[0], dims[1], dims[2], dims[3]);

    // Create input buffers + upload
    let a_strict_buf = VulkanBuffer::create_device_local(device, device_local_mt, a_strict_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &a_strict_buf, &a_strict_data)?;

    let v_prime_buf = VulkanBuffer::create_device_local(device, device_local_mt, v_prime_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &v_prime_buf, &v_prime_data)?;

    let beta_buf = VulkanBuffer::create_device_local(device, device_local_mt, beta_data.len() as u64)?;
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
        device, queue, qfi, &spirv, &all_handles, total_bindings,
        &push_constants, workgroup_count,
    )?;

    // Read back output
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;

    // Cleanup
    drop(a_strict_buf); drop(v_prime_buf); drop(beta_buf); drop(out_buf);

    let out_shape = vec![batch, heads, chunk, dv];
    create_tensor_from_data(&out_data, &out_shape, DType::F32)
        .context("failed to create gdn_forward_substitution output tensor")
}

// ---------------------------------------------------------------------------
// GDN recurrent step kernel
// ---------------------------------------------------------------------------

/// Dispatch GDN recurrent step kernel.
///
/// Recurrent state update for GDN.
/// Q: [B,T,H,dk], K: [B,T,H,dk], V: [B,T,H,dv], beta: [B,T,H], g: [B,T,H]
/// State: [B,H,dk,dv] (in/out), Output: [B,T,H,dv]
pub fn dispatch_gdn_recurrent_step(
    vk_device: &VulkanDevice,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    state: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let device = vk_device.device();
    let queue = vk_device.queue();
    let qfi = vk_device.queue_family_index();
    let device_local_mt = vk_device.device_local_mem_type();
    let host_visible_mt = vk_device.host_visible_mem_type();

    // Extract input data
    let q_data = extract_tensor_bytes(q)?.0;
    let k_data = extract_tensor_bytes(k)?.0;
    let v_data = extract_tensor_bytes(v)?.0;
    let beta_data = extract_tensor_bytes(beta)?.0;
    let g_data = extract_tensor_bytes(g)?.0;
    let state_data = extract_tensor_bytes(state)?.0;

    // Compile shader
    let glsl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/csrc/shaders/gdn_recurrent_prefill.comp"
    );
    let spirv = crate::pipeline::ShaderPipeline::compile_shader(glsl_path)?;

    // Parse shape [B, T, H, dk/dv]
    let dims = q.dims();
    let (batch, seq_len, heads, dk) = (dims[0], dims[1], dims[2], dims[3]);
    let dims_v = v.dims();
    let dv = dims_v[3];

    // Create input buffers + upload
    let q_buf = VulkanBuffer::create_device_local(device, device_local_mt, q_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &q_buf, &q_data)?;

    let k_buf = VulkanBuffer::create_device_local(device, device_local_mt, k_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &k_buf, &k_data)?;

    let v_buf = VulkanBuffer::create_device_local(device, device_local_mt, v_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &v_buf, &v_data)?;

    let beta_buf = VulkanBuffer::create_device_local(device, device_local_mt, beta_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &beta_buf, &beta_data)?;

    let g_buf = VulkanBuffer::create_device_local(device, device_local_mt, g_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &g_buf, &g_data)?;

    // State is mutable — upload, dispatch, read back
    let state_buf = VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &state_buf, &state_data)?;

    // Create output buffer (f32)
    let out_size = (batch * seq_len * heads * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, heads, seq_len, dk, dv
    let push_constants: [u32; 5] = [batch as u32, heads as u32, seq_len as u32, dk as u32, dv as u32];

    // Workgroup count: total elements / 256
    let total = batch * seq_len * heads * dv;
    let workgroup_count = ((total + 255) / 256) as u32;

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
        device, queue, qfi, &spirv, &all_handles, total_bindings,
        &push_constants, workgroup_count,
    )?;

    // Read back output and updated state
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;
    let state_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &state_buf)?;

    // Cleanup
    drop(q_buf); drop(k_buf); drop(v_buf); drop(beta_buf); drop(g_buf);
    drop(state_buf); drop(out_buf);

    let out_shape = vec![batch, seq_len, heads, dv];
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, DType::F32)?;
    let state_tensor = create_tensor_from_data(&state_data, state.dims().as_ref(), DType::F32)?;
    Ok((out_tensor, state_tensor))
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
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &g_buf, &g_data)?;

    let v_buf = VulkanBuffer::create_device_local(device, device_local_mt, v_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &v_buf, &v_data)?;

    let kkt_buf = VulkanBuffer::create_device_local(device, device_local_mt, kkt_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &kkt_buf, &kkt_data)?;

    let qkt_buf = VulkanBuffer::create_device_local(device, device_local_mt, qkt_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &qkt_buf, &qkt_data)?;

    let ks_entry_buf = VulkanBuffer::create_device_local(device, device_local_mt, ks_entry_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &ks_entry_buf, &ks_entry_data)?;

    let q_s_buf = VulkanBuffer::create_device_local(device, device_local_mt, q_s_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &q_s_buf, &q_s_data)?;

    // Create output buffers (f32)
    let cc_size = (batch * heads * chunk * chunk * 4) as u64;
    let cv_size = (batch * heads * chunk * dv * 4) as u64;
    let a_strict_buf = VulkanBuffer::create_device_local(device, device_local_mt, cc_size)?;
    let b_mask_buf = VulkanBuffer::create_device_local(device, device_local_mt, cc_size)?;
    let v_prime_buf = VulkanBuffer::create_device_local(device, device_local_mt, cv_size)?;
    let q_s_scaled_buf = VulkanBuffer::create_device_local(device, device_local_mt, cv_size)?;
    let decay_last_col_buf = VulkanBuffer::create_device_local(device, device_local_mt, cv_size)?;
    let p_last_buf = VulkanBuffer::create_device_local(device, device_local_mt, cv_size)?;

    // Push constants: batch, heads, chunk, dv
    let push_constants: [u32; 4] = [batch as u32, heads as u32, chunk as u32, dv as u32];

    // Workgroup count: total elements / 256
    let total = batch * heads * (chunk * chunk + chunk * dv * 2);
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
        device, queue, qfi, &spirv, &all_handles, total_bindings,
        &push_constants, workgroup_count,
    )?;

    // Read back all outputs
    let a_strict_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &a_strict_buf)?;
    let b_mask_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &b_mask_buf)?;
    let v_prime_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &v_prime_buf)?;
    let q_s_scaled_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &q_s_scaled_buf)?;
    let decay_last_col_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &decay_last_col_buf)?;
    let p_last_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &p_last_buf)?;

    // Cleanup
    drop(g_buf); drop(v_buf); drop(kkt_buf); drop(qkt_buf);
    drop(ks_entry_buf); drop(q_s_buf);
    drop(a_strict_buf); drop(b_mask_buf); drop(v_prime_buf); drop(q_s_scaled_buf);
    drop(decay_last_col_buf); drop(p_last_buf);

    let cc_shape = vec![batch, heads, chunk, chunk];
    let cv_shape = vec![batch, heads, chunk, dv];

    let a_strict_tensor = create_tensor_from_data(&a_strict_data, &cc_shape, DType::F32)?;
    let b_mask_tensor = create_tensor_from_data(&b_mask_data, &cc_shape, DType::F32)?;
    let v_prime_tensor = create_tensor_from_data(&v_prime_data, &cv_shape, DType::F32)?;
    let q_s_scaled_tensor = create_tensor_from_data(&q_s_scaled_data, &cv_shape, DType::F32)?;
    let decay_last_col_tensor = create_tensor_from_data(&decay_last_col_data, &cv_shape, DType::F32)?;
    let p_last_tensor = create_tensor_from_data(&p_last_data, &cv_shape, DType::F32)?;

    Ok((a_strict_tensor, b_mask_tensor, v_prime_tensor, q_s_scaled_tensor, decay_last_col_tensor, p_last_tensor))
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

    // Create input buffers + upload
    let g_buf = VulkanBuffer::create_device_local(device, device_local_mt, g_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &g_buf, &g_data)?;

    let v_buf = VulkanBuffer::create_device_local(device, device_local_mt, v_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &v_buf, &v_data)?;

    let kkt_buf = VulkanBuffer::create_device_local(device, device_local_mt, kkt_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &kkt_buf, &kkt_data)?;

    let qkt_buf = VulkanBuffer::create_device_local(device, device_local_mt, qkt_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &qkt_buf, &qkt_data)?;

    let ks_entry_buf = VulkanBuffer::create_device_local(device, device_local_mt, ks_entry_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &ks_entry_buf, &ks_entry_data)?;

    let q_s_buf = VulkanBuffer::create_device_local(device, device_local_mt, q_s_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &q_s_buf, &q_s_data)?;

    let beta_buf = VulkanBuffer::create_device_local(device, device_local_mt, beta_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &beta_buf, &beta_data)?;

    let k_t_buf = VulkanBuffer::create_device_local(device, device_local_mt, k_t_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &k_t_buf, &k_t_data)?;

    // State is mutable — upload, dispatch, read back
    let state_buf = VulkanBuffer::create_device_local(device, device_local_mt, state_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &state_buf, &state_data)?;

    // Create output buffer (f32)
    let out_size = (batch * heads * chunk * dv * 4) as u64;
    let out_buf = VulkanBuffer::create_device_local(device, device_local_mt, out_size)?;

    // Push constants: batch, heads, chunk, dk, dv
    let push_constants: [u32; 5] = [batch as u32, heads as u32, chunk as u32, dk as u32, dv as u32];

    // Workgroup count: total elements / 256
    let total = batch * heads * chunk * dv;
    let workgroup_count = ((total + 255) / 256) as u32;

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
        device, queue, qfi, &spirv, &all_handles, total_bindings,
        &push_constants, workgroup_count,
    )?;

    // Read back output and updated state
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;
    let state_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &state_buf)?;

    // Cleanup
    drop(g_buf); drop(v_buf); drop(kkt_buf); drop(qkt_buf);
    drop(ks_entry_buf); drop(q_s_buf); drop(beta_buf); drop(k_t_buf);
    drop(state_buf); drop(out_buf);

    let out_shape = vec![batch, heads, chunk, dv];
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, DType::F32)?;
    let state_tensor = create_tensor_from_data(&state_data, state.dims().as_ref(), DType::F32)?;
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
    let a_strict_buf = VulkanBuffer::create_device_local(device, device_local_mt, a_strict_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &a_strict_buf, &a_strict_data)?;

    let b_mask_buf = VulkanBuffer::create_device_local(device, device_local_mt, b_mask_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &b_mask_buf, &b_mask_data)?;

    let v_prime_buf = VulkanBuffer::create_device_local(device, device_local_mt, v_prime_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &v_prime_buf, &v_prime_data)?;

    let q_s_scaled_buf = VulkanBuffer::create_device_local(device, device_local_mt, q_s_scaled_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &q_s_scaled_buf, &q_s_scaled_data)?;

    let beta_buf = VulkanBuffer::create_device_local(device, device_local_mt, beta_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &beta_buf, &beta_data)?;

    let decay_last_col_buf = VulkanBuffer::create_device_local(device, device_local_mt, decay_last_col_data.len() as u64)?;
    VulkanBuffer::upload_data(device, host_visible_mt, queue, qfi, &decay_last_col_buf, &decay_last_col_data)?;

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
        device, queue, qfi, &spirv, &all_handles, total_bindings,
        &push_constants, workgroup_count,
    )?;

    // Read back outputs
    let out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &out_buf)?;
    let p_out_data = VulkanBuffer::read_back(device, host_visible_mt, queue, qfi, &p_out_buf)?;

    // Cleanup
    drop(a_strict_buf); drop(b_mask_buf); drop(v_prime_buf); drop(q_s_scaled_buf);
    drop(beta_buf); drop(decay_last_col_buf); drop(out_buf); drop(p_out_buf);

    let out_shape = vec![batch, heads, chunk, dv];
    let out_tensor = create_tensor_from_data(&out_data, &out_shape, DType::F32)?;
    let p_out_tensor = create_tensor_from_data(&p_out_data, &out_shape, DType::F32)?;
    Ok((out_tensor, p_out_tensor))
}

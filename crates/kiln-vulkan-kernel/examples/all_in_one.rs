//! All-in-one command buffer test: dispatch + copy in same cmd buffer.

use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::ptr::write_bytes;

fn main() -> Result<()> {
    let entry = unsafe { ash::Entry::load() }?;
    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"test\0").unwrap())
        .engine_name(CStr::from_bytes_with_nul(b"test\0").unwrap())
        .api_version(vk::make_api_version(0, 1, 2, 0))
        .build();
    let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info).build();
    let instance = unsafe { entry.create_instance(&instance_info, None)? };

    let phys_devices = unsafe { instance.enumerate_physical_devices()? };
    let phys = phys_devices[0];
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };

    let device_local_mt = (0..mem_props.memory_types.len())
        .find(|&i| mem_props.memory_types[i].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
        .context("no device local")?;
    let host_visible_mt = (0..mem_props.memory_types.len())
        .find(|&i| mem_props.memory_types[i].property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT))
        .context("no host visible")?;

    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(phys) };
    let qfi = queue_families.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)).context("no compute")? as u32;

    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(qfi).queue_priorities(&[1.0]).build();
    let device_info = vk::DeviceCreateInfo::builder().queue_create_infos(&[queue_info]).build();
    let device = unsafe { instance.create_device(phys, &device_info, None)? };
    let queue = unsafe { device.get_device_queue(qfi, 0) };

    // Compile shader
    let spirv_bytes = kiln_vulkan_kernel::pipeline::ShaderPipeline::compile_shader("/tmp/const_write_small.comp")?;
    let spirv_words: &[u32] = bytemuck::cast_slice(&spirv_bytes);

    // Shader module
    let smci = vk::ShaderModuleCreateInfo::builder().code(spirv_words).build();
    let shader_mod = unsafe { device.create_shader_module(&smci, None)? };

    // Output buffer
    let buf_size: u64 = 16;
    let bci = vk::BufferCreateInfo::builder()
        .size(buf_size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
        .build();
    let buf = unsafe { device.create_buffer(&bci, None)? };
    let reqs = unsafe { device.get_buffer_memory_requirements(buf) };
    let mai = vk::MemoryAllocateInfo::builder()
        .allocation_size(reqs.size).memory_type_index(device_local_mt as u32).build();
    let mem = unsafe { device.allocate_memory(&mai, None)? };
    unsafe { device.bind_buffer_memory(buf, mem, 0)? };

    // Staging buffer
    let sbci = vk::BufferCreateInfo::builder()
        .size(buf_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .build();
    let sbuf = unsafe { device.create_buffer(&sbci, None)? };
    let sreqs = unsafe { device.get_buffer_memory_requirements(sbuf) };
    let smai = vk::MemoryAllocateInfo::builder()
        .allocation_size(sreqs.size).memory_type_index(host_visible_mt as u32).build();
    let smem = unsafe { device.allocate_memory(&smai, None)? };
    unsafe { device.bind_buffer_memory(sbuf, smem, 0)? };

    // Buffer view
    let bvci = vk::BufferViewCreateInfo::builder()
        .buffer(buf).format(vk::Format::R32_SFLOAT).offset(0).range(vk::WHOLE_SIZE).build();
    let bview = unsafe { device.create_buffer_view(&bvci, None)? };

    // DSL
    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0).descriptor_type(vk::DescriptorType::STORAGE_TEXEL_BUFFER)
        .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build();
    let dslci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[binding]).build();
    let dsl = unsafe { device.create_descriptor_set_layout(&dslci, None)? };

    // Pipeline layout
    let pcr = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE).size(4).build();
    let plci = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[dsl]).push_constant_ranges(&[pcr]).build();
    let layout = unsafe { device.create_pipeline_layout(&plci, None)? };

    // Pipeline
    let pssci = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE).module(shader_mod)
        .name(CStr::from_bytes_with_nul(b"main\0").unwrap()).build();
    let cpci = vk::ComputePipelineCreateInfo::builder().stage(pssci).layout(layout).build();
    let pipelines = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[cpci], None) }
        .map_err(|(_, e)| anyhow::anyhow!("pipeline error: {:?}", e))?;
    let pipeline = pipelines[0];

    // Descriptor pool + set
    let dps = vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::STORAGE_TEXEL_BUFFER).descriptor_count(1).build();
    let dpci = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&[dps]).build();
    let pool = unsafe { device.create_descriptor_pool(&dpci, None)? };
    let dsaai = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool).set_layouts(&[dsl]).build();
    let sets = unsafe { device.allocate_descriptor_sets(&dsaai)? };
    let dset = sets[0];

    // Update descriptor (zero-init WriteDescriptorSet)
    let mut wds: MaybeUninit<vk::WriteDescriptorSet> = MaybeUninit::uninit();
    unsafe { write_bytes(wds.as_mut_ptr(), 0, 1); }
    unsafe {
        let p = wds.as_mut_ptr();
        (*p).s_type = vk::StructureType::WRITE_DESCRIPTOR_SET;
        (*p).dst_set = dset;
        (*p).dst_binding = 0;
        (*p).descriptor_count = 1;
        (*p).descriptor_type = vk::DescriptorType::STORAGE_TEXEL_BUFFER;
        (*p).p_texel_buffer_view = &bview as *const _;
    }
    unsafe { device.update_descriptor_sets(&[wds.assume_init()], &[]); }

    // Command pool (zero-init)
    let mut pool_info: MaybeUninit<vk::CommandPoolCreateInfo> = MaybeUninit::uninit();
    unsafe { write_bytes(pool_info.as_mut_ptr(), 0, 1); }
    unsafe {
        let p = pool_info.as_mut_ptr();
        (*p).s_type = vk::StructureType::COMMAND_POOL_CREATE_INFO;
        (*p).queue_family_index = qfi;
        (*p).flags = vk::CommandPoolCreateFlags::empty();
    }
    let cmd_pool = unsafe { device.create_command_pool(&pool_info.assume_init(), None)? };

    // Allocate command buffer (using ash wrapper)
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(cmd_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1)
        .build();
    let command_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
    let cmd = command_buffers[0];

    // Begin
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).build();
    unsafe { device.begin_command_buffer(cmd, &begin_info)? };

    // Record: bind, dispatch, barrier, copy, barrier
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[dset], &[]);
        device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&[4u32]));

        device.cmd_dispatch(cmd, 1, 1, 1);

        // Barrier: compute -> transfer
        let mut barrier: MaybeUninit<vk::MemoryBarrier> = MaybeUninit::uninit();
        unsafe { write_bytes(barrier.as_mut_ptr(), 0, 1); }
        unsafe {
            let p = barrier.as_mut_ptr();
            (*p).s_type = vk::StructureType::MEMORY_BARRIER;
            (*p).src_access_mask = vk::AccessFlags::SHADER_WRITE;
            (*p).dst_access_mask = vk::AccessFlags::TRANSFER_READ;
        }
        device.cmd_pipeline_barrier(
            cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(), &[barrier.assume_init()], &[], &[],
        );

        // Copy to staging
        let copy = vk::BufferCopy::builder().size(buf_size).build();
        device.cmd_copy_buffer(cmd, buf, sbuf, &[copy]);

        // Barrier: transfer -> host
        let mut barrier2: MaybeUninit<vk::MemoryBarrier> = MaybeUninit::uninit();
        unsafe { write_bytes(barrier2.as_mut_ptr(), 0, 1); }
        unsafe {
            let p = barrier2.as_mut_ptr();
            (*p).s_type = vk::StructureType::MEMORY_BARRIER;
            (*p).src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            (*p).dst_access_mask = vk::AccessFlags::HOST_READ;
        }
        device.cmd_pipeline_barrier(
            cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(), &[barrier2.assume_init()], &[], &[],
        );

        device.end_command_buffer(cmd)?;
    }

    // Submit (zero-init)
    let mut submit_info: MaybeUninit<vk::SubmitInfo> = MaybeUninit::uninit();
    unsafe { write_bytes(submit_info.as_mut_ptr(), 0, 1); }
    unsafe {
        let p = submit_info.as_mut_ptr();
        (*p).s_type = vk::StructureType::SUBMIT_INFO;
        (*p).command_buffer_count = 1;
        (*p).p_command_buffers = core::ptr::addr_of!(cmd);
    }
    unsafe {
        device.queue_submit(queue, &[submit_info.assume_init()], vk::Fence::null())?;
        device.queue_wait_idle(queue)?;
    }

    // Map and read
    let ptr = unsafe { device.map_memory(smem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? };
    let data: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr as *const f32, 4).to_vec() };
    println!("result = {:?}", data);
    let all_99 = data.iter().all(|&x| (x - 99.0).abs() < 1e-4);
    println!("TEST: {}", if all_99 { "PASS" } else { "FAIL" });

    Ok(())
}

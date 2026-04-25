//! Minimal raw Vulkan compute test - all in one command buffer.

use anyhow::{Context, Result};
use ash::vk;
use std::ffi::CStr;
use std::mem::MaybeUninit;
use std::ptr::write_bytes;

fn main() -> Result<()> {
    // Create Vulkan instance
    let entry = unsafe { ash::Entry::load() }.context("load entry")?;
    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"test\0").unwrap())
        .engine_name(CStr::from_bytes_with_nul(b"test\0").unwrap())
        .api_version(vk::make_api_version(0, 1, 2, 0))
        .build();
    let instance_info = vk::InstanceCreateInfo::builder().application_info(&app_info);
    let instance = unsafe { entry.create_instance(&instance_info, None).context("instance")? };

    // Get physical device
    let phys_devices = unsafe { instance.enumerate_physical_devices().context("phys devices")? };
    let phys = phys_devices[0];
    let props = unsafe { instance.get_physical_device_properties(phys) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(phys) };

    // Find memory types
    let device_local_mt = (0..mem_props.memory_types.len())
        .find(|&i| mem_props.memory_types[i].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
        .context("no device local mem")?;
    let host_visible_mt = (0..mem_props.memory_types.len())
        .find(|&i| mem_props.memory_types[i].property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT))
        .context("no host visible mem")?;

    // Find compute queue
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(phys) };
    let qfi = queue_families.iter().position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE)).context("no compute queue")? as u32;

    // Create device
    let queue_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(qfi).queue_priorities(&[1.0]).build();
    let device_info = vk::DeviceCreateInfo::builder().queue_create_infos(&[queue_info]).build();
    let device = unsafe { instance.create_device(phys, &device_info, None).context("device")? };
    let queue = unsafe { device.get_device_queue(qfi, 0) };

    println!("Device ready");

    // Compile shader
    let spirv_bytes = kiln_vulkan_kernel::pipeline::ShaderPipeline::compile_shader("/tmp/const_write_small.comp")?;
    let spirv_words: &[u32] = bytemuck::cast_slice(&spirv_bytes);

    // Shader module
    let smci = vk::ShaderModuleCreateInfo::builder().code(spirv_words).build();
    let shader_mod = unsafe { device.create_shader_module(&smci, None).context("shader module")? };
    println!("Shader module created");

    // Output buffer (device local)
    let buf_size: u64 = 16; // 4 floats
    let bci = vk::BufferCreateInfo::builder()
        .size(buf_size)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
        .build();
    let buf = unsafe { device.create_buffer(&bci, None).context("buffer")? };
    let reqs = unsafe { device.get_buffer_memory_requirements(buf) };
    let mai = vk::MemoryAllocateInfo::builder()
        .allocation_size(reqs.size).memory_type_index(device_local_mt as u32).build();
    let mem = unsafe { device.allocate_memory(&mai, None).context("memory")? };
    unsafe { device.bind_buffer_memory(buf, mem, 0).context("bind")? };
    println!("Output buffer created");

    // Buffer view
    let bvci = vk::BufferViewCreateInfo::builder()
        .buffer(buf).format(vk::Format::R32_SFLOAT).offset(0).range(vk::WHOLE_SIZE).build();
    let bview = unsafe { device.create_buffer_view(&bvci, None).context("buffer view")? };
    println!("Buffer view created");

    // Descriptor set layout
    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0).descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
        .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE).build();
    let dslci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[binding]).build();
    let dsl = unsafe { device.create_descriptor_set_layout(&dslci, None).context("dsl")? };

    // Pipeline layout
    let pcr = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE).size(4).build();
    let plci = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[dsl]).push_constant_ranges(&[pcr]).build();
    let layout = unsafe { device.create_pipeline_layout(&plci, None).context("layout")? };

    // Pipeline
    let pssci = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE).module(shader_mod)
        .name(CStr::from_bytes_with_nul(b"main\0").unwrap()).build();
    let cpci = vk::ComputePipelineCreateInfo::builder().stage(pssci).layout(layout).build();
    let pipelines = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &[cpci], None)
            .map_err(|(_, e)| anyhow::anyhow!("pipeline error: {:?}", e))?
    };
    let pipeline = pipelines[0];
    println!("Pipeline created");

    // Descriptor pool + set
    let dps = vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::UNIFORM_TEXEL_BUFFER).descriptor_count(1).build();
    let dpci = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&[dps]).build();
    let pool = unsafe { device.create_descriptor_pool(&dpci, None).context("pool")? };
    
    let dsaai = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(pool).set_layouts(&[dsl]).build();
    let sets = unsafe { device.allocate_descriptor_sets(&dsaai).context("alloc set")? };
    let dset = sets[0];

    // Update descriptor
    let wds = vk::WriteDescriptorSet::builder()
        .dst_set(dset).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_TEXEL_BUFFER)
        .texel_buffer_view(core::slice::from_ref(&bview)).build();
    unsafe { device.update_descriptor_sets(&[wds], &[]); }
    println!("Descriptor updated");

    // Command buffer
    let mut pool_info: MaybeUninit<vk::CommandPoolCreateInfo> = MaybeUninit::uninit();
    unsafe { write_bytes(pool_info.as_mut_ptr(), 0, 1); }
    unsafe {
        let p = pool_info.as_mut_ptr();
        (*p).s_type = vk::StructureType::COMMAND_POOL_CREATE_INFO;
        (*p).queue_family_index = qfi;
        (*p).flags = vk::CommandPoolCreateFlags::empty();
    }
    let cmd_pool = unsafe { device.create_command_pool(&pool_info.assume_init(), None).context("cmd pool")? };

    // Allocate command buffer (raw to avoid ash bug)
    let lib = unsafe { libloading::os::unix::Library::open::<&str>(Some("libvulkan.so.1"), 0).context("dlopen")? };
    type AllocFn = unsafe extern "system" fn(vk::Device, *const vk::CommandBufferAllocateInfo, *mut vk::CommandBuffer) -> vk::Result;
    let alloc_fn: libloading::os::unix::Symbol<AllocFn> = unsafe { lib.get(b"vkAllocateCommandBuffers\0").context("dlsym")? };

    let mut alloc_info: MaybeUninit<vk::CommandBufferAllocateInfo> = MaybeUninit::uninit();
    unsafe { write_bytes(alloc_info.as_mut_ptr(), 0, 1); }
    unsafe {
        let p = alloc_info.as_mut_ptr();
        (*p).s_type = vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO;
        (*p).command_pool = cmd_pool;
        (*p).level = vk::CommandBufferLevel::PRIMARY;
        (*p).command_buffer_count = 1;
    }
    let mut cmd: vk::CommandBuffer = vk::CommandBuffer::null();
    let res = unsafe { alloc_fn(device.handle(), &alloc_info.assume_init(), &mut cmd) };
    println!("Alloc cmd buffer result: {:?}", res);

    // Begin command buffer
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).build();
    unsafe { device.begin_command_buffer(cmd, &begin_info).context("begin cmd")? };

    // Bind pipeline + descriptors + push constants
    // Staging buffer for readback (created before cmd recording)
    let sbci = vk::BufferCreateInfo::builder()
        .size(buf_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST)
        .build();
    let sbuf = unsafe { device.create_buffer(&sbci, None).context("staging buffer")? };
    let sreqs = unsafe { device.get_buffer_memory_requirements(sbuf) };
    let smai = vk::MemoryAllocateInfo::builder()
        .allocation_size(sreqs.size).memory_type_index(host_visible_mt as u32).build();
    let smem = unsafe { device.allocate_memory(&smai, None).context("staging memory")? };
    unsafe { device.bind_buffer_memory(sbuf, smem, 0).context("bind staging")? };

    // Begin command buffer
    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT).build();
    unsafe { device.begin_command_buffer(cmd, &begin_info).context("begin cmd")? };

    // Bind pipeline + descriptors + push constants
    unsafe {
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, layout, 0, &[dset], &[]);
        device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&[4u32]));
        
        // Dispatch
        device.cmd_dispatch(cmd, 1, 1, 1);
        println!("Dispatch recorded");

        // Memory barrier
        let barrier = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .build();
        device.cmd_pipeline_barrier(
            cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(), &[barrier], &[], &[],
        );

        // Copy
        let copy = vk::BufferCopy::builder().size(buf_size).build();
        device.cmd_copy_buffer(cmd, buf, sbuf, &[copy]);

        // Second barrier for staging buffer
        let barrier2 = vk::MemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .build();
        device.cmd_pipeline_barrier(
            cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(), &[barrier2], &[], &[],
        );

        device.end_command_buffer(cmd).context("end cmd")?;
    }
    println!("Command buffer ended");

    // Submit
    let submit_info = vk::SubmitInfo::builder().command_buffers(&[cmd]).build();
    unsafe {
        device.queue_submit(queue, &[submit_info], vk::Fence::null()).context("submit")?;
        device.queue_wait_idle(queue).context("wait")?;
    }
    println!("Submit done");

    // Map and read
    let ptr = unsafe { device.map_memory(smem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).context("map")? };
    let data: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr as *const f32, 4).to_vec() };
    println!("result = {:?}", data);
    let all_99 = data.iter().all(|&x| (x - 99.0).abs() < 1e-4);
    println!("TEST: {}", if all_99 { "PASS" } else { "FAIL" });

    Ok(())
}

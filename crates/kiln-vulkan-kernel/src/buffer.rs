use anyhow::{Context, Result};
use ash::vk;
use std::mem::MaybeUninit;
use std::ptr::write_bytes;
use std::sync::Arc;

/// Vulkan buffer wrapper for Kiln tensor data.
pub struct VulkanBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    #[allow(dead_code)]
    device: Arc<ash::Device>,
}

impl std::fmt::Debug for VulkanBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanBuffer").field("size", &self.size).finish()
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.free_memory(self.memory, None);
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}

/// Create a zero-init CommandPoolCreateInfo with fixed sType.
fn make_pool_info(queue_family_index: u32) -> vk::CommandPoolCreateInfo {
    let mut info: MaybeUninit<vk::CommandPoolCreateInfo> = MaybeUninit::uninit();
    unsafe { write_bytes(info.as_mut_ptr(), 0, 1); }
    unsafe {
        let ptr = info.as_mut_ptr();
        (*ptr).s_type = vk::StructureType::COMMAND_POOL_CREATE_INFO;
        (*ptr).queue_family_index = queue_family_index;
        (*ptr).flags = vk::CommandPoolCreateFlags::empty();
    }
    unsafe { info.assume_init() }
}

/// Create a zero-init CommandBufferAllocateInfo with fixed sType.
fn make_alloc_info(pool: vk::CommandPool) -> vk::CommandBufferAllocateInfo {
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
fn make_begin_info() -> vk::CommandBufferBeginInfo {
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

impl VulkanBuffer {
    /// Create a device-local buffer (GPU-only, fast access).
    pub fn create_device_local(device: &Arc<ash::Device>, mem_type_index: u32, size: u64) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .build();

        let buffer = unsafe {
            device.create_buffer(&buffer_info, None)
                .context("failed to create storage buffer")?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index)
            .build();

        let memory = unsafe {
            device.allocate_memory(&alloc_info, None)
                .context("failed to allocate device memory")?
        };

        unsafe {
            device.bind_buffer_memory(buffer, memory, 0)
                .context("failed to bind memory to buffer")?;
        }

        Ok(Self {
            buffer,
            memory,
            size,
            device: Arc::clone(device),
        })
    }

    /// Create a host-visible buffer (for uploading data).
    pub fn create_host_visible(device: &Arc<ash::Device>, mem_type_index: u32, size: u64) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)
            .build();

        let buffer = unsafe {
            device.create_buffer(&buffer_info, None)
                .context("failed to create host buffer")?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index)
            .build();

        let memory = unsafe {
            device.allocate_memory(&alloc_info, None)
                .context("failed to allocate host memory")?
        };

        unsafe {
            device.bind_buffer_memory(buffer, memory, 0)
                .context("failed to bind memory to buffer")?;
        }

        Ok(Self {
            buffer,
            memory,
            size,
            device: Arc::clone(device),
        })
    }

    /// Upload data from CPU to this buffer via a staging buffer.
    pub fn upload_data(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        queue_family_index: u32,
        dst: &VulkanBuffer,
        data: &[u8],
    ) -> Result<()> {
        eprintln!("[upload] creating staging buffer");
        let staging = VulkanBuffer::create_host_visible(device, host_mem_type, data.len() as u64)?;
        eprintln!("[upload] staging buffer created");

        // Map and copy data to staging buffer
        eprintln!("[upload] mapping memory");
        let mapped_ptr = unsafe {
            device.map_memory(
                staging.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            ).map_err(|e| anyhow::anyhow!("failed to map memory: {:?}", e))?
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                mapped_ptr as *mut u8,
                data.len(),
            );
        }
        eprintln!("[upload] data copied to staging");

        // Create command buffer for transfer
        let pool_info = make_pool_info(queue_family_index);
        let pool = unsafe {
            device.create_command_pool(&pool_info, None)
                .context("failed to create command pool")?
        };

        let alloc_info = make_alloc_info(pool);
        let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
            .context("failed to allocate command buffer")?;
        let cmd = command_buffers[0];

        let begin_info = make_begin_info();
        unsafe { device.begin_command_buffer(cmd, &begin_info).context("failed to begin command buffer")? };

        let copy = vk::BufferCopy::builder()
            .size(data.len() as u64)
            .build();
        unsafe {
            device.cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[copy]);
        }

        unsafe { device.end_command_buffer(cmd).context("failed to end command buffer")? };

        let cmds = vec![cmd];
        let submit_info = make_submit_info(&cmds);
        unsafe {
            device.queue_submit(queue, &[submit_info], vk::Fence::null())
                .context("failed to submit transfer")?;
            device.queue_wait_idle(queue)
                .context("failed to wait for queue")?;
        }

        unsafe {
            device.unmap_memory(staging.memory);
            device.free_command_buffers(pool, &command_buffers);
            device.destroy_command_pool(pool, None);
        }

        Ok(())
    }

    /// Read data back from this buffer to CPU.
    pub fn read_back(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        queue_family_index: u32,
        src: &VulkanBuffer,
    ) -> Result<Vec<u8>> {
        let staging = VulkanBuffer::create_host_visible(device, host_mem_type, src.size)?;

        // Create command buffer
        let pool_info = make_pool_info(queue_family_index);
        let pool = unsafe {
            device.create_command_pool(&pool_info, None)
                .context("failed to create command pool")?
        };

        let alloc_info = make_alloc_info(pool);
        let command_buffers = crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
            .context("failed to allocate command buffer")?;
        let cmd = command_buffers[0];

        let begin_info = make_begin_info();
        unsafe { device.begin_command_buffer(cmd, &begin_info).context("failed to begin command buffer")? };

        let copy = vk::BufferCopy::builder()
            .size(src.size)
            .build();
        unsafe {
            device.cmd_copy_buffer(cmd, src.buffer, staging.buffer, &[copy]);
        }

        unsafe { device.end_command_buffer(cmd).context("failed to end command buffer")? };

        let cmds = vec![cmd];
        let submit_info = make_submit_info(&cmds);
        unsafe {
            device.queue_submit(queue, &[submit_info], vk::Fence::null())
                .context("failed to submit readback")?;
            device.queue_wait_idle(queue)
                .context("failed to wait for queue")?;
        }

        // Map and read data
        let mapped_ptr = unsafe {
            device.map_memory(
                staging.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            ).map_err(|e| anyhow::anyhow!("failed to map memory: {:?}", e))?
        };

        let data: Vec<u8> = unsafe {
            std::slice::from_raw_parts(mapped_ptr as *const u8, src.size as usize).to_vec()
        };

        unsafe {
            device.unmap_memory(staging.memory);
            device.free_command_buffers(pool, &command_buffers);
            device.destroy_command_pool(pool, None);
        }

        Ok(data)
    }

    /// Get the buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    /// Get the buffer size.
    pub fn size(&self) -> u64 {
        self.size
    }
}

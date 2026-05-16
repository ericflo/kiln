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
        f.debug_struct("VulkanBuffer")
            .field("size", &self.size)
            .finish()
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

/// Create a CommandPoolCreateInfo via the ash 0.38 default+chained builder.
fn make_pool_info(queue_family_index: u32) -> vk::CommandPoolCreateInfo<'static> {
    vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::empty())
}

/// CommandBufferAllocateInfo via default+chained builder.
fn make_alloc_info(pool: vk::CommandPool) -> vk::CommandBufferAllocateInfo<'static> {
    vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1)
}

/// CommandBufferBeginInfo via default+chained builder.
fn make_begin_info() -> vk::CommandBufferBeginInfo<'static> {
    vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
}

/// SubmitInfo via default+chained builder.
fn make_submit_info(cmds: &[vk::CommandBuffer]) -> vk::SubmitInfo<'_> {
    vk::SubmitInfo::default().command_buffers(cmds)
}

impl VulkanBuffer {
    /// Create a device-local buffer (GPU-only, fast access).
    pub fn create_device_local(
        device: &Arc<ash::Device>,
        mem_type_index: u32,
        size: u64,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            ;

        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .context("failed to create storage buffer")?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index)
            ;

        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .context("failed to allocate device memory")?
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
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
    pub fn create_host_visible(
        device: &Arc<ash::Device>,
        mem_type_index: u32,
        size: u64,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(
                vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER,
            )
            ;

        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .context("failed to create host buffer")?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index)
            ;

        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .context("failed to allocate host memory")?
        };

        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
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
    /// Upload `data` to `dst` starting at byte offset `dst_offset`.
    /// Used by the resident KV-cache seeding path so we only stage and
    /// copy the request's active blocks (a few tens of KB per layer)
    /// instead of the whole multi-GB pool slab.
    pub fn upload_data_at_offset(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        queue_family_index: u32,
        dst: &VulkanBuffer,
        dst_offset: u64,
        data: &[u8],
    ) -> Result<()> {
        let staging = VulkanBuffer::create_host_visible(device, host_mem_type, data.len() as u64)?;
        let mapped_ptr = unsafe {
            device
                .map_memory(
                    staging.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| anyhow::anyhow!("failed to map memory: {:?}", e))?
        };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut u8, data.len());
        }
        let pool_info = make_pool_info(queue_family_index);
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("failed to create command pool")?
        };
        let alloc_info = make_alloc_info(pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("failed to allocate command buffer")?;
        let cmd = command_buffers[0];
        let begin_info = make_begin_info();
        unsafe {
            device
                .begin_command_buffer(cmd, &begin_info)
                .context("failed to begin command buffer")?
        };
        let copy = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(dst_offset)
            .size(data.len() as u64);
        unsafe {
            device.cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[copy]);
            device
                .end_command_buffer(cmd)
                .context("failed to end command buffer")?;
        }
        let cmds = vec![cmd];
        let submit_info = make_submit_info(&cmds);
        unsafe {
            device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .context("failed to submit transfer")?;
            device
                .queue_wait_idle(queue)
                .context("failed to wait for queue")?;
            device.unmap_memory(staging.memory);
            device.free_command_buffers(pool, &command_buffers);
            device.destroy_command_pool(pool, None);
        }
        Ok(())
    }

    pub fn upload_data(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        queue_family_index: u32,
        dst: &VulkanBuffer,
        data: &[u8],
    ) -> Result<()> {
        tracing::trace!("[upload] creating staging buffer");
        let staging = VulkanBuffer::create_host_visible(device, host_mem_type, data.len() as u64)?;
        tracing::trace!("[upload] staging buffer created");

        // Map and copy data to staging buffer
        tracing::trace!("[upload] mapping memory");
        let mapped_ptr = unsafe {
            device
                .map_memory(
                    staging.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| anyhow::anyhow!("failed to map memory: {:?}", e))?
        };

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut u8, data.len());
        }
        tracing::trace!("[upload] data copied to staging");

        // Create command buffer for transfer
        let pool_info = make_pool_info(queue_family_index);
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("failed to create command pool")?
        };

        let alloc_info = make_alloc_info(pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("failed to allocate command buffer")?;
        let cmd = command_buffers[0];

        let begin_info = make_begin_info();
        unsafe {
            device
                .begin_command_buffer(cmd, &begin_info)
                .context("failed to begin command buffer")?
        };

        let copy = vk::BufferCopy::default().size(data.len() as u64);
        unsafe {
            device.cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[copy]);
        }

        unsafe {
            device
                .end_command_buffer(cmd)
                .context("failed to end command buffer")?
        };

        let cmds = vec![cmd];
        let submit_info = make_submit_info(&cmds);
        unsafe {
            device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .context("failed to submit transfer")?;
            device
                .queue_wait_idle(queue)
                .context("failed to wait for queue")?;
        }

        unsafe {
            device.unmap_memory(staging.memory);
            device.free_command_buffers(pool, &command_buffers);
            device.destroy_command_pool(pool, None);
        }

        Ok(())
    }

    /// Batch-upload multiple small payloads to multiple destination
    /// buffers in a single command buffer + single queue submission.
    ///
    /// Caller passes `&[(&dst_buffer, &payload_bytes)]`. The function
    /// allocates one staging buffer per upload (each must hold its
    /// own payload size — staging buffers can't safely overlap), but
    /// only creates ONE command pool, ONE command buffer, and submits
    /// + waits ONCE — collapsing what would otherwise be N round
    /// trips into 1. Dramatically faster for the decode hot loop where
    /// 4-5 per-token small inputs (RoPE cos/sin, block_table, seq_lens,
    /// embedding) used to take ~6 ms / token through `upload_data`.
    pub fn upload_data_batch(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        queue_family_index: u32,
        uploads: &[(&VulkanBuffer, &[u8])],
    ) -> Result<()> {
        if uploads.is_empty() {
            return Ok(());
        }
        // Allocate one staging buffer per upload, copy payload into
        // its mapped memory. Keep stagings alive in `stagings` so
        // they outlast the GPU submit.
        let mut stagings: Vec<VulkanBuffer> = Vec::with_capacity(uploads.len());
        for (_dst, data) in uploads.iter() {
            let staging = VulkanBuffer::create_host_visible(device, host_mem_type, data.len() as u64)?;
            let mapped = unsafe {
                device
                    .map_memory(staging.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                    .map_err(|e| anyhow::anyhow!("upload_data_batch: map_memory failed: {:?}", e))?
            };
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), mapped as *mut u8, data.len());
                device.unmap_memory(staging.memory);
            }
            stagings.push(staging);
        }

        let pool_info = make_pool_info(queue_family_index);
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("upload_data_batch: create_command_pool")?
        };
        let alloc_info = make_alloc_info(pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("upload_data_batch: allocate_command_buffers")?;
        let cmd = command_buffers[0];
        unsafe {
            device
                .begin_command_buffer(cmd, &make_begin_info())
                .context("upload_data_batch: begin_command_buffer")?;
            for (i, (dst, data)) in uploads.iter().enumerate() {
                let copy = vk::BufferCopy::default().size(data.len() as u64);
                device.cmd_copy_buffer(cmd, stagings[i].buffer, dst.buffer, &[copy]);
            }
            device
                .end_command_buffer(cmd)
                .context("upload_data_batch: end_command_buffer")?;
            device
                .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
                .context("upload_data_batch: queue_submit")?;
            device
                .queue_wait_idle(queue)
                .context("upload_data_batch: queue_wait_idle")?;
            device.free_command_buffers(pool, &command_buffers);
            device.destroy_command_pool(pool, None);
        }
        // Drop stagings here (their Drop releases the host-visible memory).
        drop(stagings);
        Ok(())
    }

    /// Upload data using an externally synchronized reusable command pool.
    pub fn upload_data_with_command_pool(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        dst: &VulkanBuffer,
        data: &[u8],
    ) -> Result<()> {
        let staging = VulkanBuffer::create_host_visible(device, host_mem_type, data.len() as u64)?;

        let mapped_ptr = unsafe {
            device
                .map_memory(
                    staging.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| anyhow::anyhow!("failed to map memory: {:?}", e))?
        };

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut u8, data.len());
        }

        let alloc_info = make_alloc_info(command_pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("failed to allocate command buffer")?;
        let cmd = command_buffers[0];

        unsafe {
            device
                .begin_command_buffer(cmd, &make_begin_info())
                .context("failed to begin command buffer")?;
            device.cmd_copy_buffer(
                cmd,
                staging.buffer,
                dst.buffer,
                &[vk::BufferCopy::default().size(data.len() as u64)],
            );
            device
                .end_command_buffer(cmd)
                .context("failed to end command buffer")?;
            device
                .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
                .context("failed to submit transfer")?;
            device
                .queue_wait_idle(queue)
                .context("failed to wait for queue")?;
            device.unmap_memory(staging.memory);
            device.free_command_buffers(command_pool, &command_buffers);
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
            device
                .create_command_pool(&pool_info, None)
                .context("failed to create command pool")?
        };

        let alloc_info = make_alloc_info(pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("failed to allocate command buffer")?;
        let cmd = command_buffers[0];

        let begin_info = make_begin_info();
        unsafe {
            device
                .begin_command_buffer(cmd, &begin_info)
                .context("failed to begin command buffer")?
        };

        let copy = vk::BufferCopy::default().size(src.size);
        unsafe {
            device.cmd_copy_buffer(cmd, src.buffer, staging.buffer, &[copy]);
        }

        unsafe {
            device
                .end_command_buffer(cmd)
                .context("failed to end command buffer")?
        };

        let cmds = vec![cmd];
        let submit_info = make_submit_info(&cmds);
        unsafe {
            device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .context("failed to submit readback")?;
            device
                .queue_wait_idle(queue)
                .context("failed to wait for queue")?;
        }

        // Map and read data
        let mapped_ptr = unsafe {
            device
                .map_memory(
                    staging.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| anyhow::anyhow!("failed to map memory: {:?}", e))?
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

    /// Read data back using an externally synchronized reusable command pool.
    pub fn read_back_with_command_pool(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        src: &VulkanBuffer,
    ) -> Result<Vec<u8>> {
        let staging = VulkanBuffer::create_host_visible(device, host_mem_type, src.size)?;

        let alloc_info = make_alloc_info(command_pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("failed to allocate command buffer")?;
        let cmd = command_buffers[0];

        unsafe {
            device
                .begin_command_buffer(cmd, &make_begin_info())
                .context("failed to begin command buffer")?;
            device.cmd_copy_buffer(
                cmd,
                src.buffer,
                staging.buffer,
                &[vk::BufferCopy::default().size(src.size)],
            );
            device
                .end_command_buffer(cmd)
                .context("failed to end command buffer")?;
            device
                .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
                .context("failed to submit readback")?;
            device
                .queue_wait_idle(queue)
                .context("failed to wait for queue")?;
        }

        let mapped_ptr = unsafe {
            device
                .map_memory(
                    staging.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| anyhow::anyhow!("failed to map memory: {:?}", e))?
        };
        let data: Vec<u8> = unsafe {
            std::slice::from_raw_parts(mapped_ptr as *const u8, src.size as usize).to_vec()
        };

        unsafe {
            device.unmap_memory(staging.memory);
            device.free_command_buffers(command_pool, &command_buffers);
        }

        Ok(data)
    }

    /// Write directly into a host-visible buffer.
    pub fn write_host_visible(
        device: &Arc<ash::Device>,
        dst: &VulkanBuffer,
        data: &[u8],
    ) -> Result<()> {
        if data.len() as u64 > dst.size {
            anyhow::bail!(
                "host-visible write size {} exceeds buffer size {}",
                data.len(),
                dst.size
            );
        }
        let mapped_ptr = unsafe {
            device
                .map_memory(dst.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| anyhow::anyhow!("failed to map host-visible memory: {:?}", e))?
        };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut u8, data.len());
            device.unmap_memory(dst.memory);
        }
        Ok(())
    }

    /// Read directly from a host-visible buffer.
    pub fn read_host_visible(device: &Arc<ash::Device>, src: &VulkanBuffer) -> Result<Vec<u8>> {
        let mapped_ptr = unsafe {
            device
                .map_memory(src.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| anyhow::anyhow!("failed to map host-visible memory: {:?}", e))?
        };
        let data: Vec<u8> = unsafe {
            std::slice::from_raw_parts(mapped_ptr as *const u8, src.size as usize).to_vec()
        };
        unsafe {
            device.unmap_memory(src.memory);
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

    /// Map this buffer's memory and `memcpy` `bytes` into it. Only
    /// valid for buffers created with host-visible memory
    /// (`create_host_visible`). No GPU submission; the GPU pulls the
    /// data over PCIe on demand when the next dispatch reads.
    pub fn write_mapped(&self, bytes: &[u8]) -> Result<()> {
        anyhow::ensure!(
            (bytes.len() as u64) <= self.size,
            "write_mapped: {} bytes > buffer size {}",
            bytes.len(),
            self.size,
        );
        let mapped = unsafe {
            self.device
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| anyhow::anyhow!("write_mapped: map_memory: {:?}", e))?
        };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), mapped as *mut u8, bytes.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    /// Map this buffer's memory and copy the first `bytes_len` bytes
    /// into a fresh `Vec<u8>`. Only valid for buffers created with
    /// host-visible memory (`create_host_visible`). Callers writing
    /// to host-mapped memory should use `write_mapped` instead.
    pub fn read_mapped(&self, bytes_len: usize) -> Result<Vec<u8>> {
        let mapped = unsafe {
            self.device
                .map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(|e| anyhow::anyhow!("read_mapped: map_memory: {:?}", e))?
        };
        let bytes = unsafe {
            std::slice::from_raw_parts(mapped as *const u8, bytes_len).to_vec()
        };
        unsafe {
            self.device.unmap_memory(self.memory);
        }
        Ok(bytes)
    }
}

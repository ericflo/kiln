use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VulkanBufferAllocationStats {
    pub live_device_local_buffers: u64,
    pub live_device_local_bytes: u64,
    pub live_host_visible_buffers: u64,
    pub live_host_visible_bytes: u64,
    pub peak_live_bytes: u64,
    pub device_local_allocations: u64,
    pub device_local_allocated_bytes: u64,
    pub device_local_frees: u64,
    pub device_local_freed_bytes: u64,
    pub host_visible_allocations: u64,
    pub host_visible_allocated_bytes: u64,
    pub host_visible_frees: u64,
    pub host_visible_freed_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
enum VulkanBufferMemoryKind {
    DeviceLocal,
    HostVisible,
}

static LIVE_DEVICE_LOCAL_BUFFERS: AtomicU64 = AtomicU64::new(0);
static LIVE_DEVICE_LOCAL_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_HOST_VISIBLE_BUFFERS: AtomicU64 = AtomicU64::new(0);
static LIVE_HOST_VISIBLE_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_TOTAL_BYTES: AtomicU64 = AtomicU64::new(0);
static PEAK_LIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static DEVICE_LOCAL_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEVICE_LOCAL_ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static DEVICE_LOCAL_FREES: AtomicU64 = AtomicU64::new(0);
static DEVICE_LOCAL_FREED_BYTES: AtomicU64 = AtomicU64::new(0);
static HOST_VISIBLE_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static HOST_VISIBLE_ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static HOST_VISIBLE_FREES: AtomicU64 = AtomicU64::new(0);
static HOST_VISIBLE_FREED_BYTES: AtomicU64 = AtomicU64::new(0);

fn decrement_live(counter: &AtomicU64, amount: u64, counter_name: &'static str) {
    if counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_sub(amount)
        })
        .is_err()
    {
        tracing::error!(
            counter = counter_name,
            amount,
            current = counter.load(Ordering::Relaxed),
            "Vulkan buffer allocation counter underflow"
        );
    }
}

fn record_allocation(kind: VulkanBufferMemoryKind, bytes: u64) {
    match kind {
        VulkanBufferMemoryKind::DeviceLocal => {
            LIVE_DEVICE_LOCAL_BUFFERS.fetch_add(1, Ordering::Relaxed);
            LIVE_DEVICE_LOCAL_BYTES.fetch_add(bytes, Ordering::Relaxed);
            DEVICE_LOCAL_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            DEVICE_LOCAL_ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
        }
        VulkanBufferMemoryKind::HostVisible => {
            LIVE_HOST_VISIBLE_BUFFERS.fetch_add(1, Ordering::Relaxed);
            LIVE_HOST_VISIBLE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HOST_VISIBLE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            HOST_VISIBLE_ALLOCATED_BYTES.fetch_add(bytes, Ordering::Relaxed);
        }
    }
    let live_total = LIVE_TOTAL_BYTES
        .fetch_add(bytes, Ordering::Relaxed)
        .saturating_add(bytes);
    PEAK_LIVE_BYTES.fetch_max(live_total, Ordering::Relaxed);
}

fn record_free(kind: VulkanBufferMemoryKind, bytes: u64) {
    let (live_buffers, live_bytes, frees, freed_bytes) = match kind {
        VulkanBufferMemoryKind::DeviceLocal => (
            &LIVE_DEVICE_LOCAL_BUFFERS,
            &LIVE_DEVICE_LOCAL_BYTES,
            &DEVICE_LOCAL_FREES,
            &DEVICE_LOCAL_FREED_BYTES,
        ),
        VulkanBufferMemoryKind::HostVisible => (
            &LIVE_HOST_VISIBLE_BUFFERS,
            &LIVE_HOST_VISIBLE_BYTES,
            &HOST_VISIBLE_FREES,
            &HOST_VISIBLE_FREED_BYTES,
        ),
    };
    decrement_live(live_buffers, 1, "live_buffers");
    decrement_live(live_bytes, bytes, "live_bytes");
    decrement_live(&LIVE_TOTAL_BYTES, bytes, "live_total_bytes");
    frees.fetch_add(1, Ordering::Relaxed);
    freed_bytes.fetch_add(bytes, Ordering::Relaxed);
}

pub fn allocation_stats() -> VulkanBufferAllocationStats {
    VulkanBufferAllocationStats {
        live_device_local_buffers: LIVE_DEVICE_LOCAL_BUFFERS.load(Ordering::Relaxed),
        live_device_local_bytes: LIVE_DEVICE_LOCAL_BYTES.load(Ordering::Relaxed),
        live_host_visible_buffers: LIVE_HOST_VISIBLE_BUFFERS.load(Ordering::Relaxed),
        live_host_visible_bytes: LIVE_HOST_VISIBLE_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_LIVE_BYTES.load(Ordering::Relaxed),
        device_local_allocations: DEVICE_LOCAL_ALLOCATIONS.load(Ordering::Relaxed),
        device_local_allocated_bytes: DEVICE_LOCAL_ALLOCATED_BYTES.load(Ordering::Relaxed),
        device_local_frees: DEVICE_LOCAL_FREES.load(Ordering::Relaxed),
        device_local_freed_bytes: DEVICE_LOCAL_FREED_BYTES.load(Ordering::Relaxed),
        host_visible_allocations: HOST_VISIBLE_ALLOCATIONS.load(Ordering::Relaxed),
        host_visible_allocated_bytes: HOST_VISIBLE_ALLOCATED_BYTES.load(Ordering::Relaxed),
        host_visible_frees: HOST_VISIBLE_FREES.load(Ordering::Relaxed),
        host_visible_freed_bytes: HOST_VISIBLE_FREED_BYTES.load(Ordering::Relaxed),
    }
}

/// Vulkan buffer wrapper for Kiln tensor data.
pub struct VulkanBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    allocation_size: u64,
    memory_kind: VulkanBufferMemoryKind,
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
            // Bound resources must be destroyed before their memory is freed.
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
        record_free(self.memory_kind, self.allocation_size);
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
    fn create(
        device: &Arc<ash::Device>,
        mem_type_index: u32,
        size: u64,
        memory_kind: VulkanBufferMemoryKind,
        description: &str,
    ) -> Result<Self> {
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        );
        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .with_context(|| format!("failed to create {description} buffer"))?
        };
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);
        let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(error)
                    .with_context(|| format!("failed to allocate {description} buffer memory"));
            }
        };
        if let Err(error) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
            unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            }
            return Err(error)
                .with_context(|| format!("failed to bind {description} buffer memory"));
        }
        record_allocation(memory_kind, mem_requirements.size);
        Ok(Self {
            buffer,
            memory,
            size,
            allocation_size: mem_requirements.size,
            memory_kind,
            device: Arc::clone(device),
        })
    }

    /// Create a device-local buffer (GPU-only, fast access).
    pub fn create_device_local(
        device: &Arc<ash::Device>,
        mem_type_index: u32,
        size: u64,
    ) -> Result<Self> {
        Self::create(
            device,
            mem_type_index,
            size,
            VulkanBufferMemoryKind::DeviceLocal,
            "device-local",
        )
    }

    /// Create a host-visible buffer (for uploading data).
    pub fn create_host_visible(
        device: &Arc<ash::Device>,
        mem_type_index: u32,
        size: u64,
    ) -> Result<Self> {
        Self::create(
            device,
            mem_type_index,
            size,
            VulkanBufferMemoryKind::HostVisible,
            "host-visible",
        )
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

    /// Zero the whole buffer on-device via `vkCmdFillBuffer` (no staging).
    /// Vulkan does NOT guarantee fresh device allocations read as zero —
    /// RADV usually hands back kernel-zeroed pages, lavapipe hands back
    /// host malloc garbage — so anything documented as "zero-initialized"
    /// must call this explicitly.
    pub fn fill_zero(
        device: &Arc<ash::Device>,
        queue: vk::Queue,
        queue_family_index: u32,
        dst: &VulkanBuffer,
    ) -> Result<()> {
        let pool_info = make_pool_info(queue_family_index);
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("fill_zero: failed to create command pool")?
        };
        let alloc_info = make_alloc_info(pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("fill_zero: failed to allocate command buffer")?;
        let cmd = command_buffers[0];
        let begin_info = make_begin_info();
        unsafe {
            device
                .begin_command_buffer(cmd, &begin_info)
                .context("fill_zero: failed to begin command buffer")?;
            device.cmd_fill_buffer(cmd, dst.buffer, 0, vk::WHOLE_SIZE, 0);
            device
                .end_command_buffer(cmd)
                .context("fill_zero: failed to end command buffer")?;
            device
                .queue_submit(
                    queue,
                    &[make_submit_info(&command_buffers)],
                    vk::Fence::null(),
                )
                .context("fill_zero: failed to submit fill")?;
            device
                .queue_wait_idle(queue)
                .context("fill_zero: failed to wait for queue")?;
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
    /// packs every payload into one staging buffer, creates one command
    /// pool and one command buffer, then submits and waits once.
    /// Dramatically faster for the decode hot loop where 4-5 per-token
    /// small inputs (RoPE cos/sin, block_table, seq_lens, embedding)
    /// used to take ~6 ms / token through `upload_data`.
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
        let total_staging_bytes = uploads.iter().try_fold(0u64, |acc, (_, data)| {
            acc.checked_add(data.len() as u64)
                .ok_or_else(|| anyhow::anyhow!("upload_data_batch: staging size overflow"))
        })?;
        anyhow::ensure!(
            total_staging_bytes > 0,
            "upload_data_batch: total payload size must be non-zero"
        );
        let staging =
            VulkanBuffer::create_host_visible(device, host_mem_type, total_staging_bytes)?;
        let mapped = unsafe {
            device
                .map_memory(
                    staging.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| anyhow::anyhow!("upload_data_batch: map_memory failed: {:?}", e))?
        };
        let mut src_offsets = Vec::with_capacity(uploads.len());
        let mut src_offset = 0u64;
        for (idx, (_dst, data)) in uploads.iter().enumerate() {
            anyhow::ensure!(
                !data.is_empty(),
                "upload_data_batch[{idx}]: payload must be non-empty"
            );
            src_offsets.push(src_offset);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (mapped as *mut u8).add(src_offset as usize),
                    data.len(),
                );
            }
            src_offset += data.len() as u64;
        }
        unsafe {
            device.unmap_memory(staging.memory);
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
                let copy = vk::BufferCopy::default()
                    .src_offset(src_offsets[i])
                    .size(data.len() as u64);
                device.cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[copy]);
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
        drop(staging);
        Ok(())
    }

    /// Batch-upload multiple payloads to destination offsets in a
    /// single command buffer + single queue submission.
    pub fn upload_data_at_offset_batch(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        queue_family_index: u32,
        uploads: &[(&VulkanBuffer, u64, &[u8])],
    ) -> Result<()> {
        if uploads.is_empty() {
            return Ok(());
        }

        let total_staging_bytes = uploads.iter().try_fold(0u64, |acc, (_, _, data)| {
            acc.checked_add(data.len() as u64).ok_or_else(|| {
                anyhow::anyhow!("upload_data_at_offset_batch: staging size overflow")
            })
        })?;
        anyhow::ensure!(
            total_staging_bytes > 0,
            "upload_data_at_offset_batch: total payload size must be non-zero"
        );
        let staging =
            VulkanBuffer::create_host_visible(device, host_mem_type, total_staging_bytes)?;
        let mapped = unsafe {
            device
                .map_memory(
                    staging.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| {
                    anyhow::anyhow!("upload_data_at_offset_batch: map_memory failed: {:?}", e)
                })?
        };
        let mut src_offsets = Vec::with_capacity(uploads.len());
        let mut src_offset = 0u64;
        for (idx, (dst, dst_offset, data)) in uploads.iter().enumerate() {
            let end = dst_offset.checked_add(data.len() as u64).ok_or_else(|| {
                anyhow::anyhow!("upload_data_at_offset_batch[{idx}]: offset overflow")
            })?;
            anyhow::ensure!(
                !data.is_empty(),
                "upload_data_at_offset_batch[{idx}]: payload must be non-empty"
            );
            anyhow::ensure!(
                end <= dst.size,
                "upload_data_at_offset_batch[{idx}]: range [{dst_offset}, {end}) exceeds buffer size {}",
                dst.size
            );
            src_offsets.push(src_offset);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (mapped as *mut u8).add(src_offset as usize),
                    data.len(),
                );
            }
            src_offset += data.len() as u64;
        }
        unsafe {
            device.unmap_memory(staging.memory);
        }

        let pool_info = make_pool_info(queue_family_index);
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("upload_data_at_offset_batch: create_command_pool")?
        };
        let alloc_info = make_alloc_info(pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("upload_data_at_offset_batch: allocate_command_buffers")?;
        let cmd = command_buffers[0];
        unsafe {
            device
                .begin_command_buffer(cmd, &make_begin_info())
                .context("upload_data_at_offset_batch: begin_command_buffer")?;
            for (i, (dst, dst_offset, data)) in uploads.iter().enumerate() {
                let copy = vk::BufferCopy::default()
                    .src_offset(src_offsets[i])
                    .dst_offset(*dst_offset)
                    .size(data.len() as u64);
                device.cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[copy]);
            }
            device
                .end_command_buffer(cmd)
                .context("upload_data_at_offset_batch: end_command_buffer")?;
            device
                .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
                .context("upload_data_at_offset_batch: queue_submit")?;
            device
                .queue_wait_idle(queue)
                .context("upload_data_at_offset_batch: queue_wait_idle")?;
            device.free_command_buffers(pool, &command_buffers);
            device.destroy_command_pool(pool, None);
        }
        drop(staging);
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

    /// Device-to-device copy of `size` bytes from `src[src_byte_offset]`
    /// into `dst[dst_byte_offset]`. Both buffers live on `device`. Used by
    /// the Vulkan `slice_set` (paged-KV scatter / GDN state write) so a
    /// device-resident in-place write never bounces through the host.
    /// dtype-agnostic — operates on raw bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_buffer_region(
        device: &Arc<ash::Device>,
        queue: vk::Queue,
        queue_family_index: u32,
        src: &VulkanBuffer,
        src_byte_offset: u64,
        dst: &VulkanBuffer,
        dst_byte_offset: u64,
        size: u64,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            src_byte_offset + size <= src.size,
            "copy_buffer_region: src range {}+{} exceeds buffer {}",
            src_byte_offset,
            size,
            src.size
        );
        anyhow::ensure!(
            dst_byte_offset + size <= dst.size,
            "copy_buffer_region: dst range {}+{} exceeds buffer {}",
            dst_byte_offset,
            size,
            dst.size
        );
        let pool_info = make_pool_info(queue_family_index);
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("copy_buffer_region: create_command_pool")?
        };
        let alloc_info = make_alloc_info(pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("copy_buffer_region: allocate_command_buffer")?;
        let cmd = command_buffers[0];
        let copy = vk::BufferCopy::default()
            .src_offset(src_byte_offset)
            .dst_offset(dst_byte_offset)
            .size(size);
        unsafe {
            device
                .begin_command_buffer(cmd, &make_begin_info())
                .context("copy_buffer_region: begin")?;
            device.cmd_copy_buffer(cmd, src.buffer, dst.buffer, &[copy]);
            device
                .end_command_buffer(cmd)
                .context("copy_buffer_region: end")?;
            device
                .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
                .context("copy_buffer_region: submit")?;
            device
                .queue_wait_idle(queue)
                .context("copy_buffer_region: wait")?;
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
        Self::read_back_prefix(
            device,
            host_mem_type,
            queue,
            queue_family_index,
            src,
            src.size,
        )
    }

    /// Read exactly `byte_len` bytes from the start of a device buffer.
    /// This is useful for scalar reductions backed by a bucket-rounded pooled
    /// buffer: only the logical result crosses the device/host boundary.
    pub fn read_back_prefix(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        queue_family_index: u32,
        src: &VulkanBuffer,
        byte_len: u64,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(byte_len > 0, "read_back_prefix: byte_len must be > 0");
        anyhow::ensure!(
            byte_len <= src.size,
            "read_back_prefix: requested {byte_len} bytes from a {}-byte buffer",
            src.size
        );
        // Recycle the host-visible staging buffer instead of a fresh
        // `amdgpu_bo_alloc` per call. The pooled buffer may be
        // bucket-larger than `byte_len`; we copy and read exactly
        // `byte_len` bytes below. When this
        // `Arc` drops at function end the buffer returns to the pool.
        let staging = crate::buffer_pool::pool_alloc_host_visible(device, host_mem_type, byte_len)?;

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

        let copy = vk::BufferCopy::default().size(byte_len);
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
            std::slice::from_raw_parts(mapped_ptr as *const u8, byte_len as usize).to_vec()
        };

        unsafe {
            device.unmap_memory(staging.memory);
            device.free_command_buffers(pool, &command_buffers);
            device.destroy_command_pool(pool, None);
        }

        Ok(data)
    }

    /// Read multiple device buffers back through one staging buffer and one
    /// queue submission.
    pub fn read_back_batch(
        device: &Arc<ash::Device>,
        host_mem_type: u32,
        queue: vk::Queue,
        queue_family_index: u32,
        sources: &[&VulkanBuffer],
    ) -> Result<Vec<Vec<u8>>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }

        let mut dst_offsets = Vec::with_capacity(sources.len());
        let mut total_staging_bytes = 0u64;
        for (idx, src) in sources.iter().enumerate() {
            anyhow::ensure!(
                src.size > 0,
                "read_back_batch[{idx}]: source buffer must be non-empty"
            );
            dst_offsets.push(total_staging_bytes);
            total_staging_bytes = total_staging_bytes
                .checked_add(src.size)
                .ok_or_else(|| anyhow::anyhow!("read_back_batch: staging size overflow"))?;
        }
        let total_len = usize::try_from(total_staging_bytes)
            .context("read_back_batch: staging size exceeds usize")?;
        // Recycle the host-visible staging buffer (see `read_back`). The
        // pooled buffer may exceed `total_staging_bytes`; we copy into
        // `[0, total_staging_bytes)` and read only `total_len` bytes.
        let staging = crate::buffer_pool::pool_alloc_host_visible(
            device,
            host_mem_type,
            total_staging_bytes,
        )?;

        let pool_info = make_pool_info(queue_family_index);
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("read_back_batch: create_command_pool")?
        };
        let alloc_info = make_alloc_info(pool);
        let command_buffers =
            crate::vk_raw::allocate_command_buffers(device.handle(), &alloc_info, 1)
                .context("read_back_batch: allocate_command_buffers")?;
        let cmd = command_buffers[0];

        unsafe {
            device
                .begin_command_buffer(cmd, &make_begin_info())
                .context("read_back_batch: begin_command_buffer")?;
            for (i, src) in sources.iter().enumerate() {
                let copy = vk::BufferCopy::default()
                    .dst_offset(dst_offsets[i])
                    .size(src.size);
                device.cmd_copy_buffer(cmd, src.buffer, staging.buffer, &[copy]);
            }
            device
                .end_command_buffer(cmd)
                .context("read_back_batch: end_command_buffer")?;
            device
                .queue_submit(queue, &[make_submit_info(&[cmd])], vk::Fence::null())
                .context("read_back_batch: queue_submit")?;
            device
                .queue_wait_idle(queue)
                .context("read_back_batch: queue_wait_idle")?;
        }

        let mapped_ptr = unsafe {
            device
                .map_memory(
                    staging.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| anyhow::anyhow!("read_back_batch: map_memory failed: {:?}", e))?
        };
        let mapped = unsafe { std::slice::from_raw_parts(mapped_ptr as *const u8, total_len) };
        let mut out = Vec::with_capacity(sources.len());
        for (idx, src) in sources.iter().enumerate() {
            let start = usize::try_from(dst_offsets[idx])
                .context("read_back_batch: offset exceeds usize")?;
            let len =
                usize::try_from(src.size).context("read_back_batch: source size exceeds usize")?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| anyhow::anyhow!("read_back_batch[{idx}]: slice overflow"))?;
            out.push(mapped[start..end].to_vec());
        }

        unsafe {
            device.unmap_memory(staging.memory);
            device.free_command_buffers(pool, &command_buffers);
            device.destroy_command_pool(pool, None);
        }

        Ok(out)
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

    /// Create a host-visible buffer and fill it with disjoint byte slices in
    /// order, using one map/unmap pair. Returns each slice's byte offset.
    pub fn create_host_visible_with_segments(
        device: &Arc<ash::Device>,
        mem_type_index: u32,
        segments: &[&[u8]],
    ) -> Result<(Self, Vec<u64>)> {
        anyhow::ensure!(
            !segments.is_empty(),
            "create_host_visible_with_segments: at least one segment is required"
        );

        let mut offsets = Vec::with_capacity(segments.len());
        let mut total = 0u64;
        for (idx, segment) in segments.iter().enumerate() {
            anyhow::ensure!(
                !segment.is_empty(),
                "create_host_visible_with_segments[{idx}]: segment must be non-empty"
            );
            offsets.push(total);
            total = total.checked_add(segment.len() as u64).ok_or_else(|| {
                anyhow::anyhow!("create_host_visible_with_segments: size overflow")
            })?;
        }

        let buffer = VulkanBuffer::create_host_visible(device, mem_type_index, total)
            .context("create_host_visible_with_segments: create host-visible buffer")?;
        let mapped_ptr = unsafe {
            device
                .map_memory(
                    buffer.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "create_host_visible_with_segments: map_memory failed: {:?}",
                        e
                    )
                })?
        };
        unsafe {
            for (offset, segment) in offsets.iter().zip(segments.iter()) {
                std::ptr::copy_nonoverlapping(
                    segment.as_ptr(),
                    (mapped_ptr as *mut u8).add(*offset as usize),
                    segment.len(),
                );
            }
            device.unmap_memory(buffer.memory);
        }

        Ok((buffer, offsets))
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
        let bytes = unsafe { std::slice::from_raw_parts(mapped as *const u8, bytes_len).to_vec() };
        unsafe {
            self.device.unmap_memory(self.memory);
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn read_back_batch_matches_individual_reads() -> Result<()> {
        let _guard = TEST_LOCK.lock().unwrap();
        let Ok(vk_device) = crate::VulkanDevice::new() else {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);
        let first = (0u8..16).collect::<Vec<_>>();
        let second = (32u8..56).collect::<Vec<_>>();

        let first_buffer = VulkanBuffer::create_device_local(
            vk_device.device(),
            vk_device.device_local_mem_type(),
            first.len() as u64,
        )?;
        let second_buffer = VulkanBuffer::create_device_local(
            vk_device.device(),
            vk_device.device_local_mem_type(),
            second.len() as u64,
        )?;
        VulkanBuffer::upload_data(
            vk_device.device(),
            vk_device.host_visible_mem_type(),
            vk_device.queue(),
            vk_device.queue_family_index(),
            &first_buffer,
            &first,
        )?;
        VulkanBuffer::upload_data(
            vk_device.device(),
            vk_device.host_visible_mem_type(),
            vk_device.queue(),
            vk_device.queue_family_index(),
            &second_buffer,
            &second,
        )?;

        let batched = VulkanBuffer::read_back_batch(
            vk_device.device(),
            vk_device.host_visible_mem_type(),
            vk_device.queue(),
            vk_device.queue_family_index(),
            &[&first_buffer, &second_buffer],
        )?;

        assert_eq!(batched.len(), 2);
        assert_eq!(batched[0], first);
        assert_eq!(batched[1], second);
        let empty: Vec<&VulkanBuffer> = Vec::new();
        assert!(
            VulkanBuffer::read_back_batch(
                vk_device.device(),
                vk_device.host_visible_mem_type(),
                vk_device.queue(),
                vk_device.queue_family_index(),
                &empty,
            )?
            .is_empty()
        );
        Ok(())
    }

    #[test]
    fn create_host_visible_with_segments_packs_offsets() -> Result<()> {
        let _guard = TEST_LOCK.lock().unwrap();
        let Ok(vk_device) = crate::VulkanDevice::new() else {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        };
        let vk_device = Arc::new(vk_device);
        let first = [1u8, 2, 3, 4];
        let second = [9u8, 8, 7];
        let third = [42u8, 43];
        let (buffer, offsets) = VulkanBuffer::create_host_visible_with_segments(
            vk_device.device(),
            vk_device.host_visible_mem_type(),
            &[&first, &second, &third],
        )?;

        assert_eq!(
            offsets,
            vec![0, first.len() as u64, (first.len() + second.len()) as u64]
        );
        assert_eq!(
            buffer.size(),
            (first.len() + second.len() + third.len()) as u64
        );
        let bytes = VulkanBuffer::read_host_visible(vk_device.device(), &buffer)?;
        assert_eq!(
            bytes,
            [first.as_slice(), second.as_slice(), third.as_slice()].concat()
        );
        Ok(())
    }

    #[test]
    fn allocation_stats_track_live_and_cumulative_bytes_by_memory_kind() -> Result<()> {
        let _guard = TEST_LOCK.lock().unwrap();
        let Ok(vk_device) = crate::VulkanDevice::new() else {
            eprintln!("Vulkan device unavailable, skipping");
            return Ok(());
        };
        let before = allocation_stats();
        let device_local = VulkanBuffer::create_device_local(
            vk_device.device(),
            vk_device.device_local_mem_type(),
            17,
        )?;
        let host_visible = VulkanBuffer::create_host_visible(
            vk_device.device(),
            vk_device.host_visible_mem_type(),
            19,
        )?;
        let device_local_allocation_size = device_local.allocation_size;
        let host_visible_allocation_size = host_visible.allocation_size;
        let live = allocation_stats();
        assert!(live.live_device_local_buffers >= before.live_device_local_buffers + 1);
        assert!(live.live_host_visible_buffers >= before.live_host_visible_buffers + 1);
        assert!(live.live_device_local_bytes >= before.live_device_local_bytes + 17);
        assert!(live.live_host_visible_bytes >= before.live_host_visible_bytes + 19);
        assert!(live.device_local_allocations >= before.device_local_allocations + 1);
        assert!(live.host_visible_allocations >= before.host_visible_allocations + 1);
        assert!(
            live.device_local_allocated_bytes
                >= before.device_local_allocated_bytes + device_local_allocation_size
        );
        assert!(
            live.host_visible_allocated_bytes
                >= before.host_visible_allocated_bytes + host_visible_allocation_size
        );

        drop(device_local);
        drop(host_visible);
        let after = allocation_stats();
        assert!(after.device_local_frees >= live.device_local_frees + 1);
        assert!(after.host_visible_frees >= live.host_visible_frees + 1);
        assert!(
            after.device_local_freed_bytes
                >= live.device_local_freed_bytes + device_local_allocation_size
        );
        assert!(
            after.host_visible_freed_bytes
                >= live.host_visible_freed_bytes + host_visible_allocation_size
        );
        Ok(())
    }
}

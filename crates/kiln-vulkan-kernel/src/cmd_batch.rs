//! Per-step Vulkan command-buffer batching for the resident decode
//! path.
//!
//! Gate (e.1) of `docs/vk_resident_decode_plan.md`: the resident
//! decode step's submit count needs to drop from
//! `O(layers × kernels_per_layer)` to `O(1)`. Each
//! `run_compute_pipeline` call currently begins, ends, submits, and
//! waits its own command buffer — 11+ submits per layer × 32 layers
//! ≈ 350 submits per decode token. At ~50 µs of queue overhead per
//! submit on NVIDIA Vulkan, that's ~17 ms of pure submission
//! overhead per token.
//!
//! `CommandBatch` collapses this to one submit: allocate one command
//! buffer + one descriptor pool sized for the whole step, record
//! every dispatch into it with a compute→compute memory barrier
//! between, and submit once at the end. The compute-shader reads
//! that the existing primitive emits at the end (a SHADER_WRITE →
//! TRANSFER_READ barrier sized to a single dispatch) become
//! SHADER_WRITE → SHADER_READ barriers between back-to-back
//! dispatches; the final dispatch keeps the SHADER_WRITE →
//! TRANSFER_READ barrier so the resident path's readback at the end
//! of the step is well-defined.

use crate::pipeline::ShaderPipeline;
use crate::{VulkanBuffer, VulkanDevice};
use anyhow::{Context, Result};
use ash::vk;
use std::sync::MutexGuard;

/// 1D or 3D workgroup count selector. `OneD(n)` dispatches `(n, 1, 1)`.
#[derive(Debug, Clone, Copy)]
pub enum Workgroups {
    OneD(u32),
    ThreeD(u32, u32, u32),
}

impl Workgroups {
    pub fn as_3d(&self) -> (u32, u32, u32) {
        match self {
            Workgroups::OneD(x) => (*x, 1, 1),
            Workgroups::ThreeD(x, y, z) => (*x, *y, *z),
        }
    }
}

/// One dispatch's full plan: SPIR-V bytes, buffer handles, push
/// constants, workgroup count. Building a `KernelPlan` is free; you
/// then either feed it into a single-submit
/// `vk_device.submit_plan(plan)` (existing per-call path) or append
/// it to a `CommandBatch` for batched submission.
pub struct KernelPlan<'a> {
    pub spirv: &'a [u8],
    pub handles: &'a [vk::Buffer],
    pub push_constants: &'a [u32],
    pub workgroups: Workgroups,
}

/// Recording-state batch of compute dispatches. Built once at the
/// start of a decode step; every layer's `record(...)` appends one
/// dispatch (with the right compute→compute barrier separating it
/// from the previous); `submit_and_wait()` flushes the whole step in
/// one `vkQueueSubmit`.
pub struct CommandBatch<'a> {
    vk_device: &'a VulkanDevice,
    /// Live lock on the device's long-lived batch command pool. Held
    /// for the lifetime of this batch so a single batch has exclusive
    /// use of the pool's command buffers; released on drop.
    cmd_pool_guard: MutexGuard<'a, vk::CommandPool>,
    /// Live lock on the device's long-lived batch descriptor pool.
    /// Same semantics as `cmd_pool_guard`.
    descriptor_pool_guard: MutexGuard<'a, vk::DescriptorPool>,
    cmd: vk::CommandBuffer,
    cmd_buffers_to_free: Vec<vk::CommandBuffer>,
    dispatch_count: usize,
    finished: bool,
}

const COMMAND_BATCH_MAX_DISPATCHES: u32 = 1024;
const COMMAND_BATCH_MAX_BINDINGS: u32 = 64 * COMMAND_BATCH_MAX_DISPATCHES;

impl<'a> CommandBatch<'a> {
    /// Allocate a fresh command pool + descriptor pool sized for one
    /// decode step (up to `COMMAND_BATCH_MAX_DISPATCHES` dispatches,
    /// each with up to 64 storage-buffer bindings — comfortably
    /// covering Qwen3.5-4B's 32 layers × ~11 kernels = 352 dispatches).
    pub fn new(vk_device: &'a VulkanDevice) -> Result<Self> {
        let device = vk_device.device();
        // Lock the device's long-lived batch pools. The guards are held
        // for the lifetime of this `CommandBatch` so concurrent batches
        // serialize on the pool — but the pools themselves are reused,
        // avoiding the ~5 ms-per-pool create+destroy that dominated
        // per-layer batch construction.
        let cmd_pool_guard = vk_device.batch_command_pool()?;
        let descriptor_pool_guard = vk_device.batch_descriptor_pool()?;
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(*cmd_pool_guard)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1)
            .build();
        let cmd_buffers = unsafe { device.allocate_command_buffers(&alloc_info) }
            .context("CommandBatch: allocate command buffer")?;
        let cmd = cmd_buffers[0];
        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                        .build(),
                )
                .context("CommandBatch: begin command buffer")?;
        }
        Ok(Self {
            vk_device,
            cmd_pool_guard,
            descriptor_pool_guard,
            cmd,
            cmd_buffers_to_free: cmd_buffers,
            dispatch_count: 0,
            finished: false,
        })
    }

    /// Append a single dispatch to the batch. Allocates a fresh
    /// descriptor set out of the batch's pool, writes the handle bindings,
    /// emits push constants + cmd_dispatch, and emits a
    /// SHADER_WRITE → SHADER_READ memory barrier so the NEXT dispatch
    /// (if any) sees the writes. The final-dispatch SHADER_WRITE →
    /// TRANSFER_READ barrier is emitted once by `submit_and_wait`.
    pub fn record(&mut self, plan: &KernelPlan<'_>) -> Result<()> {
        anyhow::ensure!(!self.finished, "CommandBatch: already submitted");
        anyhow::ensure!(
            (self.dispatch_count as u32) < COMMAND_BATCH_MAX_DISPATCHES,
            "CommandBatch: exceeded {COMMAND_BATCH_MAX_DISPATCHES} dispatches per batch"
        );
        anyhow::ensure!(
            plan.handles.len() <= 64,
            "CommandBatch: dispatch {} has {} handles > 64 binding limit",
            self.dispatch_count,
            plan.handles.len(),
        );
        let (set_layout, layout, pipeline) = self
            .vk_device
            .get_or_create_compute_pipeline(
                plan.spirv,
                plan.handles.len(),
                (plan.push_constants.len() * 4) as u32,
            )
            .context("CommandBatch: get/create pipeline")?;
        self.record_with_pipeline(
            set_layout,
            layout,
            pipeline,
            plan.handles,
            plan.push_constants,
            plan.workgroups,
        )
    }

    /// Inner recording path that takes the cached pipeline objects
    /// directly. Used by `record_shader` (path-keyed cache) and
    /// `record` (SPIR-V-keyed cache). Allocates a descriptor set,
    /// writes the buffer bindings, emits the compute→compute barrier
    /// (skipped on first dispatch), and dispatches.
    fn record_with_pipeline(
        &mut self,
        set_layout: vk::DescriptorSetLayout,
        layout: vk::PipelineLayout,
        pipeline: vk::Pipeline,
        handles: &[vk::Buffer],
        push_constants: &[u32],
        workgroups: Workgroups,
    ) -> Result<()> {
        anyhow::ensure!(!self.finished, "CommandBatch: already submitted");
        anyhow::ensure!(
            (self.dispatch_count as u32) < COMMAND_BATCH_MAX_DISPATCHES,
            "CommandBatch: exceeded {COMMAND_BATCH_MAX_DISPATCHES} dispatches per batch"
        );
        anyhow::ensure!(
            handles.len() <= 64,
            "CommandBatch: dispatch {} has {} handles > 64 binding limit",
            self.dispatch_count,
            handles.len(),
        );
        let device = self.vk_device.device();
        // Descriptor-set cache fast-path: keyed by `(layout, &handles)`,
        // looked up on the device. After first call per shape the set
        // is just a lookup; no `allocate_descriptor_sets` /
        // `update_descriptor_sets` round-trip per dispatch.
        let descriptor_set = self.vk_device.get_or_alloc_descriptor_set(
            set_layout,
            *self.descriptor_pool_guard,
            handles,
        )?;
        unsafe {

            // Inter-dispatch barrier — needed BEFORE this dispatch so it
            // sees the previous dispatch's writes (skip on first call).
            if self.dispatch_count > 0 {
                let barrier = vk::MemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .build();
                device.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }

            device.cmd_bind_pipeline(self.cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                self.cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                self.cmd,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::cast_slice(push_constants),
            );
            let (wx, wy, wz) = workgroups.as_3d();
            device.cmd_dispatch(self.cmd, wx, wy, wz);
        }
        self.dispatch_count += 1;
        Ok(())
    }

    /// Convenience: build a `KernelPlan` from a shader path and record it
    /// in one step. Compiles `glsl_path` to SPIR-V via the shared cache;
    /// the lifetime of the SPIR-V is bound to `self` to avoid feeding
    /// the recorder a freed buffer.
    pub fn record_shader(
        &mut self,
        glsl_path: &'static str,
        handles: &[vk::Buffer],
        push_constants: &[u32],
        workgroups: Workgroups,
    ) -> Result<()> {
        // Path-keyed pipeline cache fast-path: avoids re-hashing the
        // SPIR-V on every call (~450 calls per decode token).
        let (set_layout, layout, pipeline) = self.vk_device.get_compute_pipeline_by_path(
            glsl_path,
            handles.len(),
            (push_constants.len() * 4) as u32,
        )?;
        self.record_with_pipeline(set_layout, layout, pipeline, handles, push_constants, workgroups)
    }

    /// Record a `cmd_copy_buffer` from `src` (typically a device-local
    /// pool buffer like logits or final hidden) into `dst` (a
    /// host-visible staging buffer) inside this batch. After the
    /// batch's `submit_and_wait` returns, the caller can `map_memory`
    /// on `dst` and read the data directly — no separate readback
    /// queue submission needed.
    ///
    /// Emits a compute→transfer memory barrier first so the last
    /// dispatch's writes are visible to the copy.
    pub fn record_copy_buffer(
        &mut self,
        src: &VulkanBuffer,
        dst: &VulkanBuffer,
        size: u64,
    ) -> Result<()> {
        anyhow::ensure!(!self.finished, "CommandBatch: already submitted");
        let device = self.vk_device.device();
        unsafe {
            // SHADER_WRITE → TRANSFER_READ barrier so cmd_copy_buffer
            // sees the previous compute dispatch's writes. The
            // submit_and_wait tail barrier (added below) handles the
            // TRANSFER_WRITE → HOST_READ side for the host map.
            let barrier = vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .build();
            device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            let copy = vk::BufferCopy::builder().size(size).build();
            device.cmd_copy_buffer(self.cmd, src.handle(), dst.handle(), &[copy]);
            // TRANSFER_WRITE → HOST_READ so the post-submit map sees
            // the copy's writes.
            let host_barrier = vk::MemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .build();
            device.cmd_pipeline_barrier(
                self.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[host_barrier],
                &[],
                &[],
            );
        }
        Ok(())
    }

    /// Number of dispatches currently in the batch (excluding the final
    /// transfer barrier).
    pub fn dispatch_count(&self) -> usize {
        self.dispatch_count
    }

    /// Finalize, submit, wait for completion, free resources.
    pub fn submit_and_wait(mut self, label: &str) -> Result<()> {
        let device = self.vk_device.device();
        unsafe {
            // Final tail barrier: SHADER_WRITE → TRANSFER_READ + HOST_READ
            // so the next readback / copy off the batch's last output
            // buffer sees the writes.
            if self.dispatch_count > 0 {
                let barrier = vk::MemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::HOST_READ)
                    .build();
                device.cmd_pipeline_barrier(
                    self.cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::TRANSFER | vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
            device
                .end_command_buffer(self.cmd)
                .with_context(|| format!("CommandBatch[{label}]: end command buffer"))?;
        }
        self.vk_device.submit_and_wait(self.cmd, label)?;
        self.finished = true;
        // Drop will free resources.
        Ok(())
    }

    /// Tail buffer marker for callers that want to assert the chain
    /// terminates at a particular slot. Not used by the recorder itself.
    #[allow(dead_code)]
    pub fn assert_last_handle(&self, expected: &VulkanBuffer) {
        let _ = expected;
    }
}

impl<'a> Drop for CommandBatch<'a> {
    fn drop(&mut self) {
        let device = self.vk_device.device();
        unsafe {
            // Pools persist on `VulkanDevice`. Free this batch's
            // command buffer back to the (transient) command pool, but
            // do NOT reset the descriptor pool — the descriptor-set
            // cache in `VulkanDevice` holds long-lived references to
            // every `(layout, handles)` combination this batch
            // allocated. After the first decode token the cache is
            // warm and subsequent batches reuse the same descriptor
            // sets without any allocate/update Vulkan API calls.
            if !self.cmd_buffers_to_free.is_empty() {
                device.free_command_buffers(*self.cmd_pool_guard, &self.cmd_buffers_to_free);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::upload_tensor_f32_buffer;
    use candle_core::{Device, Tensor};

    #[test]
    fn command_batch_chains_two_adds() {
        let Ok(dev) = VulkanDevice::new() else {
            return;
        };
        let n = 256usize;
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.02).collect();
        let c: Vec<f32> = (0..n).map(|i| (i as f32) * 0.03).collect();
        let a_t = Tensor::from_vec(a.clone(), n, &Device::Cpu).unwrap();
        let b_t = Tensor::from_vec(b.clone(), n, &Device::Cpu).unwrap();
        let c_t = Tensor::from_vec(c.clone(), n, &Device::Cpu).unwrap();
        let a_buf = upload_tensor_f32_buffer(&dev, &a_t).unwrap();
        let b_buf = upload_tensor_f32_buffer(&dev, &b_t).unwrap();
        let c_buf = upload_tensor_f32_buffer(&dev, &c_t).unwrap();
        let tmp = VulkanBuffer::create_device_local(
            dev.device(),
            dev.device_local_mem_type(),
            (n * 4) as u64,
        )
        .unwrap();
        let out = VulkanBuffer::create_device_local(
            dev.device(),
            dev.device_local_mem_type(),
            (n * 4) as u64,
        )
        .unwrap();
        let glsl_path = concat!(env!("CARGO_MANIFEST_DIR"), "/csrc/shaders/add.comp");
        let push: [u32; 1] = [n as u32];
        let wg = Workgroups::OneD(n.div_ceil(256) as u32);
        {
            let mut batch = CommandBatch::new(&dev).unwrap();
            batch
                .record_shader(
                    glsl_path,
                    &[a_buf.handle(), b_buf.handle(), tmp.handle()],
                    &push,
                    wg,
                )
                .unwrap();
            batch
                .record_shader(
                    glsl_path,
                    &[tmp.handle(), c_buf.handle(), out.handle()],
                    &push,
                    wg,
                )
                .unwrap();
            assert_eq!(batch.dispatch_count(), 2);
            batch.submit_and_wait("test_chain").unwrap();
        }
        let got_bytes = VulkanBuffer::read_back(
            dev.device(),
            dev.host_visible_mem_type(),
            dev.queue(),
            dev.queue_family_index(),
            &out,
        )
        .unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&got_bytes).to_vec();
        for i in 0..n {
            let expected = a[i] + b[i] + c[i];
            assert!(
                (expected - got[i]).abs() <= 1e-6,
                "idx {i}: {expected} vs {}",
                got[i]
            );
        }
    }
}

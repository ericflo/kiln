//! Thin objc2 wrappers over `MTLIndirectCommandBuffer`.
//!
//! This is the Metal replay primitive used by `kiln-graph-metal` and the
//! future `MetalGraphRunner`: commands are encoded once into an ICB, then a
//! normal compute encoder executes the ICB range on each replay.

use super::{Buffer, ComputePipeline};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLIndirectCommandBuffer as RawMTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor,
    MTLIndirectCommandType, MTLIndirectComputeCommand as RawMTLIndirectComputeCommand, MTLSize,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndirectDispatchKind {
    Threadgroups,
    Threads,
    ThreadgroupsAndThreads,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IndirectCommandBufferDescriptor {
    pub max_kernel_buffer_bind_count: usize,
    pub max_kernel_threadgroup_memory_bind_count: usize,
    pub dispatch_kind: IndirectDispatchKind,
}

impl Default for IndirectCommandBufferDescriptor {
    fn default() -> Self {
        Self {
            max_kernel_buffer_bind_count: 31,
            max_kernel_threadgroup_memory_bind_count: 31,
            dispatch_kind: IndirectDispatchKind::Threadgroups,
        }
    }
}

impl IndirectCommandBufferDescriptor {
    pub(crate) fn to_raw(self) -> Retained<MTLIndirectCommandBufferDescriptor> {
        let raw = MTLIndirectCommandBufferDescriptor::new();
        let command_type = match self.dispatch_kind {
            IndirectDispatchKind::Threadgroups => MTLIndirectCommandType::ConcurrentDispatch,
            IndirectDispatchKind::Threads => MTLIndirectCommandType::ConcurrentDispatchThreads,
            IndirectDispatchKind::ThreadgroupsAndThreads => {
                MTLIndirectCommandType::ConcurrentDispatch
                    | MTLIndirectCommandType::ConcurrentDispatchThreads
            }
        };
        raw.setCommandTypes(command_type);
        raw.setInheritPipelineState(false);
        raw.setInheritBuffers(false);
        raw.setMaxKernelBufferBindCount(self.max_kernel_buffer_bind_count);
        unsafe {
            raw.setMaxKernelThreadgroupMemoryBindCount(
                self.max_kernel_threadgroup_memory_bind_count,
            );
        }
        raw
    }
}

#[derive(Clone, Debug)]
pub struct IndirectCommandBuffer {
    raw: Retained<ProtocolObject<dyn RawMTLIndirectCommandBuffer>>,
    max_command_count: usize,
}

unsafe impl Send for IndirectCommandBuffer {}
unsafe impl Sync for IndirectCommandBuffer {}

impl IndirectCommandBuffer {
    pub(crate) fn new(
        raw: Retained<ProtocolObject<dyn RawMTLIndirectCommandBuffer>>,
        max_command_count: usize,
    ) -> Self {
        Self {
            raw,
            max_command_count,
        }
    }

    pub fn max_command_count(&self) -> usize {
        self.max_command_count
    }

    pub fn size(&self) -> usize {
        self.raw.size()
    }

    pub fn reset(&self, start: usize, count: usize) {
        assert!(
            start <= self.max_command_count && count <= self.max_command_count - start,
            "indirect command reset range out of bounds"
        );
        unsafe {
            self.raw.resetWithRange(NSRange {
                location: start,
                length: count,
            });
        }
    }

    pub fn compute_command_at(&self, index: usize) -> IndirectComputeCommand {
        assert!(
            index < self.max_command_count,
            "indirect compute command index out of bounds"
        );
        let raw = unsafe { self.raw.indirectComputeCommandAtIndex(index) };
        IndirectComputeCommand { raw }
    }
}

impl AsRef<ProtocolObject<dyn RawMTLIndirectCommandBuffer>> for IndirectCommandBuffer {
    fn as_ref(&self) -> &ProtocolObject<dyn RawMTLIndirectCommandBuffer> {
        &self.raw
    }
}

#[derive(Debug)]
pub struct IndirectComputeCommand {
    raw: Retained<ProtocolObject<dyn RawMTLIndirectComputeCommand>>,
}

impl IndirectComputeCommand {
    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipeline) {
        self.raw.setComputePipelineState(pipeline.as_ref());
    }

    pub fn set_kernel_buffer(&self, index: usize, buffer: &Buffer, offset: usize) {
        unsafe {
            self.raw
                .setKernelBuffer_offset_atIndex(buffer.as_ref(), offset, index);
        }
    }

    pub fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        unsafe {
            self.raw.setThreadgroupMemoryLength_atIndex(length, index);
        }
    }

    pub fn dispatch_threadgroups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    ) {
        self.raw
            .concurrentDispatchThreadgroups_threadsPerThreadgroup(
                threadgroups_per_grid,
                threads_per_threadgroup,
            );
    }

    pub fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize) {
        self.raw.concurrentDispatchThreads_threadsPerThreadgroup(
            threads_per_grid,
            threads_per_threadgroup,
        );
    }

    pub fn set_barrier(&self) {
        self.raw.setBarrier();
    }

    pub fn clear_barrier(&self) {
        self.raw.clearBarrier();
    }

    pub fn reset(&self) {
        self.raw.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_rt::{Device, MTLResourceOptions};
    use crate::metal_types::MetalCompanion;

    #[test]
    fn records_and_replays_simple_compute_command() {
        let Some(device) = Device::system_default() else {
            eprintln!("skipping Metal ICB test: no system Metal device");
            return;
        };
        let companion = MetalCompanion::from_raw(device.clone()).unwrap();
        let source = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void kiln_icb_test_write(
                device uint* out [[buffer(0)]],
                constant uint& value [[buffer(1)]],
                uint gid [[thread_position_in_grid]]
            ) {
                if (gid == 0) {
                    out[0] = value + 1;
                }
            }
        "#;
        let library = device.new_library_with_source(source, None).unwrap();
        let function = library.get_function("kiln_icb_test_write", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function_for_indirect_commands(&function)
            .unwrap();
        assert!(pipeline.supports_indirect_command_buffers());
        let out = device
            .new_buffer(4, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let value = 40u32;
        let scalar = device
            .new_buffer_with_data(
                &value as *const u32 as *const std::ffi::c_void,
                std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
            .unwrap();
        let descriptor = IndirectCommandBufferDescriptor {
            dispatch_kind: IndirectDispatchKind::Threadgroups,
            ..Default::default()
        };
        let Ok(icb) = device.new_indirect_command_buffer(
            descriptor,
            1,
            MTLResourceOptions::StorageModePrivate,
        ) else {
            eprintln!("skipping Metal ICB test: device did not create an ICB");
            return;
        };
        icb.reset(0, 1);
        let command = icb.compute_command_at(0);
        command.set_compute_pipeline_state(&pipeline);
        command.set_kernel_buffer(0, &out, 0);
        command.set_kernel_buffer(1, &scalar, 0);
        command.dispatch_threadgroups(
            objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );

        let encoder = companion.command_encoder().unwrap();
        encoder.use_resource(&scalar, objc2_metal::MTLResourceUsage::Read);
        encoder.use_resource(&out, objc2_metal::MTLResourceUsage::Write);
        encoder.execute_commands_in_buffer(&icb, 0, 1);
        drop(encoder);
        companion.wait_until_completed().unwrap();

        let got = unsafe { *(out.contents() as *const u32) };
        assert_eq!(got, 41);
    }
}

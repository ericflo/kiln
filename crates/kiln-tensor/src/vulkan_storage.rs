//! Vulkan storage impl behind the `vulkan` feature flag.
//!
//! Lifts `kiln_vulkan_kernel::VulkanBuffer` + `VulkanDevice`. Unlike
//! [`CudaStorage`](crate::CudaStorage) and
//! [`MetalStorage`](crate::MetalStorage), this storage carries **no
//! transitive candle dependency** — `kiln-vulkan-kernel` is already
//! candle-free (uses `ash` + custom GLSL compute) so Phase 7 of #1082
//! does not need to replace anything here.
//!
//! # Anti-pattern 1 compliance
//!
//! `VulkanStorage` holds a `VulkanBuffer` directly. The buffer wraps
//! `vk::Buffer + vk::DeviceMemory + Arc<ash::Device>` and is owned by
//! us. No candle types appear in the type signature.
//!
//! # Phase 1.8 scope
//!
//! Storage-layer only:
//!
//! - `zeros(device, dtype, n_elements)` — device-local buffer alloc.
//! - `from_buffer(device, dtype, buffer)` — wrap an existing
//!   `VulkanBuffer` allocated by `kiln-vulkan-kernel`.
//! - `StorageBackend` impl returning `Device::Vulkan(idx)`, dtype,
//!   byte len.
//! - `buffer()` / `vulkan_device()` accessors for FFI sites in
//!   `kiln-vulkan-kernel`'s kernel-dispatch layer.
//!
//! H2D/D2H staging, the autograd tape integration, and `vk_tensor.rs`'s
//! lift into `kiln-tensor` proper are subsequent PRs.

use std::any::Any;
use std::sync::Arc;

use kiln_vulkan_kernel::buffer::VulkanBuffer;
use kiln_vulkan_kernel::device::VulkanDevice;

use crate::{DType, Device, Error, Result, StorageBackend};

/// Vulkan-backed storage. Byte-typed; dtype carried alongside for
/// dispatch.
///
/// `VulkanBuffer` is **not** `Clone` — Drop frees the underlying
/// `vk::DeviceMemory`. The expected handle is `Arc<dyn StorageBackend>`
/// from the [`crate::Storage`] alias.
#[derive(Debug)]
pub struct VulkanStorage {
    device: Device,
    dtype: DType,
    buffer: VulkanBuffer,
    /// Cached byte length. We re-record this on construction because
    /// `VulkanBuffer`'s `size: u64` field is private upstream; we know
    /// the value at allocate time and store it to avoid an upstream
    /// API change (which would require touching kiln-vulkan-kernel
    /// outside this PR's scope).
    byte_len: usize,
    vulkan_device: Arc<VulkanDevice>,
}

impl VulkanStorage {
    /// Allocate `n_elements` worth of bytes for `dtype` on
    /// `vulkan_device`. Uses
    /// `VulkanBuffer::create_device_local(device.device(),
    /// device.device_local_mem_type(), byte_len)`.
    ///
    /// `device_index` is the Vulkan physical-device index — stored as
    /// the [`Device::Vulkan`] variant. The `vulkan_device` argument's
    /// index is the source of truth; we record it explicitly so the
    /// `device()` accessor is O(1).
    pub fn zeros(
        vulkan_device: Arc<VulkanDevice>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        let buffer = VulkanBuffer::create_device_local(
            vulkan_device.device(),
            vulkan_device.device_local_mem_type(),
            byte_len as u64,
        )
        .map_err(|e| {
            Error::Msg(format!(
                "VulkanStorage::zeros: create_device_local({byte_len}) failed: {e}"
            ))
        })?;
        Ok(VulkanStorage {
            device: Device::Vulkan(device_index),
            dtype,
            buffer,
            byte_len,
            vulkan_device,
        })
    }

    /// Wrap an existing `VulkanBuffer` allocated by the caller.
    ///
    /// Validates the buffer length against `dtype.size_in_bytes()`
    /// for non-packed dtypes.
    pub fn from_buffer(
        vulkan_device: Arc<VulkanDevice>,
        device_index: usize,
        dtype: DType,
        buffer: VulkanBuffer,
        size_bytes: u64,
    ) -> Result<Self> {
        let len = size_bytes as usize;
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !len.is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "VulkanStorage::from_buffer: buffer len {len} is not a multiple of \
                     size_in_bytes({:?}) = {per}",
                    dtype
                )));
            }
        }
        Ok(VulkanStorage {
            device: Device::Vulkan(device_index),
            dtype,
            buffer,
            byte_len: len,
            vulkan_device,
        })
    }

    /// Borrow the underlying VulkanBuffer.
    pub fn buffer(&self) -> &VulkanBuffer {
        &self.buffer
    }

    /// Borrow the VulkanDevice — the queue/dispatch handle the existing
    /// `kiln-vulkan-kernel::kernels::*` entry points consume.
    pub fn vulkan_device(&self) -> &Arc<VulkanDevice> {
        &self.vulkan_device
    }
}

impl StorageBackend for VulkanStorage {
    fn device(&self) -> Device {
        self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn byte_len(&self) -> usize {
        self.byte_len
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Construct a fresh [`crate::Storage`] handle holding a [`VulkanStorage`].
pub fn vulkan_zeros(
    vulkan_device: Arc<VulkanDevice>,
    device_index: usize,
    dtype: DType,
    n_elements: usize,
) -> Result<crate::Storage> {
    let storage = VulkanStorage::zeros(vulkan_device, device_index, dtype, n_elements)?;
    Ok(Arc::new(storage))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vulkan_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_VULKAN_TEST").ok().as_deref() == Some("1")
    }

    fn maybe_vulkan_device() -> Option<Arc<VulkanDevice>> {
        if !vulkan_test_enabled() {
            return None;
        }
        VulkanDevice::new().ok().map(Arc::new)
    }

    #[test]
    fn zeros_constructs() {
        let Some(dev) = maybe_vulkan_device() else {
            eprintln!("skip: KILN_TENSOR_VULKAN_TEST unset or no Vulkan device");
            return;
        };
        let storage = VulkanStorage::zeros(dev, 0, DType::BF16, 64).unwrap();
        assert_eq!(storage.device(), Device::Vulkan(0));
        assert_eq!(storage.dtype(), DType::BF16);
    }
}

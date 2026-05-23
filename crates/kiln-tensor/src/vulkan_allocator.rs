//! `VulkanAllocator` — the Vulkan [`Allocator`] impl.
//!
//! Mirrors [`crate::CudaAllocator`] and [`crate::MetalAllocator`]:
//! Owned/Pool/Frozen modes, `warm()`, `reserved_bytes` /
//! `peak_reserved_bytes`. Backed by [`VulkanStorage::zeros`].
//!
//! **No candle dependency** — `kiln-vulkan-kernel` is candle-free
//! today (`ash` + custom GLSL compute), so the Vulkan allocator
//! requires no Phase 7 lift.

use std::collections::HashMap;
use std::sync::Arc;

use kiln_vulkan_kernel::device::VulkanDevice;

use crate::{
    allocator_frozen_error, Allocator, AllocatorMode, DType, Device, Result, Storage, VulkanStorage,
};

#[derive(Debug)]
pub struct VulkanAllocator {
    vulkan_device: Arc<VulkanDevice>,
    device_index: usize,
    mode: AllocatorMode,
    cache: HashMap<(DType, usize), Vec<Storage>>,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
}

impl VulkanAllocator {
    pub fn new(vulkan_device: Arc<VulkanDevice>, device_index: usize) -> Self {
        VulkanAllocator {
            vulkan_device,
            device_index,
            mode: AllocatorMode::Owned,
            cache: HashMap::new(),
            reserved_bytes: 0,
            peak_reserved_bytes: 0,
        }
    }

    pub fn with_mode(
        vulkan_device: Arc<VulkanDevice>,
        device_index: usize,
        mode: AllocatorMode,
    ) -> Self {
        let mut a = Self::new(vulkan_device, device_index);
        a.mode = mode;
        a
    }

    pub fn warm(&mut self, dtype: DType, n_elements: usize, count: usize) -> Result<()> {
        let bytes_per = dtype.packed_buffer_bytes(n_elements);
        let slot = self.cache.entry((dtype, n_elements)).or_default();
        for _ in 0..count {
            let v = VulkanStorage::zeros(
                self.vulkan_device.clone(),
                self.device_index,
                dtype,
                n_elements,
            )?;
            let storage: Storage = Arc::new(v);
            slot.push(storage);
            self.reserved_bytes += bytes_per;
        }
        if self.reserved_bytes > self.peak_reserved_bytes {
            self.peak_reserved_bytes = self.reserved_bytes;
        }
        Ok(())
    }

    pub fn cache_len(&self, dtype: DType, n_elements: usize) -> usize {
        self.cache
            .get(&(dtype, n_elements))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    pub fn vulkan_device(&self) -> &Arc<VulkanDevice> {
        &self.vulkan_device
    }
}

impl Allocator for VulkanAllocator {
    fn device(&self) -> Device {
        Device::Vulkan(self.device_index)
    }
    fn mode(&self) -> AllocatorMode {
        self.mode
    }
    fn set_mode(&mut self, mode: AllocatorMode) -> Result<()> {
        self.mode = mode;
        Ok(())
    }
    fn alloc(&mut self, dtype: DType, n_elements: usize) -> Result<Storage> {
        if let Some(slot) = self.cache.get_mut(&(dtype, n_elements)) {
            if let Some(s) = slot.pop() {
                return Ok(s);
            }
        }
        match self.mode {
            AllocatorMode::Frozen => Err(allocator_frozen_error(
                "VulkanAllocator::alloc",
                dtype.packed_buffer_bytes(n_elements),
            )),
            AllocatorMode::Owned | AllocatorMode::Pool => {
                let v = VulkanStorage::zeros(
                    self.vulkan_device.clone(),
                    self.device_index,
                    dtype,
                    n_elements,
                )?;
                let storage: Storage = Arc::new(v);
                let bytes = dtype.packed_buffer_bytes(n_elements);
                self.reserved_bytes += bytes;
                if self.reserved_bytes > self.peak_reserved_bytes {
                    self.peak_reserved_bytes = self.reserved_bytes;
                }
                Ok(storage)
            }
        }
    }
    fn reserved_bytes(&self) -> usize {
        self.reserved_bytes
    }
    fn peak_reserved_bytes(&self) -> usize {
        self.peak_reserved_bytes
    }
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
        // VulkanDevice construction is out of scope here; tests that
        // want a real device must arrange it. Returning None for now
        // keeps the test path safe (early-return).
        None
    }

    #[test]
    fn vulkan_allocator_skipped_when_no_device() {
        if let Some(dev) = maybe_vulkan_device() {
            let a = VulkanAllocator::new(dev, 0);
            assert_eq!(a.mode(), AllocatorMode::Owned);
            assert_eq!(a.device(), Device::Vulkan(0));
        } else {
            eprintln!("skip: no VulkanDevice available in test scope");
        }
    }
}

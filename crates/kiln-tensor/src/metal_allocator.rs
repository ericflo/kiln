//! `MetalAllocator` — the Metal [`Allocator`] impl.
//!
//! Mirrors [`crate::CudaAllocator`]'s contract: Owned/Pool/Frozen
//! mode handling, `warm()` for pre-Frozen pool population,
//! `reserved_bytes` / `peak_reserved_bytes` accounting. Backed by
//! [`MetalStorage::zeros`] through the existing
//! `Arc<MetalDevice>` plumbing.
//!
//! Phase 7 swaps the candle `MetalDevice` for a direct metal-rs
//! handle; the public API stays stable because allocation routes
//! through `MetalStorage::zeros`, which is the candle-removal lift
//! target.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::MetalDevice;

use crate::{
    allocator_frozen_error, Allocator, AllocatorMode, DType, Device, MetalStorage, Result, Storage,
};

#[derive(Debug)]
pub struct MetalAllocator {
    candle_device: Arc<MetalDevice>,
    device_index: usize,
    mode: AllocatorMode,
    cache: HashMap<(DType, usize), Vec<Storage>>,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
}

impl MetalAllocator {
    pub fn new(candle_device: Arc<MetalDevice>, device_index: usize) -> Self {
        MetalAllocator {
            candle_device,
            device_index,
            mode: AllocatorMode::Owned,
            cache: HashMap::new(),
            reserved_bytes: 0,
            peak_reserved_bytes: 0,
        }
    }

    pub fn with_mode(
        candle_device: Arc<MetalDevice>,
        device_index: usize,
        mode: AllocatorMode,
    ) -> Self {
        let mut a = Self::new(candle_device, device_index);
        a.mode = mode;
        a
    }

    pub fn warm(&mut self, dtype: DType, n_elements: usize, count: usize) -> Result<()> {
        let bytes_per = dtype.packed_buffer_bytes(n_elements);
        let slot = self.cache.entry((dtype, n_elements)).or_default();
        for _ in 0..count {
            let metal = MetalStorage::zeros(
                self.candle_device.clone(),
                self.device_index,
                dtype,
                n_elements,
            )?;
            let storage: Storage = Arc::new(metal);
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

    pub fn candle_device(&self) -> &Arc<MetalDevice> {
        &self.candle_device
    }
}

impl Allocator for MetalAllocator {
    fn device(&self) -> Device {
        Device::Metal(self.device_index)
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
                "MetalAllocator::alloc",
                dtype.packed_buffer_bytes(n_elements),
            )),
            AllocatorMode::Owned | AllocatorMode::Pool => {
                let metal = MetalStorage::zeros(
                    self.candle_device.clone(),
                    self.device_index,
                    dtype,
                    n_elements,
                )?;
                let storage: Storage = Arc::new(metal);
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
    use candle_core::Device as CandleDevice;

    fn metal_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_METAL_TEST").ok().as_deref() == Some("1")
    }

    fn maybe_metal_device() -> Option<Arc<MetalDevice>> {
        if !metal_test_enabled() {
            return None;
        }
        match CandleDevice::new_metal(0).ok()? {
            CandleDevice::Metal(d) => Some(Arc::new(d)),
            _ => None,
        }
    }

    #[test]
    fn metal_allocator_starts_in_owned_mode() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let a = MetalAllocator::new(dev, 0);
        assert_eq!(a.mode(), AllocatorMode::Owned);
        assert_eq!(a.device(), Device::Metal(0));
        assert_eq!(a.reserved_bytes(), 0);
    }

    #[test]
    fn metal_pool_warm_serves_alloc() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let mut a = MetalAllocator::with_mode(dev, 0, AllocatorMode::Pool);
        a.warm(DType::F32, 16, 2).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 2);
        let _s = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 1);
    }

    #[test]
    fn metal_frozen_alloc_fails_on_cache_miss() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let mut a = MetalAllocator::with_mode(dev, 0, AllocatorMode::Frozen);
        let e = a.alloc(DType::F32, 4).unwrap_err();
        assert!(e.to_string().contains("MetalAllocator::alloc"), "got: {e}");
    }
}

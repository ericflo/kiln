//! `CudaAllocator` — the CUDA [`Allocator`] impl.
//!
//! Wraps an `Arc<candle_core::cuda_backend::CudaDevice>` (the same
//! handle every CUDA kernel crate uses) and produces [`CudaStorage`]
//! allocations through it.
//!
//! Phase 2.1.1 of #1082 ships:
//!
//! - Owned/Pool/Frozen mode handling identical to [`crate::CpuAllocator`]
//! - `warm(dtype, n_elements, count)` for pre-Frozen pool population
//!   (used by Phase 5's [`crate::CaptureSession`] before flipping mode)
//! - `reserved_bytes` / `peak_reserved_bytes` accounting
//!
//! GPU-only tests live behind `KILN_TENSOR_CUDA_TEST=1` so the CI
//! compile path (which links `cuda` against `candle-core` but has no
//! GPU) doesn't spuriously fail.
//!
//! Phase 7 swaps `Arc<CudaDevice>` for a direct
//! `Arc<cudarc::driver::CudaContext>` — the public API of this
//! struct stays stable because allocation goes through
//! `CudaStorage::zeros`, which is the candle-removal lift target.

use std::collections::HashMap;
use std::sync::Arc;

use candle_core::cuda_backend::CudaDevice;

use crate::{
    allocator_frozen_error, Allocator, AllocatorMode, CudaStorage, DType, Device, Result, Storage,
};

#[derive(Debug)]
pub struct CudaAllocator {
    /// Candle CUDA device handle. Held for stream affinity + the
    /// `alloc_zeros::<u8>` helper. Phase 7 swaps to a direct cudarc
    /// `CudaContext`.
    candle_device: Arc<CudaDevice>,
    /// CUDA device index — matches the index of `candle_device`'s
    /// owning context.
    device_index: usize,
    mode: AllocatorMode,
    /// Free-list cache keyed on `(dtype, n_elements)`. See
    /// [`CpuAllocator`] for the mode contract.
    cache: HashMap<(DType, usize), Vec<Storage>>,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
}

impl CudaAllocator {
    /// Construct in `Owned` mode bound to `candle_device` at the
    /// given CUDA index.
    pub fn new(candle_device: Arc<CudaDevice>, device_index: usize) -> Self {
        CudaAllocator {
            candle_device,
            device_index,
            mode: AllocatorMode::Owned,
            cache: HashMap::new(),
            reserved_bytes: 0,
            peak_reserved_bytes: 0,
        }
    }

    /// Construct directly in a given mode (tests, capture session).
    pub fn with_mode(
        candle_device: Arc<CudaDevice>,
        device_index: usize,
        mode: AllocatorMode,
    ) -> Self {
        let mut a = Self::new(candle_device, device_index);
        a.mode = mode;
        a
    }

    /// Pre-warm the pool: allocate `count` storages of `(dtype,
    /// n_elements)` and stash them in the cache so subsequent
    /// `alloc()` calls (under any mode) can pull from there.
    ///
    /// Phase 5's `CaptureSession::begin()` calls this for every
    /// `(dtype, n_elements)` the captured graph needs before
    /// flipping to `Frozen`.
    pub fn warm(&mut self, dtype: DType, n_elements: usize, count: usize) -> Result<()> {
        let bytes_per = dtype.packed_buffer_bytes(n_elements);
        let slot = self.cache.entry((dtype, n_elements)).or_default();
        for _ in 0..count {
            let cuda =
                CudaStorage::zeros(self.candle_device.clone(), self.device_index, dtype, n_elements)?;
            let storage: Storage = Arc::new(cuda);
            slot.push(storage);
            self.reserved_bytes += bytes_per;
        }
        if self.reserved_bytes > self.peak_reserved_bytes {
            self.peak_reserved_bytes = self.reserved_bytes;
        }
        Ok(())
    }

    /// Number of cached storages for a given key (testing / introspection).
    pub fn cache_len(&self, dtype: DType, n_elements: usize) -> usize {
        self.cache
            .get(&(dtype, n_elements))
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Borrow the underlying candle device handle.
    pub fn candle_device(&self) -> &Arc<CudaDevice> {
        &self.candle_device
    }
}

impl Allocator for CudaAllocator {
    fn device(&self) -> Device {
        Device::Cuda(self.device_index)
    }

    fn mode(&self) -> AllocatorMode {
        self.mode
    }

    fn set_mode(&mut self, mode: AllocatorMode) -> Result<()> {
        // Same accepting transitions as CpuAllocator. The Frozen check
        // is enforced on alloc cache-miss, not at mode change.
        self.mode = mode;
        Ok(())
    }

    fn alloc(&mut self, dtype: DType, n_elements: usize) -> Result<Storage> {
        // Try the cache first.
        if let Some(slot) = self.cache.get_mut(&(dtype, n_elements)) {
            if let Some(s) = slot.pop() {
                return Ok(s);
            }
        }
        // Cache miss.
        match self.mode {
            AllocatorMode::Frozen => Err(allocator_frozen_error(
                "CudaAllocator::alloc",
                dtype.packed_buffer_bytes(n_elements),
            )),
            AllocatorMode::Owned | AllocatorMode::Pool => {
                let cuda = CudaStorage::zeros(
                    self.candle_device.clone(),
                    self.device_index,
                    dtype,
                    n_elements,
                )?;
                let storage: Storage = Arc::new(cuda);
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

    fn cuda_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_CUDA_TEST").ok().as_deref() == Some("1")
    }

    fn maybe_cuda_device() -> Option<Arc<CudaDevice>> {
        if !cuda_test_enabled() {
            return None;
        }
        match CandleDevice::new_cuda(0).ok()? {
            CandleDevice::Cuda(d) => Some(Arc::new(d)),
            _ => None,
        }
    }

    #[test]
    fn cuda_allocator_starts_in_owned_mode() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let a = CudaAllocator::new(dev, 0);
        assert_eq!(a.mode(), AllocatorMode::Owned);
        assert_eq!(a.device(), Device::Cuda(0));
        assert_eq!(a.reserved_bytes(), 0);
    }

    #[test]
    fn cuda_owned_alloc_increments_reserved() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::new(dev, 0);
        let s = a.alloc(DType::BF16, 32).unwrap();
        assert_eq!(s.dtype(), DType::BF16);
        assert_eq!(s.byte_len(), 64);
        assert_eq!(a.reserved_bytes(), 64);
        assert_eq!(a.peak_reserved_bytes(), 64);
    }

    #[test]
    fn cuda_pool_alloc_serves_from_warm_cache() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::with_mode(dev, 0, AllocatorMode::Pool);
        a.warm(DType::F32, 16, 2).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 2);
        let _s1 = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 1);
        let _s2 = a.alloc(DType::F32, 16).unwrap();
        assert_eq!(a.cache_len(DType::F32, 16), 0);
    }

    #[test]
    fn cuda_frozen_alloc_fails_on_cache_miss() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::with_mode(dev, 0, AllocatorMode::Frozen);
        let e = a.alloc(DType::F32, 4).unwrap_err();
        assert!(
            e.to_string().contains("CudaAllocator::alloc"),
            "got: {e}"
        );
    }

    #[test]
    fn cuda_frozen_alloc_succeeds_when_warm_cache_has_match() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::with_mode(dev, 0, AllocatorMode::Pool);
        a.warm(DType::F32, 4, 1).unwrap();
        a.set_mode(AllocatorMode::Frozen).unwrap();
        let s = a.alloc(DType::F32, 4).unwrap();
        assert_eq!(s.byte_len(), 16);
    }

    #[test]
    fn cuda_set_mode_transitions_freely() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let mut a = CudaAllocator::new(dev, 0);
        a.set_mode(AllocatorMode::Pool).unwrap();
        assert_eq!(a.mode(), AllocatorMode::Pool);
        a.set_mode(AllocatorMode::Frozen).unwrap();
        assert_eq!(a.mode(), AllocatorMode::Frozen);
        a.set_mode(AllocatorMode::Owned).unwrap();
        assert_eq!(a.mode(), AllocatorMode::Owned);
    }
}

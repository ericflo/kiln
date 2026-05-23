//! CUDA storage impl behind the `cuda` feature flag.
//!
//! Wraps `cudarc::driver::CudaSlice<u8>` (the actual buffer) + dtype +
//! `Arc<candle_core::cuda_backend::CudaDevice>` for stream affinity.
//!
//! # Anti-pattern 1 compliance
//!
//! Per the issue:
//!
//! > `kiln-tensor` is not a candle wrapper. Storage is
//! > `cudarc::CudaSlice` directly. No `candle_core::Tensor` field on
//! > `kiln_tensor::Tensor`.
//!
//! `CudaStorage` does **not** hold a `candle_core::Tensor`. The buffer
//! is a `CudaSlice<u8>` we own. The candle `CudaDevice` is held only
//! for its `cuda_stream()` accessor + its `alloc_zeros::<T>` helper;
//! that's the same pattern in use across `kiln-rmsnorm-kernel`,
//! `kiln-gdn-kernel`, `kiln-marlin-gemm`, `kiln-flash-attn`, etc.
//! Phase 7 of #1082 (candle removal) replaces `Arc<CudaDevice>` with a
//! direct `Arc<cudarc::driver::CudaContext>` + `Arc<CudaStream>`.
//!
//! # Phase 1.6 scope (storage layer only)
//!
//! - `zeros(device, dtype, n_elements)` — async device alloc + memset.
//! - `from_slice(device, dtype, slice)` — take ownership of an
//!   existing `CudaSlice<u8>` allocated through candle's device. This
//!   is the FFI seam that today's kernel crates plug into.
//! - `StorageBackend` impl — `device() / dtype() / byte_len() / as_any()`.
//! - `slice()` / `slice_mut()` accessors for the existing kernel-crate
//!   FFI sites that want raw byte pointers.
//!
//! Math ops, H2D/D2H helpers, pinned-host staging — separate later PRs.

use std::any::Any;
use std::sync::Arc;

use candle_core::cuda_backend::CudaDevice;
use candle_core::cuda_backend::cudarc::driver::CudaSlice;

use crate::{DType, Device, Error, Result, StorageBackend};

/// CUDA-backed storage. Byte-typed; dtype carried alongside for dispatch.
///
/// The handed-down `CudaSlice<u8>` is allocated via candle's
/// `CudaDevice` accessor today; Phase 7 swaps that for a direct cudarc
/// `CudaContext::default_stream().alloc_zeros::<u8>` once the candle
/// dep is gone.
#[derive(Debug)]
pub struct CudaStorage {
    /// Device-index variant of [`Device`]. Stored explicitly so
    /// `StorageBackend::device()` is O(1) (no context-query syscall).
    device: Device,
    /// Element dtype tag.
    dtype: DType,
    /// The actual byte buffer on the GPU.
    slice: CudaSlice<u8>,
    /// Candle CUDA device handle. Held for stream affinity (Phase 1.x
    /// `StreamPlanner` reads it) and for the in-flight kernel-crate
    /// FFI calls that take `&CudaDevice` as their first argument.
    candle_device: Arc<CudaDevice>,
}

impl CudaStorage {
    /// Allocate `n_elements` worth of bytes for `dtype` on
    /// `candle_device`. Buffer is zero-initialized via candle's
    /// `alloc_zeros::<u8>(n)`.
    ///
    /// `device_index` is the CUDA device index — must match the index
    /// of the candle device's owning context. Stored as the
    /// [`Device::Cuda`] variant.
    pub fn zeros(
        candle_device: Arc<CudaDevice>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        let slice = candle_device
            .alloc_zeros::<u8>(byte_len)
            .map_err(|e| {
                Error::Msg(format!("CudaStorage::zeros: alloc_zeros<u8>({byte_len}) failed: {e:?}"))
            })?;
        Ok(CudaStorage {
            device: Device::Cuda(device_index),
            dtype,
            slice,
            candle_device,
        })
    }

    /// Wrap an existing `CudaSlice<u8>` allocated by the caller.
    ///
    /// Validates the slice length against
    /// `dtype.size_in_bytes()` for non-packed dtypes (must be a
    /// multiple); packed dtypes have no per-element alignment.
    pub fn from_slice(
        candle_device: Arc<CudaDevice>,
        device_index: usize,
        dtype: DType,
        slice: CudaSlice<u8>,
    ) -> Result<Self> {
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !slice.len().is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "CudaStorage::from_slice: slice len {} is not a multiple of \
                     size_in_bytes({:?}) = {}",
                    slice.len(),
                    dtype,
                    per
                )));
            }
        }
        Ok(CudaStorage {
            device: Device::Cuda(device_index),
            dtype,
            slice,
            candle_device,
        })
    }

    /// Borrow the underlying byte slice. The existing kernel-crate
    /// FFI sites that want the raw device pointer reach this then
    /// call `.device_ptr(&stream)` per the cudarc 0.19 pattern.
    pub fn slice(&self) -> &CudaSlice<u8> {
        &self.slice
    }

    /// Mutable borrow for in-place ops. Bumps no version counter at
    /// this layer (anti-pattern 16 versioning is a Tensor-level concern,
    /// enforced once `kiln-autograd` lands).
    pub fn slice_mut(&mut self) -> &mut CudaSlice<u8> {
        &mut self.slice
    }

    /// The candle CUDA device handle this storage was allocated on.
    /// Used by FFI sites + the Phase 1.x `StreamPlanner` to read
    /// stream affinity.
    pub fn candle_device(&self) -> &Arc<CudaDevice> {
        &self.candle_device
    }
}

impl StorageBackend for CudaStorage {
    fn device(&self) -> Device {
        self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn byte_len(&self) -> usize {
        self.slice.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Construct a fresh [`crate::Storage`] handle (`Arc<dyn StorageBackend>`)
/// holding a [`CudaStorage`]. Convenience constructor matching the CPU
/// `cpu_zeros` helper.
pub fn cuda_zeros(
    candle_device: Arc<CudaDevice>,
    device_index: usize,
    dtype: DType,
    n_elements: usize,
) -> Result<crate::Storage> {
    let storage = CudaStorage::zeros(candle_device, device_index, dtype, n_elements)?;
    Ok(Arc::new(storage))
}

// ----------------------------------------------------------------------
// Tests are GPU-only — gated by KILN_TENSOR_CUDA_TEST=1 so a host with
// cudarc + candle's cuda feature compiled in but no actual GPU doesn't
// spuriously fail.
// ----------------------------------------------------------------------

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
    fn zeros_round_sizes() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let storage = CudaStorage::zeros(dev.clone(), 0, DType::BF16, 64).unwrap();
        assert_eq!(storage.device(), Device::Cuda(0));
        assert_eq!(storage.dtype(), DType::BF16);
        assert_eq!(storage.byte_len(), 128);

        let storage = CudaStorage::zeros(dev, 0, DType::Int4Packed, 16).unwrap();
        assert_eq!(storage.byte_len(), 8); // 16 elements packed -> 8 bytes
    }

    #[test]
    fn from_slice_validates_alignment() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let slice = dev.alloc_zeros::<u8>(17).unwrap();
        let err = CudaStorage::from_slice(dev.clone(), 0, DType::F32, slice).unwrap_err();
        assert!(err.to_string().contains("not a multiple"));
    }

    #[test]
    fn cuda_zeros_returns_arc_storage() {
        let Some(dev) = maybe_cuda_device() else {
            eprintln!("skip: KILN_TENSOR_CUDA_TEST unset or no GPU");
            return;
        };
        let s: crate::Storage = cuda_zeros(dev, 0, DType::F32, 4).unwrap();
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.byte_len(), 16);
        assert_eq!(s.device(), Device::Cuda(0));
        // Downcast to ensure the concrete type is CudaStorage.
        let cuda = s.as_any().downcast_ref::<CudaStorage>().expect("downcast");
        assert_eq!(cuda.slice().len(), 16);
    }
}

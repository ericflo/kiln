//! `kiln_tensor::Storage` — the storage trait + the CPU storage impl.
//!
//! Replaces the candle storage layer at the call sites that name
//! `candle_core::Storage::Cuda` (195), `candle_core::Storage::Metal`
//! (232), `candle_core::CpuStorage` (6), etc. — together over 600 of
//! the 1,799 candle call sites the Phase 0.1 audit captured.
//!
//! # Trait shape
//!
//! From the Phase 1 bullet:
//!
//! > **`Storage` is `Arc<dyn StorageBackend>`** with `Send + Sync`
//! > everywhere. Pin the threading model in Phase 1 — kiln-tensor ops
//! > are callable from any thread.
//!
//! The trait is intentionally narrow: read-only metadata
//! (`device`, `dtype`, `byte_len`) plus the lifecycle hooks the
//! Phase 1.5 / 1.6 / 1.7 backends need. The actual tensor operations
//! (matmul, softmax, …) hang off `DeviceOp` (Phase 1.x's CustomOpN
//! replacement) and `BackendRuntime`, not off this trait — keeping
//! `Storage` thin lets parity tests trivially A/B against the CPU
//! reference.
//!
//! # Why a trait, not an enum
//!
//! `Storage` is an `Arc<dyn StorageBackend>` rather than an enum because
//! the per-backend storage types (e.g. `cudarc::CudaSlice`) carry
//! runtime handles whose Rust types we don't want to mention in the
//! Cargo dependency tree of CPU-only callers. The trait stays
//! object-safe (no generics, no `Self`-by-value methods).

use std::any::Any;
use std::sync::Arc;

use crate::{DType, Device, Error, Result};

/// Object-safe storage trait. Implementations live in per-backend
/// modules and are erased behind [`Storage`].
pub trait StorageBackend: Any + Send + Sync + core::fmt::Debug {
    /// The device this storage lives on.
    fn device(&self) -> Device;

    /// The dtype of the elements this storage carries.
    fn dtype(&self) -> DType;

    /// Physical byte length of the storage.
    ///
    /// For packed dtypes ([`DType::Int4Packed`] / [`DType::Fp4Packed`])
    /// this is the **physical** size (≈ `n_elements / 2`), not the
    /// logical one. The conversion lives on
    /// [`DType::packed_buffer_bytes`].
    fn byte_len(&self) -> usize;

    /// `&dyn Any` for downcast-on-demand at the per-backend boundary
    /// without leaking the concrete backend type into the trait.
    fn as_any(&self) -> &dyn Any;
}

/// Reference-counted, object-erased storage.
///
/// `Arc<dyn StorageBackend>` is the production handle every Tensor
/// carries. Cloning is `O(1)`; the inner storage is shared (refcount-
/// bumped). Aliasing two `Storage` values that point to the same
/// physical buffer is supported and is how zero-copy views work
/// alongside the [`Layout`](crate::Layout) descriptor.
pub type Storage = Arc<dyn StorageBackend>;

// ----------------------------------------------------------------------
// CPU storage
// ----------------------------------------------------------------------

/// Byte-typed CPU storage. The actual element layout is the byte
/// pattern of `dtype` — e.g. 2 bytes per BF16 element, 4 per F32.
///
/// CPU is the canonical numerical reference (per the issue's DoD), so
/// `CpuStorage` is exercised on every host (no GPU stack needed). The
/// 4 GPU storage backends (Phases 1.5 / 1.6 / 1.7) plug into the same
/// `StorageBackend` trait and the parity tests compare against this
/// CPU implementation.
#[derive(Debug)]
pub struct CpuStorage {
    dtype: DType,
    bytes: Vec<u8>,
}

impl CpuStorage {
    /// Allocate a zero-initialized CPU storage with capacity for
    /// `n_elements` of `dtype`. The physical byte size is computed via
    /// [`DType::packed_buffer_bytes`] so packed dtypes are right-sized.
    pub fn zeros(dtype: DType, n_elements: usize) -> Self {
        let n_bytes = dtype.packed_buffer_bytes(n_elements);
        CpuStorage {
            dtype,
            bytes: vec![0u8; n_bytes],
        }
    }

    /// Take ownership of an existing byte buffer. Validates that the
    /// buffer length is a multiple of `dtype.size_in_bytes()` for
    /// non-packed dtypes.
    pub fn from_bytes(dtype: DType, bytes: Vec<u8>) -> Result<Self> {
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !bytes.len().is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "CpuStorage::from_bytes: byte len {} is not a multiple of size_in_bytes({:?}) = {}",
                    bytes.len(),
                    dtype,
                    per
                )));
            }
        }
        Ok(CpuStorage { dtype, bytes })
    }

    /// Read-only byte view.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Mutable byte view. Mutating in place bumps no version counter at
    /// the storage layer (versioning is a Tensor-level concern; see
    /// anti-pattern 16).
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}

impl StorageBackend for CpuStorage {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn byte_len(&self) -> usize {
        self.bytes.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Construct a fresh [`Storage`] (`Arc<dyn StorageBackend>`) on the CPU.
///
/// Convenience constructor — equivalent to
/// `Arc::new(CpuStorage::zeros(dtype, n))`.
pub fn cpu_zeros(dtype: DType, n_elements: usize) -> Storage {
    Arc::new(CpuStorage::zeros(dtype, n_elements))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_zeros_round_sizes() {
        let s = CpuStorage::zeros(DType::F32, 16);
        assert_eq!(s.byte_len(), 64);
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.device(), Device::Cpu);

        let s = CpuStorage::zeros(DType::BF16, 16);
        assert_eq!(s.byte_len(), 32);

        let s = CpuStorage::zeros(DType::Int4Packed, 16);
        // Packed: 16 elements -> 8 bytes
        assert_eq!(s.byte_len(), 8);

        let s = CpuStorage::zeros(DType::Int4Packed, 17);
        // ceil(17 / 2)
        assert_eq!(s.byte_len(), 9);
    }

    #[test]
    fn cpu_zeros_is_zeroed() {
        let s = CpuStorage::zeros(DType::F32, 4);
        assert!(s.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn from_bytes_validates_alignment() {
        let ok = CpuStorage::from_bytes(DType::F32, vec![0u8; 16]).unwrap();
        assert_eq!(ok.byte_len(), 16);

        let err = CpuStorage::from_bytes(DType::F32, vec![0u8; 17]).unwrap_err();
        assert!(err.to_string().contains("not a multiple"));
    }

    #[test]
    fn from_bytes_accepts_packed_any_size() {
        let s = CpuStorage::from_bytes(DType::Int4Packed, vec![0u8; 9]).unwrap();
        assert_eq!(s.byte_len(), 9);
    }

    #[test]
    fn as_bytes_mut_mutates_in_place() {
        let mut s = CpuStorage::zeros(DType::U8, 4);
        s.as_bytes_mut().copy_from_slice(&[1, 2, 3, 4]);
        assert_eq!(s.as_bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn cpu_zeros_returns_storage() {
        let s: Storage = cpu_zeros(DType::F32, 2);
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.byte_len(), 8);
        assert_eq!(s.device(), Device::Cpu);

        // Downcast to inspect the concrete type.
        let cpu = s.as_any().downcast_ref::<CpuStorage>().expect("downcast");
        assert_eq!(cpu.as_bytes().len(), 8);
    }

    #[test]
    fn storage_is_send_sync() {
        // Compile-time check.
        fn _check<T: Send + Sync>(_: &T) {}
        let s: Storage = cpu_zeros(DType::F32, 1);
        _check(&s);
    }
}

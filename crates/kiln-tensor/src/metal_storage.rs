//! Metal storage impl behind the `metal` feature flag.
//!
//! Wraps `Arc<metal::Buffer>` (the actual buffer) + dtype +
//! `Arc<candle_core::metal_backend::MetalDevice>` for command-queue
//! affinity. The `metal` crate (Apple's MTLBuffer binding) is reached
//! through candle's re-export; Phase 7 of #1082 (candle removal)
//! replaces `MetalDevice` with a direct `MTLDevice` + command-queue
//! handle pair.
//!
//! # Anti-pattern 1 compliance
//!
//! Per the issue:
//!
//! > `kiln-tensor` is not a candle wrapper. Storage is
//! > `metal::Buffer` directly.
//!
//! `MetalStorage` does not hold a `candle_core::Tensor`. The buffer is
//! `Arc<metal::Buffer>` we own (allocated via candle's
//! `MetalDevice::allocate_zeros` — the same allocator path used by
//! candle internally; the `Arc<Buffer>` is then fully ours).
//!
//! # Apple Silicon UMA invariant
//!
//! Per the issue's Phase 1 bullet:
//!
//! > **Apple Silicon UMA zero-copy invariant**: on M-series, CPU and
//! > GPU share physical memory; `MTLStorageModeShared` buffers are
//! > addressable from both. kiln-tensor exposes `Tensor::is_unified_memory()`
//! > and `Tensor::as_host_slice()` (zero-copy on UMA, errors elsewhere)
//! > so the safetensors loader and the optimizer don't pay a copy
//! > round-trip on Mac. Discrete-GPU Macs (Pro/Studio with M-Ultra)
//! > are still UMA — no host pinning needed.
//!
//! Candle's `allocate_zeros` returns a `Shared`-mode buffer (the
//! `RESOURCE_OPTIONS` constant in `vendor/candle-core/src/metal_backend`).
//! `MetalStorage::is_unified_memory()` returns true; the zero-copy
//! host accessor lands in a follow-up PR (it needs a stride/layout
//! check that this PR keeps off the critical path).

use std::any::Any;
use std::sync::Arc;

use candle_core::metal_backend::MetalDevice;
use candle_core::metal_backend::candle_metal_kernels::metal::Buffer as MetalBuffer;

use crate::{DType, Device, Error, Result, StorageBackend};

/// Metal-backed storage. Byte-typed; dtype carried alongside for dispatch.
///
/// Holds an `Arc<metal::Buffer>` directly (anti-pattern 1). The
/// candle `MetalDevice` is held for command-queue affinity and for
/// the `allocate_zeros` / `new_buffer` accessors that the existing
/// kernel paths in `kiln-model::backend::metal` already use.
#[derive(Debug)]
pub struct MetalStorage {
    device: Device,
    dtype: DType,
    buffer: Arc<MetalBuffer>,
    candle_device: Arc<MetalDevice>,
}

impl MetalStorage {
    /// Allocate `n_elements` worth of bytes for `dtype` on
    /// `candle_device`. Zero-initialized via candle's blit-encoder
    /// fill (the same path used by `MetalDevice::allocate_zeros`).
    ///
    /// `device_index` is the Metal device index (always 0 on Apple
    /// Silicon today; Multi-GPU Macs would use 1+).
    pub fn zeros(
        candle_device: Arc<MetalDevice>,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        let byte_len = dtype.packed_buffer_bytes(n_elements);
        let buffer = candle_device.allocate_zeros(byte_len).map_err(|e| {
            Error::Msg(format!(
                "MetalStorage::zeros: allocate_zeros({byte_len}) failed: {e:?}"
            ))
        })?;
        Ok(MetalStorage {
            device: Device::Metal(device_index),
            dtype,
            buffer,
            candle_device,
        })
    }

    /// Wrap an existing `Arc<metal::Buffer>` allocated by the caller.
    ///
    /// Validates the buffer length against `dtype.size_in_bytes()`
    /// for non-packed dtypes.
    pub fn from_buffer(
        candle_device: Arc<MetalDevice>,
        device_index: usize,
        dtype: DType,
        buffer: Arc<MetalBuffer>,
    ) -> Result<Self> {
        let len = buffer.length() as usize;
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !len.is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "MetalStorage::from_buffer: buffer len {len} is not a multiple of \
                     size_in_bytes({:?}) = {per}",
                    dtype
                )));
            }
        }
        Ok(MetalStorage {
            device: Device::Metal(device_index),
            dtype,
            buffer,
            candle_device,
        })
    }

    /// Borrow the underlying buffer. The existing kernel-crate FFI
    /// sites in `kiln-model::backend::metal` plug in via this
    /// accessor (mirrors `candle_core::metal_backend::buffer_o` 232
    /// call sites from Phase 0.1's audit).
    pub fn buffer(&self) -> &Arc<MetalBuffer> {
        &self.buffer
    }

    /// Borrow the candle Metal device — same handle the existing
    /// kernels in `kiln-model::backend::metal` consume.
    pub fn candle_device(&self) -> &Arc<MetalDevice> {
        &self.candle_device
    }

    /// Returns `true` iff this storage's buffer is in a UMA-compatible
    /// storage mode (shared / managed). On Apple Silicon today,
    /// candle's `allocate_zeros` returns a Shared-mode buffer, so this
    /// returns `true` for storage allocated via [`MetalStorage::zeros`].
    ///
    /// Phase 1.x's `Tensor::as_host_slice()` accessor reads this — UMA
    /// devices return a `&[u8]` directly; non-UMA returns an error.
    pub fn is_unified_memory(&self) -> bool {
        use candle_core::metal_backend::candle_metal_kernels::metal::MTLStorageMode;
        matches!(
            self.buffer.storage_mode(),
            MTLStorageMode::Shared | MTLStorageMode::Managed
        )
    }
}

impl StorageBackend for MetalStorage {
    fn device(&self) -> Device {
        self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn byte_len(&self) -> usize {
        self.buffer.length() as usize
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Construct a fresh [`crate::Storage`] handle holding a [`MetalStorage`].
pub fn metal_zeros(
    candle_device: Arc<MetalDevice>,
    device_index: usize,
    dtype: DType,
    n_elements: usize,
) -> Result<crate::Storage> {
    let storage = MetalStorage::zeros(candle_device, device_index, dtype, n_elements)?;
    Ok(Arc::new(storage))
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
    fn zeros_round_sizes() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let storage = MetalStorage::zeros(dev.clone(), 0, DType::BF16, 64).unwrap();
        assert_eq!(storage.device(), Device::Metal(0));
        assert_eq!(storage.dtype(), DType::BF16);
        // Candle's metal allocator rounds up to its slab size; only assert
        // that the byte_len is *at least* what we asked for.
        assert!(storage.byte_len() >= 128);

        let storage = MetalStorage::zeros(dev, 0, DType::Int4Packed, 16).unwrap();
        assert!(storage.byte_len() >= 8);
    }

    #[test]
    fn from_buffer_validates_alignment() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        // 17 bytes is not a multiple of f32 (4). The allocator may round
        // up our request, so this only tests the *post-round-up* size
        // when we explicitly pass a 17-byte buffer.
        let small = dev.allocate_zeros(17).unwrap();
        let raw_len = small.length() as usize;
        let result = MetalStorage::from_buffer(dev, 0, DType::F32, small);
        if raw_len.is_multiple_of(4) {
            // Allocator rounded up; the validation passes.
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().to_string().contains("not a multiple"));
        }
    }

    #[test]
    fn metal_zeros_returns_arc_storage() {
        let Some(dev) = maybe_metal_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        let s: crate::Storage = metal_zeros(dev, 0, DType::F32, 4).unwrap();
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.device(), Device::Metal(0));
        let metal_s = s.as_any().downcast_ref::<MetalStorage>().expect("downcast");
        // UMA invariant: Shared-mode buffer (Apple Silicon default).
        assert!(metal_s.is_unified_memory());
    }
}

//! Shared helpers for Phase 7 kt-API ports.
//!
//! Every kernel crate that adds a `kiln_tensor::Tensor`-typed
//! surface alongside its candle-typed twin (per #1082 line 322's
//! pattern) needs the same boilerplate:
//!
//! 1. Validate dtype + contiguity + CUDA-ness on each input.
//! 2. Downcast `kiln_tensor::Tensor::storage()` to
//!    `kiln_tensor::CudaStorage`.
//! 3. Compute the byte offset (`layout.start_offset() *
//!    dtype.size_in_bytes()`).
//! 4. Allocate fresh CUDA-backed outputs on the same device.
//!
//! This crate factors those four pieces into one place so the
//! kernel-crate kt-APIs (kiln-flash-attn, kiln-conv1d-kernel,
//! kiln-rmsnorm-kernel, kiln-marlin-gemm, kiln-gdn-kernel, etc.) all
//! share one canonical implementation. When a future change to
//! kiln-tensor's CUDA storage layout needs to ripple through, it
//! ripples through one file rather than seven.
//!
//! # Usage
//!
//! ```ignore
//! use kiln_kt_bridge::{alloc_cuda_tensor, cuda_storage_and_byte_offset, BridgeError};
//! use kiln_tensor::{DType, Tensor};
//!
//! fn my_op_kt(x: &Tensor) -> Result<Tensor, BridgeError> {
//!     let (st, off) = cuda_storage_and_byte_offset(x, DType::BF16, "x")?;
//!     let out = alloc_cuda_tensor(st, DType::BF16, vec![x.shape()[0]])?;
//!     // ... build device pointer, call FFI ...
//!     Ok(out)
//! }
//! ```
//!
//! Errors use the generic [`BridgeError`]; downstream crates can
//! convert via `?` into their own typed errors via `From`.

// (#1082) candle fully removed from kiln-kt-bridge. This crate is now a
// kt-native bridge only: the CUDA-touching helpers (CudaStorage downcast,
// alloc, device-pointer extraction) plus the kt-native `tape_bridge` tape
// scopes. The former candle⟷kt Device/DType mappers, the candle CUDA
// Tensor borrow/copy adapters, the `pub use candle_core;` re-export, and
// the candle GradStore bridge were all dead (no consumer activated the
// `candle` feature) and have been deleted.

/// kt-native tape scopes (#1082). The IO-mapping + tape-authoritative
/// backward machinery used by the `_kt` training adapters. Device-agnostic
/// scope plumbing; the CUDA-specific kt helpers stay
/// `#[cfg(feature = "cuda")]` inside this crate's other modules.
#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
pub mod tape_bridge;

// `KtDType` is used by the candle-free CUDA helpers
// (`cuda_storage_and_byte_offset`, `alloc_cuda_tensor`,
// `cuda_input_device_ptr`). (#1082)
#[cfg(feature = "cuda")]
use kiln_tensor::DType as KtDType;
#[cfg(feature = "cuda")]
use kiln_tensor::{CudaStorage, StorageBackend, Tensor as KtTensor};

/// Generic error for kt-API bridge operations.
///
/// Each kernel crate can keep its own typed error and convert from
/// `BridgeError` via `From` — the inner string carries the full
/// diagnostic.
#[derive(Debug, Clone)]
pub struct BridgeError {
    pub message: String,
}

impl BridgeError {
    pub fn new(message: impl Into<String>) -> Self {
        BridgeError {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for BridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BridgeError {}

/// Validate and downcast a `kiln_tensor::Tensor` to its CUDA
/// storage, returning both the storage and the start-offset
/// **expressed in bytes**.
///
/// Errors when:
/// - `t.dtype() != expected`
/// - `t.is_contiguous() == false`
/// - the storage isn't a `CudaStorage` (i.e. CPU or another GPU)
#[cfg(feature = "cuda")]
pub fn cuda_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<(&'a CudaStorage, usize), BridgeError> {
    if t.dtype() != expected {
        return Err(BridgeError::new(format!(
            "kt-bridge: {name} must be {expected}, got {}",
            t.dtype()
        )));
    }
    if !t.is_contiguous() {
        return Err(BridgeError::new(format!(
            "kt-bridge: {name} must be contiguous"
        )));
    }
    let st = t
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| BridgeError::new(format!("kt-bridge: {name} must be CUDA")))?;
    let off = t.layout().start_offset() * expected.size_in_bytes();
    Ok((st, off))
}

/// Allocate a freshly-zeroed CUDA-backed `kiln_tensor::Tensor` of
/// `dtype` and `shape`, on the same CUDA device as `source`.
///
/// Used to allocate output tensors that mirror the candle path's
/// `Tensor::zeros((shape), dtype, device)` pattern.
#[cfg(feature = "cuda")]
pub fn alloc_cuda_tensor(
    source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, BridgeError> {
    // cuda_zeros_ctx (#1082) derives the candle device internally from
    // device_index, so we don't read .candle_device() off source.
    let device_index = source.device().index().unwrap_or(0);
    let n: usize = shape.iter().product();
    let storage = kiln_tensor::cuda_zeros_ctx(device_index, dtype, n)
        .map_err(|e| BridgeError::new(format!("kt-bridge alloc: {e}")))?;
    KtTensor::from_parts(
        storage,
        kiln_tensor::Layout::contiguous(shape),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| BridgeError::new(format!("kt-bridge alloc wrap: {e}")))
}

/// Borrow the underlying [`CudaStorage`] of a fresh kt-allocated
/// output tensor. Panics if the storage isn't CUDA — only call this
/// on tensors just returned from [`alloc_cuda_tensor`].
#[cfg(feature = "cuda")]
pub fn cuda_storage_of_output(t: &KtTensor) -> &CudaStorage {
    t.storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("kt-bridge: alloc_cuda_tensor output must be CUDA")
}

/// Owner-agnostic device-pointer extraction for an input kt-Tensor.
///
/// Combines [`cuda_storage_and_byte_offset`] (dtype + contiguity +
/// CUDA-ness check, plus layout-start-offset → bytes conversion) with
/// the raw-pointer accessor that works for both `Owned` and
/// `Borrowed` storage (see `kiln_tensor::CudaStorage::device_ptr_raw`).
///
/// Returns the absolute device pointer at the tensor's first live
/// element. Use this in place of the
/// `st.slice().slice(off..).device_ptr(&stream)` chain when migrating
/// a kt-API entry point to accept Borrowed storage (Phase 7 v2).
///
/// **What this drops vs. the old chain:** the old chain returned a
/// `SyncOnDrop` guard that recorded the stream's workload to the
/// `CudaSlice`'s read event on drop, ensuring cross-stream readers of
/// the same allocation wait for prior writes. The single-stream
/// kt-API call pattern doesn't depend on this (every op uses the
/// candle device's default stream), so dropping the guard is safe in
/// the current call shapes. Future multi-stream kt-API additions
/// must re-introduce explicit synchronization at the StreamPlanner
/// layer.
#[cfg(feature = "cuda")]
pub fn cuda_input_device_ptr(
    t: &KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<u64, BridgeError> {
    let (st, byte_off) = cuda_storage_and_byte_offset(t, expected, name)?;
    let (base_ptr, byte_len) = st.device_ptr_raw();
    if byte_off > byte_len {
        return Err(BridgeError::new(format!(
            "kt-bridge: {name} byte_off {byte_off} > storage byte_len {byte_len}"
        )));
    }
    Ok(base_ptr + byte_off as u64)
}

/// Owner-agnostic device-pointer extraction for an output kt-Tensor
/// (always `Owned` since [`alloc_cuda_tensor`] produces owned
/// storage). Returns the base pointer; outputs use start_offset=0.
#[cfg(feature = "cuda")]
pub fn cuda_output_device_ptr(t: &KtTensor) -> u64 {
    let st = cuda_storage_of_output(t);
    st.device_ptr_raw().0
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_implements_display() {
        let e = BridgeError::new("something bad");
        assert_eq!(format!("{e}"), "something bad");
    }

    #[test]
    fn error_implements_error_trait() {
        let e: Box<dyn std::error::Error> = Box::new(BridgeError::new("boxed"));
        assert!(e.to_string().contains("boxed"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cpu_tensor_is_rejected_as_cuda() {
        use kiln_tensor::Tensor;
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = cuda_storage_and_byte_offset(&t, KtDType::F32, "x").unwrap_err();
        assert!(e.to_string().contains("must be CUDA"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn wrong_dtype_is_rejected() {
        use kiln_tensor::Tensor;
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = cuda_storage_and_byte_offset(&t, KtDType::BF16, "x").unwrap_err();
        assert!(e.to_string().contains("must be"));
    }

    // (#1082) The candle dtype/device-mapper tests (dtype_mapping_round_trip,
    // kt_dtype_to_candle_basic, kt_device_from_candle_cpu_roundtrip,
    // candle_device_from_kt_*) were deleted alongside the candle bridge fns
    // they exercised. The crate is candle-free now.
}

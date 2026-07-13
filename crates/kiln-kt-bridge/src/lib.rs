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
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "rocm",
    test
))]
pub mod tape_bridge;

// `KtDType` + the kt-Tensor/StorageBackend types are shared by the CUDA and
// ROCm device-pointer helpers. `CudaStorage` / `RocmStorage` are each gated to
// their own backend feature. (#1082 / R.4)
#[cfg(feature = "cuda")]
use kiln_tensor::CudaStorage;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use kiln_tensor::DType as KtDType;
#[cfg(feature = "rocm")]
use kiln_tensor::RocmStorage;
#[cfg(any(feature = "cuda", feature = "rocm"))]
use kiln_tensor::{StorageBackend, Tensor as KtTensor};
#[cfg(feature = "rocm")]
use std::sync::Arc;

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
    // #1082 illegal-address localizer: check BOTH ends of the addressable span,
    // not just the start. A kernel input whose declared shape over-runs its
    // backing storage (e.g. a batched [batch,...] state cat-assembled to the
    // wrong batch, or a mis-sized paged-KV/GDN buffer after the candle->kt
    // migration) otherwise sails past this bridge with a valid start pointer
    // and faults INSIDE the kernel as an async, sticky CUDA_ERROR_ILLEGAL_ADDRESS
    // that only surfaces (with a misleading context string) at the next stream
    // sync — making it nearly impossible to attribute. The end check converts
    // that into a clean, named Rust error naming the exact input + shape BEFORE
    // the launch. The tensor is contiguous here (verified in
    // cuda_storage_and_byte_offset), so addressable_byte_size == element_count *
    // bytes_per_element. Strictly tighter precondition: a no-op for correctly
    // sized tensors (the bs=1 and last-known-good paths).
    let addressable = t.layout().addressable_byte_size(expected.size_in_bytes());
    if byte_off + addressable > byte_len {
        return Err(BridgeError::new(format!(
            "kt-bridge: {name} OOB: start_offset_bytes {byte_off} + addressable {addressable} \
             > storage byte_len {byte_len} (shape {:?}, dtype {expected})",
            t.dims()
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

// ----------------------------------------------------------------------
// ROCm device-pointer seam (Phase R.4) — the exact analogs of the CUDA
// helpers above, swapping CudaStorage -> RocmStorage and cuda_zeros_ctx ->
// rocm_zeros_ctx. The kernel-crate kt-APIs reach these (or the backend-neutral
// dispatchers below) under `--features rocm`.
// ----------------------------------------------------------------------

/// ROCm analog of [`cuda_storage_and_byte_offset`].
#[cfg(feature = "rocm")]
pub fn rocm_storage_and_byte_offset<'a>(
    t: &'a KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<(&'a RocmStorage, usize), BridgeError> {
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
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| BridgeError::new(format!("kt-bridge: {name} must be ROCm")))?;
    let off = t.layout().start_offset() * expected.size_in_bytes();
    Ok((st, off))
}

/// ROCm analog of [`alloc_cuda_tensor`].
#[cfg(feature = "rocm")]
pub fn alloc_rocm_tensor(
    source: &RocmStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, BridgeError> {
    let device_index = source.device().index().unwrap_or(0);
    let n: usize = shape.iter().product();
    let ctx = source.context();
    let storage = RocmStorage::zeros_ctx(&ctx, device_index, dtype, n)
        .map_err(|e| BridgeError::new(format!("kt-bridge alloc: {e}")))?;
    KtTensor::from_parts(
        Arc::new(storage),
        kiln_tensor::Layout::contiguous(shape),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| BridgeError::new(format!("kt-bridge alloc wrap: {e}")))
}

/// ROCm analog of [`cuda_storage_of_output`].
#[cfg(feature = "rocm")]
pub fn rocm_storage_of_output(t: &KtTensor) -> &RocmStorage {
    t.storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .expect("kt-bridge: alloc_rocm_tensor output must be ROCm")
}

/// ROCm analog of [`cuda_input_device_ptr`] — owner-agnostic input pointer with
/// the same OOB span check.
#[cfg(feature = "rocm")]
pub fn rocm_input_device_ptr(
    t: &KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<u64, BridgeError> {
    let (st, byte_off) = rocm_storage_and_byte_offset(t, expected, name)?;
    let (base_ptr, byte_len) = st.device_ptr_raw();
    let addressable = t.layout().addressable_byte_size(expected.size_in_bytes());
    if byte_off + addressable > byte_len {
        return Err(BridgeError::new(format!(
            "kt-bridge: {name} OOB: start_offset_bytes {byte_off} + addressable {addressable} \
             > storage byte_len {byte_len} (shape {:?}, dtype {expected})",
            t.dims()
        )));
    }
    Ok(base_ptr + byte_off as u64)
}

/// ROCm analog of [`cuda_output_device_ptr`].
#[cfg(feature = "rocm")]
pub fn rocm_output_device_ptr(t: &KtTensor) -> u64 {
    rocm_storage_of_output(t).device_ptr_raw().0
}

/// Raw HIP stream pointer (`*mut c_void`) for an input kt-Tensor's ROCm storage
/// — the FFI `stream` argument kernel launchers expect. Mirrors how the CUDA
/// kt-APIs reach `CudaStorage::cuda_stream_raw`.
#[cfg(feature = "rocm")]
pub fn rocm_stream_raw_of(
    t: &KtTensor,
    name: &'static str,
) -> Result<*mut core::ffi::c_void, BridgeError> {
    let st = t
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| BridgeError::new(format!("kt-bridge: {name} must be ROCm")))?;
    Ok(st.rocm_stream_raw())
}

// ----------------------------------------------------------------------
// Backend-neutral device-pointer dispatchers (Phase R.4). Let a kernel crate's
// single `_kt` wrapper body work on either Device::Cuda or Device::Rocm without
// per-call `cfg`. Each dispatches on the tensor's backend; with only one GPU
// feature active the inactive arm is compiled out.
// ----------------------------------------------------------------------

/// Backend-neutral input device pointer — dispatches to the CUDA or ROCm helper
/// by the tensor's backend.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn device_input_ptr(
    t: &KtTensor,
    expected: KtDType,
    name: &'static str,
) -> Result<u64, BridgeError> {
    use kiln_tensor::Backend;
    match t.device().backend() {
        #[cfg(feature = "cuda")]
        Backend::Cuda => cuda_input_device_ptr(t, expected, name),
        #[cfg(feature = "rocm")]
        Backend::Rocm => rocm_input_device_ptr(t, expected, name),
        other => Err(BridgeError::new(format!(
            "kt-bridge: {name} on unsupported backend {other:?}"
        ))),
    }
}

/// Backend-neutral output device pointer.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn device_output_ptr(t: &KtTensor) -> u64 {
    use kiln_tensor::Backend;
    match t.device().backend() {
        #[cfg(feature = "cuda")]
        Backend::Cuda => cuda_output_device_ptr(t),
        #[cfg(feature = "rocm")]
        Backend::Rocm => rocm_output_device_ptr(t),
        _ => panic!("kt-bridge: device_output_ptr on unsupported backend"),
    }
}

/// Backend-neutral allocation of a fresh, zeroed GPU tensor of `dtype`/`shape`
/// on the SAME device as `source`. Dispatches to `cuda_zeros_ctx` /
/// `rocm_zeros_ctx` by the source tensor's backend.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn alloc_device_tensor_like(
    source: &KtTensor,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, BridgeError> {
    use kiln_tensor::Backend;
    let device = source.device();
    let idx = device.index().unwrap_or(0);
    let n: usize = shape.iter().product();
    let storage = match device.backend() {
        #[cfg(feature = "cuda")]
        Backend::Cuda => kiln_tensor::cuda_zeros_ctx(idx, dtype, n),
        #[cfg(feature = "rocm")]
        Backend::Rocm => kiln_tensor::rocm_zeros_ctx(idx, dtype, n),
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge: alloc_device_tensor_like on unsupported backend {other:?}"
            )));
        }
    }
    .map_err(|e| BridgeError::new(format!("kt-bridge alloc: {e}")))?;
    KtTensor::from_parts(
        storage,
        kiln_tensor::Layout::contiguous(shape),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| BridgeError::new(format!("kt-bridge alloc wrap: {e}")))
}

/// Backend-neutral raw GPU stream pointer for a kt-Tensor's storage — the FFI
/// `stream` argument kernel launchers expect.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub fn device_stream_raw_of(
    t: &KtTensor,
    name: &'static str,
) -> Result<*mut core::ffi::c_void, BridgeError> {
    use kiln_tensor::Backend;
    match t.device().backend() {
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            let st = t
                .storage()
                .as_any()
                .downcast_ref::<CudaStorage>()
                .ok_or_else(|| BridgeError::new(format!("kt-bridge: {name} must be CUDA")))?;
            Ok(st.cuda_stream_raw())
        }
        #[cfg(feature = "rocm")]
        Backend::Rocm => rocm_stream_raw_of(t, name),
        other => Err(BridgeError::new(format!(
            "kt-bridge: device_stream_raw_of {name} on unsupported backend {other:?}"
        ))),
    }
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

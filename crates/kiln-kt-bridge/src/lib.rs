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

use kiln_tensor::{CudaStorage, DType as KtDType, StorageBackend, Tensor as KtTensor};

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
pub fn alloc_cuda_tensor(
    source: &CudaStorage,
    dtype: KtDType,
    shape: Vec<usize>,
) -> Result<KtTensor, BridgeError> {
    let candle_device = source.candle_device().clone();
    let device_index = source.device().index().unwrap_or(0);
    let n: usize = shape.iter().product();
    let storage = kiln_tensor::cuda_zeros(candle_device, device_index, dtype, n)
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
pub fn cuda_storage_of_output(t: &KtTensor) -> &CudaStorage {
    t.storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("kt-bridge: alloc_cuda_tensor output must be CUDA")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::Tensor;

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

    #[test]
    fn cpu_tensor_is_rejected_as_cuda() {
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = cuda_storage_and_byte_offset(&t, KtDType::F32, "x").unwrap_err();
        assert!(e.to_string().contains("must be CUDA"));
    }

    #[test]
    fn wrong_dtype_is_rejected() {
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let e = cuda_storage_and_byte_offset(&t, KtDType::BF16, "x").unwrap_err();
        assert!(e.to_string().contains("must be"));
    }
}

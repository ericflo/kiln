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

/// Map candle's `DType` to kiln-tensor's `DType`. Returns
/// `BridgeError` for variants that have no kt equivalent today.
///
/// This is a building block for the Phase 7 candle→kiln-tensor
/// adapter; the full **zero-copy** `kt_tensor_from_candle_cuda_borrow`
/// (sharing the same CUDA buffer) ships once cudarc exposes the
/// typed→u8 slice reinterpret or a kiln-tensor `BorrowedCudaStorage`
/// variant lands. In the meantime [`kt_tensor_from_candle_cuda_copy`]
/// provides a correct-but-copying adapter that unblocks call-site
/// migration.
pub fn candle_dtype_to_kt(d: candle_core::DType) -> Result<KtDType, BridgeError> {
    use candle_core::DType as C;
    Ok(match d {
        C::F32 => KtDType::F32,
        C::BF16 => KtDType::BF16,
        C::F16 => KtDType::F16,
        C::U32 => KtDType::U32,
        C::U8 => KtDType::U8,
        C::I64 => KtDType::I64,
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge: unsupported candle dtype for kt conversion: {other:?}"
            )));
        }
    })
}

/// Phase 7 candle→kt adapter — **copying variant**.
///
/// Copies the device data backing a candle CUDA `Tensor` into a freshly
/// allocated `kiln_tensor::Tensor` of the same shape and dtype. The
/// returned kt-Tensor owns its own CUDA allocation and is independent of
/// the candle source. Stream affinity follows the candle tensor's CUDA
/// device.
///
/// Use this as the migration primitive when a call site holds a candle
/// `Tensor` and needs to call a kt-API function. Each call costs one
/// device-to-device memcpy; for hot paths, prefer waiting for the
/// zero-copy borrow variant.
///
/// **Requirements**:
/// - `t.device()` must be a CUDA device
/// - `t.is_contiguous()` must be true (caller should `.contiguous()?` first)
/// - `t.dtype()` must round-trip through [`candle_dtype_to_kt`]
///
/// Layout: returns a freshly contiguous kt-Tensor (start_offset = 0,
/// row-major strides). If the candle tensor's `layout.start_offset()` is
/// non-zero, only the live elements are copied — the kt-Tensor doesn't
/// inherit any unused prefix from the candle storage.
#[allow(clippy::needless_pass_by_value)]
pub fn kt_tensor_from_candle_cuda_copy(
    t: &candle_core::Tensor,
) -> Result<KtTensor, BridgeError> {
    use candle_core::{
        backend::BackendDevice,
        cuda_backend::cudarc::driver::{result as cudarc_result, DevicePtr},
        DType as C, DeviceLocation, Storage as CStorage,
    };
    use half::{bf16, f16};

    if !t.is_contiguous() {
        return Err(BridgeError::new(
            "kt-bridge: kt_tensor_from_candle_cuda_copy: tensor must be contiguous \
             (caller should .contiguous()? first)",
        ));
    }
    let kt_dtype = candle_dtype_to_kt(t.dtype())?;
    let shape: Vec<usize> = t.dims().to_vec();
    let n_elems: usize = shape.iter().product();
    let bytes_per_elem = kt_dtype.size_in_bytes();
    let total_bytes = n_elems * bytes_per_elem;

    let (storage_guard, layout) = t.storage_and_layout();
    let cuda_st = match &*storage_guard {
        CStorage::Cuda(c) => c,
        _ => {
            return Err(BridgeError::new(
                "kt-bridge: kt_tensor_from_candle_cuda_copy: tensor must be on CUDA",
            ))
        }
    };

    let candle_device = cuda_st.device().clone();
    let device_index = match candle_device.location() {
        DeviceLocation::Cuda { gpu_id } => gpu_id,
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge copy: expected Cuda location, got {other:?}"
            )));
        }
    };
    let stream = candle_device.cuda_stream();
    let raw_stream = stream.cu_stream();

    // Allocate the destination kt-Tensor's storage (zero-init; the
    // subsequent memcpy overwrites every byte).
    let dst_storage = kiln_tensor::cuda_zeros(candle_device, device_index, kt_dtype, n_elems)
        .map_err(|e| BridgeError::new(format!("kt-bridge copy: alloc dst: {e}")))?;
    let dst_cuda = dst_storage
        .as_any()
        .downcast_ref::<CudaStorage>()
        .expect("cuda_zeros must produce CudaStorage");
    let dst_slice = dst_cuda.slice().slice(0..);

    let off = layout.start_offset();

    // Per-dtype src ptr extraction. The slice on candle's side is
    // typed; we need to dispatch on dtype to call as_cuda_slice<T> with
    // the correct T, then `.slice(off..)` and `.device_ptr(&stream)` to
    // get the raw pointer at the right byte offset.
    let status = unsafe {
        let (dst_ptr, _dst_g) = dst_slice.device_ptr(&stream);
        let (src_ptr, _src_g) = match t.dtype() {
            C::F32 => cuda_st
                .as_cuda_slice::<f32>()
                .map_err(|e| BridgeError::new(format!("kt-bridge copy: as_cuda_slice f32: {e}")))?
                .slice(off..)
                .device_ptr(&stream),
            C::BF16 => cuda_st
                .as_cuda_slice::<bf16>()
                .map_err(|e| BridgeError::new(format!("kt-bridge copy: as_cuda_slice bf16: {e}")))?
                .slice(off..)
                .device_ptr(&stream),
            C::F16 => cuda_st
                .as_cuda_slice::<f16>()
                .map_err(|e| BridgeError::new(format!("kt-bridge copy: as_cuda_slice f16: {e}")))?
                .slice(off..)
                .device_ptr(&stream),
            C::U32 => cuda_st
                .as_cuda_slice::<u32>()
                .map_err(|e| BridgeError::new(format!("kt-bridge copy: as_cuda_slice u32: {e}")))?
                .slice(off..)
                .device_ptr(&stream),
            C::U8 => cuda_st
                .as_cuda_slice::<u8>()
                .map_err(|e| BridgeError::new(format!("kt-bridge copy: as_cuda_slice u8: {e}")))?
                .slice(off..)
                .device_ptr(&stream),
            C::I64 => cuda_st
                .as_cuda_slice::<i64>()
                .map_err(|e| BridgeError::new(format!("kt-bridge copy: as_cuda_slice i64: {e}")))?
                .slice(off..)
                .device_ptr(&stream),
            other => {
                return Err(BridgeError::new(format!(
                    "kt-bridge copy: unsupported candle dtype {other:?}"
                )));
            }
        };
        cudarc_result::memcpy_dtod_async(dst_ptr, src_ptr, total_bytes, raw_stream).map_err(
            |e| BridgeError::new(format!("kt-bridge copy: memcpy_dtod_async: {e:?}")),
        )?;
        0i32
    };
    let _ = status; // memcpy_dtod_async already returned Result; keep var to silence

    KtTensor::from_parts(
        dst_storage,
        kiln_tensor::Layout::contiguous(shape),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| BridgeError::new(format!("kt-bridge copy: wrap: {e}")))
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

    #[test]
    fn dtype_mapping_round_trip() {
        assert_eq!(candle_dtype_to_kt(candle_core::DType::F32).unwrap(), KtDType::F32);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::BF16).unwrap(), KtDType::BF16);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::F16).unwrap(), KtDType::F16);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::U32).unwrap(), KtDType::U32);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::U8).unwrap(), KtDType::U8);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::I64).unwrap(), KtDType::I64);
    }
}

//! Safetensors loader for kiln-tensor.
//!
//! Replaces `candle_core::safetensors::load` at the 14 call sites the
//! Phase 0.1 audit captured.
//!
//! # API
//!
//! The migration target is `kiln_tensor::safetensors::load_cpu`:
//!
//! ```ignore
//! use kiln_tensor as kt;
//! let tensors: std::collections::HashMap<String, kt::Tensor> =
//!     kt::safetensors::load_cpu("model.safetensors")?;
//! ```
//!
//! This matches candle's `load(P, &Device::Cpu)` shape with the device
//! made explicit in the function name. Per-backend loaders
//! (`load_cuda`, `load_metal`, `load_vulkan`) follow as separate PRs
//! once Phase 2 BLAS lands the H2D transfer paths — for now,
//! load to CPU first, then explicit `tensor.to_device(...)` migrates
//! to GPU.
//!
//! Today's call sites that explicitly want a CPU load:
//!
//! ```ignore
//! // Today:
//! let model = candle_core::safetensors::load(path, &candle_core::Device::Cpu)?;
//! // After this PR:
//! let model = kiln_tensor::safetensors::load_cpu(path)?;
//! ```
//!
//! # Why a thin wrapper
//!
//! `safetensors` (the crate) handles the file-format parsing, header
//! validation, and `TensorView` byte-slicing for us. Our wrapper:
//!
//! - Maps each safetensors `Dtype` to our [`DType`].
//! - Copies the byte slice into a `CpuStorage` (the safetensors API
//!   gives us a `&[u8]` borrow into the mmap; we own a fresh Vec).
//! - Builds a `Tensor` with a contiguous Layout.
//!
//! For mmap-friendly loaders (where the tensor data shares the mmap
//! lifetime), the `load_buffer_*` variant returns a `HashMap` that
//! borrows from the provided byte slice — useful for `safetensors`
//! files that fit fully in mmap but where we want to avoid the copy
//! into `CpuStorage`.
//!
//! Note: this PR does NOT implement zero-copy mmap views — that
//! requires a `CpuStorage` variant that borrows from a leaked `Mmap`
//! handle and the lifetime gymnastics that follow. Phase 1.x's
//! "pinned-host staging pool" bullet covers the mmap-aware path.

use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::{Dtype as StDtype, SafeTensors, TensorView};

use crate::{CpuStorage, DType, Error, Layout, Result, Storage, Tensor, TensorId};

/// Load every tensor in `filename` into CPU `Tensor`s.
///
/// Reads the file fully into memory, parses the safetensors header,
/// and copies each tensor's byte slice into a fresh `CpuStorage`.
/// Returns a name → Tensor map.
pub fn load_cpu(filename: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
    let data = std::fs::read(filename.as_ref())
        .map_err(|e| Error::Msg(format!("safetensors::load_cpu: read file failed: {e}")))?;
    load_buffer_cpu(&data)
}

/// Load every tensor from an in-memory safetensors buffer.
///
/// Same semantics as [`load_cpu`] but takes an already-read buffer.
/// Useful for HTTP body / mmap / embedded-asset workflows.
pub fn load_buffer_cpu(buffer: &[u8]) -> Result<HashMap<String, Tensor>> {
    let st = SafeTensors::deserialize(buffer)
        .map_err(|e| Error::Msg(format!("safetensors::load_buffer_cpu: deserialize failed: {e}")))?;
    let mut out = HashMap::new();
    for (name, view) in st.tensors() {
        let tensor = tensor_from_view(&view)?;
        out.insert(name, tensor);
    }
    Ok(out)
}

/// Build a single CPU `Tensor` from a safetensors `TensorView`.
///
/// Exposed so callers that iterate selectively (e.g. "only load
/// weights with prefix X") don't have to roundtrip through the full
/// `HashMap`.
pub fn tensor_from_view(view: &TensorView<'_>) -> Result<Tensor> {
    let dtype = map_dtype(view.dtype())?;
    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data().to_vec();
    let cpu = CpuStorage::from_bytes(dtype, data)?;
    let storage: Storage = std::sync::Arc::new(cpu);
    let layout = Layout::contiguous(shape);
    Tensor::from_parts(storage, layout, TensorId::next())
}

/// Map `safetensors::Dtype` to our [`DType`].
///
/// Supported: F32, F16, BF16, U8, U32, I64, F8_E4M3.
///
/// `F8_E5M2` is currently **not** mapped — safetensors 0.7 doesn't
/// have a stable variant for it (the variants enumerable in this
/// version are F4, F6_E2M3, F6_E3M2, F8_E4M3, F8_E8M0). If a future
/// safetensors version adds F8_E5M2 this mapping picks it up via the
/// `other => Err(...)` arm.
pub fn map_dtype(dt: StDtype) -> Result<DType> {
    Ok(match dt {
        StDtype::F32 => DType::F32,
        StDtype::F16 => DType::F16,
        StDtype::BF16 => DType::BF16,
        StDtype::U8 => DType::U8,
        StDtype::U32 => DType::U32,
        StDtype::I64 => DType::I64,
        StDtype::F8_E4M3 => DType::F8E4M3,
        other => {
            return Err(Error::Msg(format!(
                "kiln_tensor::safetensors: unsupported dtype {other:?}; \
                 supported: F32, F16, BF16, U8, U32, I64, F8_E4M3"
            )));
        }
    })
}

/// Reverse of [`map_dtype`]. Used by `save_*` paths in a later PR.
///
/// Packed dtypes (`Int4Packed`, `Fp4Packed`) have no safetensors
/// equivalent — they ship via the Marlin format, not safetensors.
/// `F8E5M2` has no current safetensors mapping; ship as a Marlin
/// blob alongside the safetensors file for now.
pub fn st_dtype(dtype: DType) -> Result<StDtype> {
    Ok(match dtype {
        DType::F32 => StDtype::F32,
        DType::F16 => StDtype::F16,
        DType::BF16 => StDtype::BF16,
        DType::U8 => StDtype::U8,
        DType::U32 => StDtype::U32,
        DType::I64 => StDtype::I64,
        DType::F8E4M3 => StDtype::F8_E4M3,
        DType::F8E5M2 => {
            return Err(Error::Msg(
                "kiln_tensor::safetensors: F8E5M2 has no safetensors mapping in this \
                 version; ship via Marlin format alongside the safetensors file"
                    .to_string(),
            ));
        }
        DType::Int4Packed | DType::Fp4Packed => {
            return Err(Error::Msg(format!(
                "kiln_tensor::safetensors: packed dtype {dtype} has no safetensors equivalent \
                 — packed weights ship via the Marlin format, not safetensors"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_dtype_known_values() {
        assert_eq!(map_dtype(StDtype::F32).unwrap(), DType::F32);
        assert_eq!(map_dtype(StDtype::BF16).unwrap(), DType::BF16);
        assert_eq!(map_dtype(StDtype::F16).unwrap(), DType::F16);
        assert_eq!(map_dtype(StDtype::U32).unwrap(), DType::U32);
        assert_eq!(map_dtype(StDtype::U8).unwrap(), DType::U8);
        assert_eq!(map_dtype(StDtype::I64).unwrap(), DType::I64);
        assert_eq!(map_dtype(StDtype::F8_E4M3).unwrap(), DType::F8E4M3);
    }

    #[test]
    fn st_dtype_packed_errors() {
        let e = st_dtype(DType::Int4Packed).unwrap_err();
        assert!(e.to_string().contains("packed dtype"));
        let e = st_dtype(DType::Fp4Packed).unwrap_err();
        assert!(e.to_string().contains("packed dtype"));
    }

    #[test]
    fn st_dtype_f8e5m2_errors() {
        let e = st_dtype(DType::F8E5M2).unwrap_err();
        assert!(e.to_string().contains("F8E5M2"));
    }

    #[test]
    fn st_dtype_known_values() {
        assert_eq!(st_dtype(DType::F32).unwrap(), StDtype::F32);
        assert_eq!(st_dtype(DType::BF16).unwrap(), StDtype::BF16);
        assert_eq!(st_dtype(DType::F8E4M3).unwrap(), StDtype::F8_E4M3);
    }

    #[test]
    fn load_buffer_rejects_garbage() {
        // Random bytes are unlikely to parse as a valid safetensors
        // header — the deserialize call should return Err mapped to
        // our Error::Msg.
        let garbage = vec![0u8; 64];
        let err = load_buffer_cpu(&garbage).unwrap_err();
        assert!(err.to_string().contains("deserialize"));
    }

    #[test]
    fn load_cpu_rejects_missing_path() {
        let err = load_cpu("/tmp/kiln-tensor-nonexistent-87f3b9c4d2e1.safetensors").unwrap_err();
        assert!(err.to_string().contains("read file failed"));
    }
}

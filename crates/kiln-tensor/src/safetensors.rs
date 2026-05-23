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

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use safetensors::tensor::{Dtype as StDtype, SafeTensors, TensorView, View};

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

/// Save a `HashMap<String, &Tensor>` to a safetensors file.
///
/// Counterpart to [`load_cpu`]. Reads CPU storage from each tensor,
/// builds a safetensors-compatible view, and writes the file.
///
/// All tensors must be CPU-resident and contiguous. Non-contiguous or
/// GPU tensors must be `.contiguous()`-ed + `.to_device(Cpu)`-ed first
/// (the latter lands when per-backend device transfer is in place).
///
/// Packed dtypes (`Int4Packed`, `Fp4Packed`) are rejected — they ship
/// via the Marlin format alongside the safetensors file.
pub fn save_cpu<P: AsRef<Path>, S: AsRef<str>>(
    tensors: &HashMap<S, &Tensor>,
    filename: P,
) -> Result<()> {
    // Wrap each tensor in a SerializableTensor view; the safetensors
    // crate impl-blocks on `&dyn View` so we collect Box<dyn View>.
    let mut wrappers: Vec<(String, SerializableTensor<'_>)> = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let wrapper = SerializableTensor::try_from_tensor(tensor)?;
        wrappers.push((name.as_ref().to_string(), wrapper));
    }
    // safetensors::serialize_to_file takes any `IntoIterator<Item =
    // (&str, &dyn View)>`, but the exact ergonomics depend on
    // version. We sort by name so the file is deterministic.
    wrappers.sort_by(|a, b| a.0.cmp(&b.0));
    let pairs: Vec<(&str, &SerializableTensor<'_>)> =
        wrappers.iter().map(|(n, w)| (n.as_str(), w)).collect();
    safetensors::tensor::serialize_to_file(pairs, &None, filename.as_ref()).map_err(|e| {
        Error::Msg(format!(
            "safetensors::save_cpu: serialize_to_file failed: {e}"
        ))
    })?;
    Ok(())
}

/// `View` impl wrapper for a borrowed CPU tensor. Used internally by
/// [`save_cpu`].
struct SerializableTensor<'a> {
    dtype: StDtype,
    shape: Vec<usize>,
    bytes: &'a [u8],
}

impl<'a> SerializableTensor<'a> {
    fn try_from_tensor(t: &'a Tensor) -> Result<Self> {
        let dtype = st_dtype(t.dtype())?;
        if !t.is_contiguous() {
            return Err(Error::Msg(format!(
                "safetensors::save_cpu: tensor (id {}) is not contiguous; \
                 call `.contiguous()` first",
                t.id()
            )));
        }
        let cpu = t
            .storage()
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| {
                Error::from_str(
                    "safetensors::save_cpu: only CpuStorage is serializable today; \
                     GPU tensors must be transferred to CPU first",
                )
            })?;
        Ok(SerializableTensor {
            dtype,
            shape: t.shape().to_vec(),
            bytes: cpu.as_bytes(),
        })
    }
}

impl<'a> View for SerializableTensor<'a> {
    fn dtype(&self) -> StDtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
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

    #[test]
    fn save_then_load_round_trip_f32() {
        let t = crate::Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .unwrap();
        let mut to_save: HashMap<String, &crate::Tensor> = HashMap::new();
        to_save.insert("weight".to_string(), &t);
        let tmp = std::env::temp_dir().join("kiln_save_round_trip_f32.safetensors");
        let _ = std::fs::remove_file(&tmp);
        save_cpu(&to_save, &tmp).unwrap();
        let loaded = load_cpu(&tmp).unwrap();
        let loaded_t = loaded.get("weight").expect("weight key");
        assert_eq!(loaded_t.shape(), &[2, 3]);
        assert_eq!(loaded_t.dtype(), crate::DType::F32);
        let cpu = loaded_t
            .storage()
            .as_any()
            .downcast_ref::<crate::CpuStorage>()
            .unwrap();
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
        assert_eq!(back, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn save_rejects_non_contiguous() {
        let t = crate::Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2])
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert!(!t.is_contiguous());
        let mut m: HashMap<String, &crate::Tensor> = HashMap::new();
        m.insert("w".to_string(), &t);
        let tmp = std::env::temp_dir().join("kiln_save_rejects_noncontig.safetensors");
        let _ = std::fs::remove_file(&tmp);
        let e = save_cpu(&m, &tmp).unwrap_err();
        assert!(e.to_string().contains("not contiguous"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn save_multiple_tensors_round_trip() {
        let a = crate::Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = crate::Tensor::from_slice(&[10u32, 20, 30, 40], vec![4]).unwrap();
        let mut m: HashMap<String, &crate::Tensor> = HashMap::new();
        m.insert("alpha".to_string(), &a);
        m.insert("beta".to_string(), &b);
        let tmp = std::env::temp_dir().join("kiln_save_multi.safetensors");
        let _ = std::fs::remove_file(&tmp);
        save_cpu(&m, &tmp).unwrap();
        let loaded = load_cpu(&tmp).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["alpha"].dtype(), crate::DType::F32);
        assert_eq!(loaded["beta"].dtype(), crate::DType::U32);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn save_rejects_packed_dtype() {
        let t = crate::Tensor::zeros_cpu(vec![4], crate::DType::Int4Packed);
        let mut m: HashMap<String, &crate::Tensor> = HashMap::new();
        m.insert("packed".to_string(), &t);
        let tmp = std::env::temp_dir().join("kiln_save_rejects_packed.safetensors");
        let _ = std::fs::remove_file(&tmp);
        let e = save_cpu(&m, &tmp).unwrap_err();
        assert!(e.to_string().contains("packed"));
        let _ = std::fs::remove_file(&tmp);
    }
}

//! `kiln_tensor::Tensor` — the production tensor handle.
//!
//! Combines the four foundational pieces:
//!
//! - [`Storage`] (`Arc<dyn StorageBackend>`) — the physical byte buffer
//!   + device + dtype
//! - [`Layout`] — the logical view (shape, strides, start_offset)
//! - [`TensorId`] — stable identity for optimizer + autograd bookkeeping
//!
//! Clones are O(1) (only the `Arc<Storage>` is bumped; layout copies
//! are cheap). Tensor handles are `Send + Sync` per the threading
//! model: kiln-tensor ops are callable from any thread.
//!
//! # Anti-pattern 1 — kiln-tensor is not a candle wrapper
//!
//! Notice what is **not** here:
//!
//! - No `candle_core::Tensor` field. Storage is `Arc<dyn StorageBackend>`
//!   directly.
//! - No `BackpropOp` field. Autograd lives in `kiln-autograd` (Phase 6a)
//!   and keys on [`TensorId`].
//! - No `requires_grad` flag at the tensor level. That's a property of
//!   the [`Parameter`](crate) (Phase 2.5), not the tensor.
//!
//! # Migration story
//!
//! At the 20 `candle_core::Tensor` call sites the Phase 0.1 audit
//! captured plus the thousands of method-call sites (`x.matmul(y)`,
//! `x.contiguous()`, …), the substitution is `kiln_tensor::Tensor`.
//! The method surface for math ops lands in Phase 1.x as `DeviceOp`
//! plus `BackendRuntime::dispatch`; this PR ships only the
//! Tensor + view-op surface.

use std::sync::Arc;

use crate::{
    cpu_zeros, CpuStorage, DType, Device, Element, Error, Layout, Result, Storage, StorageBackend,
    TensorId,
};

/// kiln-tensor's production tensor handle.
///
/// See the module doc for the layout. Clones are O(1).
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    layout: Layout,
    id: TensorId,
}

impl Tensor {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Construct a zero-initialized CPU tensor with the given shape and dtype.
    pub fn zeros_cpu(shape: impl Into<Vec<usize>>, dtype: DType) -> Self {
        let shape: Vec<usize> = shape.into();
        let n_elements: usize = shape.iter().product();
        let storage = cpu_zeros(dtype, n_elements);
        let layout = Layout::contiguous(shape);
        Tensor {
            storage,
            layout,
            id: TensorId::next(),
        }
    }

    /// Build a CPU tensor from a typed slice + shape. The slice length
    /// must equal the product of `shape`.
    pub fn from_slice<E: Element>(values: &[E], shape: impl Into<Vec<usize>>) -> Result<Self> {
        let shape: Vec<usize> = shape.into();
        let want: usize = shape.iter().product();
        if values.len() != want {
            return Err(Error::Msg(format!(
                "Tensor::from_slice: shape {shape:?} has {want} elements but slice has {}",
                values.len()
            )));
        }
        let bytes = E::to_bytes(values);
        let cpu = CpuStorage::from_bytes(E::DTYPE, bytes)?;
        let storage: Storage = Arc::new(cpu);
        let layout = Layout::contiguous(shape);
        Ok(Tensor {
            storage,
            layout,
            id: TensorId::next(),
        })
    }

    /// Build a CPU tensor from a typed `Vec` + shape (consumes the vec).
    pub fn from_vec<E: Element>(values: Vec<E>, shape: impl Into<Vec<usize>>) -> Result<Self> {
        Self::from_slice(&values, shape)
    }

    /// Construct a [`Tensor`] from raw parts. Used by per-backend
    /// storage impls (Phase 1.6+ CUDA, 1.7 Metal, 1.8 Vulkan) and by
    /// view ops in this module.
    pub fn from_parts(storage: Storage, layout: Layout, id: TensorId) -> Result<Self> {
        // Defense-in-depth: validate addressable_byte_size against the
        // physical buffer. A layout that points past the storage end
        // is undefined behavior on any backend.
        let per = storage.dtype().size_in_bytes();
        if !storage.dtype().is_packed() && per > 0 {
            let required = layout.addressable_byte_size(per);
            if required > storage.byte_len() {
                return Err(Error::Msg(format!(
                    "Tensor::from_parts: layout requires {} bytes but storage has {}",
                    required,
                    storage.byte_len()
                )));
            }
        }
        Ok(Tensor {
            storage,
            layout,
            id,
        })
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Storage handle (`Arc<dyn StorageBackend>`).
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Borrow the layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Borrow the shape.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Borrow the strides.
    pub fn strides(&self) -> &[usize] {
        self.layout.strides()
    }

    /// Logical rank.
    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    /// Total element count.
    pub fn element_count(&self) -> usize {
        self.layout.element_count()
    }

    /// The dtype carried by the storage.
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// The device the storage lives on.
    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Stable identity.
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Is this tensor contiguous (row-major, start_offset == 0)?
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    // ------------------------------------------------------------------
    // Zero-copy view ops (anti-pattern 10)
    // ------------------------------------------------------------------

    /// Narrow along `axis` to `[offset .. offset+length]`. Zero-copy.
    pub fn narrow(&self, axis: usize, offset: usize, length: usize) -> Result<Self> {
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            layout: self.layout.narrow_axis(axis, offset, length)?,
            id: TensorId::next(),
        })
    }

    /// Swap two axes. Zero-copy.
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            layout: self.layout.transpose(axis_a, axis_b)?,
            id: TensorId::next(),
        })
    }

    /// Apply a full axes permutation. Zero-copy.
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            layout: self.layout.permute(axes)?,
            id: TensorId::next(),
        })
    }

    /// Reshape — only valid on contiguous tensors. See [`Layout::reshape`].
    ///
    /// For non-contiguous reshape, the caller must invoke
    /// [`contiguous()`](Tensor::contiguous) first (which logs a
    /// `kiln_profile_contiguous_copy` event in Phase 1.x).
    pub fn reshape(&self, new_shape: impl Into<Vec<usize>>) -> Result<Self> {
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            layout: self.layout.reshape(new_shape)?,
            id: TensorId::next(),
        })
    }

    /// Produce a contiguous copy of this tensor. **Always allocates** —
    /// this is the "explicit `contiguous()` that logs" call site per
    /// anti-pattern 2 (the copy-counter instrumentation lands in
    /// Phase 1.x and reads this method's invocation count).
    ///
    /// On CPU, materializes a fresh `CpuStorage` and walks the strided
    /// view. The CPU backend is the canonical reference; non-CPU
    /// backends override via `BackendRuntime::contiguous` once Phase
    /// 1.5+ lands.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        if !self.device().is_cpu() {
            return Err(Error::Msg(format!(
                "Tensor::contiguous: only CPU contiguous is implemented in Phase 1.5; \
                 device {} support lands with the per-backend storage impl",
                self.device()
            )));
        }
        if self.dtype().is_packed() {
            return Err(Error::Msg(
                "Tensor::contiguous: packed dtype contiguous is not supported".to_string(),
            ));
        }

        let cpu = self
            .storage
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| Error::from_str("Tensor::contiguous: CPU device must hold CpuStorage"))?;

        let per = self.dtype().size_in_bytes();
        debug_assert!(per > 0);

        let shape = self.shape();
        let strides = self.strides();
        let start = self.layout.start_offset();
        let n_elements = self.element_count();

        let mut out = vec![0u8; n_elements * per];

        // Walk the logical iteration order and copy per-element bytes.
        // For Phase 1.5 we accept the per-element overhead; Phase 2's
        // CPU backend can replace this with a stride-aware blocked
        // copy if profile shows it matters.
        let rank = shape.len();
        let mut idx = vec![0usize; rank];
        let mut out_off = 0usize;
        loop {
            // Compute physical offset for the current logical idx.
            let mut phys = start;
            for (axis, &i) in idx.iter().enumerate() {
                phys += i * strides[axis];
            }
            let src = phys * per;
            out[out_off..out_off + per].copy_from_slice(&cpu.as_bytes()[src..src + per]);
            out_off += per;

            // Increment idx in row-major order.
            if rank == 0 {
                break;
            }
            let mut axis = rank;
            let mut bumped = false;
            while axis > 0 {
                axis -= 1;
                idx[axis] += 1;
                if idx[axis] < shape[axis] {
                    bumped = true;
                    break;
                }
                idx[axis] = 0;
            }
            if !bumped {
                break;
            }
        }

        let new_cpu = CpuStorage::from_bytes(self.dtype(), out)?;
        Ok(Tensor {
            storage: Arc::new(new_cpu),
            layout: Layout::contiguous(shape.to_vec()),
            id: TensorId::next(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_cpu_basics() {
        let t = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(t.element_count(), 12);
        assert!(t.is_contiguous());
        // Storage bytes should be 12 * 4 = 48.
        assert_eq!(t.storage().byte_len(), 48);
    }

    #[test]
    fn from_slice_typed() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&values, vec![2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.storage().byte_len(), 24);
        // Round-trip the bytes back to f32.
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
        assert_eq!(back, values);
    }

    #[test]
    fn from_slice_shape_mismatch_errors() {
        let values = vec![1.0f32, 2.0, 3.0];
        let e = Tensor::from_slice(&values, vec![2, 2]).unwrap_err();
        assert!(e.to_string().contains("has 4 elements"));
    }

    #[test]
    fn id_changes_on_view() {
        let t = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        let n = t.narrow(0, 0, 2).unwrap();
        assert_ne!(t.id(), n.id());
    }

    #[test]
    fn clone_shares_storage() {
        let t = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        let c = t.clone();
        // Same Arc pointer.
        assert!(Arc::ptr_eq(t.storage(), c.storage()));
        // Same id — clone preserves identity per anti-pattern 11.
        assert_eq!(t.id(), c.id());
    }

    #[test]
    fn narrow_is_zero_copy() {
        let t = Tensor::zeros_cpu(vec![4, 5], DType::F32);
        let n = t.narrow(0, 1, 2).unwrap();
        assert!(Arc::ptr_eq(t.storage(), n.storage()));
        assert_eq!(n.shape(), &[2, 5]);
        assert!(!n.is_contiguous());
    }

    #[test]
    fn transpose_is_zero_copy() {
        let t = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        let tt = t.transpose(0, 1).unwrap();
        assert!(Arc::ptr_eq(t.storage(), tt.storage()));
        assert_eq!(tt.shape(), &[4, 3]);
        assert!(!tt.is_contiguous());
    }

    #[test]
    fn reshape_only_on_contiguous() {
        let t = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        let r = t.reshape(vec![12]).unwrap();
        assert_eq!(r.shape(), &[12]);

        let tt = t.transpose(0, 1).unwrap();
        assert!(tt.reshape(vec![12]).is_err());
    }

    #[test]
    fn contiguous_on_contiguous_clones() {
        let t = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        let c = t.contiguous().unwrap();
        // The clone optimization should preserve the storage Arc.
        assert!(Arc::ptr_eq(t.storage(), c.storage()));
    }

    #[test]
    fn contiguous_after_transpose_materializes() {
        // Build [[1,2,3],[4,5,6]] f32, transpose to [[1,4],[2,5],[3,6]],
        // then contiguous().
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&v, vec![2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        let c = tt.contiguous().unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert!(c.is_contiguous());

        // The materialized buffer should be [1,4,2,5,3,6].
        let cpu = c.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
        assert_eq!(back, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn from_parts_validates_size() {
        let storage = cpu_zeros(DType::F32, 4);
        let layout = Layout::contiguous(vec![100]); // 100 elements != 4
        let e = Tensor::from_parts(storage, layout, TensorId::next()).unwrap_err();
        assert!(e.to_string().contains("requires"));
    }
}

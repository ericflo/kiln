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

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::{
    cpu_zeros, profile, CpuStorage, DType, Device, Element, Error, Layout, Result, Storage,
    StorageBackend, TensorId,
};

/// kiln-tensor's production tensor handle.
///
/// See the module doc for the layout. Clones are O(1) — they share
/// the underlying storage, layout (cheap copy), `TensorId`, and the
/// version counter. Mutating one clone's storage via [`bump_version`]
/// is observed by every other clone (the version is `Arc<AtomicU64>`).
///
/// # Version counter (anti-pattern 16 hook)
///
/// Per the issue's anti-pattern 16:
///
/// > In-place mutation invalidates the tape. Any in-place op
/// > (optimizer step, residual accumulate-in-place, in-place norm)
/// > bumps a per-tensor version counter; the backward path asserts
/// > the version is unchanged from when the tape recorded the
/// > forward. Failing the assertion is a programming error, not a
/// > tolerated mode.
///
/// The version is stored as `Arc<AtomicU64>` so clones of a Tensor
/// share it: if the optimizer step mutates `param`, the version
/// stored on the autograd tape's input-list also observes the bump.
/// `kiln_autograd::Tape::backward` compares the stored snapshot
/// against the live load and errors on drift.
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Storage,
    layout: Layout,
    id: TensorId,
    /// In-place mutation version counter. Bumped by callers that
    /// mutate the underlying storage (optimizer step, residual fuse,
    /// future in-place norm). Read by `kiln_autograd::Tape::backward`
    /// to detect anti-pattern 16 violations.
    version: Arc<AtomicU64>,
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
            version: Arc::new(AtomicU64::new(0)),
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
            version: Arc::new(AtomicU64::new(0)),
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
            version: Arc::new(AtomicU64::new(0)),
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
    ///
    /// View ops share the parent's version counter: if the parent's
    /// storage is mutated in place, every view sees the bump.
    pub fn narrow(&self, axis: usize, offset: usize, length: usize) -> Result<Self> {
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            layout: self.layout.narrow_axis(axis, offset, length)?,
            id: TensorId::next(),
            version: Arc::clone(&self.version),
        })
    }

    /// Swap two axes. Zero-copy.
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            layout: self.layout.transpose(axis_a, axis_b)?,
            id: TensorId::next(),
            version: Arc::clone(&self.version),
        })
    }

    /// Apply a full axes permutation. Zero-copy.
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            layout: self.layout.permute(axes)?,
            id: TensorId::next(),
            version: Arc::clone(&self.version),
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
            version: Arc::clone(&self.version),
        })
    }

    /// Insert a size-1 axis at position `axis` (`0..=rank`). Zero-copy.
    ///
    /// Inverse of [`Tensor::squeeze`]. Convenient for promoting a
    /// `[D]` tensor to `[1, D]` for matmul, or adding a head-axis
    /// before attention masking.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self> {
        let mut new_shape = self.shape().to_vec();
        if axis > new_shape.len() {
            return Err(crate::Error::Msg(format!(
                "unsqueeze: axis {axis} > rank {}",
                new_shape.len()
            )));
        }
        new_shape.insert(axis, 1);
        self.reshape(new_shape)
    }

    /// Remove the size-1 axis at position `axis`. Errors if the axis
    /// is not exactly size 1. Zero-copy.
    pub fn squeeze(&self, axis: usize) -> Result<Self> {
        let shape = self.shape();
        if axis >= shape.len() {
            return Err(crate::Error::Msg(format!(
                "squeeze: axis {axis} out of range for rank {}",
                shape.len()
            )));
        }
        if shape[axis] != 1 {
            return Err(crate::Error::Msg(format!(
                "squeeze: axis {axis} has size {}, expected 1",
                shape[axis]
            )));
        }
        let mut new_shape = shape.to_vec();
        new_shape.remove(axis);
        self.reshape(new_shape)
    }

    /// Swap the trailing two axes (matrix-style transpose). Zero-copy.
    /// Errors on rank < 2.
    pub fn t(&self) -> Result<Self> {
        if self.rank() < 2 {
            return Err(crate::Error::Msg(format!(
                "t: rank must be ≥ 2, got {}",
                self.rank()
            )));
        }
        self.transpose(self.rank() - 2, self.rank() - 1)
    }

    /// Move a single axis from `from` to `to`. Zero-copy via permute.
    pub fn move_axis(&self, from: usize, to: usize) -> Result<Self> {
        let rank = self.rank();
        if from >= rank || to >= rank {
            return Err(crate::Error::Msg(format!(
                "move_axis: from={from} to={to} out of range for rank-{rank}"
            )));
        }
        let mut axes: Vec<usize> = (0..rank).collect();
        let moved = axes.remove(from);
        axes.insert(to, moved);
        self.permute(&axes)
    }

    /// Flatten to a rank-1 tensor of `element_count()` elements.
    /// Zero-copy if input is contiguous.
    pub fn flatten(&self) -> Result<Self> {
        let n = self.element_count();
        self.reshape(vec![n])
    }

    /// Flatten contiguous range of axes `[start_axis, end_axis]`
    /// inclusive. Other axes preserved. PyTorch-style.
    pub fn flatten_range(&self, start_axis: usize, end_axis: usize) -> Result<Self> {
        let shape = self.shape();
        if start_axis > end_axis || end_axis >= shape.len() {
            return Err(crate::Error::Msg(format!(
                "flatten_range: invalid range [{start_axis}, {end_axis}] for rank-{}",
                shape.len()
            )));
        }
        let mut new_shape = Vec::with_capacity(shape.len() - (end_axis - start_axis));
        new_shape.extend_from_slice(&shape[..start_axis]);
        let flat_size: usize = shape[start_axis..=end_axis].iter().product();
        new_shape.push(flat_size);
        new_shape.extend_from_slice(&shape[end_axis + 1..]);
        self.reshape(new_shape)
    }

    /// Produce a contiguous copy of this tensor. **Always allocates
    /// when called on a non-contiguous tensor** (the fast path returns
    /// the clone on already-contiguous inputs without a copy).
    ///
    /// This is the "explicit `contiguous()` that logs" call site per
    /// anti-pattern 2 — the materializing branch bumps the
    /// [`profile::contiguous_copy_count`](crate::profile::contiguous_copy_count)
    /// counter so `bench-results/`'s "copies per token" metric stays
    /// surfaced. Phase 9's bench-gate reads it.
    ///
    /// On CPU, materializes a fresh `CpuStorage` and walks the strided
    /// view. The CPU backend is the canonical reference; non-CPU
    /// backends override via `BackendRuntime::contiguous` once Phase
    /// 1.x lands.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        // Anti-pattern 2: this is the materializing branch — count it.
        profile::emit_contiguous_copy();
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
            // Materializing copy → fresh storage → fresh version.
            // (View ops share the parent's version; this branch does
            // not.)
            version: Arc::new(AtomicU64::new(0)),
        })
    }

    // ------------------------------------------------------------------
    // Version counter (anti-pattern 16)
    // ------------------------------------------------------------------

    /// Current version counter. `0` for a fresh tensor; bumps every
    /// time the storage is mutated in place.
    ///
    /// `kiln_autograd::Tape::backward` compares this to the snapshot
    /// stored on the tape node and errors if they differ.
    pub fn current_version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// Borrow the `Arc<AtomicU64>` version handle. Used by
    /// `kiln_autograd::Tape::record` to store a live pointer so
    /// backward-time checks see the up-to-date value.
    pub fn version_handle(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.version)
    }

    /// Bump the version counter (returns the **new** version). Called
    /// by in-place ops: optimizer step, residual fuse, future
    /// in-place norm.
    ///
    /// This is the **anti-pattern 16 enforcement seam**: any code
    /// path that mutates `self.storage()`'s bytes must call this. If
    /// it doesn't, an autograd backward that reads the pre-mutation
    /// tensor will silently use the post-mutation bytes — the
    /// "silent corruption on step 2" the issue warns about.
    pub fn bump_version(&self) -> u64 {
        self.version.fetch_add(1, Ordering::Relaxed) + 1
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
    fn contiguous_emits_copy_counter_only_on_materializing_path() {
        // anti-pattern 2 contract: the fast-path clone must NOT bump
        // the counter; the materializing branch MUST.
        //
        // Serialize via the crate-wide test lock (shared with
        // `profile::tests::*`) so a parallel `emit_contiguous_copy`
        // from those tests can't bump the counter between our reset
        // and our assert. A per-module local mutex (the prior code)
        // doesn't synchronize across modules.
        let _g = crate::profile::counter_test_lock();

        crate::profile::reset_contiguous_copy_count();

        // Fast path: contiguous tensor -> no copy emitted.
        let t = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        let _c = t.contiguous().unwrap();
        assert_eq!(crate::profile::contiguous_copy_count(), 0);

        // Materializing path: transpose then contiguous -> one copy.
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&v, vec![2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        let _c = tt.contiguous().unwrap();
        assert_eq!(crate::profile::contiguous_copy_count(), 1);

        crate::profile::reset_contiguous_copy_count();
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

    #[test]
    fn fresh_tensor_version_is_zero() {
        let t = Tensor::zeros_cpu(vec![2, 3], DType::F32);
        assert_eq!(t.current_version(), 0);
    }

    #[test]
    fn bump_version_returns_new_value() {
        let t = Tensor::zeros_cpu(vec![2, 3], DType::F32);
        assert_eq!(t.bump_version(), 1);
        assert_eq!(t.bump_version(), 2);
        assert_eq!(t.bump_version(), 3);
        assert_eq!(t.current_version(), 3);
    }

    #[test]
    fn clone_shares_version_counter() {
        // Clone shares the Arc<AtomicU64>; a bump on one clone is
        // observed by all clones.
        let t = Tensor::zeros_cpu(vec![2, 3], DType::F32);
        let c1 = t.clone();
        let c2 = t.clone();
        t.bump_version();
        assert_eq!(t.current_version(), 1);
        assert_eq!(c1.current_version(), 1);
        assert_eq!(c2.current_version(), 1);
    }

    #[test]
    fn view_op_shares_version_with_parent() {
        // Views alias storage; they share the version counter.
        let t = Tensor::zeros_cpu(vec![4, 5], DType::F32);
        let v = t.narrow(0, 1, 2).unwrap();
        t.bump_version();
        assert_eq!(v.current_version(), 1);
    }

    #[test]
    fn contiguous_materialized_has_fresh_version() {
        // Materializing contiguous() copies bytes; the new tensor
        // gets a fresh version counter independent of the source.
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&v, vec![2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        // tt shares t's version counter (view op).
        t.bump_version();
        assert_eq!(tt.current_version(), 1);
        // contiguous() on a non-contig view materializes — fresh
        // version starts at 0 independent of source.
        let c = tt.contiguous().unwrap();
        assert_eq!(c.current_version(), 0);
        // Subsequent bumps on the source don't affect the materialized
        // copy.
        t.bump_version();
        assert_eq!(t.current_version(), 2);
        assert_eq!(c.current_version(), 0);
    }

    #[test]
    fn version_handle_is_arc_to_same_atomic() {
        let t = Tensor::zeros_cpu(vec![1], DType::F32);
        let h = t.version_handle();
        // The handle is to the same atomic as the tensor.
        t.bump_version();
        assert_eq!(h.load(Ordering::Relaxed), 1);
        h.fetch_add(10, Ordering::Relaxed);
        assert_eq!(t.current_version(), 11);
    }

    #[test]
    fn unsqueeze_promotes_rank() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let u = t.unsqueeze(0).unwrap();
        assert_eq!(u.shape(), &[1, 3]);
        let u1 = t.unsqueeze(1).unwrap();
        assert_eq!(u1.shape(), &[3, 1]);
    }

    #[test]
    fn unsqueeze_at_end_is_inclusive() {
        // axis = rank is valid (insert at the tail).
        let t = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let u = t.unsqueeze(1).unwrap();
        assert_eq!(u.shape(), &[2, 1]);
    }

    #[test]
    fn unsqueeze_out_of_range_errors() {
        let t = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = t.unsqueeze(5).unwrap_err();
        assert!(e.to_string().contains("unsqueeze"));
    }

    #[test]
    fn squeeze_removes_size_1_axis() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![1, 3]).unwrap();
        let s = t.squeeze(0).unwrap();
        assert_eq!(s.shape(), &[3]);
    }

    #[test]
    fn squeeze_non_size_1_errors() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let e = t.squeeze(0).unwrap_err();
        assert!(e.to_string().contains("squeeze"));
    }

    #[test]
    fn squeeze_unsqueeze_roundtrip() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let u = t.unsqueeze(0).unwrap();
        let s = u.squeeze(0).unwrap();
        assert_eq!(s.shape(), t.shape());
    }

    #[test]
    fn flatten_to_rank_1() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let f = t.flatten().unwrap();
        assert_eq!(f.shape(), &[6]);
    }

    #[test]
    fn flatten_range_collapses_axes() {
        // [2, 3, 4] flatten_range(0, 1) → [6, 4]
        let t = Tensor::zeros_cpu(vec![2, 3, 4], DType::F32);
        let f = t.flatten_range(0, 1).unwrap();
        assert_eq!(f.shape(), &[6, 4]);
    }

    #[test]
    fn flatten_range_collapse_middle() {
        // [2, 3, 4, 5] flatten_range(1, 2) → [2, 12, 5]
        let t = Tensor::zeros_cpu(vec![2, 3, 4, 5], DType::F32);
        let f = t.flatten_range(1, 2).unwrap();
        assert_eq!(f.shape(), &[2, 12, 5]);
    }

    #[test]
    fn flatten_range_invalid_errors() {
        let t = Tensor::zeros_cpu(vec![2, 3], DType::F32);
        let e = t.flatten_range(5, 6).unwrap_err();
        assert!(e.to_string().contains("flatten_range"));
    }

    #[test]
    fn t_swaps_trailing_two_axes() {
        // [2, 3] → [3, 2].
        let t = Tensor::zeros_cpu(vec![2, 3], DType::F32);
        let tt = t.t().unwrap();
        assert_eq!(tt.shape(), &[3, 2]);
    }

    #[test]
    fn t_higher_rank_swaps_last_two() {
        // [2, 3, 4] → [2, 4, 3]
        let t = Tensor::zeros_cpu(vec![2, 3, 4], DType::F32);
        let tt = t.t().unwrap();
        assert_eq!(tt.shape(), &[2, 4, 3]);
    }

    #[test]
    fn t_rank_1_errors() {
        let t = Tensor::zeros_cpu(vec![3], DType::F32);
        let e = t.t().unwrap_err();
        assert!(e.to_string().contains("rank"));
    }

    #[test]
    fn move_axis_forward() {
        // [A, B, C, D]; move 0→3 → [B, C, D, A]
        let t = Tensor::zeros_cpu(vec![2, 3, 4, 5], DType::F32);
        let m = t.move_axis(0, 3).unwrap();
        assert_eq!(m.shape(), &[3, 4, 5, 2]);
    }

    #[test]
    fn move_axis_backward() {
        // [A, B, C, D]; move 3→0 → [D, A, B, C]
        let t = Tensor::zeros_cpu(vec![2, 3, 4, 5], DType::F32);
        let m = t.move_axis(3, 0).unwrap();
        assert_eq!(m.shape(), &[5, 2, 3, 4]);
    }

    #[test]
    fn move_axis_out_of_range_errors() {
        let t = Tensor::zeros_cpu(vec![2, 3], DType::F32);
        let e = t.move_axis(5, 0).unwrap_err();
        assert!(e.to_string().contains("out of range"));
    }
}

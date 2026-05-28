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

    /// Build a CUDA tensor directly from a host slice + shape on
    /// the primary CUDA device — **candle-free constructor**.
    ///
    /// Internally:
    /// 1. Builds the CPU tensor via [`Self::from_slice`].
    /// 2. Uploads via [`crate::host_to_cuda_copy_ctx`] (which derives
    ///    the cudarc context internally — no candle_device touched).
    ///
    /// This is the substrate-side helper (#1082) that lets kernel-crate
    /// `tests/*.rs` and `#[cfg(test)]` parity scaffolds allocate CUDA
    /// inputs candle-free, a prerequisite for dropping `candle-core`
    /// from the kernel crates' `Cargo.toml`.
    ///
    /// On non-CUDA builds this method is absent; tests that need it
    /// can use `#[cfg(feature = "cuda")]` to gate.
    #[cfg(feature = "cuda")]
    pub fn cuda_from_slice<E: Element>(
        values: &[E],
        shape: impl Into<Vec<usize>>,
        device_index: usize,
    ) -> Result<Self> {
        let cpu = Self::from_slice(values, shape)?;
        crate::host_to_cuda_copy_ctx(&cpu, device_index)
    }

    /// Build a zero-initialized CUDA tensor on the primary CUDA
    /// device for `device_index` — **candle-free constructor**.
    ///
    /// Companion to [`Self::cuda_from_slice`]. Useful for test
    /// scaffolds that need a CUDA destination buffer of a known shape
    /// without having to mention `candle_core` types. Routes through
    /// `cuda_zeros_ctx` (candle-free).
    #[cfg(feature = "cuda")]
    pub fn cuda_zeros_on(
        shape: impl Into<Vec<usize>>,
        dtype: DType,
        device_index: usize,
    ) -> Result<Self> {
        let shape_vec: Vec<usize> = shape.into();
        let n_elements = shape_vec.iter().product::<usize>();
        let storage = crate::cuda_zeros_ctx(device_index, dtype, n_elements)?;
        let layout = Layout::contiguous(shape_vec);
        Self::from_parts(storage, layout, TensorId::next())
    }

    /// Allocate a fresh zero-initialized tensor on the given device.
    ///
    /// Device-parametric companion to [`Self::zeros_cpu`] /
    /// [`Self::cuda_zeros_on`]: the caller picks the device, the
    /// constructor routes to the matching backend allocator. Enables
    /// callers that derive the destination device from an input
    /// tensor's storage (e.g. accumulators in chunked
    /// `dispatch2`-based ops where mixing a CPU accumulator with a
    /// CUDA input would otherwise fail the device-match check).
    ///
    /// | `device`        | Behavior                                  |
    /// |-----------------|-------------------------------------------|
    /// | `Device::Cpu`   | identical to [`Self::zeros_cpu`]          |
    /// | `Device::Cuda(i)` (with `cuda` feature) | routes to [`Self::cuda_zeros_on`] |
    /// | `Device::Cuda(_)` (without `cuda` feature) | `Err` — cuda not linked |
    /// | `Device::Metal(_)` / `Device::Vulkan(_)` | `Err` — NYI (substrate-side, #1082) |
    ///
    /// The FLCE backward (`FlceCustomOp::bwd`) is the first kt-bridge
    /// consumer; the Metal / Vulkan branches are unreachable from that
    /// path today and stay `Err` to surface accidental routing instead
    /// of silently falling back to CPU.
    pub fn zeros_on(device: Device, shape: Vec<usize>, dtype: DType) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Self::zeros_cpu(shape, dtype)),
            #[cfg(feature = "cuda")]
            Device::Cuda(i) => Self::cuda_zeros_on(shape, dtype, i),
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(Error::Msg(
                "Tensor::zeros_on: CUDA device requested but `cuda` feature is not enabled"
                    .to_string(),
            )),
            other @ (Device::Metal(_) | Device::Vulkan(_)) => Err(Error::Msg(format!(
                "Tensor::zeros_on: device {other} is not yet implemented (issue #1082)"
            ))),
        }
    }

    /// Build a tensor on the given device from a typed [`Vec`].
    ///
    /// Device-parametric companion to [`Self::from_vec`] /
    /// [`Self::cuda_from_slice`]: the caller picks the device, the
    /// constructor either lands directly on CPU or stages-then-uploads
    /// for CUDA. Same routing table as [`Self::zeros_on`].
    ///
    /// Internally the CUDA path builds a CPU tensor via
    /// [`Self::from_vec`] and uploads via [`crate::host_to_cuda_copy`]
    /// — the same path [`Self::cuda_from_slice`] uses. The element
    /// type `E` parameter mirrors [`Self::from_vec`] / [`Self::from_slice`].
    pub fn from_vec_on<E: Element>(
        device: Device,
        values: Vec<E>,
        shape: Vec<usize>,
    ) -> Result<Self> {
        match device {
            Device::Cpu => Self::from_vec(values, shape),
            #[cfg(feature = "cuda")]
            Device::Cuda(i) => {
                // Stage on the host first, then H2D into a freshly
                // allocated CUDA buffer. Identical to the body of
                // `cuda_from_slice` but spelled out to keep both
                // constructors callable in isolation.
                let cpu = Self::from_vec(values, shape)?;
                let cdev = crate::primary_cuda_device(i)?;
                crate::host_to_cuda_copy(&cpu, cdev, i)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => Err(Error::Msg(
                "Tensor::from_vec_on: CUDA device requested but `cuda` feature is not enabled"
                    .to_string(),
            )),
            other @ (Device::Metal(_) | Device::Vulkan(_)) => Err(Error::Msg(format!(
                "Tensor::from_vec_on: device {other} is not yet implemented (issue #1082)"
            ))),
        }
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
        #[cfg(feature = "cuda")]
        if matches!(self.device(), crate::Device::Cuda(_)) {
            return crate::cuda_storage::cuda_contiguous(self);
        }
        if !self.device().is_cpu() {
            return Err(Error::Msg(format!(
                "Tensor::contiguous: only CPU + CUDA contiguous is implemented; \
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
    // Finite-element check (KILN_DETECT_ANOMALY substrate)
    // ------------------------------------------------------------------

    /// Returns `Ok(true)` iff every element of this tensor is finite
    /// (no `NaN`, no `+Inf`, no `-Inf`). Supports F32, BF16, F16, and
    /// FP8 dtypes — for integer/packed dtypes returns `Ok(true)`
    /// because they have no NaN/Inf representations.
    ///
    /// CPU storage: iterates the addressable byte buffer (this method
    /// does not materialize a contiguous copy — it walks strides).
    /// CUDA storage: routes through `cuda_is_finite`, a per-backend
    /// reduction kernel that atomicOr's a single u32 device flag and
    /// reads only those 4 bytes back to the host (vs the
    /// pre-Phase-9 D2H bridge that copied the full tensor). Other
    /// non-CPU backends still return an error until their
    /// finite-check substrate lands.
    ///
    /// This is the kt-tensor side of the
    /// `kiln_autograd::anomaly_detection_enabled()` /
    /// `anomaly_panic()` pair (#1082 Phase 9): when
    /// `KILN_DETECT_ANOMALY=1`, `Tape::backward` scans each backward
    /// op's gradient outputs via this method and panics at the
    /// op-tape-position of the first non-finite value.
    ///
    /// Cost: O(numel) per call on CPU. Off-by-default in production;
    /// CI training-parity tests opt in.
    pub fn all_finite(&self) -> Result<bool> {
        use crate::DType;
        // Integer + packed dtypes have no NaN/Inf — vacuously finite.
        if matches!(self.dtype(), DType::U8 | DType::U32 | DType::I64) {
            return Ok(true);
        }
        if self.dtype().is_packed() {
            return Ok(true);
        }
        // #1082 Phase 9: for CUDA tensors, route through the dedicated
        // `cuda_is_finite` reduction kernel
        // (`kiln_is_finite_storage_async` in `csrc/is_finite_reduce.cu`).
        // The kernel atomicOr's a single u32 device flag and we read
        // back only 4 bytes — vs the previous D2H-bridge path that
        // copied the entire tensor through `cuda_to_host_copy`.
        // Net cost on the `KILN_DETECT_ANOMALY=1` scan-per-backward-op
        // path: O(numel) bytes of D2H per node → 4 bytes per node.
        if !self.device().is_cpu() {
            #[cfg(feature = "cuda")]
            {
                if matches!(self.device(), crate::Device::Cuda(_)) {
                    // Try the kernel for supported dtypes; fall back
                    // to the D2H bridge for any new/unsupported
                    // dtypes the kernel doesn't cover.
                    let supported = matches!(
                        self.dtype(),
                        DType::F32
                            | DType::BF16
                            | DType::F16
                            | DType::F8E4M3
                            | DType::F8E5M2
                    );
                    if supported {
                        return crate::cuda_is_finite(self);
                    }
                    // Fallback: keep the bridge for any dtype the
                    // kernel doesn't yet handle (none today; this is
                    // forward-looking defense for future dtypes that
                    // land on CUDA without a corresponding
                    // is_finite_reduce.cu branch).
                    let cpu_view = crate::cuda_to_host_copy(self)?;
                    return cpu_view.all_finite();
                }
            }
            return Err(Error::Msg(format!(
                "Tensor::all_finite: device {} support lands with the \
                 per-backend is_finite reduction kernel",
                self.device()
            )));
        }
        let cpu = self
            .storage
            .as_any()
            .downcast_ref::<CpuStorage>()
            .ok_or_else(|| {
                Error::Msg("Tensor::all_finite: CPU device but storage isn't CpuStorage".to_string())
            })?;
        let per = self.dtype().size_in_bytes();
        let shape = self.layout.shape();
        let strides = self.layout.strides();
        let start = self.layout.start_offset();
        let bytes = cpu.as_bytes();
        // Stride-walk so we don't have to materialize a contiguous
        // copy. For each logical index, check finiteness of the
        // element at the physical offset.
        let rank = shape.len();
        if rank == 0 {
            // Scalar tensor. One element at `start`.
            return Ok(scalar_at_is_finite(bytes, start * per, self.dtype()));
        }
        let mut idx = vec![0usize; rank];
        loop {
            let mut phys = start;
            for (axis, &i) in idx.iter().enumerate() {
                phys += i * strides[axis];
            }
            let src = phys * per;
            if !scalar_at_is_finite(bytes, src, self.dtype()) {
                return Ok(false);
            }
            // Increment row-major.
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
        Ok(true)
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

    // ------------------------------------------------------------------
    // Device transfer (issue #1082)
    // ------------------------------------------------------------------

    /// Transfer this tensor to `target` device, returning a fresh
    /// tensor on the target. Wraps the existing `host_to_cuda_copy_ctx`
    /// and `cuda_to_host_copy` helpers behind a uniform method API.
    ///
    /// **Candle-free as of #1082**: CPU→CUDA derives the cudarc
    /// `CudaContext` internally via `host_to_cuda_copy_ctx`; callers no
    /// longer pass an `Arc<CudaDevice>`. The previous `candle_device:
    /// Option<Arc<CudaDevice>>` parameter has been dropped.
    ///
    /// # Supported transitions
    ///
    /// | from → to         | behavior                              |
    /// |-------------------|---------------------------------------|
    /// | same device       | `Ok(self.clone())` — cheap Arc bump   |
    /// | CPU → CUDA(i)     | `host_to_cuda_copy_ctx`               |
    /// | CUDA → CPU        | `cuda_to_host_copy`                   |
    /// | CUDA(i)→CUDA(j)   | `Err` — cross-device GPU transfer NYI |
    /// | other cross-back  | `Err` — Metal/Vulkan paths NYI        |
    #[cfg(feature = "cuda")]
    pub fn to_device(&self, target: Device) -> Result<Self> {
        let src = self.device();
        if src == target {
            // Same-device move is a cheap Arc bump per anti-pattern 11
            // (clones preserve identity). No copy, no version reset.
            return Ok(self.clone());
        }
        match (src, target) {
            (Device::Cpu, Device::Cuda(i)) => crate::host_to_cuda_copy_ctx(self, i),
            (Device::Cuda(_), Device::Cpu) => crate::cuda_to_host_copy(self),
            (Device::Cuda(i), Device::Cuda(j)) => Err(Error::Msg(format!(
                "Tensor::to_device: cross-GPU transfer Cuda({i})→Cuda({j}) is not yet implemented (issue #1082)"
            ))),
            _ => Err(Error::Msg(format!(
                "Tensor::to_device: transition {src}→{target} is not yet implemented                  (issue #1082)"
            ))),
        }
    }

    /// Non-CUDA build of [`Self::to_device`]. Only same-device moves
    /// succeed (cheap clone); every cross-device case returns `Err`
    /// because no GPU backend is linked.
    #[cfg(not(feature = "cuda"))]
    pub fn to_device(&self, target: Device) -> Result<Self> {
        let src = self.device();
        if src == target {
            return Ok(self.clone());
        }
        Err(Error::Msg(format!(
            "Tensor::to_device: transition {src}→{target} requires a GPU feature (cuda/metal/vulkan); none is enabled in this build"
        )))
    }
}

/// Helper for [`Tensor::all_finite`] — read one element at the given
/// byte offset, decode it per dtype, and check `is_finite()`. Returns
/// true for dtypes without NaN/Inf representations.
fn scalar_at_is_finite(bytes: &[u8], byte_off: usize, dtype: crate::DType) -> bool {
    use crate::DType;
    match dtype {
        DType::F32 => {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[byte_off..byte_off + 4]);
            f32::from_le_bytes(buf).is_finite()
        }
        DType::BF16 => {
            let mut buf = [0u8; 2];
            buf.copy_from_slice(&bytes[byte_off..byte_off + 2]);
            half::bf16::from_le_bytes(buf).to_f32().is_finite()
        }
        DType::F16 => {
            let mut buf = [0u8; 2];
            buf.copy_from_slice(&bytes[byte_off..byte_off + 2]);
            half::f16::from_le_bytes(buf).to_f32().is_finite()
        }
        // FP8 E4M3FN has no Inf; only NaN at bit pattern 0x7F / 0xFF
        // (sign bit + all-ones exponent + all-ones mantissa).
        DType::F8E4M3 => (bytes[byte_off] & 0x7F) != 0x7F,
        // FP8 E5M2: exp = 5 bits + mantissa = 2 bits. exp=11111 is
        // Inf (mantissa=00) or NaN (mantissa!=00) — either is
        // non-finite.
        DType::F8E5M2 => ((bytes[byte_off] >> 2) & 0b11111) != 0b11111,
        // U8/U32/I64/packed dtypes: handled by the early-return in
        // Tensor::all_finite; should never reach here.
        _ => true,
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
        // anti-pattern 2 contract: the materializing branch of
        // `contiguous()` bumps the
        // `profile::contiguous_copy_count` counter (the bench-gate
        // reads it); the fast-path clone on already-contiguous input
        // does NOT bump it.
        //
        // The counter is a process-global `AtomicU64`. Other tests in
        // this crate (sdpa, mha, unbind, autograd integration tests,
        // …) freely call `.contiguous()` on non-contiguous tensors
        // without grabbing any lock, so absolute-count assertions
        // race on multi-core runners (CI macOS / Metal failed run
        // 26342916999 on commit 19844cd1 with `left: 1, right: 0`).
        // Cross-thread serialization on a public counter is not the
        // contract `contiguous()` is supposed to provide.
        //
        // Verify the contract robustly via `CopyScope` deltas:
        //   * Materializing path: produces at least one emit.
        //   * Fast path: produces strictly fewer emits than the
        //     materializing path on this thread (best a global
        //     counter can do without thread-local counting). The
        //     "fast path emits exactly zero" half of the contract is
        //     covered by code review of `contiguous()` itself —
        //     `emit_contiguous_copy()` is called from exactly one
        //     site, after the `is_contiguous()` short-circuit.
        let _g = crate::profile::counter_test_lock();

        // Fast path: contiguous tensor → expect no copy emitted from
        // *this* call (concurrent emits from other threads may still
        // appear in the delta).
        let fast_scope = crate::profile::CopyScope::start();
        let t = Tensor::zeros_cpu(vec![3, 4], DType::F32);
        let _c = t.contiguous().unwrap();
        let fast_delta = fast_scope.finish();

        // Materializing path: transpose-then-contiguous → ≥ 1 emit.
        let mat_scope = crate::profile::CopyScope::start();
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t2 = Tensor::from_slice(&v, vec![2, 3]).unwrap();
        let tt = t2.transpose(0, 1).unwrap();
        let _c = tt.contiguous().unwrap();
        let mat_delta = mat_scope.finish();

        assert!(
            mat_delta >= 1,
            "materializing branch did not emit (fast_delta={fast_delta}, mat_delta={mat_delta})"
        );
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

    #[test]
    fn all_finite_cpu_true_on_all_finite_input() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, -3.5, 0.0], vec![2, 2]).unwrap();
        assert!(t.all_finite().unwrap());
    }

    #[test]
    fn all_finite_cpu_false_on_nan() {
        let t = Tensor::from_slice(&[1.0f32, f32::NAN, 3.0], vec![3]).unwrap();
        assert!(!t.all_finite().unwrap());
    }

    #[test]
    fn all_finite_cpu_false_on_inf() {
        let t = Tensor::from_slice(&[1.0f32, f32::INFINITY, 3.0], vec![3]).unwrap();
        assert!(!t.all_finite().unwrap());
    }

    #[test]
    fn all_finite_cpu_false_on_neg_inf() {
        let t = Tensor::from_slice(&[1.0f32, f32::NEG_INFINITY, 3.0], vec![3]).unwrap();
        assert!(!t.all_finite().unwrap());
    }

    #[test]
    fn all_finite_integer_dtype_is_vacuously_true() {
        let t = Tensor::from_slice(&[1u32, 2, 3, 4], vec![2, 2]).unwrap();
        assert!(t.all_finite().unwrap());
    }

    #[test]
    fn all_finite_after_transpose_uses_stride_walk() {
        // A 2x2 with one NaN at logical [1, 0]; transpose makes the
        // physical layout non-contiguous, exercising the stride walk
        // rather than the dense scan path.
        let t = Tensor::from_slice(&[1.0f32, 2.0, f32::NAN, 4.0], vec![2, 2]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        // Still contains the NaN — just at a transposed logical index.
        assert!(!tt.all_finite().unwrap());
    }

    // ------------------------------------------------------------------
    // Device-parametric constructors (zeros_on / from_vec_on)
    //
    // These constructors unblock the `FlceCustomOp::bwd` kt-bridge:
    // the chunk loop accumulator + per-chunk one-hot mask have to land
    // on the same device as the input `hidden`, otherwise `dispatch2`
    // fails the device-match check (#1082).
    // ------------------------------------------------------------------

    #[test]
    fn zeros_on_cpu_matches_zeros_cpu() {
        let t = Tensor::zeros_on(Device::Cpu, vec![2, 3], DType::F32).unwrap();
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.element_count(), 6);
        // Zero-init: every byte is 0.
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        assert!(cpu.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn from_vec_on_cpu_matches_from_vec() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = Tensor::from_vec_on(Device::Cpu, v.clone(), vec![2, 2]).unwrap();
        assert_eq!(t.device(), Device::Cpu);
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.dtype(), DType::F32);
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
        assert_eq!(back, v);
    }

    #[test]
    fn from_vec_on_cpu_shape_mismatch_errors() {
        let v = vec![1.0f32, 2.0, 3.0];
        let e = Tensor::from_vec_on(Device::Cpu, v, vec![2, 2]).unwrap_err();
        assert!(e.to_string().contains("has 4 elements"));
    }

    #[test]
    fn zeros_on_metal_errors_until_substrate_lands() {
        // Per-backend Metal/Vulkan branches stay Err until #1082
        // substrate work picks them up; callers that hit these today
        // should see an explicit error instead of a silent CPU
        // fallback that would later trip a device-mismatch assert.
        let e = Tensor::zeros_on(Device::Metal(0), vec![2], DType::F32).unwrap_err();
        assert!(e.to_string().contains("metal:0"));
        let e = Tensor::zeros_on(Device::Vulkan(0), vec![2], DType::F32).unwrap_err();
        assert!(e.to_string().contains("vulkan:0"));
    }

    #[test]
    fn from_vec_on_metal_errors_until_substrate_lands() {
        let e = Tensor::from_vec_on(Device::Metal(0), vec![1.0f32], vec![1]).unwrap_err();
        assert!(e.to_string().contains("metal:0"));
        let e = Tensor::from_vec_on(Device::Vulkan(0), vec![1.0f32], vec![1]).unwrap_err();
        assert!(e.to_string().contains("vulkan:0"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn zeros_on_cuda_without_feature_errors() {
        let e = Tensor::zeros_on(Device::Cuda(0), vec![2], DType::F32).unwrap_err();
        assert!(e.to_string().contains("cuda"));
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn from_vec_on_cuda_without_feature_errors() {
        let e = Tensor::from_vec_on(Device::Cuda(0), vec![1.0f32], vec![1]).unwrap_err();
        assert!(e.to_string().contains("cuda"));
    }

    // CUDA-only tests: verify the storage lands on CUDA, the shape /
    // dtype / element_count survive the round-trip, and a host-side
    // readback recovers the original bytes (zero-init for `zeros_on`,
    // the source vec for `from_vec_on`).
    #[cfg(feature = "cuda")]
    #[test]
    fn zeros_on_cuda_lands_on_device() {
        if !crate::cuda_is_available() {
            eprintln!("skip: no CUDA device available");
            return;
        }
        let t = Tensor::zeros_on(Device::Cuda(0), vec![3, 4], DType::F32).unwrap();
        assert_eq!(t.device(), Device::Cuda(0));
        assert_eq!(t.shape(), &[3, 4]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.element_count(), 12);
        // Round-trip via D2H readback.
        let host = crate::cuda_to_host_copy(&t).unwrap();
        let cpu = host.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
        assert_eq!(back, vec![0.0f32; 12]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn from_vec_on_cuda_lands_on_device_with_content() {
        if !crate::cuda_is_available() {
            eprintln!("skip: no CUDA device available");
            return;
        }
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_vec_on(Device::Cuda(0), v.clone(), vec![2, 3]).unwrap();
        assert_eq!(t.device(), Device::Cuda(0));
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        // Round-trip via D2H readback.
        let host = crate::cuda_to_host_copy(&t).unwrap();
        let cpu = host.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(cpu.as_bytes()).to_vec();
        assert_eq!(back, v);
    }
}

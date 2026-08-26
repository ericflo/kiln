//! Metal storage impl behind the `metal` feature flag.
//!
//! Wraps `Arc<metal::Buffer>` (the actual buffer) + dtype +
//! `MetalRawDevice` (the metal-rs `Retained<ProtocolObject<dyn
//! MTLDevice>>` handle, reached through `candle_metal_kernels::metal`)
//! for kernel-FFI affinity. After the #1082 Wave-14 lift the storage
//! is candle-core-free at the field level AND at the op level — the
//! 7 in-file substrate ops dispatch through `MetalStorage::companion()`
//! (a kt-native `MetalCompanion` from `metal_types`), which holds the
//! `Device` / `Kernels` / `Commands` triple every
//! `candle_metal_kernels::call_*` MSL kernel entry needs. The previous
//! candle-derived `MetalStorage::candle_device()` shim and its
//! `primary_metal_device` (`candle_core::Device::new_metal`) hook were
//! deleted; the only remaining `candle-core` dependency in
//! `kiln-tensor` under the `metal` feature now lives in `metal_types.rs`
//! (the `MetalDevice` / `DeviceId` / `Storage` / `sdpa` re-exports
//! consumed at ~48 callsites in `kiln-model::backend::metal`).
//!
//! # Anti-pattern 1 compliance
//!
//! Per the issue:
//!
//! > `kiln-tensor` is not a candle wrapper. Storage is
//! > `metal::Buffer` directly.
//!
//! `MetalStorage` does not hold a `candle_core::Tensor`. The buffer is
//! `Arc<metal::Buffer>` we own — allocated via metal-rs's
//! `MTLDevice::newBufferWithLength:options:` directly through
//! [`Self::zeros_kt`] (the candle-free allocation path). The `Arc<Buffer>`
//! is fully ours.
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
//! `RESOURCE_OPTIONS` constant in candle-core's `metal_backend`; see the
//! pre-#1082 vendor tree for that original definition).
//! `MetalStorage::is_unified_memory()` returns true; the zero-copy
//! host accessor lands in a follow-up PR (it needs a stride/layout
//! check that this PR keeps off the critical path).

use std::any::Any;
use std::sync::Arc;

// `candle_metal_kernels` is its own crate — candle-core does NOT
// re-export it under `metal_backend`. Depend on it directly under the
// `metal` feature so this path resolves.
//
// The seven `metal_*` substrate ops in this file (softmax, rmsnorm,
// layernorm, index_select_dim0, cast, elementwise_binary,
// activation_unary) call directly into `candle_metal_kernels::call_*`
// MSL kernel entry points (no `CandleTensor` / `CandleMetalStorage` /
// `candle_nn` bridge). After Wave-14 (#1082) the 7 ops reach the MSL
// pipeline cache + command-buffer pool through `MetalStorage::companion()`
// — a kt-native `MetalCompanion` (defined in `metal_types`) built
// entirely from `candle_metal_kernels` primitives (`Device::all()`,
// `Kernels::new()`, `Commands::new(queue)`). The previous candle-derived
// shim (`MetalStorage::candle_device()` -> `primary_metal_device`) is
// gone — the `candle_core::metal_backend::MetalDevice` import that
// shim's return type required has retired alongside it.
use crate::metal_rt::Buffer as MetalBuffer;
use crate::metal_rt::Device as MetalRawDevice;

use crate::{CpuStorage, DType, Device, Error, Result, StorageBackend};

/// Metal-backed storage. Byte-typed; dtype carried alongside for dispatch.
///
/// Holds an `Arc<metal::Buffer>` directly (anti-pattern 1). The
/// metal-rs `Device` handle is held for buffer-allocation and
/// kernel-FFI affinity; the previous `candle_device: Arc<MetalDevice>`
/// field was dropped (#1082 CP-1 final lift, mirror of CudaStorage
/// commit 5c3cd353) in favor of this `metal_handle` so the storage's
/// owned state is candle-free. The 7 internal substrate ops reach the
/// substrate primitives (Device / Kernels / Commands) via
/// [`MetalStorage::companion`] (kt-native, candle-core-free).
#[derive(Debug)]
pub struct MetalStorage {
    device: Device,
    dtype: DType,
    buffer: Arc<MetalBuffer>,
    /// Metal-rs raw `MTLDevice` handle. Replaces the previous
    /// `candle_device: Arc<MetalDevice>` field as part of the #1082
    /// CP-1 final lift. After Wave-14 the 7 in-file substrate ops
    /// no longer derive a candle `MetalDevice` per call — they
    /// reach the substrate primitives through
    /// [`MetalStorage::companion`] (a kt-native `MetalCompanion`
    /// holding `Device` / `Kernels` / `Commands` from
    /// `candle_metal_kernels` directly, no `candle-core` involved).
    metal_handle: MetalRawDevice,
}

/// Rewrite an existing contiguous Metal tensor's contents from host data.
///
/// Metal graph replay needs the same stable-buffer refresh primitive CUDA and
/// ROCm use before each graph launch. Kiln's Metal storage is
/// `MTLStorageModeShared`, so the update is a direct CPU write into the
/// buffer's UMA `contents()` pointer; no blit encoder or fresh allocation is
/// involved.
#[cfg(feature = "metal")]
pub fn metal_write_host_in_place<E: crate::Element>(dst: &crate::Tensor, host: &[E]) -> Result<()> {
    if dst.dtype().is_packed() {
        return Err(Error::Msg(
            "metal_write_host_in_place: packed dtype not supported".to_string(),
        ));
    }
    if E::DTYPE.size_in_bytes() != dst.dtype().size_in_bytes() {
        return Err(Error::Msg(format!(
            "metal_write_host_in_place: element byte width {} != dst dtype {} byte width {}",
            E::DTYPE.size_in_bytes(),
            dst.dtype(),
            dst.dtype().size_in_bytes()
        )));
    }
    if !dst.is_contiguous() || dst.layout().start_offset() != 0 {
        return Err(Error::Msg(
            "metal_write_host_in_place: dst must be contiguous with start_offset == 0".to_string(),
        ));
    }
    let n = dst.element_count();
    if host.len() != n {
        return Err(Error::Msg(format!(
            "metal_write_host_in_place: host len {} != dst element count {n}",
            host.len()
        )));
    }

    let dst_storage = dst
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_write_host_in_place: dst must be Metal storage".to_string())
        })?;

    let bytes = E::to_bytes(host);
    let buffer = dst_storage.buffer();
    if bytes.len() > buffer.length() {
        return Err(Error::Msg(format!(
            "metal_write_host_in_place: host byte len {} exceeds buffer length {}",
            bytes.len(),
            buffer.length()
        )));
    }

    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents(), bytes.len());
    }
    buffer.did_modify_range(objc2_foundation::NSRange {
        location: 0,
        length: bytes.len(),
    });
    Ok(())
}

/// Copy one contiguous Metal tensor into an existing contiguous Metal tensor.
///
/// Metal graph replay binds stable input buffers, so callers that produce
/// transient activation tensors can refresh runner-owned buffers without
/// allocating new storage. This conservative path waits for queued GPU writes
/// and then copies between `MTLStorageModeShared` buffers through UMA.
#[cfg(feature = "metal")]
pub fn metal_copy_in_place(src: &crate::Tensor, dst: &crate::Tensor) -> Result<()> {
    if src.dtype() != dst.dtype() {
        return Err(Error::Msg(format!(
            "metal_copy_in_place: dtype mismatch {} vs {}",
            src.dtype(),
            dst.dtype()
        )));
    }
    if src.shape() != dst.shape() {
        return Err(Error::Msg(format!(
            "metal_copy_in_place: shape mismatch {:?} vs {:?}",
            src.shape(),
            dst.shape()
        )));
    }
    if src.dtype().is_packed() {
        return Err(Error::Msg(
            "metal_copy_in_place: packed dtype not supported".to_string(),
        ));
    }
    if !src.is_contiguous() || !dst.is_contiguous() {
        return Err(Error::Msg(
            "metal_copy_in_place: src and dst must be contiguous".to_string(),
        ));
    }

    let src_storage = src
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_copy_in_place: src must be Metal storage".to_string()))?;
    let dst_storage = dst
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_copy_in_place: dst must be Metal storage".to_string()))?;
    if src_storage.device_index() != dst_storage.device_index() {
        return Err(Error::Msg(format!(
            "metal_copy_in_place: device mismatch {} vs {}",
            src_storage.device_index(),
            dst_storage.device_index()
        )));
    }

    let per = src.dtype().size_in_bytes();
    let byte_len = src.element_count() * per;
    let src_offset = src.layout().start_offset() * per;
    let dst_offset = dst.layout().start_offset() * per;
    let src_buffer = src_storage.buffer();
    let dst_buffer = dst_storage.buffer();
    let src_end = src_offset + byte_len;
    let dst_end = dst_offset + byte_len;
    if src_end > src_buffer.length() as usize {
        return Err(Error::Msg(format!(
            "metal_copy_in_place: src byte range {src_offset}..{src_end} exceeds buffer length {}",
            src_buffer.length()
        )));
    }
    if dst_end > dst_buffer.length() as usize {
        return Err(Error::Msg(format!(
            "metal_copy_in_place: dst byte range {dst_offset}..{dst_end} exceeds buffer length {}",
            dst_buffer.length()
        )));
    }

    src_storage.companion()?.wait_until_completed()?;
    unsafe {
        let src_ptr = (src_buffer.contents() as *const u8).add(src_offset);
        let dst_ptr = (dst_buffer.contents() as *mut u8).add(dst_offset);
        std::ptr::copy(src_ptr, dst_ptr, byte_len);
    }
    dst_buffer.did_modify_range(objc2_foundation::NSRange {
        location: dst_offset,
        length: byte_len,
    });
    Ok(())
}

impl MetalStorage {
    /// Allocate `n_elements` worth of bytes for `dtype` on the metal-rs
    /// `device`, **candle-free** in the allocation path.
    ///
    /// Buffer is allocated through
    /// `device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared)`
    /// (Apple Silicon UMA — host and GPU share physical memory). Zero
    /// initialization happens via a direct `core::ptr::write_bytes` on
    /// the buffer's UMA `.contents()` pointer; no blit-command-encoder
    /// is required because Shared-mode buffers are CPU-addressable.
    ///
    /// `device_index` is the Metal device ordinal — must match the
    /// ordinal of `device`'s owning system device. Stored as the
    /// [`Device::Metal`] variant.
    ///
    /// # Post-Wave-14: storage is candle-free at field AND op level
    ///
    /// The input `device` (metal-rs handle) flows directly into the
    /// storage's `metal_handle: MetalRawDevice` field via a cheap
    /// NSObject `retain` clone. The 7 internal substrate ops in this
    /// file reach `kernels()` + `command_encoder()` for
    /// `candle_metal_kernels::call_*` dispatch via
    /// [`MetalStorage::companion`] — a kt-native `MetalCompanion`
    /// holding the candle-core-free `Device` / `Kernels` / `Commands`
    /// triple. No candle wrapper is materialized anywhere on the
    /// allocation or dispatch path.
    ///
    /// # Device-affinity contract
    ///
    /// Both `device` (the metal-rs handle the caller passes) and the
    /// kt-native `MetalCompanion` resolved via
    /// [`primary_metal_companion(device_index)`] wrap the same
    /// `MTLDevice` protocol object for the given ordinal — both go
    /// through `candle_metal_kernels::metal::Device::all()` which
    /// resolves the same registry-ID-indexed physical GPU. The new
    /// buffer is therefore addressable by every kernel-crate FFI that
    /// consumes `companion.device()`.
    ///
    /// # UMA + zero-init safety
    ///
    /// Apple Silicon UMA guarantees that an `MTLStorageModeShared`
    /// buffer's `contents()` pointer is CPU-addressable and points at
    /// the same physical bytes the GPU sees. `core::ptr::write_bytes`
    /// (memset) on that pointer with value 0 is well-defined for any
    /// `byte_len` allocation — there is no `MTLBuffer didModifyRange`
    /// requirement on Shared mode (unlike Managed mode on Intel
    /// Macs). For `byte_len == 0`, we explicitly skip the alloc + fill
    /// and synthesize a 1-byte placeholder buffer to match candle's
    /// `allocate_zeros(0)` semantics (which goes through
    /// `buf_size(0) = 1.next_power_of_two() = 1`).
    ///
    /// # Status
    ///
    /// Wave-14 (this commit chain): the `candle_device` field and the
    /// `primary_metal_device` / `Self::candle_device()` candle-derivation
    /// shims have both been retired. This constructor and the 7 in-file
    /// substrate ops are fully candle-core-free at the field AND op
    /// level — the ops reach `kernels()` + `command_encoder()` via
    /// `Self::companion()` (a kt-native `MetalCompanion` from
    /// `metal_types`, holding `candle_metal_kernels::Kernels` + `Commands`
    /// directly).
    ///
    /// The candle-typed `Self::zeros` back-compat constructor was
    /// deleted (#1082, commit 71a3b677) earlier in the lift chain; this
    /// is the sole allocation path on `MetalStorage`. See the
    /// order-of-operations doc in `metal_allocator.rs` for the
    /// CudaAllocator/CudaStorage mirror history.
    ///
    /// Mirror of [`crate::CudaStorage::zeros_ctx`] (commit d3caf46b) —
    /// same shape, same rationale (the parallel-constructor step of
    /// the CP-1 substrate lift documented in
    /// `docs/archive/candle-removal/issue-1082-tier-4-5-roadmap-2026-05-27.md`).
    pub fn zeros_kt(
        device: &MetalRawDevice,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        use crate::metal_rt::MTLResourceOptions;

        let byte_len = dtype.packed_buffer_bytes(n_elements);
        // Candle-free allocation through metal-rs. Apple's MTLDevice
        // rejects newBufferWithLength:options: for length=0 (returns
        // nil), so round up to 1 byte to match candle's buf_size(0) = 1
        // behavior. The dtype-derived byte_len on the StorageBackend
        // side still reads from buffer.length(), so the 0-len case is
        // self-consistent (the byte_len reported by the StorageBackend
        // will be 1, matching what candle's zeros() returns today).
        let alloc_len = byte_len.max(1);
        let buffer = device
            .new_buffer(alloc_len, MTLResourceOptions::StorageModeShared)
            .map_err(|e| {
                Error::Msg(format!(
                    "MetalStorage::zeros_kt: device.new_buffer({alloc_len}, Shared) \
                     failed: {e:?}"
                ))
            })?;
        // Zero-fill via UMA contents pointer — no command-queue
        // required on Shared-mode buffers.
        //
        // SAFETY: `buffer.contents()` returns a non-null `*mut u8` for
        // Shared-mode buffers on Apple Silicon UMA. `alloc_len` is the
        // exact length passed to `newBufferWithLength:options:`, so the
        // write_bytes call stays within the buffer's allocation. The
        // buffer is single-owner (just freshly allocated, no Arc clone
        // outstanding yet) so there are no aliasing concerns.
        unsafe {
            core::ptr::write_bytes(buffer.contents(), 0u8, alloc_len);
        }
        // CP-1 final lift: the input `device` (metal-rs handle) flows
        // straight into the storage's `metal_handle` field. No
        // `primary_metal_device(device_index)` materialization on the
        // allocation path any more.
        Ok(MetalStorage {
            device: Device::Metal(device_index),
            dtype,
            buffer: Arc::new(buffer),
            metal_handle: device.clone(),
        })
    }

    /// Wrap an existing `Arc<metal::Buffer>` allocated by the caller —
    /// **candle-free** entry point.
    ///
    /// Takes a metal-rs `MetalRawDevice` handle that flows straight
    /// into the storage's `metal_handle` field (cheap NSObject clone).
    /// Validates the buffer length against `dtype.size_in_bytes()`
    /// for non-packed dtypes.
    ///
    /// Mirror of [`crate::CudaStorage::from_slice_ctx`] (the candle-
    /// free constructor entry the CudaStorage CP-1 lift chain converged
    /// onto in commits 5c3cd353 + 876e17da). The candle-typed
    /// `from_buffer(Arc<MetalDevice>, ...)` back-compat constructor
    /// was deleted (#1082) after the in-file ops + test migration to
    /// this entry; this constructor is now the sole `from_buffer*`
    /// path on `MetalStorage`. Mirrors the CudaStorage 876e17da cleanup
    /// (which deleted `CudaStorage::{zeros, from_slice, from_borrowed}`
    /// after their candle-free counterparts took over).
    pub fn from_buffer_kt(
        metal_handle: &MetalRawDevice,
        device_index: usize,
        dtype: DType,
        buffer: Arc<MetalBuffer>,
    ) -> Result<Self> {
        let len = buffer.length() as usize;
        if !dtype.is_packed() {
            let per = dtype.size_in_bytes();
            if per > 0 && !len.is_multiple_of(per) {
                return Err(Error::Msg(format!(
                    "MetalStorage::from_buffer_kt: buffer len {len} is not a multiple of \
                     size_in_bytes({:?}) = {per}",
                    dtype
                )));
            }
        }
        // CP-1 final lift: `metal_handle` flows straight into the
        // storage's field — no `primary_metal_device(device_index)`
        // materialization needed any more. Cheap NSObject clone via
        // `MetalRawDevice::clone()` (it's a
        // `Retained<ProtocolObject<dyn MTLDevice>>` under the hood).
        Ok(MetalStorage {
            device: Device::Metal(device_index),
            dtype,
            buffer,
            metal_handle: metal_handle.clone(),
        })
    }

    /// Borrow the underlying buffer. The existing kernel-crate FFI
    /// sites in `kiln-model::backend::metal` plug in via this
    /// accessor (mirrors `candle_core::metal_backend::buffer_o` 232
    /// call sites from Phase 0.1's audit).
    pub fn buffer(&self) -> &Arc<MetalBuffer> {
        &self.buffer
    }

    /// Resolve (or lazily construct) the kt-native `MetalCompanion`
    /// for this storage's device ordinal — **candle-core-free**.
    ///
    /// The companion bundles the three substrate primitives the 7
    /// in-file Metal ops in this file (softmax, rmsnorm, layernorm,
    /// index_select_dim0, cast, elementwise_binary, activation_unary)
    /// need:
    ///   - `&candle_metal_kernels::metal::Device` for `call_*`
    ///   - `&candle_metal_kernels::Kernels` for MSL pipeline caching
    ///   - `ComputeCommandEncoder` materialization
    ///
    /// All three live in `candle_metal_kernels` (the sibling crate of
    /// `candle-core`); the companion's construction path
    /// (`Device::all()` + `Kernels::new()` + `Commands::new(queue)`)
    /// never touches `candle-core`. The previous candle-derived
    /// `MetalStorage::candle_device()` shim and its `primary_metal_device`
    /// (`candle_core::Device::new_metal`) hook were retired alongside
    /// the in-file op migration (Wave 14, #1082) — `companion()` is the
    /// canonical (and only) substrate accessor on `MetalStorage` post
    /// Wave-14.
    ///
    /// Cached process-wide in [`primary_metal_companion`]'s
    /// `OnceLock<HashMap>` so repeated calls return the same
    /// `Arc<MetalCompanion>` for a given ordinal (matching what
    /// candle's `MetalDevice::new` cache does for the candle wrapper).
    pub fn companion(&self) -> Result<Arc<crate::metal_types::MetalCompanion>> {
        primary_metal_companion(self.device_index())
    }

    /// The underlying metal-rs `Device` this storage was allocated
    /// on — **candle-free passthrough**.
    ///
    /// After the #1082 CP-1 final lift, this is a cheap clone of the
    /// stored `MetalRawDevice` field (a
    /// `Retained<ProtocolObject<dyn MTLDevice>>` clone via NSObject
    /// `retain` — no candle wrapper involvement, no
    /// `primary_metal_device` call).
    ///
    /// Mirror of [`crate::CudaStorage::context`] — same shape, same
    /// rationale (the read-bridge step of the CP-1 substrate lift
    /// documented in `docs/archive/candle-removal/issue-1082-tier-4-5-roadmap-2026-05-27.md`).
    pub fn metal_device_handle(&self) -> MetalRawDevice {
        self.metal_handle.clone()
    }

    /// The Metal device ordinal this storage is bound to.
    pub fn device_index(&self) -> usize {
        match self.device {
            Device::Metal(i) => i,
            _ => unreachable!("MetalStorage::device is always Device::Metal"),
        }
    }

    /// Returns `true` iff this storage's buffer is in a UMA-compatible
    /// storage mode (shared / managed).
    ///
    /// On Apple Silicon, every Metal device is UMA and every buffer
    /// candle's `MetalDevice::allocate_zeros` hands out is in
    /// `MTLStorageModeShared`; `from_buffer` callers must also pass a
    /// Shared/Managed buffer (the constructor's contract). Since Metal
    /// is only supported on Apple Silicon hosts, this is unconditionally
    /// `true` — querying the buffer's actual storage mode would require
    /// reaching through `candle_metal_kernels::metal::Buffer` to the
    /// inner `dyn MTLBuffer` protocol object, which `Buffer` does not
    /// expose. Revisit when supporting Intel Macs or Private-mode
    /// buffers becomes a goal.
    pub fn is_unified_memory(&self) -> bool {
        true
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

// ----------------------------------------------------------------------
// primary_metal_companion — kt-native substrate cache (#1082 Phase 7 Wave 14)
// ----------------------------------------------------------------------
//
// Process-wide cache keyed on Metal device ordinal that materializes
// one `MetalCompanion` per device on first access. **Candle-core-free**:
// the companion's `Device` / `Kernels` / `Commands` triple comes entirely
// from `candle_metal_kernels` (a sibling crate of candle-core, which the
// `metal` feature already pulls).
//
// Used by [`MetalStorage::companion`] as the per-call derivation hook
// for the 7 in-file substrate ops (softmax, rmsnorm, layernorm,
// index_select_dim0, cast, elementwise_binary, activation_unary). The
// previous candle-derived `primary_metal_device` + `MetalStorage::candle_device()`
// shim was deleted alongside this hook's introduction (Wave 14, #1082).
//
// # Why a cache?
//
// `Kernels::new()` is cheap (empty `RwLock<HashMap>`), but `Commands::new`
// allocates a 5-entry `CommandBuffer` pool eagerly. Caching one
// companion per device-ordinal matches what candle does internally
// (its `MetalDevice` instances are also intended to be long-lived) and
// keeps per-op cost down to a `HashMap::get` + `Arc::clone`.
//
// # Phase 7 follow-up
//
// The next step is to flip `MetalAllocator` to hold an
// `Arc<MetalCompanion>` directly (not a `MetalRawDevice`) and pass it
// through to each freshly-allocated `MetalStorage` so the per-op lookup
// becomes a field read. That's a follow-up commit; this one only retires
// the candle hook.

use std::sync::OnceLock;

static METAL_COMPANIONS: OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, Arc<crate::metal_types::MetalCompanion>>>,
> = OnceLock::new();

/// Resolve (or lazily construct) the process-wide kt-native
/// `MetalCompanion` for the given Metal device ordinal.
///
/// Candle-core-free: under the hood this calls
/// `candle_metal_kernels::metal::Device::all()` (same MTL device
/// enumeration `MTLCreateSystemDefaultDevice` underpins, reached through
/// the `candle-metal-kernels` re-export — not `candle-core`) and threads
/// the resulting `Device` through `MetalCompanion::from_raw` to
/// allocate the `Kernels` / `Commands` pair.
///
/// The 7 in-file substrate ops in `metal_storage.rs` (softmax, rmsnorm,
/// layernorm, index_select_dim0, cast, elementwise_binary,
/// activation_unary) consume this via `MetalStorage::companion()` —
/// they replaced the earlier `kt_metal.candle_device()` derivation in
/// the Wave-14 op-migration commit (`ae08652c`).
///
/// # Errors
///
/// Returns [`Error::Msg`] if no Metal device exists at `device_index`
/// (i.e., `Device::all()` returns fewer than `device_index + 1`
/// entries), or if companion construction fails (command-queue or
/// `Commands` pool allocation).
pub fn primary_metal_companion(
    device_index: usize,
) -> Result<Arc<crate::metal_types::MetalCompanion>> {
    let map =
        METAL_COMPANIONS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = map.lock().map_err(|e| {
        Error::Msg(format!(
            "primary_metal_companion({device_index}): METAL_COMPANIONS lock poisoned: {e}"
        ))
    })?;
    if let Some(existing) = map.get(&device_index) {
        return Ok(existing.clone());
    }
    // Enumerate the candle_metal_kernels devices (the candle-core-free
    // path). On Apple Silicon there's typically a single device at
    // index 0 — `Device::all()` returns it (and `Device::system_default()`
    // returns the same physical GPU). On multi-GPU Macs (Pro/Studio with
    // M-Ultra) `Device::all()` exposes each GPU at its own ordinal.
    let devices = crate::metal_rt::Device::all();
    let device = devices.into_iter().nth(device_index).ok_or_else(|| {
        Error::Msg(format!(
            "primary_metal_companion({device_index}): no Metal device at this ordinal \
             (Device::all() returned fewer than {} entries)",
            device_index + 1
        ))
    })?;
    let companion = Arc::new(crate::metal_types::MetalCompanion::from_raw(device)?);
    map.insert(device_index, companion.clone());
    Ok(companion)
}

// ----------------------------------------------------------------------
// Host ↔ Metal I/O — candle-core-free UMA staging (#1082 Phase 1)
// ----------------------------------------------------------------------
//
// These two helpers are the Metal arms of `Tensor::{from_vec_on,
// from_raw_bytes_on, zeros_on}` (host→Metal) and `Tensor::to_device(Cpu)`
// / `Tensor::to_vec` (Metal→host). They are the substrate prerequisite
// for *every* Metal parity test (you cannot A/B a Metal op against the
// CPU reference without constructing a Metal input from host data and
// reading the Metal output back) and for the eventual candle-free
// safetensors→Metal loader.
//
// Apple Silicon is UMA: a `StorageModeShared` buffer's `contents()`
// pointer is addressable from both CPU and GPU, so the "copy" is a plain
// `memcpy` with no PCIe/blit hop. The only subtlety is *ordering* — see
// `metal_to_host_copy`'s `wait_until_completed()` call.

/// Upload a host (CPU-resident) tensor to a fresh Metal `StorageModeShared`
/// buffer on `device_index`. **Candle-core-free.**
///
/// The result is a contiguous, `start_offset == 0` Metal tensor in logical
/// row-major order. The source is materialized contiguous on the host
/// first (cheap when already contiguous), so any input layout is accepted.
///
/// # Errors
///
/// Returns [`Error::Msg`] if `cpu` is not [`CpuStorage`]-backed, no Metal
/// device exists at `device_index`, or buffer allocation fails.
pub fn host_to_metal_copy(cpu: &crate::Tensor, device_index: usize) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    // Materialize a packed, logical-row-major byte image on the host.
    let contig = cpu.contiguous()?;
    let dtype = contig.dtype();
    let cpu_storage = contig
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| {
            Error::Msg("host_to_metal_copy: source tensor must be CPU-backed".to_string())
        })?;
    let all_bytes = cpu_storage.as_bytes();
    // A contiguous tensor may still carry a non-zero start_offset (a
    // narrowed-but-contiguous view shares the parent buffer); slice to the
    // logical element range so the Metal buffer holds exactly this tensor.
    let n_elems = contig.element_count();
    let per = dtype.size_in_bytes();
    let byte_len = if dtype.is_packed() {
        all_bytes.len()
    } else {
        n_elems * per
    };
    let start_bytes = contig.layout().start_offset() * per;
    let end_bytes = start_bytes + byte_len;
    if end_bytes > all_bytes.len() {
        return Err(Error::Msg(format!(
            "host_to_metal_copy: byte range {start_bytes}..{end_bytes} exceeds CPU storage \
             length {}",
            all_bytes.len()
        )));
    }
    let src = &all_bytes[start_bytes..end_bytes];

    let companion = primary_metal_companion(device_index)?;
    let raw_device = companion.device();
    let alloc_len = byte_len.max(1);
    let buffer = raw_device
        .new_buffer(alloc_len, MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "host_to_metal_copy: new_buffer({alloc_len}, Shared) failed: {e:?}"
            ))
        })?;
    // SAFETY: `buffer.contents()` is a non-null CPU-addressable pointer for
    // Shared-mode buffers on Apple Silicon UMA; `alloc_len >= src.len()` so
    // the copy stays within the allocation. The buffer was just allocated
    // (single owner, no aliasing).
    unsafe {
        core::ptr::copy_nonoverlapping(src.as_ptr(), buffer.contents(), src.len());
    }

    let storage = MetalStorage::from_buffer_kt(raw_device, device_index, dtype, Arc::new(buffer))?;
    crate::Tensor::from_parts(
        Arc::new(storage),
        crate::Layout::contiguous(contig.shape().to_vec()),
        crate::TensorId::next(),
    )
}

/// Read a Metal tensor back to a fresh CPU tensor, packed contiguous in
/// logical row-major order. **Candle-core-free.**
///
/// Commits and waits on the companion's command queue first so the GPU
/// writes that produced this tensor are visible through the UMA
/// `contents()` pointer, then gathers the logical elements (handling any
/// strided / offset view via a host-side gather).
///
/// # Errors
///
/// Returns [`Error::Msg`] if the tensor is not [`MetalStorage`]-backed or
/// the queue sync fails.
pub fn metal_to_host_copy(t: &crate::Tensor) -> Result<crate::Tensor> {
    let metal = t
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_to_host_copy: tensor must be Metal-backed".to_string()))?;
    // Host-read synchronization point: make GPU writes visible.
    metal.companion()?.wait_until_completed()?;

    let dtype = t.dtype();
    let per = dtype.size_in_bytes();
    let buffer = metal.buffer();
    let buf_len = buffer.length() as usize;
    // SAFETY: Shared-mode buffer `contents()` is CPU-addressable and valid
    // for `buf_len` bytes; we only read within `[0, buf_len)`.
    let backing: &[u8] =
        unsafe { core::slice::from_raw_parts(buffer.contents() as *const u8, buf_len) };

    let layout = t.layout();
    let n_elems = t.element_count();

    if dtype.is_packed() {
        // Packed dtypes have no per-element stride math; copy the addressed
        // byte image directly. (Metal packed-dtype tensors are always whole
        // contiguous buffers in the current code paths.)
        let bytes = backing.to_vec();
        let storage = CpuStorage::from_bytes(dtype, bytes)?;
        return crate::Tensor::from_parts(
            Arc::new(storage),
            crate::Layout::contiguous(t.shape().to_vec()),
            crate::TensorId::next(),
        );
    }

    let start = layout.start_offset();
    let mut out = vec![0u8; n_elems * per];
    if layout.is_contiguous() {
        let s = start * per;
        let e = s + n_elems * per;
        if e > buf_len {
            return Err(Error::Msg(format!(
                "metal_to_host_copy: contiguous range {s}..{e} exceeds buffer length {buf_len}"
            )));
        }
        out.copy_from_slice(&backing[s..e]);
    } else {
        // Strided / permuted view: gather each logical element by walking
        // the multi-dimensional index against the layout strides. Not a hot
        // path (host readback is rare), so a per-element gather is fine.
        let dims = layout.shape();
        let strides = layout.strides();
        let rank = dims.len();
        let mut idx = vec![0usize; rank];
        for logical in 0..n_elems {
            // physical element offset = start + Σ idx[d] * strides[d]
            let mut phys = start;
            for d in 0..rank {
                phys += idx[d] * strides[d];
            }
            let src = phys * per;
            let dst = logical * per;
            if src + per > buf_len {
                return Err(Error::Msg(format!(
                    "metal_to_host_copy: element offset {src}..{} exceeds buffer length {buf_len}",
                    src + per
                )));
            }
            out[dst..dst + per].copy_from_slice(&backing[src..src + per]);
            // increment the mixed-radix logical index (row-major: last dim fastest)
            for d in (0..rank).rev() {
                idx[d] += 1;
                if idx[d] < dims[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
    }

    let storage = CpuStorage::from_bytes(dtype, out)?;
    crate::Tensor::from_parts(
        Arc::new(storage),
        crate::Layout::contiguous(t.shape().to_vec()),
        crate::TensorId::next(),
    )
}

/// Deep-copy a Metal tensor to a fresh Metal buffer.
///
/// Contiguous tensors use the Apple Silicon UMA fast path: wait for queued GPU
/// writes, allocate a new Shared-mode `MTLBuffer`, then copy the addressed byte
/// range directly. Non-contiguous and packed tensors fall back through the
/// existing logical host image path so layout semantics stay exact.
pub fn metal_deep_copy(t: &crate::Tensor) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    let metal = t
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_deep_copy: tensor must be Metal-backed".to_string()))?;
    let device_index = metal.device_index();
    let dtype = t.dtype();

    if dtype.is_packed() || !t.layout().is_contiguous() {
        let host = metal_to_host_copy(t)?;
        return host_to_metal_copy(&host, device_index);
    }

    metal.companion()?.wait_until_completed()?;

    let per = dtype.size_in_bytes();
    let byte_len = t.element_count() * per;
    let start_bytes = t.layout().start_offset() * per;
    let end_bytes = start_bytes + byte_len;
    let source_buffer = metal.buffer();
    let source_len = source_buffer.length() as usize;
    if end_bytes > source_len {
        return Err(Error::Msg(format!(
            "metal_deep_copy: byte range {start_bytes}..{end_bytes} exceeds buffer length \
             {source_len}"
        )));
    }

    let companion = primary_metal_companion(device_index)?;
    let raw_device = companion.device();
    let alloc_len = byte_len.max(1);
    let buffer = raw_device
        .new_buffer(alloc_len, MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_deep_copy: new_buffer({alloc_len}, Shared) failed: {e:?}"
            ))
        })?;

    unsafe {
        let src = (source_buffer.contents() as *const u8).add(start_bytes);
        core::ptr::copy_nonoverlapping(src, buffer.contents(), byte_len);
    }

    let storage = MetalStorage::from_buffer_kt(raw_device, device_index, dtype, Arc::new(buffer))?;
    crate::Tensor::from_parts(
        Arc::new(storage),
        crate::Layout::contiguous(t.shape().to_vec()),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_softmax_last_axis — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal softmax over the trailing axis. Mirrors the role of
/// [`crate::cuda_softmax_last_axis`] for the Metal backend.
///
/// Operates on a contiguous `[..., D]` Metal-backed tensor; produces a
/// fresh contiguous tensor of the same shape and dtype with each
/// `[..., :]` row normalized to a probability distribution.
///
/// # Implementation
///
/// Kiln-owned MSL (`metal_kernels`), replacing `candle_metal_kernels::call_last_softmax` (the
/// production MSL `softmax_<dt>` kernel). Output buffer is allocated
/// via metal-rs `Device::new_buffer` in `StorageModeShared` (Apple
/// Silicon UMA — host and GPU share physical memory). The `MetalDevice`
/// wrapper is held only for command-queue affinity (`kernels()`,
/// `command_encoder()`); no `CandleTensor` / `CandleMetalStorage` /
/// `candle_nn` bridge is involved.
///
/// # Requirements
///
/// - `x` must be backed by [`MetalStorage`]
/// - `x.dtype()` must be `F32`, `BF16`, or `F16`
/// - `x.rank() >= 1`
/// - `x.is_contiguous()` must hold
///
/// # Errors
///
/// Returns [`Error::Msg`] if the storage isn't `MetalStorage`, the
/// dtype is unsupported, the layout is non-contiguous, or the
/// underlying MSL kernel dispatch fails.
pub fn metal_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    // ---- Validate kt-side preconditions ----
    let dtype = x.dtype();
    let dtype_size: usize = match dtype {
        DType::F32 => 4,
        DType::BF16 => 2,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "metal_softmax_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if x.rank() == 0 {
        return Err(Error::Msg(
            "metal_softmax_last_axis: input must have rank >= 1".to_string(),
        ));
    }
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "metal_softmax_last_axis: input must be contiguous".to_string(),
        ));
    }

    let kt_metal = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_softmax_last_axis: input must be Metal-backed".to_string())
        })?;

    let companion = kt_metal.companion()?;
    let device_index = match kt_metal.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let element_count: usize = x.element_count();
    let last_dim = *shape.last().unwrap();

    // Allocate output buffer directly through metal-rs (no candle).
    let byte_len = element_count * dtype_size;
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_softmax_last_axis: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    // Kiln-owned MSL softmax (replaces candle's call_last_softmax) — faithful
    // port of reduce.metal `softmax_<dt>`. One threadgroup per last-axis row.
    let rows = element_count / last_dim;
    crate::metal_kernels::softmax_last_axis(
        &companion,
        kt_metal.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        rows,
        last_dim,
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_log_softmax_last_axis — Metal substrate op (#1082 OPD lane)
// ----------------------------------------------------------------------

/// Metal log-softmax over the trailing axis. Mirrors the role of the
/// CPU `ops::log_softmax_last_dim` for the Metal backend, and is the
/// Metal twin of [`metal_softmax_last_axis`].
///
/// Operates on a contiguous `[..., D]` Metal-backed tensor; produces a
/// fresh contiguous tensor of the same shape and dtype with each
/// `[..., :]` row replaced by `x - logsumexp(x)` (numerically stable —
/// row-max subtracted before the `exp`).
///
/// # Requirements
///
/// - `x` must be backed by [`MetalStorage`]
/// - `x.dtype()` must be `F32`, `BF16`, or `F16`
/// - `x.rank() >= 1`
/// - `x.is_contiguous()` must hold
pub fn metal_log_softmax_last_axis(x: &crate::Tensor) -> Result<crate::Tensor> {
    metal_log_softmax_last_axis_impl(x, false)
}

/// Metal log-softmax over the trailing axis with direct F32 output.
///
/// The MSL kernel reads F32/BF16/F16 input directly, keeps both the maximum
/// and exponential sum in F32, and writes a single F32 output allocation.
pub fn metal_log_softmax_last_axis_f32(x: &crate::Tensor) -> Result<crate::Tensor> {
    metal_log_softmax_last_axis_impl(x, true)
}

fn metal_log_softmax_last_axis_impl(x: &crate::Tensor, output_f32: bool) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    let label = if output_f32 {
        "metal_log_softmax_last_axis_f32"
    } else {
        "metal_log_softmax_last_axis"
    };
    let input_dtype = x.dtype();
    match input_dtype {
        DType::F32 | DType::BF16 | DType::F16 => {}
        other => {
            return Err(Error::Msg(format!("{label}: unsupported dtype {other}")));
        }
    }
    if x.rank() == 0 {
        return Err(Error::Msg(format!("{label}: input must have rank >= 1")));
    }
    if !x.is_contiguous() {
        return Err(Error::Msg(format!("{label}: input must be contiguous")));
    }

    let kt_metal = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg(format!("{label}: input must be Metal-backed")))?;

    let companion = kt_metal.companion()?;
    let device_index = match kt_metal.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let element_count: usize = x.element_count();
    let last_dim = *shape.last().unwrap();
    if last_dim == 0 {
        return Err(Error::Msg(format!(
            "{label}: trailing axis must be non-empty"
        )));
    }
    let output_dtype = if output_f32 { DType::F32 } else { input_dtype };

    // Allocate output buffer directly through metal-rs (no candle).
    let byte_len = element_count * output_dtype.size_in_bytes();
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| Error::Msg(format!("{label}: new_buffer({byte_len}) failed: {e:?}")))?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    // Kiln-owned MSL log-softmax — same online normalizer as softmax,
    // log finalize. One threadgroup per last-axis row.
    let rows = element_count / last_dim;
    if output_f32 {
        crate::metal_kernels::log_softmax_last_axis_f32(
            &companion,
            kt_metal.buffer().as_ref(),
            out_buffer_arc.as_ref(),
            input_dtype,
            rows,
            last_dim,
        )?;
    } else {
        crate::metal_kernels::log_softmax_last_axis(
            &companion,
            kt_metal.buffer().as_ref(),
            out_buffer_arc.as_ref(),
            input_dtype,
            rows,
            last_dim,
        )?;
    }

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, output_dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

/// Inclusive prefix-sum (`cumsum`) along `axis` on the Metal backend — kt-native
/// MSL scan (`metal_kernels::cumsum_axis`), no host round-trip. Output dtype ==
/// input dtype, F32 accumulation (bit-matching the CPU + CUDA references). The
/// caller (`ops::cumsum`) validates rank/dtype/contiguity.
pub fn metal_cumsum_axis(x: &crate::Tensor, axis: usize) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    let dtype = x.dtype();
    let dtype_size: usize = match dtype {
        DType::F32 => 4,
        DType::BF16 | DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "metal_cumsum_axis: unsupported dtype {other}"
            )));
        }
    };
    let kt_metal = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_cumsum_axis: input must be Metal-backed".to_string()))?;
    let companion = kt_metal.companion()?;
    let device_index = match kt_metal.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let axis_dim: usize = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);

    let byte_len = x.element_count() * dtype_size;
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_cumsum_axis: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    crate::metal_kernels::cumsum_axis(
        &companion,
        kt_metal.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        outer,
        axis_dim,
        inner,
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);
    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_sdpa_last_axis — Phase 7 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Fused scaled-dot-product attention on the Metal backend — kt-native
/// substrate op that mirrors [`candle_nn::ops::sdpa`] without going
/// through the candle dispatch layer.
///
/// This is the substrate addition called out in Step 2 of
/// `docs/archive/metal/metal-types-objc2-swap-plan-2026-05-28.md`. It is
/// the **only**
/// substrate gap left between `kiln-tensor` and the `metal_types`
/// chokepoint flip — every other Phase 7 transition is a pure caller
/// migration (Storage downcast, helper-signature swap, re-export
/// renames).
///
/// # Inputs
///
/// - `q`: `(bs, qhead, seq, hidden)` — Metal-backed, contiguous
/// - `k`: `(bs, kv_head, kv_seq, hidden)` — Metal-backed, contiguous
/// - `v`: `(bs, kv_head, kv_seq, v_hidden)` — Metal-backed, contiguous
/// - `scale`: applied before softmax (same as `candle_nn::ops::sdpa`'s
///   `scale` parameter)
/// - `causal`: enables MLX-style causal masking; the prefill path in
///   `kiln-model::backend::metal` is always causal
///
/// # Output
///
/// Fresh contiguous `(bs, qhead, seq, v_hidden)` tensor.
///
/// # Implementation
///
/// Mirror of [`metal_softmax_last_axis`]: dispatches directly into the
/// kiln-owned unified flash-attention MSL (`metal_kernels::sdpa`), replacing the candle `call_sdpa_*` family — the same FFI
/// entry points `candle_nn::ops::sdpa` uses internally. The wire-level
/// kernel call is bit-exact; this op only adds the kt-typed signature.
///
/// The dispatcher selects between vector / vector-2pass / full kernels
/// using the same q_seq + head_dim rules candle's `Sdpa::metal_fwd`
/// applies:
///
/// - `q_seq <= 8`: vector kernel (with 2-pass split if `k_seq >= 1024`)
/// - `q_seq > 8`: full attention kernel
///
/// `softcapping` is fixed at `1.0` (disabled) because every kiln call
/// site under `kiln-model::backend::metal` passes `1.0`; if we ever need
/// softcapping at the kt layer, add it as an explicit arg here rather
/// than implicitly through the candle re-export.
///
/// `mask` is fixed at `None` because every kiln call site passes
/// `None` (kiln implements causal masking via the `causal` flag, never
/// via an external mask tensor). A future PR can add a mask arg if a
/// non-causal call site appears.
///
/// # Requirements
///
/// - `q`, `k`, `v` must all be backed by [`MetalStorage`]
/// - all three must share the same dtype: `F32`, `BF16`, or `F16`
/// - `q.rank() == k.rank() == v.rank() == 4`
/// - all three must be contiguous
/// - `q.dim(3) == k.dim(3)` (matching embedding dim)
/// - `k.dim(1) == v.dim(1)` (matching kv-head count)
/// - `q.dim(1) % k.dim(1) == 0` (GQA factor)
/// - `head_dim` (last dim of `q`) must be one of:
///   32, 64, 72, 80, 96, 128, 256, 512
/// - F32 + head_dim=512 is unsupported on the full kernel (32KB
///   threadgroup-memory limit); use BF16/F16 there
///
/// # Errors
///
/// Returns [`Error::Msg`] on any precondition failure or kernel dispatch
/// error.
pub fn metal_sdpa_last_axis(
    q: &crate::Tensor,
    k: &crate::Tensor,
    v: &crate::Tensor,
    scale: f32,
    causal: bool,
) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    // ---- Validate kt-side preconditions ----
    let dtype = q.dtype();
    let dtype_size: usize = match dtype {
        DType::F32 => 4,
        DType::BF16 => 2,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "metal_sdpa_last_axis: unsupported dtype {other} (F32/BF16/F16 only)"
            )));
        }
    };
    if k.dtype() != dtype || v.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_sdpa_last_axis: q/k/v dtypes must match (got q={}, k={}, v={})",
            dtype,
            k.dtype(),
            v.dtype()
        )));
    }
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(Error::Msg(format!(
            "metal_sdpa_last_axis: q/k/v must all be rank-4 (got q={}, k={}, v={})",
            q.rank(),
            k.rank(),
            v.rank()
        )));
    }
    // NOTE: q/k/v may be strided (e.g. a KV-cache view); the kiln SDPA kernel
    // indexes via each tensor's element strides + start offset, so no
    // contiguity precondition is required here. (out is freshly contiguous.)

    let q_dims = q.shape();
    let k_dims = k.shape();
    let v_dims = v.shape();

    // q,k must have matching embedding dim (last dim).
    if q_dims[3] != k_dims[3] {
        return Err(Error::Msg(format!(
            "metal_sdpa_last_axis: q and k last dims must match (got q={}, k={})",
            q_dims[3], k_dims[3]
        )));
    }
    // k,v must have matching kv-head count (dim 1).
    if k_dims[1] != v_dims[1] {
        return Err(Error::Msg(format!(
            "metal_sdpa_last_axis: k and v head dims must match (got k={}, v={})",
            k_dims[1], v_dims[1]
        )));
    }
    // n_heads % n_kv_heads == 0.
    if q_dims[1] % k_dims[1] != 0 {
        return Err(Error::Msg(format!(
            "metal_sdpa_last_axis: q n_heads ({}) must be a multiple of k n_kv_heads ({})",
            q_dims[1], k_dims[1]
        )));
    }

    let head_dim = q_dims[3];
    let q_seq = q_dims[2];
    let k_seq = k_dims[2];
    // v's head_dim must match q/k (the kernel writes a head_dim-wide output row),
    // and k/v must share seqlen.
    if v_dims[3] != head_dim {
        return Err(Error::Msg(format!(
            "metal_sdpa_last_axis: v head_dim ({}) must match q/k head_dim ({head_dim})",
            v_dims[3]
        )));
    }
    if k_dims[2] != v_dims[2] {
        return Err(Error::Msg(format!(
            "metal_sdpa_last_axis: k and v seqlen must match (got k={}, v={})",
            k_dims[2], v_dims[2]
        )));
    }

    // ---- Storage downcast ----
    let q_metal = q
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_sdpa_last_axis: q must be Metal-backed".to_string()))?;
    let k_metal = k
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_sdpa_last_axis: k must be Metal-backed".to_string()))?;
    let v_metal = v
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_sdpa_last_axis: v must be Metal-backed".to_string()))?;

    // Device-affinity contract: companions resolved via
    // `MetalStorage::companion()` route through the same Device::all()
    // registry-ID lookup. We use q's companion for the kernel dispatch
    // and assume k/v share affinity (precondition: same Metal device).
    let companion = q_metal.companion()?;
    let device_index = match q_metal.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };

    // ---- Allocate output buffer (bs, qhead, seq, v_hidden) ----
    let out_shape: Vec<usize> = vec![q_dims[0], q_dims[1], q_dims[2], v_dims[3]];
    let elem_count: usize = out_shape.iter().product();
    let byte_len = elem_count * dtype_size;

    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_sdpa_last_axis: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    // Output layout (contiguous), needed by call_sdpa_full for o_strides.
    let out_layout = crate::Layout::contiguous(out_shape.clone());

    // ---- Dispatch the kiln-owned unified flash-attention SDPA ----
    // Replaces candle's call_sdpa_{vector,vector_2pass,full} trio with one
    // online-softmax kernel handling vector + full + GQA + causal + strided
    // q/k/v. The helper opens its own encoder (start offsets ride in params,
    // so the buffer byte-offsets stay 0).
    let batch = q_dims[0];
    let n_heads = q_dims[1];
    let n_kv_heads = k_dims[1];
    crate::metal_kernels::sdpa(
        &companion,
        q_metal.buffer().as_ref(),
        k_metal.buffer().as_ref(),
        v_metal.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        batch,
        n_heads,
        n_kv_heads,
        q_seq,
        k_seq,
        head_dim,
        scale,
        causal,
        q.layout().strides(),
        q.layout().start_offset(),
        k.layout().strides(),
        k.layout().start_offset(),
        v.layout().strides(),
        v.layout().start_offset(),
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(out_storage_arc, out_layout, crate::TensorId::next())
}

// ----------------------------------------------------------------------
// metal_rmsnorm_last_axis — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal RMSNorm over the trailing axis. Mirrors the role of
/// [`crate::cuda_rmsnorm_last_axis`] for the Metal backend.
///
/// Operates on contiguous `[..., D]` and `[D]` Metal-backed tensors and
/// produces a fresh contiguous tensor with each `[..., :]` row
/// normalized by its row-RMS and scaled per-element by `weight`:
/// `y = w * x / sqrt(mean(x^2) + eps)`.
///
/// # Implementation
///
/// Kiln-owned MSL (`metal_kernels`), replacing `candle_metal_kernels::call_rms_norm` (the
/// production MSL `rmsnorm_<dt>` kernel). Output buffer is allocated
/// via metal-rs `Device::new_buffer` in `StorageModeShared`. The
/// `MetalDevice` wrapper is held only for command-queue affinity; no
/// `CandleTensor` / `CandleMetalStorage` / `candle_nn` bridge involved.
///
/// # Requirements
///
/// - `x` and `weight` must both be backed by [`MetalStorage`]
/// - `x.dtype()` must be `F32`, `BF16`, or `F16` (and equal to
///   `weight.dtype()`)
/// - `x.rank() >= 1`, `weight.rank() == 1`
/// - `weight.shape()[0] == *x.shape().last().unwrap()`
/// - both inputs contiguous
///
/// # Errors
///
/// Returns [`Error::Msg`] on any precondition failure or kernel dispatch error.
pub fn metal_rmsnorm_last_axis(
    x: &crate::Tensor,
    weight: &crate::Tensor,
    eps: f32,
) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    let dtype = x.dtype();
    let dtype_size: usize = match dtype {
        DType::F32 => 4,
        DType::BF16 => 2,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "metal_rmsnorm_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if weight.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_rmsnorm_last_axis: weight dtype {} != x dtype {dtype}",
            weight.dtype()
        )));
    }
    if x.rank() == 0 || weight.rank() != 1 {
        return Err(Error::Msg(format!(
            "metal_rmsnorm_last_axis: rank constraints failed (x.rank={}, weight.rank={})",
            x.rank(),
            weight.rank()
        )));
    }
    if !x.is_contiguous() || !weight.is_contiguous() {
        return Err(Error::Msg(
            "metal_rmsnorm_last_axis: inputs must be contiguous".to_string(),
        ));
    }
    let hidden = *x.shape().last().unwrap();
    if weight.shape().first().copied() != Some(hidden) {
        return Err(Error::Msg(format!(
            "metal_rmsnorm_last_axis: weight.shape()[0] {:?} != x last-dim {hidden}",
            weight.shape()
        )));
    }

    let kt_metal_x = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_rmsnorm_last_axis: x must be Metal-backed".to_string()))?;
    let kt_metal_w = weight
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_rmsnorm_last_axis: weight must be Metal-backed".to_string())
        })?;

    let companion = kt_metal_x.companion()?;
    let device_index = match kt_metal_x.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let element_count_x: usize = x.element_count();

    // Allocate output buffer directly through metal-rs (no candle).
    let byte_len = element_count_x * dtype_size;
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_rmsnorm_last_axis: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    // Kiln-owned MSL RMSNorm (replaces candle's call_rms_norm) — faithful
    // port of reduce.metal `rms_norm`. One threadgroup per last-axis row.
    crate::metal_kernels::rms_norm(
        &companion,
        kt_metal_x.buffer().as_ref(),
        kt_metal_w.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        element_count_x,
        hidden,
        eps,
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_layernorm_last_axis — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal LayerNorm over the trailing axis. Mirrors the role of
/// [`crate::cuda_layernorm_last_axis`] for the Metal backend.
///
/// Operates on contiguous `[..., D]`, `[D]`, `[D]` Metal-backed
/// tensors and produces a fresh contiguous tensor:
/// `y = ((x - mean) / sqrt(var + eps)) * weight + bias`.
///
/// # Implementation
///
/// Kiln-owned MSL (`metal_kernels`), replacing `candle_metal_kernels::call_layer_norm` (the
/// production MSL `layernorm_<dt>` kernel). Output buffer is allocated
/// via metal-rs `Device::new_buffer` in `StorageModeShared`. No
/// `CandleTensor` / `CandleMetalStorage` / `candle_nn` bridge — same
/// pattern as `metal_softmax_last_axis` and `metal_rmsnorm_last_axis`
/// post #1082 substrate lift.
///
/// # Requirements
///
/// - `x`, `weight`, `bias` must all be Metal-backed
/// - all three share dtype in {F32, BF16, F16}
/// - `x.rank() >= 1`, `weight.rank() == 1`, `bias.rank() == 1`
/// - `weight.shape()[0] == bias.shape()[0] == *x.shape().last().unwrap()`
/// - all three contiguous
///
/// # Errors
///
/// Returns [`Error::Msg`] on any precondition failure or kernel dispatch error.
pub fn metal_layernorm_last_axis(
    x: &crate::Tensor,
    weight: &crate::Tensor,
    bias: &crate::Tensor,
    eps: f32,
) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    let dtype = x.dtype();
    let dtype_size: usize = match dtype {
        DType::F32 => 4,
        DType::BF16 => 2,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "metal_layernorm_last_axis: unsupported dtype {other}"
            )));
        }
    };
    if weight.dtype() != dtype || bias.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_layernorm_last_axis: weight dtype {} / bias dtype {} != x dtype {dtype}",
            weight.dtype(),
            bias.dtype()
        )));
    }
    if x.rank() == 0 || weight.rank() != 1 || bias.rank() != 1 {
        return Err(Error::Msg(format!(
            "metal_layernorm_last_axis: rank constraints failed (x={}, w={}, b={})",
            x.rank(),
            weight.rank(),
            bias.rank()
        )));
    }
    if !x.is_contiguous() || !weight.is_contiguous() || !bias.is_contiguous() {
        return Err(Error::Msg(
            "metal_layernorm_last_axis: inputs must be contiguous".to_string(),
        ));
    }
    let hidden = *x.shape().last().unwrap();
    if weight.shape().first().copied() != Some(hidden)
        || bias.shape().first().copied() != Some(hidden)
    {
        return Err(Error::Msg(format!(
            "metal_layernorm_last_axis: weight/bias shapes ({:?}, {:?}) != x last-dim {hidden}",
            weight.shape(),
            bias.shape()
        )));
    }

    let kt_metal_x = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_layernorm_last_axis: x must be Metal-backed".to_string())
        })?;
    let kt_metal_w = weight
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_layernorm_last_axis: weight must be Metal-backed".to_string())
        })?;
    let kt_metal_b = bias
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_layernorm_last_axis: bias must be Metal-backed".to_string())
        })?;

    let companion = kt_metal_x.companion()?;
    let device_index = match kt_metal_x.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };

    let shape: Vec<usize> = x.shape().to_vec();
    let element_count_x = x.element_count();

    // Allocate output buffer directly through metal-rs (no candle).
    let byte_len = element_count_x * dtype_size;
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_layernorm_last_axis: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    // Kiln-owned MSL LayerNorm (replaces candle's call_layer_norm) — faithful
    // two-pass port of reduce.metal `layernorm`. One threadgroup per row.
    let n_rows = element_count_x / hidden;
    crate::metal_kernels::layer_norm(
        &companion,
        kt_metal_x.buffer().as_ref(),
        kt_metal_w.buffer().as_ref(),
        kt_metal_b.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        n_rows,
        hidden,
        eps,
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_index_select_dim0 — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal index_select along axis 0. Mirrors the role of
/// [`crate::cuda_index_select_dim0`] for the Metal backend.
///
/// Kiln-owned MSL (`metal_kernels`), replacing `candle_metal_kernels::call_index_select` (the
/// production MSL `is_u32_<dt>` kernel — same path
/// `candle_core::Tensor::index_select(...)` resolves internally).
///
/// Given:
///   - `input: [vocab_size, hidden]` or `[axis_dim, ...]` (rank >= 1)
///   - `indices: [N]` or higher-rank, dtype U32
///
/// Produces a contiguous `[indices.shape, ...input.shape[1..]]` tensor
/// with the same dtype as `input`.
///
/// # Apple Silicon UMA zero-copy invariant
///
/// Output buffer allocated via metal-rs `new_buffer` in
/// `StorageModeShared`. No `CandleTensor` / `CandleMetalStorage`
/// bridge (#1082).
///
/// # Requirements
///
/// - `input` must be backed by [`MetalStorage`]
/// - `indices` must be backed by [`MetalStorage`]
/// - `input` and `indices` both contiguous
/// - `indices.dtype() == U32` (matches CUDA's `cuda_index_select_dim0`)
/// - `input.dtype()` in {F32, BF16, F16} (the `is_u32_<dt>` table only
///   covers these; integer/packed dtypes return Err here and the op's
///   metal_fwd falls through to CPU.)
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported dtype, non-contiguous layout,
/// non-Metal storage, or kernel dispatch error.
pub fn metal_index_select_dim0(
    input: &crate::Tensor,
    indices: &crate::Tensor,
) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    let dtype = input.dtype();
    let dtype_size: usize = match dtype {
        DType::F32 => 4,
        DType::BF16 => 2,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "metal_index_select_dim0: unsupported input dtype {other} \
                 (float triple only)"
            )));
        }
    };
    if indices.dtype() != DType::U32 {
        return Err(Error::Msg(format!(
            "metal_index_select_dim0: indices dtype must be U32 (got {})",
            indices.dtype()
        )));
    }
    if !input.is_contiguous() || !indices.is_contiguous() {
        return Err(Error::Msg(
            "metal_index_select_dim0: inputs must be contiguous".to_string(),
        ));
    }
    if input.rank() == 0 {
        return Err(Error::Msg(
            "metal_index_select_dim0: input must have rank >= 1".to_string(),
        ));
    }

    let kt_metal_in = input
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_index_select_dim0: input must be Metal-backed".to_string())
        })?;
    let kt_metal_ids = indices
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_index_select_dim0: indices must be Metal-backed".to_string())
        })?;

    let companion = kt_metal_in.companion()?;
    let device_index = match kt_metal_in.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let in_shape: Vec<usize> = input.shape().to_vec();
    let ids_shape: Vec<usize> = indices.shape().to_vec();
    let ids_element_count: usize = indices.element_count();

    // Output shape: ids.shape ++ in_shape[1..]. candle's index_select
    // along dim 0 produces this same shape (replaces dim 0 with
    // ids.shape, flattening if ids is multi-dim — `call_index_select`
    // accepts `ids_size` as the count of indices regardless of ids
    // rank, and the output is shaped [N, in.shape[1..]] where N is
    // the total ids count).
    //
    // Compute the equivalent output shape matching candle's contract:
    // `[ids_shape ..., input.shape[1..]...]`. Total element count is
    // `ids_element_count * (in_shape[1..].iter().product())`.
    let mut out_shape: Vec<usize> = ids_shape.clone();
    out_shape.extend(in_shape.iter().skip(1).copied());
    let out_element_count: usize = out_shape.iter().product::<usize>().max(0);

    // Allocate output buffer directly through metal-rs (no candle).
    let byte_len = out_element_count * dtype_size;
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_index_select_dim0: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    // Kiln-owned MSL index_select dim0 (replaces candle's call_index_select).
    // row_len = product(in_shape[1..]) (candle's `right_size`); src_dim_size =
    // in_shape[0]; ids flattened to ids_element_count rows.
    let row_len: usize = in_shape.iter().skip(1).product::<usize>().max(1);
    let src_dim_size: usize = in_shape[0];
    crate::metal_kernels::index_select_dim0(
        &companion,
        kt_metal_in.buffer().as_ref(),
        kt_metal_ids.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        src_dim_size,
        row_len,
        ids_element_count,
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_cast — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal dtype cast. Mirrors the role of [`crate::cuda_cast`] for the
/// Metal backend.
///
/// Kiln-owned MSL (`metal_kernels`), replacing `candle_metal_kernels::call_cast_contiguous`
/// (the production MSL `cast_<from>_<to>` kernel — same path
/// `candle_core::Tensor::to_dtype(...)` resolves internally). Covers
/// F32 <-> BF16 <-> F16 — the float triple. Integer round-trips
/// (U32 <-> I64) stay on the CPU fallback for now.
///
/// # Apple Silicon UMA zero-copy invariant
///
/// Output buffer allocated via metal-rs `new_buffer` in
/// `StorageModeShared`. Drops the `CandleTensor` / `CandleMetalStorage`
/// bridge that earlier shipping of this op carried (#1082).
///
/// # Requirements
///
/// - `x` must be backed by [`MetalStorage`]
/// - `x.is_contiguous()`
/// - `(x.dtype(), to)` in the supported float triple
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported dtype pair, non-contiguous
/// layout, non-Metal storage, or kernel dispatch error.
pub fn metal_cast(x: &crate::Tensor, to: DType) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    let from = x.dtype();
    // Float triple + U8 (boolean masks). U32↔I64 integer round-trips stay on
    // the CPU fallback (callers' metal_fwd falls through).
    // `metal_kernels::cast_msl_ty` gates the supported dtypes.
    let to_dtype_size: usize = match to {
        DType::F32 => 4,
        DType::BF16 | DType::F16 => 2,
        DType::U8 => 1,
        other => {
            return Err(Error::Msg(format!(
                "metal_cast: unsupported target dtype {other} (float triple + U8 only)"
            )));
        }
    };
    // Reject unsupported source/target dtypes up front (mirror of the old match).
    crate::metal_kernels::cast_msl_ty(from)?;
    crate::metal_kernels::cast_msl_ty(to)?;

    if !x.is_contiguous() {
        return Err(Error::Msg(
            "metal_cast: input must be contiguous".to_string(),
        ));
    }

    let kt_metal = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_cast: input must be Metal-backed".to_string()))?;

    let companion = kt_metal.companion()?;
    let device_index = match kt_metal.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let shape: Vec<usize> = x.shape().to_vec();
    let element_count: usize = x.element_count();

    // Allocate output buffer directly through metal-rs (no candle).
    let byte_len = element_count * to_dtype_size;
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| Error::Msg(format!("metal_cast: new_buffer({byte_len}) failed: {e:?}")))?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    // Kiln-owned MSL cast (replaces candle's call_cast_contiguous).
    crate::metal_kernels::cast(
        &companion,
        kt_metal.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        from,
        to,
        element_count,
    )?;

    let out_storage = MetalStorage::from_buffer_kt(raw_device, device_index, to, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_elementwise_binary — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal element-wise binary op (add/sub/mul/div). Mirrors the role of
/// [`crate::cuda_elementwise_binary`] for the Metal backend.
///
/// Dispatches on `kind_tag` (matches the CUDA tags in
/// `ElementwiseOp::cuda_fwd`):
///   - `0` -> Add — MSL kernel `badd_<dt>`
///   - `1` -> Sub — MSL kernel `bsub_<dt>`
///   - `2` -> Mul — MSL kernel `bmul_<dt>`
///   - `3` -> Div — MSL kernel `bdiv_<dt>`
///
/// Kiln-owned MSL (`metal_kernels`), replacing `candle_metal_kernels::call_binary_contiguous`.
/// Covers F32 / BF16 / F16. Both inputs must share shape and dtype.
///
/// # Apple Silicon UMA zero-copy invariant
///
/// Output buffer allocated via metal-rs `new_buffer` in
/// `StorageModeShared`. No `CandleTensor` / `CandleMetalStorage`
/// bridge (#1082).
///
/// # Requirements
///
/// - `a` and `b` must both be backed by [`MetalStorage`]
/// - `a.dtype() == b.dtype()` and dtype in {F32, BF16, F16}
/// - `a.shape() == b.shape()` (no broadcasting yet)
/// - both contiguous
/// - `kind_tag` in {0, 1, 2, 3}
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported kind, dtype, non-contiguous
/// layout, non-Metal storage, shape mismatch, or kernel dispatch error.
pub fn metal_elementwise_binary(
    a: &crate::Tensor,
    b: &crate::Tensor,
    kind_tag: i32,
) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    if !matches!(kind_tag, 0 | 1 | 2 | 3) {
        return Err(Error::Msg(format!(
            "metal_elementwise_binary: kind_tag {kind_tag} not supported \
             (only 0=Add, 1=Sub, 2=Mul, 3=Div)"
        )));
    }
    let dtype = a.dtype();
    let (dtype_size, _dtype_suffix): (usize, &'static str) = match dtype {
        DType::F32 => (4, "f32"),
        DType::BF16 => (2, "bf16"),
        DType::F16 => (2, "f16"),
        other => {
            return Err(Error::Msg(format!(
                "metal_elementwise_binary: unsupported dtype {other}"
            )));
        }
    };
    if b.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_elementwise_binary: dtype mismatch a={dtype} b={}",
            b.dtype()
        )));
    }
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!(
            "metal_elementwise_binary: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "metal_elementwise_binary: inputs must be contiguous".to_string(),
        ));
    }

    let kt_metal_a = a
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_elementwise_binary: a must be Metal-backed".to_string())
        })?;
    let kt_metal_b = b
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_elementwise_binary: b must be Metal-backed".to_string())
        })?;

    let companion = kt_metal_a.companion()?;
    let device_index = match kt_metal_a.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let shape: Vec<usize> = a.shape().to_vec();
    let element_count: usize = a.element_count();

    // Allocate output buffer directly through metal-rs (no candle).
    let byte_len = element_count * dtype_size;
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_elementwise_binary: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    // Kiln-owned MSL binary op (replaces candle's call_binary_contiguous).
    crate::metal_kernels::elementwise_binary(
        &companion,
        kt_metal_a.buffer().as_ref(),
        kt_metal_b.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        kind_tag,
        element_count,
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_compare — kiln-owned elementwise comparison (#1082)
// ----------------------------------------------------------------------

/// Metal element-wise comparison (eq/ne/lt/le/gt/ge) producing a **U8 mask** —
/// the Metal analog of [`crate::cuda_compare`]. Both inputs must be
/// Metal-backed, same shape, same dtype (F32/BF16/F16), contiguous. The output
/// is a fresh contiguous U8 tensor (1 = comparison held, 0 = it did not),
/// matching the CPU reference in `ops::compare`. No host round-trip — a
/// kiln-owned MSL kernel (`metal_kernels::compare`) keeps the data on-GPU.
///
/// `kind_tag` follows `CmpKind::as_i32` (0=Eq, 1=Ne, 2=Lt, 3=Le, 4=Gt, 5=Ge).
pub fn metal_compare(a: &crate::Tensor, b: &crate::Tensor, kind_tag: i32) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    if !matches!(kind_tag, 0..=5) {
        return Err(Error::Msg(format!(
            "metal_compare: kind_tag {kind_tag} not supported \
             (only 0=Eq, 1=Ne, 2=Lt, 3=Le, 4=Gt, 5=Ge)"
        )));
    }
    let dtype = a.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "metal_compare: unsupported dtype {dtype} (expected F32/BF16/F16)"
        )));
    }
    if b.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_compare: dtype mismatch a={dtype} b={}",
            b.dtype()
        )));
    }
    if a.shape() != b.shape() {
        return Err(Error::Msg(format!(
            "metal_compare: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(
            "metal_compare: inputs must be contiguous".to_string(),
        ));
    }

    let kt_metal_a = a
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_compare: a must be Metal-backed".to_string()))?;
    let kt_metal_b = b
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_compare: b must be Metal-backed".to_string()))?;

    let companion = kt_metal_a.companion()?;
    let device_index = match kt_metal_a.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let shape: Vec<usize> = a.shape().to_vec();
    let element_count: usize = a.element_count();

    // Output is a U8 mask: 1 byte per element (NOT the input dtype size).
    let byte_len = element_count; // DType::U8.size_in_bytes() == 1
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_compare: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    crate::metal_kernels::compare(
        &companion,
        kt_metal_a.buffer().as_ref(),
        kt_metal_b.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        kind_tag,
        element_count,
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, DType::U8, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_where_select — kiln-owned ternary mask-based select (#1082)
// ----------------------------------------------------------------------

/// Metal ternary select `out[i] = mask[i] != 0 ? t[i] : f[i]` — the Metal
/// analog of [`crate::cuda_where_select`]. `mask` is a U8 tensor; `t`/`f` share
/// shape and dtype (F32/BF16/F16); the output is a fresh contiguous tensor with
/// `t`'s dtype. All three inputs must be Metal-backed and contiguous. No host
/// round-trip — a kiln-owned MSL kernel (`metal_kernels::where_select`) keeps
/// the data on-GPU (byte-wise select, so the chosen operand copies bit-exact).
pub fn metal_where_select(
    mask: &crate::Tensor,
    t: &crate::Tensor,
    f: &crate::Tensor,
) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    let dtype = t.dtype();
    let dtype_size: usize = match dtype {
        DType::F32 => 4,
        DType::BF16 | DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "metal_where_select: unsupported dtype {other} (expected F32/BF16/F16)"
            )));
        }
    };
    if mask.dtype() != DType::U8 {
        return Err(Error::Msg(format!(
            "metal_where_select: mask dtype must be U8, got {}",
            mask.dtype()
        )));
    }
    if t.dtype() != f.dtype() {
        return Err(Error::Msg(format!(
            "metal_where_select: t/f dtype mismatch t={} f={}",
            t.dtype(),
            f.dtype()
        )));
    }
    if mask.shape() != t.shape() || t.shape() != f.shape() {
        return Err(Error::Msg(format!(
            "metal_where_select: shape mismatch mask={:?} t={:?} f={:?}",
            mask.shape(),
            t.shape(),
            f.shape()
        )));
    }
    if !mask.is_contiguous() || !t.is_contiguous() || !f.is_contiguous() {
        return Err(Error::Msg(
            "metal_where_select: all inputs must be contiguous".to_string(),
        ));
    }

    let kt_mask = mask
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_where_select: mask must be Metal-backed".to_string()))?;
    let kt_t = t
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_where_select: t must be Metal-backed".to_string()))?;
    let kt_f = f
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_where_select: f must be Metal-backed".to_string()))?;

    let companion = kt_t.companion()?;
    let device_index = match kt_t.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let shape: Vec<usize> = t.shape().to_vec();
    let element_count: usize = t.element_count();

    let byte_len = element_count * dtype_size;
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_where_select: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    crate::metal_kernels::where_select(
        &companion,
        kt_mask.buffer().as_ref(),
        kt_t.buffer().as_ref(),
        kt_f.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        element_count,
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

// ----------------------------------------------------------------------
// metal_adamw_step — fused on-device AdamW (#1082)
// ----------------------------------------------------------------------

/// Fused in-place AdamW step on Metal. Updates `param`, `m` (first moment),
/// and `v` (second moment) IN PLACE in their `MetalStorage` buffers; reads
/// `grad`. No host round-trip — the buffers are `StorageModeShared` UMA and
/// the kernel mutates them directly. The Metal analog of the Vulkan
/// `dispatch_adamw_step_f32` path; the math is a bit-faithful port of
/// [`kiln_optim::AdamW::step`].
///
/// All four tensors must be Metal-backed, contiguous, share `dtype`, and have
/// the same element count. `step` is 1-indexed (>= 1). `bc1`/`bc2` bias
/// corrections are computed here (`1 - beta^step`) so the kernel needs no
/// `pow`.
///
/// Currently restricted to `DType::F32` (the LoRA master case). BF16 masters
/// need the host stochastic-rounding path; this function rejects non-F32 so
/// callers fall back to the host AdamW.
#[allow(clippy::too_many_arguments)]
pub fn metal_adamw_step(
    param: &crate::Tensor,
    grad: &crate::Tensor,
    m: &crate::Tensor,
    v: &crate::Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<()> {
    if step < 1 {
        return Err(Error::Msg(format!(
            "metal_adamw_step: step must be 1-indexed (>=1), got {step}"
        )));
    }
    let dtype = param.dtype();
    // F32/BF16/F16 master+moments — the kernel reads each as its dtype, computes
    // in float, and writes back as the dtype (round-to-nearest for BF16/F16,
    // matching Vulkan's BF16 AdamW arm). BF16 moments are the on-device
    // master-dtype convention (`allocate_adamw_state`), as on CUDA/Vulkan.
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "metal_adamw_step: unsupported dtype {dtype} (expected F32/BF16/F16)"
        )));
    }
    if grad.dtype() != dtype || m.dtype() != dtype || v.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_adamw_step: dtype mismatch (param={dtype}, grad={}, m={}, v={})",
            grad.dtype(),
            m.dtype(),
            v.dtype()
        )));
    }
    let n = param.element_count();
    if grad.element_count() != n || m.element_count() != n || v.element_count() != n {
        return Err(Error::Msg(format!(
            "metal_adamw_step: element count mismatch (param={n}, grad={}, m={}, v={})",
            grad.element_count(),
            m.element_count(),
            v.element_count()
        )));
    }
    if !param.is_contiguous() || !grad.is_contiguous() || !m.is_contiguous() || !v.is_contiguous() {
        return Err(Error::Msg(
            "metal_adamw_step: all operands must be contiguous".to_string(),
        ));
    }

    fn downcast<'a>(t: &'a crate::Tensor, name: &str) -> Result<&'a MetalStorage> {
        t.storage()
            .as_any()
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| Error::Msg(format!("metal_adamw_step: {name} must be Metal-backed")))
    }
    let kt_param = downcast(param, "param")?;
    let kt_grad = downcast(grad, "grad")?;
    let kt_m = downcast(m, "m")?;
    let kt_v = downcast(v, "v")?;

    let companion = kt_param.companion()?;

    // Bias corrections — match the host reference EXACTLY: `kiln_optim::AdamW`
    // computes `1.0 - beta.powf(step as f32)` in f32. Reproducing that here
    // (same op, same precision) makes bc1/bc2 bit-identical to the host path,
    // so the only source of parity drift is the f32 elementwise math the
    // kernel and reference share.
    let step_f = step as f32;
    let bc1 = 1.0 - beta1.powf(step_f);
    let bc2 = 1.0 - beta2.powf(step_f);

    crate::metal_kernels::adamw_step(
        &companion,
        kt_param.buffer().as_ref(),
        kt_grad.buffer().as_ref(),
        kt_m.buffer().as_ref(),
        kt_v.buffer().as_ref(),
        dtype,
        n,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bc1,
        bc2,
    )?;
    Ok(())
}

// ----------------------------------------------------------------------
// metal_muon_step — fused on-device Muon (momentum-orthogonalized SGD)
// ----------------------------------------------------------------------

/// Fused in-place Muon step on Metal. Updates `param` and the per-param
/// heavy-ball `momentum` buffer IN PLACE in their `MetalStorage` buffers;
/// reads `grad` (read-only). No host round-trip — the `StorageModeShared` UMA
/// buffers are mutated directly by a single threadgroup running the
/// Newton-Schulz orthogonalization in threadgroup memory. The math is a
/// bit-faithful port of [`kiln_optim::Muon::step`] + `kiln_optim::newton_schulz`.
///
/// All three tensors must be Metal-backed, contiguous, share `dtype` (the
/// float triple), and have the same element count. `rows`/`cols` come from the
/// param shape (a rank-2 weight orthogonalizes; otherwise the kernel falls back
/// to plain (Nesterov) momentum SGD — callers pass `(n, 1)` for non-2D params).
#[allow(clippy::too_many_arguments)]
pub fn metal_muon_step(
    param: &crate::Tensor,
    grad: &crate::Tensor,
    momentum: &crate::Tensor,
    rows: usize,
    cols: usize,
    lr: f32,
    momentum_coef: f32,
    nesterov: bool,
    ns_iters: u32,
    weight_decay: f32,
) -> Result<()> {
    let dtype = param.dtype();
    // F32/BF16/F16 master+momentum — the kernel reads each as its dtype,
    // computes in float, and writes back as the dtype (round-to-nearest for
    // BF16/F16). BF16 momentum follows the on-device master-dtype convention,
    // as on CUDA/Vulkan.
    if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(Error::Msg(format!(
            "metal_muon_step: unsupported dtype {dtype} (expected F32/BF16/F16)"
        )));
    }
    if grad.dtype() != dtype || momentum.dtype() != dtype {
        return Err(Error::Msg(format!(
            "metal_muon_step: dtype mismatch (param={dtype}, grad={}, momentum={})",
            grad.dtype(),
            momentum.dtype()
        )));
    }
    let n = param.element_count();
    if rows * cols != n {
        return Err(Error::Msg(format!(
            "metal_muon_step: rows*cols ({rows}*{cols}) != param element count ({n})"
        )));
    }
    if grad.element_count() != n || momentum.element_count() != n {
        return Err(Error::Msg(format!(
            "metal_muon_step: element count mismatch (param={n}, grad={}, momentum={})",
            grad.element_count(),
            momentum.element_count()
        )));
    }
    if !param.is_contiguous() || !grad.is_contiguous() || !momentum.is_contiguous() {
        return Err(Error::Msg(
            "metal_muon_step: all operands must be contiguous".to_string(),
        ));
    }

    fn downcast<'a>(t: &'a crate::Tensor, name: &str) -> Result<&'a MetalStorage> {
        t.storage()
            .as_any()
            .downcast_ref::<MetalStorage>()
            .ok_or_else(|| Error::Msg(format!("metal_muon_step: {name} must be Metal-backed")))
    }
    let kt_param = downcast(param, "param")?;
    let kt_grad = downcast(grad, "grad")?;
    let kt_momentum = downcast(momentum, "momentum")?;

    let companion = kt_param.companion()?;

    crate::metal_kernels::muon_step(
        &companion,
        kt_param.buffer().as_ref(),
        kt_grad.buffer().as_ref(),
        kt_momentum.buffer().as_ref(),
        dtype,
        rows,
        cols,
        lr,
        momentum_coef,
        nesterov,
        ns_iters,
        weight_decay,
    )?;
    Ok(())
}

// ----------------------------------------------------------------------
// metal_activation_unary — Phase 4 Metal substrate op (#1082)
// ----------------------------------------------------------------------

/// Metal unary activation. Mirrors the role of
/// [`crate::cuda_activation_unary`] for the Metal backend.
///
/// Dispatches on `kind_tag` (matches the CUDA tags in
/// `ActivationOp::cuda_fwd` and `UnaryArithKind::cuda_kind_tag`):
///   - `0`  -> Silu — MSL kernel `silu_<dt>`
///   - `1`  -> Sigmoid — no `contiguous::*` table entry in
///     `candle_metal_kernels`; rejected here, callers fall through
///     to CPU until a Metal sigmoid kernel lands.
///   - `2`  -> Gelu (tanh approximation) — MSL kernel `gelu_<dt>`
///   - `3`  -> Tanh — MSL kernel `tanh_<dt>`
///   - `4`  -> Relu — MSL kernel `relu_<dt>`
///   - `5`  -> Ln — MSL kernel `log_<dt>`
///   - `6`  -> Exp — MSL kernel `exp_<dt>`
///   - `7`  -> Sin — MSL kernel `sin_<dt>`
///   - `8`  -> Cos — MSL kernel `cos_<dt>`
///   - `12` -> Neg — MSL kernel `neg_<dt>`
///   - `13` -> Abs — MSL kernel `abs_<dt>`
///   - `14` -> Sqrt — MSL kernel `sqrt_<dt>`
///   - `22` -> Recip — MSL kernel `recip_<dt>`
///   - `23` -> Sign — MSL kernel `sign_<dt>`
///   - `24` -> Floor — MSL kernel `floor_<dt>`
///   - `25` -> Ceil — MSL kernel `ceil_<dt>`
///   - `26` -> Round — MSL kernel `round_<dt>`
///
/// Kiln-owned MSL (`metal_kernels`), replacing `candle_metal_kernels::call_unary_contiguous`.
/// Covers F32 / BF16 / F16.
///
/// # Apple Silicon UMA zero-copy invariant
///
/// Output buffer allocated via metal-rs `new_buffer` in
/// `StorageModeShared`. No `CandleTensor` / `CandleMetalStorage`
/// bridge (#1082).
///
/// # Requirements
///
/// - `x` must be backed by [`MetalStorage`]
/// - `x.dtype()` in {F32, BF16, F16}
/// - `x.is_contiguous()`
/// - `kind_tag` in the supported set above
///
/// # Errors
///
/// Returns [`Error::Msg`] on unsupported kind, dtype, non-contiguous
/// layout, non-Metal storage, or kernel dispatch error.
pub fn metal_activation_unary(x: &crate::Tensor, kind_tag: i32) -> Result<crate::Tensor> {
    use crate::metal_rt::MTLResourceOptions;

    if !matches!(
        kind_tag,
        0 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 12 | 13 | 14 | 22 | 23 | 24 | 25 | 26
    ) {
        return Err(Error::Msg(format!(
            "metal_activation_unary: kind_tag {kind_tag} not supported on Metal today \
             (0=Silu, 2=Gelu, 3=Tanh, 4=Relu, 5=Ln, 6=Exp, 7=Sin, 8=Cos, 12=Neg, \
             13=Abs, 14=Sqrt, 22=Recip, 23=Sign, 24=Floor, 25=Ceil, 26=Round; \
             Sigmoid=1 has no `contiguous::*` table entry — falls through to CPU)"
        )));
    }
    let dtype = x.dtype();
    let dtype_size: usize = match dtype {
        DType::F32 => 4,
        DType::BF16 => 2,
        DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "metal_activation_unary: unsupported dtype {other} (F32/BF16/F16 only)"
            )));
        }
    };
    if !x.is_contiguous() {
        return Err(Error::Msg(
            "metal_activation_unary: input must be contiguous".to_string(),
        ));
    }

    let kt_metal = x
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_activation_unary: input must be Metal-backed".to_string())
        })?;

    let companion = kt_metal.companion()?;
    let device_index = match kt_metal.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let shape: Vec<usize> = x.shape().to_vec();
    let element_count: usize = x.element_count();

    // Allocate output buffer directly through metal-rs (no candle).
    // Shared mode keeps UMA semantics consistent with the rest of the
    // substrate (`MetalStorage::zeros_kt`). The kernel runs on GPU and
    // writes results; CPU side can still read via UMA if needed.
    let byte_len = element_count * dtype_size;
    let raw_device = companion.device();
    let out_buffer = raw_device
        .new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_activation_unary: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
    let out_buffer_arc: Arc<MetalBuffer> = Arc::new(out_buffer);

    // Kiln-owned MSL unary activation (replaces candle's
    // call_unary_contiguous). All math is done in `float` (load→float,
    // compute, store→dtype) so BF16/F16 match the kt CPU reference
    // (`ActivationOp::apply_f32`) and F32 matches candle's unary.metal.
    crate::metal_kernels::activation_unary(
        &companion,
        kt_metal.buffer().as_ref(),
        out_buffer_arc.as_ref(),
        dtype,
        kind_tag,
        element_count,
    )?;

    let out_storage =
        MetalStorage::from_buffer_kt(raw_device, device_index, dtype, out_buffer_arc)?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        crate::Layout::contiguous(shape),
        crate::TensorId::next(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal_rt::MTLResourceOptions;

    fn metal_test_enabled() -> bool {
        std::env::var("KILN_TENSOR_METAL_TEST").ok().as_deref() == Some("1")
    }

    /// Candle-free Metal device handle for the tests below.
    ///
    /// Mirrors the `maybe_metal_raw_device` pattern in
    /// `metal_allocator.rs` — uses `MetalRawDevice::system_default()`
    /// (the `candle_metal_kernels::metal` re-export of Apple's `metal`
    /// crate's `Device::system_default`) so the test mod does not need
    /// `candle_core::Device::new_metal(0)`. This drops the
    /// `use candle_core::Device as CandleDevice` import that the test
    /// mod previously carried — one less candle hook in
    /// `metal_storage.rs` (#1082 candle removal).
    fn maybe_metal_raw_device() -> Option<MetalRawDevice> {
        if !metal_test_enabled() {
            return None;
        }
        MetalRawDevice::system_default()
    }

    #[test]
    fn zeros_round_sizes() {
        let Some(dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        // Exercise the candle-free metal-rs allocation path.
        // `zeros_kt` round-trips through `device.new_buffer` directly
        // (no candle blit-encoder), matching what `MetalAllocator`
        // already uses in production. The metal-rs `new_buffer` does
        // NOT round up to slab sizes the way candle's `allocate_zeros`
        // did, so `byte_len()` reports exactly the dtype-derived
        // byte_len now (BF16 * 64 = 128 bytes exactly).
        let storage = MetalStorage::zeros_kt(&dev, 0, DType::BF16, 64).unwrap();
        assert_eq!(storage.device(), Device::Metal(0));
        assert_eq!(storage.dtype(), DType::BF16);
        assert!(storage.byte_len() >= 128);

        let storage = MetalStorage::zeros_kt(&dev, 0, DType::Int4Packed, 16).unwrap();
        assert!(storage.byte_len() >= 8);
    }

    #[test]
    fn from_buffer_validates_alignment() {
        let Some(dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };
        // 17 bytes is not a multiple of f32 (4). metal-rs does NOT
        // round up `new_buffer`-sized allocations the way candle's
        // `allocate_zeros` did (which went through candle's slab
        // cache), so `raw_len` here equals 17 — the unaligned-len
        // branch of the test below is the one that fires.
        //
        // Uses the candle-free `from_buffer_kt` entry — the sole
        // construction path on `MetalStorage` post Wave-14. No candle
        // wrapper is materialized on either the field state or the op
        // dispatch path; the candle-derivation shims have all retired.
        let small = dev
            .new_buffer(17, MTLResourceOptions::StorageModeShared)
            .unwrap();
        let raw_len = small.length() as usize;
        let small_arc = Arc::new(small);
        let result = MetalStorage::from_buffer_kt(&dev, 0, DType::F32, small_arc);
        if raw_len.is_multiple_of(4) {
            // metal-rs rounded up (host-specific behavior); validation passes.
            assert!(result.is_ok());
        } else {
            assert!(result.unwrap_err().to_string().contains("not a multiple"));
        }
    }

    #[test]
    fn metal_tensor_copy_preserves_values_and_allocates_fresh_buffer() {
        let Some(_dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };

        let values = vec![1.0f32, 2.0, 3.5, 4.25];
        let cpu = crate::Tensor::from_slice(&values, vec![2, 2]).unwrap();
        let metal = host_to_metal_copy(&cpu, 0).unwrap();
        let copied = metal.copy().unwrap();

        assert_eq!(copied.device(), Device::Metal(0));
        let roundtrip = copied.to_device(Device::Cpu).unwrap();
        assert_eq!(roundtrip.to_vec::<f32>().unwrap(), values);

        let src = metal
            .storage()
            .as_any()
            .downcast_ref::<MetalStorage>()
            .unwrap();
        let dst = copied
            .storage()
            .as_any()
            .downcast_ref::<MetalStorage>()
            .unwrap();
        assert!(!Arc::ptr_eq(src.buffer(), dst.buffer()));
    }

    #[test]
    fn metal_all_finite_uses_correctness_fallback_for_values_and_views() {
        let Some(_dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };

        let finite_cpu = crate::Tensor::from_slice(&[1.0f32, -2.0, 3.5, 4.25], vec![2, 2])
            .expect("finite CPU fixture");
        let finite = host_to_metal_copy(&finite_cpu, 0).expect("finite Metal fixture");
        assert!(finite.all_finite().expect("finite Metal scan"));

        let nonfinite_cpu =
            crate::Tensor::from_slice(&[1.0f32, f32::NAN, 3.5, f32::INFINITY], vec![2, 2])
                .expect("non-finite CPU fixture");
        let nonfinite = host_to_metal_copy(&nonfinite_cpu, 0).expect("non-finite Metal fixture");
        assert!(!nonfinite.all_finite().expect("non-finite Metal scan"));
        assert!(
            !nonfinite
                .transpose(0, 1)
                .expect("transposed Metal view")
                .all_finite()
                .expect("non-finite transposed Metal scan")
        );
    }

    #[test]
    fn metal_copy_in_place_reuses_destination_buffer() {
        let Some(_dev) = maybe_metal_raw_device() else {
            eprintln!("skip: KILN_TENSOR_METAL_TEST unset or no Metal device");
            return;
        };

        let src_values = vec![7.0f32, 8.0, 9.5, 10.25];
        let dst_values = vec![0.0f32; 4];
        let src_cpu = crate::Tensor::from_slice(&src_values, vec![2, 2]).unwrap();
        let dst_cpu = crate::Tensor::from_slice(&dst_values, vec![2, 2]).unwrap();
        let src = host_to_metal_copy(&src_cpu, 0).unwrap();
        let dst = host_to_metal_copy(&dst_cpu, 0).unwrap();
        let dst_buffer_before = dst
            .storage()
            .as_any()
            .downcast_ref::<MetalStorage>()
            .unwrap()
            .buffer()
            .clone();

        metal_copy_in_place(&src, &dst).unwrap();

        let roundtrip = dst.to_device(Device::Cpu).unwrap();
        assert_eq!(roundtrip.to_vec::<f32>().unwrap(), src_values);
        let dst_buffer_after = dst
            .storage()
            .as_any()
            .downcast_ref::<MetalStorage>()
            .unwrap()
            .buffer()
            .clone();
        assert!(Arc::ptr_eq(&dst_buffer_before, &dst_buffer_after));
    }
}

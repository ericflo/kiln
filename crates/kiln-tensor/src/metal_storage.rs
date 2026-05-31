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
//! `RESOURCE_OPTIONS` constant in `vendor/candle-core/src/metal_backend`).
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
use candle_metal_kernels::metal::Buffer as MetalBuffer;
use candle_metal_kernels::metal::Device as MetalRawDevice;

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
    /// `docs/issue-1082-tier-4-5-roadmap-2026-05-27.md`).
    pub fn zeros_kt(
        device: &MetalRawDevice,
        device_index: usize,
        dtype: DType,
        n_elements: usize,
    ) -> Result<Self> {
        use candle_metal_kernels::metal::MTLResourceOptions;

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
    /// documented in `docs/issue-1082-tier-4-5-roadmap-2026-05-27.md`).
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

static METAL_COMPANIONS: OnceLock<std::sync::Mutex<std::collections::HashMap<usize, Arc<crate::metal_types::MetalCompanion>>>> = OnceLock::new();

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
    let map = METAL_COMPANIONS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
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
    let devices = candle_metal_kernels::metal::Device::all();
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
    use candle_metal_kernels::metal::MTLResourceOptions;

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
        .ok_or_else(|| {
            Error::Msg("metal_to_host_copy: tensor must be Metal-backed".to_string())
        })?;
    // Host-read synchronization point: make GPU writes visible.
    metal.companion()?.wait_until_completed()?;

    let dtype = t.dtype();
    let per = dtype.size_in_bytes();
    let buffer = metal.buffer();
    let buf_len = buffer.length() as usize;
    // SAFETY: Shared-mode buffer `contents()` is CPU-addressable and valid
    // for `buf_len` bytes; we only read within `[0, buf_len)`.
    let backing: &[u8] = unsafe {
        core::slice::from_raw_parts(buffer.contents() as *const u8, buf_len)
    };

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
/// Calls directly into `candle_metal_kernels::call_last_softmax` (the
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
    use candle_metal_kernels::metal::MTLResourceOptions;

    // ---- Validate kt-side preconditions ----
    let dtype = x.dtype();
    let (dtype_size, kernel_name): (usize, &'static str) = match dtype {
        DType::F32 => (4, "softmax_f32"),
        DType::BF16 => (2, "softmax_bf16"),
        DType::F16 => (2, "softmax_f16"),
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

    let encoder = companion.command_encoder().map_err(|e| {
        Error::Msg(format!(
            "metal_softmax_last_axis: command_encoder() failed: {e:?}"
        ))
    })?;
    encoder.set_label("kt_metal_softmax_last_axis");

    // Direct `call_last_softmax` dispatch — the same MSL kernel
    // candle_nn::ops::softmax_last_dim resolves internally. Drops the
    // CandleTensor / CandleMetalStorage / candle_nn bridge (#1082).
    candle_metal_kernels::call_last_softmax(
        raw_device,
        &encoder,
        companion.kernels(),
        kernel_name,
        element_count,
        last_dim,
        kt_metal.buffer().as_ref(),
        0,
        out_buffer_arc.as_ref(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "metal_softmax_last_axis: call_last_softmax failed: {e:?}"
        ))
    })?;
    drop(encoder);

    let out_storage = MetalStorage::from_buffer_kt(
        raw_device,
        device_index,
        dtype,
        out_buffer_arc,
    )?;
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
/// `docs/metal-types-objc2-swap-plan-2026-05-28.md`. It is the **only**
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
/// `candle_metal_kernels::call_sdpa_*` MSL kernel family — the same FFI
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
    use candle_metal_kernels::metal::MTLResourceOptions;
    use candle_metal_kernels::SdpaDType;

    // ---- Validate kt-side preconditions ----
    let dtype = q.dtype();
    let (dtype_size, sdpa_dtype): (usize, SdpaDType) = match dtype {
        DType::F32 => (4, SdpaDType::F32),
        DType::BF16 => (2, SdpaDType::BF16),
        DType::F16 => (2, SdpaDType::F16),
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
    if !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() {
        return Err(Error::Msg(
            "metal_sdpa_last_axis: q/k/v must all be contiguous".to_string(),
        ));
    }

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

    let supported_head_dim = matches!(
        head_dim,
        32 | 64 | 72 | 80 | 96 | 128 | 256 | 512
    );
    if !supported_head_dim {
        return Err(Error::Msg(format!(
            "metal_sdpa_last_axis: head_dim {head_dim} not supported \
             (expected one of 32, 64, 72, 80, 96, 128, 256, 512); \
             q dims {:?}, k dims {:?}, v dims {:?}",
            q_dims, k_dims, v_dims
        )));
    }

    // F32 full attention at head_dim=512 exceeds 32KB Metal threadgroup memory.
    let supports_sdpa_full_dtype = !(head_dim == 512 && dtype == DType::F32);
    let supports_sdpa_full = q_seq > 8 && supports_sdpa_full_dtype;
    let supports_sdpa_vector = q_seq <= 8 && q_seq <= k_seq;

    if !supports_sdpa_full && !supports_sdpa_vector {
        return Err(Error::Msg(format!(
            "metal_sdpa_last_axis: q dims {:?}, k dims {:?}, v dims {:?} \
             not supported by vector or full SDPA kernels",
            q_dims, k_dims, v_dims
        )));
    }

    // ---- Storage downcast ----
    let q_metal = q
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_sdpa_last_axis: q must be Metal-backed".to_string())
        })?;
    let k_metal = k
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_sdpa_last_axis: k must be Metal-backed".to_string())
        })?;
    let v_metal = v
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| {
            Error::Msg("metal_sdpa_last_axis: v must be Metal-backed".to_string())
        })?;

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

    let encoder = companion.command_encoder().map_err(|e| {
        Error::Msg(format!(
            "metal_sdpa_last_axis: command_encoder() failed: {e:?}"
        ))
    })?;

    // ---- Dispatch ----
    // Mirrors `Sdpa::metal_fwd` in candle-nn 0.10.2 ops.rs:1080-1226.
    // The wire-level FFI is identical; the kt-typed signature is the
    // only added piece (#1082).
    if supports_sdpa_vector {
        // Route to the 2-pass fused attention when k seqlen is large.
        // (See MLX PR #1597.)
        const TWO_PASS_K_THRESHOLD: usize = 1024;
        if k_seq >= TWO_PASS_K_THRESHOLD {
            let mut intermediate_shape = [
                &out_shape[0..out_shape.len() - 2],
                &[candle_metal_kernels::SDPA_2PASS_BLOCKS],
                &[out_shape[out_shape.len() - 1]],
            ]
            .concat();
            let intermediate_elem: usize = intermediate_shape.iter().product();
            // The 2-pass intermediate is F32 (matches candle's
            // `device.new_buffer(..., DType::F32, ...)` path); use 4 bytes
            // per element here regardless of the q/k/v dtype.
            let intermediate = raw_device
                .new_buffer(
                    (intermediate_elem * 4).max(1),
                    MTLResourceOptions::StorageModeShared,
                )
                .map_err(|e| {
                    Error::Msg(format!(
                        "metal_sdpa_last_axis: new_buffer(intermediate) failed: {e:?}"
                    ))
                })?;
            let _ = intermediate_shape.pop().unwrap();
            let sums_maxs_elem: usize = intermediate_shape.iter().product();
            let sums = raw_device
                .new_buffer(
                    (sums_maxs_elem * 4).max(1),
                    MTLResourceOptions::StorageModeShared,
                )
                .map_err(|e| {
                    Error::Msg(format!(
                        "metal_sdpa_last_axis: new_buffer(sums) failed: {e:?}"
                    ))
                })?;
            let maxs = raw_device
                .new_buffer(
                    (sums_maxs_elem * 4).max(1),
                    MTLResourceOptions::StorageModeShared,
                )
                .map_err(|e| {
                    Error::Msg(format!(
                        "metal_sdpa_last_axis: new_buffer(maxs) failed: {e:?}"
                    ))
                })?;

            encoder.set_label("kt_metal_sdpa_vector_2pass");
            candle_metal_kernels::call_sdpa_vector_2pass(
                raw_device,
                &encoder,
                companion.kernels(),
                q.layout().start_offset(),
                q_dims,
                q_metal.buffer().as_ref(),
                k.layout().start_offset(),
                k_dims,
                k.layout().strides(),
                k_metal.buffer().as_ref(),
                v.layout().start_offset(),
                v.layout().strides(),
                v_metal.buffer().as_ref(),
                out_buffer_arc.as_ref(),
                &intermediate,
                &sums,
                &maxs,
                scale,
                1.0, // softcapping disabled
                sdpa_dtype,
            )
            .map_err(|e| {
                Error::Msg(format!(
                    "metal_sdpa_last_axis: call_sdpa_vector_2pass failed: {e:?}"
                ))
            })?;
        } else {
            encoder.set_label("kt_metal_sdpa_vector");
            candle_metal_kernels::call_sdpa_vector(
                raw_device,
                &encoder,
                companion.kernels(),
                q.layout().start_offset(),
                q_dims,
                q_metal.buffer().as_ref(),
                k.layout().start_offset(),
                k_dims,
                k.layout().strides(),
                k_metal.buffer().as_ref(),
                v.layout().start_offset(),
                v.layout().strides(),
                v_metal.buffer().as_ref(),
                out_buffer_arc.as_ref(),
                scale,
                1.0, // softcapping disabled
                sdpa_dtype,
            )
            .map_err(|e| {
                Error::Msg(format!(
                    "metal_sdpa_last_axis: call_sdpa_vector failed: {e:?}"
                ))
            })?;
        }
    } else {
        // supports_sdpa_full
        encoder.set_label("kt_metal_sdpa_full");
        candle_metal_kernels::call_sdpa_full(
            raw_device,
            &encoder,
            companion.kernels(),
            q.layout().start_offset(),
            q_dims,
            q.layout().strides(),
            q_metal.buffer().as_ref(),
            k.layout().start_offset(),
            k_dims,
            k.layout().strides(),
            k_metal.buffer().as_ref(),
            v.layout().start_offset(),
            v_metal.buffer().as_ref(),
            v.layout().strides(),
            None, // mask_type
            None, // mask_buffer
            None, // m_strides
            out_buffer_arc.as_ref(),
            out_layout.strides(),
            scale,
            causal,
            sdpa_dtype,
        )
        .map_err(|e| {
            Error::Msg(format!(
                "metal_sdpa_last_axis: call_sdpa_full failed: {e:?}"
            ))
        })?;
    }
    drop(encoder);

    let out_storage = MetalStorage::from_buffer_kt(
        raw_device,
        device_index,
        dtype,
        out_buffer_arc,
    )?;
    let out_storage_arc: crate::Storage = Arc::new(out_storage);

    crate::Tensor::from_parts(
        out_storage_arc,
        out_layout,
        crate::TensorId::next(),
    )
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
/// Calls directly into `candle_metal_kernels::call_rms_norm` (the
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
    use candle_metal_kernels::metal::MTLResourceOptions;

    let dtype = x.dtype();
    let (dtype_size, kernel_name): (usize, &'static str) = match dtype {
        DType::F32 => (4, "rmsnorm_f32"),
        DType::BF16 => (2, "rmsnorm_bf16"),
        DType::F16 => (2, "rmsnorm_f16"),
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
        .ok_or_else(|| {
            Error::Msg("metal_rmsnorm_last_axis: x must be Metal-backed".to_string())
        })?;
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

    let encoder = companion.command_encoder().map_err(|e| {
        Error::Msg(format!(
            "metal_rmsnorm_last_axis: command_encoder() failed: {e:?}"
        ))
    })?;
    encoder.set_label("kt_metal_rmsnorm_last_axis");

    // Direct `call_rms_norm` dispatch — same MSL kernel candle_nn's
    // `rms_norm` resolves internally. Drops the
    // CandleTensor / CandleMetalStorage / candle_nn bridge (#1082).
    candle_metal_kernels::call_rms_norm(
        raw_device,
        &encoder,
        companion.kernels(),
        kernel_name,
        element_count_x,
        hidden,
        eps,
        kt_metal_x.buffer().as_ref(),
        0,
        kt_metal_w.buffer().as_ref(),
        0,
        out_buffer_arc.as_ref(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "metal_rmsnorm_last_axis: call_rms_norm failed: {e:?}"
        ))
    })?;
    drop(encoder);

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
/// Calls directly into `candle_metal_kernels::call_layer_norm` (the
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
    use candle_metal_kernels::metal::MTLResourceOptions;

    let dtype = x.dtype();
    let (dtype_size, kernel_name): (usize, &'static str) = match dtype {
        DType::F32 => (4, "layernorm_f32"),
        DType::BF16 => (2, "layernorm_bf16"),
        DType::F16 => (2, "layernorm_f16"),
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

    let encoder = companion.command_encoder().map_err(|e| {
        Error::Msg(format!(
            "metal_layernorm_last_axis: command_encoder() failed: {e:?}"
        ))
    })?;
    encoder.set_label("kt_metal_layernorm_last_axis");

    // Direct `call_layer_norm` dispatch — same MSL kernel candle_nn's
    // `layer_norm` resolves internally. Drops the
    // CandleTensor / CandleMetalStorage / candle_nn bridge (#1082).
    candle_metal_kernels::call_layer_norm(
        raw_device,
        &encoder,
        companion.kernels(),
        kernel_name,
        element_count_x,
        hidden,
        eps,
        kt_metal_x.buffer().as_ref(),
        0,
        kt_metal_w.buffer().as_ref(),
        0,
        kt_metal_b.buffer().as_ref(),
        0,
        out_buffer_arc.as_ref(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "metal_layernorm_last_axis: call_layer_norm failed: {e:?}"
        ))
    })?;
    drop(encoder);

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
/// Calls directly into `candle_metal_kernels::call_index_select` (the
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
    use candle_metal_kernels::metal::MTLResourceOptions;
    use candle_metal_kernels::BufferOffset;

    let dtype = input.dtype();
    let (dtype_size, kernel_name): (usize, &'static str) = match dtype {
        DType::F32 => (4, "is_u32_f32"),
        DType::BF16 => (2, "is_u32_bf16"),
        DType::F16 => (2, "is_u32_f16"),
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

    // Compute contiguous strides over the source dims (call_index_select
    // takes `src_strides` even on the contiguous-true path so that the
    // kernel can index along the gather dim correctly).
    let src_strides: Vec<usize> = {
        let mut s = vec![1usize; in_shape.len()];
        for i in (0..in_shape.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * in_shape[i + 1];
        }
        s
    };

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

    let encoder = companion.command_encoder().map_err(|e| {
        Error::Msg(format!(
            "metal_index_select_dim0: command_encoder() failed: {e:?}"
        ))
    })?;
    encoder.set_label("kt_metal_index_select_dim0");

    let src = BufferOffset {
        buffer: kt_metal_in.buffer().as_ref(),
        offset_in_bytes: 0,
    };
    let ids = BufferOffset {
        buffer: kt_metal_ids.buffer().as_ref(),
        offset_in_bytes: 0,
    };

    // Direct `call_index_select` dispatch — same MSL kernel candle's
    // `Tensor::index_select` resolves internally. Drops the
    // CandleTensor / CandleMetalStorage bridge (#1082).
    //
    // `shape` parameter is the input shape (the kernel uses it for
    // bounds + axis_dim lookup). `contiguous=true` because we gate
    // on input.is_contiguous() above.
    candle_metal_kernels::call_index_select(
        raw_device,
        &encoder,
        companion.kernels(),
        kernel_name,
        in_shape.as_slice(),
        ids_element_count,
        0, // dim
        true, // contiguous
        in_shape.as_slice(),
        src_strides.as_slice(),
        src,
        ids,
        out_buffer_arc.as_ref(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "metal_index_select_dim0: call_index_select failed: {e:?}"
        ))
    })?;
    drop(encoder);

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
/// Calls directly into `candle_metal_kernels::call_cast_contiguous`
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
    use candle_metal_kernels::metal::MTLResourceOptions;

    let from = x.dtype();
    // Float triple only; integer round-trips stay on the CPU fallback
    // (callers' metal_fwd falls through). `metal_kernels::msl_ty` gates
    // the supported dtypes.
    let to_dtype_size: usize = match to {
        DType::F32 => 4,
        DType::BF16 | DType::F16 => 2,
        other => {
            return Err(Error::Msg(format!(
                "metal_cast: unsupported target dtype {other} (float triple only)"
            )));
        }
    };
    // Reject unsupported source dtypes up front (mirror of the old match).
    crate::metal_kernels::msl_ty(from)?;
    crate::metal_kernels::msl_ty(to)?;

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
        .map_err(|e| {
            Error::Msg(format!(
                "metal_cast: new_buffer({byte_len}) failed: {e:?}"
            ))
        })?;
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
/// Calls directly into `candle_metal_kernels::call_binary_contiguous`.
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
    use candle_metal_kernels::metal::MTLResourceOptions;
    use candle_metal_kernels::BufferOffset;

    if !matches!(kind_tag, 0 | 1 | 2 | 3) {
        return Err(Error::Msg(format!(
            "metal_elementwise_binary: kind_tag {kind_tag} not supported \
             (only 0=Add, 1=Sub, 2=Mul, 3=Div)"
        )));
    }
    let dtype = a.dtype();
    let (dtype_size, dtype_suffix): (usize, &'static str) = match dtype {
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
        .ok_or_else(|| Error::Msg("metal_elementwise_binary: a must be Metal-backed".to_string()))?;
    let kt_metal_b = b
        .storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg("metal_elementwise_binary: b must be Metal-backed".to_string()))?;

    let companion = kt_metal_a.companion()?;
    let device_index = match kt_metal_a.device() {
        Device::Metal(i) => i,
        _ => unreachable!("MetalStorage::device() returns Device::Metal"),
    };
    let shape: Vec<usize> = a.shape().to_vec();
    let element_count: usize = a.element_count();

    // Resolve (kind_tag, dtype) -> Metal kernel name. These are the
    // same `badd_<dt>` / `bsub_<dt>` / `bmul_<dt>` / `bdiv_<dt>`
    // MSL kernels that candle's `&a + &b` / `*` / `-` / `/`
    // operators end up dispatching through internally — we go
    // straight to `call_binary_contiguous` and skip the
    // `CandleTensor` / `CandleMetalStorage` bridge.
    //
    // candle_metal_kernels::binary::contiguous defines named
    // sub-modules (`badd`, `bsub`, `bmul`, `bdiv`) with FLOAT /
    // HALF / BFLOAT constants. Their kernel string is
    // `concat!(stringify!(<name>), "_<dt>")`, so we can equivalently
    // construct it from (op_prefix, dtype_suffix) — that keeps the
    // dispatch table flat instead of a 4x3 nested-match.
    let op_prefix = match kind_tag {
        0 => "badd",
        1 => "bsub",
        2 => "bmul",
        3 => "bdiv",
        _ => unreachable!("gated by outer matches!()"),
    };
    let kernel_name = format!("{op_prefix}_{dtype_suffix}");

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

    let encoder = companion.command_encoder().map_err(|e| {
        Error::Msg(format!(
            "metal_elementwise_binary: command_encoder() failed: {e:?}"
        ))
    })?;
    encoder.set_label("kt_metal_elementwise_binary");

    let left = BufferOffset {
        buffer: kt_metal_a.buffer().as_ref(),
        offset_in_bytes: 0,
    };
    let right = BufferOffset {
        buffer: kt_metal_b.buffer().as_ref(),
        offset_in_bytes: 0,
    };

    candle_metal_kernels::call_binary_contiguous(
        raw_device,
        &encoder,
        companion.kernels(),
        kernel_name.as_str(),
        dtype_size,
        element_count,
        left,
        right,
        out_buffer_arc.as_ref(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "metal_elementwise_binary: call_binary_contiguous(kind={kind_tag}) failed: {e:?}"
        ))
    })?;
    drop(encoder);

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
/// Calls directly into `candle_metal_kernels::call_unary_contiguous`.
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
    use candle_metal_kernels::metal::MTLResourceOptions;
    use candle_metal_kernels::unary::contiguous;
    use candle_metal_kernels::BufferOffset;

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

    // Resolve (kind, dtype) -> contiguous kernel name. These are the
    // same `unary_<op>_<dt>` MSL kernels that `CandleTensor::silu()` /
    // `.gelu()` / ... go through internally — we just skip the
    // candle_core dispatch layer (#1082 candle removal: drop the
    // `CandleTensor` / `CandleMetalStorage` bridge in this op).
    let kernel: contiguous::Kernel = match (kind_tag, dtype) {
        (0, DType::F32) => contiguous::silu::FLOAT,
        (0, DType::F16) => contiguous::silu::HALF,
        (0, DType::BF16) => contiguous::silu::BFLOAT,
        (2, DType::F32) => contiguous::gelu::FLOAT,
        (2, DType::F16) => contiguous::gelu::HALF,
        (2, DType::BF16) => contiguous::gelu::BFLOAT,
        (3, DType::F32) => contiguous::tanh::FLOAT,
        (3, DType::F16) => contiguous::tanh::HALF,
        (3, DType::BF16) => contiguous::tanh::BFLOAT,
        (4, DType::F32) => contiguous::relu::FLOAT,
        (4, DType::F16) => contiguous::relu::HALF,
        (4, DType::BF16) => contiguous::relu::BFLOAT,
        (5, DType::F32) => contiguous::log::FLOAT,
        (5, DType::F16) => contiguous::log::HALF,
        (5, DType::BF16) => contiguous::log::BFLOAT,
        (6, DType::F32) => contiguous::exp::FLOAT,
        (6, DType::F16) => contiguous::exp::HALF,
        (6, DType::BF16) => contiguous::exp::BFLOAT,
        (7, DType::F32) => contiguous::sin::FLOAT,
        (7, DType::F16) => contiguous::sin::HALF,
        (7, DType::BF16) => contiguous::sin::BFLOAT,
        (8, DType::F32) => contiguous::cos::FLOAT,
        (8, DType::F16) => contiguous::cos::HALF,
        (8, DType::BF16) => contiguous::cos::BFLOAT,
        (12, DType::F32) => contiguous::neg::FLOAT,
        (12, DType::F16) => contiguous::neg::HALF,
        (12, DType::BF16) => contiguous::neg::BFLOAT,
        (13, DType::F32) => contiguous::abs::FLOAT,
        (13, DType::F16) => contiguous::abs::HALF,
        (13, DType::BF16) => contiguous::abs::BFLOAT,
        (14, DType::F32) => contiguous::sqrt::FLOAT,
        (14, DType::F16) => contiguous::sqrt::HALF,
        (14, DType::BF16) => contiguous::sqrt::BFLOAT,
        (22, DType::F32) => contiguous::recip::FLOAT,
        (22, DType::F16) => contiguous::recip::HALF,
        (22, DType::BF16) => contiguous::recip::BFLOAT,
        (23, DType::F32) => contiguous::sign::FLOAT,
        (23, DType::F16) => contiguous::sign::HALF,
        (23, DType::BF16) => contiguous::sign::BFLOAT,
        (24, DType::F32) => contiguous::floor::FLOAT,
        (24, DType::F16) => contiguous::floor::HALF,
        (24, DType::BF16) => contiguous::floor::BFLOAT,
        (25, DType::F32) => contiguous::ceil::FLOAT,
        (25, DType::F16) => contiguous::ceil::HALF,
        (25, DType::BF16) => contiguous::ceil::BFLOAT,
        (26, DType::F32) => contiguous::round::FLOAT,
        (26, DType::F16) => contiguous::round::HALF,
        (26, DType::BF16) => contiguous::round::BFLOAT,
        _ => unreachable!("gated by outer matches!() and dtype guard"),
    };

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

    // Get a compute encoder from the candle MetalDevice. The device
    // wrapper is held purely for command-queue affinity here; the
    // encoder talks straight to metal-rs underneath.
    let encoder = companion.command_encoder().map_err(|e| {
        Error::Msg(format!(
            "metal_activation_unary: command_encoder() failed: {e:?}"
        ))
    })?;
    encoder.set_label("kt_metal_activation_unary");

    let input = BufferOffset {
        buffer: kt_metal.buffer().as_ref(),
        offset_in_bytes: 0,
    };

    candle_metal_kernels::call_unary_contiguous(
        raw_device,
        &encoder,
        companion.kernels(),
        kernel,
        dtype_size,
        element_count,
        input,
        out_buffer_arc.as_ref(),
    )
    .map_err(|e| {
        Error::Msg(format!(
            "metal_activation_unary: call_unary_contiguous(kind={kind_tag}) failed: {e:?}"
        ))
    })?;
    drop(encoder);

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
    use candle_metal_kernels::metal::MTLResourceOptions;

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
}

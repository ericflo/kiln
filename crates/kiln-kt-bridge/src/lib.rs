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

// Generic candle CustomOp shim for kt-typed forward+backward kernel pairs
// (#1082). Lets production-caller migrations be per-call-site mechanical
// transformations instead of bespoke per-kernel CustomOp wrappers.
//
// CustomOp1/2/3 + CudaStorage live behind candle-core's CUDA feature, so
// the shim is only compiled when BOTH the `cuda` AND `candle` features
// are on (it's a candle CustomOp). The kt-native helpers below stay
// unconditional so candle-free CUDA consumers can use them, and the
// pure `candle <-> kt` Device/DType enum-mapping helpers compile on any
// `candle` build (CUDA toolchain not required for the enum mappers).
// (#1082) Deleted `pub mod forward_op` (KtForwardOp1/2/3 candle CustomOp shims) — dead, no callers.

// (#1082) `inject_grad_shim` (the `InjectGradientCandleShim` candle CustomOp1 +
// `inject_gradient_via_shim`) is DELETED — its only user was the test-only
// `tape_bridge::inject_gradient_kt`, itself called only by the now-deleted
// `inject_gradient_parity` test. Gradient injection lives natively in
// `kiln_autograd::InjectGradientBackward`.

/// Re-export of `candle_core` (#1082). During the migration, consumer
/// crates that bridge to candle islands but don't carry candle as a
/// direct dependency (e.g. the `kiln-server` `kiln-bench` binary, whose
/// candle-core is dev-only) need to *name* candle types for explicit
/// signatures (`fn … -> Result<candle_core::Tensor>`) while the bridge
/// fns only let them be *inferred*. Naming via `kiln_kt_bridge::candle_core::…`
/// avoids adding a fresh direct candle dependency that the candle-drop
/// endgame would then have to remove again.
#[cfg(feature = "candle")]
pub use candle_core;

/// Phase 6a/CP-4 (#1082) — kt-tape → candle GradStore bridge.
///
/// Lets `kiln_autograd::Tape::backward` emit gradients into a
/// `candle_core::backprop::GradStore`. See module docs for the
/// "disjoint-walker" problem this bridges. Wired into the production
/// `try_tape_{rms_norm,matmul,silu}_cuda` adapters in
/// `kiln-model::tape_forward` via the registration helpers
/// `register_input_mapping` / `register_output_mapping`. Gated on both
/// `cuda` (CudaStorage) and `candle` (GradStore). (#1082)
#[cfg(all(feature = "cuda", feature = "candle"))]
pub mod tape_bridge;

// `KtDType` is used by the candle-free CUDA helpers
// (`cuda_storage_and_byte_offset`, `alloc_cuda_tensor`,
// `cuda_input_device_ptr`) AND by the candle dtype mappers, so it's
// needed whenever either `cuda` or `candle` is on. `KtDevice` is only
// referenced by the candle-gated Device mappers
// (`kt_device_from_candle` / `candle_device_from_kt`). (#1082)
#[cfg(any(feature = "cuda", feature = "candle"))]
use kiln_tensor::DType as KtDType;
#[cfg(feature = "candle")]
use kiln_tensor::Device as KtDevice;
#[cfg(feature = "cuda")]
use kiln_tensor::{CudaStorage, StorageBackend, Tensor as KtTensor};

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
    if byte_off > byte_len {
        return Err(BridgeError::new(format!(
            "kt-bridge: {name} byte_off {byte_off} > storage byte_len {byte_len}"
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
#[cfg(feature = "candle")]
pub fn candle_dtype_to_kt(d: candle_core::DType) -> Result<KtDType, BridgeError> {
    use candle_core::DType as C;
    Ok(match d {
        C::F32 => KtDType::F32,
        C::BF16 => KtDType::BF16,
        C::F16 => KtDType::F16,
        C::U32 => KtDType::U32,
        // candle's `I32` and kt's `U32` are both 4-byte integers with
        // identical CUDA storage layout. kt-tensor today doesn't have a
        // dedicated I32 variant, but several kt-API entry points need to
        // accept candle tensors that are nominally `i32` for layout
        // reasons (e.g. Marlin's packed b_packed: the kernel treats the
        // bits as opaque packed 4-bit weights, never as signed ints).
        // Map I32 -> U32 so the borrow adapter can produce a kt-Tensor
        // pointing at the same device bytes. Callers that need actual
        // signed-i32 semantics must add a real KtDType::I32 first.
        C::I32 => KtDType::U32,
        C::U8 => KtDType::U8,
        C::I64 => KtDType::I64,
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge: unsupported candle dtype for kt conversion: {other:?}"
            )));
        }
    })
}

/// Map kt's `DType` to candle's `DType`. Returns `BridgeError` for
/// variants that have no candle equivalent (e.g. `F8E4M3`, `F8E5M2`,
/// `Int4Packed`, `Fp4Packed`).
///
/// Inverse of [`candle_dtype_to_kt`], modulo the irreversible `I32→U32`
/// candle-side leg used by Marlin. This direction can produce only the
/// dtypes candle natively supports.
///
/// Phase 7 of #1082 uses this to translate kt-typed inputs at
/// `kiln-model`'s public surface (the `_kt` parallel entries) into the
/// candle dtype the underlying call path still expects.
#[cfg(feature = "candle")]
pub fn kt_dtype_to_candle(d: KtDType) -> Result<candle_core::DType, BridgeError> {
    use candle_core::DType as C;
    Ok(match d {
        KtDType::F32 => C::F32,
        KtDType::BF16 => C::BF16,
        KtDType::F16 => C::F16,
        KtDType::U32 => C::U32,
        KtDType::U8 => C::U8,
        KtDType::I64 => C::I64,
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge: kt dtype {other:?} has no candle equivalent"
            )));
        }
    })
}

/// Map a candle `Device` to a `kiln_tensor::Device`.
///
/// Multi-GPU stays out of scope for #1082 (anti-pattern 12); the
/// `Cuda` / `Metal` variants carry a `device_index: usize` that this
/// adapter reads from `candle_core::DeviceLocation`.
///
/// `candle_core::Device::Cpu` maps to `kt::Device::Cpu`.
///
/// `candle_core::Device::Cuda(_)` maps to `kt::Device::Cuda(gpu_id)`
/// via `c.location()` — the cuda adapter machinery elsewhere in this
/// crate gates on `feature = "cuda"`, but this enum-mapping arm
/// compiles unconditionally because `candle_core::Device`/`Location`
/// are part of candle's base API.
///
/// `candle_core::Device::Metal(_)` maps to `kt::Device::Metal(gpu_id)`
/// **when this crate is built with `feature = "metal"`**, which
/// forwards `candle-core/metal` so that `MetalDevice::location()`
/// returns a real `DeviceLocation::Metal { gpu_id }` instead of the
/// `dummy_metal_backend` `fail!()` stub. Without the metal feature,
/// the Metal arm is skipped and any `Device::Metal(_)` falls into the
/// catch-all CPU branch — but constructing such a device requires
/// `Device::new_metal`, which itself errors on a non-metal build, so
/// reaching that branch with a Metal device is a contradiction in
/// practice.
///
/// Vulkan and any future candle backends degrade to `kt::Device::Cpu`.
/// The kiln-server Vulkan-active path carries a candle CPU device by
/// convention (see `kiln-model::backend::mod::for_device`).
#[cfg(feature = "candle")]
pub fn kt_device_from_candle(d: &candle_core::Device) -> KtDevice {
    use candle_core::backend::BackendDevice;
    use candle_core::DeviceLocation;
    match d {
        candle_core::Device::Cpu => KtDevice::Cpu,
        candle_core::Device::Cuda(c) => match c.location() {
            DeviceLocation::Cuda { gpu_id } => KtDevice::Cuda(gpu_id),
            // Cuda backend reports a non-Cuda location — degrade to CPU
            // rather than panic. Real call sites construct via
            // `candle_core::Device::new_cuda(i)` and always carry a
            // Cuda location; this arm is defensive only.
            _ => KtDevice::Cpu,
        },
        // Metal arm is gated on `feature = "metal"` because the
        // dummy_metal_backend's MetalDevice::location() `fail!()`s at
        // runtime. With the metal feature on, candle-core/metal is
        // pulled in and the real impl returns
        // `DeviceLocation::Metal { gpu_id }`. (#1082)
        #[cfg(feature = "metal")]
        candle_core::Device::Metal(m) => match m.location() {
            DeviceLocation::Metal { gpu_id } => KtDevice::Metal(gpu_id),
            // Metal backend reports a non-Metal location — defensive
            // arm; the real `MetalDevice::location()` impl always
            // returns `DeviceLocation::Metal`.
            _ => KtDevice::Cpu,
        },
        // Without `feature = "metal"`, the Metal arm cannot call
        // `m.location()` safely. Constructing such a device is
        // impossible on a non-metal build anyway (`Device::new_metal`
        // errors at runtime), so reaching this fallthrough with a
        // Metal device is a contradiction in practice. Vulkan and
        // other future backends also degrade to CPU.
        _ => KtDevice::Cpu,
    }
}

/// Map a `kiln_tensor::Device` to a candle `Device`.
///
/// Inverse of [`kt_device_from_candle`]. Returns `BridgeError` if the kt
/// Device variant has no candle equivalent on this build.
///
/// `Cuda(i)` calls `candle_core::Device::new_cuda(i)`, which on builds
/// without candle's cuda feature returns a runtime error from the
/// `dummy_cuda_backend` stub (this helper still compiles — only the
/// runtime arm fails). On `kiln-kt-bridge --features cuda` it
/// successfully constructs a real cuda device.
///
/// `Metal(i)` calls `candle_core::Device::new_metal(i)` **when this
/// crate is built with `feature = "metal"`** (which forwards
/// `candle-core/metal`). Without the metal feature the arm is omitted
/// and `KtDevice::Metal(_)` falls through to the catch-all that
/// surfaces a typed `BridgeError`. The kiln-server Metal path is
/// expected to enable this feature.
///
/// `Vulkan(_)` has no candle backend in any feature combination —
/// kiln-server's Vulkan path keeps a candle CPU device by convention.
///
/// Phase 7 of #1082 uses this to translate kt-typed inputs at
/// `kiln-model`'s public surface (the `_kt` parallel entries) into the
/// candle `Device` the underlying call path still expects.
#[cfg(feature = "candle")]
pub fn candle_device_from_kt(d: &KtDevice) -> Result<candle_core::Device, BridgeError> {
    match d {
        KtDevice::Cpu => Ok(candle_core::Device::Cpu),
        KtDevice::Cuda(i) => candle_core::Device::new_cuda(*i).map_err(|e| {
            BridgeError::new(format!(
                "kt-bridge: candle_device_from_kt: new_cuda({i}): {e}"
            ))
        }),
        // Inverse Metal arm — gated on `feature = "metal"`. Without
        // the feature, `Device::new_metal` would route to the
        // `dummy_metal_backend` `fail!()` stub; surface a typed
        // BridgeError instead. (#1082)
        #[cfg(feature = "metal")]
        KtDevice::Metal(i) => candle_core::Device::new_metal(*i).map_err(|e| {
            BridgeError::new(format!(
                "kt-bridge: candle_device_from_kt: new_metal({i}): {e}"
            ))
        }),
        other => Err(BridgeError::new(format!(
            "kt-bridge: candle_device_from_kt: kt Device {other:?} has no candle equivalent \
             on this build"
        ))),
    }
}

/// Construct a candle CUDA device with a graph-capturable stream + event
/// tracking disabled.
///
/// kiln-server's device-selection path opens its CUDA device via
/// `candle_core::Device::new_cuda_with_stream` so the resident
/// `cudaStream_t` can be captured into a CUDA graph (the hot decode
/// path). It then `unsafe { disable_event_tracking() }` on the inner
/// `CudaDevice` to suppress candle's per-op cuEventRecord — otherwise
/// the graph capture observes the events and bloats the captured
/// instruction stream.
///
/// Both calls live in candle's API surface; kiln-server's `device.rs`
/// used to name `candle_core::Device::new_cuda_with_stream` +
/// `Device::Cuda(d) => unsafe { d.disable_event_tracking() }`
/// inline, which kept `candle_core::*` symbols in that file's source.
/// Move that two-step setup behind this bridge helper so callers can
/// build the graph-ready candle CUDA device through a kt-typed
/// surface and drop their direct `candle_core` imports.
///
/// `cuda`-feature-gated because `CudaDevice::disable_event_tracking`
/// only exists in the real candle CUDA backend (the dummy stub omits
/// it). Callers that need the candle CUDA device on a non-cuda build
/// should use [`candle_device_from_kt`] with `KtDevice::Cuda(_)`
/// instead (no graph capture). (#1082)
#[cfg(all(feature = "cuda", feature = "candle"))]
pub fn candle_cuda_device_with_stream_no_event_tracking(
    ordinal: usize,
) -> Result<candle_core::Device, BridgeError> {
    let device = candle_core::Device::new_cuda_with_stream(ordinal).map_err(|e| {
        BridgeError::new(format!(
            "kt-bridge: candle_cuda_device_with_stream_no_event_tracking: \
             new_cuda_with_stream({ordinal}): {e}"
        ))
    })?;
    if let candle_core::Device::Cuda(cuda_device) = &device {
        // SAFETY: `disable_event_tracking` mutates per-device CUDA
        // bookkeeping. It is safe to call exactly once on a freshly
        // constructed `CudaDevice` before any operations are issued
        // against it. This helper is the canonical construction
        // path for the graph-capturable stream, so the freshness
        // invariant holds at every call site by construction.
        unsafe { cuda_device.disable_event_tracking() };
    }
    Ok(device)
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
/// No-CUDA (default-build) host bridge: candle CPU tensor → kt CPU tensor
/// (typed host copy). #1082: companion to the no-CUDA `kt_tensor_to_candle_cuda_copy`;
/// the kiln-train call sites invoke this unconditionally, so the default
/// (no-CUDA / linux-default CI) build needs an implementation under the same
/// name. Value-faithful host copy over F32/BF16/F16/U32/U8/I64.
#[cfg(all(not(feature = "cuda"), feature = "candle"))]
pub fn kt_tensor_from_candle_cuda_copy(
    t: &candle_core::Tensor,
) -> Result<kiln_tensor::Tensor, BridgeError> {
    // CPU build: copy and borrow are identical (both are a host copy). Reuse the
    // borrow variant's logic.
    kt_tensor_from_candle_cuda_borrow(t)
}

#[cfg(all(feature = "cuda", feature = "candle"))]
#[allow(clippy::needless_pass_by_value)]
pub fn kt_tensor_from_candle_cuda_copy(
    t: &candle_core::Tensor,
) -> Result<KtTensor, BridgeError> {
    use candle_core::{
        backend::{BackendDevice, BackendStorage},
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

    let candle_device = std::sync::Arc::new(cuda_st.device().clone());
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
    // #1082 wave 13: `cuda_zeros(Arc<CudaDevice>, ...)` was deleted in
    // favor of the candle-free `cuda_zeros_ctx(device_index, ...)`. The
    // local `candle_device` binding is still needed for the
    // `stream`/`raw_stream` reads above; only the kt-side allocation
    // moves to the candle-free entry.
    let dst_storage = kiln_tensor::cuda_zeros_ctx(device_index, kt_dtype, n_elems)
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
    // Per-dtype dispatch. The src CudaView and its guard must outlive
    // the memcpy call, so each arm binds them locally and issues its
    // own memcpy. Macro keeps each arm to one line of intent.
    macro_rules! dispatch_copy {
        ($T:ty, $name:literal) => {{
            let slice = cuda_st.as_cuda_slice::<$T>().map_err(|e| {
                BridgeError::new(format!(
                    "kt-bridge copy: as_cuda_slice {}: {e}",
                    $name
                ))
            })?;
            let src_view = slice.slice(off..);
            unsafe {
                let (dst_ptr, _dst_g) = dst_slice.device_ptr(&stream);
                let (src_ptr, _src_g) = src_view.device_ptr(&stream);
                cudarc_result::memcpy_dtod_async(dst_ptr, src_ptr, total_bytes, raw_stream)
                    .map_err(|e| {
                        BridgeError::new(format!("kt-bridge copy: memcpy_dtod_async: {e:?}"))
                    })?;
            }
        }};
    }
    match t.dtype() {
        C::F32 => dispatch_copy!(f32, "f32"),
        C::BF16 => dispatch_copy!(bf16, "bf16"),
        C::F16 => dispatch_copy!(f16, "f16"),
        C::U32 => dispatch_copy!(u32, "u32"),
        // I32 reinterpreted as U32 (same 4-byte layout); see
        // `candle_dtype_to_kt`.
        C::I32 => dispatch_copy!(i32, "i32"),
        C::U8 => dispatch_copy!(u8, "u8"),
        C::I64 => dispatch_copy!(i64, "i64"),
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge copy: unsupported candle dtype {other:?}"
            )));
        }
    }

    KtTensor::from_parts(
        dst_storage,
        kiln_tensor::Layout::contiguous(shape),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| BridgeError::new(format!("kt-bridge copy: wrap: {e}")))
}

/// (#1082 perf — Pattern A) Per-ordinal cache of the candle `CudaDevice`.
///
/// `candle_core::Device::new_cuda(i)` constructs a FRESH `CudaBlas`
/// (`cublasCreate`) + `CudaRng` handle on every call. The kt→candle
/// copy-back ([`kt_tensor_to_candle_cuda_copy`]) runs per gated op — dozens
/// of times per decoded token, hundreds per training step — so rebuilding the
/// cuBLAS/cuRAND handle each time is pure waste (audited as the #1 Pattern-A
/// tax). A `candle_core::Device` is `Arc`-cloneable, so we build one per
/// ordinal and hand out cheap `Arc`-bump clones thereafter (same cuBLAS handle
/// + stream — exactly what the inverse `..._from_candle_cuda_borrow` already
/// does via `cuda_st.device().clone()`). Same-ordinal candle CUDA devices
/// compare equal by `gpu_id`, so downstream candle ops are unaffected.
#[cfg(all(feature = "cuda", feature = "candle"))]
fn cached_candle_cuda_device(index: usize) -> Result<candle_core::Device, BridgeError> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<usize, candle_core::Device>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(dev) = guard.get(&index) {
        return Ok(dev.clone());
    }
    let dev = candle_core::Device::new_cuda(index).map_err(|e| {
        BridgeError::new(format!(
            "kt-bridge: cached_candle_cuda_device: new_cuda({index}): {e}"
        ))
    })?;
    guard.insert(index, dev.clone());
    Ok(dev)
}

/// Phase 7 candle→kt adapter — **inverse copying variant**.
///
/// Allocates a fresh candle `Tensor` of the same shape and dtype as
/// the kt-Tensor source and device-to-device memcpys the contents.
/// The returned candle Tensor owns its own CUDA allocation; the kt
/// source can drop or live independently.
///
/// Symmetric to [`kt_tensor_from_candle_cuda_copy`]; together the two
/// adapters let call sites round-trip values between the candle and
/// kt-Tensor type systems while the v2 zero-copy borrow path is
/// pending. Cost: one device-to-device memcpy per call.
///
/// **Requirements**:
/// - `t` must be backed by `kiln_tensor::CudaStorage`
/// - `t.is_contiguous()` must be true
/// - `t.dtype()` must round-trip through [`candle_dtype_to_kt`] in
///   reverse — `F32/BF16/F16/U32/U8/I64` are supported today.
///
/// The candle Tensor's layout is contiguous (start_offset = 0,
/// row-major strides). Non-contiguous kt sources must be made
/// contiguous before this call.
/// No-CUDA (default-build) host bridge: kt CPU tensor → candle CPU tensor via a
/// typed host round-trip. #1082: the CUDA variant below is `#[cfg(cuda)]`-only,
/// but the shared kiln-train GRPO/FLCE call sites invoke this unconditionally,
/// so the default (no-CUDA / `linux-default` CI) build needs an implementation
/// under the same name. On the CPU both tensors are host-resident; this is a
/// value-faithful copy across the type systems. Supported dtypes mirror the
/// CUDA variant (F32/BF16/F16/U32/U8/I64).
#[cfg(all(not(feature = "cuda"), feature = "candle"))]
pub fn kt_tensor_to_candle_cuda_copy(
    t: &kiln_tensor::Tensor,
) -> Result<candle_core::Tensor, BridgeError> {
    let ct;
    let t = if t.is_contiguous() {
        t
    } else {
        ct = t
            .contiguous()
            .map_err(|e| BridgeError::new(format!("kt->candle cpu: contiguous: {e}")))?;
        &ct
    };
    let shape: Vec<usize> = t.shape().to_vec();
    let dev = candle_core::Device::Cpu;
    macro_rules! bridge {
        ($E:ty) => {{
            let v: Vec<$E> = t
                .to_vec::<$E>()
                .map_err(|e| BridgeError::new(format!("kt->candle cpu to_vec: {e}")))?;
            candle_core::Tensor::from_vec(v, shape, &dev)
                .map_err(|e| BridgeError::new(format!("kt->candle cpu from_vec: {e}")))
        }};
    }
    match t.dtype() {
        KtDType::F32 => bridge!(f32),
        KtDType::BF16 => bridge!(half::bf16),
        KtDType::F16 => bridge!(half::f16),
        KtDType::U32 => bridge!(u32),
        KtDType::U8 => bridge!(u8),
        KtDType::I64 => bridge!(i64),
        other => Err(BridgeError::new(format!(
            "kt->candle cpu: unsupported dtype {other:?}"
        ))),
    }
}

#[cfg(all(feature = "cuda", feature = "candle"))]
pub fn kt_tensor_to_candle_cuda_copy(
    t: &KtTensor,
) -> Result<candle_core::Tensor, BridgeError> {
    use candle_core::{
        backend::BackendDevice,
        cuda_backend::cudarc::driver::{result as cudarc_result, DevicePtr},
        DType as C, DeviceLocation, Device as CDevice,
    };
    use half::{bf16, f16};

    if !t.is_contiguous() {
        return Err(BridgeError::new(
            "kt-bridge: kt_tensor_to_candle_cuda_copy: tensor must be contiguous",
        ));
    }
    let kt_dtype = t.dtype();
    let candle_dtype = match kt_dtype {
        KtDType::F32 => C::F32,
        KtDType::BF16 => C::BF16,
        KtDType::F16 => C::F16,
        KtDType::U32 => C::U32,
        KtDType::U8 => C::U8,
        KtDType::I64 => C::I64,
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge to_candle: unsupported kt dtype {other:?}"
            )));
        }
    };
    let shape: Vec<usize> = t.shape().to_vec();
    let n_elems: usize = shape.iter().product();
    let bytes_per_elem = kt_dtype.size_in_bytes();
    let total_bytes = n_elems * bytes_per_elem;

    let src_cuda = t
        .storage()
        .as_any()
        .downcast_ref::<CudaStorage>()
        .ok_or_else(|| {
            BridgeError::new("kt-bridge to_candle: source must be CUDA storage")
        })?;
    let src_byte_off = t.layout().start_offset() * bytes_per_elem;

    // #1082: derive a candle CudaDevice for the destination candle
    // Tensor::zeros allocation from the kt source's device index.
    //
    // This bridge function actually produces a candle Tensor, so a
    // candle CudaDevice wrapper is required for the Tensor::zeros call
    // below. Construct it directly via candle_core::Device::new_cuda
    // (the same call kiln_tensor::primary_cuda_device used to make)
    // so the kt side stops exporting a candle-typed accessor purely
    // for this bridge's benefit.
    let device_index = match src_cuda.device() {
        kiln_tensor::Device::Cuda(i) => i,
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge to_candle: expected Cuda kt device, got {other:?}"
            )));
        }
    };
    // #1082 (perf, Pattern A): reuse a cached candle CudaDevice per ordinal
    // instead of building a fresh cuBLAS+cuRAND handle on every copy-back.
    let candle_device = cached_candle_cuda_device(device_index)?;
    let candle_device_arc = match &candle_device {
        CDevice::Cuda(d) => std::sync::Arc::new(d.clone()),
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge to_candle: expected Cuda candle device, got {other:?}"
            )));
        }
    };
    match candle_device_arc.location() {
        DeviceLocation::Cuda { .. } => {}
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge to_candle: expected Cuda location, got {other:?}"
            )));
        }
    }

    // Allocate the destination candle Tensor (zeros; we overwrite via
    // dtod memcpy). `Tensor::zeros` returns a candle Tensor with its
    // own freshly-allocated CudaStorage.
    let dst = candle_core::Tensor::zeros(shape.as_slice(), candle_dtype, &candle_device)
        .map_err(|e| BridgeError::new(format!("kt-bridge to_candle: alloc dst: {e}")))?;

    let stream = candle_device_arc.cuda_stream();
    let raw_stream = stream.cu_stream();

    // Extract src + dst raw pointers and memcpy. Use the same per-dtype
    // dispatch pattern as the inverse adapter so each CudaView lives
    // long enough to satisfy the device_ptr guard.
    let (dst_storage_guard, _dst_layout) = dst.storage_and_layout();
    let dst_cuda = match &*dst_storage_guard {
        candle_core::Storage::Cuda(c) => c,
        _ => {
            return Err(BridgeError::new(
                "kt-bridge to_candle: candle Tensor::zeros didn't produce Cuda storage",
            ));
        }
    };
    // #1082: `device_ptr_raw()` works for both Owned and Borrowed storage
    // (unlike `slice()`, which panics on Borrowed). Returns the buffer-start
    // pointer; add the kt layout's start-offset bytes to reach the active
    // region. The dtod memcpy below is a pure byte copy, so a borrowed
    // source is always safe to read.
    let (src_base_ptr, _src_byte_len) = src_cuda.device_ptr_raw();
    let src_ptr = src_base_ptr + src_byte_off as u64;

    macro_rules! dispatch_dst_copy {
        ($T:ty, $name:literal) => {{
            let slice = dst_cuda.as_cuda_slice::<$T>().map_err(|e| {
                BridgeError::new(format!(
                    "kt-bridge to_candle: dst as_cuda_slice {}: {e}",
                    $name
                ))
            })?;
            let dst_view = slice.slice(0..);
            unsafe {
                let (dst_ptr, _dst_g) = dst_view.device_ptr(&stream);
                cudarc_result::memcpy_dtod_async(dst_ptr, src_ptr, total_bytes, raw_stream)
                    .map_err(|e| {
                        BridgeError::new(format!(
                            "kt-bridge to_candle: memcpy_dtod_async: {e:?}"
                        ))
                    })?;
            }
        }};
    }
    match candle_dtype {
        C::F32 => dispatch_dst_copy!(f32, "f32"),
        C::BF16 => dispatch_dst_copy!(bf16, "bf16"),
        C::F16 => dispatch_dst_copy!(f16, "f16"),
        C::U32 => dispatch_dst_copy!(u32, "u32"),
        C::U8 => dispatch_dst_copy!(u8, "u8"),
        C::I64 => dispatch_dst_copy!(i64, "i64"),
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge to_candle: unsupported candle dtype {other:?}"
            )));
        }
    }
    drop(dst_storage_guard);

    Ok(dst)
}

/// Phase 7 candle→kt adapter — **zero-copy borrow variant** (v2).
///
/// Wraps a candle CUDA `Tensor` as a kt-Tensor without copying the
/// device buffer. The returned kt-Tensor shares its CUDA memory with
/// the candle source; a clone of the candle `Tensor` is held as the
/// `_keep_alive` Arc inside the borrowed `CudaStorage` so the device
/// allocation stays valid for as long as the kt-Tensor lives.
///
/// Counterpart to [`kt_tensor_from_candle_cuda_copy`]; same input
/// contract (contiguous, CUDA, dtype maps via [`candle_dtype_to_kt`])
/// but **zero memcpys**.
///
/// # Migration story
///
/// Existing kt-API call sites use `CudaStorage::slice()`, which
/// panics on a `Borrowed` storage by design (see `cuda_storage.rs`).
/// A borrowed kt-Tensor can therefore only be passed to kt-API
/// functions that have been migrated to use the dtype/owner-aware
/// raw-pointer accessor — that migration is the next step after this
/// PR. Until then, the borrow adapter is useful for tests + the
/// allocator-pool integration; the copying adapter remains the
/// production-safe path for call sites that still reach `.slice()`.
///
/// # Contiguity contract
///
/// **The candle `Tensor` MUST be contiguous.** If it isn't, this
/// function returns `BridgeError`. Callers that may receive
/// non-contig views (typical example: anything fed from
/// `Tensor::narrow` on a non-trailing dim, or `.transpose`'d
/// inputs) MUST call `t.contiguous()` first and hold the resulting
/// tensor in a local for the lifetime of the kt borrow:
///
/// ```ignore
/// let a_c = a.contiguous()?;                                  // local — keeps the storage alive
/// let a_kt = kt_tensor_from_candle_cuda_borrow(&a_c)?;       // borrow from the materialized contig copy
/// ```
///
/// Already-contiguous inputs make `.contiguous()` a cheap Arc
/// clone (no copy), so the pattern is safe to apply unconditionally
/// at sites that may see either layout. The 2026-05-26 batched-GDN
/// regression (`2d9d4fc4`) was caused by three `gdn_decode_*` kt
/// call sites in `kiln-model::backend::cuda` skipping this step
/// while the sibling `gdn_gates` kt path applied it — see that
/// commit message for the failure mode.
/// No-CUDA (default-build) host bridge: candle CPU tensor → kt CPU tensor via a
/// typed host copy. #1082: companion to the no-CUDA `kt_tensor_to_candle_cuda_copy`
/// above. The "borrow" name is retained for call-site compatibility with the
/// CUDA zero-copy variant; on the CPU build this is a value-faithful copy, not
/// a view. Supported dtypes mirror the CUDA variant.
#[cfg(all(not(feature = "cuda"), feature = "candle"))]
pub fn kt_tensor_from_candle_cuda_borrow(
    t: &candle_core::Tensor,
) -> Result<kiln_tensor::Tensor, BridgeError> {
    let t = t
        .contiguous()
        .map_err(|e| BridgeError::new(format!("candle->kt cpu: contiguous: {e}")))?;
    let shape: Vec<usize> = t.dims().to_vec();
    macro_rules! bridge {
        ($E:ty) => {{
            let v: Vec<$E> = t
                .flatten_all()
                .and_then(|f| f.to_vec1::<$E>())
                .map_err(|e| BridgeError::new(format!("candle->kt cpu to_vec1: {e}")))?;
            kiln_tensor::Tensor::from_slice(&v, shape)
                .map_err(|e| BridgeError::new(format!("candle->kt cpu from_slice: {e}")))
        }};
    }
    match t.dtype() {
        candle_core::DType::F32 => bridge!(f32),
        candle_core::DType::BF16 => bridge!(half::bf16),
        candle_core::DType::F16 => bridge!(half::f16),
        candle_core::DType::U32 => bridge!(u32),
        candle_core::DType::U8 => bridge!(u8),
        candle_core::DType::I64 => bridge!(i64),
        other => Err(BridgeError::new(format!(
            "candle->kt cpu: unsupported dtype {other:?}"
        ))),
    }
}

#[cfg(all(feature = "cuda", feature = "candle"))]
pub fn kt_tensor_from_candle_cuda_borrow(
    t: &candle_core::Tensor,
) -> Result<KtTensor, BridgeError> {
    use candle_core::{
        backend::{BackendDevice, BackendStorage},
        cuda_backend::cudarc::driver::DevicePtr,
        DType as C, DeviceLocation, Storage as CStorage,
    };
    use half::{bf16, f16};

    if !t.is_contiguous() {
        return Err(BridgeError::new(
            "kt-bridge: kt_tensor_from_candle_cuda_borrow: tensor must be contiguous",
        ));
    }
    let kt_dtype = candle_dtype_to_kt(t.dtype())?;
    let shape: Vec<usize> = t.dims().to_vec();
    let n_elems: usize = shape.iter().product();
    let bytes_per_elem = kt_dtype.size_in_bytes();

    let (storage_guard, layout) = t.storage_and_layout();
    let cuda_st = match &*storage_guard {
        CStorage::Cuda(c) => c,
        _ => {
            return Err(BridgeError::new(
                "kt-bridge: kt_tensor_from_candle_cuda_borrow: tensor must be on CUDA",
            ))
        }
    };

    let candle_device_arc = std::sync::Arc::new(cuda_st.device().clone());
    let device_index = match candle_device_arc.location() {
        DeviceLocation::Cuda { gpu_id } => gpu_id,
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge borrow: expected Cuda location, got {other:?}"
            )));
        }
    };
    // #1082: derive the cudarc CudaContext from the candle device for
    // the `from_borrowed_ctx` candle-free constructor below. The candle
    // device handle is still used here only to read its underlying
    // primary context; once the candle Tensor adapter itself is
    // candle-free this whole function changes.
    let ctx = candle_device_arc.cuda_stream().context().clone();
    let stream = candle_device_arc.cuda_stream();
    let off = layout.start_offset();
    let byte_off = off * bytes_per_elem;
    let total_bytes = n_elems * bytes_per_elem;

    // Extract the raw device pointer at the live region's start. Per
    // dtype because as_cuda_slice<T>() is typed; we discard the
    // SyncOnDrop guard since we're producing a Borrowed kt storage
    // and don't have a stream/event to record against.
    macro_rules! src_ptr {
        ($T:ty, $name:literal) => {{
            let slice = cuda_st.as_cuda_slice::<$T>().map_err(|e| {
                BridgeError::new(format!(
                    "kt-bridge borrow: as_cuda_slice {}: {e}",
                    $name
                ))
            })?;
            let view = slice.slice(off..);
            let (ptr, _g) = unsafe { view.device_ptr(&stream) };
            ptr
        }};
    }
    let src_ptr = match t.dtype() {
        C::F32 => src_ptr!(f32, "f32"),
        C::BF16 => src_ptr!(bf16, "bf16"),
        C::F16 => src_ptr!(f16, "f16"),
        C::U32 => src_ptr!(u32, "u32"),
        // I32 reinterpreted as U32 — same 4-byte layout, see
        // `candle_dtype_to_kt` for the Marlin packed-i32 rationale.
        C::I32 => src_ptr!(i32, "i32"),
        C::U8 => src_ptr!(u8, "u8"),
        C::I64 => src_ptr!(i64, "i64"),
        other => {
            return Err(BridgeError::new(format!(
                "kt-bridge borrow: unsupported candle dtype {other:?}"
            )));
        }
    };

    // The kt borrowed storage points at the active region (with
    // start_offset already applied), so the kt layout sets
    // start_offset=0. byte_len = the active region's bytes.
    let _ = byte_off; // we baked the offset into src_ptr
    drop(storage_guard);

    // Keep-alive: clone the candle Tensor (cheap — Arc clone) and
    // wrap in an Arc so the kt side has a Send+Sync handle. Holding
    // the Tensor keeps its Arc<Tensor_> alive, which in turn keeps
    // its Arc<RwLock<Storage>> alive, which keeps the CudaSlice<T>
    // (and the underlying device allocation) alive.
    let keep_alive: std::sync::Arc<dyn std::any::Any + Send + Sync> =
        std::sync::Arc::new(t.clone());

    let storage = CudaStorage::from_borrowed_ctx(
        &ctx,
        device_index,
        kt_dtype,
        src_ptr,
        total_bytes,
        keep_alive,
    )
    .map_err(|e| BridgeError::new(format!("kt-bridge borrow: from_borrowed_ctx: {e}")))?;
    let storage_arc: kiln_tensor::Storage = std::sync::Arc::new(storage);

    KtTensor::from_parts(
        storage_arc,
        kiln_tensor::Layout::contiguous(shape),
        kiln_tensor::TensorId::next(),
    )
    .map_err(|e| BridgeError::new(format!("kt-bridge borrow: wrap: {e}")))
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

    #[cfg(feature = "candle")]
    #[test]
    fn dtype_mapping_round_trip() {
        assert_eq!(candle_dtype_to_kt(candle_core::DType::F32).unwrap(), KtDType::F32);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::BF16).unwrap(), KtDType::BF16);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::F16).unwrap(), KtDType::F16);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::U32).unwrap(), KtDType::U32);
        // I32 maps to U32 (4-byte storage reinterpret) for Marlin's
        // packed-i32 b_packed and any other opaque-bytes use case.
        assert_eq!(candle_dtype_to_kt(candle_core::DType::I32).unwrap(), KtDType::U32);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::U8).unwrap(), KtDType::U8);
        assert_eq!(candle_dtype_to_kt(candle_core::DType::I64).unwrap(), KtDType::I64);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn kt_dtype_to_candle_basic() {
        assert_eq!(kt_dtype_to_candle(KtDType::F32).unwrap(), candle_core::DType::F32);
        assert_eq!(kt_dtype_to_candle(KtDType::BF16).unwrap(), candle_core::DType::BF16);
        assert_eq!(kt_dtype_to_candle(KtDType::F16).unwrap(), candle_core::DType::F16);
        assert_eq!(kt_dtype_to_candle(KtDType::U32).unwrap(), candle_core::DType::U32);
        assert_eq!(kt_dtype_to_candle(KtDType::U8).unwrap(), candle_core::DType::U8);
        assert_eq!(kt_dtype_to_candle(KtDType::I64).unwrap(), candle_core::DType::I64);
        // FP8 and packed quantized variants have no candle counterpart.
        assert!(kt_dtype_to_candle(KtDType::F8E4M3).is_err());
        assert!(kt_dtype_to_candle(KtDType::Int4Packed).is_err());
    }

    #[cfg(feature = "candle")]
    #[test]
    fn kt_device_from_candle_cpu_roundtrip() {
        // The CPU arm has no GPU dependency — exercise it unconditionally.
        let d = candle_core::Device::Cpu;
        assert_eq!(kt_device_from_candle(&d), KtDevice::Cpu);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn candle_device_from_kt_cpu_roundtrip() {
        let d = candle_device_from_kt(&KtDevice::Cpu).unwrap();
        assert!(matches!(d, candle_core::Device::Cpu));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn candle_device_from_kt_vulkan_errors() {
        // Vulkan is unsupported through candle; this should surface a
        // BridgeError rather than silently degrade.
        let e = candle_device_from_kt(&KtDevice::Vulkan(0)).unwrap_err();
        assert!(e.to_string().contains("no candle equivalent"));
    }

    /// Without `feature = "metal"`, the inverse helper must surface a
    /// typed BridgeError instead of attempting `Device::new_metal`
    /// (which would route to the `dummy_metal_backend` `fail!()`
    /// stub). Requires `candle` since `candle_device_from_kt` is
    /// candle-gated. (#1082)
    #[cfg(all(feature = "candle", not(feature = "metal")))]
    #[test]
    fn candle_device_from_kt_metal_errors_without_feature() {
        let e = candle_device_from_kt(&KtDevice::Metal(0)).unwrap_err();
        assert!(
            e.to_string().contains("no candle equivalent"),
            "expected 'no candle equivalent', got: {e}"
        );
    }

    /// With `feature = "metal"`, the inverse helper takes the Metal
    /// arm and reaches `candle_core::Device::new_metal(_)` instead of
    /// the fallthrough that surfaces "no candle equivalent". The exact
    /// outcome of `new_metal` depends on the host:
    /// - macOS with a Metal-capable GPU (M-series runners with one
    ///   visible adapter): `Ok(Device::Metal(_))`.
    /// - macOS without an enumerable adapter (some headless CI VMs):
    ///   candle's metal backend currently `swap_remove(0)`s an empty
    ///   `Vec<MetalDevice>` and panics — see
    ///   `candle-core/src/metal_backend/mod.rs:1928`. That's an
    ///   upstream bug, not a kt-bridge regression; catching the panic
    ///   here lets us still assert that the Metal arm was taken.
    /// - macOS where `new_metal` returns a typed Err: the message will
    ///   come from the metal backend, never from the kt-bridge
    ///   fallthrough.
    /// Linux/Windows hosts won't compile this test because they can't
    /// enable `feature = "metal"` (objc2 is Apple-only). (#1082)
    #[cfg(feature = "metal")]
    #[test]
    fn candle_device_from_kt_metal_dispatches_with_feature() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            candle_device_from_kt(&KtDevice::Metal(0))
        }));
        match result {
            // Panic inside `new_metal` (e.g. empty-adapter-list
            // `swap_remove(0)` on a headless macOS runner) proves the
            // call reached candle's metal backend — that's the exact
            // dispatch we're asserting on. Suppress the panic to keep
            // the test green; the kt-bridge contract is "Metal arm
            // taken", not "new_metal succeeded".
            Err(_panic) => {}
            Ok(Ok(d)) => assert!(matches!(d, candle_core::Device::Metal(_))),
            Ok(Err(e)) => assert!(
                !e.to_string().contains("no candle equivalent"),
                "metal feature on but inverse helper still fell through to the \
                 unsupported-on-this-build branch: {e}"
            ),
        }
    }
}

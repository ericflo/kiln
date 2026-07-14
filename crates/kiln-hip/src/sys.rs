//! Raw FFI declarations for the subset of the HIP runtime API that kiln's
//! ROCm backend uses. Hand-written (not bindgen) so the crate compiles with no
//! ROCm headers present. The surface is limited to device queries, allocation,
//! copy, stream/event ordering, and graph lifecycle calls.
//!
//! Grounding: `hip/hip_runtime_api.h` (ROCm 6.x/7.x). HIP's runtime API mirrors
//! the CUDA runtime API one-for-one; device pointers are plain `void*`, streams
//! and graphs are opaque handle pointers.
//!
//! The extern block carries NO `#[link]` attribute on purpose — `build.rs`
//! emits `cargo:rustc-link-lib=amdhip64` only when a ROCm root is found, so
//! `cargo check` on a toolchain-less host never needs the library.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_uchar, c_uint, c_void};

/// HIP error code. `hipSuccess == 0`. Layout matches the C `hipError_t` enum
/// (a plain C int).
pub type hipError_t = c_int;

/// `hipSuccess` — the only success code.
pub const HIP_SUCCESS: hipError_t = 0;
/// `hipErrorInvalidValue` — local argument/context validation failed.
pub const HIP_ERROR_INVALID_VALUE: hipError_t = 1;
/// `hipErrorOutOfMemory` — recoverable device allocation exhaustion.
pub const HIP_ERROR_OUT_OF_MEMORY: hipError_t = 2;
/// `hipErrorPriorLaunchFailure` — a previous asynchronous launch failed.
pub const HIP_ERROR_PRIOR_LAUNCH_FAILURE: hipError_t = 53;
/// `hipErrorNotSupported` — the requested runtime API is unavailable.
pub const HIP_ERROR_NOT_SUPPORTED: hipError_t = 801;

/// HIP runtime copy direction. The C enum is passed as a 32-bit value.
pub type hipMemcpyKind = c_uint;
/// `hipMemcpyHostToDevice`.
pub const HIP_MEMCPY_HOST_TO_DEVICE: hipMemcpyKind = 1;

/// Opaque stream handle (`hipStream_t == ihipStream_t*`).
pub type hipStream_t = *mut c_void;
/// Opaque event handle (`hipEvent_t`).
pub type hipEvent_t = *mut c_void;
/// Opaque graph handle (`hipGraph_t`).
pub type hipGraph_t = *mut c_void;
/// Opaque executable-graph handle (`hipGraphExec_t`).
pub type hipGraphExec_t = *mut c_void;

/// Stream-creation flag: non-blocking with respect to the NULL stream
/// (`hipStreamNonBlocking`). Matches CUDA's `CU_STREAM_NON_BLOCKING`.
pub const HIP_STREAM_NON_BLOCKING: c_uint = 0x01;

/// Event-creation flag: do not collect timestamps. Ordering-only events avoid
/// timing overhead and are valid inputs to `hipStreamWaitEvent`.
pub const HIP_EVENT_DISABLE_TIMING: c_uint = 0x02;

/// `hipStreamCaptureModeRelaxed` — least-restrictive capture mode. Matches the
/// `CU_STREAM_CAPTURE_MODE_RELAXED` the CUDA graph path uses.
pub const HIP_STREAM_CAPTURE_MODE_RELAXED: c_uint = 2;

/// `hipStreamCaptureStatusActive` — a capture is in progress on the stream.
pub const HIP_STREAM_CAPTURE_STATUS_ACTIVE: c_uint = 1;

/// `hipGraphInstantiateFlagAutoFreeOnLaunch` — free stream-ordered allocations
/// on graph launch. Matches `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH`.
pub const HIP_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH: u64 = 2;

unsafe extern "C" {
    // --- device ---------------------------------------------------------
    pub fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    pub fn hipSetDevice(device_id: c_int) -> hipError_t;
    pub fn hipDeviceSynchronize() -> hipError_t;
    pub fn hipRuntimeGetVersion(version: *mut c_int) -> hipError_t;
    pub fn hipGetLastError() -> hipError_t;
    pub fn hipDeviceGetStreamPriorityRange(
        least_priority: *mut c_int,
        greatest_priority: *mut c_int,
    ) -> hipError_t;
    pub fn hipGetErrorString(error: hipError_t) -> *const c_char;

    // --- memory ---------------------------------------------------------
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    pub fn hipFree(ptr: *mut c_void) -> hipError_t;
    pub fn hipMallocAsync(ptr: *mut *mut c_void, size: usize, stream: hipStream_t) -> hipError_t;
    pub fn hipFreeAsync(ptr: *mut c_void, stream: hipStream_t) -> hipError_t;
    // Stream-ordered memory-pool config. `hipMemPool_t` is an opaque pointer.
    pub fn hipDeviceGetDefaultMemPool(pool: *mut *mut c_void, device: c_int) -> hipError_t;
    pub fn hipMemPoolSetAttribute(
        pool: *mut c_void,
        attr: c_uint,
        value: *mut c_void,
    ) -> hipError_t;
    // Read a pool attribute (e.g. ReservedMemCurrent / UsedMemCurrent). The
    // process-isolated way to measure how much VRAM kiln's own pool reserves vs
    // actively uses — immune to coexisting processes (unlike the DRM counters).
    pub fn hipMemPoolGetAttribute(
        pool: *mut c_void,
        attr: c_uint,
        value: *mut c_void,
    ) -> hipError_t;
    // Release pooled-but-unused memory back to the OS, keeping at least
    // `min_bytes_to_hold` cached. Safe to call only when no in-flight work is
    // touching the freed blocks (i.e. after a device/stream sync) — that's how
    // we return VRAM to the OS / a coexisting process without re-introducing the
    // async-free decode race that the release-threshold pin prevents.
    pub fn hipMemPoolTrimTo(pool: *mut c_void, min_bytes_to_hold: usize) -> hipError_t;
    // Free/total device memory (the device-API counterpart to the OS-level
    // sysfs probe). Useful for a discrete-GPU cross-check.
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;
    pub fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size_bytes: usize,
        kind: hipMemcpyKind,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemcpyHtoDAsync(
        dst: *mut c_void,
        src: *mut c_void,
        size_bytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemcpyDtoHAsync(
        dst: *mut c_void,
        src: *mut c_void,
        size_bytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemcpyDtoDAsync(
        dst: *mut c_void,
        src: *mut c_void,
        size_bytes: usize,
        stream: hipStream_t,
    ) -> hipError_t;
    pub fn hipMemsetD8Async(
        dest: *mut c_void,
        value: c_uchar,
        count: usize,
        stream: hipStream_t,
    ) -> hipError_t;

    // --- streams --------------------------------------------------------
    pub fn hipStreamCreateWithFlags(stream: *mut hipStream_t, flags: c_uint) -> hipError_t;
    pub fn hipStreamCreateWithPriority(
        stream: *mut hipStream_t,
        flags: c_uint,
        priority: c_int,
    ) -> hipError_t;
    pub fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;
    pub fn hipStreamWaitEvent(stream: hipStream_t, event: hipEvent_t, flags: c_uint) -> hipError_t;

    // --- events ---------------------------------------------------------
    pub fn hipEventCreateWithFlags(event: *mut hipEvent_t, flags: c_uint) -> hipError_t;
    pub fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    pub fn hipEventDestroy(event: hipEvent_t) -> hipError_t;

    // --- graphs (HIP graph capture; wired in Phase R.9) ------------------
    pub fn hipStreamBeginCapture(stream: hipStream_t, mode: c_uint) -> hipError_t;
    pub fn hipStreamEndCapture(stream: hipStream_t, p_graph: *mut hipGraph_t) -> hipError_t;
    pub fn hipStreamIsCapturing(stream: hipStream_t, capture_status: *mut c_uint) -> hipError_t;
    pub fn hipGraphInstantiateWithFlags(
        p_graph_exec: *mut hipGraphExec_t,
        graph: hipGraph_t,
        flags: u64,
    ) -> hipError_t;
    pub fn hipGraphLaunch(graph_exec: hipGraphExec_t, stream: hipStream_t) -> hipError_t;
    pub fn hipGraphExecDestroy(graph_exec: hipGraphExec_t) -> hipError_t;
    pub fn hipGraphDestroy(graph: hipGraph_t) -> hipError_t;
}

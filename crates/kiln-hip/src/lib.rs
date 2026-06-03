//! `kiln-hip` — bounded, safe Rust bindings to the AMD ROCm/HIP runtime.
//!
//! This is the **cudarc analog** for kiln's ROCm backend (Phase R.1). It mirrors
//! the exact ~8-symbol surface the CUDA substrate uses
//! (`CudaContext`/`CudaStream`/`CudaSlice`/`DevicePtr` + a few `result`/`sys`
//! free functions) so that `rocm_storage.rs` / `rocm_allocator.rs` (Phase R.3)
//! are mechanical retypes of the candle-free `cuda_*.rs` files.
//!
//! Design mirrors `kiln-tensor/src/cuda_stream_priority.rs`: own the raw HIP
//! handle, implement `Drop`, expose an FFI accessor (`hip_stream()`), and carry
//! `unsafe impl Send + Sync` with the same justification cudarc uses.
//!
//! The crate compiles on hosts with no ROCm toolchain (the FFI block has no
//! `#[link]`; `build.rs` links `amdhip64` only when ROCm is present). Calling a
//! function with no runtime present returns `Err(HipError)` rather than
//! aborting — except linking, which only a ROCm host performs.

pub mod sys;

use std::ffi::CStr;
use std::fmt;
use std::os::raw::{c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;

/// Result alias for HIP calls.
pub type Result<T> = std::result::Result<T, HipError>;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A failed HIP runtime call: the numeric `hipError_t`, the API symbol that
/// returned it, and the driver's human-readable string.
#[derive(Clone)]
pub struct HipError {
    /// The raw `hipError_t` code (`hipSuccess == 0` never appears here).
    pub code: i32,
    /// The HIP API function that returned the error (e.g. `"hipMalloc"`).
    pub api: &'static str,
    /// `hipGetErrorString(code)`, if resolvable.
    pub message: String,
}

impl fmt::Debug for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HipError({} from {}: {})", self.code, self.api, self.message)
    }
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} failed: {} (hipError {})", self.api, self.message, self.code)
    }
}

impl std::error::Error for HipError {}

/// Convert a raw `hipError_t` into a `Result`, attaching the API name and the
/// driver's error string.
#[inline]
fn check(code: sys::hipError_t, api: &'static str) -> Result<()> {
    if code == sys::HIP_SUCCESS {
        return Ok(());
    }
    // SAFETY: hipGetErrorString returns a static NUL-terminated string for any
    // code (an "unknown error" string for unrecognized codes). Returns a valid
    // pointer; never null in practice.
    let message = unsafe {
        let ptr = sys::hipGetErrorString(code);
        if ptr.is_null() {
            String::from("<no error string>")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    Err(HipError { code: code as i32, api, message })
}

/// The HIP runtime version (`hipRuntimeGetVersion`), or an error if no runtime
/// is linked/available. Encoded as `10_000_000*major + 100_000*minor + patch`
/// in recent ROCm.
pub fn runtime_version() -> Result<i32> {
    let mut v: c_int = 0;
    check(unsafe { sys::hipRuntimeGetVersion(&mut v) }, "hipRuntimeGetVersion")?;
    Ok(v)
}

/// Number of visible HIP devices, or `0` if the runtime reports none.
///
/// Never errors on a missing runtime in the "no device" sense — it surfaces the
/// underlying `hipError_t` so callers can distinguish "no GPU" from "no driver".
pub fn device_count() -> Result<i32> {
    let mut n: c_int = 0;
    check(unsafe { sys::hipGetDeviceCount(&mut n) }, "hipGetDeviceCount")?;
    Ok(n)
}

/// Best-effort availability probe: `true` iff the HIP runtime links and reports
/// at least one device. Swallows errors into `false` so callers can branch
/// without handling a `Result` (mirrors `cuda_is_available()` usage).
pub fn is_available() -> bool {
    matches!(device_count(), Ok(n) if n > 0)
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// A HIP device handle + its default stream. The runtime-API analog of
/// `cudarc::driver::CudaContext`.
///
/// HIP's runtime API has no explicit context object — the "context" is the
/// device ordinal, bound per-thread via `hipSetDevice`. `RocmContext` carries
/// the ordinal and an owned default stream so the surface matches cudarc's
/// `CudaContext { default_stream(), new_stream() }`.
#[derive(Debug)]
pub struct RocmContext {
    ordinal: c_int,
    default_stream: Arc<RocmStream>,
}

// SAFETY: the ordinal is a plain int; the default stream is itself Send+Sync
// (see RocmStream). Binding is per-thread via hipSetDevice on every call, so a
// RocmContext is safe to share across threads — the same reasoning cudarc uses
// for CudaContext.
unsafe impl Send for RocmContext {}
unsafe impl Sync for RocmContext {}

impl RocmContext {
    /// Create a context for device `ordinal`, validating that the runtime is
    /// present and the ordinal is in range, then creating a non-blocking
    /// default stream.
    pub fn new(ordinal: usize) -> Result<Arc<Self>> {
        let ordinal = ordinal as c_int;
        let count = device_count()?;
        if ordinal < 0 || ordinal >= count {
            return Err(HipError {
                code: -1,
                api: "RocmContext::new",
                message: format!("device ordinal {ordinal} out of range (count={count})"),
            });
        }
        check(unsafe { sys::hipSetDevice(ordinal) }, "hipSetDevice")?;
        // Pin the stream-ordered allocator's pool to NEVER release freed memory
        // back to the OS (hipMemPoolAttrReleaseThreshold = u64::MAX). The default
        // threshold (0) makes hipMallocAsync hand freed pages back aggressively;
        // under the decode alloc/free churn that races in-flight kernels and
        // corrupts output (it's what the paged-decode order-sync was masking).
        // Keeping freed blocks pooled removes the hazard AND avoids OS-roundtrip
        // alloc latency. Best-effort: ignore if the runtime lacks mempools.
        const HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD: c_uint = 4;
        let mut pool: *mut c_void = ptr::null_mut();
        if unsafe { sys::hipDeviceGetDefaultMemPool(&mut pool, ordinal) } == sys::HIP_SUCCESS
            && !pool.is_null()
        {
            let mut threshold: u64 = u64::MAX;
            let _ = unsafe {
                sys::hipMemPoolSetAttribute(
                    pool,
                    HIP_MEM_POOL_ATTR_RELEASE_THRESHOLD,
                    &mut threshold as *mut u64 as *mut c_void,
                )
            };
        }
        let default_stream = RocmStream::create(ordinal, None)?;
        Ok(Arc::new(RocmContext { ordinal, default_stream }))
    }

    /// The device ordinal this context targets.
    pub fn ordinal(&self) -> usize {
        self.ordinal as usize
    }

    /// Bind this device to the calling thread (`hipSetDevice`). Cheap; called
    /// before driver work, mirroring cudarc's `bind_to_thread`.
    pub fn bind_to_thread(&self) -> Result<()> {
        check(unsafe { sys::hipSetDevice(self.ordinal) }, "hipSetDevice")
    }

    /// The context's default (non-blocking) stream.
    pub fn default_stream(&self) -> Arc<RocmStream> {
        self.default_stream.clone()
    }

    /// Create a fresh non-blocking stream bound to this device.
    pub fn new_stream(&self) -> Result<Arc<RocmStream>> {
        self.bind_to_thread()?;
        RocmStream::create(self.ordinal, None)
    }

    /// Create a non-blocking stream at an explicit scheduling priority. Lower
    /// integer = higher priority (HIP follows the CUDA convention). See
    /// [`stream_priority_range`].
    pub fn new_stream_with_priority(&self, priority: i32) -> Result<Arc<RocmStream>> {
        self.bind_to_thread()?;
        RocmStream::create(self.ordinal, Some(priority))
    }

    /// Block until all work on the device completes (`hipDeviceSynchronize`).
    pub fn synchronize(&self) -> Result<()> {
        self.bind_to_thread()?;
        check(unsafe { sys::hipDeviceSynchronize() }, "hipDeviceSynchronize")
    }
}

/// Query the device's stream-priority integer range as
/// `(least_priority, greatest_priority)`. `greatest <= least` (lower int =
/// higher priority), both `0` on devices without priority support. Mirrors
/// `cuda_stream_priority_range`.
pub fn stream_priority_range() -> Result<(i32, i32)> {
    let mut least: c_int = 0;
    let mut greatest: c_int = 0;
    check(
        unsafe { sys::hipDeviceGetStreamPriorityRange(&mut least, &mut greatest) },
        "hipDeviceGetStreamPriorityRange",
    )?;
    Ok((least, greatest))
}

// ---------------------------------------------------------------------------
// Stream
// ---------------------------------------------------------------------------

/// RAII owner of a HIP stream. The runtime-API analog of
/// `cudarc::driver::CudaStream`; `hip_stream()` matches `cu_stream()` so kernel
/// launch FFI accepts it unchanged.
#[derive(Debug)]
pub struct RocmStream {
    handle: sys::hipStream_t,
    ordinal: c_int,
}

// SAFETY: a hipStream_t is a raw handle bound to one device; it is safe to move
// and share across threads (binding is re-applied per call). Same reasoning as
// cudarc's `unsafe impl Send/Sync for CudaStream`.
unsafe impl Send for RocmStream {}
unsafe impl Sync for RocmStream {}

impl RocmStream {
    /// Create a non-blocking stream bound to `ordinal`, optionally at a given
    /// priority. Internal — callers go through `RocmContext`.
    fn create(ordinal: c_int, priority: Option<i32>) -> Result<Arc<Self>> {
        check(unsafe { sys::hipSetDevice(ordinal) }, "hipSetDevice")?;
        let mut handle: sys::hipStream_t = ptr::null_mut();
        match priority {
            None => check(
                unsafe {
                    sys::hipStreamCreateWithFlags(&mut handle, sys::HIP_STREAM_NON_BLOCKING)
                },
                "hipStreamCreateWithFlags",
            )?,
            Some(p) => check(
                unsafe {
                    sys::hipStreamCreateWithPriority(
                        &mut handle,
                        sys::HIP_STREAM_NON_BLOCKING,
                        p as c_int,
                    )
                },
                "hipStreamCreateWithPriority",
            )?,
        }
        Ok(Arc::new(RocmStream { handle, ordinal }))
    }

    /// The raw `hipStream_t`. Do not destroy it — the `RocmStream` owns it.
    /// Safe to pass to kernel-launch FFI while `self` is alive (matches
    /// `CudaStream::cu_stream()`).
    pub fn hip_stream(&self) -> sys::hipStream_t {
        self.handle
    }

    /// The device ordinal this stream is bound to.
    pub fn ordinal(&self) -> usize {
        self.ordinal as usize
    }

    #[inline]
    fn bind(&self) -> Result<()> {
        check(unsafe { sys::hipSetDevice(self.ordinal) }, "hipSetDevice")
    }

    /// Block until all work queued on this stream completes.
    pub fn synchronize(&self) -> Result<()> {
        self.bind()?;
        check(unsafe { sys::hipStreamSynchronize(self.handle) }, "hipStreamSynchronize")
    }

    /// Allocate `len` bytes on the device, zeroed. Stream-ordered
    /// (`hipMallocAsync`) when supported, falling back to synchronous
    /// `hipMalloc` on arches/runtimes without the stream-ordered allocator.
    pub fn alloc_zeros(self: &Arc<Self>, len: usize) -> Result<RocmSlice> {
        let slice = self.alloc(len)?;
        if len > 0 {
            self.bind()?;
            // SAFETY: slice.ptr is a valid device allocation of `len` bytes.
            check(
                unsafe { sys::hipMemsetD8Async(slice.ptr, 0, len, self.handle) },
                "hipMemsetD8Async",
            )?;
        }
        Ok(slice)
    }

    /// Allocate `len` (uninitialized) bytes on the device. See [`Self::alloc_zeros`].
    pub fn alloc(self: &Arc<Self>, len: usize) -> Result<RocmSlice> {
        self.bind()?;
        // A zero-length allocation is legal and yields a null/!owned slice.
        if len == 0 {
            return Ok(RocmSlice { ptr: ptr::null_mut(), len: 0, async_alloc: false, stream: self.clone() });
        }
        let mut ptr: *mut c_void = ptr::null_mut();
        // Prefer the stream-ordered allocator (needed for HIP-graph capture in
        // R.9); fall back to plain hipMalloc if the runtime/arch rejects it.
        let async_rc = unsafe { sys::hipMallocAsync(&mut ptr, len, self.handle) };
        let async_alloc = if async_rc == sys::HIP_SUCCESS {
            true
        } else {
            ptr = ptr::null_mut();
            check(unsafe { sys::hipMalloc(&mut ptr, len) }, "hipMalloc")?;
            false
        };
        Ok(RocmSlice { ptr, len, async_alloc, stream: self.clone() })
    }

    /// Copy host bytes into a device slice, then synchronize (the host buffer is
    /// only borrowed for the call, so the async copy must complete first).
    pub fn memcpy_htod(&self, dst: &mut RocmSlice, src: &[u8]) -> Result<()> {
        if src.len() != dst.len {
            return Err(HipError {
                code: -1,
                api: "RocmStream::memcpy_htod",
                message: format!("length mismatch: src {} != dst {}", src.len(), dst.len),
            });
        }
        if src.is_empty() {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: dst.ptr is a valid device allocation of dst.len bytes; src is
        // a valid host buffer of the same length. We synchronize before return.
        check(
            unsafe {
                sys::hipMemcpyHtoDAsync(
                    dst.ptr,
                    src.as_ptr() as *mut c_void,
                    src.len(),
                    self.handle,
                )
            },
            "hipMemcpyHtoDAsync",
        )?;
        self.synchronize()
    }

    /// Async H2D copy into a caller-supplied raw device pointer, WITHOUT a
    /// trailing synchronize. The HIP-graph replay path (R.9) uses this to
    /// refresh a graph-stable buffer's contents *in place*: the destination
    /// pointer is the one baked into the captured graph, so it must not change
    /// (no realloc). The copy is queued on this stream and is ordered before
    /// any subsequent launch on the same stream.
    ///
    /// Unlike [`Self::memcpy_htod`], this does NOT synchronize — the caller is
    /// responsible for (a) keeping `src` alive until the copy completes and
    /// (b) synchronizing (this stream, or the launch's stream after an event)
    /// before the host or another stream reads the destination.
    ///
    /// # Safety
    /// `dst` must point to at least `src.len()` bytes of a live device
    /// allocation reachable from this stream's device.
    pub unsafe fn memcpy_htod_raw_async(&self, dst: *mut c_void, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller guarantees `dst` addresses >= src.len() live device
        // bytes; `src` is a valid host slice of the same length.
        check(
            unsafe {
                sys::hipMemcpyHtoDAsync(dst, src.as_ptr() as *mut c_void, src.len(), self.handle)
            },
            "hipMemcpyHtoDAsync",
        )
    }

    /// Zero `len` bytes at a caller-supplied raw device pointer on this stream,
    /// WITHOUT synchronizing. The HIP-graph capture arena (R.9) uses this during
    /// the replay (capture) pass: issued on the active capture stream it is
    /// RECORDED into the graph, so every replay re-zeros the read-before-write
    /// arena buffers. The ROCm analog of cudarc's `result::memset_d8_async`.
    ///
    /// # Safety
    /// `dst` must point to at least `len` bytes of a live device allocation
    /// reachable from this stream's device.
    pub unsafe fn memset_zero_async(&self, dst: *mut c_void, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: caller guarantees `dst` addresses >= len live device bytes.
        check(
            unsafe { sys::hipMemsetD8Async(dst, 0, len, self.handle) },
            "hipMemsetD8Async",
        )
    }

    /// Allocate a device buffer of `src.len()` bytes and copy `src` into it
    /// (H2D), synchronizing before return. The one-shot analog of cudarc's
    /// `CudaStream::clone_htod`.
    pub fn clone_htod(self: &Arc<Self>, src: &[u8]) -> Result<RocmSlice> {
        let mut slice = self.alloc(src.len())?;
        self.memcpy_htod(&mut slice, src)?;
        Ok(slice)
    }

    /// Copy a device slice back to a freshly allocated host `Vec`, synchronizing
    /// before returning.
    pub fn memcpy_dtoh(&self, src: &RocmSlice) -> Result<Vec<u8>> {
        let mut out = vec![0u8; src.len];
        if src.len == 0 {
            return Ok(out);
        }
        self.bind()?;
        // SAFETY: src.ptr is a valid device allocation of src.len bytes; out is
        // a host buffer of the same length. Synchronized before return.
        check(
            unsafe {
                sys::hipMemcpyDtoHAsync(
                    out.as_mut_ptr() as *mut c_void,
                    src.ptr,
                    src.len,
                    self.handle,
                )
            },
            "hipMemcpyDtoHAsync",
        )?;
        self.synchronize()?;
        Ok(out)
    }

    /// Device-to-device copy on this stream (async; caller orders via the
    /// stream). `dst` and `src` must have equal length.
    pub fn memcpy_dtod(&self, dst: &mut RocmSlice, src: &RocmSlice) -> Result<()> {
        if src.len != dst.len {
            return Err(HipError {
                code: -1,
                api: "RocmStream::memcpy_dtod",
                message: format!("length mismatch: src {} != dst {}", src.len, dst.len),
            });
        }
        if src.len == 0 {
            return Ok(());
        }
        self.bind()?;
        // SAFETY: both are valid device allocations of equal length.
        check(
            unsafe { sys::hipMemcpyDtoDAsync(dst.ptr, src.ptr, src.len, self.handle) },
            "hipMemcpyDtoDAsync",
        )
    }
}

impl Drop for RocmStream {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        // Bind the device before the driver call (mirrors cudarc's Drop).
        let _ = unsafe { sys::hipSetDevice(self.ordinal) };
        // SAFETY: handle was created by hipStreamCreate* and not yet destroyed.
        let rc = unsafe { sys::hipStreamDestroy(self.handle) };
        if rc != sys::HIP_SUCCESS {
            eprintln!("RocmStream::drop: hipStreamDestroy failed (hipError {rc})");
        }
        self.handle = ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// Slice (device allocation)
// ---------------------------------------------------------------------------

/// An owned device byte buffer. The analog of `cudarc::driver::CudaSlice<u8>`.
///
/// Holds an `Arc<RocmStream>` so `Drop` can free on the same stream it was
/// allocated on (`hipFreeAsync` for stream-ordered allocations, `hipFree`
/// otherwise). Device pointers are plain `void*` normalized to a single
/// accessor so the Phase R.3 `SliceOwner::Borrowed { ptr, .. }` retype is clean.
#[derive(Debug)]
pub struct RocmSlice {
    ptr: *mut c_void,
    len: usize,
    async_alloc: bool,
    stream: Arc<RocmStream>,
}

// SAFETY: the device pointer is bound to one device and not aliased mutably
// across threads by this type; ownership is unique. Same reasoning as cudarc's
// CudaSlice (which is Send + Sync).
unsafe impl Send for RocmSlice {}
unsafe impl Sync for RocmSlice {}

impl RocmSlice {
    /// The raw device pointer (`hipDeviceptr_t == void*`). Valid while `self` is
    /// alive. The single normalized accessor referenced by the storage layer.
    pub fn device_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// The device pointer as a `usize` — convenient for the `SliceOwner` retype
    /// and for passing to `extern "C"` kernel launchers as a `u64`/pointer.
    pub fn device_ptr_usize(&self) -> usize {
        self.ptr as usize
    }

    /// Physical byte length of the allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the allocation is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The stream this slice was allocated on (and is freed on).
    pub fn stream(&self) -> &Arc<RocmStream> {
        &self.stream
    }
}

impl Drop for RocmSlice {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        let _ = unsafe { sys::hipSetDevice(self.stream.ordinal) };
        // SAFETY: ptr was produced by hipMallocAsync/hipMalloc on self.stream
        // and not yet freed. Free with the matching API.
        let rc = if self.async_alloc {
            unsafe { sys::hipFreeAsync(self.ptr, self.stream.handle) }
        } else {
            unsafe { sys::hipFree(self.ptr) }
        };
        if rc != sys::HIP_SUCCESS {
            eprintln!("RocmSlice::drop: hipFree failed (hipError {rc})");
        }
        self.ptr = ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// Graph capture (wired into kiln-graph-rocm in Phase R.9)
// ---------------------------------------------------------------------------

/// RAII owner of a captured `hipGraph_t`. Analog of `cudarc::driver::CudaGraph`.
#[derive(Debug)]
pub struct RocmGraph {
    graph: sys::hipGraph_t,
}

unsafe impl Send for RocmGraph {}
unsafe impl Sync for RocmGraph {}

/// RAII owner of an instantiated `hipGraphExec_t`. Analog of `CudaGraphExec`.
#[derive(Debug)]
pub struct RocmGraphExec {
    exec: sys::hipGraphExec_t,
}

unsafe impl Send for RocmGraphExec {}
unsafe impl Sync for RocmGraphExec {}

impl RocmStream {
    /// Begin capturing work issued on this stream into a graph
    /// (`hipStreamBeginCapture`, relaxed mode — matches the CUDA path).
    pub fn begin_capture(&self) -> Result<()> {
        self.bind()?;
        check(
            unsafe { sys::hipStreamBeginCapture(self.handle, sys::HIP_STREAM_CAPTURE_MODE_RELAXED) },
            "hipStreamBeginCapture",
        )
    }

    /// End capture and return the resulting graph (`hipStreamEndCapture`).
    pub fn end_capture(&self) -> Result<RocmGraph> {
        self.bind()?;
        let mut graph: sys::hipGraph_t = ptr::null_mut();
        check(
            unsafe { sys::hipStreamEndCapture(self.handle, &mut graph) },
            "hipStreamEndCapture",
        )?;
        Ok(RocmGraph { graph })
    }

    /// Whether a capture is currently active on this stream.
    pub fn is_capturing(&self) -> Result<bool> {
        self.bind()?;
        let mut status: c_uint = 0;
        check(
            unsafe { sys::hipStreamIsCapturing(self.handle, &mut status) },
            "hipStreamIsCapturing",
        )?;
        Ok(status == sys::HIP_STREAM_CAPTURE_STATUS_ACTIVE)
    }
}

impl RocmGraph {
    /// Instantiate into an executable graph with auto-free-on-launch (matches
    /// the CUDA `AUTO_FREE_ON_LAUNCH` flag).
    pub fn instantiate(&self) -> Result<RocmGraphExec> {
        let mut exec: sys::hipGraphExec_t = ptr::null_mut();
        check(
            unsafe {
                sys::hipGraphInstantiateWithFlags(
                    &mut exec,
                    self.graph,
                    sys::HIP_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )
            },
            "hipGraphInstantiateWithFlags",
        )?;
        Ok(RocmGraphExec { exec })
    }
}

impl Drop for RocmGraph {
    fn drop(&mut self) {
        if !self.graph.is_null() {
            let _ = unsafe { sys::hipGraphDestroy(self.graph) };
            self.graph = ptr::null_mut();
        }
    }
}

impl RocmGraphExec {
    /// Launch the executable graph on `stream`.
    pub fn launch(&self, stream: &RocmStream) -> Result<()> {
        stream.bind()?;
        check(
            unsafe { sys::hipGraphLaunch(self.exec, stream.handle) },
            "hipGraphLaunch",
        )
    }
}

impl Drop for RocmGraphExec {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            let _ = unsafe { sys::hipGraphExecDestroy(self.exec) };
            self.exec = ptr::null_mut();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — run only where a real HIP device is present; skip otherwise (mirrors
// the cuda_stream_priority.rs `try_ctx` pattern).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn try_ctx() -> Option<Arc<RocmContext>> {
        if !is_available() {
            eprintln!("ROCm device not available; skipping");
            return None;
        }
        RocmContext::new(0).ok()
    }

    #[test]
    fn runtime_and_count_are_sane() {
        let Some(_ctx) = try_ctx() else { return };
        let v = runtime_version().expect("hipRuntimeGetVersion");
        assert!(v > 0, "expected a positive HIP runtime version, got {v}");
        assert!(device_count().unwrap() >= 1);
    }

    #[test]
    fn priority_range_well_formed() {
        let Some(_ctx) = try_ctx() else { return };
        let (least, greatest) = stream_priority_range().expect("priority range");
        assert!(greatest <= least, "greatest {greatest} should be <= least {least}");
    }

    #[test]
    fn alloc_memset_roundtrip() {
        let Some(ctx) = try_ctx() else { return };
        let stream = ctx.default_stream();
        // zeros
        let z = stream.alloc_zeros(256).expect("alloc_zeros");
        assert_eq!(z.len(), 256);
        assert!(!z.device_ptr().is_null());
        let host = stream.memcpy_dtoh(&z).expect("dtoh");
        assert!(host.iter().all(|&b| b == 0), "alloc_zeros must zero the buffer");
    }

    #[test]
    fn htod_dtoh_roundtrip() {
        let Some(ctx) = try_ctx() else { return };
        let stream = ctx.default_stream();
        let src: Vec<u8> = (0..1024u32).map(|i| (i % 251) as u8).collect();
        let mut dev = stream.alloc(src.len()).expect("alloc");
        stream.memcpy_htod(&mut dev, &src).expect("htod");
        let back = stream.memcpy_dtoh(&dev).expect("dtoh");
        assert_eq!(src, back, "H2D->D2H must round-trip the bytes exactly");
    }

    #[test]
    fn dtod_roundtrip() {
        let Some(ctx) = try_ctx() else { return };
        let stream = ctx.default_stream();
        let src_host: Vec<u8> = (0..512u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut a = stream.alloc(src_host.len()).expect("alloc a");
        stream.memcpy_htod(&mut a, &src_host).expect("htod");
        let mut b = stream.alloc(src_host.len()).expect("alloc b");
        stream.memcpy_dtod(&mut b, &a).expect("dtod");
        stream.synchronize().expect("sync");
        let back = stream.memcpy_dtoh(&b).expect("dtoh");
        assert_eq!(src_host, back, "D2D copy must preserve bytes");
    }

    #[test]
    fn new_stream_with_priority_creates() {
        let Some(ctx) = try_ctx() else { return };
        let (least, greatest) = stream_priority_range().expect("range");
        let hi = ctx.new_stream_with_priority(greatest).expect("high-priority stream");
        let lo = ctx.new_stream_with_priority(least).expect("low-priority stream");
        assert!(!hi.hip_stream().is_null());
        assert!(!lo.hip_stream().is_null());
    }
}

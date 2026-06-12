//! CUDA stream-priority helpers for kiln.
//!
//! Exposes a thin wrapper around `cuStreamCreateWithPriority` so that
//! high-priority compute streams (decode hot path) and low-priority
//! background streams (prefetch, training checkpointing) can be created
//! with an explicit scheduling-priority hint to the CUDA driver.
//!
//! # API surface
//!
//! - [`StreamPriority`] — `High` / `Low` priority variant.
//! - [`PrioritizedCudaStream`] — RAII owner of a raw `CUstream`; exposes
//!   `cu_stream()` for FFI dispatch identical to `CudaStream::cu_stream()`.
//! - [`new_cuda_stream_with_priority`] — public constructor.
//! - [`cuda_stream_priority_range`] — returns `(least_priority,
//!   greatest_priority)` from `cuCtxGetStreamPriorityRange` (lower integer
//!   = higher scheduling priority on NVIDIA hardware).
//!
//! # Why not extend `CudaContext`?
//!
//! `cudarc::driver::CudaStream::cu_stream` is `pub(crate)`, so we cannot
//! construct a `CudaStream` from raw parts from outside the cudarc crate.
//! We therefore own the raw `CUstream` ourselves inside
//! [`PrioritizedCudaStream`], which has the same Drop semantics
//! (`cuStreamDestroy_v2`) and the same `cu_stream()` accessor shape.
//!
//! # cudarc API grounding (cudarc-0.19.4 `src/driver/sys/mod.rs`)
//!
//! ```text
//! // line 13047
//! pub fn cuStreamCreateWithPriority(
//!     phStream: *mut CUstream, flags: c_uint, priority: c_int,
//! ) -> CUresult;
//!
//! // line 13173
//! pub fn cuStreamGetPriority(
//!     hStream: CUstream, priority: *mut c_int,
//! ) -> CUresult;
//!
//! // line 10188
//! pub fn cuCtxGetStreamPriorityRange(
//!     leastPriority: *mut c_int, greatestPriority: *mut c_int,
//! ) -> CUresult;
//! ```
//!
//! All three functions exist in both the static-linking block
//! (`#[cfg(not(feature = "dynamic-loading"))]`) and the dynamic-loading
//! wrapper block (`#[cfg(feature = "dynamic-loading")]`). The workspace pins
//! cudarc with `features = ["dynamic-linking"]`, which in cudarc's own
//! `Cargo.toml` is `dynamic-linking = []` and maps to the `dynamic-loading`
//! code path at build time.
//!
//! `.result()` is an inherent method on `sys::CUresult`
//! (defined in `cudarc::driver::result` as `impl sys::CUresult { ... }`);
//! it converts `CUDA_SUCCESS` to `Ok(())` and anything else to
//! `Err(DriverError)`. Because it is an inherent impl it is in scope
//! wherever `sys::CUresult` is in scope — no extra import needed.

use std::mem::MaybeUninit;
use std::sync::Arc;

use cudarc::driver::sys;
use cudarc::driver::{CudaContext, DriverError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// CUDA stream scheduling priority.
///
/// On NVIDIA hardware, a higher-priority stream receives preferential SM
/// access when competing work is queued. The underlying integer range is
/// device-specific — use [`cuda_stream_priority_range`] to query it.
///
/// Per the CUDA driver API docs:
/// > Stream priorities follow a convention where lower numbers imply
/// > greater priorities. `0` represents default priority. The range of
/// > meaningful stream priorities is `[greatest_priority, least_priority]`,
/// > where `greatest_priority <= least_priority`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPriority {
    /// Maps to `greatestPriority` from `cuCtxGetStreamPriorityRange` —
    /// the most favoured scheduling tier (lowest integer value).
    High,
    /// Maps to `leastPriority` from `cuCtxGetStreamPriorityRange` —
    /// background / low-urgency work (highest integer value).
    Low,
}

/// RAII owner of a priority-scheduled CUDA stream.
///
/// Created by [`new_cuda_stream_with_priority`]. Drops by calling
/// `cuStreamDestroy_v2`, matching cudarc's own `CudaStream` Drop impl.
///
/// The `cu_stream()` accessor has the same signature and semantics as
/// `cudarc::driver::CudaStream::cu_stream()` so all kiln kernel-launch
/// FFI call sites can accept this value without change.
#[derive(Debug)]
pub struct PrioritizedCudaStream {
    cu_stream: sys::CUstream,
    ctx: Arc<CudaContext>,
}

// SAFETY: `CUstream` is a raw pointer-sized handle, but the stream is
// bound to one CUDA context and is safe to move across threads — exactly
// the same reasoning cudarc uses for its own `unsafe impl Send/Sync` on
// `CudaStream`.
unsafe impl Send for PrioritizedCudaStream {}
unsafe impl Sync for PrioritizedCudaStream {}

impl Drop for PrioritizedCudaStream {
    fn drop(&mut self) {
        // Bind the context to this thread before any driver call —
        // mirrors cudarc's own CudaStream::drop().
        if let Err(e) = self.ctx.bind_to_thread() {
            // Cannot propagate from Drop; log and proceed (same as cudarc).
            eprintln!("PrioritizedCudaStream::drop: bind_to_thread failed: {e:?}");
        }
        let cu_stream = std::mem::replace(&mut self.cu_stream, std::ptr::null_mut());
        if !cu_stream.is_null() {
            // SAFETY: cu_stream was created by cuStreamCreateWithPriority and
            // has not been destroyed yet (this is the sole Drop path).
            let res = unsafe { sys::cuStreamDestroy_v2(cu_stream) };
            if res != sys::CUresult::CUDA_SUCCESS {
                eprintln!(
                    "PrioritizedCudaStream::drop: cuStreamDestroy_v2 failed: {:?}",
                    res
                );
            }
        }
    }
}

impl PrioritizedCudaStream {
    /// The raw `CUstream` handle.
    ///
    /// # Safety
    ///
    /// Do not destroy this handle; [`PrioritizedCudaStream`] owns it and
    /// will call `cuStreamDestroy_v2` when dropped. Safe to pass to kernel
    /// launch FFI while `self` is alive, matching the contract of
    /// `CudaStream::cu_stream()`.
    pub fn cu_stream(&self) -> sys::CUstream {
        self.cu_stream
    }

    /// The `CudaContext` this stream was created on.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }
}

// ---------------------------------------------------------------------------
// Public constructors / queries
// ---------------------------------------------------------------------------

/// Query the device's stream-priority integer range.
///
/// Returns `(least_priority, greatest_priority)`. On NVIDIA hardware:
/// - `greatest_priority <= least_priority` (lower integer = higher priority).
/// - Both are `0` on devices that do not support stream priorities.
///
/// # Errors
///
/// Returns `Err(DriverError)` if `cuCtxGetStreamPriorityRange` fails
/// (e.g., no CUDA device present or driver not initialised).
pub fn cuda_stream_priority_range(ctx: &Arc<CudaContext>) -> Result<(i32, i32), DriverError> {
    ctx.bind_to_thread()?;
    let mut least = MaybeUninit::<i32>::uninit();
    let mut greatest = MaybeUninit::<i32>::uninit();
    // SAFETY: both output pointers are valid MaybeUninit<i32> allocations.
    unsafe {
        sys::cuCtxGetStreamPriorityRange(least.as_mut_ptr(), greatest.as_mut_ptr()).result()?;
        Ok((least.assume_init(), greatest.assume_init()))
    }
}

/// Create a `CU_STREAM_NON_BLOCKING` CUDA stream at the given priority.
///
/// - `StreamPriority::High` maps to `greatestPriority` (most favoured).
/// - `StreamPriority::Low` maps to `leastPriority` (least favoured).
///
/// On devices that do not support stream priorities (range endpoints are
/// both `0`), the function still succeeds — the driver clamps the
/// requested priority to `0` and accepts the call.
///
/// # Errors
///
/// Returns `Err(DriverError)` if `cuCtxGetStreamPriorityRange` or
/// `cuStreamCreateWithPriority` fails.
pub fn new_cuda_stream_with_priority(
    ctx: &Arc<CudaContext>,
    priority: StreamPriority,
) -> Result<PrioritizedCudaStream, DriverError> {
    ctx.bind_to_thread()?;

    // Query the supported priority range so we pass a value the driver
    // accepts without clamping to a possibly unexpected integer.
    let (least, greatest) = cuda_stream_priority_range(ctx)?;
    let prio_int: i32 = match priority {
        StreamPriority::High => greatest,
        StreamPriority::Low => least,
    };

    let mut cu_stream = MaybeUninit::<sys::CUstream>::uninit();
    // SAFETY: cu_stream is a valid MaybeUninit output pointer.
    // CU_STREAM_NON_BLOCKING (= 1) matches what CudaContext::new_stream()
    // uses via result::stream::StreamKind::NonBlocking.
    unsafe {
        sys::cuStreamCreateWithPriority(
            cu_stream.as_mut_ptr(),
            sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            prio_int,
        )
        .result()?;
        Ok(PrioritizedCudaStream {
            cu_stream: cu_stream.assume_init(),
            ctx: ctx.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_storage::primary_cuda_context;

    /// Try to obtain a CUDA context for device 0; return `None` and skip
    /// if unavailable — matches the pattern in
    /// `kiln-blas/tests/cublaslt_handle_smoke.rs`.
    fn try_ctx() -> Option<Arc<CudaContext>> {
        primary_cuda_context(0).ok()
    }

    #[test]
    fn priority_range_is_well_formed() {
        let Some(ctx) = try_ctx() else {
            eprintln!("CUDA device not available; skipping priority_range_is_well_formed");
            return;
        };
        let (least, greatest) =
            cuda_stream_priority_range(&ctx).expect("cuCtxGetStreamPriorityRange");
        // NVIDIA convention: greatest_priority <= least_priority.
        // On devices without priority support both are 0 — still valid.
        assert!(
            greatest <= least,
            "expected greatest_priority <= least_priority, \
             got greatest={greatest} least={least}"
        );
    }

    #[test]
    fn hi_and_lo_streams_create_and_report_ordered_priorities() {
        let Some(ctx) = try_ctx() else {
            eprintln!(
                "CUDA device not available; skipping \
                 hi_and_lo_streams_create_and_report_ordered_priorities"
            );
            return;
        };

        let hi = new_cuda_stream_with_priority(&ctx, StreamPriority::High)
            .expect("create high-priority stream");
        let lo = new_cuda_stream_with_priority(&ctx, StreamPriority::Low)
            .expect("create low-priority stream");

        // Read back the priorities the driver assigned via cuStreamGetPriority.
        let mut hi_prio = MaybeUninit::<i32>::uninit();
        let mut lo_prio = MaybeUninit::<i32>::uninit();
        // SAFETY: both streams are alive; output pointers are valid.
        unsafe {
            sys::cuStreamGetPriority(hi.cu_stream(), hi_prio.as_mut_ptr())
                .result()
                .expect("cuStreamGetPriority (high)");
            sys::cuStreamGetPriority(lo.cu_stream(), lo_prio.as_mut_ptr())
                .result()
                .expect("cuStreamGetPriority (low)");
        }
        let hi_prio = unsafe { hi_prio.assume_init() };
        let lo_prio = unsafe { lo_prio.assume_init() };

        // The driver accepted both streams (no error above — that alone
        // proves cuStreamCreateWithPriority succeeded for each priority).
        //
        // On priority-capable hardware: hi_prio < lo_prio (lower int =
        // higher priority). On hardware without priority support (range
        // [0, 0]) both clamp to 0 and are equal — that is also valid.
        assert!(
            hi_prio <= lo_prio,
            "expected hi_prio <= lo_prio (lower int = higher priority), \
             got hi={hi_prio} lo={lo_prio}"
        );
    }
}

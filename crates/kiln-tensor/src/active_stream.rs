//! Thread-local "active CUDA stream" override (#1082 CUDA-graph fix).
//!
//! # Why this exists
//!
//! Production decode opens the CUDA device through candle's
//! `new_cuda_with_stream`, which creates a fresh **non-default**
//! `CUstream`. `CudaGraphRunner::try_capture` then calls
//! `begin_capture` on *that* stream. But every kt CUDA op resolves its
//! stream as `ctx.default_stream()` — the **legacy NULL default
//! stream** (cudarc 0.19.x). Issuing a kernel launch / alloc / memcpy
//! on the NULL stream while a *different* stream is mid-capture is a
//! `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` violation: the kt flip
//! discarded the per-op stream candle used to thread through every op
//! and hardcoded `ctx.default_stream()` everywhere.
//!
//! Candle solved this by passing its chosen stream through every op.
//! Re-threading a `&CudaStream` argument through all ~70 kt op sites
//! would be a massive, invasive change. Instead this module provides a
//! **thread-local override**: every kt stream-resolution site calls
//! [`active_cuda_stream`] (which returns `ctx.default_stream()` when no
//! override is installed — i.e. ZERO behavior change for all
//! non-capture call paths) and the capture window installs the capture
//! stream via [`with_active_cuda_stream`] for the duration of the
//! captured forward pass.
//!
//! # Safety property (the key one)
//!
//! Outside a `with_active_cuda_stream` scope, `active_cuda_stream`
//! returns exactly `ctx.default_stream()`. So every one of the ~70
//! migrated op sites is a **no-op** for all existing (non-capture)
//! callers — training, eager decode, parity tests, etc. all keep
//! running on the legacy default stream exactly as before. Only the
//! CUDA-graph capture path engages the override.
//!
//! The thread-local-scope pattern mirrors
//! `kiln_autograd::tape_scope::with_thread_local_tape`: set on entry,
//! restore the prior value on exit (including on panic via an RAII
//! guard).

use std::cell::RefCell;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

thread_local! {
    /// The active CUDA stream override for this thread.
    ///
    /// `None` outside a [`with_active_cuda_stream`] scope — in which
    /// case [`active_cuda_stream`] falls back to
    /// `ctx.default_stream()`, preserving the pre-#1082-fix behavior
    /// exactly. `Some(stream)` only while a capture (or other
    /// stream-scoped) window is open on this thread.
    static ACTIVE_CUDA_STREAM: RefCell<Option<Arc<CudaStream>>> =
        const { RefCell::new(None) };
}

/// RAII guard that restores the previous `ACTIVE_CUDA_STREAM` value
/// when dropped — including on panic / early return. Storing the prior
/// `Option<Arc<CudaStream>>` (rather than assuming it was `None`) keeps
/// the override correct even if scopes ever nest.
struct ActiveStreamGuard {
    prev: Option<Arc<CudaStream>>,
}

impl Drop for ActiveStreamGuard {
    fn drop(&mut self) {
        ACTIVE_CUDA_STREAM.with(|cell| {
            *cell.borrow_mut() = self.prev.take();
        });
    }
}

/// Run `f` with `stream` installed as the active CUDA stream override
/// on the current thread, restoring the previous override afterward
/// (even if `f` panics).
///
/// Inside `f`, every kt CUDA op site that calls [`active_cuda_stream`]
/// will resolve to `stream` instead of `ctx.default_stream()`. This is
/// how `CudaGraphRunner::try_capture` makes all kt kernel launches /
/// allocs / memcpys land on the capture stream so they are recorded
/// into the graph instead of triggering
/// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
pub fn with_active_cuda_stream<R>(stream: Arc<CudaStream>, f: impl FnOnce() -> R) -> R {
    // Install the override, capturing the prior value so the guard can
    // restore it on the way out (RAII => panic-safe).
    let prev = ACTIVE_CUDA_STREAM.with(|cell| cell.borrow_mut().replace(stream));
    let _guard = ActiveStreamGuard { prev };
    f()
}

/// Resolve the CUDA stream a kt op should run on.
///
/// Returns the thread-local override installed by
/// [`with_active_cuda_stream`] if one is active; otherwise returns
/// `ctx.default_stream()` — i.e. **identical behavior to the pre-fix
/// code** for every non-capture call path. This is the chokepoint every
/// kt stream-resolution site routes through.
pub fn active_cuda_stream(ctx: &Arc<CudaContext>) -> Arc<CudaStream> {
    ACTIVE_CUDA_STREAM
        .with(|cell| cell.borrow().clone())
        .unwrap_or_else(|| ctx.default_stream())
}

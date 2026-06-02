//! Thread-local "active ROCm stream" override — the ROCm analog of
//! [`crate::active_stream`] (Phase R.3).
//!
//! Mirrors `active_cuda_stream` exactly: every kt ROCm op resolves its stream
//! through [`active_rocm_stream`], which returns `ctx.default_stream()` unless a
//! HIP-graph capture window (Phase R.9) installs an override via
//! [`with_active_rocm_stream`]. Outside a capture scope this is a no-op — the
//! op runs on the context's default stream, identical to pre-capture behavior.

use std::cell::RefCell;
use std::sync::Arc;

use kiln_hip::{RocmContext, RocmStream};

thread_local! {
    /// The active ROCm stream override for this thread. `None` outside a
    /// [`with_active_rocm_stream`] scope (→ `ctx.default_stream()`).
    static ACTIVE_ROCM_STREAM: RefCell<Option<Arc<RocmStream>>> = const { RefCell::new(None) };
}

/// RAII guard restoring the previous `ACTIVE_ROCM_STREAM` value on drop
/// (including on panic / early return), so nested scopes stay correct.
struct ActiveStreamGuard {
    prev: Option<Arc<RocmStream>>,
}

impl Drop for ActiveStreamGuard {
    fn drop(&mut self) {
        ACTIVE_ROCM_STREAM.with(|cell| {
            *cell.borrow_mut() = self.prev.take();
        });
    }
}

/// Run `f` with `stream` installed as the active ROCm stream override on the
/// current thread, restoring the previous override afterward (even on panic).
/// Used by the HIP-graph capture path (Phase R.9) so kt kernel launches /
/// allocs / memcpys land on the capture stream.
pub fn with_active_rocm_stream<R>(stream: Arc<RocmStream>, f: impl FnOnce() -> R) -> R {
    let prev = ACTIVE_ROCM_STREAM.with(|cell| cell.borrow_mut().replace(stream));
    let _guard = ActiveStreamGuard { prev };
    f()
}

/// Resolve the ROCm stream a kt op should run on: the thread-local override if
/// active, else `ctx.default_stream()` (identical behavior to pre-capture code
/// for every non-capture call path). The chokepoint every kt ROCm
/// stream-resolution site routes through.
pub fn active_rocm_stream(ctx: &Arc<RocmContext>) -> Arc<RocmStream> {
    ACTIVE_ROCM_STREAM
        .with(|cell| cell.borrow().clone())
        .unwrap_or_else(|| ctx.default_stream())
}

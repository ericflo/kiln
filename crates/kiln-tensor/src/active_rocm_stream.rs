//! Thread-local "active ROCm stream" override — the ROCm analog of
//! [`crate::active_stream`] (Phase R.3).
//!
//! Mirrors `active_cuda_stream` exactly: every kt ROCm op resolves its stream
//! through [`active_rocm_stream`], which returns `ctx.default_stream()` unless a
//! HIP-graph capture window (Phase R.9) installs an override via
//! [`with_rocm_graph_capture_stream`]. Outside a capture scope this is a no-op — the
//! op runs on the context's default stream, identical to pre-capture behavior.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use kiln_hip::{RocmContext, RocmStream};

thread_local! {
    /// The active ROCm stream override for this thread. `None` outside a
    /// [`with_rocm_graph_capture_stream`] scope (→ `ctx.default_stream()`).
    static ACTIVE_ROCM_STREAMS: RefCell<HashMap<usize, Arc<RocmStream>>> = RefCell::new(HashMap::new());
    /// Most recent stream returned to a kt ROCm operation on this thread. The
    /// strong reference is retained until a successful external-yield handoff;
    /// otherwise a temporary private stream could be destroyed with live work.
    static LAST_ROCM_PRODUCER_STREAMS: RefCell<HashMap<usize, Arc<RocmStream>>> = RefCell::new(HashMap::new());
}

/// RAII guard restoring the previous `ACTIVE_ROCM_STREAM` value on drop
/// (including on panic / early return), so nested scopes stay correct.
struct ActiveStreamGuard {
    device_ordinal: usize,
    prev: Option<Arc<RocmStream>>,
}

impl Drop for ActiveStreamGuard {
    fn drop(&mut self) {
        ACTIVE_ROCM_STREAMS.with(|cell| {
            let mut streams = cell.borrow_mut();
            match self.prev.take() {
                Some(previous) => {
                    streams.insert(self.device_ordinal, previous);
                }
                None => {
                    streams.remove(&self.device_ordinal);
                }
            }
        });
    }
}

/// Run `f` with the HIP graph's private stream installed on this thread,
/// restoring the previous override afterward (even on panic).
///
/// # Safety
/// The caller must establish input and output handoffs between the default and
/// private streams. Kiln's graph runner does so with pre-capture drains and
/// replay events. An active capture arena is also required so a private-stream
/// tensor cannot escape into ordinary safe eager execution.
pub unsafe fn with_rocm_graph_capture_stream<R>(
    stream: Arc<RocmStream>,
    f: impl FnOnce() -> R,
) -> R {
    assert!(
        crate::rocm_capture_arena_active(),
        "with_rocm_graph_capture_stream requires an active ROCm capture arena"
    );
    let device_ordinal = stream.ordinal();
    let prev = ACTIVE_ROCM_STREAMS.with(|cell| cell.borrow_mut().insert(device_ordinal, stream));
    let _guard = ActiveStreamGuard {
        device_ordinal,
        prev,
    };
    f()
}

/// Resolve the ROCm stream a kt op should run on: the thread-local override if
/// active, else `ctx.default_stream()` (identical behavior to pre-capture code
/// for every non-capture call path). The chokepoint every kt ROCm
/// stream-resolution site routes through.
pub fn active_rocm_stream(ctx: &Arc<RocmContext>) -> Arc<RocmStream> {
    let device_ordinal = ctx.ordinal();
    let stream = ACTIVE_ROCM_STREAMS
        .with(|cell| cell.borrow().get(&device_ordinal).cloned())
        .unwrap_or_else(|| ctx.default_stream());
    LAST_ROCM_PRODUCER_STREAMS.with(|cell| {
        cell.borrow_mut().insert(device_ordinal, stream.clone());
    });
    stream
}

/// Last producer stream resolved on this thread for `ctx`'s device.
pub(crate) fn last_rocm_producer_stream(ctx: &Arc<RocmContext>) -> Option<Arc<RocmStream>> {
    LAST_ROCM_PRODUCER_STREAMS.with(|cell| cell.borrow().get(&ctx.ordinal()).cloned())
}

/// Forget `stream` only after its external-yield boundary completed. A later
/// producer installed on this thread is left intact.
pub(crate) fn clear_last_rocm_producer_stream(ctx: &Arc<RocmContext>, stream: &Arc<RocmStream>) {
    LAST_ROCM_PRODUCER_STREAMS.with(|cell| {
        let mut streams = cell.borrow_mut();
        if streams
            .get(&ctx.ordinal())
            .is_some_and(|current| Arc::ptr_eq(current, stream))
        {
            streams.remove(&ctx.ordinal());
        }
    });
}

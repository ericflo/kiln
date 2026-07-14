//! Thread-local **capture arena** for the ROCm HIP-graph freeze-pointers fix
//! (R.9, Phase 2). The 1:1 ROCm twin of [`crate::capture_alloc`]; distinct
//! symbol names (`RocmCaptureArena`, `with_rocm_capture_arena`,
//! `rocm_capture_arena_alloc`, …) and a separate thread-local so a build with
//! BOTH `cuda` and `rocm` features enabled has no collision.
//!
//! # The bug this fixes
//!
//! Decode HIP-graph capture engages on a non-default stream
//! ([`crate::with_rocm_graph_capture_stream`]). The forward's Q/K/V projections and
//! per-layer activations are allocated by [`crate::RocmStorage::zeros_ctx`] /
//! [`crate::RocmStorage::alloc_uninit_ctx`] straight through the stream-ordered
//! allocator (`hipMallocAsync`) with no allocator indirection, so they are
//! `hipFreeAsync`'d when the capture closure returns and reused by the time the
//! graph replays → the recorded kernels dereference dangling device pointers
//! (`hipErrorIllegalAddress`). The capture window pins only the *output*
//! buffers, never the *input* activations.
//!
//! # The fix (mirrors [`crate::active_rocm_stream`])
//!
//! A thread-local capture arena installed for the duration of the captured
//! forward. While active, `zeros_ctx` / `alloc_uninit_ctx` route through
//! [`rocm_capture_arena_alloc`] instead of allocating fresh. Two passes:
//!
//! 1. **Record** (before `begin_capture`): each alloc makes a *real owned*
//!    buffer (`hipMallocAsync` is legal here — outside capture), retains it in
//!    the arena, and hands the forward a `Borrowed` [`RocmStorage`] *view* into
//!    it (keep-alive = the arena's `Arc`). The forward drops its view normally;
//!    the owned buffer persists in the arena.
//! 2. **Replay** (inside `begin_capture`): each alloc hands out a Borrowed view
//!    of the *same* pre-allocated buffer, in the identical order (decode is
//!    deterministic, so the alloc sequence matches pass 1). **No `hipMalloc`**
//!    happens during capture → the captured graph has no alloc/free nodes and
//!    every recorded device pointer is stable. `zeros_ctx` buffers get a
//!    *captured* `hipMemsetD8Async(0)` on the capture stream so each replay
//!    re-zeros them (read-before-write correctness).
//!
//! After capture, [`RocmCaptureArena::take_retained`] hands the owned buffers to
//! the `CapturedDecodeGraphRocm` so they outlive every replay.
//!
//! # Why Borrowed views are safe here
//!
//! Decode kernels write through raw `device_ptr_raw()` FFI, not Rust
//! `slice_mut()`, so a Borrowed (non-owning) view is fully writable through the
//! kernel path. `Storage` is `Arc<dyn StorageBackend>`, so the arena sharing the
//! underlying allocation across passes is sound.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use kiln_hip::RocmContext;

use crate::{DType, Error, Result, RocmStorage};

/// One pre-allocated, arena-retained buffer. The forward only ever receives a
/// `Borrowed` [`RocmStorage`] view of `storage`; the `Arc` here (and the
/// keep-alive clone inside each view) keeps the real allocation mapped for the
/// lifetime of the captured graph.
#[derive(Debug)]
struct ArenaBuf {
    dtype: DType,
    n_elements: usize,
    storage: Arc<RocmStorage>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ArenaMode {
    /// Pass 1: allocate real owned buffers (pre-capture), retain them.
    Record,
    /// Pass 2: hand out Borrowed views of the recorded buffers, no new alloc.
    Replay,
}

/// Owns the freeze-pointer buffers for one captured decode graph and hands the
/// forward Borrowed views into them. See the module docs.
#[derive(Debug)]
pub struct RocmCaptureArena {
    ctx: Arc<RocmContext>,
    device_index: usize,
    bufs: Vec<ArenaBuf>,
    mode: ArenaMode,
    cursor: usize,
}

impl RocmCaptureArena {
    /// Construct in `Record` mode bound to `ctx` / `device_index`.
    pub fn new_record(ctx: Arc<RocmContext>, device_index: usize) -> Self {
        Self {
            ctx,
            device_index,
            bufs: Vec::new(),
            mode: ArenaMode::Record,
            cursor: 0,
        }
    }

    /// Flip to `Replay` mode and rewind the cursor. Call between the warm pass
    /// and the captured pass.
    pub fn begin_replay(&mut self) {
        self.mode = ArenaMode::Replay;
        self.cursor = 0;
    }

    /// Number of distinct buffers recorded during the warm pass.
    pub fn buffer_count(&self) -> usize {
        self.bufs.len()
    }

    /// Borrow the owned buffers recorded so far without changing arena mode or
    /// cursor state. Capture admission uses this after the warm pass to reserve
    /// the exact requested physical bytes before native stream capture begins.
    pub fn retained_buffers(&self) -> impl ExactSizeIterator<Item = &Arc<RocmStorage>> {
        self.bufs.iter().map(|buf| &buf.storage)
    }

    /// Allocate (`Record`) or hand out (`Replay`) a buffer, returning a
    /// Borrowed [`RocmStorage`] view into an arena-owned allocation.
    fn alloc(&mut self, dtype: DType, n_elements: usize, zero: bool) -> Result<RocmStorage> {
        match self.mode {
            ArenaMode::Record => {
                let owned = if zero {
                    RocmStorage::zeros_ctx(&self.ctx, self.device_index, dtype, n_elements)?
                } else {
                    RocmStorage::alloc_uninit_ctx(&self.ctx, self.device_index, dtype, n_elements)?
                };
                let storage = Arc::new(owned);
                let view = self.borrow_view(dtype, &storage)?;
                self.bufs.push(ArenaBuf {
                    dtype,
                    n_elements,
                    storage,
                });
                Ok(view)
            }
            ArenaMode::Replay => {
                let buf = self.bufs.get(self.cursor).ok_or_else(|| {
                    Error::Msg(format!(
                        "RocmCaptureArena replay overran: cursor {} >= {} recorded buffers \
                         (forward non-deterministic between warm and capture passes)",
                        self.cursor,
                        self.bufs.len()
                    ))
                })?;
                if buf.dtype != dtype || buf.n_elements != n_elements {
                    return Err(Error::Msg(format!(
                        "RocmCaptureArena replay shape mismatch at alloc #{}: recorded \
                         ({:?}, {}) vs requested ({:?}, {})",
                        self.cursor, buf.dtype, buf.n_elements, dtype, n_elements
                    )));
                }
                let storage = buf.storage.clone();
                let view = self.borrow_view(dtype, &storage)?;
                // KILN_ARENA_FORCE_ZERO=1 forces a captured per-replay memset on
                // EVERY arena buffer (not just `zero=true` ones) — the same
                // box-102 BUG2 diagnostic as the CUDA path: if it un-freezes
                // late layers on replay, an uninitialized (`zero=false`) arena
                // buffer is being read stale by a kernel that doesn't fully
                // overwrite it. Off by default; zero production cost.
                let force_zero =
                    std::env::var("KILN_ARENA_FORCE_ZERO").ok().as_deref() == Some("1");
                if zero || force_zero {
                    // Captured memset on the active (capture) stream — recorded
                    // into the graph so every replay re-zeros the buffer.
                    self.memset_zero(&storage)?;
                }
                self.cursor += 1;
                Ok(view)
            }
        }
    }

    /// Build a Borrowed `RocmStorage` view into `storage`, keeping the owned
    /// allocation alive via the keep-alive `Arc`.
    fn borrow_view(&self, dtype: DType, storage: &Arc<RocmStorage>) -> Result<RocmStorage> {
        let (ptr, byte_len) = storage.device_ptr_raw();
        let keep_alive: Arc<dyn Any + Send + Sync> = storage.clone();
        // SAFETY: the arena owns the allocation and routes initialization plus
        // every consumer through the same explicitly ordered graph stream.
        unsafe {
            RocmStorage::from_borrowed_ctx(
                &self.ctx,
                self.device_index,
                dtype,
                ptr,
                byte_len,
                keep_alive,
            )
        }
    }

    /// Zero the buffer on the active ROCm stream (captured during the replay
    /// pass via `hipMemsetD8Async`).
    fn memset_zero(&self, storage: &Arc<RocmStorage>) -> Result<()> {
        let (ptr, byte_len) = storage.device_ptr_raw();
        let stream = crate::active_rocm_stream(&self.ctx);
        // SAFETY: `ptr` points at `byte_len` valid device bytes owned by the
        // arena (alive via `storage`); the active stream is the capture stream
        // the buffer is (re)used on. Stream-ordered zero-fill, recorded into the
        // graph during the replay (capture) pass.
        unsafe {
            stream
                .memset_zero_async(ptr as *mut core::ffi::c_void, byte_len)
                .map_err(|e| {
                    Error::Msg(format!(
                        "RocmCaptureArena::memset_zero: memset_zero_async({byte_len}) failed: {e:?}"
                    ))
                })?;
        }
        Ok(())
    }

    /// Consume the arena, returning the retained owned buffers so the caller
    /// (`CapturedDecodeGraphRocm`) can keep them mapped for every replay.
    pub fn into_retained(self) -> Vec<Arc<RocmStorage>> {
        self.bufs.into_iter().map(|b| b.storage).collect()
    }

    /// Drain the retained owned buffers out of the arena (leaving it empty),
    /// returning them so the caller can keep them mapped for every replay.
    /// Use when the arena is behind an `Rc<RefCell<…>>` and can't be consumed
    /// by value.
    pub fn take_retained(&mut self) -> Vec<Arc<RocmStorage>> {
        std::mem::take(&mut self.bufs)
            .into_iter()
            .map(|b| b.storage)
            .collect()
    }
}

thread_local! {
    /// The capture arena active on this thread, or `None` outside a
    /// [`with_rocm_capture_arena`] scope (in which case `zeros_ctx` /
    /// `alloc_uninit_ctx` take their normal direct-allocation path — ZERO
    /// behavior change for every non-capture caller).
    static ROCM_CAPTURE_ARENA: RefCell<Option<Rc<RefCell<RocmCaptureArena>>>> =
        const { RefCell::new(None) };
}

/// RAII guard restoring the previous arena on scope exit (panic-safe).
struct RocmArenaGuard {
    prev: Option<Rc<RefCell<RocmCaptureArena>>>,
}

impl Drop for RocmArenaGuard {
    fn drop(&mut self) {
        ROCM_CAPTURE_ARENA.with(|cell| {
            *cell.borrow_mut() = self.prev.take();
        });
    }
}

/// Run `f` with `arena` installed as the active capture arena on this thread,
/// restoring the previous value afterward (even on panic). Inside `f`, every
/// `zeros_ctx` / `alloc_uninit_ctx` routes through the arena.
pub fn with_rocm_capture_arena<R>(
    arena: Rc<RefCell<RocmCaptureArena>>,
    f: impl FnOnce() -> R,
) -> R {
    let prev = ROCM_CAPTURE_ARENA.with(|cell| cell.borrow_mut().replace(arena));
    let _guard = RocmArenaGuard { prev };
    f()
}

/// Allocation hook called from `RocmStorage::zeros_ctx` (`zero = true`) and
/// `alloc_uninit_ctx` (`zero = false`).
///
/// Returns `Some(Ok(view))` — a Borrowed RocmStorage view into the arena — when
/// a capture arena is active on this thread; `None` otherwise (the caller then
/// takes its normal direct `hipMallocAsync` path).
pub fn rocm_capture_arena_alloc(
    dtype: DType,
    n_elements: usize,
    zero: bool,
) -> Option<Result<RocmStorage>> {
    // Take the arena OUT of the thread-local for the duration of the alloc.
    // The `Record`-mode path calls `RocmStorage::zeros_ctx` /
    // `alloc_uninit_ctx` to make the real owned buffer — those re-enter this
    // hook, and with the arena removed they correctly take the direct
    // `hipMallocAsync` path instead of recursing (or double-borrowing the
    // `RefCell`). Restored immediately after.
    let arena = ROCM_CAPTURE_ARENA.with(|cell| cell.borrow_mut().take());
    let arena = arena?;
    let result = arena.borrow_mut().alloc(dtype, n_elements, zero);
    ROCM_CAPTURE_ARENA.with(|cell| {
        *cell.borrow_mut() = Some(arena);
    });
    Some(result)
}

/// Whether a capture arena is active on this thread.
pub fn rocm_capture_arena_active() -> bool {
    ROCM_CAPTURE_ARENA.with(|cell| cell.borrow().is_some())
}

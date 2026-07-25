//! Thread-local **capture arena** for the #1082 CUDA-graph freeze-pointers
//! fix (epic boxes 98–101: "allocator freeze-pointers mode" + "capture-lifetime
//! / dangling-pointer rule").
//!
//! # The bug this fixes
//!
//! Decode CUDA-graph capture engages on a non-default stream
//! ([`crate::with_active_cuda_stream`]), but **replay faulted with
//! `CUDA_ERROR_ILLEGAL_ADDRESS`** — compute-sanitizer pinned it to
//! `kiln_flash::flash_fwd_splitkv_kernel` reading a *garbage device pointer*
//! out of `Flash_fwd_params`. Root cause: the forward's Q/K/V projections and
//! per-layer activations are allocated by [`crate::CudaStorage::zeros_ctx`] /
//! [`crate::CudaStorage::alloc_uninit_ctx`] straight through cudarc
//! (`alloc_zeros` → `cudaMallocAsync`) with **zero allocator indirection**, so
//! they are freed when the capture closure returns and reused by the time the
//! graph replays. The capture window pins only the *output* buffers
//! (`paged_decode_outputs`, `output_logits`, …), never the *input* activations.
//!
//! # The fix (mirrors [`crate::active_stream`])
//!
//! A thread-local capture arena installed for the duration of the captured
//! forward. While active, `zeros_ctx` / `alloc_uninit_ctx` and host-to-device
//! tensor construction route through [`capture_arena_alloc`] /
//! [`capture_arena_from_host`] instead of allocating fresh. The arena runs in
//! two passes:
//!
//! 1. **Record** (before `begin_capture`): each alloc makes a *real owned*
//!    buffer (`cudaMallocAsync` is legal here — outside capture), retains it in
//!    the arena, and hands the forward a [`SliceOwner::Borrowed`] *view* into it
//!    (keep-alive = the arena's `Arc`). The forward drops its view normally; the
//!    owned buffer persists in the arena. Host-initialized tensors are uploaded
//!    and synchronized in this pass, while their source bytes are still alive.
//! 2. **Replay** (inside `begin_capture`): each alloc hands out a Borrowed view
//!    of the *same* pre-allocated buffer, in the identical order (decode is
//!    deterministic, so the alloc sequence matches pass 1). **No `cudaMalloc`
//!    or pageable-host memcpy happens in the arena path during capture** → the
//!    arena contributes no alloc/free nodes or dangling temporary host pointer,
//!    and every recorded device pointer is stable. `zeros_ctx` buffers get a
//!    *captured* `cuMemsetD8Async(0)` on the capture stream so each replay
//!    re-zeros them (read-before-write correctness). Host-initialized tensors
//!    must reproduce byte-for-byte identical contents in both passes or capture
//!    fails closed.
//!
//! After capture, [`CaptureArena::into_retained`] hands the owned buffers to the
//! `CapturedDecodeGraph` so they outlive every replay.
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

use cudarc::driver::CudaContext;

use crate::{CudaStorage, DType, Error, Result};

/// One pre-allocated, arena-retained buffer. The forward only ever receives a
/// `Borrowed` [`CudaStorage`] view of `storage`; the `Arc` here (and the
/// keep-alive clone inside each view) keeps the real allocation mapped for the
/// lifetime of the captured graph.
#[derive(Debug)]
struct ArenaBuf {
    dtype: DType,
    n_elements: usize,
    init: ArenaInit,
    storage: Arc<CudaStorage>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
enum ArenaInit {
    Zero,
    Uninit,
    Host(Vec<u8>),
}

impl ArenaInit {
    fn name(&self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Uninit => "uninit",
            Self::Host(_) => "host",
        }
    }
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
pub struct CaptureArena {
    ctx: Arc<CudaContext>,
    device_index: usize,
    bufs: Vec<ArenaBuf>,
    mode: ArenaMode,
    cursor: usize,
}

impl CaptureArena {
    /// Construct in `Record` mode bound to `ctx` / `device_index`.
    pub fn new_record(ctx: Arc<CudaContext>, device_index: usize) -> Self {
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

    /// Verify that the replay/capture pass consumed the complete allocation
    /// sequence recorded by the warm pass.
    ///
    /// An underrun is as unsafe as an overrun: it means some data-dependent or
    /// lazily initialized path changed the allocation sequence between passes,
    /// so later same-shaped allocations may have been bound to the wrong stable
    /// device pointers without tripping the per-allocation shape check.
    pub fn ensure_replay_complete(&self) -> Result<()> {
        if self.mode != ArenaMode::Replay {
            return Err(Error::Msg(
                "CaptureArena completion check requires Replay mode".to_string(),
            ));
        }
        if self.cursor != self.bufs.len() {
            return Err(Error::Msg(format!(
                "CaptureArena replay underrun: consumed {} of {} recorded buffers \
                 (forward allocation sequence changed between warm and capture passes)",
                self.cursor,
                self.bufs.len()
            )));
        }
        Ok(())
    }

    /// Allocate (`Record`) or hand out (`Replay`) a buffer, returning a
    /// Borrowed [`CudaStorage`] view into an arena-owned allocation.
    fn alloc(&mut self, dtype: DType, n_elements: usize, init: ArenaInit) -> Result<CudaStorage> {
        match self.mode {
            ArenaMode::Record => {
                let owned = match &init {
                    ArenaInit::Zero => {
                        CudaStorage::zeros_ctx(&self.ctx, self.device_index, dtype, n_elements)?
                    }
                    ArenaInit::Uninit => CudaStorage::alloc_uninit_ctx(
                        &self.ctx,
                        self.device_index,
                        dtype,
                        n_elements,
                    )?,
                    ArenaInit::Host(bytes) => {
                        let expected = dtype.packed_buffer_bytes(n_elements);
                        if bytes.len() != expected {
                            return Err(Error::Msg(format!(
                                "CaptureArena host allocation byte length mismatch: \
                                 got {}, expected {expected} for ({dtype:?}, {n_elements})",
                                bytes.len()
                            )));
                        }
                        let stream = crate::active_cuda_stream(&self.ctx);
                        let slice = stream.clone_htod(bytes).map_err(|e| {
                            Error::Msg(format!(
                                "CaptureArena host allocation clone_htod({expected}) failed: {e:?}"
                            ))
                        })?;
                        // `clone_htod` is asynchronous for pageable host memory.
                        // The retained `ArenaInit::Host` bytes are moved below,
                        // but synchronizing here also makes the one-time warm
                        // upload complete before any capture can begin.
                        stream.synchronize().map_err(|e| {
                            Error::Msg(format!(
                                "CaptureArena host allocation stream synchronize failed: {e:?}"
                            ))
                        })?;
                        CudaStorage::from_slice_ctx(&self.ctx, self.device_index, dtype, slice)?
                    }
                };
                let storage = Arc::new(owned);
                let view = self.borrow_view(dtype, &storage)?;
                self.bufs.push(ArenaBuf {
                    dtype,
                    n_elements,
                    init,
                    storage,
                });
                Ok(view)
            }
            ArenaMode::Replay => {
                let buf = self.bufs.get(self.cursor).ok_or_else(|| {
                    Error::Msg(format!(
                        "CaptureArena replay overran: cursor {} >= {} recorded buffers \
                         (forward non-deterministic between warm and capture passes)",
                        self.cursor,
                        self.bufs.len()
                    ))
                })?;
                if buf.dtype != dtype || buf.n_elements != n_elements {
                    return Err(Error::Msg(format!(
                        "CaptureArena replay shape mismatch at alloc #{}: recorded \
                         ({:?}, {}) vs requested ({:?}, {})",
                        self.cursor, buf.dtype, buf.n_elements, dtype, n_elements
                    )));
                }
                if buf.init.name() != init.name() {
                    return Err(Error::Msg(format!(
                        "CaptureArena replay initialization mismatch at alloc #{}: \
                         recorded {} vs requested {}",
                        self.cursor,
                        buf.init.name(),
                        init.name()
                    )));
                }
                if let (ArenaInit::Host(recorded), ArenaInit::Host(requested)) = (&buf.init, &init)
                {
                    if recorded != requested {
                        let first_diff = recorded
                            .iter()
                            .zip(requested)
                            .position(|(left, right)| left != right)
                            .map_or(recorded.len().min(requested.len()), |offset| offset);
                        return Err(Error::Msg(format!(
                            "CaptureArena replay host data mismatch at alloc #{}: \
                             recorded {} bytes vs requested {} bytes; first difference \
                             at byte {first_diff}",
                            self.cursor,
                            recorded.len(),
                            requested.len()
                        )));
                    }
                }
                let storage = buf.storage.clone();
                let view = self.borrow_view(dtype, &storage)?;
                if init == ArenaInit::Zero {
                    // Captured memset on the active (capture) stream — recorded
                    // into the graph so every replay re-zeros the buffer.
                    self.memset_zero(&storage)?;
                }
                self.cursor += 1;
                Ok(view)
            }
        }
    }

    /// Build a Borrowed `CudaStorage` view into `storage`, keeping the owned
    /// allocation alive via the keep-alive `Arc`.
    fn borrow_view(&self, dtype: DType, storage: &Arc<CudaStorage>) -> Result<CudaStorage> {
        let (ptr, byte_len) = storage.device_ptr_raw();
        let keep_alive: Arc<dyn Any + Send + Sync> = storage.clone();
        CudaStorage::from_borrowed_ctx(
            &self.ctx,
            self.device_index,
            dtype,
            ptr,
            byte_len,
            keep_alive,
        )
    }

    /// Zero the buffer on the active CUDA stream (captured during the replay
    /// pass via `cuMemsetD8Async`).
    fn memset_zero(&self, storage: &Arc<CudaStorage>) -> Result<()> {
        let (ptr, byte_len) = storage.device_ptr_raw();
        let stream = crate::active_cuda_stream(&self.ctx);
        let cu_stream = stream.cu_stream();
        // SAFETY: `ptr` points at `byte_len` valid device bytes owned by the
        // arena (alive via `storage`); `cu_stream` is the active capture stream
        // the buffer was (re)used on. Stream-ordered zero-fill, recorded into
        // the graph during the replay (capture) pass.
        unsafe {
            cudarc::driver::result::memset_d8_async(ptr, 0u8, byte_len, cu_stream).map_err(
                |e| {
                    Error::Msg(format!(
                        "CaptureArena::memset_zero: memset_d8_async({byte_len}) failed: {e:?}"
                    ))
                },
            )?;
        }
        Ok(())
    }

    /// Consume the arena, returning the retained owned buffers so the caller
    /// (`CapturedDecodeGraph`) can keep them mapped for every replay.
    pub fn into_retained(self) -> Vec<Arc<CudaStorage>> {
        self.bufs.into_iter().map(|b| b.storage).collect()
    }

    /// Drain the retained owned buffers out of the arena (leaving it empty),
    /// returning them so the caller can keep them mapped for every replay.
    /// Use when the arena is behind an `Rc<RefCell<…>>` and can't be consumed
    /// by value.
    pub fn take_retained(&mut self) -> Vec<Arc<CudaStorage>> {
        std::mem::take(&mut self.bufs)
            .into_iter()
            .map(|b| b.storage)
            .collect()
    }
}

thread_local! {
    /// The capture arena active on this thread, or `None` outside a
    /// [`with_capture_arena`] scope (in which case `zeros_ctx` /
    /// `alloc_uninit_ctx` / host-to-device construction take their normal
    /// direct-allocation path — ZERO behavior change for every non-capture
    /// caller).
    static CAPTURE_ARENA: RefCell<Option<Rc<RefCell<CaptureArena>>>> =
        const { RefCell::new(None) };
}

/// RAII guard restoring the previous arena on scope exit (panic-safe).
struct ArenaGuard {
    prev: Option<Rc<RefCell<CaptureArena>>>,
}

impl Drop for ArenaGuard {
    fn drop(&mut self) {
        CAPTURE_ARENA.with(|cell| {
            *cell.borrow_mut() = self.prev.take();
        });
    }
}

/// Run `f` with `arena` installed as the active capture arena on this thread,
/// restoring the previous value afterward (even on panic). Inside `f`, every
/// `zeros_ctx` / `alloc_uninit_ctx` / host-to-device tensor construction routes
/// through the arena.
pub fn with_capture_arena<R>(arena: Rc<RefCell<CaptureArena>>, f: impl FnOnce() -> R) -> R {
    let prev = CAPTURE_ARENA.with(|cell| cell.borrow_mut().replace(arena));
    let _guard = ArenaGuard { prev };
    f()
}

/// Allocation hook called from `CudaStorage::zeros_ctx` (`zero = true`) and
/// `alloc_uninit_ctx` (`zero = false`).
///
/// Returns `Some(Ok(view))` — a Borrowed CudaStorage view into the arena — when
/// a capture arena is active on this thread; `None` otherwise (the caller then
/// takes its normal direct cudarc allocation path).
pub fn capture_arena_alloc(
    dtype: DType,
    n_elements: usize,
    zero: bool,
) -> Option<Result<CudaStorage>> {
    // Take the arena OUT of the thread-local for the duration of the alloc.
    // The `Record`-mode path calls `CudaStorage::zeros_ctx` /
    // `alloc_uninit_ctx` to make the real owned buffer — those re-enter this
    // hook, and with the arena removed they correctly take the direct cudarc
    // path instead of recursing (or double-borrowing the `RefCell`). Restored
    // immediately after.
    let arena = CAPTURE_ARENA.with(|cell| cell.borrow_mut().take());
    let arena = arena?;
    let init = if zero {
        ArenaInit::Zero
    } else {
        ArenaInit::Uninit
    };
    let result = arena.borrow_mut().alloc(dtype, n_elements, init);
    CAPTURE_ARENA.with(|cell| {
        *cell.borrow_mut() = Some(arena);
    });
    Some(result)
}

/// Host-initialized allocation hook called by
/// [`crate::host_to_cuda_copy`].
///
/// Record mode uploads `bytes` once into an arena-owned device allocation.
/// Replay mode requires identical bytes and returns the same stable device
/// pointer without recording a pageable-host memcpy in the CUDA graph.
pub fn capture_arena_from_host(
    dtype: DType,
    n_elements: usize,
    bytes: &[u8],
) -> Option<Result<CudaStorage>> {
    let arena = CAPTURE_ARENA.with(|cell| cell.borrow_mut().take());
    let arena = arena?;
    let result = arena
        .borrow_mut()
        .alloc(dtype, n_elements, ArenaInit::Host(bytes.to_vec()));
    CAPTURE_ARENA.with(|cell| {
        *cell.borrow_mut() = Some(arena);
    });
    Some(result)
}

/// Whether a capture arena is active on this thread.
pub fn capture_arena_active() -> bool {
    CAPTURE_ARENA.with(|cell| cell.borrow().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host_alloc(arena: Rc<RefCell<CaptureArena>>, bytes: &[u8]) -> Result<CudaStorage> {
        with_capture_arena(arena, || {
            capture_arena_from_host(DType::U8, bytes.len(), bytes)
                .expect("capture arena must be active")
        })
    }

    fn zero_alloc(arena: Rc<RefCell<CaptureArena>>, n_elements: usize) -> Result<CudaStorage> {
        with_capture_arena(arena, || {
            capture_arena_alloc(DType::U8, n_elements, true).expect("capture arena must be active")
        })
    }

    #[test]
    fn host_initialized_replay_reuses_pointer_and_rejects_drift() -> Result<()> {
        let ctx = crate::primary_cuda_context(0)?;
        let arena = Rc::new(RefCell::new(CaptureArena::new_record(ctx, 0)));
        let host_bytes = [3u8, 1, 4, 1, 5, 9];

        let recorded_host = host_alloc(arena.clone(), &host_bytes)?;
        let recorded_zero = zero_alloc(arena.clone(), 8)?;
        assert_eq!(arena.borrow().buffer_count(), 2);

        arena.borrow_mut().begin_replay();
        let replayed_host = host_alloc(arena.clone(), &host_bytes)?;
        assert_eq!(
            recorded_host.device_ptr_raw(),
            replayed_host.device_ptr_raw(),
            "host-initialized replay must return the retained device allocation"
        );
        let underrun = arena.borrow().ensure_replay_complete().unwrap_err();
        assert!(underrun.to_string().contains("replay underrun"));
        let replayed_zero = zero_alloc(arena.clone(), 8)?;
        assert_eq!(
            recorded_zero.device_ptr_raw(),
            replayed_zero.device_ptr_raw(),
            "zero-initialized replay must return the retained device allocation"
        );
        arena.borrow().ensure_replay_complete()?;

        arena.borrow_mut().begin_replay();
        let drift = host_alloc(arena.clone(), &[3u8, 1, 4, 2, 5, 9]).unwrap_err();
        assert!(drift.to_string().contains("host data mismatch"));
        assert!(drift.to_string().contains("byte 3"));

        arena.borrow_mut().begin_replay();
        let kind_drift = zero_alloc(arena.clone(), host_bytes.len()).unwrap_err();
        assert!(kind_drift.to_string().contains("initialization mismatch"));
        Ok(())
    }
}

//! Vendored from candle-metal-kernels 0.10.2 `src/metal/commands.rs`
//! (MIT/Apache-2.0), then **re-architected for correctness** (#1082).
//!
//! # Why this diverges from the candle original
//!
//! candle's `Commands` kept a *pool* of N independent `MTLCommandBuffer`s
//! and spread encoders across them (`select_entry` round-robin), with
//! deferred commit (each `command_encoder()` accumulates a compute encoder
//! onto a pool buffer until `compute_per_buffer`; `flush_and_wait()` commits
//! + waits ALL pool buffers). That design is correct ONLY when the work
//! encoded onto different pool buffers is mutually independent.
//!
//! kiln chains **data-dependent** ops on-GPU with no intermediate host
//! reads: op N+1 consumes op N's freshly-written output buffer as its input
//! (the entire forward/backward pass is one long dependency chain). Metal
//! guarantees in-order execution *only* for command buffers committed in
//! order to the same queue. With the multi-buffer pool, the consumer's
//! command buffer could be committed (and thus execute) before the
//! producer's — so the consumer reads a not-yet-written input. Under
//! cross-thread contention (server inference racing a training job, or
//! cargo's parallel test threads) the commit interleaving varies, so the
//! SAME deterministic computation produced DIFFERENT results
//! (a read-before-write race — observed as a tiny SFT loss differing
//! between single-threaded and concurrent runs). With `pool_size == 1` the
//! race vanished; `pool_size >= 2` reproduced it deterministically.
//!
//! # The fix: a single strictly-ordered command-buffer stream
//!
//! `Commands` now keeps a **single** active command buffer that every
//! encoder appends to, committing it (and swapping in a fresh one) only at
//! the batch boundary (`compute_per_buffer`) or on `flush`/`flush_and_wait`,
//! in strict encode order. This makes
//!
//!   encode order == commit order == GPU execution order,
//!
//! so every data dependency between chained ops is honored. On a single GPU
//! this costs nothing in execution parallelism (one queue runs serially
//! regardless); it only removes the *false* parallelism that was corrupting
//! results. The deferred-commit batching (CPU/GPU overlap) and the async
//! in-flight wait list are preserved.
//!
//! Concurrency correctness is enforced by serializing the encode→commit
//! critical section on one `Mutex` (`state`): an encoder is handed out under
//! the lock with the stream marked `Encoding`, and the next encoder /
//! commit / flush waits (via the condvar) until that encoder signals
//! `end_encoding` (its `Drop`). Two encoders are therefore never live on the
//! same command buffer at once, and commits never race a live encoder.
//!
//! Only crate-internal paths are renamed from the candle source
//! (`crate::metal::` -> `super::`, `MetalKernelError` -> `MetalRtError`).
//! (#1082)

use super::{
    BlitCommandEncoder, CommandBuffer, CommandSemaphore, CommandStatus, ComputeCommandEncoder,
};
use super::MetalRtError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBufferStatus, MTLCommandQueue};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// Use Retained when appropriate. Gives us a more elegant way of handling memory (peaks) than autoreleasepool.
// https://docs.rs/objc2/latest/objc2/rc/struct.Retained.html
pub type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;

const DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: usize = 50;

/// Creates a new command buffer from the queue with an attached semaphore for tracking its state.
pub fn create_command_buffer(
    command_queue: &CommandQueue,
    semaphore: Arc<CommandSemaphore>,
) -> Result<CommandBuffer, MetalRtError> {
    command_queue
        .commandBuffer()
        .map(|raw| CommandBuffer::new(raw, semaphore))
        .ok_or(MetalRtError::FailedToCreateResource(
            "CommandBuffer".to_string(),
        ))
}

/// State of the single ordered command-buffer stream, guarded by one mutex.
///
/// `current` is the active (not-yet-committed) command buffer that the next
/// encoder appends to. `in_flight` holds buffers already committed (in
/// strict commit order) but not yet waited on — `flush_and_wait` drains
/// them. The shared `semaphore` tracks whether an encoder is currently live
/// on `current` (`Encoding`) or not (`Available`); a freshly-swapped buffer
/// reuses the same semaphore.
struct StreamState {
    current: CommandBuffer,
    in_flight: Vec<CommandBuffer>,
}

pub struct Commands {
    /// The single ordered command-buffer stream's state.
    state: Mutex<StreamState>,
    /// Tracks whether an encoder is live on `state.current` (`Encoding`) so
    /// commits/new-encoders wait for `end_encoding` before proceeding. One
    /// `CommandSemaphore` is shared across the lifetime of the stream and is
    /// re-attached to each freshly-created `current` buffer.
    semaphore: Arc<CommandSemaphore>,
    /// Count of compute encoders appended to the *current* buffer since it
    /// was last swapped. Drives the batch-boundary commit at
    /// `compute_per_buffer`.
    compute_count: AtomicUsize,
    /// Single command queue for the entire device.
    command_queue: CommandQueue,
    /// The maximum amount of [compute command encoder](https://developer.apple.com/documentation/metal/mtlcomputecommandencoder?language=objc) per [command buffer](https://developer.apple.com/documentation/metal/mtlcommandbuffer?language=objc)
    compute_per_buffer: usize,
}

impl std::fmt::Debug for Commands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Commands")
            .field("compute_per_buffer", &self.compute_per_buffer)
            .field("compute_count", &self.compute_count)
            .finish_non_exhaustive()
    }
}

unsafe impl Send for Commands {}
unsafe impl Sync for Commands {}

impl Commands {
    pub fn new(command_queue: CommandQueue) -> Result<Self, MetalRtError> {
        let compute_per_buffer = match std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER") {
            Ok(val) => val
                .parse()
                .unwrap_or(DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER),
            _ => DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
        };
        // NOTE: `CANDLE_METAL_COMMAND_POOL_SIZE` is intentionally ignored —
        // a multi-buffer pool re-introduces the cross-thread out-of-order
        // commit data race this module was re-architected to fix (a single
        // ordered stream is required for correctness on a single queue). The
        // env var is kept readable for back-compat but has no effect.

        let semaphore = Arc::new(CommandSemaphore::new());
        let current = create_command_buffer(&command_queue, Arc::clone(&semaphore))?;

        Ok(Self {
            state: Mutex::new(StreamState {
                current,
                in_flight: Vec::new(),
            }),
            semaphore,
            compute_count: AtomicUsize::new(0),
            command_queue,
            compute_per_buffer,
        })
    }

    pub fn command_encoder(&self) -> Result<(bool, ComputeCommandEncoder), MetalRtError> {
        self.encoder(|cb| cb.compute_command_encoder())
    }

    pub fn blit_command_encoder(&self) -> Result<(bool, BlitCommandEncoder), MetalRtError> {
        self.encoder(|cb| cb.blit_command_encoder())
    }

    pub fn wait_until_completed(&self) -> Result<(), MetalRtError> {
        self.flush_and_wait()
    }

    /// Hand out an encoder on the single ordered stream.
    ///
    /// Waits (via the condvar) until no encoder is live on `current`
    /// (`Available`), marks the stream `Encoding`, then — under the `state`
    /// lock — recycles `current` if it has hit `compute_per_buffer`
    /// (committing the old buffer in order and swapping in a fresh one) and
    /// creates the encoder. The returned encoder restores the stream to
    /// `Available` on `Drop` (its `end_encoding`), which the next
    /// encoder/commit/flush waits for. This guarantees two encoders are
    /// never live on the same command buffer concurrently and that commits
    /// never race a live encoder.
    fn encoder<F, E>(&self, create_encoder: F) -> Result<(bool, E), MetalRtError>
    where
        F: FnOnce(&mut CommandBuffer) -> E,
    {
        // Block until the stream is idle (no live encoder), then claim it.
        {
            let mut guard = self
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));
            *guard = CommandStatus::Encoding;
        }

        let mut state = self.state.lock()?;

        let count = self.compute_count.fetch_add(1, Ordering::Relaxed);
        let flush = count >= self.compute_per_buffer;
        if flush {
            // Batch boundary: commit the current buffer (in order) and swap
            // a fresh one in before encoding the next op onto it.
            self.commit_swap_locked(&mut state, 1)?;
        }

        let encoder = create_encoder(&mut state.current);

        Ok((flush, encoder))
    }

    /// Flushes the stream and waits for completion.
    ///
    /// Commits the current buffer (if it has pending work) in order, then
    /// waits on every committed-but-not-yet-waited buffer (including prior
    /// batch-boundary recycles) in commit order. This is the host-read
    /// synchronization point — after it returns, every op encoded so far has
    /// completed on the GPU, so freshly-written `StorageModeShared` buffers
    /// reflect their GPU writes through the UMA `contents()` pointer.
    pub fn flush_and_wait(&self) -> Result<(), MetalRtError> {
        let to_wait: Vec<CommandBuffer> = {
            // Ensure no encoder is still live on `current` before committing.
            let _guard = self
                .semaphore
                .wait_until(|s| matches!(s, CommandStatus::Available));

            let mut state = self.state.lock()?;

            if self.compute_count.load(Ordering::Acquire) > 0 {
                self.commit_swap_locked(&mut state, 0)?;
            }

            std::mem::take(&mut state.in_flight)
        };

        for cb in to_wait {
            Self::ensure_completed(&cb)?;
        }

        Ok(())
    }

    /// Flushes the stream without waiting for completion.
    /// Commits the current buffer (if it has pending work) in order and
    /// swaps in a fresh one; the committed buffer joins `in_flight`.
    pub fn flush(&self) -> Result<(), MetalRtError> {
        let _guard = self
            .semaphore
            .wait_until(|s| matches!(s, CommandStatus::Available));

        let mut state = self.state.lock()?;

        if self.compute_count.load(Ordering::Acquire) > 0 {
            self.commit_swap_locked(&mut state, 0)?;
        }

        Ok(())
    }

    /// Commit the current command buffer, swap in a fresh one, push the old
    /// into `in_flight`, and reset `compute_count` to `reset_to`.
    ///
    /// Must be called with `state` locked AND the stream `Available` (no live
    /// encoder) — both invariants are upheld by the two callers
    /// (`encoder`/`flush_and_wait`/`flush`). Committing in this single,
    /// lock-ordered path is what makes commit order == encode order.
    fn commit_swap_locked(
        &self,
        state: &mut StreamState,
        reset_to: usize,
    ) -> Result<(), MetalRtError> {
        state.current.commit();
        let new_cb = create_command_buffer(&self.command_queue, Arc::clone(&self.semaphore))?;
        let old_cb = std::mem::replace(&mut state.current, new_cb);
        state.in_flight.push(old_cb);
        self.compute_count.store(reset_to, Ordering::Release);

        Ok(())
    }

    fn ensure_completed(cb: &CommandBuffer) -> Result<(), MetalRtError> {
        match cb.status() {
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued => {
                cb.commit();
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled => {
                cb.wait_until_completed();
            }
            MTLCommandBufferStatus::Completed => {}
            MTLCommandBufferStatus::Error => {
                let msg = cb
                    .error()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                return Err(MetalRtError::CommandBufferError(msg));
            }
            _ => unreachable!(),
        }

        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // TODO: Avoid redundant allocation before drop
        let _ = self.flush();
    }
}

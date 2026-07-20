//! HIP (ROCm) graph capture and replay for decode forward passes — the ROCm
//! twin of [`crate::cuda_graph`] (R.9).
//!
//! During decode each step processes exactly one token with identical tensor
//! shapes, so the kernel sequence is captured once (`hipStreamBeginCapture`)
//! and replayed (`hipGraphLaunch`) to eliminate per-step host launch overhead.
//! The machinery mirrors `CudaGraphRunner` and rests on the R.9 foundations:
//!   * `kiln_tensor::rocm_write_host_in_place` — refresh per-step inputs through
//!     a graph-stable device pointer (Phase 1).
//!   * `kiln_tensor::RocmCaptureArena` — freeze every activation pointer the
//!     captured forward touches across capture→replay (Phase 2).
//!   * the `any(cuda,rocm)`-gated graph-inputs forward in `forward.rs`, whose kt
//!     consumption arms already dispatch to the ROCm SDPA / paged-KV kernels
//!     (Phase 3B.1).
//!
//! Like the CUDA path, the captured forward stops at the PRE-final-norm hidden
//! (`LmHeadMode::HiddenOnly`); `final_norm` + lm_head run EAGERLY off the graph
//! (`lm_head_from_hidden_eager`) because the lm_head GEMV is not replay-safe.
//!
//! Graph execution is selected once, when the model runner is constructed. It
//! is deliberately eager by default; callers must explicitly request lazy
//! capture/replay through [`RocmGraphExecutionPolicy`]. Capture or replay
//! failures trip a runner-local circuit breaker and fall back to eager.

use anyhow::{Context, Result};
use serde::Serialize;
use tracing;

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;

use crate::PagedKvCacheKt;
use crate::backend::BackendRuntime;
use crate::forward::{
    GpuWeights, LinearAttentionState, model_forward_paged,
    model_forward_paged_batched_decode_hidden, model_forward_paged_next_token_greedy,
};
use crate::lora_loader::LoraWeights;

#[cfg(feature = "rocm")]
use crate::forward::{
    BatchedPagedDecodeGraphInputs, PagedDecodeGraphInputs,
    model_forward_paged_batched_hidden_with_graph_inputs,
    model_forward_paged_decode_contiguous_batch_hidden_with_ids,
    model_forward_paged_hidden_with_graph_inputs,
};
#[cfg(feature = "rocm")]
use kiln_graph::{
    CaptureError, InvalidateReason, ReplayInputs, ReplayKey, ReplayOutputs, ReplayPlan,
    ReplayResourceStability, ReplayState, ResidentResourceRef,
};
#[cfg(feature = "rocm")]
use std::collections::{HashMap, HashSet};

#[cfg(feature = "rocm")]
use kiln_tensor::{Backend, StorageBackend};
use kiln_tensor::{Device, Tensor};

/// Runtime behavior for the ROCm decode graph state machine.
///
/// The modes describe observable execution, rather than exposing independent
/// booleans that can form contradictory configurations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RocmGraphExecutionMode {
    /// Bypass graph warmup, capture, and replay.
    #[default]
    Disabled,
    /// Run one graph-shaped eager warmup, then remain eager.
    WarmupThenEager,
    /// Warm up eagerly, then lazily capture and replay eligible decode shapes.
    LazyCaptureReplay,
}

impl RocmGraphExecutionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::WarmupThenEager => "warmup_then_eager",
            Self::LazyCaptureReplay => "lazy_capture_replay",
        }
    }
}

impl std::fmt::Display for RocmGraphExecutionMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Immutable ROCm decode graph policy installed with a model runner.
///
/// The default is intentionally eager. Product surfaces that expose
/// experimental graph execution must select [`Self::lazy_capture_replay`]
/// explicitly, which keeps stable and compatibility callers from enabling
/// native capture as a side effect of process environment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RocmGraphExecutionPolicy {
    mode: RocmGraphExecutionMode,
    max_cached_graphs: usize,
    max_retained_bytes: u64,
    force_eager_decode: bool,
}

impl RocmGraphExecutionPolicy {
    pub const DEFAULT_MAX_CACHED_GRAPHS: usize = 8;
    pub const MAX_CACHED_GRAPHS: usize = 64;
    pub const DEFAULT_MAX_RETAINED_BYTES: u64 = 1 << 30;
    pub const MIN_MAX_RETAINED_BYTES: u64 = 64 << 20;
    pub const MAX_MAX_RETAINED_BYTES: u64 = 16 << 30;

    /// Fully eager execution with no graph-shaped warmup.
    pub const fn disabled() -> Self {
        Self {
            mode: RocmGraphExecutionMode::Disabled,
            max_cached_graphs: Self::DEFAULT_MAX_CACHED_GRAPHS,
            max_retained_bytes: Self::DEFAULT_MAX_RETAINED_BYTES,
            force_eager_decode: false,
        }
    }

    /// Graph-shaped warmup followed by eager steady-state decode.
    pub const fn warmup_then_eager() -> Self {
        Self {
            mode: RocmGraphExecutionMode::WarmupThenEager,
            max_cached_graphs: Self::DEFAULT_MAX_CACHED_GRAPHS,
            max_retained_bytes: Self::DEFAULT_MAX_RETAINED_BYTES,
            force_eager_decode: false,
        }
    }

    /// Lazy native capture/replay with the default bounded graph cache.
    pub const fn lazy_capture_replay() -> Self {
        Self {
            mode: RocmGraphExecutionMode::LazyCaptureReplay,
            max_cached_graphs: Self::DEFAULT_MAX_CACHED_GRAPHS,
            max_retained_bytes: Self::DEFAULT_MAX_RETAINED_BYTES,
            force_eager_decode: false,
        }
    }

    /// Build a policy from typed product configuration.
    pub fn try_new(
        mode: RocmGraphExecutionMode,
        max_cached_graphs: usize,
        max_retained_bytes: u64,
        force_eager_decode: bool,
    ) -> Result<Self> {
        anyhow::ensure!(
            max_cached_graphs > 0,
            "ROCm graph cache capacity must be greater than zero"
        );
        anyhow::ensure!(
            max_cached_graphs <= Self::MAX_CACHED_GRAPHS,
            "ROCm graph cache capacity must not exceed {}",
            Self::MAX_CACHED_GRAPHS
        );
        anyhow::ensure!(
            max_retained_bytes >= Self::MIN_MAX_RETAINED_BYTES,
            "ROCm graph retained-byte budget must be at least {} bytes",
            Self::MIN_MAX_RETAINED_BYTES
        );
        anyhow::ensure!(
            max_retained_bytes <= Self::MAX_MAX_RETAINED_BYTES,
            "ROCm graph retained-byte budget must not exceed {} bytes",
            Self::MAX_MAX_RETAINED_BYTES
        );
        Ok(Self {
            mode,
            max_cached_graphs,
            max_retained_bytes,
            force_eager_decode,
        })
    }

    pub const fn mode(self) -> RocmGraphExecutionMode {
        self.mode
    }

    pub const fn max_cached_graphs(self) -> usize {
        self.max_cached_graphs
    }

    pub const fn max_retained_bytes(self) -> u64 {
        self.max_retained_bytes
    }

    pub const fn force_eager_decode(self) -> bool {
        self.force_eager_decode
    }

    /// Qualification-only override that preserves the configured graph mode
    /// while routing decode through eager execution for an A/B run.
    pub const fn with_force_eager_decode(mut self, force_eager_decode: bool) -> Self {
        self.force_eager_decode = force_eager_decode;
        self
    }

    pub fn with_max_cached_graphs(self, max_cached_graphs: usize) -> Result<Self> {
        Self::try_new(
            self.mode,
            max_cached_graphs,
            self.max_retained_bytes,
            self.force_eager_decode,
        )
    }

    pub fn with_max_retained_bytes(self, max_retained_bytes: u64) -> Result<Self> {
        Self::try_new(
            self.mode,
            self.max_cached_graphs,
            max_retained_bytes,
            self.force_eager_decode,
        )
    }
}

impl Default for RocmGraphExecutionPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(all(test, feature = "rocm"))]
static ROCM_TEST_NATIVE_REPLAY_LAUNCHES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(feature = "rocm")]
fn attributed_rocm_graph_synchronize(
    phase: &'static str,
    reason: &'static str,
    affected_bytes: u64,
    byte_scope: &'static str,
    synchronize: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let started = std::time::Instant::now();
    let result = synchronize();
    let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
    match &result {
        Ok(()) => tracing::info!(
            event = "gpu_memory_operation",
            operation = "synchronize",
            reason,
            outcome = "completed",
            phase,
            affected_bytes,
            byte_scope,
            wait_ms = duration_ms,
            duration_ms,
            "ROCm graph host synchronization completed"
        ),
        Err(error) => tracing::warn!(
            event = "gpu_memory_operation",
            operation = "synchronize",
            reason,
            outcome = "failed",
            phase,
            affected_bytes,
            byte_scope,
            error = %format!("{error:#}"),
            wait_ms = duration_ms,
            duration_ms,
            "ROCm graph host synchronization failed"
        ),
    }
    result
}

#[cfg(feature = "rocm")]
fn graph_tensor_bytes<'a>(tensors: impl IntoIterator<Item = &'a Tensor>) -> u64 {
    tensors.into_iter().fold(0u64, |total, tensor| {
        total.saturating_add(
            (tensor.element_count() as u64).saturating_mul(tensor.dtype().size_in_bytes() as u64),
        )
    })
}

#[cfg(feature = "rocm")]
fn quarantine_rocm_tensor_context(tensor: &Tensor) {
    if let Some(storage) = tensor
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::RocmStorage>()
    {
        storage.context().quarantine_execution();
    }
}

#[cfg(feature = "rocm")]
fn fail_closed_after_rocm_warmup<T>(weights: &GpuWeights, error: anyhow::Error) -> Result<T> {
    quarantine_rocm_tensor_context(&weights.embed_tokens);
    let settlement = weights
        .embed_tokens
        .device()
        .index()
        .ok_or_else(|| anyhow::anyhow!("ROCm graph warmup lost its device ordinal"))
        .and_then(|device_index| {
            kiln_tensor::rocm_synchronize_device_for(
                device_index,
                kiln_tensor::RocmSyncReason::ErrorRecovery,
            )
            .map_err(|sync_error| anyhow::anyhow!("{sync_error}"))
        });
    match settlement {
        Ok(()) => Err(error).context(
            "ROCm graph-shaped warmup failed after state may have advanced; execution is quarantined until process restart",
        ),
        Err(sync_error) => Err(error).context(format!(
            "ROCm graph-shaped warmup failed and device settlement also failed ({sync_error:#}); execution and cleanup are quarantined until process restart"
        )),
    }
}

#[cfg(feature = "rocm")]
fn synchronize_after_rocm_graph_capture_failure(device: &Device) -> Result<()> {
    let Some(device_idx) = device.index() else {
        return Ok(());
    };
    attributed_rocm_graph_synchronize(
        "failure_recovery_default_stream",
        "rocm_graph_capture_failure_recovery",
        0,
        "unknown_in_flight_device_work",
        || {
            kiln_tensor::rocm_synchronize_device_for(
                device_idx,
                kiln_tensor::RocmSyncReason::CaptureRollback,
            )
            .map_err(|error| anyhow::anyhow!("{error}"))
        },
    )
    .context("ROCm graph capture failure recovery could not settle the device")?;
    anyhow::ensure!(
        !kiln_tensor::rocm_cleanup_quarantined(device_idx)
            .context("query ROCm quarantine after graph capture recovery")?,
        "ROCm graph capture failure left execution quarantined; restart the process"
    );
    Ok(())
}

/// Cache key for a captured decode graph. With
/// stable paged metadata the block table / seq_len are refreshed in place each
/// replay, so they don't enter the key — only the FA2-bucketed K/V geometry
/// and batch width do. Stable metadata is a graph correctness invariant, not a
/// runtime knob.
#[cfg(feature = "rocm")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RocmGraphKey {
    batch_size: usize,
    max_seqlen_k: usize,
    max_blocks_per_seq: usize,
}

#[cfg(feature = "rocm")]
impl RocmGraphKey {
    fn exact_max_seqlen_k(attention_len: usize) -> usize {
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        attention_len.div_ceil(kblock_n) * kblock_n
    }

    fn new(paged_cache: &PagedKvCacheKt, seq_len: usize) -> Self {
        let attention_len = seq_len + 1;
        // Match eager's exact FA2 split geometry. A coarser graph bucket changes
        // the number of split-K partials and can perturb BF16 reductions even
        // when `seqused_k` masks the same logical K/V length.
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let max_seqlen_k = Self::exact_max_seqlen_k(attention_len);
        let pages_per_chunk = kblock_n / paged_cache.block_size();
        let max_blocks_per_seq = (max_seqlen_k / kblock_n) * pages_per_chunk;
        Self {
            batch_size: 1,
            max_seqlen_k,
            max_blocks_per_seq,
        }
    }

    fn new_batched(paged_cache: &PagedKvCacheKt, sequence_lengths: &[usize]) -> Result<Self> {
        anyhow::ensure!(
            sequence_lengths.len() > 1,
            "batched ROCm graph key requires more than one row"
        );
        let max_seq_len = sequence_lengths
            .iter()
            .copied()
            .max()
            .context("batched ROCm graph key requires sequence lengths")?;
        let mut key = Self::new(paged_cache, max_seq_len);
        key.batch_size = sequence_lengths.len();
        Ok(key)
    }
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum RocmGraphOwner {
    Slot(u64),
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmGraphBindOutcome {
    Bound(RocmGraphOwner),
    Fallback(RocmGraphFallbackReason),
}

#[cfg(feature = "rocm")]
impl RocmGraphOwner {
    fn slot_id(self) -> u64 {
        match self {
            Self::Slot(slot_id) => slot_id,
        }
    }
}

#[cfg(feature = "rocm")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct RocmGraphCacheKey {
    owner: RocmGraphOwner,
    graph: RocmGraphKey,
}

#[cfg(feature = "rocm")]
impl RocmGraphCacheKey {
    fn new(owner: RocmGraphOwner, graph: RocmGraphKey) -> Self {
        Self { owner, graph }
    }
}

#[cfg(feature = "rocm")]
#[derive(Default)]
struct RocmGraphOwnerTimeline {
    last_decode_seq_len: Option<usize>,
    last_decode_block0: Option<u32>,
}

#[cfg(feature = "rocm")]
struct RocmGraphSlotState {
    assigned_row: Option<u64>,
    /// `Some(width)` reserves this otherwise-idle slot for a shared batched
    /// graph. Single-row request owners must never adopt it.
    batch_size: Option<usize>,
    linear_state: LinearAttentionState,
    allocations: Vec<RocmAllocationRecord>,
    accounting_complete: bool,
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy)]
struct RocmGraphCandidateSlot<'a> {
    owner: RocmGraphOwner,
    allocations: &'a [RocmAllocationRecord],
    accounting_complete: bool,
}

/// Ensures every failed capture attempt settles device work before any
/// pointer-bearing graph buffers leave scope. A failed recovery marks the HIP
/// context cleanup-quarantined, so low-level owners retain their handles and
/// allocations until process exit instead of freeing uncertain live memory.
#[cfg(any(feature = "rocm", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmCaptureGateAction {
    KeepOpen,
    PublishStop,
}

#[cfg(any(feature = "rocm", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RocmCaptureRollbackState {
    armed: bool,
    physically_settled: bool,
}

#[cfg(any(feature = "rocm", test))]
impl RocmCaptureRollbackState {
    const fn new() -> Self {
        Self {
            armed: true,
            physically_settled: false,
        }
    }

    fn record_settlement(&mut self, succeeded: bool) -> RocmCaptureGateAction {
        if succeeded {
            self.physically_settled = true;
            RocmCaptureGateAction::KeepOpen
        } else {
            self.armed = false;
            RocmCaptureGateAction::PublishStop
        }
    }

    fn record_logical_rollback(&mut self, succeeded: bool) -> RocmCaptureGateAction {
        self.armed = false;
        if succeeded && self.physically_settled {
            RocmCaptureGateAction::KeepOpen
        } else {
            RocmCaptureGateAction::PublishStop
        }
    }

    fn disarm_after_success(&mut self) {
        self.armed = false;
    }

    const fn exit_action(self) -> RocmCaptureGateAction {
        if self.armed {
            RocmCaptureGateAction::PublishStop
        } else {
            RocmCaptureGateAction::KeepOpen
        }
    }
}

#[cfg(feature = "rocm")]
struct RocmCaptureFailureGuard {
    context: std::sync::Arc<kiln_hip::RocmContext>,
    graph: Option<kiln_hip::RocmGraph>,
    exec: Option<kiln_hip::RocmGraphExec>,
    rollback: RocmCaptureRollbackState,
}

#[cfg(feature = "rocm")]
impl RocmCaptureFailureGuard {
    fn new(context: std::sync::Arc<kiln_hip::RocmContext>) -> Self {
        Self {
            context,
            graph: None,
            exec: None,
            rollback: RocmCaptureRollbackState::new(),
        }
    }

    fn disarm(&mut self) {
        self.rollback.disarm_after_success();
    }

    fn settle_before_rollback(&mut self) -> Result<()> {
        // This is the explicitly recoverable path: settle physical capture work
        // while admission remains open. STOP is published only if settlement
        // itself fails or logical state cannot subsequently be restored.
        let result = self
            .context
            .synchronize_device_for(kiln_tensor::RocmSyncReason::CaptureRollback)
            .map_err(|error| anyhow::anyhow!("{error}"));
        if self.rollback.record_settlement(result.is_ok()) == RocmCaptureGateAction::PublishStop {
            // The low-level synchronization path has already set the shared,
            // process-lifetime device quarantine. Do not retry from Drop: a
            // later successful drain must not make incomplete logical rollback
            // look recoverable.
            self.context.quarantine_execution();
        }
        result
    }

    fn complete_rollback(&mut self, rollback: Result<()>) -> Result<()> {
        if self.rollback.record_logical_rollback(rollback.is_ok())
            == RocmCaptureGateAction::PublishStop
        {
            // Physical quiescence is insufficient when recurrent model state
            // could not be restored (or settlement was not acknowledged).
            self.context.quarantine_execution();
        }
        rollback
    }
}

#[cfg(feature = "rocm")]
impl Drop for RocmCaptureFailureGuard {
    fn drop(&mut self) {
        if self.rollback.exit_action() == RocmCaptureGateAction::KeepOpen {
            return;
        }
        // An armed Drop has already lost the caller's logical rollback result.
        // Publish STOP before the last-chance drain so no new work can race it.
        self.context.quarantine_execution();
        let result = self
            .context
            .synchronize_device_for(kiln_tensor::RocmSyncReason::ErrorRecovery);
        // Any armed Drop means the caller did not acknowledge a completed
        // logical rollback. Keep the device quarantined even if this last-chance
        // physical drain succeeds.
        match result {
            Ok(()) => tracing::error!(
                "ROCm capture exited without completed logical rollback; execution is quarantined"
            ),
            Err(error) => tracing::error!(
                error = %error,
                "ROCm capture failure recovery could not settle the device; execution and cleanup are quarantined"
            ),
        }
    }
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RocmAllocationKey {
    device_index: usize,
    allocation_id: u64,
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RocmAllocationRecord {
    key: RocmAllocationKey,
    bytes: u64,
}

#[cfg(feature = "rocm")]
#[derive(Clone, Debug, Default)]
struct RocmGraphEntryAccounting {
    stable_io: Vec<RocmAllocationRecord>,
    capture_arena: Vec<RocmAllocationRecord>,
    blaslt_workspace: Option<RocmAllocationRecord>,
}

#[cfg(feature = "rocm")]
impl RocmGraphEntryAccounting {
    /// Exact deduplicated bytes owned by this transient/cached graph entry.
    /// Persistent recurrent slot allocations are intentionally excluded.
    fn retained_bytes_excluding_slot(&self) -> u64 {
        let mut seen = HashSet::new();
        let mut bytes = 0;
        for allocation in self
            .stable_io
            .iter()
            .chain(self.capture_arena.iter())
            .copied()
            .chain(self.blaslt_workspace)
        {
            RocmGraphMemoryAccounting::add_record(&mut bytes, &mut seen, allocation);
        }
        bytes
    }
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RocmGraphMemoryAccounting {
    stable_io_bytes: u64,
    capture_arena_bytes: u64,
    blaslt_workspace_bytes: u64,
    slot_state_bytes: u64,
    retained_bytes: u64,
    opaque_native_object_count: usize,
    complete: bool,
}

#[cfg(feature = "rocm")]
impl RocmGraphMemoryAccounting {
    fn add_record(
        total: &mut u64,
        seen: &mut HashSet<RocmAllocationKey>,
        record: RocmAllocationRecord,
    ) {
        if record.bytes > 0 && seen.insert(record.key) {
            *total = total.saturating_add(record.bytes);
        }
    }

    fn finish(&mut self) {
        self.retained_bytes = self
            .stable_io_bytes
            .saturating_add(self.capture_arena_bytes)
            .saturating_add(self.blaslt_workspace_bytes)
            .saturating_add(self.slot_state_bytes);
    }
}

#[cfg(feature = "rocm")]
fn rocm_tensor_allocation(tensor: &Tensor) -> Option<RocmAllocationRecord> {
    let storage = tensor
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::RocmStorage>()?;
    let device_index = match storage.device() {
        Device::Rocm(device_index) => device_index,
        _ => return None,
    };
    let (allocation_id, bytes) = storage.device_ptr_raw();
    Some(RocmAllocationRecord {
        key: RocmAllocationKey {
            device_index,
            allocation_id,
        },
        bytes: bytes as u64,
    })
}

#[cfg(feature = "rocm")]
fn unique_rocm_tensor_allocations<'a>(
    tensors: impl IntoIterator<Item = &'a Tensor>,
) -> Result<Vec<RocmAllocationRecord>> {
    let (allocations, complete) = inspect_rocm_tensor_allocations(tensors);
    anyhow::ensure!(
        complete,
        "ROCm graph accounting encountered non-ROCm tensor storage"
    );
    Ok(allocations)
}

#[cfg(feature = "rocm")]
fn inspect_rocm_tensor_allocations<'a>(
    tensors: impl IntoIterator<Item = &'a Tensor>,
) -> (Vec<RocmAllocationRecord>, bool) {
    let mut seen = HashSet::new();
    let mut allocations = Vec::new();
    let mut complete = true;
    for tensor in tensors {
        match rocm_tensor_allocation(tensor) {
            Some(allocation) if seen.insert(allocation.key) => allocations.push(allocation),
            Some(_) => {}
            None => complete = false,
        }
    }
    (allocations, complete)
}

#[cfg(feature = "rocm")]
fn unique_rocm_storage_allocations<'a>(
    device_index: usize,
    storages: impl IntoIterator<Item = &'a std::sync::Arc<kiln_tensor::RocmStorage>>,
) -> Vec<RocmAllocationRecord> {
    let mut seen = HashSet::new();
    let mut allocations = Vec::new();
    for storage in storages {
        let (allocation_id, bytes) = storage.device_ptr_raw();
        let allocation = RocmAllocationRecord {
            key: RocmAllocationKey {
                device_index,
                allocation_id,
            },
            bytes: bytes as u64,
        };
        if seen.insert(allocation.key) {
            allocations.push(allocation);
        }
    }
    allocations
}

/// A captured HIP graph ready for replay, plus every graph-stable buffer whose
/// device pointer the graph baked in. Mirrors `CapturedDecodeGraph`.
#[cfg(feature = "rocm")]
struct CapturedDecodeGraphRocm {
    accounting: RocmGraphEntryAccounting,
    last_used_tick: u64,
    /// The source graph — retained because dropping it `hipGraphDestroy`s the
    /// handle; the exec is launched, the graph is kept alive alongside it.
    _graph: kiln_hip::RocmGraph,
    /// The instantiated, launchable graph. ROCm uses plain instantiation
    /// (`flags = 0`); auto-free was rejected on gfx1151 / ROCm 7.2.4.
    exec: kiln_hip::RocmGraphExec,
    /// Graph-stable PRE-final-norm hidden `[batch, 1, hidden]`; refreshed in
    /// place by the captured forward, read eagerly by lm_head after launch.
    output_hidden: Tensor,
    /// The non-default capture stream the graph launches on. Replay completion
    /// is ordered into `default_stream` without blocking the host.
    capture_stream: std::sync::Arc<kiln_hip::RocmStream>,
    /// Exact context whose cleanup quarantine is shared by every retained HIP
    /// resource in this graph.
    context: std::sync::Arc<kiln_hip::RocmContext>,
    /// The kt default stream that receives refreshed inputs and consumes the
    /// graph-stable hidden output.
    default_stream: std::sync::Arc<kiln_hip::RocmStream>,
    /// Reusable default-to-capture dependency for refreshed replay inputs.
    replay_inputs_ready_event: std::sync::Arc<kiln_hip::RocmEvent>,
    /// Reusable capture-to-default dependency for replayed hidden output.
    replay_complete_event: std::sync::Arc<kiln_hip::RocmEvent>,
    adapter_gen: u64,
    /// Exact paged-KV allocation/generation whose pool pointers are embedded in
    /// the captured kernels.
    kv_pool_identity: crate::KvPoolIdentity,
    token_buffer: Tensor,
    position_buffer: Tensor,
    block_table_buffer: Option<Tensor>,
    seqused_k_buffer: Option<Tensor>,
    kv_slot_buffer: Option<Tensor>,
    rotary_cos_buffer: Tensor,
    rotary_sin_buffer: Tensor,
    _paged_decode_outputs: Vec<Tensor>,
    _paged_decode_lse: Vec<Tensor>,
    max_seqlen_k: usize,
    _gdn_decode_outputs: Vec<Tensor>,
    /// The freeze-pointer arena buffers (Q/K/V/activations); retained so their
    /// device pointers stay mapped for every replay.
    _capture_arena_buffers: Vec<std::sync::Arc<kiln_tensor::RocmStorage>>,
    /// Shared graph-layer replay contract state captured alongside the native
    /// HIP graph. The production replay path validates this before launching.
    replay_state: ReplayState,
    /// Declared last so native graph handles and graph-stable tensors are
    /// destroyed before the capture stream's hipBLASLt workspace is reclaimed.
    _blaslt_workspace_lease: kiln_tensor::HipblasLtWorkspaceLease,
}

#[cfg(feature = "rocm")]
struct RocmDecodeReplayPlan<'a> {
    captured: &'a CapturedDecodeGraphRocm,
}

#[cfg(feature = "rocm")]
impl<'a> RocmDecodeReplayPlan<'a> {
    fn new(captured: &'a CapturedDecodeGraphRocm) -> Self {
        Self { captured }
    }
}

#[cfg(feature = "rocm")]
impl std::fmt::Debug for RocmDecodeReplayPlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmDecodeReplayPlan")
            .field("key", &self.captured.replay_state.key)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "rocm")]
impl ReplayPlan for RocmDecodeReplayPlan<'_> {
    fn backend(&self) -> Backend {
        Backend::Rocm
    }

    fn key(&self) -> ReplayKey {
        self.captured.replay_state.key.clone()
    }

    fn validate_inputs(&self, inputs: ReplayInputs<'_>) -> Result<(), CaptureError> {
        self.captured
            .replay_state
            .validate(inputs.key, inputs.resources)
    }

    fn replay(&mut self, inputs: ReplayInputs<'_>) -> Result<ReplayOutputs, CaptureError> {
        self.validate_inputs(inputs)?;
        #[cfg(test)]
        ROCM_TEST_NATIVE_REPLAY_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Err(error) = self.captured.exec.launch(&self.captured.capture_stream) {
            self.captured.context.quarantine_execution();
            return Err(CaptureError::Backend(format!("ROCm graph launch: {error}")));
        }
        if let Err(error) = self
            .captured
            .capture_stream
            .record_event(&self.captured.replay_complete_event)
        {
            self.captured.context.quarantine_execution();
            return Err(CaptureError::Backend(format!(
                "record ROCm graph replay completion: {error}"
            )));
        }
        if let Err(error) = self
            .captured
            .default_stream
            .wait_event(&self.captured.replay_complete_event)
        {
            self.captured.context.quarantine_execution();
            return Err(CaptureError::Backend(format!(
                "order ROCm graph replay output handoff: {error}"
            )));
        }
        Ok(ReplayOutputs::new(inputs.resources.to_vec(), 1))
    }

    fn invalidate_reason(&self, state: &ReplayState) -> Option<InvalidateReason> {
        self.captured
            .replay_state
            .invalidate_reason(&state.key, &state.inputs)
    }
}

#[cfg(feature = "rocm")]
enum RocmCaptureStep {
    CapturedHidden(Tensor),
    CapturedHiddenUncached(Tensor),
    FallbackEager {
        reason: RocmGraphFallbackReason,
        cleanup_timer: Option<RocmGraphPhaseTimer>,
    },
}

#[cfg(feature = "rocm")]
impl RocmCaptureStep {
    fn fallback(reason: RocmGraphFallbackReason) -> Self {
        Self::FallbackEager {
            reason,
            cleanup_timer: None,
        }
    }

    fn fallback_after_candidate(
        reason: RocmGraphFallbackReason,
        telemetry: &RocmGraphTelemetryHandle,
    ) -> Self {
        Self::FallbackEager {
            reason,
            cleanup_timer: Some(telemetry.timer(RocmGraphPhase::RejectedCandidateCleanup)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RocmGraphCounters {
    capture_attempts: u64,
    capture_successes: u64,
    capture_deferrals: u64,
    capture_failures: u64,
    replay_attempts: u64,
    replay_successes: u64,
    replay_failures: u64,
    decode_owner_release_count: u64,
    decode_owner_graph_release_count: u64,
    graph_slot_create_count: u64,
    graph_slot_reuse_count: u64,
    cache_admission_successes: u64,
    cache_evictions: u64,
    cache_evicted_bytes: u64,
    budget_evictions: u64,
    pressure_evictions: u64,
    invalidation_evictions: u64,
    recovery_evictions: u64,
    entry_capacity_rejections: u64,
    byte_budget_rejections: u64,
    accounting_incomplete_rejections: u64,
    pre_capture_entry_capacity_skips: u64,
    pre_capture_byte_budget_skips: u64,
    pre_capture_accounting_incomplete_skips: u64,
    pre_capture_memory_reservation_denied_skips: u64,
    memory_governor_selector_mismatch_skips: u64,
    quarantined_retained_bytes: u64,
    fallbacks: RocmGraphFallbackStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmGraphCaptureOutcome {
    SucceededRetained,
    SucceededUncached,
    Deferred,
    Failed,
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmGraphEvictionReason {
    Budget,
    Pressure,
    Invalidation,
    Recovery,
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmGraphAdmissionRejection {
    EntryCapacity,
    ByteBudget,
    CandidateByteBudget,
    AccountingIncomplete,
}

#[cfg(feature = "rocm")]
#[derive(Debug, Default, Eq, PartialEq)]
struct RocmGraphAdmissionPlan {
    evict_owners: Vec<RocmGraphOwner>,
    evict_keys: Vec<RocmGraphCacheKey>,
}

#[cfg(feature = "rocm")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmGraphPressureDecision {
    Normal,
    ReplayOnly(RocmGraphFallbackReason),
    EagerOnly(RocmGraphFallbackReason),
}

#[cfg(feature = "rocm")]
fn non_evicting_pressure_decision(
    pressure: kiln_memory::MemoryPressure,
) -> Option<RocmGraphPressureDecision> {
    match pressure {
        kiln_memory::MemoryPressure::Comfortable => Some(RocmGraphPressureDecision::Normal),
        kiln_memory::MemoryPressure::Moderate => Some(RocmGraphPressureDecision::ReplayOnly(
            RocmGraphFallbackReason::ModerateMemoryPressure,
        )),
        kiln_memory::MemoryPressure::Tight | kiln_memory::MemoryPressure::Critical => None,
    }
}

fn memory_governor_selector_matches(
    expected: kiln_memory::VramProbeSelector,
    configured: kiln_memory::VramProbeSelector,
) -> bool {
    expected == configured
}

#[cfg(feature = "rocm")]
fn sort_idle_owner_lru(records: &mut [(u64, u64, RocmGraphOwner)]) {
    records.sort_unstable_by_key(|(last_used, slot_id, _)| (*last_used, *slot_id));
}

/// Order the exact graph entries that may be retired after all idle owners have
/// been exhausted. The incoming candidate counts toward its owner's projected
/// share. Each selection comes from the currently most-represented owner, then
/// uses exact LRU and stable owner/geometry tie-breakers. At least one graph is
/// retained for every active owner after candidate admission.
#[cfg(feature = "rocm")]
fn fair_active_geometry_eviction_order<'a>(
    candidate_owner: RocmGraphOwner,
    active_owners: impl IntoIterator<Item = RocmGraphOwner>,
    geometries: impl Iterator<Item = (&'a RocmGraphCacheKey, u64)>,
) -> Vec<RocmGraphCacheKey> {
    let active_owners: HashSet<_> = active_owners.into_iter().collect();
    let mut by_owner: HashMap<RocmGraphOwner, Vec<(RocmGraphCacheKey, u64)>> = HashMap::new();
    for (key, last_used_tick) in geometries {
        if active_owners.contains(&key.owner) {
            by_owner
                .entry(key.owner)
                .or_default()
                .push((key.clone(), last_used_tick));
        }
    }
    for entries in by_owner.values_mut() {
        entries.sort_unstable_by_key(|(key, last_used_tick)| {
            (
                *last_used_tick,
                key.graph.max_seqlen_k,
                key.graph.max_blocks_per_seq,
            )
        });
    }

    let mut projected_counts: HashMap<_, _> = active_owners
        .iter()
        .copied()
        .map(|owner| {
            let retained = by_owner.get(&owner).map_or(0, Vec::len);
            let incoming = usize::from(owner == candidate_owner);
            (owner, retained.saturating_add(incoming))
        })
        .collect();
    let mut order = Vec::new();
    loop {
        let selected_owner = by_owner
            .iter()
            .filter(|(owner, entries)| {
                !entries.is_empty() && projected_counts.get(owner).copied().unwrap_or(0) > 1
            })
            .min_by_key(|(owner, entries)| {
                let (oldest, last_used_tick) = &entries[0];
                (
                    std::cmp::Reverse(projected_counts.get(owner).copied().unwrap_or(0)),
                    *last_used_tick,
                    owner.slot_id(),
                    oldest.graph.max_seqlen_k,
                    oldest.graph.max_blocks_per_seq,
                )
            })
            .map(|(owner, _)| *owner);
        let Some(owner) = selected_owner else {
            break;
        };
        let (key, _) = by_owner
            .get_mut(&owner)
            .expect("selected fair ROCm graph owner must remain present")
            .remove(0);
        if let Some(count) = projected_counts.get_mut(&owner) {
            *count = count.saturating_sub(1);
        }
        order.push(key);
    }
    order
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RocmGraphFallbackReason {
    MultiRowBatchUnsupported,
    ColdCacheHostRoundTrip,
    PersistentHostRoundTrip,
    ShapeDependentAttention,
    GraphCacheCapacity,
    GraphCacheByteBudget,
    GraphAccountingIncomplete,
    ModerateMemoryPressure,
    TightMemoryPressure,
    CriticalMemoryPressure,
    MemoryReservationDenied,
    MemoryGovernorSelectorMismatch,
    CaptureFailure,
    ReplayFailure,
}

impl RocmGraphFallbackReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::MultiRowBatchUnsupported => "multi_row_batch_unsupported",
            Self::ColdCacheHostRoundTrip => "cold_cache_host_round_trip",
            Self::PersistentHostRoundTrip => "persistent_host_round_trip",
            Self::ShapeDependentAttention => "shape_dependent_attention",
            Self::GraphCacheCapacity => "graph_cache_capacity",
            Self::GraphCacheByteBudget => "graph_cache_byte_budget",
            Self::GraphAccountingIncomplete => "graph_accounting_incomplete",
            Self::ModerateMemoryPressure => "moderate_memory_pressure",
            Self::TightMemoryPressure => "tight_memory_pressure",
            Self::CriticalMemoryPressure => "critical_memory_pressure",
            Self::MemoryReservationDenied => "memory_reservation_denied",
            Self::MemoryGovernorSelectorMismatch => "memory_governor_selector_mismatch",
            Self::CaptureFailure => "capture_failure",
            Self::ReplayFailure => "replay_failure",
        }
    }
}

/// Bounded ROCm graph fallback counts and end-to-end eager-fallback latency.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RocmGraphFallbackStats {
    pub total: u64,
    pub multi_row_batch_unsupported: u64,
    pub cold_cache_host_round_trip: u64,
    pub persistent_host_round_trip: u64,
    pub shape_dependent_attention: u64,
    pub graph_cache_capacity: u64,
    pub graph_cache_byte_budget: u64,
    pub graph_accounting_incomplete: u64,
    pub moderate_memory_pressure: u64,
    pub tight_memory_pressure: u64,
    pub critical_memory_pressure: u64,
    pub memory_reservation_denied: u64,
    pub memory_governor_selector_mismatch: u64,
    pub capture_failure: u64,
    pub replay_failure: u64,
    pub slow: u64,
    pub total_duration_micros: u64,
    pub max_duration_micros: u64,
}

impl RocmGraphFallbackStats {
    const SLOW_DURATION: std::time::Duration = std::time::Duration::from_millis(100);

    fn record(&mut self, reason: RocmGraphFallbackReason, duration: std::time::Duration) -> u64 {
        self.total = self.total.saturating_add(1);
        let reason_count = match reason {
            RocmGraphFallbackReason::MultiRowBatchUnsupported => {
                &mut self.multi_row_batch_unsupported
            }
            RocmGraphFallbackReason::ColdCacheHostRoundTrip => &mut self.cold_cache_host_round_trip,
            RocmGraphFallbackReason::PersistentHostRoundTrip => {
                &mut self.persistent_host_round_trip
            }
            RocmGraphFallbackReason::ShapeDependentAttention => &mut self.shape_dependent_attention,
            RocmGraphFallbackReason::GraphCacheCapacity => &mut self.graph_cache_capacity,
            RocmGraphFallbackReason::GraphCacheByteBudget => &mut self.graph_cache_byte_budget,
            RocmGraphFallbackReason::GraphAccountingIncomplete => {
                &mut self.graph_accounting_incomplete
            }
            RocmGraphFallbackReason::ModerateMemoryPressure => &mut self.moderate_memory_pressure,
            RocmGraphFallbackReason::TightMemoryPressure => &mut self.tight_memory_pressure,
            RocmGraphFallbackReason::CriticalMemoryPressure => &mut self.critical_memory_pressure,
            RocmGraphFallbackReason::MemoryReservationDenied => &mut self.memory_reservation_denied,
            RocmGraphFallbackReason::MemoryGovernorSelectorMismatch => {
                &mut self.memory_governor_selector_mismatch
            }
            RocmGraphFallbackReason::CaptureFailure => &mut self.capture_failure,
            RocmGraphFallbackReason::ReplayFailure => &mut self.replay_failure,
        };
        *reason_count = reason_count.saturating_add(1);
        let occurrence = *reason_count;
        let duration_micros = duration.as_micros().min(u64::MAX as u128) as u64;
        self.total_duration_micros = self.total_duration_micros.saturating_add(duration_micros);
        self.max_duration_micros = self.max_duration_micros.max(duration_micros);
        if duration >= Self::SLOW_DURATION {
            self.slow = self.slow.saturating_add(1);
        }
        occurrence
    }
}

/// Bounded latency telemetry for one fixed ROCm graph-capture phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RocmGraphPhaseStats {
    pub calls: u64,
    /// Calls whose phase duration was at least 100 milliseconds.
    pub slow: u64,
    pub total_duration_micros: u64,
    pub max_duration_micros: u64,
}

impl RocmGraphPhaseStats {
    const SLOW_DURATION: std::time::Duration = std::time::Duration::from_millis(100);

    fn record(&mut self, duration: std::time::Duration) {
        self.calls = self.calls.saturating_add(1);
        let duration_micros = duration.as_micros().min(u64::MAX as u128) as u64;
        self.total_duration_micros = self.total_duration_micros.saturating_add(duration_micros);
        self.max_duration_micros = self.max_duration_micros.max(duration_micros);
        if duration >= Self::SLOW_DURATION {
            self.slow = self.slow.saturating_add(1);
        }
    }
}

/// Closed capture-phase labels for graph-runner-lock-independent observability.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RocmGraphPhase {
    PreCandidateHeadroom,
    CandidateWarm,
    PreNativeReservation,
    NativeCapture,
    RejectedCandidateCleanup,
}

/// Snapshot available through [`RocmGraphTelemetryHandle`] without locking the
/// graph runner. The active phase elapsed time is derived from a monotonic clock.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RocmGraphLiveTelemetry {
    pub current_phase: Option<RocmGraphPhase>,
    pub current_phase_elapsed_micros: u64,
    pub pre_candidate_headroom_phase: RocmGraphPhaseStats,
    pub candidate_warm_phase: RocmGraphPhaseStats,
    pub pre_native_reservation_phase: RocmGraphPhaseStats,
    pub native_capture_phase: RocmGraphPhaseStats,
    pub rejected_candidate_cleanup_phase: RocmGraphPhaseStats,
    pub last_transient_candidate_bytes: u64,
    pub peak_transient_candidate_bytes: u64,
}

/// Closed reasons why a full graph-runner snapshot could not be acquired.
///
/// Live phase telemetry remains available through
/// [`RocmGraphTelemetryHandle`] while the runner is busy.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RocmGraphStatsUnavailable {
    Busy,
    Poisoned,
}

impl std::fmt::Display for RocmGraphStatsUnavailable {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Busy => formatter.write_str("ROCm graph runner is busy"),
            Self::Poisoned => formatter.write_str("ROCm graph runner lock is poisoned"),
        }
    }
}

impl std::error::Error for RocmGraphStatsUnavailable {}

#[derive(Clone, Copy, Debug)]
struct RocmGraphActivePhase {
    phase: RocmGraphPhase,
    generation: u64,
    started: std::time::Instant,
}

#[derive(Debug, Default)]
struct RocmGraphTelemetryState {
    completed: RocmGraphLiveTelemetry,
    active: Option<RocmGraphActivePhase>,
    next_generation: u64,
}

/// Cloneable telemetry channel independent of the graph-runner mutex.
#[derive(Clone, Debug, Default)]
pub struct RocmGraphTelemetryHandle(std::sync::Arc<std::sync::Mutex<RocmGraphTelemetryState>>);

impl RocmGraphTelemetryHandle {
    fn timer(&self, phase: RocmGraphPhase) -> RocmGraphPhaseTimer {
        let started = std::time::Instant::now();
        let generation = {
            let mut telemetry = self.0.lock().unwrap_or_else(|error| error.into_inner());
            telemetry.next_generation = telemetry.next_generation.checked_add(1).unwrap_or(1);
            let generation = telemetry.next_generation;
            telemetry.active = Some(RocmGraphActivePhase {
                phase,
                generation,
                started,
            });
            generation
        };
        RocmGraphPhaseTimer {
            telemetry: self.clone(),
            phase,
            generation,
            started,
        }
    }

    fn record_transient_candidate_bytes(&self, bytes: u64) {
        let mut telemetry = self.0.lock().unwrap_or_else(|error| error.into_inner());
        telemetry.completed.last_transient_candidate_bytes = bytes;
        telemetry.completed.peak_transient_candidate_bytes = telemetry
            .completed
            .peak_transient_candidate_bytes
            .max(bytes);
    }

    pub fn snapshot(&self) -> RocmGraphLiveTelemetry {
        let telemetry = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let mut snapshot = telemetry.completed;
        if let Some(active) = telemetry.active {
            snapshot.current_phase = Some(active.phase);
            snapshot.current_phase_elapsed_micros =
                active.started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        }
        snapshot
    }
}

struct RocmGraphPhaseTimer {
    telemetry: RocmGraphTelemetryHandle,
    phase: RocmGraphPhase,
    generation: u64,
    started: std::time::Instant,
}

impl Drop for RocmGraphPhaseTimer {
    fn drop(&mut self) {
        let duration = self.started.elapsed();
        let mut telemetry = self
            .telemetry
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        phase_stats_mut(&mut telemetry.completed, self.phase).record(duration);
        if telemetry
            .active
            .is_some_and(|active| active.generation == self.generation)
        {
            telemetry.active = None;
        }
    }
}

fn phase_stats_mut(
    telemetry: &mut RocmGraphLiveTelemetry,
    phase: RocmGraphPhase,
) -> &mut RocmGraphPhaseStats {
    match phase {
        RocmGraphPhase::PreCandidateHeadroom => &mut telemetry.pre_candidate_headroom_phase,
        RocmGraphPhase::CandidateWarm => &mut telemetry.candidate_warm_phase,
        RocmGraphPhase::PreNativeReservation => &mut telemetry.pre_native_reservation_phase,
        RocmGraphPhase::NativeCapture => &mut telemetry.native_capture_phase,
        RocmGraphPhase::RejectedCandidateCleanup => &mut telemetry.rejected_candidate_cleanup_phase,
    }
}

impl RocmGraphCounters {
    fn record_capture_outcome(&mut self, outcome: RocmGraphCaptureOutcome) {
        self.capture_attempts = self.capture_attempts.saturating_add(1);
        match outcome {
            RocmGraphCaptureOutcome::SucceededRetained
            | RocmGraphCaptureOutcome::SucceededUncached => {
                self.capture_successes = self.capture_successes.saturating_add(1);
            }
            RocmGraphCaptureOutcome::Deferred => {
                self.capture_deferrals = self.capture_deferrals.saturating_add(1);
            }
            RocmGraphCaptureOutcome::Failed => {
                self.capture_failures = self.capture_failures.saturating_add(1);
            }
        }
    }

    fn record_replay_outcome(&mut self, succeeded: bool) {
        self.replay_attempts = self.replay_attempts.saturating_add(1);
        if succeeded {
            self.replay_successes = self.replay_successes.saturating_add(1);
        } else {
            self.replay_failures = self.replay_failures.saturating_add(1);
        }
    }

    fn record_decode_owner_release(&mut self, released_graphs: usize) {
        self.decode_owner_release_count = self.decode_owner_release_count.saturating_add(1);
        self.decode_owner_graph_release_count = self
            .decode_owner_graph_release_count
            .saturating_add(released_graphs as u64);
    }

    fn record_graph_slot_create(&mut self) {
        self.graph_slot_create_count = self.graph_slot_create_count.saturating_add(1);
    }

    fn record_graph_slot_reuse(&mut self) {
        self.graph_slot_reuse_count = self.graph_slot_reuse_count.saturating_add(1);
    }

    fn record_cache_admission(&mut self) {
        self.cache_admission_successes = self.cache_admission_successes.saturating_add(1);
    }

    #[cfg(feature = "rocm")]
    fn record_cache_rejection(&mut self, rejection: RocmGraphAdmissionRejection) {
        match rejection {
            RocmGraphAdmissionRejection::EntryCapacity => {
                self.entry_capacity_rejections = self.entry_capacity_rejections.saturating_add(1);
            }
            RocmGraphAdmissionRejection::ByteBudget
            | RocmGraphAdmissionRejection::CandidateByteBudget => {
                self.byte_budget_rejections = self.byte_budget_rejections.saturating_add(1);
            }
            RocmGraphAdmissionRejection::AccountingIncomplete => {
                self.accounting_incomplete_rejections =
                    self.accounting_incomplete_rejections.saturating_add(1);
            }
        }
    }

    #[cfg(feature = "rocm")]
    fn record_pre_capture_skip(&mut self, rejection: RocmGraphAdmissionRejection) {
        match rejection {
            RocmGraphAdmissionRejection::EntryCapacity => {
                self.pre_capture_entry_capacity_skips =
                    self.pre_capture_entry_capacity_skips.saturating_add(1);
            }
            RocmGraphAdmissionRejection::ByteBudget
            | RocmGraphAdmissionRejection::CandidateByteBudget => {
                self.pre_capture_byte_budget_skips =
                    self.pre_capture_byte_budget_skips.saturating_add(1);
            }
            RocmGraphAdmissionRejection::AccountingIncomplete => {
                self.pre_capture_accounting_incomplete_skips = self
                    .pre_capture_accounting_incomplete_skips
                    .saturating_add(1);
            }
        }
    }

    #[cfg(feature = "rocm")]
    fn record_cache_eviction(
        &mut self,
        graphs: usize,
        bytes: u64,
        reason: RocmGraphEvictionReason,
    ) {
        self.cache_evictions = self.cache_evictions.saturating_add(graphs as u64);
        self.cache_evicted_bytes = self.cache_evicted_bytes.saturating_add(bytes);
        match reason {
            RocmGraphEvictionReason::Budget => {
                self.budget_evictions = self.budget_evictions.saturating_add(graphs as u64);
            }
            RocmGraphEvictionReason::Pressure => {
                self.pressure_evictions = self.pressure_evictions.saturating_add(graphs as u64);
            }
            RocmGraphEvictionReason::Invalidation => {
                self.invalidation_evictions =
                    self.invalidation_evictions.saturating_add(graphs as u64);
            }
            RocmGraphEvictionReason::Recovery => {
                self.recovery_evictions = self.recovery_evictions.saturating_add(graphs as u64);
            }
        }
    }

    fn record_fallback(
        &mut self,
        reason: RocmGraphFallbackReason,
        duration: std::time::Duration,
    ) -> u64 {
        self.fallbacks.record(reason, duration)
    }
}

/// Point-in-time ROCm HIP-graph execution state.
///
/// The attempt/success/failure fields are monotonic for the lifetime of the
/// runner, including across adapter invalidations. The graph and owner counts
/// are live gauges and may decrease when state is released or invalidated.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct RocmGraphStats {
    /// The ROCm device and typed policy requested the graph runner when it was
    /// constructed. This remains true after a circuit break.
    pub requested: bool,
    /// The typed policy requested lazy native capture/replay.
    pub capture_requested: bool,
    /// The runner is still armed. Capture or replay failures set this false.
    pub enabled: bool,
    /// Native capture is both requested and currently armed.
    pub capture_enabled: bool,
    /// Configured maximum number of retained native graphs.
    pub max_cached_graphs: usize,
    /// Configured maximum requested physical bytes retained by graph-owned
    /// buffers. Opaque HIP graph objects and allocator-pool slack are excluded.
    pub max_retained_bytes: u64,
    /// Calls that entered the capture state machine, including pre-native
    /// deferrals after a settled warm Record pass.
    pub capture_attempts: u64,
    /// Native graphs successfully instantiated and launched once. This includes
    /// exact post-capture cache rejections; `cache_admission_successes` counts
    /// the subset retained for replay. After successful rejected-candidate
    /// cleanup, this equals admissions plus the three post-capture rejection
    /// counters below.
    pub capture_successes: u64,
    /// Attempts deferred before native capture, such as cold-cache warm passes.
    pub capture_deferrals: u64,
    /// Capture attempts that returned an error and tripped the circuit breaker.
    pub capture_failures: u64,
    /// Native replay launches attempted from any decode API.
    pub replay_attempts: u64,
    /// Native replay launches whose input/output stream dependencies and launch
    /// were queued successfully. External-yield settlement confirms device
    /// completion before progress is published.
    pub replay_successes: u64,
    /// Native replay launches that failed and tripped the circuit breaker.
    pub replay_failures: u64,
    /// Saturating sum of capture and replay failures.
    pub failures: u64,
    /// Finished decode owners whose tracked timeline or graph state was removed.
    pub decode_owner_release_count: u64,
    /// Captured graphs evicted by finished-owner cleanup.
    pub decode_owner_graph_release_count: u64,
    /// Persistent graph slots created over the runner lifetime.
    pub graph_slot_create_count: u64,
    /// Finished slots rebound to a later logical decode row.
    pub graph_slot_reuse_count: u64,
    /// Successful insertions into the bounded graph cache.
    pub cache_admission_successes: u64,
    /// Native graph entries evicted after safe device settlement.
    pub cache_evictions: u64,
    /// Requested physical device bytes released by successful evictions.
    pub cache_evicted_bytes: u64,
    /// Entries evicted to satisfy the entry or retained-byte budget.
    pub budget_evictions: u64,
    /// Entries evicted in response to Tight or Critical memory pressure.
    pub pressure_evictions: u64,
    /// Entries evicted by explicit adapter-generation invalidation.
    pub invalidation_evictions: u64,
    /// Entries evicted while recovering from replay/capture state mismatch.
    pub recovery_evictions: u64,
    /// Candidates rejected because no inactive-owner eviction plan could satisfy
    /// the entry cap.
    pub entry_capacity_rejections: u64,
    /// Candidates rejected because they alone exceeded the retained-byte cap or
    /// no inactive-owner eviction plan could satisfy it.
    pub byte_budget_rejections: u64,
    /// Successfully captured candidates rejected because exact allocation
    /// accounting could not be completed.
    pub accounting_incomplete_rejections: u64,
    /// Capture state-machine calls skipped before native capture at the entry
    /// cap. These are deliberately separate from post-capture rejections.
    pub pre_capture_entry_capacity_skips: u64,
    /// Capture state-machine calls skipped before native capture at the byte
    /// cap, including persistent-slot publication denials.
    pub pre_capture_byte_budget_skips: u64,
    /// Capture state-machine calls skipped before native capture because exact
    /// allocation accounting was incomplete.
    pub pre_capture_accounting_incomplete_skips: u64,
    /// Capture attempts abandoned because the process-wide governor could not
    /// atomically reserve the measured transient candidate bytes.
    pub pre_capture_memory_reservation_denied_skips: u64,
    /// Decode steps kept eager because the process-wide governor observes a
    /// different accelerator than this graph runner.
    pub memory_governor_selector_mismatch_skips: u64,
    /// Graphs currently retained in the live cache.
    pub captured_graph_count: usize,
    /// Persistent recurrent-state slots currently retained.
    pub graph_slot_count: usize,
    /// Retained slots currently assigned to live logical decode rows.
    pub active_graph_slot_count: usize,
    /// Retained slots available for a later logical decode row.
    pub idle_graph_slot_count: usize,
    /// Decode owners whose continuity timeline is currently retained.
    pub tracked_decode_owner_count: usize,
    /// Exact requested physical bytes held by graph-stable direct tensors.
    pub retained_stable_io_bytes: u64,
    /// Exact requested physical bytes held by freeze-pointer capture arenas.
    pub retained_capture_arena_bytes: u64,
    /// Exact requested physical bytes held by leased private-stream workspaces.
    pub retained_blaslt_workspace_bytes: u64,
    /// Exact requested physical bytes held by every runner-owned recurrent/conv
    /// owner slot, including active slots whose native graphs were evicted.
    pub retained_slot_state_bytes: u64,
    /// Deduplicated sum of all graph-owned requested physical bytes.
    pub retained_bytes: u64,
    /// Highest `retained_bytes` observed after slot creation or cache admission.
    pub peak_retained_bytes: u64,
    /// Native graph/exec/stream/event objects whose driver allocation size is not
    /// queryable through the current HIP API surface.
    pub opaque_native_object_count: usize,
    /// Whether every retained tensor could be mapped to ROCm allocation metadata.
    pub retained_bytes_accounting_complete: bool,
    /// Logical bytes whose destruction or post-drop settlement failed after the
    /// process-lifetime device quarantine became sticky.
    pub quarantined_retained_bytes: u64,
    /// Initial pressure reconciliation and settled eviction before allocating a
    /// warm candidate.
    pub pre_candidate_headroom_phase: RocmGraphPhaseStats,
    /// Candidate stream/buffer allocation plus the settled warm Record forward.
    pub candidate_warm_phase: RocmGraphPhaseStats,
    /// Exact accounting, governor reservation, and settled idle-owner eviction
    /// immediately before native capture.
    pub pre_native_reservation_phase: RocmGraphPhaseStats,
    /// Stream capture, graph instantiation, the settled first native launch,
    /// defensive cache admission, and committed governor publication.
    pub native_capture_phase: RocmGraphPhaseStats,
    /// Settled destruction of successfully captured but unretained candidates.
    pub rejected_candidate_cleanup_phase: RocmGraphPhaseStats,
    /// Exact deduplicated requested bytes in queryable tensor, arena, and
    /// workspace allocations for the most recently measured transient graph
    /// candidate. This excludes the already runner-owned recurrent slot and
    /// opaque stream/event/native-graph objects.
    pub last_transient_candidate_bytes: u64,
    /// High-water mark of `last_transient_candidate_bytes` for this runner.
    pub peak_transient_candidate_bytes: u64,
    /// Closed-reason eager fallback counts and latency.
    pub fallbacks: RocmGraphFallbackStats,
}

/// Runs decode steps through captured HIP graphs when enabled, falling back to
/// eager execution otherwise. ROCm analog of `CudaGraphRunner`.
pub struct RocmGraphRunner {
    policy: RocmGraphExecutionPolicy,
    requested: bool,
    capture_requested: bool,
    enabled: bool,
    counters: RocmGraphCounters,
    phase_telemetry: RocmGraphTelemetryHandle,
    memory_probe_selector: kiln_memory::VramProbeSelector,
    peak_retained_bytes: u64,
    adapter_generation: u64,
    warmup_done: bool,
    #[cfg(feature = "rocm")]
    captured: HashMap<RocmGraphCacheKey, CapturedDecodeGraphRocm>,
    /// Stable recurrent/conv buffers captured by each graph slot. This field is
    /// declared after `captured` so native graphs are destroyed before their
    /// state buffers when the runner is dropped or invalidated.
    #[cfg(feature = "rocm")]
    graph_slots: HashMap<RocmGraphOwner, RocmGraphSlotState>,
    #[cfg(feature = "rocm")]
    decode_row_slots: HashMap<u64, RocmGraphOwner>,
    /// Shared recurrent-state owner for each multi-row graph width.
    #[cfg(feature = "rocm")]
    batched_graph_slots: HashMap<usize, RocmGraphOwner>,
    /// Logical rows denied a new persistent slot by the hard retained-byte cap.
    /// Entries live until `release_decode_row` so decode cannot retry the same
    /// expensive graph setup on every token.
    #[cfg(feature = "rocm")]
    graph_ineligible_rows: HashMap<u64, RocmGraphFallbackReason>,
    #[cfg(feature = "rocm")]
    next_graph_slot_id: u64,
    #[cfg(feature = "rocm")]
    next_access_tick: u64,
    /// Changes whenever graph-cache or persistent-slot ownership changes.
    #[cfg(feature = "rocm")]
    ownership_generation: u64,
    /// Advances only when retained bytes or protected-owner constraints are
    /// relaxed. Aggregate budget suppression keys off this generation; binds
    /// and admissions consume headroom and therefore do not reopen capture.
    #[cfg(feature = "rocm")]
    budget_relief_generation: u64,
    #[cfg(feature = "rocm")]
    budget_rejection_generation: HashMap<RocmGraphKey, u64>,
    #[cfg(feature = "rocm")]
    budget_rejection_generation_wide: Option<u64>,
    /// Measured transient bytes from governor-denied candidates. Matching
    /// geometries skip the warm pass until published headroom can fit one retry.
    #[cfg(feature = "rocm")]
    reservation_denied_bytes: HashMap<RocmGraphKey, u64>,
    #[cfg(feature = "rocm")]
    reservation_denied_wide_bytes: Option<u64>,
    /// Geometries whose decode forward is not capture-safe and the typed eager
    /// fallback reason to reuse on subsequent steps. This includes persistent
    /// host round-trips, attention paths whose tensor shapes depend on the
    /// current sequence length, and candidates that cannot fit the byte budget
    /// even when every other owner is excluded.
    #[cfg(feature = "rocm")]
    non_capture_safe: std::collections::HashMap<RocmGraphKey, RocmGraphFallbackReason>,
    /// Per-geometry count of consecutive capture attempts whose warm pass did a
    /// host round-trip. The first attempt in a `max_seqlen_k` bucket fills the
    /// shape-keyed global caches (broadcast gather indices, gqa-expand indices)
    /// with a one-time host upload; the NEXT attempt in the same (bucket-stable)
    /// geometry finds them warm and captures cleanly. So we don't condemn a
    /// geometry to `non_capture_safe` on the first htod bump — we retry up to
    /// `CAPTURE_RETRY_LIMIT` times and only give up if the round-trip persists
    /// (a genuine per-step fallback, not a cold cache).
    #[cfg(feature = "rocm")]
    capture_retry: std::collections::HashMap<RocmGraphKey, u32>,
    #[cfg(feature = "rocm")]
    cache_full_warned: bool,
    /// Per-owner request-boundary detection. Captured bs=1 graphs carry GDN
    /// recurrent/conv state in their own buffers, evolved in place across that
    /// owner's replays; sharing one timeline across interleaved streaming rows
    /// lets unrelated requests evict or reuse each other's graph state.
    #[cfg(feature = "rocm")]
    decode_timelines: HashMap<RocmGraphOwner, RocmGraphOwnerTimeline>,
}

impl RocmGraphRunner {
    /// Construct a runner for `device` with an already-resolved policy.
    pub fn new(device: &Device, policy: RocmGraphExecutionPolicy) -> Self {
        let is_rocm = matches!(device, Device::Rocm(_));
        let requested = is_rocm && policy.mode() != RocmGraphExecutionMode::Disabled;
        #[cfg(feature = "rocm")]
        let capture_requested =
            requested && policy.mode() == RocmGraphExecutionMode::LazyCaptureReplay;
        #[cfg(not(feature = "rocm"))]
        let capture_requested = false;
        if requested && capture_requested {
            tracing::info!("ROCm HIP graphs enabled for decode");
        } else if requested {
            tracing::info!("ROCm graph runner configured for graph-shaped warmup and eager decode");
        } else if is_rocm {
            tracing::debug!("ROCm graphs disabled; using eager decode");
        }
        Self {
            policy,
            requested,
            capture_requested,
            enabled: requested,
            counters: RocmGraphCounters::default(),
            phase_telemetry: RocmGraphTelemetryHandle::default(),
            memory_probe_selector: device.memory_probe_selector(),
            peak_retained_bytes: 0,
            adapter_generation: 0,
            warmup_done: false,
            #[cfg(feature = "rocm")]
            captured: HashMap::new(),
            #[cfg(feature = "rocm")]
            graph_slots: HashMap::new(),
            #[cfg(feature = "rocm")]
            decode_row_slots: HashMap::new(),
            #[cfg(feature = "rocm")]
            batched_graph_slots: HashMap::new(),
            #[cfg(feature = "rocm")]
            graph_ineligible_rows: HashMap::new(),
            #[cfg(feature = "rocm")]
            next_graph_slot_id: 1,
            #[cfg(feature = "rocm")]
            next_access_tick: 1,
            #[cfg(feature = "rocm")]
            ownership_generation: 1,
            #[cfg(feature = "rocm")]
            budget_relief_generation: 1,
            #[cfg(feature = "rocm")]
            budget_rejection_generation: HashMap::new(),
            #[cfg(feature = "rocm")]
            budget_rejection_generation_wide: None,
            #[cfg(feature = "rocm")]
            reservation_denied_bytes: HashMap::new(),
            #[cfg(feature = "rocm")]
            reservation_denied_wide_bytes: None,
            #[cfg(feature = "rocm")]
            non_capture_safe: std::collections::HashMap::new(),
            #[cfg(feature = "rocm")]
            capture_retry: std::collections::HashMap::new(),
            #[cfg(feature = "rocm")]
            cache_full_warned: false,
            #[cfg(feature = "rocm")]
            decode_timelines: HashMap::new(),
        }
    }

    /// Max consecutive capture attempts (per bucket-stable geometry) whose warm
    /// pass may do a host round-trip before the geometry is condemned to
    /// `non_capture_safe`. A cold shape-cache fill clears in one pass, so the
    /// 2nd attempt normally captures; the budget gives margin for multiple
    /// distinct caches warming across a step or two.
    #[cfg(feature = "rocm")]
    const CAPTURE_RETRY_LIMIT: u32 = 3;
    /// Transient aggregate-budget denial history is independent of the live
    /// graph-entry cap: a long context can visit many exact attention buckets
    /// under one unchanged owner set. At saturation, capture fails closed for
    /// that ownership generation instead of cycling old denials.
    #[cfg(feature = "rocm")]
    const MAX_BUDGET_REJECTION_GEOMETRIES: usize = 256;

    /// Whether captured-graph decode is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Clone the capture telemetry channel, which is independent of the graph
    /// runner lock. Owners should keep it beside, rather than behind, that lock.
    pub fn telemetry_handle(&self) -> RocmGraphTelemetryHandle {
        self.phase_telemetry.clone()
    }

    /// Return configuration, circuit-breaker state, and lifetime execution
    /// counters without resetting them.
    pub fn stats(&self) -> RocmGraphStats {
        #[cfg(feature = "rocm")]
        let captured_graph_count = self.captured.len();
        #[cfg(not(feature = "rocm"))]
        let captured_graph_count = 0;
        #[cfg(feature = "rocm")]
        let tracked_decode_owner_count = self.decode_timelines.len();
        #[cfg(not(feature = "rocm"))]
        let tracked_decode_owner_count = 0;
        #[cfg(feature = "rocm")]
        let graph_slot_count = self.graph_slots.len();
        #[cfg(not(feature = "rocm"))]
        let graph_slot_count = 0;
        #[cfg(feature = "rocm")]
        let active_graph_slot_count = self
            .graph_slots
            .values()
            .filter(|slot| slot.assigned_row.is_some() || slot.batch_size.is_some())
            .count();
        #[cfg(not(feature = "rocm"))]
        let active_graph_slot_count = 0;
        #[cfg(feature = "rocm")]
        let memory = self.memory_accounting(&HashSet::new(), None);
        let phase_telemetry = self.phase_telemetry.snapshot();
        #[cfg(not(feature = "rocm"))]
        let (
            retained_stable_io_bytes,
            retained_capture_arena_bytes,
            retained_blaslt_workspace_bytes,
            retained_slot_state_bytes,
            retained_bytes,
            opaque_native_object_count,
            retained_bytes_accounting_complete,
        ) = (0, 0, 0, 0, 0, 0, true);
        #[cfg(feature = "rocm")]
        let (
            retained_stable_io_bytes,
            retained_capture_arena_bytes,
            retained_blaslt_workspace_bytes,
            retained_slot_state_bytes,
            retained_bytes,
            opaque_native_object_count,
            retained_bytes_accounting_complete,
        ) = (
            memory.stable_io_bytes,
            memory.capture_arena_bytes,
            memory.blaslt_workspace_bytes,
            memory.slot_state_bytes,
            memory.retained_bytes,
            memory.opaque_native_object_count,
            memory.complete,
        );

        RocmGraphStats {
            requested: self.requested,
            capture_requested: self.capture_requested,
            enabled: self.enabled,
            capture_enabled: self.capture_requested && self.enabled,
            max_cached_graphs: self.policy.max_cached_graphs(),
            max_retained_bytes: self.policy.max_retained_bytes(),
            capture_attempts: self.counters.capture_attempts,
            capture_successes: self.counters.capture_successes,
            capture_deferrals: self.counters.capture_deferrals,
            capture_failures: self.counters.capture_failures,
            replay_attempts: self.counters.replay_attempts,
            replay_successes: self.counters.replay_successes,
            replay_failures: self.counters.replay_failures,
            failures: self
                .counters
                .capture_failures
                .saturating_add(self.counters.replay_failures),
            decode_owner_release_count: self.counters.decode_owner_release_count,
            decode_owner_graph_release_count: self.counters.decode_owner_graph_release_count,
            graph_slot_create_count: self.counters.graph_slot_create_count,
            graph_slot_reuse_count: self.counters.graph_slot_reuse_count,
            cache_admission_successes: self.counters.cache_admission_successes,
            cache_evictions: self.counters.cache_evictions,
            cache_evicted_bytes: self.counters.cache_evicted_bytes,
            budget_evictions: self.counters.budget_evictions,
            pressure_evictions: self.counters.pressure_evictions,
            invalidation_evictions: self.counters.invalidation_evictions,
            recovery_evictions: self.counters.recovery_evictions,
            entry_capacity_rejections: self.counters.entry_capacity_rejections,
            byte_budget_rejections: self.counters.byte_budget_rejections,
            accounting_incomplete_rejections: self.counters.accounting_incomplete_rejections,
            pre_capture_entry_capacity_skips: self.counters.pre_capture_entry_capacity_skips,
            pre_capture_byte_budget_skips: self.counters.pre_capture_byte_budget_skips,
            pre_capture_accounting_incomplete_skips: self
                .counters
                .pre_capture_accounting_incomplete_skips,
            pre_capture_memory_reservation_denied_skips: self
                .counters
                .pre_capture_memory_reservation_denied_skips,
            memory_governor_selector_mismatch_skips: self
                .counters
                .memory_governor_selector_mismatch_skips,
            captured_graph_count,
            graph_slot_count,
            active_graph_slot_count,
            idle_graph_slot_count: graph_slot_count.saturating_sub(active_graph_slot_count),
            tracked_decode_owner_count,
            retained_stable_io_bytes,
            retained_capture_arena_bytes,
            retained_blaslt_workspace_bytes,
            retained_slot_state_bytes,
            retained_bytes,
            peak_retained_bytes: self.peak_retained_bytes.max(retained_bytes),
            opaque_native_object_count,
            retained_bytes_accounting_complete,
            quarantined_retained_bytes: self.counters.quarantined_retained_bytes,
            pre_candidate_headroom_phase: phase_telemetry.pre_candidate_headroom_phase,
            candidate_warm_phase: phase_telemetry.candidate_warm_phase,
            pre_native_reservation_phase: phase_telemetry.pre_native_reservation_phase,
            native_capture_phase: phase_telemetry.native_capture_phase,
            rejected_candidate_cleanup_phase: phase_telemetry.rejected_candidate_cleanup_phase,
            last_transient_candidate_bytes: phase_telemetry.last_transient_candidate_bytes,
            peak_transient_candidate_bytes: phase_telemetry.peak_transient_candidate_bytes,
            fallbacks: self.counters.fallbacks,
        }
    }

    /// Invalidate any captured graphs (LoRA swap changes weight pointers) and
    /// force a fresh warmup.
    pub fn invalidate(&mut self) -> Result<()> {
        self.adapter_generation += 1;
        self.warmup_done = false;
        #[cfg(feature = "rocm")]
        {
            let owners: HashSet<_> = self
                .graph_owners()
                .into_iter()
                .chain(self.graph_slots.keys().copied())
                .collect();
            self.evict_graph_owners(
                &owners,
                "adapter_invalidation",
                RocmGraphEvictionReason::Invalidation,
                true,
            )?;
            self.record_budget_relief();
            self.decode_row_slots.clear();
            self.batched_graph_slots.clear();
            self.graph_ineligible_rows.clear();
            self.non_capture_safe.clear();
            self.capture_retry.clear();
            self.reservation_denied_bytes.clear();
            self.reservation_denied_wide_bytes = None;
            self.cache_full_warned = false;
            self.decode_timelines.clear();
        }
        Ok(())
    }

    /// Settle all device work before and after releasing pointer-bearing graph
    /// state. Failed settlement leaves the sticky low-level quarantine in charge
    /// of retaining any uncertain resources until process restart.
    #[cfg(feature = "rocm")]
    fn release_captured_after_device_settlement(&mut self, boundary: &'static str) -> Result<()> {
        let owners = self.graph_owners();
        self.evict_graph_owners(&owners, boundary, RocmGraphEvictionReason::Recovery, false)?;
        Ok(())
    }

    /// Return a finished logical decode row's graph slot to the bounded reuse
    /// pool. Native graphs and the exact recurrent-state buffers they captured
    /// remain resident until explicit runner invalidation; a graphless slot is
    /// discarded immediately.
    pub fn release_decode_row(&mut self, row_id: u64) {
        #[cfg(feature = "rocm")]
        {
            self.graph_ineligible_rows.remove(&row_id);
            let Some(owner) = self.decode_row_slots.remove(&row_id) else {
                return;
            };
            let removed_timeline = self.decode_timelines.remove(&owner).is_some();
            let retained_graphs = self
                .captured
                .keys()
                .filter(|key| key.owner == owner)
                .count();
            if retained_graphs == 0 {
                self.graph_slots.remove(&owner);
            } else if let Some(slot) = self.graph_slots.get_mut(&owner) {
                slot.assigned_row = None;
            }
            self.record_budget_relief();
            self.counters.record_decode_owner_release(0);
            tracing::debug!(
                event = "rocm_graph_decode_owner_released",
                row_id,
                graph_slot_id = owner.slot_id(),
                retained_graphs,
                removed_timeline,
                graph_slot_count = self.graph_slots.len(),
                active_graph_slot_count = self.decode_row_slots.len(),
                "rocm_graph_decode_owner_released"
            );
        }
        #[cfg(not(feature = "rocm"))]
        let _ = row_id;
    }

    #[cfg(feature = "rocm")]
    fn prepare_owner_decode(
        &mut self,
        owner: RocmGraphOwner,
        row_id: u64,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> bool {
        let block0 = block_table.blocks.first().copied();
        let owner_started = !self.decode_timelines.contains_key(&owner);
        if owner_started {
            tracing::debug!(
                event = "rocm_graph_decode_owner_started",
                row_id,
                graph_slot_id = owner.slot_id(),
                seq_len,
                block0 = block0.unwrap_or_default(),
                block0_present = block0.is_some(),
                "rocm_graph_decode_owner_started"
            );
        }
        let timeline = self.decode_timelines.entry(owner).or_default();
        let continues = block0.is_some()
            && timeline.last_decode_seq_len == Some(seq_len.wrapping_sub(1))
            && timeline.last_decode_block0 == block0;
        timeline.last_decode_seq_len = Some(seq_len);
        timeline.last_decode_block0 = block0;
        continues
    }

    #[cfg(feature = "rocm")]
    fn clone_linear_state_handles(state: &LinearAttentionState) -> LinearAttentionState {
        LinearAttentionState {
            recurrent_states: state.recurrent_states.clone(),
            conv_states: state.conv_states.clone(),
        }
    }

    #[cfg(feature = "rocm")]
    fn linear_state_handles_match(
        left: &LinearAttentionState,
        right: &LinearAttentionState,
    ) -> bool {
        left.recurrent_states.len() == right.recurrent_states.len()
            && left.conv_states.len() == right.conv_states.len()
            && left
                .recurrent_states
                .iter()
                .zip(&right.recurrent_states)
                .all(|(left, right)| left.id() == right.id())
            && left
                .conv_states
                .iter()
                .zip(&right.conv_states)
                .all(|(left, right)| left.id() == right.id())
    }

    #[cfg(any(feature = "rocm", test))]
    fn restore_linear_state_in_place(
        state: &mut LinearAttentionState,
        snapshot: &LinearAttentionState,
        context: &'static str,
    ) -> Result<()> {
        anyhow::ensure!(
            state.recurrent_states.len() == snapshot.recurrent_states.len()
                && state.conv_states.len() == snapshot.conv_states.len(),
            "{context}: recurrent/conv layer-count mismatch"
        );
        for (destination, source) in state
            .recurrent_states
            .iter()
            .zip(snapshot.recurrent_states.iter())
        {
            anyhow::ensure!(
                destination.dims() == source.dims(),
                "{context}: recurrent-state shape mismatch"
            );
            destination
                .slice_set(source, 0, 0)
                .with_context(|| context)?;
        }
        for (destination, source) in state.conv_states.iter().zip(snapshot.conv_states.iter()) {
            anyhow::ensure!(
                destination.dims() == source.dims(),
                "{context}: convolution-state shape mismatch"
            );
            destination
                .slice_set(source, 0, 0)
                .with_context(|| context)?;
        }
        Ok(())
    }

    #[cfg(feature = "rocm")]
    fn restore_linear_state_after_execution(
        rocm_context: &kiln_hip::RocmContext,
        state: &mut LinearAttentionState,
        snapshot: &LinearAttentionState,
        operation: &'static str,
    ) -> Result<()> {
        let result = Self::restore_linear_state_in_place(state, snapshot, operation);
        if result.is_err() {
            // A completed HIP wait proves physical quiescence, not validity of
            // recurrent model state. Failed rollback must prevent all future
            // dispatch on this device.
            rocm_context.quarantine_execution();
        }
        result
    }

    #[cfg(feature = "rocm")]
    fn publish_new_graph_slot(
        &mut self,
        row_id: u64,
        owner: RocmGraphOwner,
        linear_state: LinearAttentionState,
        allocations: Vec<RocmAllocationRecord>,
        accounting_complete: bool,
    ) -> std::result::Result<(), RocmGraphFallbackReason> {
        let candidate_slot = RocmGraphCandidateSlot {
            owner,
            allocations: &allocations,
            accounting_complete,
        };
        let projected = self.memory_accounting_with_retained_slots(
            &HashSet::new(),
            None,
            &HashSet::new(),
            Some(candidate_slot),
        );
        if !projected.complete || projected.retained_bytes > self.max_retained_bytes() {
            let (rejection, reason) = if !projected.complete {
                (
                    RocmGraphAdmissionRejection::AccountingIncomplete,
                    RocmGraphFallbackReason::GraphAccountingIncomplete,
                )
            } else {
                (
                    RocmGraphAdmissionRejection::ByteBudget,
                    RocmGraphFallbackReason::GraphCacheByteBudget,
                )
            };
            self.graph_ineligible_rows.insert(row_id, reason);
            self.counters.record_pre_capture_skip(rejection);
            tracing::warn!(
                row_id,
                graph_slot_id = owner.slot_id(),
                projected_retained_bytes = projected.retained_bytes,
                max_retained_bytes = self.max_retained_bytes(),
                accounting_complete = projected.complete,
                "ROCm graph slot rejected by the hard retained-byte budget"
            );
            return Err(reason);
        }

        self.graph_slots.insert(
            owner,
            RocmGraphSlotState {
                assigned_row: None,
                batch_size: None,
                linear_state,
                allocations,
                accounting_complete,
            },
        );
        self.peak_retained_bytes = self.peak_retained_bytes.max(projected.retained_bytes);
        self.counters.record_graph_slot_create();
        Ok(())
    }

    #[cfg(feature = "rocm")]
    fn publish_batched_graph_slot(
        &mut self,
        batch_size: usize,
        owner: RocmGraphOwner,
        linear_state: LinearAttentionState,
        allocations: Vec<RocmAllocationRecord>,
        accounting_complete: bool,
    ) -> std::result::Result<(), RocmGraphFallbackReason> {
        let candidate_slot = RocmGraphCandidateSlot {
            owner,
            allocations: &allocations,
            accounting_complete,
        };
        let projected = self.memory_accounting_with_retained_slots(
            &HashSet::new(),
            None,
            &HashSet::new(),
            Some(candidate_slot),
        );
        if !projected.complete || projected.retained_bytes > self.max_retained_bytes() {
            let (rejection, reason) = if !projected.complete {
                (
                    RocmGraphAdmissionRejection::AccountingIncomplete,
                    RocmGraphFallbackReason::GraphAccountingIncomplete,
                )
            } else {
                (
                    RocmGraphAdmissionRejection::ByteBudget,
                    RocmGraphFallbackReason::GraphCacheByteBudget,
                )
            };
            self.counters.record_pre_capture_skip(rejection);
            tracing::warn!(
                batch_size,
                graph_slot_id = owner.slot_id(),
                projected_retained_bytes = projected.retained_bytes,
                max_retained_bytes = self.max_retained_bytes(),
                accounting_complete = projected.complete,
                "ROCm batched graph slot rejected by the hard retained-byte budget"
            );
            return Err(reason);
        }

        self.graph_slots.insert(
            owner,
            RocmGraphSlotState {
                assigned_row: None,
                batch_size: Some(batch_size),
                linear_state,
                allocations,
                accounting_complete,
            },
        );
        self.batched_graph_slots.insert(batch_size, owner);
        self.peak_retained_bytes = self.peak_retained_bytes.max(projected.retained_bytes);
        self.counters.record_graph_slot_create();
        self.record_ownership_mutation();
        Ok(())
    }

    #[cfg(feature = "rocm")]
    fn bind_batched_state_to_slot(
        &mut self,
        batch_size: usize,
        linear_state: &mut LinearAttentionState,
    ) -> Result<RocmGraphBindOutcome> {
        if let Some(owner) = self.batched_graph_slots.get(&batch_size).copied() {
            let valid = self
                .graph_slots
                .get(&owner)
                .is_some_and(|slot| slot.batch_size == Some(batch_size));
            if !valid {
                self.batched_graph_slots.remove(&batch_size);
            }
        }

        let owner = if let Some(owner) = self.batched_graph_slots.get(&batch_size).copied() {
            owner
        } else {
            let owner = RocmGraphOwner::Slot(self.next_graph_slot_id);
            self.next_graph_slot_id = self.next_graph_slot_id.saturating_add(1);
            let (allocations, accounting_complete) = inspect_rocm_tensor_allocations(
                linear_state
                    .recurrent_states
                    .iter()
                    .chain(linear_state.conv_states.iter()),
            );
            if let Err(reason) = self.publish_batched_graph_slot(
                batch_size,
                owner,
                Self::clone_linear_state_handles(linear_state),
                allocations,
                accounting_complete,
            ) {
                return Ok(RocmGraphBindOutcome::Fallback(reason));
            }
            owner
        };

        let slot = self
            .graph_slots
            .get_mut(&owner)
            .context("ROCm batched graph width points to a missing persistent slot")?;
        anyhow::ensure!(
            slot.assigned_row.is_none() && slot.batch_size == Some(batch_size),
            "ROCm graph slot {} is not reserved for batch width {batch_size}",
            owner.slot_id()
        );
        if !Self::linear_state_handles_match(&slot.linear_state, linear_state) {
            Self::restore_linear_state_in_place(
                &mut slot.linear_state,
                linear_state,
                "refresh shared ROCm batched graph slot state",
            )?;
            linear_state
                .recurrent_states
                .clone_from(&slot.linear_state.recurrent_states);
            linear_state
                .conv_states
                .clone_from(&slot.linear_state.conv_states);
        }
        Ok(RocmGraphBindOutcome::Bound(owner))
    }

    #[cfg(feature = "rocm")]
    fn bind_decode_row_to_slot(
        &mut self,
        row_id: u64,
        requested_key: &RocmGraphKey,
        linear_state: &mut LinearAttentionState,
    ) -> Result<RocmGraphBindOutcome> {
        if let Some(reason) = self.graph_ineligible_rows.get(&row_id).copied() {
            return Ok(RocmGraphBindOutcome::Fallback(reason));
        }
        let existing = self.decode_row_slots.get(&row_id).copied();
        let owner = if let Some(owner) = existing {
            owner
        } else {
            let preferred = self
                .graph_slots
                .iter()
                .filter(|(_, slot)| slot.assigned_row.is_none() && slot.batch_size.is_none())
                .map(|(owner, _)| *owner)
                .filter(|owner| {
                    self.captured
                        .contains_key(&RocmGraphCacheKey::new(*owner, requested_key.clone()))
                })
                .min_by_key(|owner| owner.slot_id());
            let idle = preferred.or_else(|| {
                self.graph_slots
                    .iter()
                    .filter(|(_, slot)| slot.assigned_row.is_none() && slot.batch_size.is_none())
                    .map(|(owner, _)| *owner)
                    .min_by_key(|owner| owner.slot_id())
            });
            let owner = if let Some(owner) = idle {
                self.counters.record_graph_slot_reuse();
                owner
            } else {
                let owner = RocmGraphOwner::Slot(self.next_graph_slot_id);
                self.next_graph_slot_id = self.next_graph_slot_id.saturating_add(1);
                let (allocations, accounting_complete) = inspect_rocm_tensor_allocations(
                    linear_state
                        .recurrent_states
                        .iter()
                        .chain(linear_state.conv_states.iter()),
                );
                if let Err(reason) = self.publish_new_graph_slot(
                    row_id,
                    owner,
                    Self::clone_linear_state_handles(linear_state),
                    allocations,
                    accounting_complete,
                ) {
                    return Ok(RocmGraphBindOutcome::Fallback(reason));
                }
                owner
            };
            self.decode_row_slots.insert(row_id, owner);
            self.decode_timelines.remove(&owner);
            self.graph_slots
                .get_mut(&owner)
                .expect("new or idle ROCm graph slot must exist")
                .assigned_row = Some(row_id);
            self.record_ownership_mutation();
            owner
        };

        let slot = self
            .graph_slots
            .get_mut(&owner)
            .context("ROCm graph row points to a missing persistent slot")?;
        anyhow::ensure!(
            slot.assigned_row == Some(row_id),
            "ROCm graph slot {} is assigned to {:?}, not row {row_id}",
            owner.slot_id(),
            slot.assigned_row
        );
        if !Self::linear_state_handles_match(&slot.linear_state, linear_state) {
            slot.linear_state
                .refresh_batched_state_from_rows_in_place(&[linear_state])
                .context("refresh reusable ROCm graph slot state")?;
            linear_state
                .recurrent_states
                .clone_from(&slot.linear_state.recurrent_states);
            linear_state
                .conv_states
                .clone_from(&slot.linear_state.conv_states);
        }
        Ok(RocmGraphBindOutcome::Bound(owner))
    }

    #[cfg(feature = "rocm")]
    fn max_cached_graphs(&self) -> usize {
        self.policy.max_cached_graphs()
    }

    #[cfg(feature = "rocm")]
    fn max_retained_bytes(&self) -> u64 {
        self.policy.max_retained_bytes()
    }

    #[cfg(feature = "rocm")]
    fn matching_memory_governor(&self) -> bool {
        memory_governor_selector_matches(
            self.memory_probe_selector,
            kiln_memory::MemoryGovernor::global_configuration().selector,
        )
    }

    #[cfg(feature = "rocm")]
    fn next_access_tick(&mut self) -> u64 {
        let tick = self.next_access_tick;
        self.next_access_tick = self.next_access_tick.saturating_add(1);
        tick
    }

    #[cfg(feature = "rocm")]
    fn record_ownership_mutation(&mut self) {
        self.ownership_generation = self.ownership_generation.checked_add(1).unwrap_or(1);
    }

    #[cfg(feature = "rocm")]
    fn record_budget_relief(&mut self) {
        self.record_ownership_mutation();
        self.budget_relief_generation = self.budget_relief_generation.checked_add(1).unwrap_or(1);
        self.budget_rejection_generation.clear();
        self.budget_rejection_generation_wide = None;
    }

    #[cfg(feature = "rocm")]
    fn budget_capture_suppressed(&self, key: &RocmGraphKey) -> bool {
        self.budget_rejection_generation_wide == Some(self.budget_relief_generation)
            || self.budget_rejection_generation.get(key) == Some(&self.budget_relief_generation)
    }

    #[cfg(feature = "rocm")]
    fn remember_reservation_denial(&mut self, key: &RocmGraphKey, required_bytes: u64) {
        if let Some(wide) = self.reservation_denied_wide_bytes.as_mut() {
            *wide = (*wide).max(required_bytes);
            return;
        }
        if let Some(existing) = self.reservation_denied_bytes.get_mut(key) {
            *existing = (*existing).max(required_bytes);
            return;
        }
        if self.reservation_denied_bytes.len() >= Self::MAX_BUDGET_REJECTION_GEOMETRIES {
            let wide = self
                .reservation_denied_bytes
                .values()
                .copied()
                .fold(required_bytes, u64::max);
            self.reservation_denied_bytes.clear();
            self.reservation_denied_wide_bytes = Some(wide);
            return;
        }
        self.reservation_denied_bytes
            .insert(key.clone(), required_bytes);
    }

    #[cfg(feature = "rocm")]
    fn reservation_retry_suppressed_with_available(
        &mut self,
        key: &RocmGraphKey,
        available_bytes: Option<u64>,
    ) -> bool {
        let required_bytes = self
            .reservation_denied_bytes
            .get(key)
            .copied()
            .or(self.reservation_denied_wide_bytes);
        let Some(required_bytes) = required_bytes else {
            return false;
        };
        if available_bytes.is_some_and(|available| available >= required_bytes) {
            self.reservation_denied_bytes.remove(key);
            self.reservation_denied_wide_bytes = None;
            false
        } else {
            true
        }
    }

    #[cfg(feature = "rocm")]
    fn reservation_retry_suppressed(
        &mut self,
        key: &RocmGraphKey,
    ) -> Option<RocmGraphFallbackReason> {
        if !self.reservation_denied_bytes.contains_key(key)
            && self.reservation_denied_wide_bytes.is_none()
        {
            return None;
        }
        if !self.matching_memory_governor() {
            self.counters.memory_governor_selector_mismatch_skips = self
                .counters
                .memory_governor_selector_mismatch_skips
                .saturating_add(1);
            return Some(RocmGraphFallbackReason::MemoryGovernorSelectorMismatch);
        }
        let available = kiln_memory::MemoryGovernor::try_global_cached_available_bytes();
        if self.reservation_retry_suppressed_with_available(key, available) {
            self.counters.pre_capture_memory_reservation_denied_skips = self
                .counters
                .pre_capture_memory_reservation_denied_skips
                .saturating_add(1);
            Some(RocmGraphFallbackReason::MemoryReservationDenied)
        } else {
            None
        }
    }

    #[cfg(feature = "rocm")]
    fn touch_captured_graph(&mut self, key: &RocmGraphCacheKey) {
        let tick = self.next_access_tick();
        if let Some(captured) = self.captured.get_mut(key) {
            captured.last_used_tick = tick;
        }
    }

    /// Derive exact requested physical bytes from live allocation metadata. This
    /// never synchronizes a stream or calls the HIP runtime.
    #[cfg(feature = "rocm")]
    fn memory_accounting(
        &self,
        excluded_owners: &HashSet<RocmGraphOwner>,
        candidate: Option<(&RocmGraphCacheKey, &RocmGraphEntryAccounting)>,
    ) -> RocmGraphMemoryAccounting {
        self.memory_accounting_with_retained_slots(
            excluded_owners,
            candidate,
            &HashSet::new(),
            None,
        )
    }

    #[cfg(feature = "rocm")]
    fn memory_accounting_with_retained_slots(
        &self,
        excluded_owners: &HashSet<RocmGraphOwner>,
        candidate: Option<(&RocmGraphCacheKey, &RocmGraphEntryAccounting)>,
        retained_slot_owners: &HashSet<RocmGraphOwner>,
        candidate_slot: Option<RocmGraphCandidateSlot<'_>>,
    ) -> RocmGraphMemoryAccounting {
        self.memory_accounting_with_exclusions(
            excluded_owners,
            &HashSet::new(),
            candidate,
            retained_slot_owners,
            candidate_slot,
        )
    }

    #[cfg(feature = "rocm")]
    fn memory_accounting_with_exclusions(
        &self,
        excluded_owners: &HashSet<RocmGraphOwner>,
        excluded_keys: &HashSet<RocmGraphCacheKey>,
        candidate: Option<(&RocmGraphCacheKey, &RocmGraphEntryAccounting)>,
        retained_slot_owners: &HashSet<RocmGraphOwner>,
        candidate_slot: Option<RocmGraphCandidateSlot<'_>>,
    ) -> RocmGraphMemoryAccounting {
        const OPAQUE_NATIVE_OBJECTS_PER_GRAPH: usize = 5; // graph, exec, stream, two events

        let included = || {
            self.captured
                .iter()
                .filter(|(key, _)| {
                    !excluded_owners.contains(&key.owner) && !excluded_keys.contains(*key)
                })
                .map(|(key, captured)| (key, &captured.accounting))
                .chain(candidate.into_iter())
        };
        let mut accounting = RocmGraphMemoryAccounting {
            complete: true,
            ..RocmGraphMemoryAccounting::default()
        };
        let mut seen = HashSet::new();

        for (_, entry) in included() {
            for allocation in &entry.stable_io {
                RocmGraphMemoryAccounting::add_record(
                    &mut accounting.stable_io_bytes,
                    &mut seen,
                    *allocation,
                );
            }
        }
        for (_, entry) in included() {
            for allocation in &entry.capture_arena {
                RocmGraphMemoryAccounting::add_record(
                    &mut accounting.capture_arena_bytes,
                    &mut seen,
                    *allocation,
                );
            }
        }
        for (_, entry) in included() {
            if let Some(allocation) = entry.blaslt_workspace {
                RocmGraphMemoryAccounting::add_record(
                    &mut accounting.blaslt_workspace_bytes,
                    &mut seen,
                    allocation,
                );
            }
        }

        let represented_owners: HashSet<_> = self
            .graph_slots
            .keys()
            .filter(|owner| !excluded_owners.contains(*owner))
            .copied()
            .chain(included().map(|(key, _)| key.owner))
            .chain(retained_slot_owners.iter().copied())
            .chain(candidate_slot.map(|slot| slot.owner))
            .collect();
        for owner in represented_owners {
            if let Some(slot) = candidate_slot.filter(|slot| slot.owner == owner) {
                accounting.complete &= slot.accounting_complete;
                for allocation in slot.allocations {
                    RocmGraphMemoryAccounting::add_record(
                        &mut accounting.slot_state_bytes,
                        &mut seen,
                        *allocation,
                    );
                }
                continue;
            }
            let Some(slot) = self.graph_slots.get(&owner) else {
                accounting.complete = false;
                continue;
            };
            accounting.complete &= slot.accounting_complete;
            for allocation in &slot.allocations {
                RocmGraphMemoryAccounting::add_record(
                    &mut accounting.slot_state_bytes,
                    &mut seen,
                    *allocation,
                );
            }
        }

        accounting.opaque_native_object_count = included()
            .count()
            .saturating_mul(OPAQUE_NATIVE_OBJECTS_PER_GRAPH);
        accounting.finish();
        accounting
    }

    #[cfg(feature = "rocm")]
    fn idle_owner_lru(&self, protected_owner: RocmGraphOwner) -> Vec<RocmGraphOwner> {
        let mut owners: Vec<_> = self
            .graph_slots
            .iter()
            .filter(|(owner, slot)| {
                **owner != protected_owner
                    && slot.assigned_row.is_none()
                    && self.captured.keys().any(|key| key.owner == **owner)
            })
            .map(|(owner, _)| {
                let last_used = self
                    .captured
                    .iter()
                    .filter(|(key, _)| key.owner == *owner)
                    .map(|(_, captured)| captured.last_used_tick)
                    .max()
                    .unwrap_or(0);
                (last_used, owner.slot_id(), *owner)
            })
            .collect();
        sort_idle_owner_lru(&mut owners);
        owners.into_iter().map(|(_, _, owner)| owner).collect()
    }

    /// Order active graph entries for fair, narrow admission relief after every
    /// reclaimable idle owner has been exhausted.
    #[cfg(feature = "rocm")]
    fn active_geometry_eviction_order(
        &self,
        candidate_owner: RocmGraphOwner,
    ) -> Vec<RocmGraphCacheKey> {
        let active_owners = self
            .graph_slots
            .iter()
            .filter_map(|(owner, slot)| slot.assigned_row.map(|_| *owner));
        fair_active_geometry_eviction_order(
            candidate_owner,
            active_owners,
            self.captured
                .iter()
                .map(|(key, captured)| (key, captured.last_used_tick)),
        )
    }

    #[cfg(feature = "rocm")]
    fn retained_entry_count_excluding(
        &self,
        excluded_owners: &HashSet<RocmGraphOwner>,
        excluded_keys: &HashSet<RocmGraphCacheKey>,
    ) -> usize {
        self.captured
            .keys()
            .filter(|key| !excluded_owners.contains(&key.owner) && !excluded_keys.contains(*key))
            .count()
    }

    /// Plan all ordinary cache evictions before mutating ownership. Idle owners
    /// are considered first as reclaimable units. If those cannot provide the
    /// required headroom, active-owner entries are considered in fair LRU order
    /// and retired narrowly without discarding live slot state.
    #[cfg(feature = "rocm")]
    fn plan_candidate_admission(
        &self,
        key: &RocmGraphCacheKey,
        candidate: &RocmGraphEntryAccounting,
    ) -> std::result::Result<RocmGraphAdmissionPlan, RocmGraphAdmissionRejection> {
        if self.captured.contains_key(key) {
            return Err(RocmGraphAdmissionRejection::EntryCapacity);
        }

        let all_existing_owners: HashSet<_> = self
            .graph_slots
            .keys()
            .copied()
            .chain(self.captured.keys().map(|key| key.owner))
            .collect();
        let candidate_alone = self.memory_accounting(&all_existing_owners, Some((key, candidate)));
        if !candidate_alone.complete {
            return Err(RocmGraphAdmissionRejection::AccountingIncomplete);
        }
        if candidate_alone.retained_bytes > self.max_retained_bytes() {
            return Err(RocmGraphAdmissionRejection::CandidateByteBudget);
        }

        let mut excluded_owners = HashSet::new();
        let mut excluded_keys = HashSet::new();
        let mut plan = RocmGraphAdmissionPlan::default();
        let mut idle_victims = self.idle_owner_lru(key.owner).into_iter();
        let mut active_victims = self.active_geometry_eviction_order(key.owner).into_iter();
        loop {
            let entry_count = self
                .retained_entry_count_excluding(&excluded_owners, &excluded_keys)
                .saturating_add(1);
            let projected = self.memory_accounting_with_exclusions(
                &excluded_owners,
                &excluded_keys,
                Some((key, candidate)),
                &HashSet::new(),
                None,
            );
            if entry_count <= self.max_cached_graphs()
                && projected.complete
                && projected.retained_bytes <= self.max_retained_bytes()
            {
                return Ok(plan);
            }

            let rejection = if !projected.complete {
                RocmGraphAdmissionRejection::AccountingIncomplete
            } else if entry_count > self.max_cached_graphs() {
                RocmGraphAdmissionRejection::EntryCapacity
            } else {
                RocmGraphAdmissionRejection::ByteBudget
            };
            if let Some(owner) = idle_victims.next() {
                excluded_owners.insert(owner);
                plan.evict_owners.push(owner);
            } else if let Some(key) = active_victims.next() {
                excluded_keys.insert(key.clone());
                plan.evict_keys.push(key);
            } else {
                return Err(rejection);
            }
        }
    }

    #[cfg(feature = "rocm")]
    const fn admission_fallback_reason(
        rejection: RocmGraphAdmissionRejection,
    ) -> RocmGraphFallbackReason {
        match rejection {
            RocmGraphAdmissionRejection::EntryCapacity => {
                RocmGraphFallbackReason::GraphCacheCapacity
            }
            RocmGraphAdmissionRejection::ByteBudget
            | RocmGraphAdmissionRejection::CandidateByteBudget => {
                RocmGraphFallbackReason::GraphCacheByteBudget
            }
            RocmGraphAdmissionRejection::AccountingIncomplete => {
                RocmGraphFallbackReason::GraphAccountingIncomplete
            }
        }
    }

    /// Reserve both cache-entry and exact retained-byte headroom after the warm
    /// Record pass but before native stream capture. Any idle-owner evictions
    /// are fully settled here, so capture never creates a transient N+1 entry or
    /// a candidate that is already known to be unretainable.
    #[cfg(feature = "rocm")]
    fn reserve_capture_candidate(
        &mut self,
        key: &RocmGraphCacheKey,
        accounting: &RocmGraphEntryAccounting,
    ) -> Result<Option<RocmGraphFallbackReason>> {
        let plan = match self.plan_candidate_admission(key, accounting) {
            Ok(plan) => plan,
            Err(rejection) => {
                self.counters.record_pre_capture_skip(rejection);
                self.apply_capture_rejection_suppression(key, rejection);
                return Ok(Some(Self::admission_fallback_reason(rejection)));
            }
        };

        if !plan.evict_owners.is_empty() {
            let owners = plan.evict_owners.into_iter().collect();
            self.evict_graph_owners(
                &owners,
                "pre_capture_budget_reservation",
                RocmGraphEvictionReason::Budget,
                false,
            )?;
        }
        if !plan.evict_keys.is_empty() {
            self.evict_graph_keys(
                plan.evict_keys,
                "pre_capture_active_fair_share",
                RocmGraphEvictionReason::Budget,
            )?;
        }

        let projected = self.memory_accounting(&HashSet::new(), Some((key, accounting)));
        let rejection = if !projected.complete {
            Some(RocmGraphAdmissionRejection::AccountingIncomplete)
        } else if self.captured.len().saturating_add(1) > self.max_cached_graphs() {
            Some(RocmGraphAdmissionRejection::EntryCapacity)
        } else if projected.retained_bytes > self.max_retained_bytes() {
            Some(RocmGraphAdmissionRejection::ByteBudget)
        } else {
            None
        };
        if let Some(rejection) = rejection {
            self.counters.record_pre_capture_skip(rejection);
            self.apply_capture_rejection_suppression(key, rejection);
            return Ok(Some(Self::admission_fallback_reason(rejection)));
        }
        Ok(None)
    }

    /// Reclaim deterministic idle owners before allocating a warm candidate
    /// when either retained limit has no headroom. Candidate working memory is
    /// necessarily allocated by the Record pass before its exact size is
    /// knowable; this phase prevents that transient from also overlapping an
    /// already entry- or byte-saturated cache.
    #[cfg(feature = "rocm")]
    fn reserve_capture_entry_capacity(
        &mut self,
        key: &RocmGraphCacheKey,
    ) -> Result<Option<RocmGraphFallbackReason>> {
        let current = self.memory_accounting(&HashSet::new(), None);
        let entry_saturated = self.captured.len() >= self.max_cached_graphs();
        let byte_saturated = current.retained_bytes >= self.max_retained_bytes();
        if current.complete && !entry_saturated && !byte_saturated {
            return Ok(None);
        }

        let mut projected_entries = self.captured.len();
        let mut plan = RocmGraphAdmissionPlan::default();
        let mut excluded_owners = HashSet::new();
        let mut excluded_keys = HashSet::new();
        let mut idle_victims = self.idle_owner_lru(key.owner).into_iter();
        let mut active_victims = self.active_geometry_eviction_order(key.owner).into_iter();
        let mut projected = current;
        loop {
            if projected_entries < self.max_cached_graphs()
                && projected.complete
                && projected.retained_bytes < self.max_retained_bytes()
            {
                break;
            }
            if let Some(owner) = idle_victims.next() {
                projected_entries = projected_entries.saturating_sub(
                    self.captured
                        .keys()
                        .filter(|captured_key| captured_key.owner == owner)
                        .count(),
                );
                excluded_owners.insert(owner);
                plan.evict_owners.push(owner);
            } else if let Some(victim) = active_victims.next() {
                projected_entries = projected_entries.saturating_sub(1);
                excluded_keys.insert(victim.clone());
                plan.evict_keys.push(victim);
            } else {
                break;
            }
            projected = self.memory_accounting_with_exclusions(
                &excluded_owners,
                &excluded_keys,
                None,
                &HashSet::new(),
                None,
            );
        }
        let rejection = if !projected.complete {
            Some(RocmGraphAdmissionRejection::AccountingIncomplete)
        } else if projected_entries >= self.max_cached_graphs() {
            Some(RocmGraphAdmissionRejection::EntryCapacity)
        } else if projected.retained_bytes >= self.max_retained_bytes() {
            Some(RocmGraphAdmissionRejection::ByteBudget)
        } else {
            None
        };
        if let Some(rejection) = rejection {
            self.counters.record_pre_capture_skip(rejection);
            return Ok(Some(Self::admission_fallback_reason(rejection)));
        }

        if !plan.evict_owners.is_empty() {
            self.evict_graph_owners(
                &plan.evict_owners.into_iter().collect(),
                "pre_capture_entry_reservation",
                RocmGraphEvictionReason::Budget,
                false,
            )?;
        }
        if !plan.evict_keys.is_empty() {
            self.evict_graph_keys(
                plan.evict_keys,
                "pre_capture_active_fair_share",
                RocmGraphEvictionReason::Budget,
            )?;
        }

        let current = self.memory_accounting(&HashSet::new(), None);
        let rejection = if !current.complete {
            Some(RocmGraphAdmissionRejection::AccountingIncomplete)
        } else if self.captured.len() >= self.max_cached_graphs() {
            Some(RocmGraphAdmissionRejection::EntryCapacity)
        } else if current.retained_bytes >= self.max_retained_bytes() {
            Some(RocmGraphAdmissionRejection::ByteBudget)
        } else {
            None
        };
        if let Some(rejection) = rejection {
            self.counters.record_pre_capture_skip(rejection);
            return Ok(Some(Self::admission_fallback_reason(rejection)));
        }
        Ok(None)
    }

    #[cfg(feature = "rocm")]
    fn pre_capture_rejection(
        &mut self,
        key: &RocmGraphCacheKey,
    ) -> Option<RocmGraphFallbackReason> {
        let current = self.memory_accounting(&HashSet::new(), None);
        if self.captured.len() < self.max_cached_graphs()
            && current.complete
            && current.retained_bytes < self.max_retained_bytes()
        {
            return None;
        }
        if !self.idle_owner_lru(key.owner).is_empty()
            || !self.active_geometry_eviction_order(key.owner).is_empty()
        {
            return None;
        }
        let (rejection, reason) = if !current.complete {
            (
                RocmGraphAdmissionRejection::AccountingIncomplete,
                RocmGraphFallbackReason::GraphAccountingIncomplete,
            )
        } else if self.captured.len() >= self.max_cached_graphs() {
            (
                RocmGraphAdmissionRejection::EntryCapacity,
                RocmGraphFallbackReason::GraphCacheCapacity,
            )
        } else {
            (
                RocmGraphAdmissionRejection::ByteBudget,
                RocmGraphFallbackReason::GraphCacheByteBudget,
            )
        };
        self.counters.record_pre_capture_skip(rejection);
        Some(reason)
    }

    #[cfg(feature = "rocm")]
    fn graph_owners(&self) -> HashSet<RocmGraphOwner> {
        self.captured.keys().map(|key| key.owner).collect()
    }

    /// Drop selected graph owners as one settled transaction. Pre-drop
    /// settlement protects pointer lifetimes; the post-drop settlement completes
    /// async frees. Destructor-triggered quarantine is checked before any caller
    /// can dispatch fallback work.
    #[cfg(feature = "rocm")]
    fn evict_graph_owners(
        &mut self,
        owners: &HashSet<RocmGraphOwner>,
        boundary: &'static str,
        reason: RocmGraphEvictionReason,
        remove_selected_slots: bool,
    ) -> Result<RocmGraphMemoryAccounting> {
        self.evict_graph_entries(owners, boundary, reason, remove_selected_slots, None)
    }

    /// Retire selected geometries while retaining their active owners' slots
    /// and every other cached geometry. Fair admission uses this narrow path;
    /// broad pressure, invalidation, and recovery still operate on owners.
    #[cfg(feature = "rocm")]
    fn evict_graph_keys(
        &mut self,
        keys: Vec<RocmGraphCacheKey>,
        boundary: &'static str,
        reason: RocmGraphEvictionReason,
    ) -> Result<RocmGraphMemoryAccounting> {
        anyhow::ensure!(!keys.is_empty(), "{boundary}: no ROCm graph keys selected");
        anyhow::ensure!(
            keys.iter().all(|key| self.captured.contains_key(key)),
            "{boundary}: selected ROCm graph key is no longer retained"
        );
        let owners: HashSet<_> = keys.iter().map(|key| key.owner).collect();
        anyhow::ensure!(
            owners.iter().all(|owner| self
                .graph_slots
                .get(owner)
                .is_some_and(|slot| slot.assigned_row.is_some())),
            "{boundary}: geometry-only ROCm graph eviction requires active owners"
        );
        self.evict_graph_entries(
            &owners,
            boundary,
            reason,
            false,
            Some(keys.into_iter().collect()),
        )
    }

    #[cfg(feature = "rocm")]
    fn evict_graph_entries(
        &mut self,
        owners: &HashSet<RocmGraphOwner>,
        boundary: &'static str,
        reason: RocmGraphEvictionReason,
        remove_selected_slots: bool,
        selected_keys: Option<HashSet<RocmGraphCacheKey>>,
    ) -> Result<RocmGraphMemoryAccounting> {
        let keys: Vec<_> = match selected_keys.as_ref() {
            Some(selected) => selected.iter().cloned().collect(),
            None => self
                .captured
                .keys()
                .filter(|key| owners.contains(&key.owner))
                .cloned()
                .collect(),
        };
        let prospective_selected_slots: HashSet<_> = owners
            .iter()
            .copied()
            .filter(|owner| {
                let idle = self
                    .graph_slots
                    .get(owner)
                    .is_some_and(|slot| slot.assigned_row.is_none());
                remove_selected_slots || idle
            })
            .collect();
        let retained_slot_owners: HashSet<_> = owners
            .difference(&prospective_selected_slots)
            .copied()
            .collect();
        let active_graph_only_owner_count = retained_slot_owners
            .iter()
            .filter(|owner| {
                self.graph_slots
                    .get(owner)
                    .is_some_and(|slot| slot.assigned_row.is_some())
            })
            .count();
        let context = keys
            .first()
            .and_then(|key| self.captured.get(key))
            .map(|captured| captured.context.clone())
            .or_else(|| {
                prospective_selected_slots.iter().find_map(|owner| {
                    self.graph_slots.get(owner).and_then(|slot| {
                        slot.linear_state
                            .recurrent_states
                            .iter()
                            .chain(slot.linear_state.conv_states.iter())
                            .find_map(|tensor| {
                                tensor
                                    .storage()
                                    .as_any()
                                    .downcast_ref::<kiln_tensor::RocmStorage>()
                                    .map(kiln_tensor::RocmStorage::context)
                            })
                    })
                })
            });
        let before = self.memory_accounting(&HashSet::new(), None);
        let after = if let Some(selected) = selected_keys.as_ref() {
            self.memory_accounting_with_exclusions(
                &HashSet::new(),
                selected,
                None,
                &HashSet::new(),
                None,
            )
        } else {
            self.memory_accounting_with_retained_slots(owners, None, &retained_slot_owners, None)
        };
        let released = RocmGraphMemoryAccounting {
            stable_io_bytes: before.stable_io_bytes.saturating_sub(after.stable_io_bytes),
            capture_arena_bytes: before
                .capture_arena_bytes
                .saturating_sub(after.capture_arena_bytes),
            blaslt_workspace_bytes: before
                .blaslt_workspace_bytes
                .saturating_sub(after.blaslt_workspace_bytes),
            slot_state_bytes: before
                .slot_state_bytes
                .saturating_sub(after.slot_state_bytes),
            retained_bytes: before.retained_bytes.saturating_sub(after.retained_bytes),
            opaque_native_object_count: keys.len().saturating_mul(5),
            complete: before.complete && after.complete,
        };

        if let Some(context) = context.as_ref() {
            context
                .synchronize_device_for(kiln_tensor::RocmSyncReason::MemoryReclaim)
                .with_context(|| format!("{boundary}: settle ROCm device before graph eviction"))?;
            anyhow::ensure!(
                !context.cleanup_quarantined(),
                "{boundary}: ROCm execution is quarantined before graph eviction"
            );
        }

        let mut removed_graphs = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(captured) = self.captured.remove(&key) {
                removed_graphs.push(captured);
            }
        }
        let mut removed_slots = Vec::new();
        let selected_slots: Vec<_> = prospective_selected_slots
            .into_iter()
            .filter(|owner| !self.captured.keys().any(|key| key.owner == *owner))
            .collect();
        for owner in &selected_slots {
            if let Some(slot) = self.graph_slots.remove(owner) {
                removed_slots.push(slot);
            }
            self.decode_timelines.remove(owner);
        }
        self.batched_graph_slots
            .retain(|_, owner| !selected_slots.contains(owner));
        if remove_selected_slots {
            self.decode_row_slots
                .retain(|_, owner| !selected_slots.contains(owner));
        }

        let removed_graph_count = removed_graphs.len();
        let removed_slot_count = removed_slots.len();
        drop(removed_graphs);
        drop(removed_slots);
        if removed_graph_count > 0 || removed_slot_count > 0 {
            self.record_budget_relief();
        }

        let Some(context) = context else {
            self.counters.record_cache_eviction(
                removed_graph_count,
                released.retained_bytes,
                reason,
            );
            tracing::info!(
                event = "rocm_graph_cache_eviction",
                boundary,
                ?reason,
                removed_graph_count,
                removed_slot_count,
                active_graph_only_owner_count,
                released_bytes = released.retained_bytes,
                "ROCm graph cache eviction completed"
            );
            return Ok(released);
        };
        if context.cleanup_quarantined() {
            self.counters.quarantined_retained_bytes = self
                .counters
                .quarantined_retained_bytes
                .saturating_add(released.retained_bytes);
            anyhow::bail!(
                "{boundary}: ROCm graph destructor quarantined execution; restart the process"
            );
        }
        if let Err(error) =
            context.synchronize_device_for(kiln_tensor::RocmSyncReason::MemoryReclaim)
        {
            self.counters.quarantined_retained_bytes = self
                .counters
                .quarantined_retained_bytes
                .saturating_add(released.retained_bytes);
            return Err(anyhow::anyhow!(error))
                .with_context(|| format!("{boundary}: settle ROCm device after graph eviction"));
        }
        if context.cleanup_quarantined() {
            self.counters.quarantined_retained_bytes = self
                .counters
                .quarantined_retained_bytes
                .saturating_add(released.retained_bytes);
            anyhow::bail!(
                "{boundary}: ROCm graph post-drop settlement quarantined execution; restart the process"
            );
        }

        self.counters
            .record_cache_eviction(removed_graph_count, released.retained_bytes, reason);
        tracing::info!(
            event = "rocm_graph_cache_eviction",
            boundary,
            ?reason,
            removed_graph_count,
            removed_slot_count,
            active_graph_only_owner_count,
            released_bytes = released.retained_bytes,
            "ROCm graph cache eviction completed"
        );
        self.cache_full_warned = false;
        Ok(released)
    }

    #[cfg(feature = "rocm")]
    fn release_uncached_candidate(
        &mut self,
        candidate: CapturedDecodeGraphRocm,
        boundary: &'static str,
    ) -> Result<()> {
        let _cleanup_timer = self
            .phase_telemetry
            .timer(RocmGraphPhase::RejectedCandidateCleanup);
        let candidate = std::mem::ManuallyDrop::new(candidate);
        let context = candidate.context.clone();
        let mut seen = HashSet::new();
        let mut retained_bytes = 0u64;
        for allocation in candidate
            .accounting
            .stable_io
            .iter()
            .chain(candidate.accounting.capture_arena.iter())
            .copied()
            .chain(candidate.accounting.blaslt_workspace)
        {
            RocmGraphMemoryAccounting::add_record(&mut retained_bytes, &mut seen, allocation);
        }
        if let Err(error) =
            context.synchronize_device_for(kiln_tensor::RocmSyncReason::MemoryReclaim)
        {
            self.counters.quarantined_retained_bytes = self
                .counters
                .quarantined_retained_bytes
                .saturating_add(retained_bytes);
            return Err(anyhow::anyhow!(error))
                .with_context(|| format!("{boundary}: settle uncached ROCm graph before drop"));
        }
        if context.cleanup_quarantined() {
            self.counters.quarantined_retained_bytes = self
                .counters
                .quarantined_retained_bytes
                .saturating_add(retained_bytes);
            anyhow::bail!("{boundary}: ROCm execution is quarantined before candidate drop");
        }
        let candidate = std::mem::ManuallyDrop::into_inner(candidate);
        drop(candidate);
        if context.cleanup_quarantined() {
            self.counters.quarantined_retained_bytes = self
                .counters
                .quarantined_retained_bytes
                .saturating_add(retained_bytes);
            anyhow::bail!("{boundary}: uncached ROCm graph drop quarantined execution");
        }
        if let Err(error) =
            context.synchronize_device_for(kiln_tensor::RocmSyncReason::MemoryReclaim)
        {
            self.counters.quarantined_retained_bytes = self
                .counters
                .quarantined_retained_bytes
                .saturating_add(retained_bytes);
            return Err(anyhow::anyhow!(error))
                .with_context(|| format!("{boundary}: settle uncached ROCm graph frees"));
        }
        if context.cleanup_quarantined() {
            self.counters.quarantined_retained_bytes = self
                .counters
                .quarantined_retained_bytes
                .saturating_add(retained_bytes);
            anyhow::bail!("{boundary}: uncached ROCm graph cleanup quarantined execution");
        }
        Ok(())
    }

    #[cfg(feature = "rocm")]
    fn record_post_capture_rejection(
        &mut self,
        key: &RocmGraphCacheKey,
        rejection: RocmGraphAdmissionRejection,
    ) -> bool {
        self.counters.record_cache_rejection(rejection);
        self.apply_capture_rejection_suppression(key, rejection)
    }

    #[cfg(feature = "rocm")]
    fn apply_capture_rejection_suppression(
        &mut self,
        key: &RocmGraphCacheKey,
        rejection: RocmGraphAdmissionRejection,
    ) -> bool {
        let future_captures_suppressed = match rejection {
            RocmGraphAdmissionRejection::CandidateByteBudget => {
                self.non_capture_safe.insert(
                    key.graph.clone(),
                    RocmGraphFallbackReason::GraphCacheByteBudget,
                );
                true
            }
            RocmGraphAdmissionRejection::ByteBudget => {
                if self.budget_rejection_generation_wide == Some(self.budget_relief_generation) {
                    return true;
                }
                if !self.budget_rejection_generation.contains_key(&key.graph)
                    && self.budget_rejection_generation.len()
                        >= Self::MAX_BUDGET_REJECTION_GEOMETRIES
                {
                    self.budget_rejection_generation.clear();
                    self.budget_rejection_generation_wide = Some(self.budget_relief_generation);
                    return true;
                }
                self.budget_rejection_generation
                    .insert(key.graph.clone(), self.budget_relief_generation);
                true
            }
            RocmGraphAdmissionRejection::AccountingIncomplete => {
                self.non_capture_safe.insert(
                    key.graph.clone(),
                    RocmGraphFallbackReason::GraphAccountingIncomplete,
                );
                true
            }
            RocmGraphAdmissionRejection::EntryCapacity => false,
        };
        future_captures_suppressed
    }

    #[cfg(feature = "rocm")]
    fn admit_captured_graph(
        &mut self,
        key: RocmGraphCacheKey,
        mut candidate: CapturedDecodeGraphRocm,
        native_capture_timer: &mut Option<RocmGraphPhaseTimer>,
    ) -> Result<bool> {
        let plan = match self.plan_candidate_admission(&key, &candidate.accounting) {
            Ok(plan) => plan,
            Err(rejection) => {
                drop(native_capture_timer.take());
                self.release_uncached_candidate(candidate, "cache_admission_rejection")?;
                let future_captures_suppressed =
                    self.record_post_capture_rejection(&key, rejection);
                tracing::warn!(
                    rejection = ?rejection,
                    candidate_owner = key.owner.slot_id(),
                    future_captures_suppressed,
                    "ROCm graph captured successfully but was not retained by the bounded cache"
                );
                return Ok(false);
            }
        };

        if !plan.evict_owners.is_empty() {
            let owners: HashSet<_> = plan.evict_owners.into_iter().collect();
            if let Err(eviction_error) = self.evict_graph_owners(
                &owners,
                "cache_budget_eviction",
                RocmGraphEvictionReason::Budget,
                false,
            ) {
                drop(native_capture_timer.take());
                return match self.release_uncached_candidate(
                    candidate,
                    "cache_budget_eviction_candidate_cleanup",
                ) {
                    Ok(()) => Err(eviction_error),
                    Err(cleanup_error) => Err(cleanup_error).with_context(|| {
                        format!(
                            "cache budget eviction failed before candidate cleanup: {eviction_error:#}"
                        )
                    }),
                };
            }
        }
        if !plan.evict_keys.is_empty()
            && let Err(eviction_error) = self.evict_graph_keys(
                plan.evict_keys,
                "cache_active_fair_share",
                RocmGraphEvictionReason::Budget,
            )
        {
            drop(native_capture_timer.take());
            return match self.release_uncached_candidate(
                candidate,
                "cache_active_fair_share_candidate_cleanup",
            ) {
                Ok(()) => Err(eviction_error),
                Err(cleanup_error) => Err(cleanup_error).with_context(|| {
                    format!(
                        "active fair-share eviction failed before candidate cleanup: {eviction_error:#}"
                    )
                }),
            };
        }
        let projected =
            self.memory_accounting(&HashSet::new(), Some((&key, &candidate.accounting)));
        if !projected.complete
            || self.captured.len().saturating_add(1) > self.max_cached_graphs()
            || projected.retained_bytes > self.max_retained_bytes()
        {
            let rejection = if !projected.complete {
                RocmGraphAdmissionRejection::AccountingIncomplete
            } else if self.captured.len().saturating_add(1) > self.max_cached_graphs() {
                RocmGraphAdmissionRejection::EntryCapacity
            } else {
                RocmGraphAdmissionRejection::ByteBudget
            };
            drop(native_capture_timer.take());
            self.release_uncached_candidate(candidate, "cache_admission_recheck")?;
            self.record_post_capture_rejection(&key, rejection);
            tracing::warn!(
                rejection = ?rejection,
                candidate_owner = key.owner.slot_id(),
                "ROCm graph admission recheck rejected a captured candidate"
            );
            return Ok(false);
        }
        candidate.last_used_tick = self.next_access_tick();
        if self.captured.contains_key(&key) {
            drop(native_capture_timer.take());
            self.release_uncached_candidate(candidate, "cache_admission_key_collision")?;
            self.counters
                .record_cache_rejection(RocmGraphAdmissionRejection::EntryCapacity);
            return Ok(false);
        }
        self.captured.insert(key, candidate);
        self.record_ownership_mutation();
        self.counters.record_cache_admission();
        let retained = self.memory_accounting(&HashSet::new(), None).retained_bytes;
        self.peak_retained_bytes = self.peak_retained_bytes.max(retained);
        Ok(true)
    }

    #[cfg(feature = "rocm")]
    fn reconcile_memory_pressure(
        &mut self,
        protected_owner: RocmGraphOwner,
    ) -> Result<RocmGraphPressureDecision> {
        if !self.matching_memory_governor() {
            self.counters.memory_governor_selector_mismatch_skips = self
                .counters
                .memory_governor_selector_mismatch_skips
                .saturating_add(1);
            return Ok(RocmGraphPressureDecision::EagerOnly(
                RocmGraphFallbackReason::MemoryGovernorSelectorMismatch,
            ));
        }
        let pressure = kiln_memory::MemoryGovernor::try_global_cached_pressure()
            .unwrap_or(kiln_memory::MemoryPressure::Critical);
        if let Some(decision) = non_evicting_pressure_decision(pressure) {
            return Ok(decision);
        }
        match pressure {
            kiln_memory::MemoryPressure::Tight => {
                let owners: HashSet<_> = self.idle_owner_lru(protected_owner).into_iter().collect();
                if !owners.is_empty() {
                    self.evict_graph_owners(
                        &owners,
                        "tight_memory_pressure",
                        RocmGraphEvictionReason::Pressure,
                        true,
                    )?;
                }
                Ok(RocmGraphPressureDecision::ReplayOnly(
                    RocmGraphFallbackReason::TightMemoryPressure,
                ))
            }
            kiln_memory::MemoryPressure::Critical => {
                let owners = self.graph_owners();
                if !owners.is_empty() {
                    self.evict_graph_owners(
                        &owners,
                        "critical_memory_pressure",
                        RocmGraphEvictionReason::Pressure,
                        false,
                    )?;
                }
                Ok(RocmGraphPressureDecision::EagerOnly(
                    RocmGraphFallbackReason::CriticalMemoryPressure,
                ))
            }
            kiln_memory::MemoryPressure::Comfortable | kiln_memory::MemoryPressure::Moderate => {
                unreachable!("non-evicting pressure decisions returned above")
            }
        }
    }

    #[cfg(feature = "rocm")]
    fn run_eager_fallback<T>(
        &mut self,
        reason: RocmGraphFallbackReason,
        seq_len: usize,
        attempt_duration: std::time::Duration,
        eager: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let eager_started = std::time::Instant::now();
        let result = eager();
        let eager_duration = eager_started.elapsed();
        let duration = attempt_duration.saturating_add(eager_duration);
        let occurrence = self.counters.record_fallback(reason, duration);
        let attempt_duration_ms = attempt_duration.as_secs_f64() * 1000.0;
        let eager_duration_ms = eager_duration.as_secs_f64() * 1000.0;
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let slow = duration >= RocmGraphFallbackStats::SLOW_DURATION;
        match &result {
            Err(error) => tracing::warn!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_failed",
                %error,
                occurrence,
                seq_len,
                attempt_duration_ms,
                eager_duration_ms,
                duration_ms,
                slow,
                "ROCm graph eager fallback failed"
            ),
            Ok(_) if slow => tracing::warn!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_completed",
                occurrence,
                seq_len,
                attempt_duration_ms,
                eager_duration_ms,
                duration_ms,
                slow,
                "slow ROCm graph eager fallback completed"
            ),
            Ok(_) if occurrence == 1 => tracing::info!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_completed",
                occurrence,
                seq_len,
                attempt_duration_ms,
                eager_duration_ms,
                duration_ms,
                slow,
                "ROCm graph eager fallback activated"
            ),
            Ok(_) => tracing::debug!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_completed",
                occurrence,
                seq_len,
                attempt_duration_ms,
                eager_duration_ms,
                duration_ms,
                slow,
                "ROCm graph eager fallback completed"
            ),
        }
        result
    }

    /// Preserve the legacy fallback counter contract for historical receipt
    /// and accounting tests. Supported production multi-row routes now enter
    /// `decode_step_paged_batched_hidden` instead of calling this helper.
    #[cfg(test)]
    fn record_multi_row_eager_fallback(
        &mut self,
        batch_rows: usize,
        duration: std::time::Duration,
    ) {
        if !self.capture_requested || batch_rows <= 1 {
            return;
        }
        let reason = RocmGraphFallbackReason::MultiRowBatchUnsupported;
        let occurrence = self.counters.record_fallback(reason, duration);
        let duration_ms = duration.as_secs_f64() * 1000.0;
        let slow = duration >= RocmGraphFallbackStats::SLOW_DURATION;
        if occurrence == 1 {
            tracing::warn!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_completed",
                occurrence,
                batch_rows,
                duration_ms,
                slow,
                "ROCm graph capture is single-row; accounting multi-row eager decode"
            );
        } else {
            tracing::debug!(
                event = "rocm_graph_fallback",
                reason = reason.as_str(),
                outcome = "eager_completed",
                occurrence,
                batch_rows,
                duration_ms,
                slow,
                "ROCm multi-row eager graph fallback completed"
            );
        }
    }

    /// Run a true multi-row ROCm decode through a width/attention-geometry
    /// keyed HIP graph. `Ok(None)` means graph execution was not requested for
    /// this runner; once requested, every typed deferral is executed and
    /// accounted here so callers cannot accidentally double-advance state.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_batched_hidden(
        &mut self,
        backend: &dyn BackendRuntime,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        sequence_lengths: &[usize],
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
        row_ids: Option<&[u64]>,
    ) -> Result<Option<Tensor>> {
        let batch_size = token_ids.len();
        if !self.enabled
            || !self.capture_requested
            || self.policy.force_eager_decode()
            || batch_size <= 1
        {
            return Ok(None);
        }
        anyhow::ensure!(
            block_tables.len() == batch_size && sequence_lengths.len() == batch_size,
            "ROCm batched graph decode metadata row-count mismatch"
        );
        if let Some(row_ids) = row_ids {
            anyhow::ensure!(
                row_ids.len() == batch_size,
                "ROCm batched graph decode row-id count mismatch"
            );
        }

        #[cfg(feature = "rocm")]
        {
            let max_seq_len = sequence_lengths.iter().copied().max().unwrap_or(0);
            let eager = |linear_state: &mut LinearAttentionState| {
                model_forward_paged_decode_contiguous_batch_hidden_with_ids(
                    backend,
                    token_ids,
                    weights,
                    config,
                    paged_cache,
                    block_tables,
                    sequence_lengths,
                    Some(linear_state),
                    lora,
                    row_ids,
                )
                .context("ROCm batched eager hidden fallback")
            };
            Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
            let requested_key = RocmGraphKey::new_batched(paged_cache, sequence_lengths)?;
            let owner = match self.bind_batched_state_to_slot(batch_size, linear_state)? {
                RocmGraphBindOutcome::Bound(owner) => owner,
                RocmGraphBindOutcome::Fallback(reason) => {
                    return self
                        .run_eager_fallback(reason, max_seq_len, std::time::Duration::ZERO, || {
                            eager(linear_state)
                        })
                        .map(Some);
                }
            };
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());
            let pressure_decision = self.reconcile_memory_pressure(owner)?;
            if let RocmGraphPressureDecision::EagerOnly(reason) = pressure_decision {
                return self
                    .run_eager_fallback(reason, max_seq_len, std::time::Duration::ZERO, || {
                        eager(linear_state)
                    })
                    .map(Some);
            }
            if let Some(reason) = self.non_capture_safe.get(&requested_key).copied() {
                return self
                    .run_eager_fallback(reason, max_seq_len, std::time::Duration::ZERO, || {
                        eager(linear_state)
                    })
                    .map(Some);
            }

            if let Some(adapter_gen) = self
                .captured
                .get(&cache_key)
                .map(|captured| captured.adapter_gen)
            {
                if adapter_gen == self.adapter_generation {
                    let replay_started = std::time::Instant::now();
                    match self.replay_hidden_batched(
                        &cache_key,
                        token_ids,
                        weights,
                        paged_cache,
                        block_tables,
                        sequence_lengths,
                    ) {
                        Ok(hidden) => {
                            self.touch_captured_graph(&cache_key);
                            return Ok(Some(hidden));
                        }
                        Err(error) => {
                            tracing::warn!(
                                batch_size,
                                error = %format!("{error:#}"),
                                "ROCm batched graph replay failed; disabling graphs for this runner"
                            );
                            self.enabled = false;
                            self.release_captured_after_device_settlement(
                                "batched_replay_failure",
                            )
                            .with_context(|| {
                                format!(
                                    "ROCm batched graph replay failed ({error:#}) and containment failed"
                                )
                            })?;
                            return self
                                .run_eager_fallback(
                                    RocmGraphFallbackReason::ReplayFailure,
                                    max_seq_len,
                                    replay_started.elapsed(),
                                    || eager(linear_state),
                                )
                                .map(Some);
                        }
                    }
                } else {
                    self.release_captured_after_device_settlement(
                        "batched_adapter_generation_mismatch",
                    )?;
                }
            }

            if let RocmGraphPressureDecision::ReplayOnly(reason) = pressure_decision {
                return self
                    .run_eager_fallback(reason, max_seq_len, std::time::Duration::ZERO, || {
                        eager(linear_state)
                    })
                    .map(Some);
            }
            if self.budget_capture_suppressed(&requested_key) {
                return self
                    .run_eager_fallback(
                        RocmGraphFallbackReason::GraphCacheByteBudget,
                        max_seq_len,
                        std::time::Duration::ZERO,
                        || eager(linear_state),
                    )
                    .map(Some);
            }
            if let Some(reason) = self.pre_capture_rejection(&cache_key) {
                return self
                    .run_eager_fallback(reason, max_seq_len, std::time::Duration::ZERO, || {
                        eager(linear_state)
                    })
                    .map(Some);
            }

            let capture_started = std::time::Instant::now();
            match self.try_capture_batched_hidden(
                backend,
                owner,
                token_ids,
                weights,
                config,
                paged_cache,
                block_tables,
                sequence_lengths,
                linear_state,
                lora,
            ) {
                Ok(RocmCaptureStep::CapturedHidden(hidden))
                | Ok(RocmCaptureStep::CapturedHiddenUncached(hidden)) => Ok(Some(hidden)),
                Ok(RocmCaptureStep::FallbackEager { reason, .. }) => self
                    .run_eager_fallback(reason, max_seq_len, capture_started.elapsed(), || {
                        eager(linear_state)
                    })
                    .map(Some),
                Err(error) => {
                    tracing::warn!(
                        batch_size,
                        error = %format!("{error:#}"),
                        "ROCm batched graph capture failed; disabling graphs and containing device work"
                    );
                    self.enabled = false;
                    synchronize_after_rocm_graph_capture_failure(&weights.embed_tokens.device())
                        .with_context(|| {
                            format!("batched capture failed before recovery: {error:#}")
                        })?;
                    self.run_eager_fallback(
                        RocmGraphFallbackReason::CaptureFailure,
                        max_seq_len,
                        capture_started.elapsed(),
                        || eager(linear_state),
                    )
                    .map(Some)
                }
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = (
                backend,
                weights,
                config,
                paged_cache,
                block_tables,
                sequence_lengths,
                linear_state,
                lora,
                row_ids,
            );
            Ok(None)
        }
    }

    /// Run one bs=1 paged decode step, returning kt logits `[1, 1, vocab]`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
        graph_row_id: u64,
    ) -> Result<Tensor> {
        if !self.enabled || self.policy.force_eager_decode() {
            return Self::eager_forward(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            );
        }

        #[cfg(feature = "rocm")]
        {
            // NOTE: FP8 paged KV caches ARE capturable on ROCm (unlike CUDA): the
            // FP8 graph-slot write consumes the slot ON-DEVICE and scatters the
            // quantized U8 rows via `rocm_index_copy_dim0` (device_ptr_raw, never
            // `slice()`), so it records into the captured graph and is safe on the
            // Borrowed freeze-pointer arena buffers. See
            // `PagedKvCacheKt::write_token_major_native_graph_slot`. Any residual
            // host round-trip is still caught by the warm-pass htod check below
            // (graceful eager), so no explicit FP8 guard is needed here.

            // Warmup: first decode step runs eagerly (graph-shaped position
            // buffer) to prime the allocator pools before the first capture.
            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: graph-shaped warmup decode step");
                match Self::eager_forward_with_position_buffer(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                ) {
                    Ok(logits) => return Ok(logits),
                    Err(e) => {
                        self.enabled = false;
                        tracing::error!(
                            "ROCm graph-shaped warmup failed: {e:#}; quarantining instead of retrying partially advanced state"
                        );
                        return fail_closed_after_rocm_warmup(weights, e);
                    }
                }
            }

            // Warmup-only policy keeps the graph-shaped first step but uses
            // eager steady-state decode.
            if !self.capture_requested {
                return Self::eager_forward(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                );
            }

            Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
            let requested_key = RocmGraphKey::new(paged_cache, seq_len);
            let owner =
                match self.bind_decode_row_to_slot(graph_row_id, &requested_key, linear_state)? {
                    RocmGraphBindOutcome::Bound(owner) => owner,
                    RocmGraphBindOutcome::Fallback(reason) => {
                        return self.run_eager_fallback(
                            reason,
                            seq_len,
                            std::time::Duration::ZERO,
                            || {
                                Self::eager_forward(
                                    backend,
                                    token_id,
                                    weights,
                                    config,
                                    paged_cache,
                                    block_table,
                                    seq_len,
                                    linear_state,
                                    lora,
                                )
                            },
                        );
                    }
                };
            self.prepare_owner_decode(owner, graph_row_id, block_table, seq_len);
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());
            let pressure_decision = self.reconcile_memory_pressure(owner)?;
            if let RocmGraphPressureDecision::EagerOnly(reason) = pressure_decision {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            // Geometry previously found non-capture-safe: skip the warm pass +
            // capture attempt and reuse its typed eager-fallback reason.
            if let Some(reason) = self.non_capture_safe.get(&requested_key).copied() {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            // Replay if we have a valid captured graph for this geometry.
            if let Some(adapter_gen) = self
                .captured
                .get(&cache_key)
                .map(|captured| captured.adapter_gen)
            {
                if adapter_gen == self.adapter_generation {
                    let replay_started = std::time::Instant::now();
                    match self.replay(
                        &cache_key,
                        token_id,
                        backend,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                    ) {
                        Ok(logits) => {
                            self.touch_captured_graph(&cache_key);
                            tracing::trace!(seq_len, "ROCm graph: replayed captured decode graph");
                            return Ok(logits);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "ROCm graph replay failed: {e:#}; disabling ROCm HIP graphs for this runner; if execution was attempted, the device is quarantined and restart is required"
                            );
                            self.enabled = false;
                            self.release_captured_after_device_settlement("replay_failure")
                                .with_context(|| {
                                    format!(
                                        "ROCm graph replay failed ({e:#}) and containment failed; the device is quarantined and restart is required"
                                    )
                                })?;
                            return self.run_eager_fallback(
                                RocmGraphFallbackReason::ReplayFailure,
                                seq_len,
                                replay_started.elapsed(),
                                || {
                                    Self::eager_forward(
                                        backend,
                                        token_id,
                                        weights,
                                        config,
                                        paged_cache,
                                        block_table,
                                        seq_len,
                                        linear_state,
                                        lora,
                                    )
                                },
                            );
                        }
                    }
                } else {
                    self.release_captured_after_device_settlement("adapter_generation_mismatch")
                        .context(
                            "ROCm graph adapter-generation recovery failed; cleanup is quarantined",
                        )?;
                }
            }

            if let RocmGraphPressureDecision::ReplayOnly(reason) = pressure_decision {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            if self.budget_capture_suppressed(&requested_key) {
                return self.run_eager_fallback(
                    RocmGraphFallbackReason::GraphCacheByteBudget,
                    seq_len,
                    std::time::Duration::ZERO,
                    || {
                        Self::eager_forward(
                            backend,
                            token_id,
                            weights,
                            config,
                            paged_cache,
                            block_table,
                            seq_len,
                            linear_state,
                            lora,
                        )
                    },
                );
            }

            if let Some(reason) = self.pre_capture_rejection(&cache_key) {
                if !self.cache_full_warned {
                    self.cache_full_warned = true;
                    let memory = self.memory_accounting(&HashSet::new(), None);
                    tracing::warn!(
                        cached = self.captured.len(),
                        retained_bytes = memory.retained_bytes,
                        max_cached_graphs = self.max_cached_graphs(),
                        max_retained_bytes = self.max_retained_bytes(),
                        reason = reason.as_str(),
                        "ROCm graph capture skipped: bounded cache has no settled eviction path"
                    );
                }
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            // Capture.
            let capture_started = std::time::Instant::now();
            match self.try_capture(
                backend,
                owner,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            ) {
                Ok(logits) => return Ok(logits),
                Err(e) => {
                    tracing::warn!(
                        "ROCm graph capture failed: {e:#}; disabling HIP graphs and attempting \
                         containment (a quarantined device requires process restart)"
                    );
                    self.enabled = false;
                    // A failed capture can leave pending work. Eager fallback
                    // is permitted only after recovery proves device quiescence.
                    synchronize_after_rocm_graph_capture_failure(&weights.embed_tokens.device())
                        .with_context(|| format!("capture failed before recovery: {e:#}"))?;
                    return self.run_eager_fallback(
                        RocmGraphFallbackReason::CaptureFailure,
                        seq_len,
                        capture_started.elapsed(),
                        || {
                            Self::eager_forward(
                                backend,
                                token_id,
                                weights,
                                config,
                                paged_cache,
                                block_table,
                                seq_len,
                                linear_state,
                                lora,
                            )
                        },
                    );
                }
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = graph_row_id;
            let _ = linear_state;
            Self::eager_forward(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            )
        }
    }

    /// Run one bs=1 paged decode step, returning the pre-final-norm hidden
    /// `[1, 1, hidden]` so callers can keep sampling outside the graph.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_hidden(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
        graph_row_id: u64,
    ) -> Result<Tensor> {
        if !self.enabled || self.policy.force_eager_decode() {
            return Self::eager_forward_hidden(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            );
        }

        #[cfg(feature = "rocm")]
        {
            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: graph-shaped warmup decode step");
                match Self::eager_forward_hidden_with_position_buffer(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                ) {
                    Ok(hidden) => return Ok(hidden),
                    Err(e) => {
                        self.enabled = false;
                        tracing::error!(
                            "ROCm graph-shaped warmup failed: {e:#}; quarantining instead of retrying partially advanced state"
                        );
                        return fail_closed_after_rocm_warmup(weights, e);
                    }
                }
            }

            if !self.capture_requested {
                return Self::eager_forward_hidden(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                );
            }

            Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
            let requested_key = RocmGraphKey::new(paged_cache, seq_len);
            let owner =
                match self.bind_decode_row_to_slot(graph_row_id, &requested_key, linear_state)? {
                    RocmGraphBindOutcome::Bound(owner) => owner,
                    RocmGraphBindOutcome::Fallback(reason) => {
                        return self.run_eager_fallback(
                            reason,
                            seq_len,
                            std::time::Duration::ZERO,
                            || {
                                Self::eager_forward_hidden(
                                    backend,
                                    token_id,
                                    weights,
                                    config,
                                    paged_cache,
                                    block_table,
                                    seq_len,
                                    linear_state,
                                    lora,
                                )
                            },
                        );
                    }
                };
            self.prepare_owner_decode(owner, graph_row_id, block_table, seq_len);
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());
            let pressure_decision = self.reconcile_memory_pressure(owner)?;
            if let RocmGraphPressureDecision::EagerOnly(reason) = pressure_decision {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_hidden(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            if let Some(reason) = self.non_capture_safe.get(&requested_key).copied() {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_hidden(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            if let Some(adapter_gen) = self
                .captured
                .get(&cache_key)
                .map(|captured| captured.adapter_gen)
            {
                if adapter_gen == self.adapter_generation {
                    let replay_started = std::time::Instant::now();
                    match self.replay_hidden(
                        &cache_key,
                        token_id,
                        weights,
                        paged_cache,
                        block_table,
                        seq_len,
                    ) {
                        Ok(hidden) => {
                            self.touch_captured_graph(&cache_key);
                            tracing::trace!(seq_len, "ROCm graph: replayed captured decode graph");
                            return Ok(hidden);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "ROCm graph replay failed: {e:#}; disabling ROCm HIP graphs for this runner; if execution was attempted, the device is quarantined and restart is required"
                            );
                            self.enabled = false;
                            self.release_captured_after_device_settlement("replay_failure")
                                .with_context(|| {
                                    format!(
                                        "ROCm graph replay failed ({e:#}) and containment failed; the device is quarantined and restart is required"
                                    )
                                })?;
                            return self.run_eager_fallback(
                                RocmGraphFallbackReason::ReplayFailure,
                                seq_len,
                                replay_started.elapsed(),
                                || {
                                    Self::eager_forward_hidden(
                                        backend,
                                        token_id,
                                        weights,
                                        config,
                                        paged_cache,
                                        block_table,
                                        seq_len,
                                        linear_state,
                                        lora,
                                    )
                                },
                            );
                        }
                    }
                } else {
                    self.release_captured_after_device_settlement("adapter_generation_mismatch")
                        .context(
                            "ROCm graph adapter-generation recovery failed; cleanup is quarantined",
                        )?;
                }
            }

            if let RocmGraphPressureDecision::ReplayOnly(reason) = pressure_decision {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_hidden(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            if self.budget_capture_suppressed(&requested_key) {
                return self.run_eager_fallback(
                    RocmGraphFallbackReason::GraphCacheByteBudget,
                    seq_len,
                    std::time::Duration::ZERO,
                    || {
                        Self::eager_forward_hidden(
                            backend,
                            token_id,
                            weights,
                            config,
                            paged_cache,
                            block_table,
                            seq_len,
                            linear_state,
                            lora,
                        )
                    },
                );
            }

            if let Some(reason) = self.pre_capture_rejection(&cache_key) {
                if !self.cache_full_warned {
                    self.cache_full_warned = true;
                    let memory = self.memory_accounting(&HashSet::new(), None);
                    tracing::warn!(
                        cached = self.captured.len(),
                        retained_bytes = memory.retained_bytes,
                        max_cached_graphs = self.max_cached_graphs(),
                        max_retained_bytes = self.max_retained_bytes(),
                        reason = reason.as_str(),
                        "ROCm graph capture skipped: bounded cache has no settled eviction path"
                    );
                }
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_hidden(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            let capture_started = std::time::Instant::now();
            match self.try_capture_hidden(
                backend,
                owner,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            ) {
                Ok(RocmCaptureStep::CapturedHidden(hidden))
                | Ok(RocmCaptureStep::CapturedHiddenUncached(hidden)) => return Ok(hidden),
                Ok(RocmCaptureStep::FallbackEager { reason, .. }) => {
                    return self.run_eager_fallback(
                        reason,
                        seq_len,
                        capture_started.elapsed(),
                        || {
                            Self::eager_forward_hidden(
                                backend,
                                token_id,
                                weights,
                                config,
                                paged_cache,
                                block_table,
                                seq_len,
                                linear_state,
                                lora,
                            )
                        },
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        "ROCm graph capture failed: {e:#}; disabling HIP graphs and attempting \
                         containment (a quarantined device requires process restart)"
                    );
                    self.enabled = false;
                    synchronize_after_rocm_graph_capture_failure(&weights.embed_tokens.device())
                        .with_context(|| format!("capture failed before recovery: {e:#}"))?;
                    return self.run_eager_fallback(
                        RocmGraphFallbackReason::CaptureFailure,
                        seq_len,
                        capture_started.elapsed(),
                        || {
                            Self::eager_forward_hidden(
                                backend,
                                token_id,
                                weights,
                                config,
                                paged_cache,
                                block_table,
                                seq_len,
                                linear_state,
                                lora,
                            )
                        },
                    );
                }
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = graph_row_id;
            let _ = linear_state;
            Self::eager_forward_hidden(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            )
        }
    }

    /// Run one bs=1 paged decode step, returning the greedy token directly.
    ///
    /// This follows the same graph warmup/capture/replay state machine as
    /// [`Self::decode_step_paged`] but keeps the eager tail on the fast
    /// `final_norm + lm_head argmax` path, avoiding materializing
    /// `[1, 1, vocab]` logits only to reduce them immediately.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_greedy(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
        graph_row_id: u64,
    ) -> Result<u32> {
        if !self.enabled || self.policy.force_eager_decode() {
            return Self::eager_forward_greedy(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            );
        }

        #[cfg(feature = "rocm")]
        {
            if !self.warmup_done {
                self.warmup_done = true;
                tracing::info!("ROCm graph runner: graph-shaped warmup decode step");
                match Self::eager_forward_greedy_with_position_buffer(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                ) {
                    Ok(token) => return Ok(token),
                    Err(e) => {
                        self.enabled = false;
                        tracing::error!(
                            "ROCm graph-shaped warmup failed: {e:#}; quarantining instead of retrying partially advanced state"
                        );
                        return fail_closed_after_rocm_warmup(weights, e);
                    }
                }
            }

            if !self.capture_requested {
                return Self::eager_forward_greedy(
                    backend,
                    token_id,
                    weights,
                    config,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state,
                    lora,
                );
            }

            Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
            let requested_key = RocmGraphKey::new(paged_cache, seq_len);
            let owner =
                match self.bind_decode_row_to_slot(graph_row_id, &requested_key, linear_state)? {
                    RocmGraphBindOutcome::Bound(owner) => owner,
                    RocmGraphBindOutcome::Fallback(reason) => {
                        return self.run_eager_fallback(
                            reason,
                            seq_len,
                            std::time::Duration::ZERO,
                            || {
                                Self::eager_forward_greedy(
                                    backend,
                                    token_id,
                                    weights,
                                    config,
                                    paged_cache,
                                    block_table,
                                    seq_len,
                                    linear_state,
                                    lora,
                                )
                            },
                        );
                    }
                };
            self.prepare_owner_decode(owner, graph_row_id, block_table, seq_len);
            let cache_key = RocmGraphCacheKey::new(owner, requested_key.clone());
            let pressure_decision = self.reconcile_memory_pressure(owner)?;
            if let RocmGraphPressureDecision::EagerOnly(reason) = pressure_decision {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_greedy(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            if let Some(reason) = self.non_capture_safe.get(&requested_key).copied() {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_greedy(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            if let Some(adapter_gen) = self
                .captured
                .get(&cache_key)
                .map(|captured| captured.adapter_gen)
            {
                if adapter_gen == self.adapter_generation {
                    let replay_started = std::time::Instant::now();
                    match self.replay_greedy(
                        &cache_key,
                        token_id,
                        backend,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                    ) {
                        Ok(token) => {
                            self.touch_captured_graph(&cache_key);
                            tracing::trace!(seq_len, "ROCm graph: replayed captured decode graph");
                            return Ok(token);
                        }
                        Err(e) => {
                            tracing::warn!(
                                "ROCm graph replay failed: {e:#}; disabling ROCm HIP graphs for this runner; if execution was attempted, the device is quarantined and restart is required"
                            );
                            self.enabled = false;
                            self.release_captured_after_device_settlement("replay_failure")
                                .with_context(|| {
                                    format!(
                                        "ROCm graph replay failed ({e:#}) and containment failed; the device is quarantined and restart is required"
                                    )
                                })?;
                            return self.run_eager_fallback(
                                RocmGraphFallbackReason::ReplayFailure,
                                seq_len,
                                replay_started.elapsed(),
                                || {
                                    Self::eager_forward_greedy(
                                        backend,
                                        token_id,
                                        weights,
                                        config,
                                        paged_cache,
                                        block_table,
                                        seq_len,
                                        linear_state,
                                        lora,
                                    )
                                },
                            );
                        }
                    }
                } else {
                    self.release_captured_after_device_settlement("adapter_generation_mismatch")
                        .context(
                            "ROCm graph adapter-generation recovery failed; cleanup is quarantined",
                        )?;
                }
            }

            if let RocmGraphPressureDecision::ReplayOnly(reason) = pressure_decision {
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_greedy(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            if self.budget_capture_suppressed(&requested_key) {
                return self.run_eager_fallback(
                    RocmGraphFallbackReason::GraphCacheByteBudget,
                    seq_len,
                    std::time::Duration::ZERO,
                    || {
                        Self::eager_forward_greedy(
                            backend,
                            token_id,
                            weights,
                            config,
                            paged_cache,
                            block_table,
                            seq_len,
                            linear_state,
                            lora,
                        )
                    },
                );
            }

            if let Some(reason) = self.pre_capture_rejection(&cache_key) {
                if !self.cache_full_warned {
                    self.cache_full_warned = true;
                    let memory = self.memory_accounting(&HashSet::new(), None);
                    tracing::warn!(
                        cached = self.captured.len(),
                        retained_bytes = memory.retained_bytes,
                        max_cached_graphs = self.max_cached_graphs(),
                        max_retained_bytes = self.max_retained_bytes(),
                        reason = reason.as_str(),
                        "ROCm graph capture skipped: bounded cache has no settled eviction path"
                    );
                }
                return self.run_eager_fallback(reason, seq_len, std::time::Duration::ZERO, || {
                    Self::eager_forward_greedy(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                });
            }

            let capture_started = std::time::Instant::now();
            match self.try_capture_greedy(
                backend,
                owner,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            ) {
                Ok(token) => return Ok(token),
                Err(e) => {
                    tracing::warn!(
                        "ROCm graph capture failed: {e:#}; disabling HIP graphs and attempting \
                         containment (a quarantined device requires process restart)"
                    );
                    self.enabled = false;
                    synchronize_after_rocm_graph_capture_failure(&weights.embed_tokens.device())
                        .with_context(|| format!("capture failed before recovery: {e:#}"))?;
                    return self.run_eager_fallback(
                        RocmGraphFallbackReason::CaptureFailure,
                        seq_len,
                        capture_started.elapsed(),
                        || {
                            Self::eager_forward_greedy(
                                backend,
                                token_id,
                                weights,
                                config,
                                paged_cache,
                                block_table,
                                seq_len,
                                linear_state,
                                lora,
                            )
                        },
                    );
                }
            }
        }

        #[cfg(not(feature = "rocm"))]
        {
            let _ = graph_row_id;
            let _ = linear_state;
            Self::eager_forward_greedy(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            )
        }
    }

    /// Plain eager decode forward — `model_forward_paged` over a single token.
    #[allow(clippy::too_many_arguments)]
    fn eager_forward(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        model_forward_paged(
            backend,
            &[token_id],
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            None,
        )
        .context("eager decode forward pass failed (rocm)")
    }

    /// Plain eager greedy decode forward.
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_greedy(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<u32> {
        model_forward_paged_next_token_greedy(
            backend,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            None,
        )
        .context("eager greedy decode forward pass failed (rocm)")
    }

    /// Plain eager hidden-only decode.
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_hidden(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        let sequence_lengths = [seq_len];
        let mut linear_states: [&mut LinearAttentionState; 1] = [linear_state];
        model_forward_paged_batched_decode_hidden(
            backend,
            &[token_id],
            weights,
            config,
            paged_cache,
            std::slice::from_ref(block_table),
            &sequence_lengths,
            &mut linear_states,
            lora,
        )
        .context("eager hidden-only decode forward pass failed (rocm)")
    }

    /// Eager decode with a graph-shaped position buffer (warms the allocator
    /// with the capture-shaped allocation sequence).
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_with_position_buffer(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        let device = weights.embed_tokens.device();
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        model_forward_paged(
            backend,
            &[token_id],
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            Some(&position_buffer),
        )
        .context("graph-shaped eager decode forward pass failed (rocm)")
    }

    /// Eager greedy decode with a graph-shaped position buffer.
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_greedy_with_position_buffer(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<u32> {
        let device = weights.embed_tokens.device();
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        model_forward_paged_next_token_greedy(
            backend,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            Some(&position_buffer),
        )
        .context("graph-shaped eager greedy decode forward pass failed (rocm)")
    }

    /// Eager hidden-only decode with a graph-shaped position buffer.
    #[cfg(feature = "rocm")]
    #[allow(clippy::too_many_arguments)]
    fn eager_forward_hidden_with_position_buffer(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        let device = weights.embed_tokens.device();
        let token_buffer = Self::new_token_buffer(device, token_id)?;
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        model_forward_paged_hidden_with_graph_inputs(
            backend,
            &[token_id],
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            &token_buffer,
            &position_buffer,
            None,
        )
        .context("graph-shaped eager hidden-only decode forward pass failed (rocm)")
    }

    fn new_position_buffer(device: Device, position: usize) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![position as f32], vec![1])
            .context("create ROCm graph position buffer")
    }
}

#[cfg(feature = "rocm")]
impl Drop for RocmGraphRunner {
    fn drop(&mut self) {
        let owners: HashSet<_> = self
            .graph_owners()
            .into_iter()
            .chain(self.graph_slots.keys().copied())
            .collect();
        if owners.is_empty() {
            return;
        }
        if let Err(error) = self.evict_graph_owners(
            &owners,
            "runner_drop",
            RocmGraphEvictionReason::Recovery,
            true,
        ) {
            tracing::error!(
                error = %format!("{error:#}"),
                "ROCm graph runner drop could not settle retained resources; cleanup quarantine remains authoritative"
            );
        }
    }
}

// ======================================================================
// Capture / replay machinery (ROCm only).
// ======================================================================
#[cfg(feature = "rocm")]
impl RocmGraphRunner {
    /// Replay a cached graph: refresh the per-step input buffers, sync the
    /// default stream, launch, sync the capture stream, then run lm_head eagerly
    /// on the replayed hidden.
    #[allow(clippy::too_many_arguments)]
    fn replay(
        &mut self,
        key: &RocmGraphCacheKey,
        token_id: u32,
        backend: &dyn BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<Tensor> {
        let context = self
            .captured
            .get(key)
            .map(|captured| captured.context.clone())
            .ok_or_else(|| anyhow::anyhow!("replay: key vanished"))?;
        let hidden =
            self.replay_hidden(key, token_id, weights, paged_cache, block_table, seq_len)?;
        match crate::forward::lm_head_from_hidden_eager(backend, &hidden, weights, config) {
            Ok(logits) => Ok(logits),
            Err(error) => {
                // The graph has already advanced recurrent state. Retrying the
                // whole step eagerly would advance it twice.
                context.quarantine_execution();
                Err(error).context("eager lm_head on replayed hidden")
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_greedy(
        &mut self,
        key: &RocmGraphCacheKey,
        token_id: u32,
        backend: &dyn BackendRuntime,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<u32> {
        let context = self
            .captured
            .get(key)
            .map(|captured| captured.context.clone())
            .ok_or_else(|| anyhow::anyhow!("replay: key vanished"))?;
        let hidden =
            self.replay_hidden(key, token_id, weights, paged_cache, block_table, seq_len)?;
        match crate::forward::lm_head_argmax_from_hidden_eager(backend, &hidden, weights, config) {
            Ok(token) => Ok(token),
            Err(error) => {
                context.quarantine_execution();
                Err(error).context("eager lm_head argmax on replayed hidden")
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_hidden(
        &mut self,
        key: &RocmGraphCacheKey,
        token_id: u32,
        weights: &GpuWeights,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<Tensor> {
        let result =
            self.replay_hidden_inner(key, token_id, weights, paged_cache, block_table, seq_len);
        self.counters.record_replay_outcome(result.is_ok());
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_hidden_inner(
        &self,
        key: &RocmGraphCacheKey,
        token_id: u32,
        weights: &GpuWeights,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<Tensor> {
        let captured = self
            .captured
            .get(key)
            .ok_or_else(|| anyhow::anyhow!("replay: key vanished"))?;

        paged_cache
            .ensure_pool_identity(captured.kv_pool_identity)
            .context("refuse ROCm graph replay after paged-KV pool replacement")?;

        Self::update_token_buffer(&captured.token_buffer, token_id)?;
        Self::update_position_buffer(&captured.position_buffer, seq_len)?;
        Self::update_rotary_buffers(
            &captured.rotary_cos_buffer,
            &captured.rotary_sin_buffer,
            &weights.rotary_inv_freq,
            &captured.position_buffer,
        )?;
        if let (Some(bt), Some(sk), Some(slot)) = (
            captured.block_table_buffer.as_ref(),
            captured.seqused_k_buffer.as_ref(),
            captured.kv_slot_buffer.as_ref(),
        ) {
            Self::update_paged_metadata_buffers(
                bt,
                sk,
                slot,
                block_table,
                paged_cache,
                seq_len,
                captured.max_seqlen_k,
            )?;
        }

        // The writes above land on the kt default stream while the graph
        // launches on its capture stream. The event dependency prevents stale
        // replay inputs without forcing the host to wait for either stream.
        captured
            .default_stream
            .record_event(&captured.replay_inputs_ready_event)
            .context("record per-replay ROCm graph inputs")?;
        captured
            .capture_stream
            .wait_event(&captured.replay_inputs_ready_event)
            .context("order per-replay ROCm graph input handoff")?;

        let mut plan = RocmDecodeReplayPlan::new(captured);
        let replay_key = kiln_graph::ReplayPlan::key(&plan);
        let replay_inputs = ReplayInputs::new(&replay_key, &captured.replay_state.inputs);
        kiln_graph::ReplayPlan::replay(&mut plan, replay_inputs)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        Ok(captured.output_hidden.clone())
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_hidden_batched(
        &mut self,
        key: &RocmGraphCacheKey,
        token_ids: &[u32],
        weights: &GpuWeights,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        sequence_lengths: &[usize],
    ) -> Result<Tensor> {
        let result = self.replay_hidden_batched_inner(
            key,
            token_ids,
            weights,
            paged_cache,
            block_tables,
            sequence_lengths,
        );
        self.counters.record_replay_outcome(result.is_ok());
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_hidden_batched_inner(
        &self,
        key: &RocmGraphCacheKey,
        token_ids: &[u32],
        weights: &GpuWeights,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        sequence_lengths: &[usize],
    ) -> Result<Tensor> {
        let captured = self
            .captured
            .get(key)
            .context("batched ROCm replay: key vanished")?;
        anyhow::ensure!(
            key.graph.batch_size == token_ids.len()
                && block_tables.len() == token_ids.len()
                && sequence_lengths.len() == token_ids.len(),
            "batched ROCm replay row-count mismatch"
        );
        paged_cache
            .ensure_pool_identity(captured.kv_pool_identity)
            .context("refuse batched ROCm graph replay after paged-KV pool replacement")?;

        Self::update_batched_token_buffer(&captured.token_buffer, token_ids)?;
        Self::update_batched_position_buffer(&captured.position_buffer, sequence_lengths)?;
        Self::update_rotary_buffers(
            &captured.rotary_cos_buffer,
            &captured.rotary_sin_buffer,
            &weights.rotary_inv_freq,
            &captured.position_buffer,
        )?;
        let (block_table_buffer, seqused_k_buffer, kv_slot_buffer) = (
            captured
                .block_table_buffer
                .as_ref()
                .context("batched ROCm graph missing stable block table")?,
            captured
                .seqused_k_buffer
                .as_ref()
                .context("batched ROCm graph missing stable sequence lengths")?,
            captured
                .kv_slot_buffer
                .as_ref()
                .context("batched ROCm graph missing stable KV slots")?,
        );
        Self::update_batched_paged_metadata_buffers(
            block_table_buffer,
            seqused_k_buffer,
            kv_slot_buffer,
            block_tables,
            paged_cache,
            sequence_lengths,
            captured.max_seqlen_k,
        )?;

        captured
            .default_stream
            .record_event(&captured.replay_inputs_ready_event)
            .context("record batched ROCm graph replay inputs")?;
        captured
            .capture_stream
            .wait_event(&captured.replay_inputs_ready_event)
            .context("order batched ROCm graph replay input handoff")?;

        let mut plan = RocmDecodeReplayPlan::new(captured);
        let replay_key = kiln_graph::ReplayPlan::key(&plan);
        let replay_inputs = ReplayInputs::new(&replay_key, &captured.replay_state.inputs);
        kiln_graph::ReplayPlan::replay(&mut plan, replay_inputs)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        Ok(captured.output_hidden.clone())
    }

    fn replay_state_for_capture(
        key: &RocmGraphKey,
        output_hidden: &Tensor,
        token_buffer: &Tensor,
        position_buffer: &Tensor,
        block_table_buffer: Option<&Tensor>,
        seqused_k_buffer: Option<&Tensor>,
        kv_slot_buffer: Option<&Tensor>,
        rotary_cos_buffer: &Tensor,
        rotary_sin_buffer: &Tensor,
        paged_decode_outputs: &[Tensor],
        paged_decode_lse: &[Tensor],
        gdn_decode_outputs: &[Tensor],
    ) -> ReplayState {
        let replay_key = ReplayKey::new(
            Backend::Rocm,
            "paged_decode_graph_outputs",
            vec![key.batch_size, key.max_seqlen_k, key.max_blocks_per_seq],
            Some(output_hidden.dtype()),
            key.batch_size,
            true,
        );
        let mut resources = vec![
            Self::stable_replay_resource(output_hidden),
            Self::stable_replay_resource(token_buffer),
            Self::stable_replay_resource(position_buffer),
            Self::stable_replay_resource(rotary_cos_buffer),
            Self::stable_replay_resource(rotary_sin_buffer),
        ];
        for tensor in [block_table_buffer, seqused_k_buffer, kv_slot_buffer]
            .into_iter()
            .flatten()
        {
            resources.push(Self::stable_replay_resource(tensor));
        }
        resources.extend(
            paged_decode_outputs
                .iter()
                .map(Self::stable_replay_resource),
        );
        resources.extend(paged_decode_lse.iter().map(Self::stable_replay_resource));
        resources.extend(gdn_decode_outputs.iter().map(Self::stable_replay_resource));
        ReplayState::new(replay_key, resources)
    }

    fn stable_replay_resource(tensor: &Tensor) -> ResidentResourceRef {
        ResidentResourceRef::from_tensor(
            tensor,
            Backend::Rocm,
            ReplayResourceStability::StableAcrossReplay,
        )
    }

    fn exact_tensor_values_match(left: &Tensor, right: &Tensor) -> Result<bool> {
        anyhow::ensure!(
            left.dims() == right.dims(),
            "ROCm graph parity tensor shape mismatch ({:?} vs {:?})",
            left.dims(),
            right.dims()
        );
        anyhow::ensure!(
            left.dtype() == right.dtype(),
            "ROCm graph parity tensor dtype mismatch ({} vs {})",
            left.dtype(),
            right.dtype()
        );
        if left.element_count() == 0 {
            return Ok(true);
        }
        let equal =
            kiln_tensor::ops::eq(left, right).context("compare ROCm graph parity tensor values")?;
        let all_equal = kiln_tensor::ops::all_axis(
            &equal
                .flatten_all()
                .context("flatten ROCm graph parity equality mask")?,
            0,
        )
        .context("reduce ROCm graph parity equality mask")?;
        Ok(all_equal
            .to_scalar::<u8>()
            .context("read ROCm graph parity equality scalar")?
            != 0)
    }

    fn capture_parity_tensor_temporary_bytes(
        output_hidden: &Tensor,
        linear_state: &LinearAttentionState,
    ) -> (u64, u64) {
        let state_tensors = linear_state
            .recurrent_states
            .iter()
            .chain(linear_state.conv_states.iter());
        let state_snapshot_bytes = graph_tensor_bytes(state_tensors.clone());
        let largest_comparison_mask = std::iter::once(output_hidden)
            .chain(state_tensors)
            .map(|tensor| tensor.element_count() as u64)
            .max()
            .unwrap_or(0);

        let retained_copies = state_snapshot_bytes
            .saturating_mul(2)
            .saturating_add(graph_tensor_bytes([output_hidden]));
        (retained_copies, largest_comparison_mask.saturating_add(1))
    }

    fn capture_parity_kv_temporary_bytes(
        paged_cache: &PagedKvCacheKt,
        batch_size: usize,
    ) -> Result<(u64, u64)> {
        let mut retained_copies = 0u64;
        let mut largest_comparison_scratch = 0u64;
        for layer_idx in 0..paged_cache.num_layers() {
            let (key_pool, value_pool) = paged_cache
                .pool_tensors(layer_idx)
                .with_context(|| format!("missing ROCm graph parity KV layer {layer_idx}"))?;
            anyhow::ensure!(
                key_pool.dims() == value_pool.dims()
                    && key_pool.dtype() == value_pool.dtype()
                    && key_pool.dims().len() == 3,
                "ROCm graph parity KV pool mismatch at layer {layer_idx}"
            );
            let row_elements = key_pool.dims()[1]
                .saturating_mul(key_pool.dims()[2])
                .saturating_mul(batch_size) as u64;
            let row_bytes = row_elements.saturating_mul(key_pool.dtype().size_in_bytes() as u64);
            retained_copies = retained_copies.saturating_add(row_bytes.saturating_mul(2));
            // Comparing a saved row set with the live pool requires one
            // gathered candidate tensor, one U8 equality mask, and one scalar.
            largest_comparison_scratch = largest_comparison_scratch
                .max(row_bytes.saturating_add(row_elements).saturating_add(1));
        }
        Ok((retained_copies, largest_comparison_scratch))
    }

    fn capture_parity_temporary_bytes(
        output_hidden: &Tensor,
        linear_state: &LinearAttentionState,
        paged_cache: &PagedKvCacheKt,
        batch_size: usize,
    ) -> Result<u64> {
        let (tensor_copies, tensor_scratch) =
            Self::capture_parity_tensor_temporary_bytes(output_hidden, linear_state);
        let (kv_copies, kv_scratch) =
            Self::capture_parity_kv_temporary_bytes(paged_cache, batch_size)?;
        Ok(tensor_copies
            .saturating_add(kv_copies)
            .saturating_add(tensor_scratch.max(kv_scratch)))
    }

    fn snapshot_paged_kv_slots(
        paged_cache: &PagedKvCacheKt,
        slots: &Tensor,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let mut snapshots = Vec::with_capacity(paged_cache.num_layers());
        for layer_idx in 0..paged_cache.num_layers() {
            let (key_pool, value_pool) = paged_cache
                .pool_tensors(layer_idx)
                .with_context(|| format!("missing ROCm graph parity KV layer {layer_idx}"))?;
            let key = key_pool.index_select(slots, 0).with_context(|| {
                format!("snapshot ROCm graph parity K slots at layer {layer_idx}")
            })?;
            let value = value_pool.index_select(slots, 0).with_context(|| {
                format!("snapshot ROCm graph parity V slots at layer {layer_idx}")
            })?;
            snapshots.push((key, value));
        }
        Ok(snapshots)
    }

    fn exact_paged_kv_slots_match(
        expected: &[(Tensor, Tensor)],
        paged_cache: &PagedKvCacheKt,
        slots: &Tensor,
    ) -> Result<(Option<usize>, Option<usize>)> {
        anyhow::ensure!(
            expected.len() == paged_cache.num_layers(),
            "ROCm graph capture parity KV layer-count mismatch"
        );
        let mut key_mismatch = None;
        let mut value_mismatch = None;
        for (layer_idx, (expected_key, expected_value)) in expected.iter().enumerate() {
            let (key_pool, value_pool) = paged_cache
                .pool_tensors(layer_idx)
                .with_context(|| format!("missing ROCm graph parity KV layer {layer_idx}"))?;
            if key_mismatch.is_none() {
                let actual_key = key_pool.index_select(slots, 0).with_context(|| {
                    format!("gather ROCm graph parity K slots at layer {layer_idx}")
                })?;
                if !Self::exact_tensor_values_match(expected_key, &actual_key)? {
                    key_mismatch = Some(layer_idx);
                }
            }
            if value_mismatch.is_none() {
                let actual_value = value_pool.index_select(slots, 0).with_context(|| {
                    format!("gather ROCm graph parity V slots at layer {layer_idx}")
                })?;
                if !Self::exact_tensor_values_match(expected_value, &actual_value)? {
                    value_mismatch = Some(layer_idx);
                }
            }
            if key_mismatch.is_some() && value_mismatch.is_some() {
                break;
            }
        }
        Ok((key_mismatch, value_mismatch))
    }

    fn exact_capture_outputs_match(
        expected_hidden: &Tensor,
        actual_hidden: &Tensor,
        expected_state: &LinearAttentionState,
        actual_state: &LinearAttentionState,
    ) -> Result<(bool, Option<usize>, Option<usize>)> {
        anyhow::ensure!(
            expected_state.recurrent_states.len() == actual_state.recurrent_states.len()
                && expected_state.conv_states.len() == actual_state.conv_states.len(),
            "ROCm graph capture parity state layer-count mismatch"
        );
        let hidden_match = Self::exact_tensor_values_match(expected_hidden, actual_hidden)?;
        let mut recurrent_mismatch = None;
        for (layer_idx, (expected, actual)) in expected_state
            .recurrent_states
            .iter()
            .zip(&actual_state.recurrent_states)
            .enumerate()
        {
            if !Self::exact_tensor_values_match(expected, actual)? {
                recurrent_mismatch = Some(layer_idx);
                break;
            }
        }
        let mut conv_mismatch = None;
        for (layer_idx, (expected, actual)) in expected_state
            .conv_states
            .iter()
            .zip(&actual_state.conv_states)
            .enumerate()
        {
            if !Self::exact_tensor_values_match(expected, actual)? {
                conv_mismatch = Some(layer_idx);
                break;
            }
        }
        Ok((hidden_match, recurrent_mismatch, conv_mismatch))
    }

    // --- per-replay in-place buffer refresh (frozen device pointers) ---

    fn update_token_buffer(token_buffer: &Tensor, token_id: u32) -> Result<()> {
        kiln_tensor::rocm_write_host_in_place(token_buffer, &[token_id])
            .context("update ROCm graph token buffer")
    }

    fn update_batched_token_buffer(token_buffer: &Tensor, token_ids: &[u32]) -> Result<()> {
        anyhow::ensure!(token_ids.len() > 1, "batched ROCm graph requires width > 1");
        kiln_tensor::rocm_write_host_in_place(token_buffer, token_ids)
            .context("update ROCm graph batched token buffer")
    }

    fn update_position_buffer(position_buffer: &Tensor, position: usize) -> Result<()> {
        kiln_tensor::rocm_write_host_in_place(position_buffer, &[position as f32])
            .context("update ROCm graph position buffer")
    }

    fn update_batched_position_buffer(position_buffer: &Tensor, positions: &[usize]) -> Result<()> {
        anyhow::ensure!(positions.len() > 1, "batched ROCm graph requires width > 1");
        let positions: Vec<f32> = positions.iter().map(|&position| position as f32).collect();
        kiln_tensor::rocm_write_host_in_place(position_buffer, &positions)
            .context("update ROCm graph batched position buffer")
    }

    fn update_rotary_buffers(
        rotary_cos_buffer: &Tensor,
        rotary_sin_buffer: &Tensor,
        rotary_inv_freq: &Tensor,
        position_buffer: &Tensor,
    ) -> Result<()> {
        // #34 BUG2 FIX: compute the rotary tables on the GPU via eager's exact
        // path (`forward::rotary_tables_from_tensor` -> device `cos`/`sin`), not
        // host CPU cos/sin. CPU cos != GPU cos (range reduction) perturbs only the
        // RoPE full-attention layers on replay -> divergence from eager. Same root
        // cause + fix as the CUDA path.
        let (cos, sin) =
            crate::forward::rotary_tables_from_tensor(position_buffer, rotary_inv_freq)?;
        let cos = cos
            .to_dtype(rotary_cos_buffer.dtype())?
            .reshape(rotary_cos_buffer.dims().to_vec())?;
        let sin = sin
            .to_dtype(rotary_sin_buffer.dtype())?
            .reshape(rotary_sin_buffer.dims().to_vec())?;
        rotary_cos_buffer
            .slice_set(&cos, 0, 0)
            .context("update ROCm graph rotary cos buffer (gpu)")?;
        rotary_sin_buffer
            .slice_set(&sin, 0, 0)
            .context("update ROCm graph rotary sin buffer (gpu)")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn update_paged_metadata_buffers(
        block_table_buffer: &Tensor,
        seqused_k_buffer: &Tensor,
        kv_slot_buffer: &Tensor,
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        seq_len: usize,
        max_seqlen_k: usize,
    ) -> Result<()> {
        let padded = Self::padded_block_table(block_table, paged_cache, max_seqlen_k)?;
        kiln_tensor::rocm_write_host_in_place(block_table_buffer, padded.as_slice())
            .context("update ROCm graph block table buffer")?;
        let attention_len = [(seq_len + 1) as u32];
        kiln_tensor::rocm_write_host_in_place(seqused_k_buffer, &attention_len)
            .context("update ROCm graph seqused_k buffer")?;
        let slot = [block_table
            .slot_for(seq_len, paged_cache.block_size())
            .with_context(|| format!("no slot for decode position {seq_len}"))?
            as u32];
        kiln_tensor::rocm_write_host_in_place(kv_slot_buffer, &slot)
            .context("update ROCm graph KV slot buffer")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn update_batched_paged_metadata_buffers(
        block_table_buffer: &Tensor,
        seqused_k_buffer: &Tensor,
        kv_slot_buffer: &Tensor,
        block_tables: &[&BlockTable],
        paged_cache: &PagedKvCacheKt,
        sequence_lengths: &[usize],
        max_seqlen_k: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            block_tables.len() > 1 && block_tables.len() == sequence_lengths.len(),
            "batched ROCm graph paged metadata row-count mismatch"
        );
        let mut flat = Vec::with_capacity(block_table_buffer.elem_count());
        for table in block_tables {
            flat.extend(Self::padded_block_table(table, paged_cache, max_seqlen_k)?);
        }
        anyhow::ensure!(
            flat.len() == block_table_buffer.elem_count(),
            "batched ROCm graph block-table shape changed across replay"
        );
        kiln_tensor::rocm_write_host_in_place(block_table_buffer, &flat)
            .context("update ROCm graph batched block table")?;

        let sequence_used: Vec<u32> = sequence_lengths
            .iter()
            .map(|&length| {
                u32::try_from(length + 1).context("ROCm graph sequence length exceeds u32")
            })
            .collect::<Result<_>>()?;
        kiln_tensor::rocm_write_host_in_place(seqused_k_buffer, &sequence_used)
            .context("update ROCm graph batched sequence lengths")?;
        let slots = paged_cache.resolve_unique_decode_slots(block_tables, sequence_lengths)?;
        kiln_tensor::rocm_write_host_in_place(kv_slot_buffer, &slots)
            .context("update ROCm graph batched KV slots")?;
        Ok(())
    }

    fn padded_block_table(
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        max_seqlen_k: usize,
    ) -> Result<Vec<u32>> {
        let block_size = paged_cache.block_size();
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let pages_per_chunk = kblock_n / block_size;
        let max_blocks_per_seq = (max_seqlen_k / kblock_n) * pages_per_chunk;
        let take = max_blocks_per_seq.min(block_table.blocks.len());
        let mut padded = Vec::with_capacity(max_blocks_per_seq);
        padded.extend_from_slice(&block_table.blocks[..take]);
        if padded.is_empty() {
            anyhow::bail!("paged decode graph block table is empty");
        }
        // Pad with the LAST REAL block (repeated), never an incrementing one —
        // an out-of-pool page baked into the graph would fault on replay; a
        // repeated valid block stays in-bounds and is masked by seqused_k.
        let pad_block = *padded.last().expect("non-empty");
        while padded.len() < max_blocks_per_seq {
            padded.push(pad_block);
        }
        Ok(padded)
    }

    // --- graph-stable buffer constructors ---

    fn new_block_table_buffer(
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        max_seqlen_k: usize,
        device: Device,
    ) -> Result<Tensor> {
        let padded = Self::padded_block_table(block_table, paged_cache, max_seqlen_k)?;
        let len = padded.len();
        Tensor::from_vec_on(device, padded, vec![1, len])
            .context("create ROCm graph block table buffer")
    }

    fn new_seqused_k_buffer(device: Device, attention_len: usize) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![attention_len as u32], vec![1])
            .context("create ROCm graph seqused_k buffer")
    }

    fn new_kv_slot_buffer(
        block_table: &BlockTable,
        paged_cache: &PagedKvCacheKt,
        seq_len: usize,
        device: Device,
    ) -> Result<Tensor> {
        let slot = block_table
            .slot_for(seq_len, paged_cache.block_size())
            .with_context(|| format!("no slot for decode position {seq_len}"))?
            as u32;
        Tensor::from_vec_on(device, vec![slot], vec![1]).context("create ROCm graph KV slot buffer")
    }

    fn new_batched_block_table_buffer(
        block_tables: &[&BlockTable],
        paged_cache: &PagedKvCacheKt,
        max_seqlen_k: usize,
        device: Device,
    ) -> Result<Tensor> {
        anyhow::ensure!(
            block_tables.len() > 1,
            "batched ROCm graph requires width > 1"
        );
        let mut flat = Vec::new();
        let mut width = None;
        for table in block_tables {
            let padded = Self::padded_block_table(table, paged_cache, max_seqlen_k)?;
            match width {
                Some(expected) => anyhow::ensure!(
                    expected == padded.len(),
                    "batched ROCm graph block-table widths differ"
                ),
                None => width = Some(padded.len()),
            }
            flat.extend(padded);
        }
        Tensor::from_vec_on(
            device,
            flat,
            vec![block_tables.len(), width.expect("non-empty batch")],
        )
        .context("create ROCm graph batched block table")
    }

    fn new_batched_seqused_k_buffer(device: Device, sequence_lengths: &[usize]) -> Result<Tensor> {
        let values = sequence_lengths
            .iter()
            .map(|&length| {
                u32::try_from(length + 1).context("ROCm graph sequence length exceeds u32")
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::from_vec_on(device, values, vec![sequence_lengths.len()])
            .context("create ROCm graph batched sequence lengths")
    }

    fn new_batched_kv_slot_buffer(
        block_tables: &[&BlockTable],
        paged_cache: &PagedKvCacheKt,
        sequence_lengths: &[usize],
        device: Device,
    ) -> Result<Tensor> {
        let slots = paged_cache.resolve_unique_decode_slots(block_tables, sequence_lengths)?;
        Tensor::from_vec_on(device, slots, vec![block_tables.len()])
            .context("create ROCm graph batched KV slots")
    }

    fn new_rotary_cos_buffer(
        config: &ModelConfig,
        device: Device,
        position: usize,
    ) -> Result<Tensor> {
        // #34 BUG2 FIX: GPU rotary (matches eager + update_rotary_buffers), not host CPU cos.
        let inv_freq = crate::forward::compute_rotary_inv_freq(
            config.rotary_dim(),
            config.rope_theta,
            &device,
        )?;
        let pos = Tensor::from_vec_on(device, vec![position as f32], vec![1])?;
        let (cos, _) = crate::forward::rotary_tables_from_tensor(&pos, &inv_freq)?;
        cos.to_dtype(kiln_tensor::DType::F32)?
            .contiguous()
            .context("create ROCm graph rotary cos buffer (gpu)")
    }

    fn new_rotary_sin_buffer(
        config: &ModelConfig,
        device: Device,
        position: usize,
    ) -> Result<Tensor> {
        // #34 BUG2 FIX: GPU rotary (see new_rotary_cos_buffer).
        let inv_freq = crate::forward::compute_rotary_inv_freq(
            config.rotary_dim(),
            config.rope_theta,
            &device,
        )?;
        let pos = Tensor::from_vec_on(device, vec![position as f32], vec![1])?;
        let (_, sin) = crate::forward::rotary_tables_from_tensor(&pos, &inv_freq)?;
        sin.to_dtype(kiln_tensor::DType::F32)?
            .contiguous()
            .context("create ROCm graph rotary sin buffer (gpu)")
    }

    fn new_output_hidden(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
    ) -> Result<Tensor> {
        Tensor::zeros_on(device, vec![1, 1, config.hidden_size], dtype)
            .context("create ROCm graph output hidden")
    }

    fn new_batched_rotary_buffers(
        weights: &GpuWeights,
        positions: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (cos, sin) =
            crate::forward::rotary_tables_from_tensor(positions, &weights.rotary_inv_freq)?;
        Ok((
            cos.to_dtype(kiln_tensor::DType::F32)?
                .contiguous()
                .context("create ROCm graph batched rotary cos")?,
            sin.to_dtype(kiln_tensor::DType::F32)?
                .contiguous()
                .context("create ROCm graph batched rotary sin")?,
        ))
    }

    fn new_batched_output_hidden(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
        batch_size: usize,
    ) -> Result<Tensor> {
        Tensor::zeros_on(device, vec![batch_size, 1, config.hidden_size], dtype)
            .context("create ROCm graph batched output hidden")
    }

    fn new_paged_decode_outputs(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let mut outputs = Vec::with_capacity(config.num_full_attention_layers);
        let mut lse = Vec::with_capacity(config.num_full_attention_layers);
        for _ in 0..config.num_full_attention_layers {
            outputs.push(
                Tensor::zeros_on(
                    device,
                    vec![1, 1, config.num_attention_heads, config.head_dim],
                    dtype,
                )
                .context("create ROCm graph paged decode output")?,
            );
            lse.push(
                Tensor::zeros_on(
                    device,
                    vec![1, config.num_attention_heads, 1],
                    kiln_tensor::DType::F32,
                )
                .context("create ROCm graph paged decode LSE")?,
            );
        }
        Ok((outputs, lse))
    }

    fn new_batched_paged_decode_outputs(
        config: &ModelConfig,
        device: Device,
        dtype: kiln_tensor::DType,
        batch_size: usize,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let mut outputs = Vec::with_capacity(config.num_full_attention_layers);
        let mut lse = Vec::with_capacity(config.num_full_attention_layers);
        for _ in 0..config.num_full_attention_layers {
            outputs.push(Tensor::zeros_on(
                device,
                vec![batch_size, 1, config.num_attention_heads, config.head_dim],
                dtype,
            )?);
            lse.push(Tensor::zeros_on(
                device,
                vec![batch_size, config.num_attention_heads, 1],
                kiln_tensor::DType::F32,
            )?);
        }
        Ok((outputs, lse))
    }

    fn prepare_gdn_recurrent_state_for_capture(
        linear_state: &mut LinearAttentionState,
    ) -> Result<()> {
        for state in &mut linear_state.recurrent_states {
            if state.dtype() != kiln_tensor::DType::BF16 {
                *state = state
                    .to_dtype(kiln_tensor::DType::BF16)
                    .context("prepare ROCm graph GDN recurrent state")?;
            }
        }
        Ok(())
    }

    fn new_gdn_decode_outputs(config: &ModelConfig, device: Device) -> Result<Vec<Tensor>> {
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let mut outputs = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            outputs.push(
                Tensor::zeros_on(
                    device,
                    vec![
                        1,
                        1,
                        config.linear_num_value_heads,
                        config.linear_value_head_dim,
                    ],
                    kiln_tensor::DType::BF16,
                )
                .context("create ROCm graph GDN decode output")?,
            );
        }
        Ok(outputs)
    }

    fn new_batched_gdn_decode_outputs(
        config: &ModelConfig,
        device: Device,
        batch_size: usize,
    ) -> Result<Vec<Tensor>> {
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let mut outputs = Vec::with_capacity(num_linear_layers);
        for _ in 0..num_linear_layers {
            outputs.push(Tensor::zeros_on(
                device,
                vec![
                    batch_size,
                    1,
                    config.linear_num_value_heads,
                    config.linear_value_head_dim,
                ],
                kiln_tensor::DType::BF16,
            )?);
        }
        Ok(outputs)
    }

    /// Capture a HIP graph for this decode step (bs=1), launch it once to
    /// compute + advance state, and return this step's logits. Mirrors
    /// `CudaGraphRunner::try_capture`.
    #[allow(clippy::too_many_arguments)]
    fn try_capture_batched_hidden(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        sequence_lengths: &[usize],
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<RocmCaptureStep> {
        let mut result = self.try_capture_batched_hidden_inner(
            backend,
            owner,
            token_ids,
            weights,
            config,
            paged_cache,
            block_tables,
            sequence_lengths,
            linear_state,
            lora,
        );
        if let Ok(RocmCaptureStep::FallbackEager { cleanup_timer, .. }) = &mut result {
            drop(cleanup_timer.take());
        }
        let outcome = match &result {
            Ok(RocmCaptureStep::CapturedHidden(_)) => RocmGraphCaptureOutcome::SucceededRetained,
            Ok(RocmCaptureStep::CapturedHiddenUncached(_)) => {
                RocmGraphCaptureOutcome::SucceededUncached
            }
            Ok(RocmCaptureStep::FallbackEager { .. }) => RocmGraphCaptureOutcome::Deferred,
            Err(_) => RocmGraphCaptureOutcome::Failed,
        };
        self.counters.record_capture_outcome(outcome);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn try_capture_batched_hidden_inner(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        sequence_lengths: &[usize],
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<RocmCaptureStep> {
        let batch_size = token_ids.len();
        anyhow::ensure!(
            batch_size > 1
                && block_tables.len() == batch_size
                && sequence_lengths.len() == batch_size,
            "batched ROCm graph capture row-count mismatch"
        );
        let device = weights.embed_tokens.device();
        let dtype = weights.embed_tokens.dtype();
        let device_idx = match device {
            Device::Rocm(index) => index,
            _ => anyhow::bail!("ROCm graphs require a ROCm device"),
        };
        let key = RocmGraphKey::new_batched(paged_cache, sequence_lengths)?;
        let cache_key = RocmGraphCacheKey::new(owner, key.clone());
        let pre_candidate_headroom_timer = self
            .phase_telemetry
            .timer(RocmGraphPhase::PreCandidateHeadroom);
        match self.reconcile_memory_pressure(owner)? {
            RocmGraphPressureDecision::Normal => {}
            RocmGraphPressureDecision::ReplayOnly(reason)
            | RocmGraphPressureDecision::EagerOnly(reason) => {
                return Ok(RocmCaptureStep::fallback(reason));
            }
        }
        if let Some(reason) = self.reservation_retry_suppressed(&key) {
            return Ok(RocmCaptureStep::fallback(reason));
        }
        if let Some(reason) = self.reserve_capture_entry_capacity(&cache_key)? {
            return Ok(RocmCaptureStep::fallback(reason));
        }
        drop(pre_candidate_headroom_timer);
        let candidate_warm_timer = self.phase_telemetry.timer(RocmGraphPhase::CandidateWarm);

        let context = kiln_tensor::primary_rocm_context(device_idx)
            .context("batched ROCm graph capture context")?;
        let stream = context
            .new_stream()
            .map_err(|error| anyhow::anyhow!("create batched ROCm capture stream: {error}"))?;
        let blaslt_workspace_lease =
            kiln_tensor::rocm_blaslt_workspace_lease(device_idx, &context, &stream)
                .context("lease batched ROCm graph hipBLASLt workspace")?;
        let default_stream = context.default_stream();
        let replay_inputs_ready_event = context
            .new_event()
            .map_err(|error| anyhow::anyhow!("create batched ROCm input event: {error}"))?;
        let replay_complete_event = context
            .new_event()
            .map_err(|error| anyhow::anyhow!("create batched ROCm completion event: {error}"))?;

        let token_buffer = Tensor::from_vec_on(device, token_ids.to_vec(), vec![batch_size])
            .context("create ROCm graph batched token buffer")?;
        let positions: Vec<f32> = sequence_lengths.iter().map(|&value| value as f32).collect();
        let position_buffer = Tensor::from_vec_on(device, positions, vec![batch_size])
            .context("create ROCm graph batched position buffer")?;
        let mut output_hidden = Self::new_batched_output_hidden(config, device, dtype, batch_size)?;
        let (rotary_cos_buffer, rotary_sin_buffer) =
            Self::new_batched_rotary_buffers(weights, &position_buffer)?;
        let block_table_buffer = Some(Self::new_batched_block_table_buffer(
            block_tables,
            paged_cache,
            key.max_seqlen_k,
            device,
        )?);
        let seqused_k_buffer = Some(Self::new_batched_seqused_k_buffer(
            device,
            sequence_lengths,
        )?);
        let kv_slot_buffer = Some(Self::new_batched_kv_slot_buffer(
            block_tables,
            paged_cache,
            sequence_lengths,
            device,
        )?);
        let (paged_decode_outputs, paged_decode_lse) =
            Self::new_batched_paged_decode_outputs(config, device, dtype, batch_size)?;
        let gdn_decode_outputs = Self::new_batched_gdn_decode_outputs(config, device, batch_size)?;
        Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
        let stable_graph_io_bytes = graph_tensor_bytes(
            [
                &token_buffer,
                &position_buffer,
                &output_hidden,
                &rotary_cos_buffer,
                &rotary_sin_buffer,
            ]
            .into_iter()
            .chain(
                [
                    block_table_buffer.as_ref(),
                    seqused_k_buffer.as_ref(),
                    kv_slot_buffer.as_ref(),
                ]
                .into_iter()
                .flatten(),
            )
            .chain(paged_decode_outputs.iter())
            .chain(paged_decode_lse.iter())
            .chain(gdn_decode_outputs.iter()),
        );
        let capture_parity_temporary_bytes = Self::capture_parity_temporary_bytes(
            &output_hidden,
            linear_state,
            paged_cache,
            batch_size,
        )?;
        let Some(_capture_parity_reservation) =
            kiln_memory::MemoryGovernor::try_global_cached_reserve(capture_parity_temporary_bytes)
        else {
            self.remember_reservation_denial(&key, capture_parity_temporary_bytes);
            self.counters.pre_capture_memory_reservation_denied_skips = self
                .counters
                .pre_capture_memory_reservation_denied_skips
                .saturating_add(1);
            drop(candidate_warm_timer);
            return Ok(RocmCaptureStep::fallback_after_candidate(
                RocmGraphFallbackReason::MemoryReservationDenied,
                &self.phase_telemetry,
            ));
        };

        let arena_context = kiln_tensor::primary_rocm_context(device_idx)
            .context("batched ROCm capture arena context")?;
        let arena = std::rc::Rc::new(std::cell::RefCell::new(
            kiln_tensor::RocmCaptureArena::new_record(arena_context, device_idx),
        ));
        let gdn_snapshot = linear_state
            .snapshot()
            .context("snapshot batched GDN state before graph warm pass")?;
        attributed_rocm_graph_synchronize(
            "batched_default_inputs_before_warmup",
            "rocm_graph_capture_warmup",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(
                        &default_stream,
                        kiln_tensor::RocmSyncReason::GraphBoundary,
                    )
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        )?;
        let (warm_result, warm_htod) =
            kiln_tensor::with_rocm_htod_observer_detailed(device_idx, || {
                kiln_tensor::with_rocm_capture_arena(arena.clone(), || unsafe {
                    kiln_tensor::with_rocm_graph_capture_stream(stream.clone(), || {
                        let mut graph_inputs = BatchedPagedDecodeGraphInputs {
                            token_ids: &token_buffer,
                            positions: &position_buffer,
                            block_table: block_table_buffer
                                .as_ref()
                                .expect("batched block table allocated"),
                            seqused_k: seqused_k_buffer
                                .as_ref()
                                .expect("batched sequence lengths allocated"),
                            kv_slot: kv_slot_buffer.as_ref().expect("batched KV slots allocated"),
                            max_seqlen_k: key.max_seqlen_k,
                            rotary_cos: &rotary_cos_buffer,
                            rotary_sin: &rotary_sin_buffer,
                            attn_out: &paged_decode_outputs,
                            softmax_lse: &paged_decode_lse,
                            output_hidden: &mut output_hidden,
                            linear_state,
                        };
                        model_forward_paged_batched_hidden_with_graph_inputs(
                            backend,
                            token_ids,
                            weights,
                            config,
                            paged_cache,
                            block_tables,
                            sequence_lengths,
                            lora,
                            &mut graph_inputs,
                        )
                    })
                })
            });
        let warm_sync_result = attributed_rocm_graph_synchronize(
            "batched_capture_stream_warmup_completion",
            "rocm_graph_capture_warmup",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(&stream, kiln_tensor::RocmSyncReason::GraphBoundary)
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        );
        if let Err(error) = warm_result {
            if let Err(sync_error) = warm_sync_result {
                return fail_closed_after_rocm_warmup(
                    weights,
                    sync_error.context(format!(
                        "capture stream synchronization failed after batched warm-forward failure ({error:#})"
                    )),
                );
            }
            Self::restore_linear_state_after_execution(
                &context,
                linear_state,
                &gdn_snapshot,
                "restore batched GDN state after warm-forward failure",
            )?;
            if crate::forward::is_rocm_graph_shape_dependent_attention(&error) {
                self.non_capture_safe
                    .insert(key, RocmGraphFallbackReason::ShapeDependentAttention);
                drop(candidate_warm_timer);
                return Ok(RocmCaptureStep::fallback_after_candidate(
                    RocmGraphFallbackReason::ShapeDependentAttention,
                    &self.phase_telemetry,
                ));
            }
            return Err(error).context("batched frozen-pointer warm pass failed");
        }
        if let Err(sync_error) = warm_sync_result {
            return fail_closed_after_rocm_warmup(
                weights,
                sync_error
                    .context("capture stream synchronization failed after batched warm forward"),
            );
        }
        let parity_snapshots = (|| {
            let warm_hidden = output_hidden
                .copy()
                .context("retain batched ROCm eager warm hidden for capture parity")?;
            let warm_gdn_state = linear_state
                .snapshot()
                .context("retain batched ROCm eager warm state for capture parity")?;
            let warm_kv_slots = Self::snapshot_paged_kv_slots(
                paged_cache,
                kv_slot_buffer.as_ref().expect("batched KV slots allocated"),
            )
            .context("retain batched ROCm eager warm KV slots for capture parity")?;
            Ok::<_, anyhow::Error>((warm_hidden, warm_gdn_state, warm_kv_slots))
        })();
        let restore_result = Self::restore_linear_state_after_execution(
            &context,
            linear_state,
            &gdn_snapshot,
            "restore batched GDN state after warm pass",
        );
        let (warm_hidden, warm_gdn_state, warm_kv_slots) = match parity_snapshots {
            Ok(snapshots) => {
                restore_result?;
                snapshots
            }
            Err(error) => {
                restore_result
                    .context("restore batched GDN state after capture-parity snapshot failure")?;
                return Err(error).context("snapshot batched ROCm eager capture parity");
            }
        };
        drop(candidate_warm_timer);
        let pre_native_reservation_timer = self
            .phase_telemetry
            .timer(RocmGraphPhase::PreNativeReservation);

        let reserved_stable_io = unique_rocm_tensor_allocations(
            [
                &token_buffer,
                &position_buffer,
                &output_hidden,
                &rotary_cos_buffer,
                &rotary_sin_buffer,
            ]
            .into_iter()
            .chain(
                [
                    block_table_buffer.as_ref(),
                    seqused_k_buffer.as_ref(),
                    kv_slot_buffer.as_ref(),
                ]
                .into_iter()
                .flatten(),
            )
            .chain(paged_decode_outputs.iter())
            .chain(paged_decode_lse.iter())
            .chain(gdn_decode_outputs.iter()),
        )
        .context("measure batched ROCm graph-stable allocations")?;
        let reserved_capture_arena = {
            let arena = arena.borrow();
            unique_rocm_storage_allocations(device_idx, arena.retained_buffers())
        };
        let workspace_stats = blaslt_workspace_lease
            .stats()
            .map_err(|error| anyhow::anyhow!(error))
            .context("measure batched ROCm graph hipBLASLt workspace")?;
        let reserved_workspace =
            (workspace_stats.retained_bytes > 0).then_some(RocmAllocationRecord {
                key: RocmAllocationKey {
                    device_index: device_idx,
                    allocation_id: workspace_stats.allocation_id as u64,
                },
                bytes: workspace_stats.retained_bytes,
            });
        let reserved_accounting = RocmGraphEntryAccounting {
            stable_io: reserved_stable_io,
            capture_arena: reserved_capture_arena,
            blaslt_workspace: reserved_workspace,
        };
        let transient_candidate_bytes = reserved_accounting.retained_bytes_excluding_slot();
        let peak_transient_candidate_bytes =
            transient_candidate_bytes.saturating_add(capture_parity_temporary_bytes);
        self.phase_telemetry
            .record_transient_candidate_bytes(peak_transient_candidate_bytes);
        if warm_htod.copy_count > 0 {
            let attempts = self.capture_retry.entry(key.clone()).or_insert(0);
            *attempts += 1;
            let reason = if *attempts >= Self::CAPTURE_RETRY_LIMIT {
                self.non_capture_safe.insert(
                    key.clone(),
                    RocmGraphFallbackReason::PersistentHostRoundTrip,
                );
                self.capture_retry.remove(&key);
                RocmGraphFallbackReason::PersistentHostRoundTrip
            } else {
                RocmGraphFallbackReason::ColdCacheHostRoundTrip
            };
            for site in &warm_htod.sites {
                tracing::warn!(
                    event = "rocm_graph_capture_host_transfer",
                    reason = reason.as_str(),
                    batch_size,
                    source_file = site.source_file,
                    source_line = site.source_line,
                    source_column = site.source_column,
                    bytes_per_copy = site.bytes_per_copy,
                    copy_count = site.copy_count,
                    total_bytes = site.total_bytes,
                    "batched ROCm graph warm pass observed a host-to-device transfer"
                );
            }
            drop(pre_native_reservation_timer);
            return Ok(RocmCaptureStep::fallback_after_candidate(
                reason,
                &self.phase_telemetry,
            ));
        }
        self.capture_retry.remove(&key);

        match self.reconcile_memory_pressure(owner)? {
            RocmGraphPressureDecision::Normal => {}
            RocmGraphPressureDecision::ReplayOnly(reason)
            | RocmGraphPressureDecision::EagerOnly(reason) => {
                drop(pre_native_reservation_timer);
                return Ok(RocmCaptureStep::fallback_after_candidate(
                    reason,
                    &self.phase_telemetry,
                ));
            }
        }
        if !self.matching_memory_governor() {
            self.counters.memory_governor_selector_mismatch_skips = self
                .counters
                .memory_governor_selector_mismatch_skips
                .saturating_add(1);
            drop(pre_native_reservation_timer);
            return Ok(RocmCaptureStep::fallback_after_candidate(
                RocmGraphFallbackReason::MemoryGovernorSelectorMismatch,
                &self.phase_telemetry,
            ));
        }
        let Some(governor_candidate_reservation) =
            kiln_memory::MemoryGovernor::try_global_cached_reserve(transient_candidate_bytes)
        else {
            self.remember_reservation_denial(&key, peak_transient_candidate_bytes);
            self.counters.pre_capture_memory_reservation_denied_skips = self
                .counters
                .pre_capture_memory_reservation_denied_skips
                .saturating_add(1);
            drop(pre_native_reservation_timer);
            return Ok(RocmCaptureStep::fallback_after_candidate(
                RocmGraphFallbackReason::MemoryReservationDenied,
                &self.phase_telemetry,
            ));
        };
        self.reservation_denied_bytes.remove(&key);
        self.reservation_denied_wide_bytes = None;
        if let Some(reason) = self.reserve_capture_candidate(&cache_key, &reserved_accounting)? {
            drop(pre_native_reservation_timer);
            return Ok(RocmCaptureStep::fallback_after_candidate(
                reason,
                &self.phase_telemetry,
            ));
        }
        drop(pre_native_reservation_timer);

        let mut native_capture_timer =
            Some(self.phase_telemetry.timer(RocmGraphPhase::NativeCapture));
        arena.borrow_mut().begin_replay();
        let capture_snapshot = gdn_snapshot;
        let mut capture_failure_guard = RocmCaptureFailureGuard::new(context.clone());
        attributed_rocm_graph_synchronize(
            "batched_default_inputs_before_capture",
            "rocm_graph_capture_begin",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(
                        &default_stream,
                        kiln_tensor::RocmSyncReason::GraphBoundary,
                    )
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        )?;
        attributed_rocm_graph_synchronize(
            "batched_capture_stream_before_begin",
            "rocm_graph_capture_begin",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(&stream, kiln_tensor::RocmSyncReason::GraphBoundary)
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        )?;
        stream
            .begin_capture()
            .map_err(|error| anyhow::anyhow!("begin batched ROCm capture: {error}"))?;
        let capture_result = kiln_tensor::with_rocm_capture_arena(arena.clone(), || unsafe {
            kiln_tensor::with_rocm_graph_capture_stream(stream.clone(), || {
                let mut graph_inputs = BatchedPagedDecodeGraphInputs {
                    token_ids: &token_buffer,
                    positions: &position_buffer,
                    block_table: block_table_buffer
                        .as_ref()
                        .expect("batched block table allocated"),
                    seqused_k: seqused_k_buffer
                        .as_ref()
                        .expect("batched sequence lengths allocated"),
                    kv_slot: kv_slot_buffer.as_ref().expect("batched KV slots allocated"),
                    max_seqlen_k: key.max_seqlen_k,
                    rotary_cos: &rotary_cos_buffer,
                    rotary_sin: &rotary_sin_buffer,
                    attn_out: &paged_decode_outputs,
                    softmax_lse: &paged_decode_lse,
                    output_hidden: &mut output_hidden,
                    linear_state,
                };
                model_forward_paged_batched_hidden_with_graph_inputs(
                    backend,
                    token_ids,
                    weights,
                    config,
                    paged_cache,
                    block_tables,
                    sequence_lengths,
                    lora,
                    &mut graph_inputs,
                )
            })
        });
        let graph_result = stream.end_capture();
        if let Err(error) = capture_result {
            capture_failure_guard.settle_before_rollback()?;
            capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                linear_state,
                &capture_snapshot,
                "restore batched GDN state after capture-forward failure",
            ))?;
            return Err(error).context("batched forward failed during ROCm graph capture");
        }
        let graph = match graph_result {
            Ok(graph) => graph,
            Err(error) => {
                capture_failure_guard.settle_before_rollback()?;
                capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                    linear_state,
                    &capture_snapshot,
                    "restore batched GDN state after end-capture failure",
                ))?;
                return Err(anyhow::anyhow!("end batched ROCm capture: {error}"));
            }
        };
        capture_failure_guard.graph = Some(graph);
        let exec = match capture_failure_guard
            .graph
            .as_ref()
            .expect("captured batched graph installed")
            .instantiate()
        {
            Ok(exec) => exec,
            Err(error) => {
                capture_failure_guard.settle_before_rollback()?;
                capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                    linear_state,
                    &capture_snapshot,
                    "restore batched GDN state after graph-instantiation failure",
                ))?;
                return Err(anyhow::anyhow!("instantiate batched ROCm graph: {error}"));
            }
        };
        capture_failure_guard.exec = Some(exec);
        if let Err(error) = capture_failure_guard
            .exec
            .as_ref()
            .expect("batched ROCm graph exec installed")
            .launch(&stream)
        {
            capture_failure_guard.settle_before_rollback()?;
            capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                linear_state,
                &capture_snapshot,
                "restore batched GDN state after first launch failure",
            ))?;
            return Err(anyhow::anyhow!("first batched ROCm graph launch: {error}"));
        }
        if let Err(error) = attributed_rocm_graph_synchronize(
            "batched_first_launch_completion",
            "rocm_graph_first_launch",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(&stream, kiln_tensor::RocmSyncReason::GraphBoundary)
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        ) {
            capture_failure_guard.settle_before_rollback()?;
            capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                linear_state,
                &capture_snapshot,
                "restore batched GDN state after first-launch wait failure",
            ))?;
            return Err(error).context("settle first batched ROCm graph launch");
        }

        let parity_started = std::time::Instant::now();
        let parity_bytes = graph_tensor_bytes(
            [&warm_hidden, &output_hidden]
                .into_iter()
                .chain(warm_gdn_state.recurrent_states.iter())
                .chain(warm_gdn_state.conv_states.iter())
                .chain(linear_state.recurrent_states.iter())
                .chain(linear_state.conv_states.iter()),
        )
        .saturating_add(
            graph_tensor_bytes(warm_kv_slots.iter().flat_map(|(key, value)| [key, value]))
                .saturating_mul(2),
        );
        let parity = Self::exact_capture_outputs_match(
            &warm_hidden,
            &output_hidden,
            &warm_gdn_state,
            linear_state,
        )
        .and_then(|(hidden_match, recurrent_mismatch, conv_mismatch)| {
            let (kv_key_mismatch, kv_value_mismatch) = Self::exact_paged_kv_slots_match(
                &warm_kv_slots,
                paged_cache,
                kv_slot_buffer.as_ref().expect("batched KV slots allocated"),
            )?;
            Ok((
                hidden_match,
                recurrent_mismatch,
                conv_mismatch,
                kv_key_mismatch,
                kv_value_mismatch,
            ))
        });
        let parity_duration = parity_started.elapsed();
        let (hidden_match, recurrent_mismatch, conv_mismatch, kv_key_mismatch, kv_value_mismatch) =
            match parity {
                Ok(parity) => parity,
                Err(error) => {
                    tracing::error!(
                        event = "rocm_graph_capture_parity_check",
                        batch_size,
                        outcome = "error",
                        comparison_complete = false,
                        compared_bytes = parity_bytes,
                        duration_ms = parity_duration.as_secs_f64() * 1000.0,
                        error = %format!("{error:#}"),
                        "ROCm batched graph first-launch comparison failed"
                    );
                    capture_failure_guard.settle_before_rollback()?;
                    capture_failure_guard.complete_rollback(
                        Self::restore_linear_state_in_place(
                            linear_state,
                            &capture_snapshot,
                            "restore batched GDN state after capture parity-check failure",
                        ),
                    )?;
                    return Err(error).context("compare batched ROCm graph capture parity");
                }
            };
        tracing::info!(
            event = "rocm_graph_capture_parity_check",
            batch_size,
            outcome = if hidden_match
                && recurrent_mismatch.is_none()
                && conv_mismatch.is_none()
                && kv_key_mismatch.is_none()
                && kv_value_mismatch.is_none()
            {
                "passed"
            } else {
                "failed"
            },
            comparison_complete = true,
            compared_bytes = parity_bytes,
            duration_ms = parity_duration.as_secs_f64() * 1000.0,
            hidden_match,
            recurrent_mismatch_layer = recurrent_mismatch.map(|layer| layer as u64),
            conv_mismatch_layer = conv_mismatch.map(|layer| layer as u64),
            kv_key_mismatch_layer = kv_key_mismatch.map(|layer| layer as u64),
            kv_value_mismatch_layer = kv_value_mismatch.map(|layer| layer as u64),
            "ROCm batched graph first launch compared with its eager warm pass"
        );
        if !hidden_match
            || recurrent_mismatch.is_some()
            || conv_mismatch.is_some()
            || kv_key_mismatch.is_some()
            || kv_value_mismatch.is_some()
        {
            capture_failure_guard.settle_before_rollback()?;
            capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                linear_state,
                &capture_snapshot,
                "restore batched GDN state after capture parity failure",
            ))?;
            anyhow::bail!(
                "batched ROCm graph capture parity failed: hidden_match={hidden_match}, \
                 recurrent_mismatch_layer={recurrent_mismatch:?}, \
                 conv_mismatch_layer={conv_mismatch:?}, \
                 kv_key_mismatch_layer={kv_key_mismatch:?}, \
                 kv_value_mismatch_layer={kv_value_mismatch:?}"
            );
        }

        let captured_hidden = output_hidden.clone();
        let arena_buffers = arena.borrow_mut().take_retained();
        let workspace_stats = blaslt_workspace_lease
            .stats()
            .map_err(|error| anyhow::anyhow!(error))?;
        let blaslt_workspace =
            (workspace_stats.retained_bytes > 0).then_some(RocmAllocationRecord {
                key: RocmAllocationKey {
                    device_index: device_idx,
                    allocation_id: workspace_stats.allocation_id as u64,
                },
                bytes: workspace_stats.retained_bytes,
            });
        let RocmGraphEntryAccounting {
            stable_io,
            capture_arena,
            blaslt_workspace: _,
        } = reserved_accounting;
        let accounting = RocmGraphEntryAccounting {
            stable_io,
            capture_arena,
            blaslt_workspace,
        };
        let replay_state = Self::replay_state_for_capture(
            &key,
            &output_hidden,
            &token_buffer,
            &position_buffer,
            block_table_buffer.as_ref(),
            seqused_k_buffer.as_ref(),
            kv_slot_buffer.as_ref(),
            &rotary_cos_buffer,
            &rotary_sin_buffer,
            &paged_decode_outputs,
            &paged_decode_lse,
            &gdn_decode_outputs,
        );
        let graph = capture_failure_guard
            .graph
            .take()
            .expect("successful batched capture retains source graph");
        let exec = capture_failure_guard
            .exec
            .take()
            .expect("successful batched capture retains graph exec");
        capture_failure_guard.disarm();
        let candidate = CapturedDecodeGraphRocm {
            accounting,
            last_used_tick: 0,
            _graph: graph,
            exec,
            output_hidden,
            capture_stream: stream,
            context,
            default_stream,
            replay_inputs_ready_event,
            replay_complete_event,
            adapter_gen: self.adapter_generation,
            kv_pool_identity: paged_cache.pool_identity(),
            token_buffer,
            position_buffer,
            block_table_buffer,
            seqused_k_buffer,
            kv_slot_buffer,
            rotary_cos_buffer,
            rotary_sin_buffer,
            _paged_decode_outputs: paged_decode_outputs,
            _paged_decode_lse: paged_decode_lse,
            max_seqlen_k: key.max_seqlen_k,
            _gdn_decode_outputs: gdn_decode_outputs,
            _capture_arena_buffers: arena_buffers,
            replay_state,
            _blaslt_workspace_lease: blaslt_workspace_lease,
        };
        let retained =
            self.admit_captured_graph(cache_key, candidate, &mut native_capture_timer)?;
        if retained {
            governor_candidate_reservation.commit_allocated();
        }
        drop(native_capture_timer.take());
        Ok(if retained {
            RocmCaptureStep::CapturedHidden(captured_hidden)
        } else {
            RocmCaptureStep::CapturedHiddenUncached(captured_hidden)
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn try_capture(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Tensor> {
        let capture_started = std::time::Instant::now();
        match self.try_capture_hidden(
            backend,
            owner,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            linear_state,
            lora,
        )? {
            RocmCaptureStep::CapturedHidden(hidden)
            | RocmCaptureStep::CapturedHiddenUncached(hidden) => {
                match crate::forward::lm_head_from_hidden_eager(backend, &hidden, weights, config) {
                    Ok(logits) => Ok(logits),
                    Err(error) => {
                        // The first graph launch already advanced recurrent
                        // state. A whole-step eager retry would advance it twice.
                        quarantine_rocm_tensor_context(&hidden);
                        Err(error).context("eager lm_head on captured hidden (first launch)")
                    }
                }
            }
            RocmCaptureStep::FallbackEager { reason, .. } => {
                self.run_eager_fallback(reason, seq_len, capture_started.elapsed(), || {
                    Self::eager_forward(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                })
            }
        }
    }

    /// Capture a HIP graph for this decode step (bs=1), launch it once to
    /// compute + advance state, and return this step's greedy token.
    #[allow(clippy::too_many_arguments)]
    fn try_capture_greedy(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<u32> {
        let capture_started = std::time::Instant::now();
        match self.try_capture_hidden(
            backend,
            owner,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            linear_state,
            lora,
        )? {
            RocmCaptureStep::CapturedHidden(hidden)
            | RocmCaptureStep::CapturedHiddenUncached(hidden) => {
                match crate::forward::lm_head_argmax_from_hidden_eager(
                    backend, &hidden, weights, config,
                ) {
                    Ok(token) => Ok(token),
                    Err(error) => {
                        quarantine_rocm_tensor_context(&hidden);
                        Err(error).context("eager lm_head argmax on captured hidden (first launch)")
                    }
                }
            }
            RocmCaptureStep::FallbackEager { reason, .. } => {
                self.run_eager_fallback(reason, seq_len, capture_started.elapsed(), || {
                    Self::eager_forward_greedy(
                        backend,
                        token_id,
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state,
                        lora,
                    )
                })
            }
        }
    }

    /// Capture a HIP graph for this decode step (bs=1), launch it once to
    /// compute + advance state, and return the graph-stable hidden.
    #[allow(clippy::too_many_arguments)]
    fn try_capture_hidden(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<RocmCaptureStep> {
        let mut result = self.try_capture_hidden_inner(
            backend,
            owner,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            linear_state,
            lora,
        );
        if let Ok(RocmCaptureStep::FallbackEager { cleanup_timer, .. }) = &mut result {
            drop(cleanup_timer.take());
        }
        let outcome = match &result {
            Ok(RocmCaptureStep::CapturedHidden(_)) => RocmGraphCaptureOutcome::SucceededRetained,
            Ok(RocmCaptureStep::CapturedHiddenUncached(_)) => {
                RocmGraphCaptureOutcome::SucceededUncached
            }
            Ok(RocmCaptureStep::FallbackEager { .. }) => RocmGraphCaptureOutcome::Deferred,
            Err(_) => RocmGraphCaptureOutcome::Failed,
        };
        self.counters.record_capture_outcome(outcome);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn try_capture_hidden_inner(
        &mut self,
        backend: &dyn BackendRuntime,
        owner: RocmGraphOwner,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<RocmCaptureStep> {
        let device = weights.embed_tokens.device();
        let dtype = weights.embed_tokens.dtype();
        let device_idx = match device {
            Device::Rocm(i) => i,
            _ => anyhow::bail!("ROCm graphs require a Rocm device"),
        };
        let key = RocmGraphKey::new(paged_cache, seq_len);
        let cache_key = RocmGraphCacheKey::new(owner, key.clone());
        let pre_candidate_headroom_timer = self
            .phase_telemetry
            .timer(RocmGraphPhase::PreCandidateHeadroom);

        // Pressure may have changed since the decode entry point checked it.
        // Only Comfortable permits fresh candidate allocations; Moderate keeps
        // replay available but stops cache growth, and tighter states use their
        // normal settled eviction/fallback policy.
        match self.reconcile_memory_pressure(owner)? {
            RocmGraphPressureDecision::Normal => {}
            RocmGraphPressureDecision::ReplayOnly(reason)
            | RocmGraphPressureDecision::EagerOnly(reason) => {
                return Ok(RocmCaptureStep::fallback(reason));
            }
        }
        if let Some(reason) = self.reservation_retry_suppressed(&key) {
            return Ok(RocmCaptureStep::fallback(reason));
        }
        if let Some(reason) = self.reserve_capture_entry_capacity(&cache_key)? {
            return Ok(RocmCaptureStep::fallback(reason));
        }
        drop(pre_candidate_headroom_timer);
        let candidate_warm_timer = self.phase_telemetry.timer(RocmGraphPhase::CandidateWarm);

        // Capture on a FRESH non-default stream (mirror the CUDA discipline; the
        // graph-capture stream scope routes every kt op onto it).
        let context = kiln_tensor::primary_rocm_context(device_idx)
            .context("ROCm graph capture: primary_rocm_context for capture stream")?;
        let stream = context
            .new_stream()
            .map_err(|e| anyhow::anyhow!("ROCm graph capture: create capture stream: {e}"))?;
        // Acquire before the warm pass so success, fallback, and failure paths
        // all release any workspace allocated for this fresh private stream.
        let blaslt_workspace_lease =
            kiln_tensor::rocm_blaslt_workspace_lease(device_idx, &context, &stream)
                .context("ROCm graph capture: lease hipBLASLt workspace")?;
        let default_stream = context.default_stream();
        let replay_inputs_ready_event = context
            .new_event()
            .map_err(|e| anyhow::anyhow!("ROCm graph capture: create input event: {e}"))?;
        let replay_complete_event = context
            .new_event()
            .map_err(|e| anyhow::anyhow!("ROCm graph capture: create completion event: {e}"))?;

        // Pre-allocate graph-stable buffers before capture.
        let token_buffer = Self::new_token_buffer(device, token_id)?;
        let position_buffer = Self::new_position_buffer(device, seq_len)?;
        let output_hidden = Self::new_output_hidden(config, device, dtype)?;
        let rotary_cos_buffer = Self::new_rotary_cos_buffer(config, device, seq_len)?;
        let rotary_sin_buffer = Self::new_rotary_sin_buffer(config, device, seq_len)?;
        let block_table_buffer = Some(Self::new_block_table_buffer(
            block_table,
            paged_cache,
            key.max_seqlen_k,
            device,
        )?);
        let seqused_k_buffer = Some(Self::new_seqused_k_buffer(device, seq_len + 1)?);
        let kv_slot_buffer = Some(Self::new_kv_slot_buffer(
            block_table,
            paged_cache,
            seq_len,
            device,
        )?);
        let (paged_decode_outputs, paged_decode_lse) =
            Self::new_paged_decode_outputs(config, device, dtype)?;
        let graph_inputs = match (
            block_table_buffer.as_ref(),
            seqused_k_buffer.as_ref(),
            kv_slot_buffer.as_ref(),
        ) {
            (Some(bt), Some(sk), Some(slot)) => Some(PagedDecodeGraphInputs {
                block_table: bt,
                seqused_k: sk,
                kv_slot: slot,
                max_seqlen_k: key.max_seqlen_k,
                rotary_cos: &rotary_cos_buffer,
                rotary_sin: &rotary_sin_buffer,
                attn_out: &paged_decode_outputs[..],
                softmax_lse: &paged_decode_lse[..],
            }),
            _ => None,
        };
        let gdn_decode_outputs = Self::new_gdn_decode_outputs(config, device)?;
        Self::prepare_gdn_recurrent_state_for_capture(linear_state)?;
        let stable_graph_io_bytes = graph_tensor_bytes(
            [
                &token_buffer,
                &position_buffer,
                &output_hidden,
                &rotary_cos_buffer,
                &rotary_sin_buffer,
            ]
            .into_iter()
            .chain(
                [
                    block_table_buffer.as_ref(),
                    seqused_k_buffer.as_ref(),
                    kv_slot_buffer.as_ref(),
                ]
                .into_iter()
                .flatten(),
            )
            .chain(paged_decode_outputs.iter())
            .chain(paged_decode_lse.iter())
            .chain(gdn_decode_outputs.iter()),
        );

        // === freeze-pointers Pass 1 (Record / warm) ===
        let arena_ctx = kiln_tensor::primary_rocm_context(device_idx)
            .context("freeze-pointers: primary_rocm_context for capture arena")?;
        let arena = std::rc::Rc::new(std::cell::RefCell::new(
            kiln_tensor::RocmCaptureArena::new_record(arena_ctx, device_idx),
        ));
        let gdn_snapshot = linear_state
            .snapshot()
            .context("freeze-pointers: snapshot GDN recurrent state before warm pass")?;
        // The graph-stable input buffers and GDN snapshot above were filled on
        // the kt default stream. The warm pass deliberately runs on the SAME
        // non-default stream that will be captured so hipBLASLt's per-stream
        // workspace is allocated before `hipStreamBeginCapture`; sync first so
        // that stream sees initialized inputs/state.
        attributed_rocm_graph_synchronize(
            "default_inputs_before_warmup",
            "rocm_graph_capture_warmup",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(
                        &default_stream,
                        kiln_tensor::RocmSyncReason::GraphBoundary,
                    )
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        )?;
        // Observe host→device copies issued by this thread to this device
        // across the warm forward. A process-global delta is unsafe here:
        // unrelated inference on another thread or device can advance it. Any
        // host_to_rocm_copy issued by the forward (e.g. a GDN-gates/softplus or
        // KV-write FALLBACK path that allocates a device tensor from host) does a
        // hipStreamSynchronize, which is ILLEGAL inside begin_capture and ABORTS
        // the capture — poisoning the device so even the eager fallback then
        // fails (an empty response under load). The warm pass runs the SAME
        // forward OUTSIDE capture, so if it did a host round-trip the captured
        // pass would too: skip capture for this geometry and fall back to eager
        // BEFORE begin_capture, leaving the device clean.
        let (warm_result, warm_htod) =
            kiln_tensor::with_rocm_htod_observer_detailed(device_idx, || {
                kiln_tensor::with_rocm_capture_arena(arena.clone(), || {
                    // SAFETY: the default input stream was drained above; the warm pass
                    // is settled on `stream` before any buffer can leave this scope.
                    unsafe {
                        kiln_tensor::with_rocm_graph_capture_stream(stream.clone(), || {
                            let hidden = model_forward_paged_hidden_with_graph_inputs(
                                backend,
                                &[token_id],
                                weights,
                                config,
                                paged_cache,
                                block_table,
                                seq_len,
                                Some(linear_state),
                                lora,
                                &token_buffer,
                                &position_buffer,
                                graph_inputs.as_ref(),
                            )?;
                            kiln_tensor::rocm_slice_set_dim0(&output_hidden, &hidden, 0).context(
                                "freeze-pointers warm pass: copy hidden into stable output",
                            )?;
                            Ok::<(), anyhow::Error>(())
                        })
                    }
                })
            });
        let warm_sync_result = attributed_rocm_graph_synchronize(
            "capture_stream_warmup_completion",
            "rocm_graph_capture_warmup",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(&stream, kiln_tensor::RocmSyncReason::GraphBoundary)
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        );
        if let Err(err) = warm_result {
            if let Err(sync_err) = warm_sync_result {
                return fail_closed_after_rocm_warmup(
                    weights,
                    sync_err.context(format!(
                        "capture stream synchronization failed after warm-forward failure ({err:#})"
                    )),
                );
            }
            Self::restore_linear_state_after_execution(
                &context,
                linear_state,
                &gdn_snapshot,
                "restore graph-slot GDN state after warm-forward failure",
            )?;
            if crate::forward::is_rocm_graph_shape_dependent_attention(&err) {
                self.non_capture_safe
                    .insert(key, RocmGraphFallbackReason::ShapeDependentAttention);
                drop(candidate_warm_timer);
                return Ok(RocmCaptureStep::fallback_after_candidate(
                    RocmGraphFallbackReason::ShapeDependentAttention,
                    &self.phase_telemetry,
                ));
            }
            return Err(err).context("freeze-pointers warm (Record) pass failed");
        }
        if let Err(sync_err) = warm_sync_result {
            return fail_closed_after_rocm_warmup(
                weights,
                sync_err.context("capture stream synchronization failed after warm forward"),
            );
        }
        // Restore values without replacing the graph slot's tensor handles.
        // Captured graphs retain these exact addresses across request reuse.
        Self::restore_linear_state_after_execution(
            &context,
            linear_state,
            &gdn_snapshot,
            "restore graph-slot GDN state after warm pass",
        )?;
        drop(gdn_snapshot);
        drop(candidate_warm_timer);
        let pre_native_reservation_timer = self
            .phase_telemetry
            .timer(RocmGraphPhase::PreNativeReservation);

        // The Record pass is the first point where every transient candidate
        // allocation has an exact physical identity, including candidates that
        // are about to defer because their warm forward performed a host copy.
        let reserved_stable_io = unique_rocm_tensor_allocations(
            [
                &token_buffer,
                &position_buffer,
                &output_hidden,
                &rotary_cos_buffer,
                &rotary_sin_buffer,
            ]
            .into_iter()
            .chain(
                [
                    block_table_buffer.as_ref(),
                    seqused_k_buffer.as_ref(),
                    kv_slot_buffer.as_ref(),
                ]
                .into_iter()
                .flatten(),
            )
            .chain(paged_decode_outputs.iter())
            .chain(paged_decode_lse.iter())
            .chain(gdn_decode_outputs.iter()),
        )
        .context("measure ROCm graph-stable direct allocations")?;
        let reserved_capture_arena = {
            let arena = arena.borrow();
            unique_rocm_storage_allocations(device_idx, arena.retained_buffers())
        };
        let reserved_workspace_stats = blaslt_workspace_lease
            .stats()
            .map_err(|error| anyhow::anyhow!(error))
            .context("measure ROCm graph private-stream hipBLASLt workspace")?;
        let reserved_blaslt_workspace =
            (reserved_workspace_stats.retained_bytes > 0).then_some(RocmAllocationRecord {
                key: RocmAllocationKey {
                    device_index: device_idx,
                    allocation_id: reserved_workspace_stats.allocation_id as u64,
                },
                bytes: reserved_workspace_stats.retained_bytes,
            });
        let reserved_accounting = RocmGraphEntryAccounting {
            stable_io: reserved_stable_io,
            capture_arena: reserved_capture_arena,
            blaslt_workspace: reserved_blaslt_workspace,
        };
        let transient_candidate_bytes = reserved_accounting.retained_bytes_excluding_slot();
        self.phase_telemetry
            .record_transient_candidate_bytes(transient_candidate_bytes);
        if warm_htod.copy_count > 0 {
            // The warm forward did a host round-trip. This is EITHER a one-time
            // cold-cache fill (shape-keyed broadcast/gqa-expand gather indices
            // upload once per `max_seqlen_k` bucket, then every step + replay in
            // that bucket reuses the device buffer) OR a genuine per-step fallback
            // (e.g. a GDN-gates/softplus path that rebuilds host data every step).
            // The two are indistinguishable from a single warm pass, so retry: a
            // cold fill is absorbed by THIS pass and the next attempt for the same
            // (bucket-stable) geometry sees htod==0 and captures; a real per-step
            // round-trip keeps bumping htod and exhausts the retry budget. The warm
            // pass advanced then restored the recurrent state, and begin_capture was
            // never called, so the device stays clean either way.
            let attempts = self.capture_retry.entry(key.clone()).or_insert(0);
            *attempts += 1;
            let fallback_reason = if *attempts >= Self::CAPTURE_RETRY_LIMIT {
                tracing::debug!(
                    htod = warm_htod.copy_count,
                    attempts = *attempts,
                    "ROCm graph: geometry not capture-safe (persistent host round-trip); \
                     caching skip + running eager"
                );
                self.non_capture_safe.insert(
                    key.clone(),
                    RocmGraphFallbackReason::PersistentHostRoundTrip,
                );
                self.capture_retry.remove(&key);
                RocmGraphFallbackReason::PersistentHostRoundTrip
            } else {
                tracing::debug!(
                    htod = warm_htod.copy_count,
                    attempts = *attempts,
                    "ROCm graph: warm pass did a host round-trip (likely cold cache fill); \
                     running eager, will retry capture next step"
                );
                RocmGraphFallbackReason::ColdCacheHostRoundTrip
            };
            for site in &warm_htod.sites {
                tracing::warn!(
                    event = "rocm_graph_capture_host_transfer",
                    reason = fallback_reason.as_str(),
                    source_file = site.source_file,
                    source_line = site.source_line,
                    source_column = site.source_column,
                    dtype = %site.dtype,
                    elements_per_copy = site.elements_per_copy,
                    bytes_per_copy = site.bytes_per_copy,
                    copy_count = site.copy_count,
                    total_bytes = site.total_bytes,
                    "ROCm graph capture-safety observer attributed a host-to-device transfer"
                );
            }
            if warm_htod.unattributed_copy_count > 0 {
                tracing::warn!(
                    event = "rocm_graph_capture_host_transfer",
                    reason = fallback_reason.as_str(),
                    source_file = "bounded_site_overflow",
                    source_line = 0,
                    source_column = 0,
                    dtype = "mixed",
                    elements_per_copy = 0,
                    bytes_per_copy = 0,
                    copy_count = warm_htod.unattributed_copy_count,
                    total_bytes = warm_htod.unattributed_bytes,
                    "ROCm graph capture-safety observer omitted unique host-to-device transfer sites"
                );
            }
            drop(pre_native_reservation_timer);
            return Ok(RocmCaptureStep::fallback_after_candidate(
                fallback_reason,
                &self.phase_telemetry,
            ));
        }
        // Capture-safe: clear any retry bookkeeping for this geometry.
        self.capture_retry.remove(&key);

        match self.reconcile_memory_pressure(owner)? {
            RocmGraphPressureDecision::Normal => {}
            RocmGraphPressureDecision::ReplayOnly(reason)
            | RocmGraphPressureDecision::EagerOnly(reason) => {
                drop(pre_native_reservation_timer);
                return Ok(RocmCaptureStep::fallback_after_candidate(
                    reason,
                    &self.phase_telemetry,
                ));
            }
        }

        // Reserve retained-cache headroom now, before `begin_capture`. The warm
        // buffers are already physical, so the governor reservation below is a
        // conservative concurrent-planner debt until this attempt publishes or
        // drops them; it can double-count a concurrently refreshed snapshot.
        if !self.matching_memory_governor() {
            self.counters.memory_governor_selector_mismatch_skips = self
                .counters
                .memory_governor_selector_mismatch_skips
                .saturating_add(1);
            drop(pre_native_reservation_timer);
            return Ok(RocmCaptureStep::fallback_after_candidate(
                RocmGraphFallbackReason::MemoryGovernorSelectorMismatch,
                &self.phase_telemetry,
            ));
        }
        let Some(governor_candidate_reservation) =
            kiln_memory::MemoryGovernor::try_global_cached_reserve(transient_candidate_bytes)
        else {
            self.remember_reservation_denial(&key, transient_candidate_bytes);
            self.counters.pre_capture_memory_reservation_denied_skips = self
                .counters
                .pre_capture_memory_reservation_denied_skips
                .saturating_add(1);
            drop(pre_native_reservation_timer);
            return Ok(RocmCaptureStep::fallback_after_candidate(
                RocmGraphFallbackReason::MemoryReservationDenied,
                &self.phase_telemetry,
            ));
        };
        self.reservation_denied_bytes.remove(&key);
        self.reservation_denied_wide_bytes = None;
        if let Some(reason) = self.reserve_capture_candidate(&cache_key, &reserved_accounting)? {
            drop(pre_native_reservation_timer);
            return Ok(RocmCaptureStep::fallback_after_candidate(
                reason,
                &self.phase_telemetry,
            ));
        }
        drop(pre_native_reservation_timer);

        let mut native_capture_timer =
            Some(self.phase_telemetry.timer(RocmGraphPhase::NativeCapture));
        arena.borrow_mut().begin_replay();
        let capture_snapshot = linear_state
            .snapshot()
            .context("snapshot GDN recurrent state before capture pass")?;
        // Declared after every pointer-bearing capture buffer so its Drop runs
        // first on every error path. Graph/exec handles are moved into the
        // guard as soon as they are created for the same ordering guarantee.
        let mut capture_failure_guard = RocmCaptureFailureGuard::new(context.clone());

        // Capture establishment makes a host-side success/rollback decision,
        // so these one-time waits remain explicit and attributed.
        attributed_rocm_graph_synchronize(
            "default_inputs_before_capture",
            "rocm_graph_capture_begin",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(
                        &default_stream,
                        kiln_tensor::RocmSyncReason::GraphBoundary,
                    )
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        )?;
        attributed_rocm_graph_synchronize(
            "capture_stream_before_begin",
            "rocm_graph_capture_begin",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(&stream, kiln_tensor::RocmSyncReason::GraphBoundary)
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        )?;

        // === capture Pass 2 (Replay arena views, on the capture stream) ===
        let _ = &gdn_decode_outputs;
        stream
            .begin_capture()
            .map_err(|e| anyhow::anyhow!("begin_capture: {e}"))?;
        let capture_result = kiln_tensor::with_rocm_capture_arena(arena.clone(), || {
            // SAFETY: both streams were drained immediately above. Replay uses
            // explicit input-ready and completion events for later launches.
            unsafe {
                kiln_tensor::with_rocm_graph_capture_stream(stream.clone(), || {
                    let hidden = model_forward_paged_hidden_with_graph_inputs(
                        backend,
                        &[token_id],
                        weights,
                        config,
                        paged_cache,
                        block_table,
                        seq_len,
                        Some(linear_state),
                        lora,
                        &token_buffer,
                        &position_buffer,
                        graph_inputs.as_ref(),
                    )?;
                    kiln_tensor::rocm_slice_set_dim0(&output_hidden, &hidden, 0)
                        .context("ROCm graph: copy kt hidden into stable output_hidden")?;
                    Ok::<(), anyhow::Error>(())
                })
            }
        });
        let graph_result = stream.end_capture();
        if let Err(err) = capture_result {
            capture_failure_guard
                .settle_before_rollback()
                .context("settle device before capture-forward rollback")?;
            capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                linear_state,
                &capture_snapshot,
                "restore graph-slot GDN state after capture-forward failure",
            ))?;
            return Err(err).context("forward pass failed during graph capture");
        }
        drop(graph_inputs);

        let graph = match graph_result {
            Ok(graph) => graph,
            Err(err) => {
                capture_failure_guard
                    .settle_before_rollback()
                    .context("settle device before end-capture rollback")?;
                capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                    linear_state,
                    &capture_snapshot,
                    "restore graph-slot GDN state after end-capture failure",
                ))?;
                return Err(anyhow::anyhow!("end_capture failed: {err}"));
            }
        };
        capture_failure_guard.graph = Some(graph);
        let exec = match capture_failure_guard
            .graph
            .as_ref()
            .expect("captured graph installed in failure guard")
            .instantiate()
        {
            Ok(exec) => exec,
            Err(err) => {
                capture_failure_guard
                    .settle_before_rollback()
                    .context("settle device before graph-instantiation rollback")?;
                capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                    linear_state,
                    &capture_snapshot,
                    "restore graph-slot GDN state after graph-instantiation failure",
                ))?;
                return Err(anyhow::anyhow!("instantiate captured graph: {err}"));
            }
        };
        capture_failure_guard.exec = Some(exec);
        tracing::info!(
            "ROCm HIP graph captured for decode ({} layers)",
            config.num_layers
        );

        // Stream capture only RECORDED the forward; launch once now to actually
        // compute this step + advance state, then sync so output_hidden is valid.
        if let Err(err) = capture_failure_guard
            .exec
            .as_ref()
            .expect("captured graph exec installed in failure guard")
            .launch(&stream)
        {
            capture_failure_guard
                .settle_before_rollback()
                .context("settle device before first-launch rollback")?;
            capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                linear_state,
                &capture_snapshot,
                "restore graph-slot GDN state after first-launch failure",
            ))?;
            return Err(anyhow::anyhow!(
                "execute captured decode graph (first run): {err}"
            ));
        }
        if let Err(err) = attributed_rocm_graph_synchronize(
            "first_launch_completion",
            "rocm_graph_first_launch",
            stable_graph_io_bytes,
            "stable_graph_io",
            || {
                context
                    .synchronize_stream_for(&stream, kiln_tensor::RocmSyncReason::GraphBoundary)
                    .map_err(|error| anyhow::anyhow!("{error}"))
            },
        ) {
            capture_failure_guard
                .settle_before_rollback()
                .context("settle device after first-launch stream wait failed")?;
            capture_failure_guard.complete_rollback(Self::restore_linear_state_in_place(
                linear_state,
                &capture_snapshot,
                "restore graph-slot GDN state after first-launch synchronization failure",
            ))?;
            return Err(anyhow::anyhow!(
                "sync after first captured-graph launch: {err}"
            ));
        }
        let captured_hidden = output_hidden.clone();
        let max_seqlen_k = key.max_seqlen_k;
        let arena_buffers = arena.borrow_mut().take_retained();
        let workspace_stats = blaslt_workspace_lease
            .stats()
            .map_err(|error| anyhow::anyhow!(error))
            .context("account ROCm graph private-stream hipBLASLt workspace")?;
        let blaslt_workspace =
            (workspace_stats.retained_bytes > 0).then_some(RocmAllocationRecord {
                key: RocmAllocationKey {
                    device_index: device_idx,
                    allocation_id: workspace_stats.allocation_id as u64,
                },
                bytes: workspace_stats.retained_bytes,
            });
        let RocmGraphEntryAccounting {
            stable_io,
            capture_arena,
            blaslt_workspace: _,
        } = reserved_accounting;
        let accounting = RocmGraphEntryAccounting {
            stable_io,
            capture_arena,
            blaslt_workspace,
        };
        let replay_state = Self::replay_state_for_capture(
            &key,
            &output_hidden,
            &token_buffer,
            &position_buffer,
            block_table_buffer.as_ref(),
            seqused_k_buffer.as_ref(),
            kv_slot_buffer.as_ref(),
            &rotary_cos_buffer,
            &rotary_sin_buffer,
            &paged_decode_outputs,
            &paged_decode_lse,
            &gdn_decode_outputs,
        );
        let graph = capture_failure_guard
            .graph
            .take()
            .expect("successful capture retains source graph");
        let exec = capture_failure_guard
            .exec
            .take()
            .expect("successful capture retains graph exec");
        capture_failure_guard.disarm();
        let candidate = CapturedDecodeGraphRocm {
            accounting,
            last_used_tick: 0,
            _graph: graph,
            exec,
            output_hidden,
            capture_stream: stream,
            context,
            default_stream,
            replay_inputs_ready_event,
            replay_complete_event,
            adapter_gen: self.adapter_generation,
            kv_pool_identity: paged_cache.pool_identity(),
            token_buffer,
            position_buffer,
            block_table_buffer,
            seqused_k_buffer,
            kv_slot_buffer,
            rotary_cos_buffer,
            rotary_sin_buffer,
            _paged_decode_outputs: paged_decode_outputs,
            _paged_decode_lse: paged_decode_lse,
            max_seqlen_k,
            _gdn_decode_outputs: gdn_decode_outputs,
            _capture_arena_buffers: arena_buffers,
            replay_state,
            _blaslt_workspace_lease: blaslt_workspace_lease,
        };
        let retained =
            self.admit_captured_graph(cache_key, candidate, &mut native_capture_timer)?;
        if retained {
            governor_candidate_reservation.commit_allocated();
        }
        drop(native_capture_timer.take());
        Ok(if retained {
            RocmCaptureStep::CapturedHidden(captured_hidden)
        } else {
            RocmCaptureStep::CapturedHiddenUncached(captured_hidden)
        })
    }

    fn new_token_buffer(device: Device, token_id: u32) -> Result<Tensor> {
        Tensor::from_vec_on(device, vec![token_id], vec![1])
            .context("create ROCm graph token buffer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_capture_rollback_keeps_execution_gate_open() {
        let mut state = RocmCaptureRollbackState::new();
        assert_eq!(
            state.record_settlement(true),
            RocmCaptureGateAction::KeepOpen
        );
        assert_eq!(
            state.record_logical_rollback(true),
            RocmCaptureGateAction::KeepOpen
        );
        assert_eq!(state.exit_action(), RocmCaptureGateAction::KeepOpen);
    }

    #[test]
    fn failed_or_unclassified_capture_rollback_publishes_sticky_stop() {
        let mut logical_failure = RocmCaptureRollbackState::new();
        assert_eq!(
            logical_failure.record_settlement(true),
            RocmCaptureGateAction::KeepOpen
        );
        assert_eq!(
            logical_failure.record_logical_rollback(false),
            RocmCaptureGateAction::PublishStop
        );

        let mut settlement_failure = RocmCaptureRollbackState::new();
        assert_eq!(
            settlement_failure.record_settlement(false),
            RocmCaptureGateAction::PublishStop
        );

        let unclassified_exit = RocmCaptureRollbackState::new();
        assert_eq!(
            unclassified_exit.exit_action(),
            RocmCaptureGateAction::PublishStop
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn graph_attention_geometry_matches_exact_fa2_splits() {
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        for attention_len in [1, kblock_n, kblock_n + 1, 2395, 2560, 2561] {
            let expected = attention_len.div_ceil(kblock_n) * kblock_n;
            assert_eq!(RocmGraphKey::exact_max_seqlen_k(attention_len), expected);
        }
        assert_eq!(RocmGraphKey::exact_max_seqlen_k(2395), 2432);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn batched_graph_key_includes_width_and_max_attention_geometry() {
        let cache = PagedKvCacheKt::new(0, 8, 64, 1, 8, kiln_tensor::DType::BF16, Device::Cpu)
            .expect("metadata-only paged cache");

        let width_two = RocmGraphKey::new_batched(&cache, &[62, 64]).expect("width-two key");
        let width_three = RocmGraphKey::new_batched(&cache, &[1, 64, 3]).expect("width-three key");

        assert_eq!(width_two.batch_size, 2);
        assert_eq!(width_two.max_seqlen_k, 128);
        assert_eq!(width_two.max_blocks_per_seq, 2);
        assert_eq!(width_three.batch_size, 3);
        assert_eq!(width_three.max_seqlen_k, width_two.max_seqlen_k);
        assert_eq!(width_three.max_blocks_per_seq, width_two.max_blocks_per_seq);
        assert_ne!(width_two, width_three);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn retained_byte_accounting_deduplicates_physical_allocations_across_categories() {
        let first = RocmAllocationRecord {
            key: RocmAllocationKey {
                device_index: 0,
                allocation_id: 17,
            },
            bytes: 64,
        };
        let second = RocmAllocationRecord {
            key: RocmAllocationKey {
                device_index: 0,
                allocation_id: 18,
            },
            bytes: 128,
        };
        let mut seen = HashSet::new();
        let mut accounting = RocmGraphMemoryAccounting {
            complete: true,
            ..RocmGraphMemoryAccounting::default()
        };

        RocmGraphMemoryAccounting::add_record(&mut accounting.stable_io_bytes, &mut seen, first);
        RocmGraphMemoryAccounting::add_record(
            &mut accounting.capture_arena_bytes,
            &mut seen,
            first,
        );
        RocmGraphMemoryAccounting::add_record(
            &mut accounting.capture_arena_bytes,
            &mut seen,
            second,
        );
        accounting.finish();

        assert_eq!(accounting.stable_io_bytes, 64);
        assert_eq!(accounting.capture_arena_bytes, 128);
        assert_eq!(accounting.retained_bytes, 192);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn idle_owner_lru_is_deterministic_by_access_then_slot() {
        let mut records = [
            (7, 2, RocmGraphOwner::Slot(2)),
            (3, 9, RocmGraphOwner::Slot(9)),
            (3, 4, RocmGraphOwner::Slot(4)),
        ];
        sort_idle_owner_lru(&mut records);
        assert_eq!(
            records.map(|(_, _, owner)| owner),
            [
                RocmGraphOwner::Slot(4),
                RocmGraphOwner::Slot(9),
                RocmGraphOwner::Slot(2),
            ]
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn active_geometry_relief_is_fair_lru_and_retains_one_per_owner() {
        let candidate = RocmGraphOwner::Slot(7);
        let peer = RocmGraphOwner::Slot(8);
        let singleton = RocmGraphOwner::Slot(9);
        let inactive = RocmGraphOwner::Slot(10);
        let candidate_oldest = RocmGraphCacheKey::new(
            candidate,
            RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 128,
                max_blocks_per_seq: 2,
            },
        );
        let candidate_middle = RocmGraphCacheKey::new(
            candidate,
            RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 256,
                max_blocks_per_seq: 4,
            },
        );
        let candidate_newest = RocmGraphCacheKey::new(
            candidate,
            RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 512,
                max_blocks_per_seq: 8,
            },
        );
        let peer_oldest = RocmGraphCacheKey::new(
            peer,
            RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 128,
                max_blocks_per_seq: 2,
            },
        );
        let peer_newest = RocmGraphCacheKey::new(
            peer,
            RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 256,
                max_blocks_per_seq: 4,
            },
        );
        let singleton_key = RocmGraphCacheKey::new(
            singleton,
            RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 128,
                max_blocks_per_seq: 2,
            },
        );
        let inactive_key = RocmGraphCacheKey::new(
            inactive,
            RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 64,
                max_blocks_per_seq: 1,
            },
        );
        let geometries = [
            (&candidate_newest, 9),
            (&peer_newest, 8),
            (&singleton_key, 3),
            (&candidate_middle, 4),
            (&peer_oldest, 2),
            (&candidate_oldest, 1),
            (&inactive_key, 0),
        ];
        assert_eq!(
            fair_active_geometry_eviction_order(
                candidate,
                [candidate, peer, singleton],
                geometries.into_iter(),
            ),
            vec![
                candidate_oldest,
                candidate_middle,
                peer_oldest,
                candidate_newest,
            ]
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn graphless_candidate_uses_surplus_active_geometry_without_starvation() {
        let first = RocmGraphOwner::Slot(1);
        let second = RocmGraphOwner::Slot(2);
        let third = RocmGraphOwner::Slot(3);
        let candidate = RocmGraphOwner::Slot(4);
        let key = |owner, max_seqlen_k, tick| {
            (
                RocmGraphCacheKey::new(
                    owner,
                    RocmGraphKey {
                        batch_size: 1,
                        max_seqlen_k,
                        max_blocks_per_seq: max_seqlen_k / 64,
                    },
                ),
                tick,
            )
        };
        let entries = [
            key(first, 64, 1),
            key(first, 128, 4),
            key(first, 192, 9),
            key(second, 64, 2),
            key(second, 128, 8),
            key(third, 64, 3),
        ];
        let order = fair_active_geometry_eviction_order(
            candidate,
            [first, second, third, candidate],
            entries.iter().map(|(key, tick)| (key, *tick)),
        );
        assert_eq!(
            order,
            vec![
                entries[0].0.clone(),
                entries[3].0.clone(),
                entries[1].0.clone()
            ]
        );
        assert!(!order.contains(&entries[2].0));
        assert!(!order.contains(&entries[4].0));
        assert!(!order.contains(&entries[5].0));
    }

    #[test]
    fn graph_slot_restore_preserves_tensor_handles() {
        let mut state = LinearAttentionState {
            recurrent_states: vec![Tensor::from_vec(vec![1.0f32, 2.0], vec![1, 2]).unwrap()],
            conv_states: vec![Tensor::from_vec(vec![3.0f32, 4.0], vec![1, 2]).unwrap()],
        };
        let snapshot = LinearAttentionState {
            recurrent_states: vec![Tensor::from_vec(vec![5.0f32, 6.0], vec![1, 2]).unwrap()],
            conv_states: vec![Tensor::from_vec(vec![7.0f32, 8.0], vec![1, 2]).unwrap()],
        };
        let recurrent_id = state.recurrent_states[0].id();
        let conv_id = state.conv_states[0].id();

        RocmGraphRunner::restore_linear_state_in_place(
            &mut state,
            &snapshot,
            "test graph-slot restore",
        )
        .unwrap();

        assert_eq!(state.recurrent_states[0].id(), recurrent_id);
        assert_eq!(state.conv_states[0].id(), conv_id);
        assert_eq!(
            state.recurrent_states[0].to_vec2::<f32>().unwrap(),
            vec![vec![5.0, 6.0]]
        );
        assert_eq!(
            state.conv_states[0].to_vec2::<f32>().unwrap(),
            vec![vec![7.0, 8.0]]
        );
    }

    #[cfg(feature = "rocm")]
    fn rocm_graph_test_fixture(
        device: &Device,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> (ModelConfig, GpuWeights) {
        use crate::forward::{
            GpuAttentionWeights, GpuFfnWeights, GpuFullAttentionWeights, GpuLayerWeights,
        };

        let config = ModelConfig {
            hidden_size: num_attention_heads * head_dim,
            num_layers: 1,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            intermediate_size: num_attention_heads * head_dim * 2,
            vocab_size: 32,
            max_position_embeddings: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            dtype: kiln_core::config::DType::BF16,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: 0,
            linear_key_head_dim: 0,
            linear_num_value_heads: 0,
            linear_value_head_dim: 0,
            linear_conv_kernel_dim: 0,
            partial_rotary_factor: 1.0,
        };
        let random_bf16 = |shape: Vec<usize>| {
            Tensor::randn(0.0_f32, 0.02, shape, device)
                .expect("random ROCm graph fixture tensor")
                .to_dtype(kiln_tensor::DType::BF16)
                .expect("cast ROCm graph fixture tensor")
        };
        let transposed = |tensor: &Tensor| {
            tensor
                .t()
                .expect("transpose ROCm graph fixture tensor")
                .contiguous()
                .expect("materialize ROCm graph fixture transpose")
        };

        let embed_tokens = random_bf16(vec![config.vocab_size, config.hidden_size]);
        let q_proj = random_bf16(vec![
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
        ]);
        let k_proj = random_bf16(vec![
            config.num_kv_heads * config.head_dim,
            config.hidden_size,
        ]);
        let v_proj = random_bf16(vec![
            config.num_kv_heads * config.head_dim,
            config.hidden_size,
        ]);
        let o_proj = random_bf16(vec![
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        ]);
        let gate_proj = random_bf16(vec![config.intermediate_size, config.hidden_size]);
        let up_proj = random_bf16(vec![config.intermediate_size, config.hidden_size]);
        let down_proj = random_bf16(vec![config.hidden_size, config.intermediate_size]);
        let layer = GpuLayerWeights {
            input_layernorm: Tensor::ones(config.hidden_size, kiln_tensor::DType::BF16, device)
                .expect("input norm"),
            post_attention_layernorm: Tensor::ones(
                config.hidden_size,
                kiln_tensor::DType::BF16,
                device,
            )
            .expect("post-attention norm"),
            attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                q_proj_t: transposed(&q_proj),
                k_proj_t: transposed(&k_proj),
                v_proj_t: transposed(&v_proj),
                o_proj_t: transposed(&o_proj),
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm: Tensor::ones(config.head_dim, kiln_tensor::DType::BF16, device)
                    .expect("q norm"),
                k_norm: Tensor::ones(config.head_dim, kiln_tensor::DType::BF16, device)
                    .expect("k norm"),
                qkv_proj_t: None,
                qkv_proj_w8: None,
                o_proj_w8: None,
                q_proj_marlin: None,
            }),
            mlp: GpuFfnWeights {
                gate_proj_t: transposed(&gate_proj),
                up_proj_t: transposed(&up_proj),
                down_proj_t: transposed(&down_proj),
                gate_proj,
                up_proj,
                down_proj,
                gate_up_proj_t: None,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
                gate_up_proj_w8: None,
                down_proj_w8: None,
            },
        };
        let rotary_inv_freq =
            crate::forward::compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, device)
                .expect("rotary frequencies");
        let weights = GpuWeights {
            source_content_sha256: None,
            base_weight_shard_manifest: None,
            execution_provenance: None,
            embed_tokens_t: transposed(&embed_tokens),
            embed_tokens,
            layers: vec![layer],
            final_norm: Tensor::ones(config.hidden_size, kiln_tensor::DType::BF16, device)
                .expect("final norm"),
            rotary_inv_freq,
            lm_head_w8: None,
            mtp: None,
        };
        (config, weights)
    }

    #[cfg(feature = "rocm")]
    fn stale_generation_test_fixture(device: &Device) -> (ModelConfig, GpuWeights) {
        rocm_graph_test_fixture(device, 2, 1, 128)
    }

    #[cfg(feature = "rocm")]
    fn rocm_graph_hybrid_test_fixture(device: &Device) -> (ModelConfig, GpuWeights) {
        use crate::forward::{
            GpuAttentionWeights, GpuFfnWeights, GpuLayerWeights, GpuLinearAttentionWeights,
        };

        let (mut config, mut weights) = stale_generation_test_fixture(device);
        config.num_layers = 2;
        config.num_full_attention_layers = 1;
        config.full_attention_interval = 2;
        config.linear_num_key_heads = config.num_kv_heads;
        config.linear_key_head_dim = config.head_dim;
        config.linear_num_value_heads = config.num_attention_heads;
        config.linear_value_head_dim = config.head_dim;
        config.linear_conv_kernel_dim = 4;

        let random_bf16 = |shape: Vec<usize>| {
            Tensor::randn(0.0_f32, 0.02, shape, device)
                .expect("random ROCm hybrid graph fixture tensor")
                .to_dtype(kiln_tensor::DType::BF16)
                .expect("cast ROCm hybrid graph fixture tensor")
        };
        let transposed = |tensor: &Tensor| {
            tensor
                .t()
                .expect("transpose ROCm hybrid graph fixture tensor")
                .contiguous()
                .expect("materialize ROCm hybrid graph fixture transpose")
        };

        let qkv_dim = config.linear_qkv_dim();
        let value_dim = config.linear_v_dim();
        let value_heads = config.linear_num_value_heads;
        let in_proj_qkv = random_bf16(vec![qkv_dim, config.hidden_size]);
        let in_proj_z = random_bf16(vec![value_dim, config.hidden_size]);
        let out_proj = random_bf16(vec![config.hidden_size, value_dim]);
        let in_proj_a = random_bf16(vec![value_heads, config.hidden_size]);
        let in_proj_b = random_bf16(vec![value_heads, config.hidden_size]);
        let gate_proj = random_bf16(vec![config.intermediate_size, config.hidden_size]);
        let up_proj = random_bf16(vec![config.intermediate_size, config.hidden_size]);
        let down_proj = random_bf16(vec![config.hidden_size, config.intermediate_size]);
        let linear_layer = GpuLayerWeights {
            input_layernorm: Tensor::ones(config.hidden_size, kiln_tensor::DType::BF16, device)
                .expect("hybrid input norm"),
            post_attention_layernorm: Tensor::ones(
                config.hidden_size,
                kiln_tensor::DType::BF16,
                device,
            )
            .expect("hybrid post-attention norm"),
            attention: GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                in_proj_qkv_t: transposed(&in_proj_qkv),
                in_proj_z_t: transposed(&in_proj_z),
                in_proj_a_t: transposed(&in_proj_a),
                in_proj_b_t: transposed(&in_proj_b),
                out_proj_t: transposed(&out_proj),
                in_proj_qkv,
                in_proj_z,
                out_proj,
                in_proj_a,
                in_proj_b,
                conv1d: random_bf16(vec![qkv_dim, 1, config.linear_conv_kernel_dim]),
                norm: Tensor::ones(
                    config.linear_value_head_dim,
                    kiln_tensor::DType::F32,
                    device,
                )
                .expect("hybrid GDN norm"),
                a_log: Tensor::zeros(value_heads, kiln_tensor::DType::F32, *device)
                    .expect("hybrid GDN a_log"),
                a_log_gates: Tensor::zeros(value_heads, kiln_tensor::DType::F32, *device)
                    .expect("hybrid GDN a_log gates"),
                dt_bias: Tensor::zeros(value_heads, kiln_tensor::DType::BF16, *device)
                    .expect("hybrid GDN dt bias"),
                in_proj_ab_t: None,
                out_proj_marlin: None,
                in_proj_qkvzab_w8: None,
            }),
            mlp: GpuFfnWeights {
                gate_proj_t: transposed(&gate_proj),
                up_proj_t: transposed(&up_proj),
                down_proj_t: transposed(&down_proj),
                gate_proj,
                up_proj,
                down_proj,
                gate_up_proj_t: None,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
                gate_up_proj_w8: None,
                down_proj_w8: None,
            },
        };
        weights.layers.insert(0, linear_layer);
        (config, weights)
    }

    #[cfg(feature = "rocm")]
    fn rocm_graph_qwen_depth_test_fixture(device: &Device) -> (ModelConfig, GpuWeights) {
        let (mut config, mut weights) = rocm_graph_hybrid_test_fixture(device);

        config.num_layers = 32;
        config.num_full_attention_layers = 8;
        config.full_attention_interval = 4;
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            // Real checkpoint layers have distinct allocations. Building each
            // small synthetic layer independently makes capture preserve 32
            // different weight-pointer sets instead of replaying aliases of
            // one GDN and one full-attention layer.
            let (_, mut source) = rocm_graph_hybrid_test_fixture(device);
            let mut layer = if config.is_full_attention_layer(layer_idx) {
                source.layers.remove(1)
            } else {
                source.layers.remove(0)
            };

            let gate_up_rows = Tensor::cat(&[&layer.mlp.gate_proj, &layer.mlp.up_proj], 0)
                .expect("concatenate ROCm graph fixture MLP rows")
                .contiguous()
                .expect("materialize ROCm graph fixture MLP rows");
            layer.mlp.gate_up_proj_w8 = crate::rocm_w8_proj::pack_from_bf16_rows(&gate_up_rows)
                .expect("pack ROCm graph fixture gate/up rows");
            layer.mlp.down_proj_w8 = crate::rocm_w8_proj::pack_from_bf16_rows(&layer.mlp.down_proj)
                .expect("pack ROCm graph fixture down rows");

            match &mut layer.attention {
                crate::forward::GpuAttentionWeights::Full(full) => {
                    let qkv_rows = Tensor::cat(&[&full.q_proj, &full.k_proj, &full.v_proj], 0)
                        .expect("concatenate ROCm graph fixture QKV rows")
                        .contiguous()
                        .expect("materialize ROCm graph fixture QKV rows");
                    full.qkv_proj_w8 = crate::rocm_w8_proj::pack_from_bf16_rows(&qkv_rows)
                        .expect("pack ROCm graph fixture QKV rows");
                    full.o_proj_w8 = crate::rocm_w8_proj::pack_from_bf16_rows(&full.o_proj)
                        .expect("pack ROCm graph fixture attention output rows");
                }
                crate::forward::GpuAttentionWeights::Linear(linear) => {
                    let rows = Tensor::cat(
                        &[
                            &linear.in_proj_qkv,
                            &linear.in_proj_z,
                            &linear.in_proj_a,
                            &linear.in_proj_b,
                        ],
                        0,
                    )
                    .expect("concatenate ROCm graph fixture GDN rows")
                    .contiguous()
                    .expect("materialize ROCm graph fixture GDN rows");
                    linear.in_proj_qkvzab_w8 = crate::rocm_w8_proj::pack_from_bf16_rows(&rows)
                        .expect("pack ROCm graph fixture GDN rows");
                }
            }
            layers.push(layer);
        }
        weights.layers = layers;

        (config, weights)
    }

    #[cfg(feature = "rocm")]
    fn hidden_f32(tensor: &Tensor) -> Vec<f32> {
        tensor
            .to_device(Device::Cpu)
            .expect("copy hidden to CPU")
            .to_dtype(kiln_tensor::DType::F32)
            .expect("cast hidden to f32")
            .to_vec()
            .expect("read hidden")
    }

    #[cfg(feature = "rocm")]
    fn configure_rocm_graph_test_memory_governor(device: &Device) {
        let selector = device.memory_probe_selector();
        kiln_memory::MemoryGovernor::configure_global(
            selector,
            kiln_memory::GovernorConfig::default(),
        )
        .expect("configure ROCm graph test memory governor");
        let snapshot = kiln_memory::MemoryGovernor::global().refresh();
        assert!(
            !snapshot.observations.probe_failed,
            "ROCm graph test memory probe must publish a usable snapshot"
        );
        assert_eq!(
            kiln_memory::MemoryGovernor::global_configuration().selector,
            selector,
            "ROCm graph runner and process memory governor must name the same device"
        );
    }

    #[cfg(feature = "rocm")]
    fn require_explicit_rocm_qualification() {
        assert_eq!(
            std::env::var("KILN_QUALIFICATION").ok().as_deref(),
            Some("1"),
            "set KILN_QUALIFICATION=1 for the explicit hardware run"
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn capture_parity_check_is_exact_and_attributes_state_layer() -> Result<()> {
        let device = Device::Cpu;
        let tensor = |values: &[f32]| {
            Tensor::from_vec_on(device, values.to_vec(), vec![1, values.len()])
                .expect("build ROCm capture parity fixture tensor")
        };
        let expected_hidden = tensor(&[1.0, 2.0]);
        let expected_state = LinearAttentionState {
            recurrent_states: vec![tensor(&[3.0, 4.0]), tensor(&[5.0, 6.0])],
            conv_states: vec![tensor(&[7.0, 8.0]), tensor(&[9.0, 10.0])],
        };
        let matching_state = expected_state.snapshot()?;
        let (tensor_copies, tensor_scratch) =
            RocmGraphRunner::capture_parity_tensor_temporary_bytes(
                &expected_hidden,
                &expected_state,
            );
        assert_eq!(
            tensor_copies.saturating_add(tensor_scratch),
            75,
            "two state snapshots, one hidden copy, and the largest U8 comparison mask must be reserved",
        );
        let paged_cache = PagedKvCacheKt::new(2, 1, 8, 2, 3, kiln_tensor::DType::F32, device)?;
        assert_eq!(
            RocmGraphRunner::capture_parity_temporary_bytes(
                &expected_hidden,
                &expected_state,
                &paged_cache,
                4,
            )?,
            577,
            "KV reference rows and one gathered candidate row must share the parity reservation",
        );
        let kv_slots = Tensor::from_vec_on(device, vec![0u32, 2, 4, 6], vec![4])?;
        for layer_idx in 0usize..2 {
            let values = (0usize..48)
                .map(|index| (layer_idx * 100 + index) as f32)
                .collect::<Vec<_>>();
            let keys = Tensor::from_vec_on(device, values.clone(), vec![8, 2, 3])?;
            let values = Tensor::from_vec_on(
                device,
                values.into_iter().map(|value| value + 50.0).collect(),
                vec![8, 2, 3],
            )?;
            let (key_pool, value_pool) = paged_cache
                .pool_tensors(layer_idx)
                .expect("capture parity KV fixture layer");
            key_pool.slice_set(&keys, 0, 0)?;
            value_pool.slice_set(&values, 0, 0)?;
        }
        let expected_kv = RocmGraphRunner::snapshot_paged_kv_slots(&paged_cache, &kv_slots)?;
        assert_eq!(
            RocmGraphRunner::exact_paged_kv_slots_match(&expected_kv, &paged_cache, &kv_slots,)?,
            (None, None),
        );
        let changed_row = Tensor::from_vec_on(device, vec![-1.0f32; 6], vec![1, 2, 3])?;
        let (layer_one_keys, _) = paged_cache
            .pool_tensors(1)
            .expect("capture parity KV fixture layer one");
        layer_one_keys.slice_set(&changed_row, 0, 2)?;
        assert_eq!(
            RocmGraphRunner::exact_paged_kv_slots_match(&expected_kv, &paged_cache, &kv_slots,)?,
            (Some(1), None),
        );
        let (_, layer_zero_values) = paged_cache
            .pool_tensors(0)
            .expect("capture parity KV fixture layer zero");
        layer_zero_values.slice_set(&changed_row, 0, 0)?;
        assert_eq!(
            RocmGraphRunner::exact_paged_kv_slots_match(&expected_kv, &paged_cache, &kv_slots,)?,
            (Some(1), Some(0)),
        );

        assert_eq!(
            RocmGraphRunner::exact_capture_outputs_match(
                &expected_hidden,
                &expected_hidden.copy()?,
                &expected_state,
                &matching_state,
            )?,
            (true, None, None)
        );

        let recurrent_mismatch = LinearAttentionState {
            recurrent_states: vec![tensor(&[3.0, 4.0]), tensor(&[5.0, 6.5])],
            conv_states: vec![tensor(&[7.0, 8.0]), tensor(&[9.0, 10.0])],
        };
        assert_eq!(
            RocmGraphRunner::exact_capture_outputs_match(
                &expected_hidden,
                &tensor(&[1.0, 2.5]),
                &expected_state,
                &recurrent_mismatch,
            )?,
            (false, Some(1), None)
        );
        assert!(
            !RocmGraphRunner::exact_tensor_values_match(
                &tensor(&[f32::NAN]),
                &tensor(&[f32::NAN]),
            )?,
            "NaN parity must fail closed",
        );
        assert!(
            RocmGraphRunner::exact_tensor_values_match(&tensor(&[]), &tensor(&[]))?,
            "empty tensor parity is vacuously true",
        );
        assert!(
            RocmGraphRunner::exact_tensor_values_match(
                &tensor(&[1.0]),
                &tensor(&[1.0]).to_dtype(kiln_tensor::DType::BF16)?,
            )
            .is_err(),
            "dtype drift must not be normalized away",
        );
        Ok(())
    }

    #[test]
    fn graph_policy_defaults_are_eager_and_cache_bounds_are_validated() {
        let default_policy = RocmGraphExecutionPolicy::default();
        assert_eq!(default_policy, RocmGraphExecutionPolicy::disabled());
        assert_eq!(default_policy.mode(), RocmGraphExecutionMode::Disabled);
        assert_eq!(default_policy.mode().as_str(), "disabled");
        assert_eq!(
            default_policy.max_cached_graphs(),
            RocmGraphExecutionPolicy::DEFAULT_MAX_CACHED_GRAPHS
        );
        assert_eq!(
            default_policy.max_retained_bytes(),
            RocmGraphExecutionPolicy::DEFAULT_MAX_RETAINED_BYTES
        );
        assert!(!default_policy.force_eager_decode());

        let lazy = RocmGraphExecutionPolicy::try_new(
            RocmGraphExecutionMode::LazyCaptureReplay,
            16,
            RocmGraphExecutionPolicy::MIN_MAX_RETAINED_BYTES,
            true,
        )
        .expect("bounded graph cache policy");
        assert_eq!(lazy.mode(), RocmGraphExecutionMode::LazyCaptureReplay);
        assert_eq!(lazy.max_cached_graphs(), 16);
        assert_eq!(
            lazy.max_retained_bytes(),
            RocmGraphExecutionPolicy::MIN_MAX_RETAINED_BYTES
        );
        assert!(lazy.force_eager_decode());

        assert!(
            RocmGraphExecutionPolicy::try_new(
                RocmGraphExecutionMode::WarmupThenEager,
                0,
                RocmGraphExecutionPolicy::DEFAULT_MAX_RETAINED_BYTES,
                false,
            )
            .is_err(),
            "zero-capacity graph caches must fail during config resolution"
        );
        assert!(
            RocmGraphExecutionPolicy::try_new(
                RocmGraphExecutionMode::LazyCaptureReplay,
                RocmGraphExecutionPolicy::MAX_CACHED_GRAPHS,
                RocmGraphExecutionPolicy::DEFAULT_MAX_RETAINED_BYTES,
                false,
            )
            .is_ok(),
            "the documented upper bound must be accepted"
        );
        assert!(
            RocmGraphExecutionPolicy::try_new(
                RocmGraphExecutionMode::LazyCaptureReplay,
                RocmGraphExecutionPolicy::MAX_CACHED_GRAPHS + 1,
                RocmGraphExecutionPolicy::DEFAULT_MAX_RETAINED_BYTES,
                false,
            )
            .is_err(),
            "embedding callers cannot construct an unbounded graph cache"
        );
        assert!(
            RocmGraphExecutionPolicy::try_new(
                RocmGraphExecutionMode::LazyCaptureReplay,
                1,
                RocmGraphExecutionPolicy::MIN_MAX_RETAINED_BYTES - 1,
                false,
            )
            .is_err(),
            "retained-byte budgets below the documented floor must fail"
        );
        assert!(
            RocmGraphExecutionPolicy::try_new(
                RocmGraphExecutionMode::LazyCaptureReplay,
                1,
                RocmGraphExecutionPolicy::MAX_MAX_RETAINED_BYTES,
                false,
            )
            .is_ok(),
            "the documented retained-byte ceiling must be accepted"
        );
        assert!(
            RocmGraphExecutionPolicy::try_new(
                RocmGraphExecutionMode::LazyCaptureReplay,
                1,
                RocmGraphExecutionPolicy::MAX_MAX_RETAINED_BYTES + 1,
                false,
            )
            .is_err(),
            "retained-byte budgets above the documented ceiling must fail"
        );
    }

    #[test]
    fn runner_derives_requested_state_from_typed_mode() {
        let device = Device::Rocm(0);

        let disabled = RocmGraphRunner::new(&device, RocmGraphExecutionPolicy::disabled()).stats();
        assert!(!disabled.requested);
        assert!(!disabled.capture_requested);
        assert!(!disabled.enabled);

        let warmup =
            RocmGraphRunner::new(&device, RocmGraphExecutionPolicy::warmup_then_eager()).stats();
        assert!(warmup.requested);
        assert!(!warmup.capture_requested);
        assert!(warmup.enabled);
        assert!(!warmup.capture_enabled);

        let lazy =
            RocmGraphRunner::new(&device, RocmGraphExecutionPolicy::lazy_capture_replay()).stats();
        assert!(lazy.requested);
        #[cfg(feature = "rocm")]
        {
            assert!(lazy.capture_requested);
            assert!(lazy.capture_enabled);
        }
        #[cfg(not(feature = "rocm"))]
        {
            assert!(!lazy.capture_requested);
            assert!(!lazy.capture_enabled);
        }
        assert!(lazy.enabled);
    }

    #[test]
    fn disabled_off_device() {
        let mut r = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        r.release_decode_row(7);
        assert!(!r.is_enabled());
        assert_eq!(
            r.stats(),
            RocmGraphStats {
                requested: false,
                capture_requested: false,
                enabled: false,
                capture_enabled: false,
                max_cached_graphs: RocmGraphExecutionPolicy::DEFAULT_MAX_CACHED_GRAPHS,
                max_retained_bytes: RocmGraphExecutionPolicy::DEFAULT_MAX_RETAINED_BYTES,
                retained_bytes_accounting_complete: true,
                ..RocmGraphStats::default()
            }
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn release_decode_row_discards_only_the_finished_graphless_slot() {
        let target = RocmGraphOwner::Slot(7);
        let survivor = RocmGraphOwner::Slot(8);
        let second_survivor = RocmGraphOwner::Slot(9);
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        for (owner, row_id) in [(target, 7), (survivor, 8), (second_survivor, 9)] {
            runner.graph_slots.insert(
                owner,
                RocmGraphSlotState {
                    assigned_row: Some(row_id),
                    batch_size: None,
                    linear_state: LinearAttentionState {
                        recurrent_states: Vec::new(),
                        conv_states: Vec::new(),
                    },
                    allocations: Vec::new(),
                    accounting_complete: true,
                },
            );
            runner.decode_row_slots.insert(row_id, owner);
        }
        runner.decode_timelines.insert(target, Default::default());
        runner.decode_timelines.insert(survivor, Default::default());
        runner
            .decode_timelines
            .insert(second_survivor, Default::default());
        assert_eq!(runner.stats().tracked_decode_owner_count, 3);
        runner.release_decode_row(7);

        assert!(!runner.graph_slots.contains_key(&target));
        assert!(runner.graph_slots.contains_key(&survivor));
        assert!(runner.graph_slots.contains_key(&second_survivor));
        assert!(!runner.decode_timelines.contains_key(&target));
        assert!(runner.decode_timelines.contains_key(&survivor));
        assert!(runner.decode_timelines.contains_key(&second_survivor));
        let stats = runner.stats();
        assert_eq!(stats.tracked_decode_owner_count, 2);
        assert_eq!(stats.graph_slot_count, 2);
        assert_eq!(stats.active_graph_slot_count, 2);
        assert_eq!(stats.decode_owner_release_count, 1);
        assert_eq!(stats.decode_owner_graph_release_count, 0);

        runner.release_decode_row(7);
        let stats = runner.stats();
        assert_eq!(stats.tracked_decode_owner_count, 2);
        assert_eq!(stats.decode_owner_release_count, 1);
        assert_eq!(stats.decode_owner_graph_release_count, 0);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn active_graphless_slot_stays_in_stats_and_budget_after_graph_eviction() {
        let owner = RocmGraphOwner::Slot(41);
        let budget = RocmGraphExecutionPolicy::MIN_MAX_RETAINED_BYTES;
        let policy = RocmGraphExecutionPolicy::lazy_capture_replay()
            .with_max_retained_bytes(budget)
            .expect("minimum retained-byte budget");
        let mut runner = RocmGraphRunner::new(&Device::Cpu, policy);
        runner.graph_slots.insert(
            owner,
            RocmGraphSlotState {
                assigned_row: Some(9),
                batch_size: None,
                linear_state: LinearAttentionState {
                    recurrent_states: Vec::new(),
                    conv_states: Vec::new(),
                },
                allocations: vec![RocmAllocationRecord {
                    key: RocmAllocationKey {
                        device_index: 0,
                        allocation_id: 99,
                    },
                    bytes: budget,
                }],
                accounting_complete: true,
            },
        );
        runner.decode_row_slots.insert(9, owner);

        let before = runner.stats();
        assert_eq!(before.captured_graph_count, 0);
        assert_eq!(before.active_graph_slot_count, 1);
        assert_eq!(before.retained_slot_state_bytes, budget);
        assert_eq!(before.retained_bytes, budget);
        assert!(before.peak_retained_bytes >= before.retained_bytes);
        assert!(before.retained_bytes_accounting_complete);

        let owners = HashSet::from([owner]);
        let released = runner
            .evict_graph_owners(
                &owners,
                "active_slot_accounting_test",
                RocmGraphEvictionReason::Pressure,
                false,
            )
            .expect("graph-only eviction projection");
        assert_eq!(released.retained_bytes, 0);

        let after = runner.stats();
        assert_eq!(after.active_graph_slot_count, 1);
        assert_eq!(after.retained_slot_state_bytes, budget);
        assert_eq!(after.retained_bytes, budget);
        assert!(after.peak_retained_bytes >= after.retained_bytes);
        let requested = RocmGraphCacheKey::new(
            owner,
            RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 512,
                max_blocks_per_seq: 8,
            },
        );
        assert_eq!(
            runner.pre_capture_rejection(&requested),
            Some(RocmGraphFallbackReason::GraphCacheByteBudget)
        );
        assert_eq!(runner.counters.byte_budget_rejections, 0);
        assert_eq!(runner.counters.pre_capture_byte_budget_skips, 1);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn oversized_slot_is_memoized_until_row_release_without_reopening_budget_generation() {
        let budget = RocmGraphExecutionPolicy::MIN_MAX_RETAINED_BYTES;
        let policy = RocmGraphExecutionPolicy::lazy_capture_replay()
            .with_max_retained_bytes(budget)
            .expect("minimum retained-byte budget");
        let mut runner = RocmGraphRunner::new(&Device::Cpu, policy);
        let row_id = 71;
        let declined_owner = RocmGraphOwner::Slot(99);
        let empty_state = LinearAttentionState {
            recurrent_states: Vec::new(),
            conv_states: Vec::new(),
        };
        let reason = runner
            .publish_new_graph_slot(
                row_id,
                declined_owner,
                empty_state,
                vec![RocmAllocationRecord {
                    key: RocmAllocationKey {
                        device_index: 0,
                        allocation_id: 7001,
                    },
                    bytes: budget + 1,
                }],
                true,
            )
            .expect_err("oversized slot must be declined before publication");
        assert_eq!(reason, RocmGraphFallbackReason::GraphCacheByteBudget);
        assert!(!runner.graph_slots.contains_key(&declined_owner));
        assert!(!runner.decode_row_slots.contains_key(&row_id));
        assert_eq!(runner.counters.pre_capture_byte_budget_skips, 1);

        let key = RocmGraphKey {
            batch_size: 1,
            max_seqlen_k: 512,
            max_blocks_per_seq: 8,
        };
        let mut retry_state = LinearAttentionState {
            recurrent_states: Vec::new(),
            conv_states: Vec::new(),
        };
        assert_eq!(
            runner
                .bind_decode_row_to_slot(row_id, &key, &mut retry_state)
                .expect("memoized bind"),
            RocmGraphBindOutcome::Fallback(RocmGraphFallbackReason::GraphCacheByteBudget)
        );
        assert_eq!(runner.counters.pre_capture_byte_budget_skips, 1);

        let relief_generation = runner.budget_relief_generation;
        runner.release_decode_row(row_id);
        assert!(!runner.graph_ineligible_rows.contains_key(&row_id));
        assert_eq!(runner.budget_relief_generation, relief_generation);
        assert!(matches!(
            runner
                .bind_decode_row_to_slot(row_id, &key, &mut retry_state)
                .expect("released row id can retry"),
            RocmGraphBindOutcome::Bound(_)
        ));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn incomplete_slot_accounting_has_distinct_typed_skip() {
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        let row_id = 81;
        let owner = RocmGraphOwner::Slot(101);
        let reason = runner
            .publish_new_graph_slot(
                row_id,
                owner,
                LinearAttentionState {
                    recurrent_states: Vec::new(),
                    conv_states: Vec::new(),
                },
                Vec::new(),
                false,
            )
            .expect_err("incomplete accounting must fail closed");
        assert_eq!(reason, RocmGraphFallbackReason::GraphAccountingIncomplete);
        assert_eq!(
            runner.graph_ineligible_rows.get(&row_id),
            Some(&RocmGraphFallbackReason::GraphAccountingIncomplete)
        );
        assert_eq!(runner.counters.accounting_incomplete_rejections, 0);
        assert_eq!(runner.counters.pre_capture_accounting_incomplete_skips, 1);
        assert!(!runner.graph_slots.contains_key(&owner));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn candidate_alone_byte_rejection_suppresses_repeat_capture_for_geometry() {
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        let graph = RocmGraphKey {
            batch_size: 1,
            max_seqlen_k: 512,
            max_blocks_per_seq: 8,
        };
        let key = RocmGraphCacheKey::new(RocmGraphOwner::Slot(7), graph.clone());

        assert!(
            runner.record_post_capture_rejection(
                &key,
                RocmGraphAdmissionRejection::CandidateByteBudget,
            )
        );
        assert_eq!(
            runner.non_capture_safe.get(&graph),
            Some(&RocmGraphFallbackReason::GraphCacheByteBudget)
        );
        assert_eq!(runner.counters.byte_budget_rejections, 1);
        assert_eq!(runner.counters.entry_capacity_rejections, 0);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn exact_pre_native_plan_deduplicates_candidate_and_owner_slot() {
        let budget = RocmGraphExecutionPolicy::MIN_MAX_RETAINED_BYTES;
        let policy = RocmGraphExecutionPolicy::lazy_capture_replay()
            .with_max_retained_bytes(budget)
            .expect("minimum retained-byte budget");
        let mut runner = RocmGraphRunner::new(&Device::Cpu, policy);
        let owner = RocmGraphOwner::Slot(12);
        let shared = RocmAllocationRecord {
            key: RocmAllocationKey {
                device_index: 0,
                allocation_id: 1200,
            },
            bytes: budget / 2,
        };
        runner.graph_slots.insert(
            owner,
            RocmGraphSlotState {
                assigned_row: Some(12),
                batch_size: None,
                linear_state: LinearAttentionState {
                    recurrent_states: Vec::new(),
                    conv_states: Vec::new(),
                },
                allocations: vec![shared],
                accounting_complete: true,
            },
        );
        let key = RocmGraphCacheKey::new(
            owner,
            RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 512,
                max_blocks_per_seq: 8,
            },
        );
        let candidate = RocmGraphEntryAccounting {
            stable_io: vec![shared],
            capture_arena: vec![RocmAllocationRecord {
                key: RocmAllocationKey {
                    device_index: 0,
                    allocation_id: 1201,
                },
                bytes: budget / 2,
            }],
            blaslt_workspace: None,
        };
        assert_eq!(candidate.retained_bytes_excluding_slot(), budget);
        assert_eq!(
            runner
                .plan_candidate_admission(&key, &candidate)
                .expect("deduplicated candidate fits exactly"),
            RocmGraphAdmissionPlan::default()
        );

        let oversized = RocmGraphEntryAccounting {
            stable_io: vec![RocmAllocationRecord {
                key: RocmAllocationKey {
                    device_index: 0,
                    allocation_id: 1202,
                },
                bytes: budget + 1,
            }],
            ..RocmGraphEntryAccounting::default()
        };
        assert_eq!(
            runner.plan_candidate_admission(&key, &oversized),
            Err(RocmGraphAdmissionRejection::CandidateByteBudget)
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn aggregate_byte_rejection_retries_only_after_owner_release() {
        let owner = RocmGraphOwner::Slot(17);
        let row_id = 23;
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        runner.graph_slots.insert(
            owner,
            RocmGraphSlotState {
                assigned_row: Some(row_id),
                batch_size: None,
                linear_state: LinearAttentionState {
                    recurrent_states: Vec::new(),
                    conv_states: Vec::new(),
                },
                allocations: Vec::new(),
                accounting_complete: true,
            },
        );
        runner.decode_row_slots.insert(row_id, owner);
        let graph = RocmGraphKey {
            batch_size: 1,
            max_seqlen_k: 1024,
            max_blocks_per_seq: 16,
        };
        let key = RocmGraphCacheKey::new(owner, graph.clone());
        let rejected_generation = runner.budget_relief_generation;

        assert!(
            runner.record_post_capture_rejection(&key, RocmGraphAdmissionRejection::ByteBudget)
        );
        assert!(runner.budget_capture_suppressed(&graph));
        assert!(runner.budget_capture_suppressed(&graph));
        assert_eq!(runner.budget_relief_generation, rejected_generation);

        // New binds/admissions consume headroom and must not reopen a geometry
        // denied under the same retained-owner constraints.
        runner.record_ownership_mutation();
        assert_eq!(runner.budget_relief_generation, rejected_generation);
        assert!(runner.budget_capture_suppressed(&graph));

        runner.release_decode_row(row_id);
        assert_ne!(runner.budget_relief_generation, rejected_generation);
        assert!(!runner.budget_capture_suppressed(&graph));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn aggregate_budget_suppression_does_not_cycle_at_cache_entry_cap() {
        let policy = RocmGraphExecutionPolicy::lazy_capture_replay()
            .with_max_cached_graphs(1)
            .expect("one-entry cache policy");
        let mut runner = RocmGraphRunner::new(&Device::Cpu, policy);
        let owner = RocmGraphOwner::Slot(5);
        let mut rejected = Vec::new();

        for bucket in 0..2 {
            let graph = RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 512 + bucket,
                max_blocks_per_seq: 8,
            };
            let key = RocmGraphCacheKey::new(owner, graph.clone());
            assert!(
                runner.record_post_capture_rejection(&key, RocmGraphAdmissionRejection::ByteBudget)
            );
            rejected.push(graph);
        }
        assert!(runner.budget_rejection_generation.len() > runner.max_cached_graphs());
        assert!(
            rejected
                .iter()
                .all(|graph| runner.budget_capture_suppressed(graph))
        );

        for bucket in 2..=RocmGraphRunner::MAX_BUDGET_REJECTION_GEOMETRIES {
            let graph = RocmGraphKey {
                batch_size: 1,
                max_seqlen_k: 512 + bucket,
                max_blocks_per_seq: 8,
            };
            let key = RocmGraphCacheKey::new(owner, graph);
            assert!(
                runner.record_post_capture_rejection(&key, RocmGraphAdmissionRejection::ByteBudget)
            );
        }
        assert_eq!(
            runner.budget_rejection_generation_wide,
            Some(runner.budget_relief_generation)
        );
        let unseen = RocmGraphKey {
            batch_size: 1,
            max_seqlen_k: usize::MAX,
            max_blocks_per_seq: usize::MAX,
        };
        assert!(runner.budget_capture_suppressed(&unseen));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn governor_denial_skips_rewarm_until_measured_headroom_returns() {
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        let graph = RocmGraphKey {
            batch_size: 1,
            max_seqlen_k: 2048,
            max_blocks_per_seq: 32,
        };
        runner.remember_reservation_denial(&graph, 4096);
        assert!(runner.reservation_retry_suppressed_with_available(&graph, Some(1024)));
        assert!(runner.reservation_retry_suppressed_with_available(&graph, Some(4095)));
        assert!(runner.reservation_denied_bytes.contains_key(&graph));
        assert!(!runner.reservation_retry_suppressed_with_available(&graph, Some(4096)));
        assert!(!runner.reservation_denied_bytes.contains_key(&graph));

        for bucket in 0..=RocmGraphRunner::MAX_BUDGET_REJECTION_GEOMETRIES {
            runner.remember_reservation_denial(
                &RocmGraphKey {
                    batch_size: 1,
                    max_seqlen_k: bucket,
                    max_blocks_per_seq: 1,
                },
                bucket as u64 + 1,
            );
        }
        assert!(runner.reservation_denied_bytes.is_empty());
        assert_eq!(
            runner.reservation_denied_wide_bytes,
            Some(RocmGraphRunner::MAX_BUDGET_REJECTION_GEOMETRIES as u64 + 1)
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn recycled_block_continuity_never_crosses_graph_slots() {
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        let recycled_table = BlockTable { blocks: vec![11] };
        let first = RocmGraphOwner::Slot(41);
        let second = RocmGraphOwner::Slot(42);

        assert!(!runner.prepare_owner_decode(first, 401, &recycled_table, 63));
        assert!(runner.prepare_owner_decode(first, 401, &recycled_table, 64));

        // Even though both block zero and sequence continuity match the prior
        // call, a different generation must start a fresh recurrent timeline.
        assert!(!runner.prepare_owner_decode(second, 402, &recycled_table, 65));
        assert!(runner.prepare_owner_decode(second, 402, &recycled_table, 66));

        runner.decode_timelines.remove(&second);
        assert!(!runner.prepare_owner_decode(second, 403, &recycled_table, 67));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn idle_graph_slot_refreshes_and_adopts_a_new_rows_state() {
        fn state(recurrent: [f32; 2], conv: [f32; 2]) -> LinearAttentionState {
            LinearAttentionState {
                recurrent_states: vec![
                    Tensor::from_vec(recurrent.to_vec(), (1, 2)).expect("recurrent state"),
                ],
                conv_states: vec![Tensor::from_vec(conv.to_vec(), (1, 2)).expect("conv state")],
            }
        }

        let key = RocmGraphKey {
            batch_size: 1,
            max_seqlen_k: 512,
            max_blocks_per_seq: 8,
        };
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        let mut first = state([1.0, 2.0], [3.0, 4.0]);
        let owner = RocmGraphOwner::Slot(runner.next_graph_slot_id);
        runner.next_graph_slot_id += 1;
        runner
            .publish_new_graph_slot(
                1001,
                owner,
                RocmGraphRunner::clone_linear_state_handles(&first),
                Vec::new(),
                true,
            )
            .expect("install explicitly accounted test slot");
        runner.decode_row_slots.insert(1001, owner);
        runner.graph_slots.get_mut(&owner).unwrap().assigned_row = Some(1001);
        let owner = match runner
            .bind_decode_row_to_slot(1001, &key, &mut first)
            .expect("bind first row")
        {
            RocmGraphBindOutcome::Bound(owner) => owner,
            RocmGraphBindOutcome::Fallback(reason) => {
                panic!("unexpected first-row graph fallback: {reason:?}")
            }
        };
        assert!(RocmGraphRunner::linear_state_handles_match(
            &runner.graph_slots[&owner].linear_state,
            &first
        ));

        runner.decode_row_slots.remove(&1001);
        runner.graph_slots.get_mut(&owner).unwrap().assigned_row = None;
        let mut second = state([11.0, 12.0], [13.0, 14.0]);
        let rebound = match runner
            .bind_decode_row_to_slot(1002, &key, &mut second)
            .expect("reuse slot for second row")
        {
            RocmGraphBindOutcome::Bound(owner) => owner,
            RocmGraphBindOutcome::Fallback(reason) => {
                panic!("unexpected reused-row graph fallback: {reason:?}")
            }
        };
        assert_eq!(rebound, owner);
        assert!(RocmGraphRunner::linear_state_handles_match(
            &runner.graph_slots[&owner].linear_state,
            &second
        ));
        assert_eq!(
            second.recurrent_states[0].to_vec::<f32>().unwrap(),
            vec![11.0, 12.0]
        );
        assert_eq!(
            second.conv_states[0].to_vec::<f32>().unwrap(),
            vec![13.0, 14.0]
        );
        assert_eq!(runner.stats().graph_slot_create_count, 1);
        assert_eq!(runner.stats().graph_slot_reuse_count, 1);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn batched_graph_slot_refreshes_cohort_without_replacing_handles() {
        fn state(recurrent: [f32; 4], conv: [f32; 4]) -> LinearAttentionState {
            LinearAttentionState {
                recurrent_states: vec![
                    Tensor::from_vec(recurrent.to_vec(), (2, 2)).expect("recurrent state"),
                ],
                conv_states: vec![
                    Tensor::from_vec(conv.to_vec(), (2, 2)).expect("convolution state"),
                ],
            }
        }

        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        let mut first = state([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]);
        let owner = RocmGraphOwner::Slot(runner.next_graph_slot_id);
        runner.next_graph_slot_id += 1;
        runner
            .publish_batched_graph_slot(
                2,
                owner,
                RocmGraphRunner::clone_linear_state_handles(&first),
                Vec::new(),
                true,
            )
            .expect("install explicitly accounted batched test slot");
        assert!(matches!(
            runner
                .bind_batched_state_to_slot(2, &mut first)
                .expect("bind initial cohort"),
            RocmGraphBindOutcome::Bound(bound) if bound == owner
        ));
        let recurrent_id = first.recurrent_states[0].id();
        let conv_id = first.conv_states[0].id();

        let mut second = state([11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]);
        assert!(matches!(
            runner
                .bind_batched_state_to_slot(2, &mut second)
                .expect("refresh changed cohort"),
            RocmGraphBindOutcome::Bound(bound) if bound == owner
        ));

        assert_eq!(second.recurrent_states[0].id(), recurrent_id);
        assert_eq!(second.conv_states[0].id(), conv_id);
        assert_eq!(
            second.recurrent_states[0].to_vec::<f32>().unwrap(),
            vec![11.0, 12.0, 13.0, 14.0]
        );
        assert_eq!(
            second.conv_states[0].to_vec::<f32>().unwrap(),
            vec![15.0, 16.0, 17.0, 18.0]
        );
        assert_eq!(runner.batched_graph_slots.get(&2), Some(&owner));
        let stats = runner.stats();
        assert_eq!(stats.graph_slot_create_count, 1);
        assert_eq!(stats.graph_slot_reuse_count, 0);
        assert_eq!(stats.active_graph_slot_count, 1);
        assert_eq!(stats.idle_graph_slot_count, 0);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn single_row_binding_never_adopts_a_batched_graph_slot() {
        let state = || LinearAttentionState {
            recurrent_states: vec![Tensor::from_vec(vec![1.0f32, 2.0], (1, 2)).unwrap()],
            conv_states: vec![Tensor::from_vec(vec![3.0f32, 4.0], (1, 2)).unwrap()],
        };
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        let batch_owner = RocmGraphOwner::Slot(1);
        runner
            .publish_batched_graph_slot(
                2,
                batch_owner,
                LinearAttentionState {
                    recurrent_states: Vec::new(),
                    conv_states: Vec::new(),
                },
                Vec::new(),
                true,
            )
            .expect("install batched slot");
        let row_owner = RocmGraphOwner::Slot(2);
        runner
            .publish_new_graph_slot(7, row_owner, state(), Vec::new(), true)
            .expect("install row slot");

        let mut row_state = state();
        let key = RocmGraphKey {
            batch_size: 1,
            max_seqlen_k: 64,
            max_blocks_per_seq: 1,
        };
        let bound = runner
            .bind_decode_row_to_slot(7, &key, &mut row_state)
            .expect("bind row");
        assert_eq!(bound, RocmGraphBindOutcome::Bound(row_owner));
        assert_eq!(runner.decode_row_slots.get(&7), Some(&row_owner));
        assert_eq!(runner.batched_graph_slots.get(&2), Some(&batch_owner));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn evicting_a_batched_owner_removes_its_width_mapping() {
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        let owner = RocmGraphOwner::Slot(9);
        runner
            .publish_batched_graph_slot(
                4,
                owner,
                LinearAttentionState {
                    recurrent_states: Vec::new(),
                    conv_states: Vec::new(),
                },
                Vec::new(),
                true,
            )
            .expect("install batched slot");

        runner
            .evict_graph_owners(
                &HashSet::from([owner]),
                "test_batched_slot_eviction",
                RocmGraphEvictionReason::Invalidation,
                true,
            )
            .expect("evict batched owner");

        assert!(!runner.graph_slots.contains_key(&owner));
        assert!(!runner.batched_graph_slots.contains_key(&4));
    }

    /// Real gfx1151 proof that one width-four graph can refresh heterogeneous
    /// row metadata without changing semantics. Every step is compared against
    /// an independent eager cache, including the complete K/V pool contents.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn batched_graph_width_four_matches_eager_hidden_and_kv() -> Result<()> {
        require_explicit_rocm_qualification();
        assert!(
            kiln_tensor::rocm_is_available(),
            "ROCm qualification requested but no ROCm device is available"
        );

        let device = Device::Rocm(0);
        configure_rocm_graph_test_memory_governor(&device);
        let backend = crate::backend::for_device_kt(&device);
        let (config, weights) = stale_generation_test_fixture(&device);
        let graph_cache = PagedKvCacheKt::new(
            1,
            32,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("batched graph paged cache");
        let eager_cache = PagedKvCacheKt::new(
            1,
            32,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("batched eager paged cache");
        let tables = [
            BlockTable { blocks: vec![0, 1] },
            BlockTable { blocks: vec![4, 5] },
            BlockTable { blocks: vec![8, 9] },
            BlockTable {
                blocks: vec![12, 13],
            },
        ];
        let table_refs: Vec<&BlockTable> = tables.iter().collect();
        let row_ids = [4101, 4102, 4103, 4104];
        let mut graph_state = LinearAttentionState {
            recurrent_states: Vec::new(),
            conv_states: Vec::new(),
        };
        let mut eager_state = LinearAttentionState {
            recurrent_states: Vec::new(),
            conv_states: Vec::new(),
        };
        let mut graph_runner =
            RocmGraphRunner::new(&device, RocmGraphExecutionPolicy::lazy_capture_replay());

        for step in 1usize..=24 {
            let token_ids = [
                (step % config.vocab_size) as u32,
                ((step + 5) % config.vocab_size) as u32,
                ((step + 11) % config.vocab_size) as u32,
                ((step + 17) % config.vocab_size) as u32,
            ];
            let sequence_lengths = [step, step + 1, step + 2, step + 3];
            let graph_hidden = graph_runner
                .decode_step_paged_batched_hidden(
                    backend.as_ref(),
                    &token_ids,
                    &weights,
                    &config,
                    &graph_cache,
                    &table_refs,
                    &sequence_lengths,
                    &mut graph_state,
                    None,
                    Some(&row_ids),
                )
                .expect("batched graph decode")
                .expect("lazy ROCm graph runner must own the requested decode");
            let eager_hidden = model_forward_paged_decode_contiguous_batch_hidden_with_ids(
                backend.as_ref(),
                &token_ids,
                &weights,
                &config,
                &eager_cache,
                &table_refs,
                &sequence_lengths,
                Some(&mut eager_state),
                None,
                Some(&row_ids),
            )
            .expect("batched eager oracle");

            let graph_values = hidden_f32(&graph_hidden);
            let eager_values = hidden_f32(&eager_hidden);
            if graph_values != eager_values {
                let first = graph_values
                    .iter()
                    .zip(&eager_values)
                    .enumerate()
                    .find(|(_, (graph, eager))| graph != eager)
                    .map(|(index, (&graph, &eager))| (index, graph, eager));
                let max_abs_diff = graph_values
                    .iter()
                    .zip(&eager_values)
                    .map(|(&graph, &eager)| (graph - eager).abs())
                    .fold(0.0f32, f32::max);
                let row_diffs: Vec<_> = graph_values
                    .chunks_exact(config.hidden_size)
                    .zip(eager_values.chunks_exact(config.hidden_size))
                    .enumerate()
                    .map(|(row, (graph, eager))| {
                        let mismatch_count = graph
                            .iter()
                            .zip(eager)
                            .filter(|(graph, eager)| graph != eager)
                            .count();
                        let max_abs_diff = graph
                            .iter()
                            .zip(eager)
                            .map(|(&graph, &eager)| (graph - eager).abs())
                            .fold(0.0f32, f32::max);
                        (row, mismatch_count, max_abs_diff)
                    })
                    .collect();
                let captured_inputs = graph_runner.captured.values().next().map(|captured| {
                    let read_u32 = |tensor: &Tensor| {
                        tensor
                            .to_device(Device::Cpu)
                            .expect("copy captured u32 input to CPU")
                            .to_vec::<u32>()
                            .expect("read captured u32 input")
                    };
                    let expected_rotary = crate::forward::rotary_tables_from_tensor(
                        &captured.position_buffer,
                        &weights.rotary_inv_freq,
                    )
                    .expect("recompute expected replay rotary tables");
                    (
                        read_u32(&captured.token_buffer),
                        hidden_f32(&captured.position_buffer),
                        captured.block_table_buffer.as_ref().map(read_u32),
                        captured.seqused_k_buffer.as_ref().map(read_u32),
                        captured.kv_slot_buffer.as_ref().map(read_u32),
                        hidden_f32(&captured.rotary_cos_buffer) == hidden_f32(&expected_rotary.0),
                        hidden_f32(&captured.rotary_sin_buffer) == hidden_f32(&expected_rotary.1),
                    )
                });
                let stats = graph_runner.stats();
                graph_runner
                    .invalidate()
                    .context("settle batched graph after hidden parity mismatch")?;
                anyhow::bail!(
                    "batched hidden mismatch at step {step}: first={first:?}, \
                     max_abs_diff={max_abs_diff}, row_diffs={row_diffs:?}, \
                     captured_inputs={captured_inputs:?}, stats={stats:?}"
                );
            }
            for layer in 0..graph_cache.num_layers() {
                let (graph_k, graph_v) =
                    graph_cache.pool_tensors(layer).expect("graph cache layer");
                let (eager_k, eager_v) =
                    eager_cache.pool_tensors(layer).expect("eager cache layer");
                anyhow::ensure!(
                    hidden_f32(&graph_k) == hidden_f32(&eager_k),
                    "batched K-pool mismatch at layer {layer}, step {step}"
                );
                anyhow::ensure!(
                    hidden_f32(&graph_v) == hidden_f32(&eager_v),
                    "batched V-pool mismatch at layer {layer}, step {step}"
                );
            }
        }

        let stats = graph_runner.stats();
        let retained_width_four = graph_runner.captured.keys().any(|key| {
            key.graph.batch_size == 4
                && key.graph.max_seqlen_k == 64
                && key.graph.max_blocks_per_seq == 4
        });
        eprintln!("[rocm-batched-graph-parity] stats={stats:?}");
        graph_runner
            .invalidate()
            .expect("settle batched graphs before evaluating test assertions");

        anyhow::ensure!(
            stats.capture_successes > 0,
            "native capture never succeeded: {stats:?}"
        );
        anyhow::ensure!(
            stats.replay_successes == 23,
            "expected 23 native replays after capture: {stats:?}"
        );
        anyhow::ensure!(stats.failures == 0, "native graph failure: {stats:?}");
        anyhow::ensure!(retained_width_four, "width-four graph was not retained");
        anyhow::ensure!(
            stats.active_graph_slot_count == 1 && stats.idle_graph_slot_count == 0,
            "retained width-four slot must be reported active: {stats:?}"
        );
        anyhow::ensure!(
            stats.fallbacks.multi_row_batch_unsupported == 0,
            "native multi-row route must not use the historical eager fallback"
        );
        eprintln!(
            "[rocm-batched-graph-parity] captures={}; deferrals={}; replays={}; fallbacks={}",
            stats.capture_successes,
            stats.capture_deferrals,
            stats.replay_successes,
            stats.fallbacks.total,
        );
        Ok(())
    }

    /// Real gfx1151 proof that captured GDN recurrent and convolution state
    /// advances exactly once per native batch replay.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn batched_graph_hybrid_gdn_state_matches_eager() -> Result<()> {
        require_explicit_rocm_qualification();
        assert!(
            kiln_tensor::rocm_is_available(),
            "ROCm qualification requested but no ROCm device is available"
        );

        let device = Device::Rocm(0);
        configure_rocm_graph_test_memory_governor(&device);
        let backend = crate::backend::for_device_kt(&device);
        let (config, weights) = rocm_graph_hybrid_test_fixture(&device);
        let graph_cache = PagedKvCacheKt::new(
            config.num_full_attention_layers,
            32,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )?;
        let eager_cache = PagedKvCacheKt::new(
            config.num_full_attention_layers,
            32,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )?;
        let tables = [
            BlockTable { blocks: vec![0, 1] },
            BlockTable { blocks: vec![4, 5] },
            BlockTable { blocks: vec![8, 9] },
            BlockTable {
                blocks: vec![12, 13],
            },
        ];
        let table_refs: Vec<&BlockTable> = tables.iter().collect();
        let row_ids = [4201, 4202, 4203, 4204];
        let mut graph_state = LinearAttentionState::new_with_batch_for_inference_runtime(
            &config,
            row_ids.len(),
            &device,
            backend.as_ref(),
        )?;
        let mut eager_state = LinearAttentionState::new_with_batch_for_inference_runtime(
            &config,
            row_ids.len(),
            &device,
            backend.as_ref(),
        )?;
        let mut graph_runner =
            RocmGraphRunner::new(&device, RocmGraphExecutionPolicy::lazy_capture_replay());

        for step in 1usize..=8 {
            let token_ids = [
                (step % config.vocab_size) as u32,
                ((step + 5) % config.vocab_size) as u32,
                ((step + 11) % config.vocab_size) as u32,
                ((step + 17) % config.vocab_size) as u32,
            ];
            let sequence_lengths = [step, step + 1, step + 2, step + 3];
            let graph_hidden = graph_runner
                .decode_step_paged_batched_hidden(
                    backend.as_ref(),
                    &token_ids,
                    &weights,
                    &config,
                    &graph_cache,
                    &table_refs,
                    &sequence_lengths,
                    &mut graph_state,
                    None,
                    Some(&row_ids),
                )?
                .context("hybrid graph runner declined a requested batch")?;
            let eager_hidden = model_forward_paged_decode_contiguous_batch_hidden_with_ids(
                backend.as_ref(),
                &token_ids,
                &weights,
                &config,
                &eager_cache,
                &table_refs,
                &sequence_lengths,
                Some(&mut eager_state),
                None,
                Some(&row_ids),
            )?;

            let parity = hidden_f32(&graph_hidden) == hidden_f32(&eager_hidden)
                && graph_state
                    .recurrent_states
                    .iter()
                    .zip(&eager_state.recurrent_states)
                    .all(|(graph, eager)| hidden_f32(graph) == hidden_f32(eager))
                && graph_state
                    .conv_states
                    .iter()
                    .zip(&eager_state.conv_states)
                    .all(|(graph, eager)| hidden_f32(graph) == hidden_f32(eager));
            if !parity {
                graph_runner.invalidate()?;
                anyhow::bail!("hybrid ROCm graph hidden/GDN state mismatch at step {step}");
            }
            for layer in 0..graph_cache.num_layers() {
                let (graph_k, graph_v) = graph_cache
                    .pool_tensors(layer)
                    .context("hybrid graph cache layer")?;
                let (eager_k, eager_v) = eager_cache
                    .pool_tensors(layer)
                    .context("hybrid eager cache layer")?;
                if hidden_f32(&graph_k) != hidden_f32(&eager_k)
                    || hidden_f32(&graph_v) != hidden_f32(&eager_v)
                {
                    graph_runner.invalidate()?;
                    anyhow::bail!("hybrid ROCm graph K/V mismatch at layer {layer}, step {step}");
                }
            }
        }

        let stats = graph_runner.stats();
        eprintln!("[rocm-batched-graph-hybrid-parity] stats={stats:?}");
        graph_runner.invalidate()?;
        anyhow::ensure!(
            stats.capture_successes == 1 && stats.replay_successes == 7,
            "hybrid graph did not capture once and replay seven times: {stats:?}"
        );
        anyhow::ensure!(stats.failures == 0 && stats.fallbacks.total == 0);
        anyhow::ensure!(stats.retained_slot_state_bytes > 0);
        anyhow::ensure!(stats.active_graph_slot_count == 1 && stats.idle_graph_slot_count == 0);
        Ok(())
    }

    /// Real gfx1151 proof that one persistent width slot can alternate between
    /// unrelated, nonzero GDN cohorts without leaking or staling either
    /// cohort's state. The small tensors retain Qwen's production 32-layer
    /// topology (three GDN layers followed by full attention, repeated eight
    /// times), so capture covers every production state and KV layer index.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn batched_graph_alternating_gdn_cohorts_match_eager() -> Result<()> {
        require_explicit_rocm_qualification();
        assert!(
            kiln_tensor::rocm_is_available(),
            "ROCm qualification requested but no ROCm device is available"
        );

        let device = Device::Rocm(0);
        configure_rocm_graph_test_memory_governor(&device);
        let backend = crate::backend::for_device_kt(&device);
        let (config, weights) = rocm_graph_qwen_depth_test_fixture(&device);
        let graph_cache = PagedKvCacheKt::new(
            config.num_full_attention_layers,
            32,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )?;
        let eager_cache = PagedKvCacheKt::new(
            config.num_full_attention_layers,
            32,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )?;
        let tables = [
            [
                BlockTable { blocks: vec![0, 1] },
                BlockTable { blocks: vec![4, 5] },
                BlockTable { blocks: vec![8, 9] },
                BlockTable {
                    blocks: vec![12, 13],
                },
            ],
            [
                BlockTable {
                    blocks: vec![16, 17],
                },
                BlockTable {
                    blocks: vec![20, 21],
                },
                BlockTable {
                    blocks: vec![24, 25],
                },
                BlockTable {
                    blocks: vec![28, 29],
                },
            ],
        ];
        let row_ids = [[4301, 4302, 4303, 4304], [4401, 4402, 4403, 4404]];
        let new_row = || {
            LinearAttentionState::new_with_batch_for_inference_runtime(
                &config,
                1,
                &device,
                backend.as_ref(),
            )
        };
        let new_cohort = || {
            (0..row_ids[0].len())
                .map(|_| new_row())
                .collect::<Result<Vec<_>>>()
        };
        let mut graph_rows = [new_cohort()?, new_cohort()?];
        let mut eager_rows = [new_cohort()?, new_cohort()?];
        let mut graph_runner =
            RocmGraphRunner::new(&device, RocmGraphExecutionPolicy::lazy_capture_replay());

        // Production rows enter decode with prompt-derived recurrent and
        // convolution state. Seed both independent cohorts through the eager
        // path before capture so a zero-state fixture cannot hide a stale-copy
        // or layer-index error.
        for cohort in 0usize..2 {
            let offset = cohort * 29;
            let token_ids = [
                (offset % config.vocab_size) as u32,
                ((offset + 5) % config.vocab_size) as u32,
                ((offset + 11) % config.vocab_size) as u32,
                ((offset + 17) % config.vocab_size) as u32,
            ];
            let sequence_lengths = [16usize; 4];
            let table_refs: Vec<&BlockTable> = tables[cohort].iter().collect();
            let graph_row_refs: Vec<&LinearAttentionState> = graph_rows[cohort].iter().collect();
            let eager_row_refs: Vec<&LinearAttentionState> = eager_rows[cohort].iter().collect();
            let mut graph_state = LinearAttentionState::from_batch_rows(&graph_row_refs)?;
            let mut eager_state = LinearAttentionState::from_batch_rows(&eager_row_refs)?;
            model_forward_paged_decode_contiguous_batch_hidden_with_ids(
                backend.as_ref(),
                &token_ids,
                &weights,
                &config,
                &graph_cache,
                &table_refs,
                &sequence_lengths,
                Some(&mut graph_state),
                None,
                Some(&row_ids[cohort]),
            )?;
            model_forward_paged_decode_contiguous_batch_hidden_with_ids(
                backend.as_ref(),
                &token_ids,
                &weights,
                &config,
                &eager_cache,
                &table_refs,
                &sequence_lengths,
                Some(&mut eager_state),
                None,
                Some(&row_ids[cohort]),
            )?;
            graph_rows[cohort] = graph_state.split_batch_rows()?;
            eager_rows[cohort] = eager_state.split_batch_rows()?;
        }

        for turn in 0usize..16 {
            // The production server owns a background memory sampler. This
            // standalone hardware test does not, so publish a fresh live probe
            // before each turn rather than letting the 500 ms cached-sample
            // freshness guard force an unrelated eager fallback.
            let memory_snapshot = kiln_memory::MemoryGovernor::global().refresh();
            anyhow::ensure!(
                !memory_snapshot.observations.probe_failed,
                "ROCm graph depth test memory probe failed before turn {turn}"
            );
            let cohort = turn % 2;
            let step = turn / 2 + 1;
            let offset = cohort * 29;
            let token_ids = [
                ((step + offset) % config.vocab_size) as u32,
                ((step + offset + 5) % config.vocab_size) as u32,
                ((step + offset + 11) % config.vocab_size) as u32,
                ((step + offset + 17) % config.vocab_size) as u32,
            ];
            let sequence_lengths = [16 + step; 4];
            let table_refs: Vec<&BlockTable> = tables[cohort].iter().collect();
            let graph_row_refs: Vec<&LinearAttentionState> = graph_rows[cohort].iter().collect();
            let eager_row_refs: Vec<&LinearAttentionState> = eager_rows[cohort].iter().collect();
            let mut graph_state = LinearAttentionState::from_batch_rows(&graph_row_refs)?;
            let mut eager_state = LinearAttentionState::from_batch_rows(&eager_row_refs)?;
            let recurrent_input_match = graph_state
                .recurrent_states
                .iter()
                .zip(&eager_state.recurrent_states)
                .all(|(graph, eager)| hidden_f32(graph) == hidden_f32(eager));
            let conv_input_match = graph_state
                .conv_states
                .iter()
                .zip(&eager_state.conv_states)
                .all(|(graph, eager)| hidden_f32(graph) == hidden_f32(eager));
            anyhow::ensure!(
                recurrent_input_match && conv_input_match,
                "alternating-cohort oracle inputs diverged before cohort {cohort}, step {step}"
            );
            let graph_hidden = graph_runner
                .decode_step_paged_batched_hidden(
                    backend.as_ref(),
                    &token_ids,
                    &weights,
                    &config,
                    &graph_cache,
                    &table_refs,
                    &sequence_lengths,
                    &mut graph_state,
                    None,
                    Some(&row_ids[cohort]),
                )?
                .context("alternating-cohort graph runner declined a requested batch")?;
            let eager_hidden = model_forward_paged_decode_contiguous_batch_hidden_with_ids(
                backend.as_ref(),
                &token_ids,
                &weights,
                &config,
                &eager_cache,
                &table_refs,
                &sequence_lengths,
                Some(&mut eager_state),
                None,
                Some(&row_ids[cohort]),
            )?;

            let hidden_match = hidden_f32(&graph_hidden) == hidden_f32(&eager_hidden);
            let recurrent_match = graph_state
                .recurrent_states
                .iter()
                .zip(&eager_state.recurrent_states)
                .all(|(graph, eager)| hidden_f32(graph) == hidden_f32(eager));
            let conv_match = graph_state
                .conv_states
                .iter()
                .zip(&eager_state.conv_states)
                .all(|(graph, eager)| hidden_f32(graph) == hidden_f32(eager));
            if !(hidden_match && recurrent_match && conv_match) {
                let stats = graph_runner.stats();
                graph_runner.invalidate()?;
                anyhow::bail!(
                    "alternating-cohort ROCm graph mismatch for cohort {cohort} at step {step}: \
                     hidden_match={hidden_match}, recurrent_match={recurrent_match}, \
                     conv_match={conv_match}, stats={stats:?}"
                );
            }
            graph_rows[cohort] = graph_state.split_batch_rows()?;
            eager_rows[cohort] = eager_state.split_batch_rows()?;
            for layer in 0..graph_cache.num_layers() {
                let (graph_k, graph_v) = graph_cache
                    .pool_tensors(layer)
                    .context("alternating graph cache layer")?;
                let (eager_k, eager_v) = eager_cache
                    .pool_tensors(layer)
                    .context("alternating eager cache layer")?;
                if hidden_f32(&graph_k) != hidden_f32(&eager_k)
                    || hidden_f32(&graph_v) != hidden_f32(&eager_v)
                {
                    graph_runner.invalidate()?;
                    anyhow::bail!(
                        "alternating-cohort ROCm graph K/V mismatch at layer {layer}, \
                         cohort {cohort}, step {step}"
                    );
                }
            }
        }

        let stats = graph_runner.stats();
        eprintln!("[rocm-batched-graph-alternating-parity] stats={stats:?}");
        graph_runner.invalidate()?;
        anyhow::ensure!(
            stats.capture_successes == 1 && stats.replay_successes == 15,
            "alternating graph did not capture once and replay fifteen times: {stats:?}"
        );
        anyhow::ensure!(stats.failures == 0 && stats.fallbacks.total == 0);
        anyhow::ensure!(stats.active_graph_slot_count == 1 && stats.idle_graph_slot_count == 0);
        Ok(())
    }

    /// A decode geometry that cannot use graph-stable native attention must
    /// remain eager. Capturing its sequence-length-shaped fallback would appear
    /// to succeed but would silently reuse the captured K/V length on replay.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn shape_dependent_attention_is_cached_as_typed_eager_fallback() {
        require_explicit_rocm_qualification();
        assert!(kiln_tensor::rocm_is_available());

        let device = Device::Rocm(0);
        configure_rocm_graph_test_memory_governor(&device);
        let backend = crate::backend::for_device_kt(&device);
        let (config, weights) = rocm_graph_test_fixture(&device, 1, 1, 64);
        let graph_cache = PagedKvCacheKt::new(
            1,
            4,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("shape-dependent graph cache");
        let eager_cache = PagedKvCacheKt::new(
            1,
            4,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("shape-dependent eager cache");
        let table = BlockTable { blocks: vec![0] };
        let mut graph_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("graph state");
        let mut eager_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("eager state");
        let mut graph_runner =
            RocmGraphRunner::new(&device, RocmGraphExecutionPolicy::lazy_capture_replay());

        for seq_len in 1..=4 {
            let token_id = (seq_len % config.vocab_size) as u32;
            let guarded = graph_runner
                .decode_step_paged_hidden(
                    backend.as_ref(),
                    token_id,
                    &weights,
                    &config,
                    &graph_cache,
                    &table,
                    seq_len,
                    &mut graph_state,
                    None,
                    811,
                )
                .expect("guarded graph decode");
            let eager = RocmGraphRunner::eager_forward_hidden(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &eager_cache,
                &table,
                seq_len,
                &mut eager_state,
                None,
            )
            .expect("shape-dependent eager oracle");
            assert_eq!(hidden_f32(&guarded), hidden_f32(&eager));
        }

        let stats = graph_runner.stats();
        assert!(stats.enabled);
        assert_eq!(stats.capture_attempts, 1);
        assert_eq!(stats.capture_deferrals, 1);
        assert_eq!(stats.capture_successes, 0);
        assert_eq!(stats.replay_attempts, 0);
        assert_eq!(stats.failures, 0);
        assert_eq!(stats.captured_graph_count, 0);
        assert_eq!(stats.fallbacks.total, 3);
        assert_eq!(stats.fallbacks.shape_dependent_attention, 3);
    }

    /// Real gfx1151 parity over the graph lifecycle that serving exercises:
    /// bucket transitions, growing metadata, retained-prefix reuse, cancelled
    /// owner cleanup, and adapter-generation invalidation.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn graph_parity_across_buckets_prefix_cancellation_and_adapter_boundary() {
        require_explicit_rocm_qualification();
        assert!(kiln_tensor::rocm_is_available());

        let device = Device::Rocm(0);
        configure_rocm_graph_test_memory_governor(&device);
        let backend = crate::backend::for_device_kt(&device);
        let (config, weights) = stale_generation_test_fixture(&device);
        let graph_cache = PagedKvCacheKt::new(
            1,
            16,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("graph parity paged cache");
        let eager_cache = PagedKvCacheKt::new(
            1,
            16,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("eager parity paged cache");
        let mut graph_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("graph state");
        let mut eager_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("eager state");
        let mut graph_runner =
            RocmGraphRunner::new(&device, RocmGraphExecutionPolicy::lazy_capture_replay());
        assert!(graph_runner.stats().capture_enabled);

        let mut assert_hidden_step = |runner: &mut RocmGraphRunner, row_id: u64, seq_len: usize| {
            let block_count = (seq_len + 1).div_ceil(graph_cache.block_size());
            let table = BlockTable {
                blocks: (0..block_count as u32).collect(),
            };
            let token_id = (seq_len % config.vocab_size) as u32;
            let graph_hidden = runner
                .decode_step_paged_hidden(
                    backend.as_ref(),
                    token_id,
                    &weights,
                    &config,
                    &graph_cache,
                    &table,
                    seq_len,
                    &mut graph_state,
                    None,
                    row_id,
                )
                .expect("graph hidden");
            let eager_hidden = RocmGraphRunner::eager_forward_hidden(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &eager_cache,
                &table,
                seq_len,
                &mut eager_state,
                None,
            )
            .expect("eager hidden");
            let graph_values = hidden_f32(&graph_hidden);
            let eager_values = hidden_f32(&eager_hidden);
            assert_eq!(
                graph_values, eager_values,
                "hidden mismatch at row {row_id}, sequence {seq_len}"
            );
        };

        const FIRST_OWNER: u64 = 901;
        for seq_len in 1..=80 {
            assert_hidden_step(&mut graph_runner, FIRST_OWNER, seq_len);
        }
        let before_cancel = graph_runner.stats();
        assert!(before_cancel.capture_successes >= 2);
        assert!(before_cancel.replay_successes > 0);
        assert!(before_cancel.captured_graph_count >= 2);

        // Cancellation returns the first owner's graph slot to the bounded
        // reuse pool. A new row then starts from already-populated KV pages,
        // matching prefix-cache reuse without native graph churn.
        graph_runner.release_decode_row(FIRST_OWNER);
        let after_cancel = graph_runner.stats();
        assert_eq!(
            after_cancel.captured_graph_count,
            before_cancel.captured_graph_count
        );
        assert_eq!(after_cancel.active_graph_slot_count, 0);
        assert_eq!(after_cancel.idle_graph_slot_count, 1);
        assert_eq!(after_cancel.decode_owner_release_count, 1);
        assert_eq!(after_cancel.decode_owner_graph_release_count, 0);

        // Rebind the idle native graph to an unrelated request with fresh GDN
        // state and disjoint KV pages. This exercises the real ROCm in-place
        // state refresh before replay rather than only continuing a prefix.
        const FRESH_OWNER: u64 = 903;
        let mut fresh_graph_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("fresh graph state");
        let mut fresh_eager_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("fresh eager state");
        let fresh_table = BlockTable { blocks: vec![8] };
        for seq_len in 1..=4 {
            let token_id = ((seq_len + 17) % config.vocab_size) as u32;
            let graph_hidden = graph_runner
                .decode_step_paged_hidden(
                    backend.as_ref(),
                    token_id,
                    &weights,
                    &config,
                    &graph_cache,
                    &fresh_table,
                    seq_len,
                    &mut fresh_graph_state,
                    None,
                    FRESH_OWNER,
                )
                .expect("fresh graph hidden");
            let eager_hidden = RocmGraphRunner::eager_forward_hidden(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &eager_cache,
                &fresh_table,
                seq_len,
                &mut fresh_eager_state,
                None,
            )
            .expect("fresh eager hidden");
            assert_eq!(hidden_f32(&graph_hidden), hidden_f32(&eager_hidden));
        }
        graph_runner.release_decode_row(FRESH_OWNER);

        const PREFIX_OWNER: u64 = 902;
        for seq_len in 64..=72 {
            assert_hidden_step(&mut graph_runner, PREFIX_OWNER, seq_len);
        }
        let before_adapter_boundary = graph_runner.stats();
        assert!(before_adapter_boundary.captured_graph_count > 0);
        assert!(before_adapter_boundary.graph_slot_reuse_count >= 2);
        let captures_before_adapter_boundary = before_adapter_boundary.capture_successes;

        // Adapter swaps invalidate pointer-bearing graph state before the next
        // request quantum. The fixture keeps weights numerically unchanged so
        // eager equality remains an exact oracle for lifecycle behavior.
        graph_runner
            .invalidate()
            .expect("settle graphs before adapter-boundary invalidation");
        assert_eq!(graph_runner.stats().captured_graph_count, 0);
        for seq_len in 73..=78 {
            assert_hidden_step(&mut graph_runner, PREFIX_OWNER, seq_len);
        }
        drop(assert_hidden_step);

        let seq_len = 79usize;
        let table = BlockTable {
            blocks: (0..(seq_len + 1).div_ceil(graph_cache.block_size()) as u32).collect(),
        };
        let token_id = (seq_len % config.vocab_size) as u32;
        let graph_logits = graph_runner
            .decode_step_paged(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &graph_cache,
                &table,
                seq_len,
                &mut graph_state,
                None,
                PREFIX_OWNER,
            )
            .expect("graph logits");
        let eager_logits = RocmGraphRunner::eager_forward(
            backend.as_ref(),
            token_id,
            &weights,
            &config,
            &eager_cache,
            &table,
            seq_len,
            &mut eager_state,
            None,
        )
        .expect("eager logits");
        assert_eq!(hidden_f32(&graph_logits), hidden_f32(&eager_logits));

        let seq_len = 80usize;
        let table = BlockTable {
            blocks: (0..(seq_len + 1).div_ceil(graph_cache.block_size()) as u32).collect(),
        };
        let token_id = (seq_len % config.vocab_size) as u32;
        let graph_token = graph_runner
            .decode_step_paged_greedy(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &graph_cache,
                &table,
                seq_len,
                &mut graph_state,
                None,
                PREFIX_OWNER,
            )
            .expect("graph greedy token");
        let eager_token = RocmGraphRunner::eager_forward_greedy(
            backend.as_ref(),
            token_id,
            &weights,
            &config,
            &eager_cache,
            &table,
            seq_len,
            &mut eager_state,
            None,
        )
        .expect("eager greedy token");
        assert_eq!(graph_token, eager_token);

        let final_stats = graph_runner.stats();
        assert!(final_stats.capture_successes > captures_before_adapter_boundary);
        assert_eq!(final_stats.failures, 0);
        assert_eq!(final_stats.fallbacks.total, 0);
        eprintln!(
            "[rocm-graph-parity] captures={}; replays={}; owner_releases={}; graph_releases={}; fallbacks={}",
            final_stats.capture_successes,
            final_stats.replay_successes,
            final_stats.decode_owner_release_count,
            final_stats.decode_owner_graph_release_count,
            final_stats.fallbacks.total,
        );
    }

    /// Deliberately corrupt only the generation retained by a live graph. The
    /// physical allocation remains valid, so a missing guard would safely but
    /// observably reach `hipGraphLaunch` via the unit-build launch probe.
    #[cfg(feature = "rocm")]
    #[test]
    #[ignore = "requires an explicit real-ROCm qualification run"]
    fn stale_pool_generation_refuses_native_replay_and_falls_back_eager() {
        require_explicit_rocm_qualification();
        assert!(
            kiln_tensor::rocm_is_available(),
            "ROCm qualification requested but no ROCm device is available"
        );

        let device = Device::Rocm(0);
        configure_rocm_graph_test_memory_governor(&device);
        let backend = crate::backend::for_device_kt(&device);
        let (config, weights) = stale_generation_test_fixture(&device);
        let graph_cache = PagedKvCacheKt::new(
            1,
            64,
            16,
            config.num_kv_heads,
            config.head_dim,
            kiln_tensor::DType::BF16,
            device,
        )
        .expect("graph paged cache");
        let table = BlockTable {
            blocks: vec![0, 1, 2, 3],
        };
        let mut graph_state =
            LinearAttentionState::new_for_inference(&config, &device).expect("graph linear state");
        let mut graph_runner =
            RocmGraphRunner::new(&device, RocmGraphExecutionPolicy::lazy_capture_replay());
        assert!(graph_runner.stats().capture_enabled);

        let row_id = 77;
        let mut next_seq_len = 1usize;
        while graph_runner.stats().replay_successes == 0 && next_seq_len < 24 {
            let token_id = (next_seq_len % config.vocab_size) as u32;
            graph_runner
                .decode_step_paged_hidden(
                    backend.as_ref(),
                    token_id,
                    &weights,
                    &config,
                    &graph_cache,
                    &table,
                    next_seq_len,
                    &mut graph_state,
                    None,
                    row_id,
                )
                .expect("graph warmup/capture/replay hidden");
            next_seq_len += 1;
        }

        let before = graph_runner.stats();
        assert!(
            before.capture_successes > 0,
            "native capture never succeeded"
        );
        assert!(before.replay_successes > 0, "native replay never succeeded");
        assert_eq!(before.failures, 0);
        assert_eq!(before.captured_graph_count, 1);

        // Compute the fallback oracle on this same cache before corrupting only
        // the graph's retained identity. Repeating a full-attention step is
        // idempotent: it overwrites the same slot with the same token-derived
        // KV and reads the same prefix.
        let token_id = (next_seq_len % config.vocab_size) as u32;
        let mut reference_state = LinearAttentionState::new_for_inference(&config, &device)
            .expect("reference linear state");
        let eager_hidden = RocmGraphRunner::eager_forward_hidden(
            backend.as_ref(),
            token_id,
            &weights,
            &config,
            &graph_cache,
            &table,
            next_seq_len,
            &mut reference_state,
            None,
        )
        .expect("same-cache eager mismatch-step control");

        let live_identity = graph_cache.pool_identity();
        for captured in graph_runner.captured.values_mut() {
            assert_eq!(captured.kv_pool_identity, live_identity);
            captured.kv_pool_identity.generation = live_identity
                .generation
                .checked_add(1)
                .expect("test generation increment");
        }

        let native_launches_before =
            ROCM_TEST_NATIVE_REPLAY_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed);
        let guarded_hidden = graph_runner
            .decode_step_paged_hidden(
                backend.as_ref(),
                token_id,
                &weights,
                &config,
                &graph_cache,
                &table,
                next_seq_len,
                &mut graph_state,
                None,
                row_id,
            )
            .expect("identity mismatch must take eager fallback");
        assert_eq!(hidden_f32(&guarded_hidden), hidden_f32(&eager_hidden));

        let after = graph_runner.stats();
        let native_launches_after =
            ROCM_TEST_NATIVE_REPLAY_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(after.replay_attempts, before.replay_attempts + 1);
        assert_eq!(after.replay_successes, before.replay_successes);
        assert_eq!(after.replay_failures, before.replay_failures + 1);
        assert_eq!(after.fallbacks.total, before.fallbacks.total + 1);
        assert_eq!(
            after.fallbacks.replay_failure,
            before.fallbacks.replay_failure + 1
        );
        assert_eq!(after.failures, before.failures + 1);
        assert_eq!(native_launches_after, native_launches_before);
        assert!(
            !after.enabled,
            "mismatch must trip the graph circuit breaker"
        );
        assert_eq!(after.captured_graph_count, 0);
        assert_eq!(graph_cache.pool_identity(), live_identity);
        eprintln!(
            "[rocm-stale-generation] replay_attempts={} -> {}; replay_failures={} -> {}; native_launches={} -> {}",
            before.replay_attempts,
            after.replay_attempts,
            before.replay_failures,
            after.replay_failures,
            native_launches_before,
            native_launches_after,
        );
    }

    #[test]
    fn invalidate_bumps_generation_and_resets_warmup() {
        let mut r = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );
        r.warmup_done = true;
        r.counters
            .record_capture_outcome(RocmGraphCaptureOutcome::SucceededRetained);
        r.counters.record_replay_outcome(false);
        let gen0 = r.adapter_generation;
        r.invalidate().expect("CPU runner invalidation");
        assert_eq!(r.adapter_generation, gen0 + 1);
        assert!(!r.warmup_done);
        let stats = r.stats();
        assert_eq!(stats.capture_attempts, 1);
        assert_eq!(stats.capture_successes, 1);
        assert_eq!(stats.capture_deferrals, 0);
        assert_eq!(stats.capture_failures, 0);
        assert_eq!(stats.replay_attempts, 1);
        assert_eq!(stats.replay_successes, 0);
        assert_eq!(stats.replay_failures, 1);
        assert_eq!(stats.failures, 1);
        assert_eq!(stats.captured_graph_count, 0);
    }

    #[test]
    fn counters_record_monotonic_capture_and_replay_outcomes() {
        let mut counters = RocmGraphCounters::default();
        counters.record_capture_outcome(RocmGraphCaptureOutcome::Failed);
        counters.record_capture_outcome(RocmGraphCaptureOutcome::Deferred);
        counters.record_capture_outcome(RocmGraphCaptureOutcome::SucceededRetained);
        counters.record_capture_outcome(RocmGraphCaptureOutcome::SucceededUncached);
        counters.record_replay_outcome(true);
        counters.record_replay_outcome(false);
        counters.record_decode_owner_release(2);
        counters.record_decode_owner_release(1);
        counters.record_fallback(
            RocmGraphFallbackReason::CaptureFailure,
            std::time::Duration::from_millis(50),
        );
        counters.record_fallback(
            RocmGraphFallbackReason::ReplayFailure,
            std::time::Duration::from_millis(120),
        );
        counters.record_fallback(
            RocmGraphFallbackReason::ReplayFailure,
            std::time::Duration::from_millis(10),
        );
        counters.record_fallback(
            RocmGraphFallbackReason::MultiRowBatchUnsupported,
            std::time::Duration::from_millis(130),
        );

        assert_eq!(counters.capture_attempts, 4);
        assert_eq!(counters.capture_successes, 2);
        assert_eq!(counters.capture_deferrals, 1);
        assert_eq!(counters.capture_failures, 1);
        assert_eq!(counters.replay_attempts, 2);
        assert_eq!(counters.replay_successes, 1);
        assert_eq!(counters.replay_failures, 1);
        assert_eq!(counters.decode_owner_release_count, 2);
        assert_eq!(counters.decode_owner_graph_release_count, 3);
        assert_eq!(counters.fallbacks.total, 4);
        assert_eq!(counters.fallbacks.multi_row_batch_unsupported, 1);
        assert_eq!(counters.fallbacks.capture_failure, 1);
        assert_eq!(counters.fallbacks.replay_failure, 2);
        assert_eq!(counters.fallbacks.slow, 2);
        assert_eq!(counters.fallbacks.total_duration_micros, 310_000);
        assert_eq!(counters.fallbacks.max_duration_micros, 130_000);
    }

    #[test]
    fn multi_row_eager_fallback_requires_requested_capture_and_batch() {
        let mut runner = RocmGraphRunner::new(
            &Device::Cpu,
            RocmGraphExecutionPolicy::lazy_capture_replay(),
        );

        runner.record_multi_row_eager_fallback(4, std::time::Duration::from_millis(130));
        assert_eq!(runner.counters.fallbacks.total, 0);

        runner.capture_requested = true;
        runner.record_multi_row_eager_fallback(1, std::time::Duration::from_millis(130));
        assert_eq!(runner.counters.fallbacks.total, 0);

        runner.record_multi_row_eager_fallback(4, std::time::Duration::from_millis(130));
        assert_eq!(runner.counters.fallbacks.total, 1);
        assert_eq!(runner.counters.fallbacks.multi_row_batch_unsupported, 1);
        assert_eq!(runner.counters.fallbacks.slow, 1);
        assert_eq!(runner.counters.fallbacks.total_duration_micros, 130_000);
        assert_eq!(runner.counters.fallbacks.max_duration_micros, 130_000);
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn capture_admission_and_eviction_counters_reconcile_by_closed_cause() {
        let mut counters = RocmGraphCounters::default();
        counters.record_capture_outcome(RocmGraphCaptureOutcome::SucceededRetained);
        counters.record_cache_admission();
        for rejection in [
            RocmGraphAdmissionRejection::EntryCapacity,
            RocmGraphAdmissionRejection::CandidateByteBudget,
            RocmGraphAdmissionRejection::AccountingIncomplete,
        ] {
            counters.record_capture_outcome(RocmGraphCaptureOutcome::SucceededUncached);
            counters.record_cache_rejection(rejection);
        }
        assert_eq!(
            counters.capture_successes,
            counters
                .cache_admission_successes
                .saturating_add(counters.entry_capacity_rejections)
                .saturating_add(counters.byte_budget_rejections)
                .saturating_add(counters.accounting_incomplete_rejections)
        );
        assert_eq!(counters.pre_capture_entry_capacity_skips, 0);
        assert_eq!(counters.pre_capture_byte_budget_skips, 0);
        assert_eq!(counters.pre_capture_accounting_incomplete_skips, 0);
        assert_eq!(counters.pre_capture_memory_reservation_denied_skips, 0);
        assert_eq!(counters.memory_governor_selector_mismatch_skips, 0);

        for (graphs, reason) in [
            (1, RocmGraphEvictionReason::Budget),
            (2, RocmGraphEvictionReason::Pressure),
            (3, RocmGraphEvictionReason::Invalidation),
            (4, RocmGraphEvictionReason::Recovery),
        ] {
            counters.record_cache_eviction(graphs, graphs as u64 * 1024, reason);
        }
        assert_eq!(
            counters.cache_evictions,
            counters
                .budget_evictions
                .saturating_add(counters.pressure_evictions)
                .saturating_add(counters.invalidation_evictions)
                .saturating_add(counters.recovery_evictions)
        );
    }

    #[test]
    fn fixed_phase_telemetry_is_bounded_and_tracks_transient_high_water() {
        let telemetry = RocmGraphTelemetryHandle::default();
        let idle = telemetry.snapshot();
        assert_eq!(idle.current_phase, None);
        assert_eq!(idle.current_phase_elapsed_micros, 0);

        let older = telemetry.timer(RocmGraphPhase::CandidateWarm);
        let first_elapsed = telemetry.snapshot().current_phase_elapsed_micros;
        let second_elapsed = telemetry.snapshot().current_phase_elapsed_micros;
        assert!(second_elapsed >= first_elapsed);
        let newer = telemetry.timer(RocmGraphPhase::NativeCapture);
        drop(older);
        let while_newer = telemetry.snapshot();
        assert_eq!(
            while_newer.current_phase,
            Some(RocmGraphPhase::NativeCapture)
        );
        assert_eq!(
            serde_json::to_value(while_newer).expect("serialize live telemetry")["current_phase"],
            "native_capture"
        );
        drop(newer);
        let idle_again = telemetry.snapshot();
        assert_eq!(idle_again.current_phase, None);
        assert_eq!(idle_again.current_phase_elapsed_micros, 0);

        for phase in [
            RocmGraphPhase::PreCandidateHeadroom,
            RocmGraphPhase::PreNativeReservation,
            RocmGraphPhase::RejectedCandidateCleanup,
        ] {
            let _timer = telemetry.timer(phase);
        }
        telemetry.record_transient_candidate_bytes(4096);
        telemetry.record_transient_candidate_bytes(1024);
        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.candidate_warm_phase.calls, 1);
        assert_eq!(snapshot.pre_candidate_headroom_phase.calls, 1);
        assert_eq!(snapshot.pre_native_reservation_phase.calls, 1);
        assert_eq!(snapshot.native_capture_phase.calls, 1);
        assert_eq!(snapshot.rejected_candidate_cleanup_phase.calls, 1);
        assert_eq!(snapshot.last_transient_candidate_bytes, 1024);
        assert_eq!(snapshot.peak_transient_candidate_bytes, 4096);

        let mut phase = RocmGraphPhaseStats::default();
        phase.record(std::time::Duration::from_millis(50));
        phase.record(std::time::Duration::from_millis(120));
        assert_eq!(phase.calls, 2);
        assert_eq!(phase.slow, 1);
        assert_eq!(phase.total_duration_micros, 170_000);
        assert_eq!(phase.max_duration_micros, 120_000);
    }

    #[test]
    fn fallback_reason_labels_are_closed_and_distinct() {
        let labels = [
            RocmGraphFallbackReason::ColdCacheHostRoundTrip,
            RocmGraphFallbackReason::PersistentHostRoundTrip,
            RocmGraphFallbackReason::ShapeDependentAttention,
            RocmGraphFallbackReason::GraphCacheCapacity,
            RocmGraphFallbackReason::GraphCacheByteBudget,
            RocmGraphFallbackReason::GraphAccountingIncomplete,
            RocmGraphFallbackReason::ModerateMemoryPressure,
            RocmGraphFallbackReason::TightMemoryPressure,
            RocmGraphFallbackReason::CriticalMemoryPressure,
            RocmGraphFallbackReason::MemoryReservationDenied,
            RocmGraphFallbackReason::MemoryGovernorSelectorMismatch,
            RocmGraphFallbackReason::CaptureFailure,
            RocmGraphFallbackReason::ReplayFailure,
        ]
        .map(RocmGraphFallbackReason::as_str);
        let mut distinct = labels.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), labels.len());
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn moderate_pressure_stops_capture_growth_without_requesting_eviction() {
        assert_eq!(
            non_evicting_pressure_decision(kiln_memory::MemoryPressure::Comfortable),
            Some(RocmGraphPressureDecision::Normal)
        );
        assert_eq!(
            non_evicting_pressure_decision(kiln_memory::MemoryPressure::Moderate),
            Some(RocmGraphPressureDecision::ReplayOnly(
                RocmGraphFallbackReason::ModerateMemoryPressure
            ))
        );
        assert_eq!(
            non_evicting_pressure_decision(kiln_memory::MemoryPressure::Tight),
            None
        );
        assert_eq!(
            non_evicting_pressure_decision(kiln_memory::MemoryPressure::Critical),
            None
        );
    }

    #[test]
    fn graph_governor_selector_must_match_the_active_device() {
        use kiln_memory::VramProbeSelector;

        assert!(memory_governor_selector_matches(
            VramProbeSelector::Nvidia(0),
            VramProbeSelector::Nvidia(0)
        ));
        assert!(!memory_governor_selector_matches(
            VramProbeSelector::Nvidia(0),
            VramProbeSelector::Nvidia(1)
        ));
        assert!(!memory_governor_selector_matches(
            VramProbeSelector::Nvidia(0),
            VramProbeSelector::Auto
        ));
    }
}

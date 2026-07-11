//! Real-model batching engine actor scaffolding.
//!
//! Phase 1 keeps the actor in `kiln-server` so HTTP request routing, prefix
//! cache ownership, GPU coordination, and metrics can be wired incrementally.
//! Decode now issues a multi-row forward by default; the rowwise path remains
//! only as an operator-forced comparison or fallback mode.

use std::collections::{HashSet, VecDeque};
use std::sync::mpsc as std_mpsc;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use kiln_core::block::BlockManager;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_model::{
    BackendHealthHandle, CancelHandle, DecodeBatcherPolicy, FinishReason, GenerationOutput,
    ModelRunner, PagedBatchedDecodeState, PagedBatchedPrefillStart, PagedBatchedPrefillState,
    PagedKvCacheKt, PagedPrefixRegistration, PagedPrefixReuse,
};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::config::{
    BatchTokenBudget, ConfigValueSource, PrefillLayerBudget, PrefillTokenBudget, StreamStallGrace,
};
use crate::response_delivery::{
    DeliveryBarrierError, DeliveryBatch, DeliveryCommand, DeliveryKey, DeliveryResult,
    DeliveryResultNotifyError, DeliveryResultSink, DeliveryResultSinkError, DeliveryTerminal,
    DeliveryWorker,
};
use crate::state::{
    GpuCoordinationLock, LoadedAdapterIdentity, RealPrefixCache, RealPrefixCacheRequest,
    gpu_coordination_read_guard, gpu_coordination_write_guard_while_healthy,
};

const DEFAULT_ENGINE_CHANNEL: usize = 1024;
const DEFAULT_RESPONSE_CHANNEL: usize = 64;
const DEFAULT_MAX_DECODE_BATCH: usize = 8;
const DEFAULT_PREFILL_ADMISSION_QUANTUM: usize = 4;
const DEFAULT_PREFIX_AWARE_ADMISSION: bool = true;

/// Actor work above this wall time is material to the qualification stall
/// gate and gets one bounded structured event after the phase completes.
const SLOW_ACTOR_PHASE_THRESHOLD: Duration = Duration::from_millis(100);

/// Fair worker retry cadence while a response lane is inside its grace window.
const RESPONSE_DELIVERY_POLL_CADENCE: Duration = Duration::from_millis(10);

/// Delivery settings resolved and validated before the batching actor starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseDeliveryPolicy {
    stream_stall_grace: Duration,
    stream_stall_grace_source: ConfigValueSource,
}

impl ResponseDeliveryPolicy {
    pub fn stream_stall_grace_ms(self) -> u64 {
        duration_millis_saturating(self.stream_stall_grace)
    }

    pub fn stream_stall_grace_source(self) -> ConfigValueSource {
        self.stream_stall_grace_source
    }
}

impl From<StreamStallGrace> for ResponseDeliveryPolicy {
    fn from(grace: StreamStallGrace) -> Self {
        Self {
            stream_stall_grace: grace.duration(),
            stream_stall_grace_source: grace.source(),
        }
    }
}

impl Default for ResponseDeliveryPolicy {
    fn default() -> Self {
        StreamStallGrace::default().into()
    }
}

fn duration_millis_saturating(duration: Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}

fn blocks_needed_for_tokens(num_tokens: usize, block_size: usize) -> usize {
    num_tokens.div_ceil(block_size)
}

/// Resolve the actor's `max_decode_batch` from the environment, falling back to
/// the active backend's decode policy when `KILN_MAX_DECODE_BATCH` is unset or
/// cannot be parsed as a positive integer. The actor caps `active.len()` at
/// this value, so it is the effective concurrent-decode width.
pub(crate) fn env_max_decode_batch_for_policy(policy: Option<DecodeBatcherPolicy>) -> usize {
    std::env::var("KILN_MAX_DECODE_BATCH")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            policy.map_or(DEFAULT_MAX_DECODE_BATCH, |policy| {
                // The engine's width, not the legacy batcher's: CUDA keeps
                // its serial legacy row loop (max_batch 1) while the engine
                // decodes concurrently.
                policy.engine_max_decode_batch.unwrap_or(policy.max_batch)
            })
        })
}

fn env_prefix_aware_admission() -> bool {
    match std::env::var("KILN_BATCH_PREFIX_AWARE_ADMISSION") {
        Ok(raw) => !matches!(
            raw.trim(),
            "0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO"
        ),
        Err(_) => DEFAULT_PREFIX_AWARE_ADMISSION,
    }
}

/// Cap how many queued requests the actor prefills before yielding to a decode
/// step. Vulkan defaults to filling the resident decode width before the first
/// decode step, while other backends keep a smaller latency-oriented default.
pub(crate) fn env_prefill_admission_quantum_for_policy(
    max_decode_batch: usize,
    policy: Option<DecodeBatcherPolicy>,
) -> usize {
    let max_decode_batch = max_decode_batch.max(1);
    std::env::var("KILN_BATCH_PREFILL_ADMISSION_QUANTUM")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| {
            // #1082 CUDA concurrency regression: this quantum governs how many
            // waiting requests are prompt-prefilled per actor cycle before the
            // decode width is reached. At LKG 2d9d4fc4 `admit_waiting`
            // burst-filled the whole decode width in one cycle and CUDA scaled
            // ~5.9x to ~498 tok/s @ bs=64. The Vulkan admission-tuning commits
            // (568d82a4 / 07607d5d) restored that full-width burst ONLY for
            // Vulkan, leaving CUDA pinned at the latency default (4): the 64
            // prompt prefills then drained ~1/decode-cycle on the single actor
            // thread, ballooning TTFT to ~38s and collapsing aggregate
            // throughput to ~22 tok/s. GPU backends saturate via wide decode
            // batches, so give CUDA the same full-width quantum as Vulkan; CPU
            // keeps the latency-oriented default. Per-deploy override:
            // KILN_BATCH_PREFILL_ADMISSION_QUANTUM.
            if policy.is_some_and(|policy| policy.use_decode_width_prefill_admission) {
                max_decode_batch
            } else {
                DEFAULT_PREFILL_ADMISSION_QUANTUM
            }
        })
        .clamp(1, max_decode_batch)
}

/// Decide whether multi-row decode steps should be issued one-row-at-a-time
/// instead of as a single batched forward. Backends now default to true batched
/// decode; `KILN_BATCH_DECODE_ROWWISE` (0/1) remains as an operator override
/// for focused comparisons and emergency fallback.
fn default_rowwise_decode() -> bool {
    if let Ok(raw) = std::env::var("KILN_BATCH_DECODE_ROWWISE") {
        return !matches!(
            raw.trim(),
            "0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO"
        );
    }
    false
}

#[derive(Debug, Clone)]
pub struct EngineRequest {
    pub request_id: Uuid,
    pub prompt_tokens: Vec<TokenId>,
    pub sampling: SamplingParams,
    pub adapter: Option<LoadedAdapterIdentity>,
    pub cancel: CancelHandle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    Token { token: TokenId, ready_at: Instant },
    Done { output: BatchedGenerationOutput },
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchedGenerationOutput {
    pub text: String,
    pub token_ids: Vec<TokenId>,
    pub finish_reason: FinishReason,
    pub completion_tokens: usize,
    pub prefill_duration: Duration,
    pub decode_duration: Duration,
}

fn completion_usage_tokens(visible_token_count: usize, finish_reason: &FinishReason) -> usize {
    visible_token_count + usize::from(matches!(finish_reason, FinishReason::Eos))
}

pub struct DecodeForwardOutput {
    pub output: GenerationOutput,
    pub prefill_duration: Duration,
    pub decode_duration: Duration,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchingEngineSnapshot {
    /// Age of the shared published snapshot at read time. Strong actor-barrier
    /// snapshots return zero; [`BatchingEngineHandle::cached_snapshot`] fills
    /// this from the publication timestamp without contacting the actor.
    pub snapshot_age_ms: u64,
    /// Effective full-response-channel grace injected when this actor started.
    pub stream_stall_grace_ms: u64,
    /// Startup source that selected `stream_stall_grace_ms`.
    pub stream_stall_grace_source: ConfigValueSource,
    pub accepting: bool,
    pub queue_depth: usize,
    pub active_decode: usize,
    pub active_prefill: usize,
    pub max_batch_tokens: usize,
    pub max_batch_tokens_source: ConfigValueSource,
    pub max_prefill_tokens_per_cycle: usize,
    pub max_prefill_tokens_per_cycle_source: ConfigValueSource,
    pub max_prefill_layers_per_cycle: usize,
    pub max_prefill_layers_per_cycle_source: ConfigValueSource,
    pub max_prefill_admission_quantum: usize,
    pub current_batch_size: usize,
    pub last_batch_size: usize,
    pub max_observed_batch_size: usize,
    pub last_forward_ms: f64,
    pub max_decode_forward_ms: f64,
    pub total_decode_forward_ms: f64,
    pub slow_decode_forward_count: u64,
    pub last_prefill_ms: f64,
    pub max_prefill_forward_ms: f64,
    pub total_prefill_forward_ms: f64,
    pub slow_prefill_forward_count: u64,
    pub last_prefill_tokens: usize,
    pub last_prefill_layers: usize,
    pub last_admission_ms: f64,
    pub max_admission_ms: f64,
    pub total_admission_ms: f64,
    pub total_admission_calls: u64,
    pub slow_admission_count: u64,
    pub total_decode_forwards: u64,
    pub total_batched_decode_forwards: u64,
    pub total_decode_rows: u64,
    pub total_prefill_admission_cycles: u64,
    pub total_prefill_forwards: u64,
    pub total_prefill_layers: u64,
    pub total_prefill_layer_yields: u64,
    pub total_prefill_token_budget_deferrals: u64,
    pub total_decode_tokens: u64,
    pub total_prefill_tokens: u64,
    pub total_errors: u64,
    /// Response batches that encountered a full per-request event channel.
    /// Each sequence counts once even if token progress starts a fresh grace episode.
    pub response_backpressure_events: u64,
    /// Cumulative time spent polling full response channels during bounded grace.
    pub response_backpressure_wait_ms: u64,
    /// Requests evicted after their response channel remained full for the grace window.
    pub response_stall_evictions: u64,
    /// Active requests discarded because their response receiver was already closed.
    pub response_channel_closed: u64,
    pub response_delivery_in_flight: usize,
    pub response_delivery_backpressured: usize,
    pub response_delivery_pending_terminal: usize,
    pub adapter_groups_waiting: usize,
    /// Last prefix-deferral gauge sampled by a strong actor-barrier snapshot.
    /// Cheap cached publications preserve that sample rather than repeating
    /// the O(waiting x active x prompt_len) prefix scan on every decode token.
    pub prefix_deferred_waiting: usize,
    pub prefix_admission_deferrals: u64,
}

#[derive(Debug, Clone)]
struct PublishedBatchingEngineSnapshot {
    snapshot: BatchingEngineSnapshot,
    published_at: Instant,
}

type SharedBatchingEngineSnapshot = Arc<RwLock<PublishedBatchingEngineSnapshot>>;

pub enum DecodeSlot {
    Mock {
        next_token: TokenId,
        generated_tokens: Vec<TokenId>,
    },
    Real {
        state: PagedBatchedDecodeState,
        prefix_request: Option<RealPrefixCacheRequest>,
        first_token_pending: bool,
    },
    RealPrefill {
        state: Option<PagedBatchedPrefillState>,
        prefix_request: Option<RealPrefixCacheRequest>,
    },
}

/// Ownership returned while a request moves from admission to decode-ready.
pub enum RequestPreparation {
    Prefilling {
        slot: DecodeSlot,
        tokens_processed: usize,
        layers_processed: usize,
    },
    Ready {
        slot: DecodeSlot,
        tokens_processed: usize,
        layers_processed: usize,
    },
}

fn collect_ready_decode_indices(
    slots: &mut [&mut DecodeSlot],
    sampling: &[SamplingParams],
    output: &mut [TokenId],
) -> Result<(Vec<usize>, Vec<SamplingParams>)> {
    anyhow::ensure!(
        slots.len() == sampling.len() && slots.len() == output.len(),
        "decode slots length {} sampling length {} output length {} mismatch",
        slots.len(),
        sampling.len(),
        output.len()
    );

    let mut decode_indices = Vec::new();
    let mut decode_params = Vec::new();
    for (idx, slot) in slots.iter_mut().enumerate() {
        match slot {
            DecodeSlot::Real {
                state,
                first_token_pending,
                ..
            } if *first_token_pending => {
                output[idx] = state.next_token;
                *first_token_pending = false;
            }
            DecodeSlot::Real { .. } => {
                decode_indices.push(idx);
                decode_params.push(sampling[idx].clone());
            }
            DecodeSlot::RealPrefill { .. } => {
                anyhow::bail!("prefilling slot sent to real decode forward")
            }
            DecodeSlot::Mock { .. } => anyhow::bail!("mock slot sent to real decode forward"),
        }
    }

    Ok((decode_indices, decode_params))
}

pub trait DecodeForward: Send + Sync + 'static {
    fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot>;
    fn supports_resumable_prefill(&self) -> bool {
        false
    }
    fn prepare_request_chunked(
        &self,
        req: &EngineRequest,
        max_tokens: usize,
    ) -> Result<RequestPreparation> {
        anyhow::ensure!(
            req.prompt_tokens.len() <= max_tokens,
            "decode forward does not support resumable prefill for a {}-token prompt under a {max_tokens}-token budget",
            req.prompt_tokens.len()
        );
        Ok(RequestPreparation::Ready {
            slot: self.prepare_request(req)?,
            tokens_processed: req.prompt_tokens.len(),
            layers_processed: 0,
        })
    }
    fn advance_prefill(
        &self,
        _slot: DecodeSlot,
        _max_tokens: usize,
        _max_layers: usize,
        _sampling: &SamplingParams,
        _cancel: &CancelHandle,
    ) -> Result<RequestPreparation> {
        anyhow::bail!("decode forward does not support resumable prefill")
    }
    /// Cheap, deterministic classification used on the actor hot path. Custom
    /// forwards may keep resumable state outside [`DecodeSlot`] and override
    /// this method; the production forward uses `RealPrefill` ownership.
    fn is_prefilling(&self, slot: &DecodeSlot) -> bool {
        matches!(slot, DecodeSlot::RealPrefill { .. })
    }
    /// Distinguish a valid transformer-layer yield from a broken prefill that
    /// reported zero token and zero internal progress.
    fn has_inflight_prefill_layer_progress(&self, _slot: &DecodeSlot) -> bool {
        false
    }
    /// Token width fixed when a layer-resumable chunk began. The actor must
    /// not resume it in a later cycle with less remaining token budget.
    fn inflight_prefill_token_width(&self, _slot: &DecodeSlot) -> Option<usize> {
        None
    }
    fn can_reuse_as_strict_prefix(&self, _prompt_token_len: usize) -> bool {
        false
    }
    /// Grow KV capacity for the ready decode rows BEFORE the forward.
    /// Returns the indices of slots that could NOT be grown because the
    /// block pool is exhausted — the actor finishes those requests as
    /// `length` casualties (they outgrew the pool) instead of letting a
    /// later atomic-grow failure kill the ENTIRE batch. Non-capacity
    /// errors still propagate.
    fn grow_for_decode(&self, _slots: &mut [&mut DecodeSlot]) -> Result<Vec<usize>> {
        Ok(Vec::new())
    }
    fn forward_decode(
        &self,
        slots: &mut [&mut DecodeSlot],
        sampling: &[SamplingParams],
    ) -> Result<Vec<TokenId>>;
    fn is_eos_token(&self, _token: TokenId) -> Result<bool> {
        Ok(false)
    }
    fn stop_reason_after_emit(
        &self,
        generated_tokens: &[TokenId],
        sampling: &SamplingParams,
    ) -> Result<Option<FinishReason>> {
        if generated_tokens.len() >= sampling.max_tokens {
            Ok(Some(FinishReason::MaxTokens))
        } else {
            Ok(None)
        }
    }
    fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize>;
    fn finish_request(
        &self,
        slot: DecodeSlot,
        finish_reason: FinishReason,
    ) -> Result<DecodeForwardOutput>;
    fn discard_request(&self, _slot: DecodeSlot) {}

    /// Physically resize the KV cache to `target_blocks` usable blocks — the
    /// memory-governor actuator (#26/#24). SHRINK hands KV VRAM back to the pool
    /// for a coexisting training run to reuse; GROW reclaims it when pressure
    /// eases. Called ONLY from the engine actor between decode steps (the
    /// barrier), and takes exclusive GPU access internally so the other decode
    /// actor / training are excluded while the pools swap. Returns the achieved
    /// block count (a shrink may stop above `target_blocks` if live requests
    /// still hold high blocks). Default: no-op (mock/non-paged backends).
    fn resize_kv(&self, _target_blocks: usize) -> Result<usize> {
        Ok(0)
    }

    /// Current usable KV block count, for the governor's resize policy. `None`
    /// if this forward has no paged cache. Default: `None`.
    fn kv_num_blocks(&self) -> Option<usize> {
        None
    }
}

pub struct RealDecodeForward {
    runner: Arc<RwLock<ModelRunner>>,
    backend_health: BackendHealthHandle,
    block_manager: Arc<Mutex<BlockManager>>,
    paged_cache: Arc<PagedKvCacheKt>,
    prefix_cache: Arc<Mutex<RealPrefixCache>>,
    gpu_lock: GpuCoordinationLock,
    loaded_adapter: Arc<RwLock<Option<LoadedAdapterIdentity>>>,
    allow_dynamic_kv_resize: bool,
    // When set, multi-row decode steps are dispatched as a loop of single-row
    // forwards instead of one batched forward. Defaults off so Vulkan reaches
    // the native multi-row resident decode route; the env override is kept for
    // focused comparisons and fallback.
    rowwise_decode: bool,
}

impl RealDecodeForward {
    pub fn new(
        runner: Arc<RwLock<ModelRunner>>,
        block_manager: Arc<Mutex<BlockManager>>,
        paged_cache: Arc<PagedKvCacheKt>,
        prefix_cache: Arc<Mutex<RealPrefixCache>>,
        gpu_lock: GpuCoordinationLock,
        loaded_adapter: Arc<RwLock<Option<LoadedAdapterIdentity>>>,
        allow_dynamic_kv_resize: bool,
    ) -> Self {
        let rowwise_decode = default_rowwise_decode();
        let backend_health = runner
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .backend_health_handle();
        Self {
            runner,
            backend_health,
            block_manager,
            paged_cache,
            prefix_cache,
            gpu_lock,
            loaded_adapter,
            allow_dynamic_kv_resize,
            rowwise_decode,
        }
    }

    pub fn with_rowwise_decode(mut self, rowwise: bool) -> Self {
        self.rowwise_decode = rowwise;
        self
    }

    fn runner_guard(&self) -> Result<std::sync::RwLockReadGuard<'_, ModelRunner>> {
        let runner = self
            .runner
            .read()
            .map_err(|e| anyhow::anyhow!("model runner lock poisoned: {e}"))?;
        runner.ensure_backend_healthy()?;
        Ok(runner)
    }

    fn runner_guard_for_finish(&self) -> (std::sync::RwLockReadGuard<'_, ModelRunner>, bool) {
        match self.runner.read() {
            Ok(runner) => (runner, false),
            Err(poisoned) => {
                tracing::error!(
                    "recovering poisoned model runner solely to release batched decode ownership"
                );
                (poisoned.into_inner(), true)
            }
        }
    }

    fn prefix_cache_guard(&self) -> Result<std::sync::MutexGuard<'_, RealPrefixCache>> {
        self.prefix_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("prefix cache lock poisoned: {e}"))
    }

    fn block_manager_guard(&self) -> Result<std::sync::MutexGuard<'_, BlockManager>> {
        self.block_manager
            .lock()
            .map_err(|e| anyhow::anyhow!("block manager lock poisoned: {e}"))
    }

    fn free_uncached_blocks(
        &self,
        output: &mut kiln_model::PrefixCachedGenerationOutput,
        prefix_request: Option<RealPrefixCacheRequest>,
    ) -> Result<()> {
        self.finish_prefix_resources(
            output.registration.take(),
            std::mem::take(&mut output.extra_registrations),
            std::mem::take(&mut output.allocated_blocks),
            prefix_request,
        )
    }

    fn finish_prefix_resources(
        &self,
        registration: Option<PagedPrefixRegistration>,
        mut extra_registrations: Vec<PagedPrefixRegistration>,
        allocated_blocks: Vec<u32>,
        prefix_request: Option<RealPrefixCacheRequest>,
    ) -> Result<()> {
        let mut registrations = Vec::new();
        if let Some(registration) = registration {
            registrations.push(registration);
        }
        registrations.append(&mut extra_registrations);
        if let Some(prefix_request) = prefix_request {
            prefix_request.finish(registrations, allocated_blocks);
        } else if !allocated_blocks.is_empty() {
            if !registrations.is_empty() {
                tracing::warn!(
                    registrations = registrations.len(),
                    "discarding unfenced prefix-cache registrations"
                );
            }
            let mut bm_guard = self.block_manager_guard()?;
            bm_guard.free_all(&allocated_blocks);
        }
        Ok(())
    }

    fn grow_ready_decode_slots(&self, slots: &mut [&mut DecodeSlot]) -> Result<()> {
        let block_size = self.block_manager_guard()?.block_size();

        let missing_by_slot: Vec<usize> = slots
            .iter()
            .map(|slot| match slot {
                DecodeSlot::Real {
                    state,
                    first_token_pending: false,
                    ..
                } => {
                    let required_blocks =
                        blocks_needed_for_tokens(state.seq_len.saturating_add(1), block_size);
                    required_blocks.saturating_sub(state.block_table.blocks.len())
                }
                _ => 0,
            })
            .collect();
        let total_missing: usize = missing_by_slot.iter().sum();
        if total_missing == 0 {
            return Ok(());
        }

        let allocated_blocks = {
            let mut bm_guard = self.block_manager_guard()?;
            bm_guard
                .allocate(total_missing)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };

        let mut cursor = 0;
        for (slot, missing) in slots.iter_mut().zip(missing_by_slot) {
            if missing == 0 {
                continue;
            }
            let new_blocks = &allocated_blocks[cursor..cursor + missing];
            cursor += missing;
            let DecodeSlot::Real { state, .. } = &mut **slot else {
                unreachable!("missing block count is only set for real decode slots");
            };
            state.block_table.blocks.extend(new_blocks.iter().copied());
            state.allocated_blocks.extend(new_blocks.iter().copied());
        }

        Ok(())
    }

    /// Per-slot KV growth for the actor's pre-decode pass. Unlike the
    /// atomic `grow_ready_decode_slots` (which the forward keeps as a
    /// backstop), a pool-exhausted allocation here starves ONLY the slots
    /// that needed blocks — everyone else keeps decoding.
    fn grow_for_decode_per_slot(&self, slots: &mut [&mut DecodeSlot]) -> Result<Vec<usize>> {
        let block_size = self.block_manager_guard()?.block_size();
        let mut starved = Vec::new();
        for (idx, slot) in slots.iter_mut().enumerate() {
            let DecodeSlot::Real {
                state,
                first_token_pending: false,
                ..
            } = &mut **slot
            else {
                continue;
            };
            let required_blocks =
                blocks_needed_for_tokens(state.seq_len.saturating_add(1), block_size);
            let missing = required_blocks.saturating_sub(state.block_table.blocks.len());
            if missing == 0 {
                continue;
            }
            let allocated = {
                let mut bm_guard = self.block_manager_guard()?;
                bm_guard.allocate(missing)
            };
            match allocated {
                Ok(new_blocks) => {
                    state.block_table.blocks.extend(new_blocks.iter().copied());
                    state.allocated_blocks.extend(new_blocks.iter().copied());
                }
                Err(kiln_core::block::BlockError::OutOfMemory { .. }) => {
                    starved.push(idx);
                }
            }
        }
        Ok(starved)
    }
}

impl DecodeForward for RealDecodeForward {
    fn supports_resumable_prefill(&self) -> bool {
        true
    }

    fn grow_for_decode(&self, slots: &mut [&mut DecodeSlot]) -> Result<Vec<usize>> {
        self.grow_for_decode_per_slot(slots)
    }

    fn can_reuse_as_strict_prefix(&self, prompt_token_len: usize) -> bool {
        self.prefix_cache
            .lock()
            .map(|cache| cache.can_register_strict_prefix_len(prompt_token_len))
            .unwrap_or(false)
    }

    fn has_inflight_prefill_layer_progress(&self, slot: &DecodeSlot) -> bool {
        matches!(
            slot,
            DecodeSlot::RealPrefill {
                state: Some(state),
                ..
            } if state.has_pending_layer_progress()
        )
    }

    fn inflight_prefill_token_width(&self, slot: &DecodeSlot) -> Option<usize> {
        let DecodeSlot::RealPrefill {
            state: Some(state), ..
        } = slot
        else {
            return None;
        };
        state.pending_layer_chunk_tokens()
    }

    fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
        let mut preparation = self.prepare_request_chunked(req, usize::MAX)?;
        loop {
            preparation = match preparation {
                RequestPreparation::Ready { slot, .. } => return Ok(slot),
                RequestPreparation::Prefilling { slot, .. } => {
                    self.advance_prefill(slot, usize::MAX, usize::MAX, &req.sampling, &req.cancel)?
                }
            };
        }
    }

    fn prepare_request_chunked(
        &self,
        req: &EngineRequest,
        _max_tokens: usize,
    ) -> Result<RequestPreparation> {
        let loaded = self
            .loaded_adapter
            .read()
            .map_err(|error| anyhow::anyhow!("loaded adapter identity lock poisoned: {error}"))?;
        anyhow::ensure!(
            *loaded == req.adapter,
            "queued request adapter revision is stale: expected {:?}, loaded {:?}",
            req.adapter,
            *loaded
        );
        drop(loaded);
        let gpu_guard = gpu_coordination_read_guard(&self.gpu_lock);
        let runner_guard = self.runner_guard()?;
        let prefix_cache_enabled = self.prefix_cache_guard()?.is_enabled();
        let lookup = if prefix_cache_enabled {
            let pending_lookup = match RealPrefixCacheRequest::begin(
                &self.prefix_cache,
                &self.block_manager,
                req.adapter.clone(),
                &req.prompt_tokens,
                &req.sampling,
            ) {
                Ok(lookup) => lookup,
                Err(failure) => {
                    let error = failure.settle(&runner_guard);
                    if runner_guard.backend_health_snapshot().quarantined {
                        std::mem::forget(gpu_guard);
                    }
                    return Err(error);
                }
            };
            let lookup = match pending_lookup.settle(&runner_guard) {
                Ok(lookup) => lookup,
                Err(error) => {
                    std::mem::forget(gpu_guard);
                    return Err(error);
                }
            };
            Some(lookup)
        } else {
            None
        };
        let (prefix_request, hit) = lookup
            .map(|lookup| (Some(lookup.request), lookup.hit))
            .unwrap_or((None, None));
        let cached_prefix = hit.map(|hit| PagedPrefixReuse {
            cached_tokens: hit.cached_tokens,
            block_ids: hit.block_ids,
            linear_state: hit.linear_state,
            next_token: hit.next_token,
        });

        let prepared = runner_guard.begin_paged_batched_decode_with_prefix_cache(
            &req.prompt_tokens,
            &req.sampling,
            self.block_manager.as_ref(),
            self.paged_cache.as_ref(),
            cached_prefix,
            prefix_cache_enabled,
            Some(&req.cancel),
        );
        let synchronized =
            runner_guard.synchronize_external_yield("batched request prefill initialization");
        drop(runner_guard);
        if let Err(err) = synchronized {
            std::mem::forget(prepared);
            std::mem::forget(prefix_request);
            std::mem::forget(gpu_guard);
            return Err(err);
        }

        match prepared {
            Ok(PagedBatchedPrefillStart::Ready(state)) => Ok(RequestPreparation::Ready {
                slot: DecodeSlot::Real {
                    state,
                    prefix_request,
                    first_token_pending: true,
                },
                tokens_processed: 0,
                layers_processed: 0,
            }),
            Ok(PagedBatchedPrefillStart::Prefilling(state)) => Ok(RequestPreparation::Prefilling {
                slot: DecodeSlot::RealPrefill {
                    state: Some(state),
                    prefix_request,
                },
                tokens_processed: 0,
                layers_processed: 0,
            }),
            Err(err) => Err(err),
        }
    }

    fn advance_prefill(
        &self,
        slot: DecodeSlot,
        max_tokens: usize,
        max_layers: usize,
        sampling: &SamplingParams,
        cancel: &CancelHandle,
    ) -> Result<RequestPreparation> {
        let DecodeSlot::RealPrefill {
            mut state,
            prefix_request,
        } = slot
        else {
            anyhow::bail!("non-prefill slot sent to resumable prefill")
        };
        let gpu_guard = gpu_coordination_read_guard(&self.gpu_lock);
        let runner_guard = match self.runner_guard() {
            Ok(runner) => runner,
            Err(error) => {
                std::mem::forget(state);
                std::mem::forget(prefix_request);
                std::mem::forget(gpu_guard);
                return Err(error);
            }
        };
        let progress = runner_guard.advance_paged_batched_prefill_with_layer_budget(
            &mut state,
            sampling,
            self.paged_cache.as_ref(),
            max_tokens,
            max_layers,
            Some(cancel),
        );
        let synchronized = runner_guard.synchronize_external_yield("batched prefill quantum");
        drop(runner_guard);
        if let Err(error) = synchronized {
            std::mem::forget(progress);
            std::mem::forget(state);
            std::mem::forget(prefix_request);
            std::mem::forget(gpu_guard);
            return Err(error);
        }

        match progress {
            Ok(progress) => match progress.decode_state {
                Some(decode_state) => Ok(RequestPreparation::Ready {
                    slot: DecodeSlot::Real {
                        state: decode_state,
                        prefix_request,
                        first_token_pending: true,
                    },
                    tokens_processed: progress.tokens_processed,
                    layers_processed: progress.layers_processed,
                }),
                None => Ok(RequestPreparation::Prefilling {
                    slot: DecodeSlot::RealPrefill {
                        state,
                        prefix_request,
                    },
                    tokens_processed: progress.tokens_processed,
                    layers_processed: progress.layers_processed,
                }),
            },
            Err(error) => {
                let allocated_blocks = state
                    .take()
                    .map(PagedBatchedPrefillState::into_allocated_blocks)
                    .unwrap_or_default();
                if let Err(cleanup_error) =
                    self.finish_prefix_resources(None, Vec::new(), allocated_blocks, prefix_request)
                {
                    return Err(anyhow::anyhow!(
                        "{error:#}; resumable prefill cleanup also failed: {cleanup_error:#}"
                    ));
                }
                Err(error)
            }
        }
    }

    fn forward_decode(
        &self,
        slots: &mut [&mut DecodeSlot],
        sampling: &[SamplingParams],
    ) -> Result<Vec<TokenId>> {
        self.grow_ready_decode_slots(slots)?;
        let mut output = vec![0; slots.len()];
        let (decode_indices, decode_params) =
            collect_ready_decode_indices(slots, sampling, &mut output)?;

        if !decode_indices.is_empty() {
            let gpu_guard = gpu_coordination_read_guard(&self.gpu_lock);
            let mut row_refs: Vec<&mut PagedBatchedDecodeState> =
                Vec::with_capacity(decode_indices.len());
            let mut next_decode_index = decode_indices.iter().copied().peekable();
            for (idx, slot) in slots.iter_mut().enumerate() {
                if next_decode_index.peek() != Some(&idx) {
                    continue;
                }
                match &mut **slot {
                    DecodeSlot::Real {
                        state,
                        first_token_pending: false,
                        ..
                    } => {
                        row_refs.push(state);
                        next_decode_index.next();
                    }
                    DecodeSlot::Real { .. } => {
                        anyhow::bail!("decode row {idx} became first-token pending")
                    }
                    DecodeSlot::RealPrefill { .. } => {
                        anyhow::bail!("prefilling row {idx} entered decode batch")
                    }
                    DecodeSlot::Mock { .. } => {
                        anyhow::bail!("mock slot sent to real decode forward")
                    }
                }
            }
            anyhow::ensure!(
                row_refs.len() == decode_params.len(),
                "decode row length {} != params length {} after row selection",
                row_refs.len(),
                decode_params.len()
            );
            let runner_guard = self.runner_guard()?;
            let decode_result = (|| -> Result<Vec<TokenId>> {
                if self.rowwise_decode && row_refs.len() > 1 {
                    // Operator-forced comparison/fallback path: dispatch one
                    // single-row forward per active slot instead of one batched
                    // decode step.
                    let mut tokens = Vec::with_capacity(row_refs.len());
                    for (row, params) in row_refs.iter_mut().zip(decode_params.iter()) {
                        let mut single_row: [&mut PagedBatchedDecodeState; 1] = [&mut **row];
                        let single_params = std::slice::from_ref(params);
                        let mut next = runner_guard.paged_batched_decode_step(
                            &mut single_row,
                            single_params,
                            self.paged_cache.as_ref(),
                        )?;
                        anyhow::ensure!(
                            next.len() == 1,
                            "rowwise decode returned {} tokens for a 1-row step",
                            next.len()
                        );
                        tokens.push(next.remove(0));
                    }
                    Ok(tokens)
                } else {
                    runner_guard.paged_batched_decode_step(
                        &mut row_refs,
                        &decode_params,
                        self.paged_cache.as_ref(),
                    )
                }
            })();
            let synchronized = runner_guard.synchronize_external_yield("batched decode step");
            drop(runner_guard);
            if let Err(err) = synchronized {
                std::mem::forget(gpu_guard);
                return Err(err);
            }
            let next_tokens = decode_result?;
            for (idx, token) in decode_indices.into_iter().zip(next_tokens) {
                output[idx] = token;
            }
        }

        Ok(output)
    }

    fn is_eos_token(&self, token: TokenId) -> Result<bool> {
        Ok(self.runner_guard()?.is_eos_token(token))
    }

    fn stop_reason_after_emit(
        &self,
        generated_tokens: &[TokenId],
        sampling: &SamplingParams,
    ) -> Result<Option<FinishReason>> {
        if let Some(stop) = self
            .runner_guard()?
            .stop_sequence_match(generated_tokens, sampling)?
        {
            return Ok(Some(FinishReason::StopSequence(stop)));
        }
        if generated_tokens.len() >= sampling.max_tokens {
            Ok(Some(FinishReason::MaxTokens))
        } else {
            Ok(None)
        }
    }

    fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize> {
        let DecodeSlot::Real { state, .. } = slot else {
            anyhow::bail!("mock slot sent to real accept_token");
        };
        state.generated_tokens.push(token);
        state.next_token = token;
        if let Some(seed) = state.step_seed.as_mut() {
            *seed = seed.wrapping_add(1);
        }
        Ok(state.generated_tokens.len())
    }

    fn finish_request(
        &self,
        slot: DecodeSlot,
        finish_reason: FinishReason,
    ) -> Result<DecodeForwardOutput> {
        let DecodeSlot::Real {
            state,
            prefix_request,
            ..
        } = slot
        else {
            anyhow::bail!("mock slot sent to real finish_request");
        };
        let fallback_allocated = state.allocated_blocks.clone();
        let (runner, runner_poisoned) = self.runner_guard_for_finish();
        if let Err(err) = runner.ensure_backend_healthy() {
            drop(runner);
            std::mem::forget(state);
            std::mem::forget(prefix_request);
            return Err(err.context(
                "batched request ownership quarantined instead of releasing unknown GPU state",
            ));
        }
        let finished = runner.finish_paged_batched_decode(state, finish_reason);
        drop(runner);

        match finished {
            Ok(mut output) if !runner_poisoned => {
                let prefill_duration = output.prefill_duration;
                let decode_duration = output.decode_duration;
                self.free_uncached_blocks(&mut output, prefix_request)?;
                Ok(DecodeForwardOutput {
                    output: output.output,
                    prefill_duration,
                    decode_duration,
                })
            }
            Ok(mut output) => {
                let cleanup = self.finish_prefix_resources(
                    None,
                    Vec::new(),
                    std::mem::take(&mut output.allocated_blocks),
                    prefix_request,
                );
                cleanup?;
                anyhow::bail!(
                    "model runner lock was poisoned; output discarded after ownership cleanup"
                )
            }
            Err(err) => {
                if let Err(cleanup_err) = self.finish_prefix_resources(
                    None,
                    Vec::new(),
                    fallback_allocated,
                    prefix_request,
                ) {
                    return Err(anyhow::anyhow!(
                        "{err:#}; request ownership cleanup also failed: {cleanup_err:#}"
                    ));
                }
                Err(err)
            }
        }
    }

    fn discard_request(&self, slot: DecodeSlot) {
        let slot = match slot {
            DecodeSlot::RealPrefill {
                mut state,
                prefix_request,
            } => {
                if self.backend_health.snapshot().quarantined {
                    std::mem::forget(state);
                    std::mem::forget(prefix_request);
                    tracing::error!("discarded prefill ownership retained with unhealthy backend");
                    return;
                }
                let allocated_blocks = state
                    .take()
                    .map(PagedBatchedPrefillState::into_allocated_blocks)
                    .unwrap_or_default();
                if let Err(error) =
                    self.finish_prefix_resources(None, Vec::new(), allocated_blocks, prefix_request)
                {
                    tracing::warn!(
                        error = %format!("{error:#}"),
                        "failed to free discarded prefill blocks"
                    );
                }
                return;
            }
            slot => slot,
        };
        if let DecodeSlot::Real {
            state,
            prefix_request,
            ..
        } = slot
        {
            // Route through the SAME finish path as a completed request so
            // the prefill's prefix-cache registration survives. The old
            // shape here dropped `state.registration` on the floor
            // (registration: None), so a cancelled request — every pi
            // ESC/steer — paid a full re-prefill of the identical
            // multi-thousand-token context on the very next message.
            let fallback_allocated = state.allocated_blocks.clone();
            let (runner, runner_poisoned) = self.runner_guard_for_finish();
            if let Err(err) = runner.ensure_backend_healthy() {
                drop(runner);
                std::mem::forget(state);
                std::mem::forget(prefix_request);
                tracing::error!(
                    error = %format!("{err:#}"),
                    "discarded request ownership quarantined with unhealthy backend"
                );
                return;
            }
            let finished = runner.finish_paged_batched_decode(state, FinishReason::MaxTokens);
            drop(runner);
            let cleanup = match finished {
                Ok(mut output) if !runner_poisoned => {
                    self.free_uncached_blocks(&mut output, prefix_request)
                }
                Ok(mut output) => self.finish_prefix_resources(
                    None,
                    Vec::new(),
                    std::mem::take(&mut output.allocated_blocks),
                    prefix_request,
                ),
                Err(err) => {
                    tracing::warn!(
                        error = %format!("{err:#}"),
                        "failed to materialize discarded request output; releasing private ownership"
                    );
                    self.finish_prefix_resources(
                        None,
                        Vec::new(),
                        fallback_allocated,
                        prefix_request,
                    )
                }
            };
            if let Err(err) = cleanup {
                tracing::warn!(
                    error = %format!("{err:#}"),
                    "failed to free discarded request blocks"
                );
            }
        }
    }

    fn kv_num_blocks(&self) -> Option<usize> {
        Some(self.paged_cache.num_blocks())
    }

    fn resize_kv(&self, target_blocks: usize) -> Result<usize> {
        anyhow::ensure!(
            self.allow_dynamic_kv_resize,
            "physical KV resize is prohibited by the active serving profile"
        );
        // Check before taking the exclusive GPU lock: quarantine deliberately
        // retains a read owner forever, so entering the write wait would hang.
        drop(self.runner_guard()?);
        let device = match self.paged_cache.device() {
            Some(d) => d,
            None => return Ok(0),
        };
        let cur = self.paged_cache.num_blocks();
        if target_blocks == cur || target_blocks == 0 {
            return Ok(cur);
        }
        // EXCLUSIVE GPU access for the pool swap: the write guard blocks BOTH
        // decode actors (they hold the read guard) and any training step (also a
        // write guard) until the resize completes — so no kernel is reading a
        // pool we are about to drop. Combined with the device-sync inside
        // `physical_resize_to`, the swap is race-free. Resize is rare
        // (governor-driven under pressure), so the brief decode stall is fine.
        let _gpu =
            gpu_coordination_write_guard_while_healthy(&self.gpu_lock, &self.backend_health)?;
        if target_blocks < cur {
            // SHRINK. Logical first: lower the ceiling + retire free high blocks.
            // We can only physically drop to the live high-water mark right now;
            // if requests still hold high blocks the shrink stops there, and a
            // later resize finishes it once they drain.
            let achievable = {
                let mut bm = self.block_manager_guard()?;
                bm.set_target_usable(target_blocks);
                let achievable = target_blocks.max(bm.physical_floor());
                bm.physical_truncate(achievable)
                    .map_err(|e| anyhow::anyhow!("kv shrink truncate to {achievable}: {e}"))?;
                achievable
            };
            self.paged_cache.physical_resize_to(achievable, device)?;
            tracing::info!(
                from = cur,
                to = achievable,
                target = target_blocks,
                "KV cache physically shrunk (VRAM returned to pool for reuse)"
            );
            Ok(achievable)
        } else {
            // GROW. Physical first (alloc bigger, copy existing KV), then publish
            // the new blocks to the manager and raise the ceiling.
            self.paged_cache.physical_resize_to(target_blocks, device)?;
            {
                let mut bm = self.block_manager_guard()?;
                bm.physical_grow(target_blocks);
                bm.set_target_usable(target_blocks);
            }
            tracing::info!(from = cur, to = target_blocks, "KV cache physically grown");
            Ok(target_blocks)
        }
    }
}

#[derive(Clone)]
pub struct BatchingEngineHandle {
    tx: mpsc::Sender<EngineCommand>,
    published_snapshot: SharedBatchingEngineSnapshot,
}

impl BatchingEngineHandle {
    pub fn start(forward: Arc<dyn DecodeForward>) -> Self {
        Self::start_with_options(forward, env_max_decode_batch_for_policy(None))
    }

    pub fn start_with_options(forward: Arc<dyn DecodeForward>, max_decode_batch: usize) -> Self {
        Self::start_with_backend_options(
            forward,
            max_decode_batch,
            None,
            ResponseDeliveryPolicy::default(),
        )
    }

    pub fn start_with_backend_options(
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        policy: Option<DecodeBatcherPolicy>,
        response_delivery_policy: ResponseDeliveryPolicy,
    ) -> Self {
        Self::start_with_runtime_options(
            forward,
            max_decode_batch,
            policy,
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            response_delivery_policy,
        )
    }

    pub fn start_with_runtime_options(
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        policy: Option<DecodeBatcherPolicy>,
        max_batch_tokens: BatchTokenBudget,
        max_prefill_tokens_per_cycle: PrefillTokenBudget,
        max_prefill_layers_per_cycle: PrefillLayerBudget,
        response_delivery_policy: ResponseDeliveryPolicy,
    ) -> Self {
        let max_decode_batch = max_decode_batch.max(1);
        Self::start_with_policy(
            forward,
            max_decode_batch,
            max_batch_tokens,
            max_prefill_tokens_per_cycle,
            max_prefill_layers_per_cycle,
            env_prefix_aware_admission(),
            env_prefill_admission_quantum_for_policy(max_decode_batch, policy),
            policy.is_some_and(|policy| policy.burst_prefill_admission),
            response_delivery_policy,
        )
    }

    fn start_with_policy(
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        max_batch_tokens: BatchTokenBudget,
        max_prefill_tokens_per_cycle: PrefillTokenBudget,
        max_prefill_layers_per_cycle: PrefillLayerBudget,
        prefix_aware_admission: bool,
        prefill_admission_quantum: usize,
        burst_refill: bool,
        response_delivery_policy: ResponseDeliveryPolicy,
    ) -> Self {
        let (tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let (delivery_result_tx, delivery_results) = std_mpsc::channel();
        let delivery_worker = DeliveryWorker::start(
            response_delivery_policy.stream_stall_grace,
            RESPONSE_DELIVERY_POLL_CADENCE,
            EngineDeliveryResultSink {
                result_tx: delivery_result_tx,
                pending_results: Vec::new(),
                engine_tx: tx.downgrade(),
            },
        )
        .expect("spawn response delivery worker");
        let actor = BatchingEngineActor::new(
            rx,
            forward,
            max_decode_batch.max(1),
            max_batch_tokens,
            max_prefill_tokens_per_cycle,
            max_prefill_layers_per_cycle,
            prefix_aware_admission,
            prefill_admission_quantum,
            burst_refill,
            response_delivery_policy,
            delivery_worker,
            delivery_results,
        );
        let published_snapshot = actor.published_snapshot.clone();
        thread::Builder::new()
            .name("kiln-batching-engine".to_string())
            .spawn(move || actor.run())
            .expect("spawn batching engine actor");
        Self {
            tx,
            published_snapshot,
        }
    }

    pub async fn enqueue(&self, req: EngineRequest) -> Result<mpsc::Receiver<EngineEvent>> {
        let (response_tx, response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
        self.tx
            .send(EngineCommand::Enqueue { req, response_tx })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        Ok(response_rx)
    }

    #[cfg(test)]
    async fn enqueue_with_response_capacity(
        &self,
        req: EngineRequest,
        capacity: usize,
    ) -> Result<mpsc::Receiver<EngineEvent>> {
        let (response_tx, response_rx) = mpsc::channel(capacity);
        self.tx
            .send(EngineCommand::Enqueue { req, response_tx })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        Ok(response_rx)
    }

    pub async fn cancel(&self, request_id: Uuid) -> Result<()> {
        self.tx
            .send(EngineCommand::Cancel { request_id })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))
    }

    pub async fn drain(&self) -> Result<()> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::Drain { reply })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("batching engine stopped during drain"))
    }

    pub async fn stop(&self) -> Result<()> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::Stop { reply })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("batching engine stopped before ack"))
    }

    pub async fn snapshot(&self) -> Result<BatchingEngineSnapshot> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::Snapshot { reply })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("batching engine stopped before snapshot"))?
            .map_err(anyhow::Error::msg)
    }

    /// Return the latest snapshot published by the actor without enqueueing a
    /// command or waiting for a model forward to finish.
    ///
    /// Cheap actor fields are published at state boundaries and immediately
    /// before a decode forward. `snapshot_age_ms` exposes how long the actor has
    /// been inside a forward (or otherwise unable to publish). The expensive
    /// `prefix_deferred_waiting` field is the last sample taken by
    /// [`Self::snapshot`], drain, or stop and may be older than the cheap fields.
    pub fn cached_snapshot(&self) -> BatchingEngineSnapshot {
        let published = self
            .published_snapshot
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut snapshot = published.snapshot.clone();
        snapshot.snapshot_age_ms = duration_millis_saturating(published.published_at.elapsed());
        snapshot
    }

    /// Physically resize the KV cache to `target_blocks` (#26). Returns the
    /// achieved block count. Async variant for request-handler / API callers.
    pub async fn resize_kv(&self, target_blocks: usize) -> Result<usize> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::ResizeKv {
                target_blocks,
                reply,
            })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("batching engine stopped before resize ack"))?
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Blocking variant of [`Self::resize_kv`] for the memory governor's
    /// (non-async) monitor thread.
    pub fn resize_kv_blocking(&self, target_blocks: usize) -> Result<usize> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .blocking_send(EngineCommand::ResizeKv {
                target_blocks,
                reply,
            })
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.blocking_recv()
            .map_err(|_| anyhow::anyhow!("batching engine stopped before resize ack"))?
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Run `swap` at the engine's between-requests barrier: admission pauses
    /// and the closure executes only once every in-flight request has
    /// finished, so generation never continues across a weight change (KV
    /// computed under the old weights + decode steps under the new weights
    /// is silent garbage). Queued requests resume right after.
    pub async fn swap_adapter(&self, swap: AdapterSwapClosure) -> Result<()> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::SwapAdapter { swap, reply })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("batching engine stopped before swap ack"))?
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Health-aware adapter barrier. If inference quarantines while the actor
    /// drains its active batch, the caller returns promptly. The queued swap
    /// closure must still recheck the same latch before mutating weights.
    pub async fn swap_adapter_while_healthy(
        &self,
        swap: AdapterSwapClosure,
        backend_health: &BackendHealthHandle,
    ) -> Result<()> {
        let (reply, mut rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::SwapAdapter { swap, reply })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        loop {
            backend_health.ensure_healthy()?;
            tokio::select! {
                result = &mut rx => {
                    return result
                        .map_err(|_| anyhow::anyhow!("batching engine stopped before swap ack"))?
                        .map_err(|error| anyhow::anyhow!(error));
                }
                _ = tokio::time::sleep(Duration::from_millis(5)) => {}
            }
        }
    }

    /// Blocking variant of [`Self::swap_adapter`] for the training worker's
    /// (non-async) job thread.
    pub fn swap_adapter_blocking(&self, swap: AdapterSwapClosure) -> Result<()> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .blocking_send(EngineCommand::SwapAdapter { swap, reply })
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.blocking_recv()
            .map_err(|_| anyhow::anyhow!("batching engine stopped before swap ack"))?
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Blocking counterpart of [`Self::swap_adapter_while_healthy`].
    pub fn swap_adapter_blocking_while_healthy(
        &self,
        swap: AdapterSwapClosure,
        backend_health: &BackendHealthHandle,
    ) -> Result<()> {
        let (reply, mut rx) = oneshot::channel();
        self.tx
            .blocking_send(EngineCommand::SwapAdapter { swap, reply })
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        loop {
            backend_health.ensure_healthy()?;
            match rx.try_recv() {
                Ok(result) => return result.map_err(|error| anyhow::anyhow!(error)),
                Err(oneshot::error::TryRecvError::Empty) => {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(oneshot::error::TryRecvError::Closed) => {
                    anyhow::bail!("batching engine stopped before swap ack")
                }
            }
        }
    }
}

/// Deferred adapter-swap work executed on the engine thread at the
/// between-requests barrier. The closure owns everything it needs (runner
/// handle, pre-loaded LoRA weights, cache handles) so the engine stays
/// ignorant of LoRA specifics.
pub type AdapterSwapClosure = Box<dyn FnOnce() -> std::result::Result<(), String> + Send>;

enum EngineCommand {
    Enqueue {
        req: EngineRequest,
        response_tx: mpsc::Sender<EngineEvent>,
    },
    Cancel {
        request_id: Uuid,
    },
    DeliveryWake,
    Drain {
        reply: oneshot::Sender<()>,
    },
    Stop {
        reply: oneshot::Sender<()>,
    },
    Snapshot {
        reply: oneshot::Sender<std::result::Result<BatchingEngineSnapshot, String>>,
    },
    /// Physically resize the KV cache to `target_blocks` usable blocks (#26).
    /// Handled at the between-steps barrier so no forward is in flight.
    ResizeKv {
        target_blocks: usize,
        reply: oneshot::Sender<std::result::Result<usize, String>>,
    },
    /// Swap adapter weights at the between-REQUESTS barrier: queued until
    /// every active request has finished (admission pauses meanwhile), then
    /// executed on the engine thread with no forward in flight anywhere.
    SwapAdapter {
        swap: AdapterSwapClosure,
        reply: oneshot::Sender<std::result::Result<(), String>>,
    },
}

/// A queued adapter swap waiting for the active batch to drain.
struct PendingAdapterSwap {
    swap: AdapterSwapClosure,
    reply: oneshot::Sender<std::result::Result<(), String>>,
}

struct QueuedRequest {
    req: EngineRequest,
    delivery_key: DeliveryKey,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveDeliveryState {
    Ready,
    InFlight { sequence: u64 },
}

struct ActiveRequest {
    req: EngineRequest,
    delivery_key: DeliveryKey,
    delivery_state: ActiveDeliveryState,
    next_delivery_sequence: u64,
    slot: DecodeSlot,
}

#[derive(Default)]
struct AdmissionOutcome {
    submitted_first_tokens: bool,
    tokens_processed: usize,
}

struct EngineDeliveryResultSink {
    // One worker flush becomes one channel item. The actor cannot observe a
    // prefix of a decode cohort and accidentally launch a narrower forward.
    result_tx: std_mpsc::Sender<Vec<DeliveryResult>>,
    pending_results: Vec<DeliveryResult>,
    engine_tx: mpsc::WeakSender<EngineCommand>,
}

impl DeliveryResultSink for EngineDeliveryResultSink {
    fn try_send(&mut self, result: DeliveryResult) -> Result<(), DeliveryResultSinkError> {
        self.pending_results.push(result);
        Ok(())
    }

    fn notify(&mut self) -> Result<(), DeliveryResultNotifyError> {
        let results = std::mem::take(&mut self.pending_results);
        self.result_tx
            .send(results)
            .map_err(|_| DeliveryResultNotifyError)?;
        let Some(engine_tx) = self.engine_tx.upgrade() else {
            return Ok(());
        };
        match engine_tx.try_send(EngineCommand::DeliveryWake) {
            Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(DeliveryResultNotifyError),
        }
    }
}

struct BatchingEngineActor {
    rx: mpsc::Receiver<EngineCommand>,
    forward: Arc<dyn DecodeForward>,
    waiting: VecDeque<QueuedRequest>,
    active: Vec<ActiveRequest>,
    accepting: bool,
    stopped: bool,
    max_decode_batch: usize,
    max_batch_tokens: usize,
    max_prefill_tokens_per_cycle: usize,
    max_prefill_layers_per_cycle: usize,
    next_prefill_index: usize,
    prefix_aware_admission: bool,
    max_prefill_admissions_per_cycle: usize,
    // #1082 CUDA concurrency regression: when true, admit_waiting refills the
    // decode batch toward `max_decode_batch` every cycle (the LKG 2d9d4fc4
    // burst-fill that scaled CUDA to ~498 tok/s @ bs=64) instead of yielding to
    // ready decode rows by admitting only 1 prefill/cycle. The `has_ready_decode_row`
    // 1/cycle yield (added for Vulkan by 2c52565d) starves the SUSTAINED batch
    // width on CUDA (max_decode_batch=8): inline prefills interleave into every
    // decode cycle so a wide batch never forms, producing the measured
    // anti-scaling (n=32 slower than n=8). Set only for CUDA; Vulkan/Metal keep
    // their tuned yield behavior.
    burst_refill: bool,
    response_delivery_policy: ResponseDeliveryPolicy,
    delivery_worker: Option<DeliveryWorker>,
    delivery_results: std_mpsc::Receiver<Vec<DeliveryResult>>,
    next_delivery_generation: u64,
    delivery_backpressured: HashSet<(DeliveryKey, u64)>,
    delivery_pending_terminal: HashSet<DeliveryKey>,
    delivery_outbox: Vec<(DeliveryKey, DeliveryBatch)>,
    defer_delivery_flush: bool,
    stop_replies: Vec<oneshot::Sender<()>>,
    /// Adapter swaps waiting for the active batch to drain. While any swap
    /// is pending, admission pauses (waiting requests stay queued) so the
    /// barrier is reached promptly; swaps then run FIFO on this thread.
    pending_swaps: VecDeque<PendingAdapterSwap>,
    snapshot: BatchingEngineSnapshot,
    published_snapshot: SharedBatchingEngineSnapshot,
}

impl BatchingEngineActor {
    fn new(
        rx: mpsc::Receiver<EngineCommand>,
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        max_batch_tokens: BatchTokenBudget,
        max_prefill_tokens_per_cycle: PrefillTokenBudget,
        max_prefill_layers_per_cycle: PrefillLayerBudget,
        prefix_aware_admission: bool,
        prefill_admission_quantum: usize,
        burst_refill: bool,
        response_delivery_policy: ResponseDeliveryPolicy,
        delivery_worker: DeliveryWorker,
        delivery_results: std_mpsc::Receiver<Vec<DeliveryResult>>,
    ) -> Self {
        let configured_max_decode_batch = max_decode_batch.max(1);
        let max_decode_batch = configured_max_decode_batch
            .min(max_batch_tokens.tokens())
            .max(1);
        if max_decode_batch != configured_max_decode_batch {
            tracing::warn!(
                configured_max_decode_batch,
                effective_max_decode_batch = max_decode_batch,
                max_batch_tokens = max_batch_tokens.tokens(),
                "decode width reduced to the combined per-cycle token budget"
            );
        }
        let max_prefill_admissions_per_cycle = prefill_admission_quantum.clamp(1, max_decode_batch);
        let max_prefill_tokens_per_cycle_source = max_prefill_tokens_per_cycle.source();
        let configured_max_prefill_tokens_per_cycle = max_prefill_tokens_per_cycle.tokens();
        let max_prefill_tokens_per_cycle = configured_max_prefill_tokens_per_cycle
            .min(max_batch_tokens.tokens())
            .max(1);
        if max_prefill_tokens_per_cycle != configured_max_prefill_tokens_per_cycle {
            tracing::info!(
                configured_max_prefill_tokens_per_cycle,
                effective_max_prefill_tokens_per_cycle = max_prefill_tokens_per_cycle,
                max_batch_tokens = max_batch_tokens.tokens(),
                "prefill token ceiling reduced to the combined actor-cycle budget"
            );
        }
        let max_prefill_layers_per_cycle_source = max_prefill_layers_per_cycle.source();
        let max_prefill_layers_per_cycle = max_prefill_layers_per_cycle.layers();
        let snapshot = BatchingEngineSnapshot {
            accepting: true,
            max_batch_tokens: max_batch_tokens.tokens(),
            max_batch_tokens_source: max_batch_tokens.source(),
            max_prefill_tokens_per_cycle,
            max_prefill_tokens_per_cycle_source,
            max_prefill_layers_per_cycle,
            max_prefill_layers_per_cycle_source,
            max_prefill_admission_quantum: max_prefill_admissions_per_cycle,
            stream_stall_grace_ms: duration_millis_saturating(
                response_delivery_policy.stream_stall_grace,
            ),
            stream_stall_grace_source: response_delivery_policy.stream_stall_grace_source,
            ..BatchingEngineSnapshot::default()
        };
        let published_snapshot = Arc::new(RwLock::new(PublishedBatchingEngineSnapshot {
            snapshot: snapshot.clone(),
            published_at: Instant::now(),
        }));
        Self {
            rx,
            forward,
            waiting: VecDeque::new(),
            active: Vec::new(),
            accepting: true,
            stopped: false,
            max_decode_batch,
            max_batch_tokens: max_batch_tokens.tokens(),
            max_prefill_tokens_per_cycle,
            max_prefill_layers_per_cycle,
            next_prefill_index: 0,
            prefix_aware_admission,
            max_prefill_admissions_per_cycle,
            burst_refill,
            response_delivery_policy,
            delivery_worker: Some(delivery_worker),
            delivery_results,
            next_delivery_generation: 0,
            delivery_backpressured: HashSet::new(),
            delivery_pending_terminal: HashSet::new(),
            delivery_outbox: Vec::new(),
            defer_delivery_flush: false,
            stop_replies: Vec::new(),
            pending_swaps: VecDeque::new(),
            snapshot,
            published_snapshot,
        }
    }

    fn run(mut self) {
        while !self.stopped {
            self.drain_delivery_results();
            if self.stopped {
                break;
            }
            // Between-requests barrier: with no decode step in flight and
            // the active batch drained, queued adapter swaps execute now —
            // before blocking on the channel, so a swap queued behind a
            // just-finished batch doesn't wait for the next command.
            self.run_pending_swaps_at_barrier();

            if self.active.is_empty() && self.waiting.is_empty() && self.pending_swaps.is_empty() {
                match self.rx.blocking_recv() {
                    Some(cmd) => self.handle_command(cmd),
                    None => break,
                }
                // A swap that arrived while idle executes immediately.
                self.run_pending_swaps_at_barrier();
                if self.stopped {
                    break;
                }
            }

            // Only sleep when we have nothing to do. Sleeping unconditionally
            // before every decode step adds ~5% c=1 regression and starves
            // already-active rows of GPU time. With active work, drain commands
            // non-blockingly, admit any new arrivals, and run the decode step
            // immediately.
            if self.active.is_empty() {
                thread::sleep(Duration::from_millis(1));
            }
            self.drain_commands();
            self.drain_delivery_results();
            if self.stopped {
                break;
            }
            // Existing decode rows reserve their one-token steps before any
            // fallback forward may do admission-time prompt work. Production
            // resumable admission is allocation-only and reports zero tokens.
            let decode_reservation = self
                .ready_decode_indices_with_limit(self.max_batch_tokens)
                .len();
            let admission_budget = self.max_batch_tokens.saturating_sub(decode_reservation);
            let admission_budget = admission_budget.min(self.max_prefill_tokens_per_cycle);
            let admission = self.admit_waiting_with_budget(admission_budget);
            if self.stopped {
                break;
            }
            if admission.submitted_first_tokens {
                let barrier = self
                    .delivery_worker
                    .as_ref()
                    .ok_or(DeliveryBarrierError)
                    .and_then(DeliveryWorker::barrier);
                if let Err(error) = barrier {
                    self.accepting = false;
                    self.stopped = true;
                    tracing::error!(
                        error = %error,
                        "response delivery worker unavailable at admission barrier; stopping batching actor"
                    );
                    break;
                }
                self.drain_delivery_results();
                if self.stopped {
                    break;
                }
            }
            let decode_budget = self
                .max_batch_tokens
                .saturating_sub(admission.tokens_processed);
            let decoded_tokens = if decode_budget > 0 && self.has_ready_decode_row() {
                self.run_decode_batch_with_budget(decode_budget)
            } else {
                0
            };
            if self.stopped {
                break;
            }
            let prefill_budget = self
                .max_batch_tokens
                .saturating_sub(admission.tokens_processed)
                .saturating_sub(decoded_tokens)
                .min(
                    self.max_prefill_tokens_per_cycle
                        .saturating_sub(admission.tokens_processed),
                );
            let advanced_prefill = self.run_prefill_budget(prefill_budget);
            if decoded_tokens > 0 || advanced_prefill {
                continue;
            }

            // Every live row may be waiting for its one in-flight delivery
            // batch. Block for a delivery wakeup or control command instead
            // of issuing duplicate model work or spinning an empty loop.
            if !self.active.is_empty() {
                match self.rx.blocking_recv() {
                    Some(cmd) => self.handle_command(cmd),
                    None => break,
                }
                continue;
            }

            thread::sleep(Duration::from_millis(1));
        }

        // Reject new commands before collecting every Stop that was already
        // accepted. Otherwise a concurrent caller can enqueue Stop behind the
        // command that ended the run loop and wait forever for its reply.
        self.rx.close();
        while let Ok(command) = self.rx.try_recv() {
            if let EngineCommand::Stop { reply } = command {
                self.stop_replies.push(reply);
            }
        }
        self.fail_all("batching engine stopped");
        for reply in self.stop_replies.drain(..) {
            let _ = reply.send(());
        }
    }

    fn record_admission_duration(
        &mut self,
        elapsed: Duration,
        request_id: Uuid,
        prompt_tokens: usize,
        token_budget: usize,
    ) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.snapshot.last_admission_ms = elapsed_ms;
        self.snapshot.max_admission_ms = self.snapshot.max_admission_ms.max(elapsed_ms);
        self.snapshot.total_admission_ms += elapsed_ms;
        self.snapshot.total_admission_calls = self.snapshot.total_admission_calls.saturating_add(1);
        if elapsed >= SLOW_ACTOR_PHASE_THRESHOLD {
            self.snapshot.slow_admission_count =
                self.snapshot.slow_admission_count.saturating_add(1);
            tracing::warn!(
                event = "slow_batching_actor_phase",
                phase = "admission",
                %request_id,
                prompt_tokens,
                token_budget,
                elapsed_ms,
                threshold_ms = duration_millis_saturating(SLOW_ACTOR_PHASE_THRESHOLD),
                active_requests = self.active.len(),
                waiting_requests = self.waiting.len(),
                "slow_batching_actor_phase"
            );
        }
    }

    fn record_prefill_forward_duration(
        &mut self,
        elapsed: Duration,
        request_id: Uuid,
        token_budget: usize,
        layer_budget: usize,
    ) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.snapshot.last_prefill_ms = elapsed_ms;
        self.snapshot.max_prefill_forward_ms = self.snapshot.max_prefill_forward_ms.max(elapsed_ms);
        self.snapshot.total_prefill_forward_ms += elapsed_ms;
        if elapsed >= SLOW_ACTOR_PHASE_THRESHOLD {
            self.snapshot.slow_prefill_forward_count =
                self.snapshot.slow_prefill_forward_count.saturating_add(1);
            tracing::warn!(
                event = "slow_batching_actor_phase",
                phase = "prefill",
                %request_id,
                token_budget,
                layer_budget,
                elapsed_ms,
                threshold_ms = duration_millis_saturating(SLOW_ACTOR_PHASE_THRESHOLD),
                active_requests = self.active.len(),
                waiting_requests = self.waiting.len(),
                "slow_batching_actor_phase"
            );
        }
    }

    fn record_decode_forward_duration(&mut self, elapsed: Duration, batch_rows: usize) {
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.snapshot.last_forward_ms = elapsed_ms;
        self.snapshot.max_decode_forward_ms = self.snapshot.max_decode_forward_ms.max(elapsed_ms);
        self.snapshot.total_decode_forward_ms += elapsed_ms;
        if elapsed >= SLOW_ACTOR_PHASE_THRESHOLD {
            self.snapshot.slow_decode_forward_count =
                self.snapshot.slow_decode_forward_count.saturating_add(1);
            tracing::warn!(
                event = "slow_batching_actor_phase",
                phase = "decode",
                batch_rows,
                elapsed_ms,
                threshold_ms = duration_millis_saturating(SLOW_ACTOR_PHASE_THRESHOLD),
                active_requests = self.active.len(),
                waiting_requests = self.waiting.len(),
                "slow_batching_actor_phase"
            );
        }
    }

    fn drain_commands(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(cmd) => self.handle_command(cmd),
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    self.accepting = false;
                    self.stopped = true;
                    break;
                }
            }
        }
    }

    fn has_ready_decode_row(&self) -> bool {
        self.active.iter().any(|active| {
            active.delivery_state == ActiveDeliveryState::Ready
                && !self.forward.is_prefilling(&active.slot)
                && match &active.slot {
                    DecodeSlot::Real {
                        first_token_pending,
                        ..
                    } => !*first_token_pending,
                    DecodeSlot::RealPrefill { .. } => false,
                    DecodeSlot::Mock { .. } => true,
                }
        })
    }

    fn send_delivery(&self, command: DeliveryCommand) -> bool {
        self.delivery_worker
            .as_ref()
            .is_some_and(|worker| worker.command(command).is_ok())
    }

    fn queue_delivery(&mut self, key: DeliveryKey, batch: DeliveryBatch) {
        self.delivery_outbox.push((key, batch));
        if !self.defer_delivery_flush {
            self.flush_delivery_outbox();
        }
    }

    fn flush_delivery_outbox(&mut self) {
        if self.delivery_outbox.is_empty() {
            return;
        }
        let mut deliveries = std::mem::take(&mut self.delivery_outbox);
        let command = if deliveries.len() == 1 {
            let (key, batch) = deliveries.pop().expect("single delivery remains present");
            DeliveryCommand::Deliver { key, batch }
        } else {
            DeliveryCommand::DeliverMany { deliveries }
        };
        if !self.send_delivery(command) {
            self.accepting = false;
            self.stopped = true;
        }
    }

    fn register_delivery(
        &mut self,
        request_id: Uuid,
        response_tx: mpsc::Sender<EngineEvent>,
    ) -> Option<DeliveryKey> {
        let generation = self.next_delivery_generation;
        self.next_delivery_generation = self.next_delivery_generation.checked_add(1)?;
        let key = DeliveryKey::new(request_id, generation);
        self.send_delivery(DeliveryCommand::Register { key, response_tx })
            .then_some(key)
    }

    fn drain_delivery_results(&mut self) {
        while let Ok(results) = self.delivery_results.try_recv() {
            for result in results {
                self.handle_delivery_result(result);
            }
        }
    }

    fn handle_delivery_result(&mut self, result: DeliveryResult) {
        match result {
            DeliveryResult::BackpressureStarted {
                key,
                sequence,
                capacity,
            } => {
                if self.delivery_backpressured.insert((key, sequence)) {
                    self.snapshot.response_backpressure_events =
                        self.snapshot.response_backpressure_events.saturating_add(1);
                    tracing::info!(
                        event = "response_channel_backpressure",
                        request_id = %key.request_id,
                        generation = key.generation,
                        sequence,
                        channel_capacity = capacity,
                        grace_ms = self.response_delivery_policy.stream_stall_grace_ms(),
                        "response_channel_backpressure"
                    );
                }
            }
            DeliveryResult::Delivered {
                key,
                sequence,
                terminal,
                waited,
            } => {
                let was_backpressured = self.delivery_backpressured.remove(&(key, sequence));
                if terminal {
                    self.delivery_pending_terminal.remove(&key);
                }
                if was_backpressured {
                    self.record_response_backpressure_wait(waited);
                }
                if !terminal
                    && let Some(active) = self.active.iter_mut().find(|active| {
                        active.delivery_key == key
                            && active.delivery_state == ActiveDeliveryState::InFlight { sequence }
                    })
                {
                    active.delivery_state = ActiveDeliveryState::Ready;
                }
            }
            DeliveryResult::Closed {
                key,
                sequence,
                waited,
                backpressured,
            } => {
                self.delivery_backpressured.remove(&(key, sequence));
                let pending_terminal = self.delivery_pending_terminal.remove(&key);
                if backpressured {
                    self.record_response_backpressure_wait(waited);
                }
                let active_idx = self.active.iter().position(|active| {
                    active.delivery_key == key
                        && active.delivery_state == ActiveDeliveryState::InFlight { sequence }
                });
                if let Some(idx) = active_idx {
                    let active = self.active.remove(idx);
                    active.req.cancel.cancel();
                    self.forward.discard_request(active.slot);
                }
                if pending_terminal || active_idx.is_some() {
                    self.snapshot.response_channel_closed =
                        self.snapshot.response_channel_closed.saturating_add(1);
                }
                tracing::info!(
                    event = "response_channel_closed",
                    request_id = %key.request_id,
                    generation = key.generation,
                    sequence,
                    backpressured,
                    waited_ms = duration_millis_saturating(waited),
                    "stream response receiver closed"
                );
            }
            DeliveryResult::TimedOut {
                key,
                sequence,
                waited,
            } => {
                self.delivery_backpressured.remove(&(key, sequence));
                let pending_terminal = self.delivery_pending_terminal.remove(&key);
                self.record_response_backpressure_wait(waited);
                let active_idx = self.active.iter().position(|active| {
                    active.delivery_key == key
                        && active.delivery_state == ActiveDeliveryState::InFlight { sequence }
                });
                if let Some(idx) = active_idx {
                    let active = self.active.remove(idx);
                    active.req.cancel.cancel();
                    self.forward.discard_request(active.slot);
                }
                if pending_terminal || active_idx.is_some() {
                    self.snapshot.response_stall_evictions =
                        self.snapshot.response_stall_evictions.saturating_add(1);
                    self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                }
                tracing::warn!(
                    event = "response_channel_backpressure_timeout",
                    request_id = %key.request_id,
                    generation = key.generation,
                    sequence,
                    grace_ms = self.response_delivery_policy.stream_stall_grace_ms(),
                    waited_ms = duration_millis_saturating(waited),
                    "response_channel_backpressure_timeout"
                );
            }
            DeliveryResult::ProtocolError(error) => {
                self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                self.accepting = false;
                self.stopped = true;
                tracing::error!(
                    event = "response_delivery_protocol_error",
                    error = %error,
                    "response delivery protocol violation; stopping batching actor"
                );
            }
        }
        self.refresh_snapshot();
    }

    fn record_response_backpressure_wait(&mut self, waited: Duration) {
        self.snapshot.response_backpressure_wait_ms = self
            .snapshot
            .response_backpressure_wait_ms
            .saturating_add(duration_millis_saturating(waited));
    }

    fn handle_command(&mut self, cmd: EngineCommand) {
        match cmd {
            EngineCommand::Enqueue { req, response_tx } => {
                let Some(delivery_key) = self.register_delivery(req.request_id, response_tx) else {
                    req.cancel.cancel();
                    self.accepting = false;
                    self.stopped = true;
                    tracing::error!(
                        request_id = %req.request_id,
                        "response delivery worker unavailable; stopping batching actor"
                    );
                    return;
                };
                if self.accepting {
                    self.waiting.push_back(QueuedRequest { req, delivery_key });
                    self.refresh_snapshot();
                } else {
                    req.cancel.cancel();
                    self.terminate_delivery(
                        delivery_key,
                        "batching engine is draining".to_string(),
                    );
                }
            }
            EngineCommand::Cancel { request_id } => self.cancel(request_id),
            EngineCommand::DeliveryWake => self.drain_delivery_results(),
            EngineCommand::Drain { reply } => {
                self.accepting = false;
                self.refresh_snapshot();
                self.refresh_deferral_gauge();
                self.publish_snapshot();
                let _ = reply.send(());
            }
            EngineCommand::Stop { reply } => {
                self.accepting = false;
                self.stopped = true;
                self.stop_replies.push(reply);
                self.refresh_snapshot();
                self.refresh_deferral_gauge();
                self.publish_snapshot();
            }
            EngineCommand::Snapshot { reply } => {
                let delivery_barrier = self
                    .delivery_worker
                    .as_ref()
                    .ok_or(DeliveryBarrierError)
                    .and_then(DeliveryWorker::barrier);
                if let Err(error) = delivery_barrier {
                    let _ = reply.send(Err(format!(
                        "response delivery worker unavailable at snapshot barrier: {error}"
                    )));
                    return;
                }
                self.drain_delivery_results();
                self.refresh_snapshot();
                self.refresh_deferral_gauge();
                self.publish_snapshot();
                let _ = reply.send(Ok(self.snapshot.clone()));
            }
            EngineCommand::ResizeKv {
                target_blocks,
                reply,
            } => {
                // Runs at the barrier (drain_commands, between decode steps): the
                // previous step's `run_decode_batch` has returned, so no forward
                // is in flight in THIS actor. `resize_kv` additionally takes the
                // GPU write lock to exclude the other decode actor / training.
                let result = self
                    .forward
                    .resize_kv(target_blocks)
                    .map_err(|e| format!("{e:#}"));
                let _ = reply.send(result);
            }
            EngineCommand::SwapAdapter { swap, reply } => {
                // Unlike ResizeKv, a weight swap must also wait for the
                // active batch to DRAIN (KV computed under the old weights
                // can't continue under new ones), so it queues here and the
                // run loop executes it at the between-requests barrier.
                self.pending_swaps
                    .push_back(PendingAdapterSwap { swap, reply });
            }
        }
    }

    /// Execute queued adapter swaps when the active batch has drained. The
    /// run loop calls this between decode steps; while swaps are pending,
    /// `admit_waiting` pauses admission so the barrier is reached promptly.
    fn run_pending_swaps_at_barrier(&mut self) {
        if self.pending_swaps.is_empty() || !self.active.is_empty() {
            return;
        }
        while let Some(pending) = self.pending_swaps.pop_front() {
            let result = (pending.swap)();
            let _ = pending.reply.send(result);
        }
    }

    fn cancel(&mut self, request_id: Uuid) {
        let mut waiting_idx = 0;
        while waiting_idx < self.waiting.len() {
            if self.waiting[waiting_idx].req.request_id == request_id {
                let queued = self
                    .waiting
                    .remove(waiting_idx)
                    .expect("waiting request index remains valid");
                queued.req.cancel.cancel();
                self.terminate_delivery(queued.delivery_key, "request cancelled".to_string());
            } else {
                waiting_idx += 1;
            }
        }

        let mut idx = 0;
        while idx < self.active.len() {
            if self.active[idx].req.request_id == request_id {
                let active = self.active.remove(idx);
                active.req.cancel.cancel();
                self.forward.discard_request(active.slot);
                self.terminate_delivery(active.delivery_key, "request cancelled".to_string());
            } else {
                idx += 1;
            }
        }
        self.refresh_snapshot();
    }

    fn terminate_delivery(&mut self, key: DeliveryKey, error: String) {
        self.delivery_pending_terminal.insert(key);
        if !self.send_delivery(DeliveryCommand::Terminate { key, error }) {
            self.accepting = false;
            self.stopped = true;
        }
    }

    /// Admit queued requests and report whether this cycle submitted one or
    /// more prefill-produced first tokens to the delivery worker.
    fn admit_waiting_with_budget(&mut self, mut token_budget: usize) -> AdmissionOutcome {
        // A pending adapter swap needs the active batch to drain — admitting
        // new requests now would (a) delay the swap arbitrarily and (b) run
        // them under weights the caller is replacing. They stay queued and
        // resume right after the swap executes.
        if !self.pending_swaps.is_empty() {
            return AdmissionOutcome::default();
        }
        let initial_token_budget = token_budget;
        let mut admitted = 0usize;
        let mut submitted_first_tokens = false;
        // #1082 CUDA concurrency regression fix: CUDA (burst_refill) refills the
        // batch toward max_decode_batch every cycle — the LKG 2d9d4fc4 burst-fill
        // that sustained a wide decode batch and scaled to ~498 tok/s @ bs=64.
        // Vulkan/Metal keep yielding to ready decode rows (admit 1 prefill/cycle
        // once any row is decoding) — their tuned latency behavior. The while
        // loop below still caps total active at max_decode_batch, so burst_refill
        // only ever fills the deficit, never over-admits.
        let admission_limit = if self.has_ready_decode_row() && !self.burst_refill {
            1
        } else {
            self.max_prefill_admissions_per_cycle
        };
        while self.active.len() < self.max_decode_batch
            && admitted < admission_limit
            && !self.waiting.is_empty()
            && token_budget > 0
        {
            // Count deferrals during the admission scan itself — when every
            // waiting row is deferred, position() has already evaluated the
            // predicate for all of them, so a second full filter pass would
            // double the (prefix-cache-locking) work for the same answer.
            let mut deferred_seen: u64 = 0;
            let Some(waiting_idx) = self.waiting.iter().position(|queued| {
                let defer = self.should_defer_for_active_prefix(queued);
                deferred_seen += u64::from(defer);
                !defer
            }) else {
                self.snapshot.prefix_admission_deferrals = self
                    .snapshot
                    .prefix_admission_deferrals
                    .saturating_add(deferred_seen);
                break;
            };
            let queued = self
                .waiting
                .remove(waiting_idx)
                .expect("waiting index selected from VecDeque");
            if !self.forward.supports_resumable_prefill()
                && queued.req.prompt_tokens.len() > token_budget
            {
                self.waiting.insert(waiting_idx, queued);
                break;
            }
            let started = Instant::now();
            let preparation = self
                .forward
                .prepare_request_chunked(&queued.req, token_budget);
            self.record_admission_duration(
                started.elapsed(),
                queued.req.request_id,
                queued.req.prompt_tokens.len(),
                token_budget,
            );
            match preparation {
                Ok(preparation) => {
                    let (slot, tokens_processed, ready) = match preparation {
                        RequestPreparation::Prefilling {
                            slot,
                            tokens_processed,
                            ..
                        } => (slot, tokens_processed, false),
                        RequestPreparation::Ready {
                            slot,
                            tokens_processed,
                            ..
                        } => (slot, tokens_processed, true),
                    };
                    if tokens_processed > token_budget {
                        self.forward.discard_request(slot);
                        self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                        self.terminate_delivery(
                            queued.delivery_key,
                            format!(
                                "prefill admission processed {tokens_processed} tokens beyond the {token_budget}-token remainder"
                            ),
                        );
                        continue;
                    }
                    self.snapshot.last_prefill_tokens = tokens_processed;
                    self.snapshot.total_prefill_tokens = self
                        .snapshot
                        .total_prefill_tokens
                        .saturating_add(tokens_processed as u64);
                    token_budget -= tokens_processed;
                    admitted += 1;
                    let active_idx = self.active.len();
                    self.active.push(ActiveRequest {
                        req: queued.req,
                        delivery_key: queued.delivery_key,
                        delivery_state: ActiveDeliveryState::Ready,
                        next_delivery_sequence: 0,
                        slot,
                    });
                    // Publish admission before first-token delivery, which can
                    // itself encounter a slow response channel.
                    self.refresh_snapshot();
                    if ready {
                        submitted_first_tokens |= self.emit_pending_first_token_at(active_idx);
                    }
                }
                Err(err) => {
                    // A block-pool shortage while other requests are active
                    // is TRANSIENT — they free blocks as they finish. Put
                    // the request back at the front and retry next cycle
                    // (the caller's request timeout bounds the wait)
                    // instead of failing it instantly under concurrent
                    // load. With nothing active, nothing will ever free —
                    // fail honestly.
                    let msg = format!("{err:#}");
                    if msg.contains("out of memory: no free blocks") && !self.active.is_empty() {
                        tracing::debug!(
                            request_id = %queued.req.request_id,
                            "block pool busy — request stays queued for the next cycle"
                        );
                        self.waiting.push_front(queued);
                        break;
                    }
                    self.snapshot.total_errors += 1;
                    self.terminate_delivery(queued.delivery_key, msg);
                }
            }
        }
        if admitted > 0 {
            self.snapshot.total_prefill_admission_cycles = self
                .snapshot
                .total_prefill_admission_cycles
                .saturating_add(1);
        }
        self.refresh_snapshot();
        AdmissionOutcome {
            submitted_first_tokens,
            tokens_processed: initial_token_budget.saturating_sub(token_budget),
        }
    }

    #[cfg(test)]
    fn admit_waiting(&mut self) -> bool {
        self.admit_waiting_with_budget(self.max_batch_tokens.min(self.max_prefill_tokens_per_cycle))
            .submitted_first_tokens
    }

    /// Spend at most one combined-cycle token remainder on resumable prefills.
    /// Partial rows are selected round-robin so a 16K prompt cannot hide a 1K
    /// prompt behind repeated quanta. Every forward returns to the actor before
    /// another decode cohort or control-command drain.
    fn run_prefill_budget(&mut self, mut budget: usize) -> bool {
        let mut advanced = false;
        while budget > 0 && !self.active.is_empty() {
            let active_len = self.active.len();
            let Some(idx) = (0..active_len)
                .map(|offset| (self.next_prefill_index + offset) % active_len)
                .find(|&idx| {
                    self.active[idx].delivery_state == ActiveDeliveryState::Ready
                        && self.forward.is_prefilling(&self.active[idx].slot)
                        && self
                            .forward
                            .inflight_prefill_token_width(&self.active[idx].slot)
                            .is_none_or(|tokens| tokens <= budget)
                })
            else {
                break;
            };

            let ActiveRequest {
                req,
                delivery_key,
                delivery_state,
                next_delivery_sequence,
                slot,
            } = self.active.remove(idx);
            let started = Instant::now();
            let result = self.forward.advance_prefill(
                slot,
                budget,
                self.max_prefill_layers_per_cycle,
                &req.sampling,
                &req.cancel,
            );
            let elapsed = started.elapsed();
            self.record_prefill_forward_duration(
                elapsed,
                req.request_id,
                budget,
                self.max_prefill_layers_per_cycle,
            );
            self.snapshot.total_prefill_forwards =
                self.snapshot.total_prefill_forwards.saturating_add(1);

            let preparation = match result {
                Ok(preparation) => preparation,
                Err(error) => {
                    self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                    self.terminate_delivery(delivery_key, format!("{error:#}"));
                    self.next_prefill_index = if self.active.is_empty() {
                        0
                    } else {
                        idx % self.active.len()
                    };
                    self.refresh_snapshot();
                    advanced = true;
                    continue;
                }
            };
            let (slot, tokens_processed, layers_processed, ready) = match preparation {
                RequestPreparation::Prefilling {
                    slot,
                    tokens_processed,
                    layers_processed,
                } => (slot, tokens_processed, layers_processed, false),
                RequestPreparation::Ready {
                    slot,
                    tokens_processed,
                    layers_processed,
                } => (slot, tokens_processed, layers_processed, true),
            };
            if layers_processed == 0 || layers_processed > self.max_prefill_layers_per_cycle {
                self.forward.discard_request(slot);
                self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                self.terminate_delivery(
                    delivery_key,
                    format!(
                        "prefill forward reported {layers_processed} layers for a {}-layer budget",
                        self.max_prefill_layers_per_cycle
                    ),
                );
                self.next_prefill_index = if self.active.is_empty() {
                    0
                } else {
                    idx % self.active.len()
                };
                self.refresh_snapshot();
                advanced = true;
                continue;
            }
            self.snapshot.last_prefill_layers = layers_processed;
            self.snapshot.total_prefill_layers = self
                .snapshot
                .total_prefill_layers
                .saturating_add(layers_processed as u64);
            if tokens_processed == 0 {
                if ready || !self.forward.has_inflight_prefill_layer_progress(&slot) {
                    self.forward.discard_request(slot);
                    self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                    self.terminate_delivery(
                        delivery_key,
                        "prefill forward reported zero tokens without retained layer progress"
                            .to_string(),
                    );
                    self.next_prefill_index = if self.active.is_empty() {
                        0
                    } else {
                        idx % self.active.len()
                    };
                    self.refresh_snapshot();
                    return true;
                }
                self.active.insert(
                    idx,
                    ActiveRequest {
                        req,
                        delivery_key,
                        delivery_state,
                        next_delivery_sequence,
                        slot,
                    },
                );
                self.next_prefill_index = (idx + 1) % self.active.len();
                self.snapshot.total_prefill_layer_yields =
                    self.snapshot.total_prefill_layer_yields.saturating_add(1);
                self.refresh_snapshot();
                return true;
            }
            if tokens_processed > budget {
                self.forward.discard_request(slot);
                self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                self.terminate_delivery(
                    delivery_key,
                    format!(
                        "prefill forward reported {tokens_processed} tokens for a {budget}-token budget"
                    ),
                );
                self.next_prefill_index = if self.active.is_empty() {
                    0
                } else {
                    idx % self.active.len()
                };
                self.refresh_snapshot();
                return true;
            }

            budget -= tokens_processed;
            self.snapshot.last_prefill_tokens = tokens_processed;
            self.snapshot.total_prefill_tokens = self
                .snapshot
                .total_prefill_tokens
                .saturating_add(tokens_processed as u64);
            self.active.insert(
                idx,
                ActiveRequest {
                    req,
                    delivery_key,
                    delivery_state,
                    next_delivery_sequence,
                    slot,
                },
            );
            self.next_prefill_index = (idx + 1) % self.active.len();
            if ready {
                self.emit_pending_first_token_at(idx);
            }
            self.refresh_snapshot();
            advanced = true;
        }
        let token_budget_deferred = self.active.iter().any(|active| {
            active.delivery_state == ActiveDeliveryState::Ready
                && self.forward.is_prefilling(&active.slot)
                && self
                    .forward
                    .inflight_prefill_token_width(&active.slot)
                    .is_some_and(|tokens| tokens > budget)
        });
        if token_budget_deferred {
            self.snapshot.total_prefill_token_budget_deferrals = self
                .snapshot
                .total_prefill_token_budget_deferrals
                .saturating_add(1);
            self.refresh_snapshot();
        }
        advanced || token_budget_deferred
    }

    fn pending_first_token_at(&mut self, idx: usize) -> Option<TokenId> {
        match self.active.get_mut(idx).map(|active| &mut active.slot) {
            Some(DecodeSlot::Real {
                state,
                first_token_pending,
                ..
            }) if *first_token_pending => {
                *first_token_pending = false;
                Some(state.next_token)
            }
            _ => None,
        }
    }

    fn emit_pending_first_token_at(&mut self, idx: usize) -> bool {
        let Some(token) = self.pending_first_token_at(idx) else {
            return false;
        };

        self.emit_output_token_at(idx, token);
        true
    }

    fn emit_output_token_at(&mut self, idx: usize, token: TokenId) {
        let ready_at = Instant::now();
        let token = {
            let generated_tokens = self.generated_tokens_for(idx);
            self.active[idx]
                .req
                .sampling
                .apply_thinking_budget(generated_tokens, token)
        };
        match self.forward.is_eos_token(token) {
            Ok(true) => {
                self.finish_active(idx, FinishReason::Eos, None);
                return;
            }
            Ok(false) => {}
            Err(err) => {
                self.finish_one_with_error(idx, format!("{err:#}"), None);
                return;
            }
        }

        let generated_count = match self.forward.accept_token(&mut self.active[idx].slot, token) {
            Ok(count) => count,
            Err(err) => {
                self.finish_one_with_error(idx, format!("{err:#}"), None);
                return;
            }
        };
        self.snapshot.total_decode_tokens += 1;

        // No per-token Vec clone: `forward` is an Arc, so a cheap handle
        // clone lets the generated-tokens slice borrow `self.active`
        // directly. The old `.to_vec()` here re-copied the entire
        // generated sequence on EVERY token — O(n²) churn on long
        // completions, in the decode hot path.
        let stop = {
            let forward = self.forward.clone();
            let generated_tokens = self.generated_tokens_for(idx);
            let sampling = &self.active[idx].req.sampling;
            forward.stop_reason_after_emit(generated_tokens, sampling)
        };
        match stop {
            Ok(Some(reason)) => {
                self.finish_active(idx, reason, Some((token, ready_at)));
            }
            Ok(None) if generated_count >= self.active[idx].req.sampling.max_tokens => {
                self.finish_active(idx, FinishReason::MaxTokens, Some((token, ready_at)));
            }
            Ok(None) => self.submit_token_delivery(idx, token, ready_at),
            Err(err) => {
                self.finish_one_with_error(idx, format!("{err:#}"), Some((token, ready_at)));
            }
        }
    }

    fn submit_token_delivery(&mut self, idx: usize, token: TokenId, ready_at: Instant) {
        let (key, sequence) = {
            let active = &mut self.active[idx];
            debug_assert_eq!(active.delivery_state, ActiveDeliveryState::Ready);
            let sequence = active.next_delivery_sequence;
            let Some(next_sequence) = sequence.checked_add(1) else {
                self.finish_one_with_error(
                    idx,
                    "response delivery sequence exhausted".to_string(),
                    Some((token, ready_at)),
                );
                return;
            };
            active.next_delivery_sequence = next_sequence;
            active.delivery_state = ActiveDeliveryState::InFlight { sequence };
            (active.delivery_key, sequence)
        };
        self.queue_delivery(
            key,
            DeliveryBatch::Token {
                token,
                ready_at,
                sequence,
            },
        );
        self.refresh_snapshot();
    }

    fn should_defer_for_active_prefix(&self, queued: &QueuedRequest) -> bool {
        self.prefix_aware_admission
            && self.active.iter().any(|active| {
                active.req.adapter == queued.req.adapter
                    && active.req.prompt_tokens.len() < queued.req.prompt_tokens.len()
                    && queued
                        .req
                        .prompt_tokens
                        .starts_with(&active.req.prompt_tokens)
                    && self
                        .forward
                        .can_reuse_as_strict_prefix(active.req.prompt_tokens.len())
            })
    }

    fn run_decode_batch_with_budget(&mut self, max_rows: usize) -> usize {
        let mut ready_indices = self.ready_decode_indices_with_limit(max_rows);
        if ready_indices.is_empty() {
            return 0;
        }

        // Pre-grow KV per slot: a request that has outgrown the block pool
        // finishes as a `length` casualty HERE — the old order let the
        // forward's atomic grow fail and `finish_batch_with_error` killed
        // EVERY active request because one conversation got long.
        {
            let mut probe_slots: Vec<&mut DecodeSlot> = self
                .active
                .iter_mut()
                .enumerate()
                .filter_map(|(idx, active)| {
                    ready_indices.contains(&idx).then_some(&mut active.slot)
                })
                .collect();
            match self.forward.grow_for_decode(&mut probe_slots) {
                Ok(starved) => {
                    drop(probe_slots);
                    let mut starved_indices: Vec<usize> = starved
                        .into_iter()
                        .filter_map(|relative_idx| ready_indices.get(relative_idx).copied())
                        .collect();
                    starved_indices.sort_unstable();
                    starved_indices.dedup();
                    for idx in starved_indices.into_iter().rev() {
                        if idx >= self.active.len()
                            || self.active[idx].delivery_state != ActiveDeliveryState::Ready
                        {
                            continue;
                        }
                        tracing::warn!(
                            request_id = %self.active[idx].req.request_id,
                            "KV block pool exhausted for this request — finishing it as \
                             `length`; other requests keep decoding"
                        );
                        self.finish_active(idx, FinishReason::MaxTokens, None);
                    }
                    ready_indices = self.ready_decode_indices_with_limit(max_rows);
                    if ready_indices.is_empty() {
                        self.refresh_snapshot();
                        return 0;
                    }
                }
                Err(err) => {
                    self.finish_indices_with_error(&ready_indices, format!("{err:#}"));
                    self.refresh_snapshot();
                    return 0;
                }
            }
        }

        let batch_len = ready_indices.len();
        let sampling: Vec<SamplingParams> = ready_indices
            .iter()
            .map(|&idx| self.active[idx].req.sampling.clone())
            .collect();

        self.snapshot.current_batch_size = batch_len;
        self.snapshot.max_observed_batch_size =
            self.snapshot.max_observed_batch_size.max(batch_len);
        self.snapshot.total_decode_forwards = self.snapshot.total_decode_forwards.saturating_add(1);
        self.snapshot.total_decode_rows = self
            .snapshot
            .total_decode_rows
            .saturating_add(batch_len as u64);
        if batch_len > 1 {
            self.snapshot.total_batched_decode_forwards = self
                .snapshot
                .total_batched_decode_forwards
                .saturating_add(1);
        }
        // Publish the in-flight batch before entering a potentially long GPU
        // forward. Control-plane readers can now remain responsive while also
        // seeing both the batch and an increasing snapshot age.
        self.refresh_snapshot();
        let mut slots: Vec<&mut DecodeSlot> = self
            .active
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, active)| ready_indices.contains(&idx).then_some(&mut active.slot))
            .collect();
        let started = Instant::now();
        let result = self.forward.forward_decode(&mut slots, &sampling);
        let elapsed = started.elapsed();
        drop(slots);
        self.record_decode_forward_duration(elapsed, batch_len);
        self.snapshot.last_batch_size = batch_len;
        self.snapshot.current_batch_size = 0;
        self.refresh_snapshot();

        let output_tokens = match result {
            Ok(tokens) if tokens.len() == batch_len => tokens,
            Ok(tokens) => {
                self.finish_indices_with_error(
                    &ready_indices,
                    format!(
                        "batched decode returned {} rows for batch size {batch_len}",
                        tokens.len()
                    ),
                );
                self.refresh_snapshot();
                return 0;
            }
            Err(err) => {
                self.finish_indices_with_error(&ready_indices, format!("{err:#}"));
                self.refresh_snapshot();
                return 0;
            }
        };

        debug_assert!(!self.defer_delivery_flush);
        self.defer_delivery_flush = true;
        for (idx, token) in ready_indices.into_iter().zip(output_tokens).rev() {
            if idx >= self.active.len()
                || self.active[idx].delivery_state != ActiveDeliveryState::Ready
            {
                continue;
            }
            self.emit_output_token_at(idx, token);
            if self.stopped {
                break;
            }
        }
        self.defer_delivery_flush = false;
        self.flush_delivery_outbox();
        self.refresh_snapshot();
        batch_len
    }

    #[cfg(test)]
    fn run_decode_batch(&mut self) -> usize {
        self.run_decode_batch_with_budget(self.max_batch_tokens)
    }

    fn ready_decode_indices_with_limit(&self, max_rows: usize) -> Vec<usize> {
        self.active
            .iter()
            .enumerate()
            .filter_map(|(idx, active)| {
                if active.delivery_state != ActiveDeliveryState::Ready {
                    return None;
                }
                if self.forward.is_prefilling(&active.slot) {
                    return None;
                }
                match &active.slot {
                    DecodeSlot::Real {
                        first_token_pending,
                        ..
                    } if *first_token_pending => None,
                    DecodeSlot::RealPrefill { .. } => None,
                    DecodeSlot::Real { .. } | DecodeSlot::Mock { .. } => Some(idx),
                }
            })
            .take(self.max_decode_batch.min(max_rows))
            .collect()
    }

    fn generated_tokens_for(&self, idx: usize) -> &[TokenId] {
        match &self.active[idx].slot {
            DecodeSlot::Mock {
                generated_tokens, ..
            } => generated_tokens,
            DecodeSlot::Real { state, .. } => &state.generated_tokens,
            DecodeSlot::RealPrefill { .. } => {
                unreachable!("prefilling row has no generated tokens")
            }
        }
    }

    fn finish_active(
        &mut self,
        idx: usize,
        finish_reason: FinishReason,
        preceding_token: Option<(TokenId, Instant)>,
    ) {
        let active = self.active.remove(idx);
        let key = active.delivery_key;
        let sequence = active.next_delivery_sequence;
        self.refresh_snapshot();
        let terminal = match self.forward.finish_request(active.slot, finish_reason) {
            Ok(output) => {
                let completion_tokens = completion_usage_tokens(
                    output.output.token_ids.len(),
                    &output.output.finish_reason,
                );
                DeliveryTerminal::Done(BatchedGenerationOutput {
                    text: output.output.text,
                    token_ids: output.output.token_ids,
                    finish_reason: output.output.finish_reason,
                    completion_tokens,
                    prefill_duration: output.prefill_duration,
                    decode_duration: output.decode_duration,
                })
            }
            Err(err) => {
                self.snapshot.total_errors += 1;
                self.publish_snapshot();
                DeliveryTerminal::Error(err.to_string())
            }
        };
        self.submit_terminal_delivery(key, sequence, preceding_token, terminal);
    }

    fn finish_one_with_error(
        &mut self,
        idx: usize,
        error: String,
        preceding_token: Option<(TokenId, Instant)>,
    ) {
        self.snapshot.total_errors += 1;
        let active = self.active.remove(idx);
        let key = active.delivery_key;
        let sequence = active.next_delivery_sequence;
        self.refresh_snapshot();
        self.forward.discard_request(active.slot);
        self.submit_terminal_delivery(
            key,
            sequence,
            preceding_token,
            DeliveryTerminal::Error(error),
        );
    }

    fn finish_indices_with_error(&mut self, indices: &[usize], error: String) {
        for &idx in indices.iter().rev() {
            if idx < self.active.len() {
                self.finish_one_with_error(idx, error.clone(), None);
            }
        }
    }

    fn submit_terminal_delivery(
        &mut self,
        key: DeliveryKey,
        sequence: u64,
        preceding_token: Option<(TokenId, Instant)>,
        terminal: DeliveryTerminal,
    ) {
        self.delivery_pending_terminal.insert(key);
        self.queue_delivery(
            key,
            DeliveryBatch::Terminal {
                preceding_token,
                terminal,
                sequence,
            },
        );
        self.refresh_snapshot();
    }

    fn fail_all(&mut self, error: &str) {
        self.accepting = false;
        self.refresh_snapshot();
        while let Some(queued) = self.waiting.pop_front() {
            queued.req.cancel.cancel();
            self.refresh_snapshot();
        }
        for active in self.active.drain(..) {
            active.req.cancel.cancel();
            self.forward.discard_request(active.slot);
        }
        self.refresh_snapshot();
        if let Some(mut worker) = self.delivery_worker.take() {
            let _ = worker.shutdown(error.to_string());
        }
        self.delivery_outbox.clear();
        self.defer_delivery_flush = false;
        self.delivery_backpressured.clear();
        self.delivery_pending_terminal.clear();
        while let Some(pending) = self.pending_swaps.pop_front() {
            let _ = pending.reply.send(Err(error.to_string()));
        }
        self.refresh_snapshot();
    }

    fn refresh_snapshot(&mut self) {
        self.snapshot.snapshot_age_ms = 0;
        self.snapshot.accepting = self.accepting;
        self.snapshot.queue_depth = self.waiting.len();
        self.snapshot.active_prefill = self
            .active
            .iter()
            .filter(|active| self.forward.is_prefilling(&active.slot))
            .count();
        self.snapshot.active_decode = self
            .active
            .len()
            .saturating_sub(self.snapshot.active_prefill);
        self.snapshot.response_delivery_in_flight = self
            .active
            .iter()
            .filter(|active| matches!(active.delivery_state, ActiveDeliveryState::InFlight { .. }))
            .count();
        self.snapshot.response_delivery_backpressured = self.delivery_backpressured.len();
        self.snapshot.response_delivery_pending_terminal = self.delivery_pending_terminal.len();
        self.snapshot.adapter_groups_waiting = usize::from(!self.waiting.is_empty());
        self.publish_snapshot();
    }

    fn publish_snapshot(&mut self) {
        self.snapshot.snapshot_age_ms = 0;
        let snapshot = self.snapshot.clone();
        let mut published = self
            .published_snapshot
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        published.snapshot = snapshot;
        published.published_at = Instant::now();
    }

    /// Recompute the `prefix_deferred_waiting` gauge. O(waiting x active x
    /// prompt_len) with a prefix-cache lock per matching pair, so it runs
    /// only when an observer asks for the snapshot (Snapshot/Drain/Stop) —
    /// not on the per-decode-step hot path, where it burned CPU for a value
    /// nobody read between observations.
    fn refresh_deferral_gauge(&mut self) {
        self.snapshot.prefix_deferred_waiting = self
            .waiting
            .iter()
            .filter(|queued| self.should_defer_for_active_prefix(queued))
            .count();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_core::block::BlockTable;
    use kiln_core::sampling::ThinkingBudget;
    use kiln_model::LinearAttentionState;
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct MockForward {
        calls: StdMutex<Vec<Vec<TokenId>>>,
        reusable_prefixes: bool,
        prefix_probe_calls: std::sync::atomic::AtomicUsize,
    }

    #[derive(Default)]
    struct PendingFirstTokenForward {
        calls: StdMutex<Vec<Vec<TokenId>>>,
        prepare_delay: Duration,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum SchedulingEvent {
        Decode(Vec<TokenId>),
        Prefill {
            key: TokenId,
            tokens: usize,
            remaining: usize,
        },
        PrefillLayers {
            key: TokenId,
            layers: usize,
            remaining: usize,
        },
        Discard(TokenId),
    }

    #[derive(Default)]
    struct SyntheticPrefillForward {
        remaining: StdMutex<HashMap<TokenId, usize>>,
        pending_layers: StdMutex<HashMap<TokenId, usize>>,
        pending_token_widths: StdMutex<HashMap<TokenId, usize>>,
        events: StdMutex<Vec<SchedulingEvent>>,
        layers_per_chunk: usize,
        layer_delay: Duration,
    }

    impl SyntheticPrefillForward {
        fn mock_slot(key: TokenId) -> DecodeSlot {
            DecodeSlot::Mock {
                next_token: key,
                generated_tokens: Vec::new(),
            }
        }

        fn slot_key(slot: &DecodeSlot) -> TokenId {
            match slot {
                DecodeSlot::Mock { next_token, .. } => *next_token,
                DecodeSlot::Real { .. } | DecodeSlot::RealPrefill { .. } => unreachable!(),
            }
        }
    }

    impl DecodeForward for SyntheticPrefillForward {
        fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
            Ok(Self::mock_slot(
                req.prompt_tokens.last().copied().unwrap_or_default(),
            ))
        }

        fn supports_resumable_prefill(&self) -> bool {
            true
        }

        fn prepare_request_chunked(
            &self,
            req: &EngineRequest,
            _max_tokens: usize,
        ) -> Result<RequestPreparation> {
            let key = req.prompt_tokens.last().copied().unwrap_or_default();
            let slot = Self::mock_slot(key);
            if req.prompt_tokens.len() <= 1 {
                return Ok(RequestPreparation::Ready {
                    slot,
                    tokens_processed: 0,
                    layers_processed: 0,
                });
            }
            self.remaining
                .lock()
                .unwrap()
                .insert(key, req.prompt_tokens.len());
            Ok(RequestPreparation::Prefilling {
                slot,
                tokens_processed: 0,
                layers_processed: 0,
            })
        }

        fn is_prefilling(&self, slot: &DecodeSlot) -> bool {
            let DecodeSlot::Mock { next_token, .. } = slot else {
                return matches!(slot, DecodeSlot::RealPrefill { .. });
            };
            self.remaining.lock().unwrap().contains_key(next_token)
        }

        fn advance_prefill(
            &self,
            slot: DecodeSlot,
            max_tokens: usize,
            max_layers: usize,
            _sampling: &SamplingParams,
            cancel: &CancelHandle,
        ) -> Result<RequestPreparation> {
            anyhow::ensure!(!cancel.is_cancelled(), "synthetic prefill cancelled");
            let key = Self::slot_key(&slot);
            let reserved_tokens = if self.layers_per_chunk == 0 {
                None
            } else {
                let remaining_tokens =
                    *self.remaining.lock().unwrap().get(&key).ok_or_else(|| {
                        anyhow::anyhow!("missing synthetic prefill state for {key}")
                    })?;
                let mut pending_token_widths = self.pending_token_widths.lock().unwrap();
                Some(
                    *pending_token_widths
                        .entry(key)
                        .or_insert(remaining_tokens.min(max_tokens)),
                )
            };
            let layers_processed = if self.layers_per_chunk == 0 {
                1
            } else {
                anyhow::ensure!(
                    max_layers > 0,
                    "synthetic prefill received an empty layer budget"
                );
                let (layers, remaining_after) = {
                    let mut pending_layers = self.pending_layers.lock().unwrap();
                    let remaining_layers =
                        pending_layers.entry(key).or_insert(self.layers_per_chunk);
                    let layers = (*remaining_layers).min(max_layers);
                    *remaining_layers -= layers;
                    let remaining_after = *remaining_layers;
                    if remaining_after == 0 {
                        pending_layers.remove(&key);
                    }
                    (layers, remaining_after)
                };
                if !self.layer_delay.is_zero() {
                    thread::sleep(self.layer_delay);
                }
                self.events
                    .lock()
                    .unwrap()
                    .push(SchedulingEvent::PrefillLayers {
                        key,
                        layers,
                        remaining: remaining_after,
                    });
                if remaining_after > 0 {
                    return Ok(RequestPreparation::Prefilling {
                        slot,
                        tokens_processed: 0,
                        layers_processed: layers,
                    });
                }
                layers
            };
            let (tokens, remaining_after) = {
                let mut remaining = self.remaining.lock().unwrap();
                let remaining_tokens = remaining
                    .get_mut(&key)
                    .ok_or_else(|| anyhow::anyhow!("missing synthetic prefill state for {key}"))?;
                let tokens = reserved_tokens.unwrap_or_else(|| (*remaining_tokens).min(max_tokens));
                anyhow::ensure!(tokens > 0, "synthetic prefill received an empty budget");
                anyhow::ensure!(
                    tokens <= *remaining_tokens,
                    "synthetic reserved width {tokens} exceeds {remaining_tokens} remaining tokens"
                );
                *remaining_tokens -= tokens;
                let remaining_after = *remaining_tokens;
                if remaining_after == 0 {
                    remaining.remove(&key);
                }
                (tokens, remaining_after)
            };
            if reserved_tokens.is_some() {
                self.pending_token_widths.lock().unwrap().remove(&key);
            }
            thread::sleep(Duration::from_micros(100));
            self.events.lock().unwrap().push(SchedulingEvent::Prefill {
                key,
                tokens,
                remaining: remaining_after,
            });
            if remaining_after == 0 {
                Ok(RequestPreparation::Ready {
                    slot,
                    tokens_processed: tokens,
                    layers_processed,
                })
            } else {
                Ok(RequestPreparation::Prefilling {
                    slot,
                    tokens_processed: tokens,
                    layers_processed,
                })
            }
        }

        fn has_inflight_prefill_layer_progress(&self, slot: &DecodeSlot) -> bool {
            self.pending_layers
                .lock()
                .unwrap()
                .contains_key(&Self::slot_key(slot))
        }

        fn inflight_prefill_token_width(&self, slot: &DecodeSlot) -> Option<usize> {
            self.pending_token_widths
                .lock()
                .unwrap()
                .get(&Self::slot_key(slot))
                .copied()
        }

        fn forward_decode(
            &self,
            slots: &mut [&mut DecodeSlot],
            _sampling: &[SamplingParams],
        ) -> Result<Vec<TokenId>> {
            let keys: Vec<_> = slots.iter().map(|slot| Self::slot_key(slot)).collect();
            self.events
                .lock()
                .unwrap()
                .push(SchedulingEvent::Decode(keys.clone()));
            Ok(keys.into_iter().map(|key| key + 1).collect())
        }

        fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize> {
            let DecodeSlot::Mock {
                next_token,
                generated_tokens,
            } = slot
            else {
                anyhow::bail!("non-mock slot sent to synthetic accept_token")
            };
            generated_tokens.push(token);
            *next_token = token;
            Ok(generated_tokens.len())
        }

        fn finish_request(
            &self,
            slot: DecodeSlot,
            finish_reason: FinishReason,
        ) -> Result<DecodeForwardOutput> {
            let DecodeSlot::Mock {
                generated_tokens, ..
            } = slot
            else {
                anyhow::bail!("non-mock slot sent to synthetic finish_request")
            };
            Ok(DecodeForwardOutput {
                output: GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason,
                },
                prefill_duration: Duration::ZERO,
                decode_duration: Duration::ZERO,
            })
        }

        fn discard_request(&self, slot: DecodeSlot) {
            let key = Self::slot_key(&slot);
            let removed_tokens = self.remaining.lock().unwrap().remove(&key).is_some();
            let removed_layers = self.pending_layers.lock().unwrap().remove(&key).is_some();
            let removed_width = self
                .pending_token_widths
                .lock()
                .unwrap()
                .remove(&key)
                .is_some();
            if removed_tokens || removed_layers || removed_width {
                self.events
                    .lock()
                    .unwrap()
                    .push(SchedulingEvent::Discard(key));
            }
        }
    }

    impl DecodeForward for MockForward {
        fn can_reuse_as_strict_prefix(&self, prompt_token_len: usize) -> bool {
            self.prefix_probe_calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.reusable_prefixes && prompt_token_len > 0
        }

        fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
            Ok(DecodeSlot::Mock {
                next_token: req.prompt_tokens.last().copied().unwrap_or_default(),
                generated_tokens: Vec::new(),
            })
        }

        fn forward_decode(
            &self,
            slots: &mut [&mut DecodeSlot],
            _sampling: &[SamplingParams],
        ) -> Result<Vec<TokenId>> {
            let input_tokens: Vec<TokenId> = slots
                .iter()
                .map(|slot| match slot {
                    DecodeSlot::Mock { next_token, .. } => *next_token,
                    DecodeSlot::Real { .. } | DecodeSlot::RealPrefill { .. } => unreachable!(),
                })
                .collect();
            self.calls.lock().unwrap().push(input_tokens.clone());
            Ok(input_tokens.iter().map(|token| token + 10).collect())
        }

        fn is_eos_token(&self, token: TokenId) -> Result<bool> {
            Ok(token == 10)
        }

        fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize> {
            let DecodeSlot::Mock {
                next_token,
                generated_tokens,
            } = slot
            else {
                unreachable!();
            };
            generated_tokens.push(token);
            *next_token = token;
            Ok(generated_tokens.len())
        }

        fn finish_request(
            &self,
            slot: DecodeSlot,
            finish_reason: FinishReason,
        ) -> Result<DecodeForwardOutput> {
            let DecodeSlot::Mock {
                generated_tokens, ..
            } = slot
            else {
                unreachable!();
            };
            Ok(DecodeForwardOutput {
                output: GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason,
                },
                prefill_duration: Duration::ZERO,
                decode_duration: Duration::ZERO,
            })
        }
    }

    impl DecodeForward for PendingFirstTokenForward {
        fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
            if !self.prepare_delay.is_zero() {
                thread::sleep(self.prepare_delay);
            }
            let next_token = req
                .prompt_tokens
                .last()
                .copied()
                .unwrap_or_default()
                .saturating_add(10);
            Ok(real_slot(next_token, true))
        }

        fn forward_decode(
            &self,
            slots: &mut [&mut DecodeSlot],
            _sampling: &[SamplingParams],
        ) -> Result<Vec<TokenId>> {
            let input_tokens: Vec<TokenId> = slots
                .iter()
                .map(|slot| match slot {
                    DecodeSlot::Real { state, .. } => state.next_token,
                    DecodeSlot::Mock { .. } | DecodeSlot::RealPrefill { .. } => unreachable!(),
                })
                .collect();
            self.calls.lock().unwrap().push(input_tokens.clone());
            Ok(input_tokens.iter().map(|token| token + 10).collect())
        }

        fn is_eos_token(&self, _token: TokenId) -> Result<bool> {
            Ok(false)
        }

        fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize> {
            let DecodeSlot::Real { state, .. } = slot else {
                unreachable!();
            };
            state.generated_tokens.push(token);
            state.next_token = token;
            Ok(state.generated_tokens.len())
        }

        fn finish_request(
            &self,
            slot: DecodeSlot,
            finish_reason: FinishReason,
        ) -> Result<DecodeForwardOutput> {
            let DecodeSlot::Real { state, .. } = slot else {
                unreachable!();
            };
            Ok(DecodeForwardOutput {
                output: GenerationOutput {
                    text: String::new(),
                    token_ids: state.generated_tokens,
                    finish_reason,
                },
                prefill_duration: Duration::ZERO,
                decode_duration: Duration::ZERO,
            })
        }
    }

    fn request(prompt_last: TokenId, max_tokens: usize) -> EngineRequest {
        request_with_tokens(vec![1, prompt_last], max_tokens)
    }

    fn request_with_tokens(prompt_tokens: Vec<TokenId>, max_tokens: usize) -> EngineRequest {
        EngineRequest {
            request_id: Uuid::new_v4(),
            prompt_tokens,
            sampling: SamplingParams {
                max_tokens,
                ..SamplingParams::default()
            },
            adapter: None,
            cancel: CancelHandle::new(),
        }
    }

    fn real_slot(next_token: TokenId, first_token_pending: bool) -> DecodeSlot {
        DecodeSlot::Real {
            state: PagedBatchedDecodeState {
                block_table: BlockTable::new(),
                linear_state: LinearAttentionState {
                    recurrent_states: Vec::new(),
                    conv_states: Vec::new(),
                },
                seq_len: 1,
                next_token,
                generated_tokens: Vec::new(),
                step_seed: None,
                registration: None,
                allocated_blocks: Vec::new(),
                prefill_duration: Duration::ZERO,
                decode_duration: Duration::ZERO,
                prompt_tokens: Vec::new(),
                block_size: 16,
                prefill_split_snapshot: None,
                rolling_snapshot: None,
                id: 0,
            },
            prefix_request: None,
            first_token_pending,
        }
    }

    fn assert_token_event(event: Option<EngineEvent>, expected: TokenId) {
        match event {
            Some(EngineEvent::Token { token, ready_at }) => {
                assert_eq!(token, expected);
                assert!(ready_at <= Instant::now());
            }
            other => panic!("expected token {expected}, got {other:?}"),
        }
    }

    fn test_actor(
        rx: mpsc::Receiver<EngineCommand>,
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        prefix_aware_admission: bool,
        prefill_admission_quantum: usize,
        burst_refill: bool,
        response_delivery_policy: ResponseDeliveryPolicy,
    ) -> BatchingEngineActor {
        let (result_tx, delivery_results) = std_mpsc::channel();
        let (wake_tx, _wake_rx) = mpsc::channel(1);
        let delivery_worker = DeliveryWorker::start(
            response_delivery_policy.stream_stall_grace,
            Duration::from_millis(1),
            EngineDeliveryResultSink {
                result_tx,
                pending_results: Vec::new(),
                engine_tx: wake_tx.downgrade(),
            },
        )
        .expect("spawn test response delivery worker");
        BatchingEngineActor::new(
            rx,
            forward,
            max_decode_batch,
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            prefix_aware_admission,
            prefill_admission_quantum,
            burst_refill,
            response_delivery_policy,
            delivery_worker,
            delivery_results,
        )
    }

    fn queue_test_request(
        actor: &mut BatchingEngineActor,
        req: EngineRequest,
        response_tx: mpsc::Sender<EngineEvent>,
    ) {
        let delivery_key = actor
            .register_delivery(req.request_id, response_tx)
            .expect("test delivery lane registers");
        actor.waiting.push_back(QueuedRequest { req, delivery_key });
    }

    fn push_test_active(
        actor: &mut BatchingEngineActor,
        req: EngineRequest,
        response_tx: mpsc::Sender<EngineEvent>,
        slot: DecodeSlot,
    ) {
        let delivery_key = actor
            .register_delivery(req.request_id, response_tx)
            .expect("test delivery lane registers");
        actor.active.push(ActiveRequest {
            req,
            delivery_key,
            delivery_state: ActiveDeliveryState::Ready,
            next_delivery_sequence: 0,
            slot,
        });
    }

    fn settle_active_deliveries(actor: &mut BatchingEngineActor) {
        let deadline = Instant::now() + Duration::from_secs(1);
        loop {
            actor.drain_delivery_results();
            if actor
                .active
                .iter()
                .all(|active| active.delivery_state == ActiveDeliveryState::Ready)
                && actor.delivery_pending_terminal.is_empty()
            {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "test delivery worker did not settle: {:?}",
                actor.snapshot
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn actor_phase_accounting_tracks_totals_maxima_and_slow_counts() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let mut actor = test_actor(
            rx,
            Arc::new(MockForward::default()),
            8,
            false,
            4,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let request_id = Uuid::new_v4();

        actor.record_admission_duration(Duration::from_millis(25), request_id, 128, 512);
        actor.record_admission_duration(Duration::from_millis(125), request_id, 128, 512);
        actor.record_prefill_forward_duration(Duration::from_millis(90), request_id, 512, 4);
        actor.record_prefill_forward_duration(Duration::from_millis(510), request_id, 512, 4);
        actor.record_decode_forward_duration(Duration::from_millis(75), 4);
        actor.record_decode_forward_duration(Duration::from_millis(175), 8);

        assert_eq!(actor.snapshot.total_admission_calls, 2);
        assert_eq!(actor.snapshot.total_admission_ms, 150.0);
        assert_eq!(actor.snapshot.max_admission_ms, 125.0);
        assert_eq!(actor.snapshot.slow_admission_count, 1);
        assert_eq!(actor.snapshot.total_prefill_forward_ms, 600.0);
        assert_eq!(actor.snapshot.max_prefill_forward_ms, 510.0);
        assert_eq!(actor.snapshot.slow_prefill_forward_count, 1);
        assert_eq!(actor.snapshot.total_decode_forward_ms, 250.0);
        assert_eq!(actor.snapshot.max_decode_forward_ms, 175.0);
        assert_eq!(actor.snapshot.slow_decode_forward_count, 1);
    }

    #[tokio::test]
    async fn thinking_budget_forces_close_tokens_into_batched_decode_history() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward.clone(), 4);
        let mut req = request(1, 4);
        req.sampling.thinking_budget = Some(
            ThinkingBudget::new(Some(0), None, req.sampling.max_tokens, vec![90, 91]).unwrap(),
        );

        let mut events = handle.enqueue(req).await.unwrap();
        let output = loop {
            match events.recv().await {
                Some(EngineEvent::Done { output }) => break output,
                Some(EngineEvent::Error(error)) => panic!("generation failed: {error}"),
                Some(EngineEvent::Token { .. }) => {}
                None => panic!("engine closed before completion"),
            }
        };

        assert_eq!(&output.token_ids[..2], &[90, 91]);
        let calls = forward.calls.lock().unwrap();
        assert!(
            calls.iter().any(|batch| batch.contains(&90)),
            "the second forced close token must be decoded from KV state containing the first"
        );
        assert!(output.token_ids.len() > 2, "answer decoding must resume");
    }

    /// Test double whose `forward_decode` blocks until the test releases a
    /// step, and which records prepare/decode events into a shared log —
    /// the deterministic harness for asserting barrier ordering.
    struct GatedForward {
        gate: StdMutex<std::sync::mpsc::Receiver<()>>,
        events: Arc<StdMutex<Vec<String>>>,
    }

    impl GatedForward {
        fn new(events: Arc<StdMutex<Vec<String>>>) -> (Self, std::sync::mpsc::Sender<()>) {
            let (tx, rx) = std::sync::mpsc::channel();
            (
                Self {
                    gate: StdMutex::new(rx),
                    events,
                },
                tx,
            )
        }
    }

    impl DecodeForward for GatedForward {
        fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
            self.events.lock().unwrap().push(format!(
                "prepare:{}",
                req.prompt_tokens.last().copied().unwrap_or_default()
            ));
            Ok(DecodeSlot::Mock {
                next_token: req.prompt_tokens.last().copied().unwrap_or_default(),
                generated_tokens: Vec::new(),
            })
        }

        fn forward_decode(
            &self,
            slots: &mut [&mut DecodeSlot],
            _sampling: &[SamplingParams],
        ) -> Result<Vec<TokenId>> {
            // Hold the decode step until the test releases it.
            self.gate.lock().unwrap().recv().ok();
            self.events.lock().unwrap().push("decode".to_string());
            Ok(slots
                .iter()
                .map(|slot| match slot {
                    DecodeSlot::Mock { next_token, .. } => *next_token + 10,
                    DecodeSlot::Real { .. } | DecodeSlot::RealPrefill { .. } => unreachable!(),
                })
                .collect())
        }

        fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize> {
            let DecodeSlot::Mock {
                next_token,
                generated_tokens,
            } = slot
            else {
                unreachable!();
            };
            generated_tokens.push(token);
            *next_token = token;
            Ok(generated_tokens.len())
        }

        fn finish_request(
            &self,
            slot: DecodeSlot,
            finish_reason: FinishReason,
        ) -> Result<DecodeForwardOutput> {
            let DecodeSlot::Mock {
                generated_tokens, ..
            } = slot
            else {
                unreachable!();
            };
            Ok(DecodeForwardOutput {
                output: GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason,
                },
                prefill_duration: Duration::ZERO,
                decode_duration: Duration::ZERO,
            })
        }
    }

    #[tokio::test]
    async fn cached_snapshot_remains_immediate_during_blocked_forward() {
        let events: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let (forward, release) = GatedForward::new(events);
        let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 1);
        let mut response = handle.enqueue(request(100, 1)).await.unwrap();

        let publication_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let cached = handle.cached_snapshot();
            if cached.current_batch_size == 1 && cached.total_decode_forwards == 1 {
                break;
            }
            assert!(
                Instant::now() < publication_deadline,
                "actor did not publish the in-flight batch before entering forward: {cached:?}"
            );
            tokio::task::yield_now().await;
        }

        tokio::time::sleep(Duration::from_millis(25)).await;
        let reader = handle.clone();
        let (snapshot_tx, snapshot_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = snapshot_tx.send(reader.cached_snapshot());
        });
        let cached = snapshot_rx
            .recv_timeout(Duration::from_millis(100))
            .expect("cached read must not wait for the blocked actor");
        assert_eq!(cached.current_batch_size, 1, "{cached:?}");
        assert_eq!(cached.active_decode, 1, "{cached:?}");
        assert_eq!(cached.total_decode_forwards, 1, "{cached:?}");
        assert!(cached.snapshot_age_ms >= 20, "{cached:?}");

        let strong_snapshot = {
            let handle = handle.clone();
            tokio::spawn(async move { handle.snapshot().await })
        };
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(
            !strong_snapshot.is_finished(),
            "strong snapshot must remain an actor barrier while forward is blocked"
        );

        let later = handle.cached_snapshot();
        assert!(
            later.snapshot_age_ms >= cached.snapshot_age_ms,
            "cache age must expose the actor's time inside forward: first={cached:?} later={later:?}"
        );

        release.send(()).unwrap();
        let strong = tokio::time::timeout(Duration::from_secs(2), strong_snapshot)
            .await
            .expect("strong snapshot must finish after forward is released")
            .unwrap()
            .unwrap();
        assert_eq!(strong.current_batch_size, 0, "{strong:?}");
        assert_eq!(strong.snapshot_age_ms, 0, "{strong:?}");

        loop {
            match response.recv().await {
                Some(EngineEvent::Done { .. }) => break,
                Some(EngineEvent::Error(error)) => panic!("generation failed: {error}"),
                Some(EngineEvent::Token { .. }) => {}
                None => panic!("engine closed before completion"),
            }
        }
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn adapter_swap_waits_for_active_request_and_pauses_admission() {
        let events: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let (forward, release) = GatedForward::new(events.clone());
        let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 8);

        // Request A needs two decode steps; hold it mid-generation after
        // the first release.
        let mut rx_a = handle.enqueue(request(100, 2)).await.unwrap();
        release.send(()).unwrap(); // A's first step runs; A stays active.
        assert!(matches!(rx_a.recv().await, Some(EngineEvent::Token { .. })));

        // With A mid-generation: queue a swap, then request B behind it.
        let swap_events = events.clone();
        let swap_task = {
            let handle = handle.clone();
            tokio::spawn(async move {
                handle
                    .swap_adapter(Box::new(move || {
                        swap_events.lock().unwrap().push("swap".to_string());
                        Ok(())
                    }))
                    .await
            })
        };
        // Give the swap command time to land in the actor's pending queue
        // while A is still blocked on the gate.
        tokio::time::sleep(Duration::from_millis(50)).await;
        let mut rx_b = handle.enqueue(request(200, 1)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Release everything: A's second step, then B's step after the swap.
        release.send(()).unwrap();
        release.send(()).unwrap();

        swap_task.await.unwrap().unwrap();
        loop {
            match rx_a.recv().await {
                Some(EngineEvent::Done { .. }) => break,
                Some(_) => {}
                None => panic!("request A channel closed before Done"),
            }
        }
        loop {
            match rx_b.recv().await {
                Some(EngineEvent::Done { .. }) => break,
                Some(_) => {}
                None => panic!("request B channel closed before Done"),
            }
        }

        let log = events.lock().unwrap().clone();
        let swap_idx = log.iter().position(|e| e == "swap").expect("swap ran");
        let prepare_b_idx = log
            .iter()
            .position(|e| e == "prepare:200")
            .expect("B admitted");
        // The swap executed only after A's final decode step (both steps
        // precede it), and B was admitted only after the swap — so no
        // request ever spans a weight change.
        assert!(
            swap_idx < prepare_b_idx,
            "B must be admitted after the swap: {log:?}"
        );
        let decodes_before_swap = log[..swap_idx].iter().filter(|e| *e == "decode").count();
        assert_eq!(
            decodes_before_swap, 2,
            "both of A's decode steps precede the swap: {log:?}"
        );

        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn health_checked_adapter_swap_rejects_while_active_request_drains() {
        let events: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let (forward, release) = GatedForward::new(events.clone());
        let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 8);
        let mut response = handle.enqueue(request(100, 2)).await.unwrap();
        release.send(()).unwrap();
        assert!(matches!(
            response.recv().await,
            Some(EngineEvent::Token { .. })
        ));

        let backend_health = BackendHealthHandle::default();
        let swap_health = backend_health.clone();
        let swap_handle = handle.clone();
        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fired_in_swap = fired.clone();
        let swap = tokio::spawn(async move {
            swap_handle
                .swap_adapter_while_healthy(
                    Box::new(move || {
                        fired_in_swap.store(true, std::sync::atomic::Ordering::SeqCst);
                        Ok(())
                    }),
                    &swap_health,
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!swap.is_finished());

        backend_health.quarantine("injected unknown completion during adapter barrier");
        let error = tokio::time::timeout(Duration::from_millis(250), swap)
            .await
            .expect("quarantine must interrupt the adapter barrier wait")
            .unwrap()
            .expect_err("quarantined adapter barrier must reject");
        assert!(error.to_string().contains("requires restart"));
        assert!(!fired.load(std::sync::atomic::Ordering::SeqCst));

        // Let the actor settle so this test does not leave its worker thread
        // blocked. Production adapter closures recheck health and reject here.
        release.send(()).unwrap();
        handle.stop().await.unwrap();
    }

    /// A healthy lane must complete while another lane is still inside its
    /// backpressure grace window. The stalled lane is evicted only after that
    /// window elapses; neither phase may park compute or the control plane.
    #[tokio::test]
    async fn stalled_streaming_client_is_evicted_and_others_proceed() {
        let forward = Arc::new(MockForward::default());
        let response_delivery_policy = ResponseDeliveryPolicy {
            stream_stall_grace: Duration::from_millis(1000),
            stream_stall_grace_source: ConfigValueSource::ConfigFile,
        };
        let handle = BatchingEngineHandle::start_with_policy(
            forward,
            8,
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            env_prefix_aware_admission(),
            env_prefill_admission_quantum_for_policy(8, None),
            false,
            response_delivery_policy,
        );

        // A wants 200 tokens but its events are NEVER read — the channel
        // (cap 64) fills and the client looks suspended.
        let mut rx_a = handle.enqueue(request(101, 200)).await.unwrap();
        let pressure_deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let snapshot = handle.cached_snapshot();
            if snapshot.response_delivery_backpressured == 1 {
                assert_eq!(snapshot.response_stall_evictions, 0, "{snapshot:?}");
                break;
            }
            assert!(
                Instant::now() < pressure_deadline,
                "stalled request never reached backpressure: {snapshot:?}"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // B arrives after A is observably full but before A's grace expires.
        let mut rx_b = handle.enqueue(request(7, 1)).await.unwrap();
        let b_done = tokio::time::timeout(Duration::from_millis(500), async {
            loop {
                match rx_b.recv().await {
                    Some(EngineEvent::Done { .. }) => break true,
                    Some(_) => {}
                    None => break false,
                }
            }
        })
        .await
        .expect("engine must not stay parked on the stalled client");
        assert!(b_done, "healthy request completes despite the stalled one");
        let during_grace = handle.snapshot().await.unwrap();
        assert_eq!(during_grace.response_stall_evictions, 0, "{during_grace:?}");

        // A is cancelled only once its full grace has elapsed. Draining its
        // channel ends in the stall error (or close), never a Done.
        let eviction_deadline = Instant::now() + Duration::from_secs(3);
        loop {
            let snapshot = handle.cached_snapshot();
            if snapshot.response_stall_evictions == 1 {
                break;
            }
            assert!(
                Instant::now() < eviction_deadline,
                "stalled request was not evicted after its grace: {snapshot:?}"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let a_outcome = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                match rx_a.recv().await {
                    Some(EngineEvent::Error(e)) => break Some(e),
                    Some(EngineEvent::Done { .. }) => {
                        panic!("stalled request must be cancelled, not completed")
                    }
                    Some(_) => {}
                    None => break None,
                }
            }
        })
        .await
        .unwrap();
        if let Some(err) = a_outcome {
            assert!(err.contains("stalled"), "{err}");
        }

        let snapshot = handle.snapshot().await.unwrap();
        assert_eq!(snapshot.stream_stall_grace_ms, 1000);
        assert_eq!(
            snapshot.stream_stall_grace_source,
            ConfigValueSource::ConfigFile
        );
        assert_eq!(snapshot.response_backpressure_events, 1);
        assert!(
            snapshot.response_backpressure_wait_ms >= 1000,
            "{snapshot:?}"
        );
        assert_eq!(snapshot.response_stall_evictions, 1);
        assert_eq!(snapshot.response_channel_closed, 0);

        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn final_token_precedes_done_with_one_response_slot() {
        let response_delivery_policy = ResponseDeliveryPolicy {
            stream_stall_grace: Duration::from_secs(2),
            stream_stall_grace_source: ConfigValueSource::ConfigFile,
        };
        let handle = BatchingEngineHandle::start_with_policy(
            Arc::new(MockForward::default()),
            1,
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            false,
            1,
            false,
            response_delivery_policy,
        );
        let mut response = handle
            .enqueue_with_response_capacity(request(101, 1), 1)
            .await
            .unwrap();

        let pressure_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = handle.cached_snapshot();
            if snapshot.response_delivery_backpressured == 1 {
                assert_eq!(snapshot.active_decode, 0, "{snapshot:?}");
                assert_eq!(snapshot.response_delivery_in_flight, 0, "{snapshot:?}");
                assert_eq!(
                    snapshot.response_delivery_pending_terminal, 1,
                    "{snapshot:?}"
                );
                assert_eq!(snapshot.response_stall_evictions, 0, "{snapshot:?}");
                break;
            }
            assert!(
                Instant::now() < pressure_deadline,
                "terminal delivery never observed its full slot: {snapshot:?}"
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // The actor remains responsive while Done is waiting for the sole
        // channel slot occupied by the final token.
        let snapshot = tokio::time::timeout(Duration::from_millis(250), handle.snapshot())
            .await
            .expect("control plane must remain responsive")
            .unwrap();
        assert_eq!(snapshot.response_delivery_pending_terminal, 1);

        assert_token_event(response.recv().await, 111);
        assert!(matches!(
            tokio::time::timeout(Duration::from_secs(1), response.recv())
                .await
                .expect("Done must follow after the final-token slot is released"),
            Some(EngineEvent::Done {
                output: BatchedGenerationOutput {
                    token_ids,
                    completion_tokens: 1,
                    finish_reason: FinishReason::MaxTokens,
                    ..
                }
            }) if token_ids == vec![111]
        ));

        let settled = handle.snapshot().await.unwrap();
        assert_eq!(settled.response_delivery_pending_terminal, 0, "{settled:?}");
        assert_eq!(settled.response_delivery_backpressured, 0, "{settled:?}");
        assert_eq!(settled.response_stall_evictions, 0, "{settled:?}");
        handle.stop().await.unwrap();
    }

    #[test]
    fn cancellation_after_token_delivery_keeps_terminal_generation_accounted() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(MockForward::default());
        let mut actor = test_actor(
            rx,
            forward,
            1,
            false,
            1,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let req = request(101, 2);
        let request_id = req.request_id;
        let (response_tx, mut response_rx) = mpsc::channel(2);
        push_test_active(
            &mut actor,
            req,
            response_tx,
            DecodeSlot::Mock {
                next_token: 101,
                generated_tokens: Vec::new(),
            },
        );

        actor.submit_token_delivery(0, 111, Instant::now());
        assert_token_event(response_rx.blocking_recv(), 111);
        assert!(matches!(
            actor.active[0].delivery_state,
            ActiveDeliveryState::InFlight { sequence: 0 }
        ));

        // The delivery worker has made progress, but the actor has not yet
        // consumed that acknowledgement. Cancellation must account for the
        // terminal sequence the worker creates after the delivered token.
        actor.cancel(request_id);
        assert_eq!(actor.snapshot.response_delivery_pending_terminal, 1);
        assert!(matches!(
            response_rx.blocking_recv(),
            Some(EngineEvent::Error(error)) if error == "request cancelled"
        ));

        settle_active_deliveries(&mut actor);
        assert!(actor.active.is_empty());
        assert_eq!(actor.snapshot.response_delivery_pending_terminal, 0);
        assert_eq!(actor.snapshot.response_channel_closed, 0);
        assert_eq!(actor.snapshot.response_stall_evictions, 0);
    }

    #[test]
    fn cancellation_after_worker_retires_closed_lane_is_nonfatal() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let mut actor = test_actor(
            rx,
            Arc::new(MockForward::default()),
            1,
            false,
            1,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let req = request(101, 2);
        let request_id = req.request_id;
        let (response_tx, response_rx) = mpsc::channel(1);
        drop(response_rx);
        push_test_active(
            &mut actor,
            req,
            response_tx,
            DecodeSlot::Mock {
                next_token: 101,
                generated_tokens: Vec::new(),
            },
        );

        actor.submit_token_delivery(0, 111, Instant::now());
        let mut held_closed = actor
            .delivery_results
            .recv_timeout(Duration::from_secs(1))
            .expect("worker must retire the disconnected lane");
        assert_eq!(held_closed.len(), 1);
        let held_closed = held_closed.pop().unwrap();
        assert!(matches!(held_closed, DeliveryResult::Closed { .. }));

        // Hold the Closed result so Cancel still sees an InFlight row after
        // the worker has retired its lane. Terminate must be idempotent for
        // that generation rather than becoming a fatal UnknownKey protocol
        // error.
        actor.cancel(request_id);
        actor.handle_delivery_result(held_closed);

        // A later delivery is a command-channel barrier: by the time its
        // result arrives, the raced Terminate has also been processed.
        let (barrier_tx, mut barrier_rx) = mpsc::channel(1);
        let barrier_key = actor
            .register_delivery(Uuid::from_u128(999), barrier_tx)
            .unwrap();
        assert!(actor.send_delivery(DeliveryCommand::Deliver {
            key: barrier_key,
            batch: DeliveryBatch::Token {
                token: 999,
                ready_at: Instant::now(),
                sequence: 0,
            },
        }));
        assert_token_event(barrier_rx.blocking_recv(), 999);

        loop {
            let results = actor
                .delivery_results
                .recv_timeout(Duration::from_secs(1))
                .expect("barrier delivery result cohort must arrive");
            let mut barrier_delivered = false;
            for result in results {
                barrier_delivered |= matches!(
                    &result,
                    DeliveryResult::Delivered {
                        key,
                        sequence: 0,
                        terminal: false,
                        ..
                    } if *key == barrier_key
                );
                actor.handle_delivery_result(result);
            }
            if barrier_delivered {
                break;
            }
        }

        assert!(!actor.stopped);
        assert!(actor.delivery_pending_terminal.is_empty());
        assert_eq!(actor.snapshot.response_channel_closed, 1);
        assert_eq!(actor.snapshot.total_errors, 0);
    }

    #[tokio::test]
    async fn concurrent_stop_callers_receive_acknowledgements() {
        let events = Arc::new(StdMutex::new(Vec::new()));
        let (forward, release) = GatedForward::new(events);
        let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 1);
        let _response = handle.enqueue(request(101, 4)).await.unwrap();

        let forward_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = handle.cached_snapshot();
            if snapshot.current_batch_size == 1 {
                break;
            }
            assert!(
                Instant::now() < forward_deadline,
                "request did not enter its gated forward: {snapshot:?}"
            );
            tokio::task::yield_now().await;
        }

        let first = tokio::spawn({
            let handle = handle.clone();
            async move { handle.stop().await }
        });
        let second = tokio::spawn({
            let handle = handle.clone();
            async move { handle.stop().await }
        });
        tokio::task::yield_now().await;
        release.send(()).unwrap();

        let (first, second) = tokio::time::timeout(Duration::from_secs(2), async {
            tokio::join!(first, second)
        })
        .await
        .expect("both accepted Stop commands must be acknowledged");
        first.unwrap().unwrap();
        second.unwrap().unwrap();
    }

    #[tokio::test]
    async fn dropping_final_handle_stops_actor_and_delivery_worker() {
        let events = Arc::new(StdMutex::new(Vec::new()));
        let (forward, release) = GatedForward::new(events);
        let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 1);
        let req = request(101, 4);
        let cancel = req.cancel.clone();
        let mut response = handle.enqueue(req).await.unwrap();

        let forward_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = handle.cached_snapshot();
            if snapshot.current_batch_size == 1 {
                break;
            }
            assert!(
                Instant::now() < forward_deadline,
                "request did not enter its gated forward: {snapshot:?}"
            );
            tokio::task::yield_now().await;
        }

        drop(handle);
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while response.recv().await.is_some() {}
        })
        .await
        .expect("dropping the final strong handle must tear down both threads");
        assert!(cancel.is_cancelled());
    }

    #[test]
    fn closed_response_channel_is_counted_without_backpressure() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(MockForward::default());
        let mut actor = test_actor(
            rx,
            forward,
            8,
            false,
            1,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let (response_tx, response_rx) = mpsc::channel(1);
        drop(response_rx);
        let req = request(101, 2);
        push_test_active(
            &mut actor,
            req,
            response_tx,
            DecodeSlot::Mock {
                next_token: 101,
                generated_tokens: Vec::new(),
            },
        );

        actor.submit_token_delivery(0, 111, Instant::now());
        settle_active_deliveries(&mut actor);
        assert!(actor.active.is_empty());
        assert_eq!(actor.snapshot.response_channel_closed, 1);
        assert_eq!(actor.snapshot.response_backpressure_events, 0);
        assert_eq!(actor.snapshot.response_backpressure_wait_ms, 0);
        assert_eq!(actor.snapshot.response_stall_evictions, 0);
    }

    #[test]
    fn delivered_token_carries_response_ready_timestamp() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(MockForward::default());
        let mut actor = test_actor(
            rx,
            forward,
            8,
            false,
            1,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let (response_tx, mut response_rx) = mpsc::channel(1);
        push_test_active(
            &mut actor,
            request(101, 2),
            response_tx,
            DecodeSlot::Mock {
                next_token: 101,
                generated_tokens: Vec::new(),
            },
        );

        let before = Instant::now();
        let ready_at = Instant::now();
        actor.submit_token_delivery(0, 111, ready_at);
        let after = Instant::now();
        match response_rx.blocking_recv() {
            Some(EngineEvent::Token { token, ready_at }) => {
                assert_eq!(token, 111);
                assert!(ready_at >= before);
                assert!(ready_at <= after);
            }
            other => panic!("expected timed token, got {other:?}"),
        }
    }

    /// Forward whose `prepare_request` reports a block-pool shortage for a
    /// marked prompt while the flag is up, and whose decode steps block on
    /// a test-released gate (so a request stays ACTIVE deterministically)
    /// — the transient-admission case.
    struct FlakyPrepareForward {
        inner: MockForward,
        fail_oom_for: TokenId,
        failing: std::sync::atomic::AtomicBool,
        gate: StdMutex<std::sync::mpsc::Receiver<()>>,
    }

    impl DecodeForward for FlakyPrepareForward {
        fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
            if req.prompt_tokens.last().copied() == Some(self.fail_oom_for)
                && self.failing.load(std::sync::atomic::Ordering::SeqCst)
            {
                anyhow::bail!("out of memory: no free blocks available (need 2, have 0)");
            }
            self.inner.prepare_request(req)
        }
        fn forward_decode(
            &self,
            slots: &mut [&mut DecodeSlot],
            sampling: &[SamplingParams],
        ) -> Result<Vec<TokenId>> {
            self.gate.lock().unwrap().recv().ok();
            self.inner.forward_decode(slots, sampling)
        }
        fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize> {
            self.inner.accept_token(slot, token)
        }
        fn finish_request(
            &self,
            slot: DecodeSlot,
            finish_reason: FinishReason,
        ) -> Result<DecodeForwardOutput> {
            self.inner.finish_request(slot, finish_reason)
        }
    }

    /// A transient block shortage under concurrent load must keep the
    /// request QUEUED for the next admission cycle — not fail it
    /// instantly. (Blocks free up as other requests finish.)
    #[tokio::test]
    async fn transient_block_shortage_keeps_request_queued() {
        let (release, gate) = std::sync::mpsc::channel();
        let forward = Arc::new(FlakyPrepareForward {
            inner: MockForward::default(),
            fail_oom_for: 999,
            failing: std::sync::atomic::AtomicBool::new(true),
            gate: StdMutex::new(gate),
        });
        let handle = BatchingEngineHandle::start_with_options(forward.clone(), 8);

        // A needs 3 gated decode steps — it stays active while B knocks.
        let mut rx_a = handle.enqueue(request(100, 3)).await.unwrap();
        release.send(()).unwrap();
        assert!(matches!(rx_a.recv().await, Some(EngineEvent::Token { .. })));

        // B hits the (transient) shortage — it must NOT receive an error.
        let mut rx_b = handle.enqueue(request(999, 1)).await.unwrap();
        release.send(()).unwrap(); // A step 2; B's admission retries with A active
        assert!(matches!(rx_a.recv().await, Some(EngineEvent::Token { .. })));
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            matches!(rx_b.try_recv(), Err(mpsc::error::TryRecvError::Empty)),
            "request must stay queued through a transient shortage"
        );

        // Blocks "free up": B admits on a later cycle and completes.
        forward
            .failing
            .store(false, std::sync::atomic::Ordering::SeqCst);
        release.send(()).unwrap(); // A step 3 (done)
        release.send(()).unwrap(); // B's step
        let b_done = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                match rx_b.recv().await {
                    Some(EngineEvent::Done { .. }) => break true,
                    Some(EngineEvent::Error(e)) => panic!("B must not fail: {e}"),
                    Some(_) => {}
                    None => break false,
                }
            }
        })
        .await
        .unwrap();
        assert!(b_done);
        handle.stop().await.unwrap();
    }

    /// Forward that starves one slot's KV growth exactly once — the
    /// single-victim decode-growth case.
    struct StarvingGrowForward {
        inner: MockForward,
        starve_once: std::sync::atomic::AtomicBool,
    }

    impl DecodeForward for StarvingGrowForward {
        fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
            self.inner.prepare_request(req)
        }
        fn grow_for_decode(&self, slots: &mut [&mut DecodeSlot]) -> Result<Vec<usize>> {
            if slots.len() > 1
                && self
                    .starve_once
                    .swap(false, std::sync::atomic::Ordering::SeqCst)
            {
                return Ok(vec![0]);
            }
            Ok(Vec::new())
        }
        fn forward_decode(
            &self,
            slots: &mut [&mut DecodeSlot],
            sampling: &[SamplingParams],
        ) -> Result<Vec<TokenId>> {
            self.inner.forward_decode(slots, sampling)
        }
        fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize> {
            self.inner.accept_token(slot, token)
        }
        fn finish_request(
            &self,
            slot: DecodeSlot,
            finish_reason: FinishReason,
        ) -> Result<DecodeForwardOutput> {
            self.inner.finish_request(slot, finish_reason)
        }
    }

    /// A request that outgrows the KV pool finishes as a `length`
    /// casualty — the rest of the batch keeps decoding. (The old path
    /// called finish_batch_with_error and killed EVERY active request
    /// because one conversation got long.)
    #[tokio::test]
    async fn kv_growth_starvation_finishes_only_the_victim() {
        let forward = Arc::new(StarvingGrowForward {
            inner: MockForward::default(),
            starve_once: std::sync::atomic::AtomicBool::new(true),
        });
        let handle = BatchingEngineHandle::start_with_options(forward, 8);

        let rx_a = handle.enqueue(request(100, 5)).await.unwrap();
        let rx_b = handle.enqueue(request(200, 5)).await.unwrap();

        let outcome = |mut rx: mpsc::Receiver<EngineEvent>| async move {
            loop {
                match rx.recv().await {
                    Some(EngineEvent::Done { output }) => break Ok(output),
                    Some(EngineEvent::Error(e)) => break Err(e),
                    Some(_) => {}
                    None => break Err("closed".to_string()),
                }
            }
        };
        let (a, b) = tokio::join!(
            tokio::time::timeout(Duration::from_secs(10), outcome(rx_a)),
            tokio::time::timeout(Duration::from_secs(10), outcome(rx_b)),
        );
        let a = a
            .unwrap()
            .expect("victim finishes cleanly, not with an engine error");
        let b = b.unwrap().expect("survivor completes");
        // One of the two was starved (admission order isn't pinned);
        // whichever it was finished early as `length`, the other decoded
        // all 5 tokens.
        let (victim, survivor) = if a.completion_tokens < 5 {
            (a, b)
        } else {
            (b, a)
        };
        assert!(victim.completion_tokens < 5, "victim was cut short");
        assert_eq!(survivor.completion_tokens, 5, "survivor unaffected");
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn adapter_swap_executes_immediately_when_idle() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward, 8);
        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = fired.clone();
        handle
            .swap_adapter(Box::new(move || {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            }))
            .await
            .unwrap();
        assert!(fired.load(std::sync::atomic::Ordering::SeqCst));
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn adapter_swap_error_propagates_to_caller() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward, 8);
        let err = handle
            .swap_adapter(Box::new(|| Err("load exploded".to_string())))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("load exploded"));
        // The engine keeps serving after a failed swap.
        let mut rx = handle.enqueue(request(7, 1)).await.unwrap();
        assert!(matches!(rx.recv().await, Some(EngineEvent::Token { .. })));
        handle.stop().await.unwrap();
    }

    #[test]
    fn decode_capacity_block_count_rounds_up() {
        assert_eq!(blocks_needed_for_tokens(0, 16), 0);
        assert_eq!(blocks_needed_for_tokens(1, 16), 1);
        assert_eq!(blocks_needed_for_tokens(16, 16), 1);
        assert_eq!(blocks_needed_for_tokens(17, 16), 2);
    }

    #[test]
    fn default_rowwise_decode_uses_batched_decode_unless_overridden() {
        // Snapshot + restore the env var so other tests that share this
        // process see the original value.
        let prior = std::env::var("KILN_BATCH_DECODE_ROWWISE").ok();
        // SAFETY: tests in this crate that touch this env var must not run in
        // parallel; cargo defaults to serial within a single test binary.
        unsafe {
            std::env::remove_var("KILN_BATCH_DECODE_ROWWISE");
        }
        assert!(!default_rowwise_decode());
        unsafe {
            std::env::set_var("KILN_BATCH_DECODE_ROWWISE", "0");
        }
        assert!(!default_rowwise_decode());
        unsafe {
            std::env::set_var("KILN_BATCH_DECODE_ROWWISE", "1");
        }
        assert!(default_rowwise_decode());
        match prior {
            Some(v) => unsafe { std::env::set_var("KILN_BATCH_DECODE_ROWWISE", v) },
            None => unsafe { std::env::remove_var("KILN_BATCH_DECODE_ROWWISE") },
        }
    }

    #[test]
    fn max_decode_batch_default_is_backend_aware() {
        let prior = std::env::var("KILN_MAX_DECODE_BATCH").ok();
        // SAFETY: tests in this crate that touch this env var must not run in
        // parallel; cargo defaults to serial within a single test binary.
        unsafe {
            std::env::remove_var("KILN_MAX_DECODE_BATCH");
        }
        let vulkan_policy =
            DecodeBatcherPolicy::for_backend("vulkan", kiln_tensor::Device::Vulkan(0));
        let metal_policy = DecodeBatcherPolicy::for_backend("metal", kiln_tensor::Device::Metal(0));
        assert_eq!(env_max_decode_batch_for_policy(None), 8);
        // CUDA: the legacy batcher stays serial (max_batch 1) but the
        // ENGINE width must be the engine default — the policy-routing
        // change that reused max_batch serialized all concurrent CUDA
        // requests.
        let cuda_policy = DecodeBatcherPolicy::for_backend("cuda", kiln_tensor::Device::Cuda(0));
        assert_eq!(cuda_policy.max_batch, 1);
        assert_eq!(env_max_decode_batch_for_policy(Some(cuda_policy)), 8);
        assert_eq!(env_max_decode_batch_for_policy(Some(vulkan_policy)), 64);
        assert_eq!(env_max_decode_batch_for_policy(Some(metal_policy)), 8);
        unsafe {
            std::env::set_var("KILN_MAX_DECODE_BATCH", "24");
        }
        assert_eq!(env_max_decode_batch_for_policy(None), 24);
        assert_eq!(env_max_decode_batch_for_policy(Some(vulkan_policy)), 24);
        match prior {
            Some(v) => unsafe { std::env::set_var("KILN_MAX_DECODE_BATCH", v) },
            None => unsafe { std::env::remove_var("KILN_MAX_DECODE_BATCH") },
        }
    }

    #[test]
    fn prefill_admission_quantum_default_and_override() {
        let prior = std::env::var("KILN_BATCH_PREFILL_ADMISSION_QUANTUM").ok();
        // SAFETY: tests in this crate that touch this env var must not run in
        // parallel; cargo defaults to serial within a single test binary.
        unsafe {
            std::env::remove_var("KILN_BATCH_PREFILL_ADMISSION_QUANTUM");
        }
        let cuda_policy = DecodeBatcherPolicy::for_backend("cuda", kiln_tensor::Device::Cuda(0));
        let vulkan_policy =
            DecodeBatcherPolicy::for_backend("vulkan", kiln_tensor::Device::Vulkan(0));
        let metal_policy = DecodeBatcherPolicy::for_backend("metal", kiln_tensor::Device::Metal(0));
        assert_eq!(env_prefill_admission_quantum_for_policy(64, None), 4);
        assert_eq!(
            env_prefill_admission_quantum_for_policy(64, Some(vulkan_policy)),
            64
        );
        // #1082: CUDA gets the same full-width quantum as Vulkan (regression fix).
        assert_eq!(
            env_prefill_admission_quantum_for_policy(64, Some(cuda_policy)),
            64
        );
        // Metal stays at the latency default (the Metal lane can opt in separately).
        assert_eq!(
            env_prefill_admission_quantum_for_policy(64, Some(metal_policy)),
            4
        );
        // CUDA still clamps to demand when the decode width is small.
        assert_eq!(
            env_prefill_admission_quantum_for_policy(2, Some(cuda_policy)),
            2
        );
        assert_eq!(env_prefill_admission_quantum_for_policy(2, None), 2);
        assert_eq!(
            env_prefill_admission_quantum_for_policy(2, Some(vulkan_policy)),
            2
        );
        assert_eq!(env_prefill_admission_quantum_for_policy(0, None), 1);
        unsafe {
            std::env::set_var("KILN_BATCH_PREFILL_ADMISSION_QUANTUM", "24");
        }
        assert_eq!(env_prefill_admission_quantum_for_policy(64, None), 24);
        assert_eq!(
            env_prefill_admission_quantum_for_policy(64, Some(vulkan_policy)),
            24
        );
        unsafe {
            std::env::set_var("KILN_BATCH_PREFILL_ADMISSION_QUANTUM", "999");
        }
        assert_eq!(env_prefill_admission_quantum_for_policy(64, None), 64);
        unsafe {
            std::env::set_var("KILN_BATCH_PREFILL_ADMISSION_QUANTUM", "0");
        }
        assert_eq!(env_prefill_admission_quantum_for_policy(64, None), 4);
        assert_eq!(
            env_prefill_admission_quantum_for_policy(64, Some(vulkan_policy)),
            64
        );
        unsafe {
            std::env::set_var("KILN_BATCH_PREFILL_ADMISSION_QUANTUM", "bad");
        }
        assert_eq!(env_prefill_admission_quantum_for_policy(64, None), 4);
        assert_eq!(
            env_prefill_admission_quantum_for_policy(64, Some(vulkan_policy)),
            64
        );
        match prior {
            Some(v) => unsafe { std::env::set_var("KILN_BATCH_PREFILL_ADMISSION_QUANTUM", v) },
            None => unsafe { std::env::remove_var("KILN_BATCH_PREFILL_ADMISSION_QUANTUM") },
        }
    }

    #[test]
    fn real_decode_selection_skips_first_token_rows_for_model_step() {
        let mut pending_a = real_slot(101, true);
        let mut ready_a = real_slot(202, false);
        let mut pending_b = real_slot(303, true);
        let mut ready_b = real_slot(404, false);
        let mut slots = vec![&mut pending_a, &mut ready_a, &mut pending_b, &mut ready_b];
        let sampling = vec![
            SamplingParams {
                max_tokens: 11,
                ..SamplingParams::default()
            },
            SamplingParams {
                max_tokens: 22,
                ..SamplingParams::default()
            },
            SamplingParams {
                max_tokens: 33,
                ..SamplingParams::default()
            },
            SamplingParams {
                max_tokens: 44,
                ..SamplingParams::default()
            },
        ];
        let mut output = vec![0; slots.len()];

        let (decode_indices, decode_params) =
            collect_ready_decode_indices(&mut slots, &sampling, &mut output).unwrap();

        assert_eq!(output, vec![101, 0, 303, 0]);
        assert_eq!(decode_indices, vec![1, 3]);
        assert_eq!(decode_params.len(), decode_indices.len());
        assert_eq!(decode_params[0].max_tokens, 22);
        assert_eq!(decode_params[1].max_tokens, 44);
        drop(slots);
        assert!(matches!(
            pending_a,
            DecodeSlot::Real {
                first_token_pending: false,
                ..
            }
        ));
        assert!(matches!(
            pending_b,
            DecodeSlot::Real {
                first_token_pending: false,
                ..
            }
        ));
    }

    #[test]
    fn prefill_admission_quantum_limits_each_actor_cycle() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(MockForward::default());
        let mut actor = test_actor(
            rx,
            forward,
            8,
            false,
            3,
            false,
            ResponseDeliveryPolicy::default(),
        );
        for idx in 0..8 {
            let (response_tx, _response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
            queue_test_request(&mut actor, request(100 + idx as TokenId, 1), response_tx);
        }

        actor.admit_waiting();
        assert_eq!(actor.active.len(), 3);
        assert_eq!(actor.waiting.len(), 5);
        assert_eq!(actor.snapshot.max_prefill_admission_quantum, 3);
        assert_eq!(actor.snapshot.total_prefill_admission_cycles, 1);
        assert_eq!(actor.snapshot.total_prefill_tokens, 6);

        actor.run_decode_batch();
        assert_eq!(actor.active.len(), 0);

        actor.admit_waiting();
        assert_eq!(actor.active.len(), 3);
        assert_eq!(actor.waiting.len(), 2);
        assert_eq!(actor.snapshot.total_prefill_admission_cycles, 2);
        assert_eq!(actor.snapshot.total_prefill_tokens, 12);
    }

    #[test]
    fn ready_decode_rows_limit_followup_prefill_admission_to_one() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(PendingFirstTokenForward::default());
        let mut actor = test_actor(
            rx,
            forward.clone(),
            8,
            false,
            3,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let mut receivers = Vec::new();
        for idx in 0..6 {
            let (response_tx, response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
            queue_test_request(&mut actor, request(100 + idx as TokenId, 2), response_tx);
            receivers.push(response_rx);
        }

        actor.admit_waiting();
        assert_eq!(actor.active.len(), 3);
        assert_eq!(actor.waiting.len(), 3);
        assert_eq!(actor.snapshot.total_prefill_admission_cycles, 1);
        assert_eq!(actor.snapshot.total_prefill_tokens, 6);
        assert_eq!(actor.snapshot.total_decode_tokens, 3);
        assert!(forward.calls.lock().unwrap().is_empty());

        settle_active_deliveries(&mut actor);
        actor.admit_waiting();
        assert_eq!(actor.active.len(), 4);
        assert_eq!(actor.waiting.len(), 2);
        assert_eq!(actor.snapshot.total_prefill_admission_cycles, 2);
        assert_eq!(actor.snapshot.total_prefill_tokens, 8);
        assert_eq!(actor.snapshot.total_decode_tokens, 4);
        assert!(forward.calls.lock().unwrap().is_empty());

        for (idx, rx) in receivers.iter_mut().take(4).enumerate() {
            assert_token_event(rx.blocking_recv(), 110 + idx as TokenId);
        }
    }

    #[test]
    fn admission_emits_prefill_first_tokens_without_model_decode() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(PendingFirstTokenForward::default());
        let mut actor = test_actor(
            rx,
            forward.clone(),
            8,
            false,
            3,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let mut receivers = Vec::new();
        for idx in 0..4 {
            let (response_tx, response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
            queue_test_request(&mut actor, request(100 + idx as TokenId, 1), response_tx);
            receivers.push(response_rx);
        }

        actor.admit_waiting();

        assert_eq!(actor.active.len(), 0);
        assert_eq!(actor.waiting.len(), 1);
        assert_eq!(actor.snapshot.total_prefill_admission_cycles, 1);
        assert_eq!(actor.snapshot.total_prefill_tokens, 6);
        assert_eq!(actor.snapshot.total_decode_tokens, 3);
        assert_eq!(actor.snapshot.total_decode_forwards, 0);
        assert_eq!(actor.snapshot.total_batched_decode_forwards, 0);
        assert_eq!(actor.snapshot.total_decode_rows, 0);
        assert!(forward.calls.lock().unwrap().is_empty());

        for (idx, rx) in receivers.iter_mut().take(3).enumerate() {
            let expected = 110 + idx as TokenId;
            assert_token_event(rx.blocking_recv(), expected);
            assert!(matches!(
                rx.blocking_recv(),
                Some(EngineEvent::Done {
                    output: BatchedGenerationOutput {
                        completion_tokens: 1,
                        token_ids,
                        finish_reason: FinishReason::MaxTokens,
                        ..
                    }
                }) if token_ids == vec![expected]
            ));
        }
        assert!(receivers[3].try_recv().is_err());
    }

    #[tokio::test]
    async fn enqueue_batches_forward_shape_and_routes_responses() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward.clone(), 8);

        let mut rx1 = handle.enqueue(request(101, 1)).await.unwrap();
        let mut rx2 = handle.enqueue(request(202, 1)).await.unwrap();

        assert_token_event(rx1.recv().await, 111);
        assert!(matches!(
            rx1.recv().await,
            Some(EngineEvent::Done {
                output: BatchedGenerationOutput {
                    completion_tokens: 1,
                    token_ids,
                    ..
                }
            }) if token_ids == vec![111]
        ));
        assert_token_event(rx2.recv().await, 212);
        assert!(matches!(
            rx2.recv().await,
            Some(EngineEvent::Done {
                output: BatchedGenerationOutput {
                    completion_tokens: 1,
                    token_ids,
                    ..
                }
            }) if token_ids == vec![212]
        ));

        let calls = forward.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![vec![101, 202]]);
        let snapshot = handle.snapshot().await.unwrap();
        assert_eq!(snapshot.last_batch_size, 2);
        assert_eq!(snapshot.max_observed_batch_size, 2);
        assert_eq!(snapshot.total_decode_forwards, 1);
        assert_eq!(snapshot.total_batched_decode_forwards, 1);
        assert_eq!(snapshot.total_decode_rows, 2);
        assert_eq!(snapshot.total_decode_tokens, 2);
        handle.stop().await.unwrap();
    }

    #[test]
    fn delivery_ack_cohort_preserves_sustained_wide_decode() {
        let forward = Arc::new(PendingFirstTokenForward {
            calls: StdMutex::new(Vec::new()),
            // Earlier rows are acknowledged while later rows are still being
            // prepared. Without the post-admission FIFO barrier this reliably
            // splits the first decode turn into a wide prefix plus one row.
            prepare_delay: Duration::from_millis(5),
        });
        let (command_tx, command_rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let (delivery_result_tx, delivery_results) = std_mpsc::channel();
        let response_delivery_policy = ResponseDeliveryPolicy::default();
        let delivery_worker = DeliveryWorker::start(
            response_delivery_policy.stream_stall_grace,
            Duration::from_millis(1),
            EngineDeliveryResultSink {
                result_tx: delivery_result_tx,
                pending_results: Vec::new(),
                engine_tx: command_tx.downgrade(),
            },
        )
        .expect("spawn test response delivery worker");
        let mut actor = BatchingEngineActor::new(
            command_rx,
            forward.clone(),
            8,
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            false,
            8,
            false,
            response_delivery_policy,
            delivery_worker,
            delivery_results,
        );

        let mut receivers = Vec::new();
        for prompt in 100..108 {
            let (response_tx, response_rx) = mpsc::channel(64);
            queue_test_request(&mut actor, request(prompt, 5), response_tx);
            receivers.push(response_rx);
        }

        let actor_thread = thread::spawn(move || actor.run());
        for (row, response_rx) in receivers.iter_mut().enumerate() {
            let mut token_count = 0;
            loop {
                match response_rx.blocking_recv() {
                    Some(EngineEvent::Token { .. }) => token_count += 1,
                    Some(EngineEvent::Done { output }) => {
                        assert_eq!(token_count, 5, "row {row}");
                        assert_eq!(output.completion_tokens, 5, "row {row}");
                        break;
                    }
                    Some(EngineEvent::Error(error)) => panic!("row {row} failed: {error}"),
                    None => panic!("row {row} closed before Done"),
                }
            }
        }

        let (stop_reply, stop_ack) = oneshot::channel();
        command_tx
            .blocking_send(EngineCommand::Stop { reply: stop_reply })
            .unwrap();
        stop_ack.blocking_recv().unwrap();
        actor_thread.join().unwrap();

        let calls = forward.calls.lock().unwrap().clone();
        let expected: Vec<Vec<TokenId>> = (1..5)
            .map(|step| (100..108).map(|token| token + step * 10).collect())
            .collect();
        assert_eq!(calls, expected);
        assert!(calls.iter().all(|batch| batch.len() == 8));
    }

    /// The O(waiting x active x prompt_len) deferral predicate (which also
    /// takes a prefix-cache lock per matching pair) must stay off the
    /// per-decode-step hot path: decode steps evaluate it zero times, one
    /// admission scan evaluates it exactly once per waiting row, and the
    /// snapshot observer path still reports a fresh gauge.
    #[test]
    fn deferral_predicate_stays_off_the_decode_hot_path() {
        let forward = Arc::new(MockForward {
            reusable_prefixes: true,
            ..MockForward::default()
        });
        let (_cmd_tx, cmd_rx) = mpsc::channel(8);
        let mut actor = test_actor(
            cmd_rx,
            forward.clone(),
            8,
            true,
            8,
            false,
            ResponseDeliveryPolicy::default(),
        );

        // Root row becomes active; keep its receiver alive so decode steps
        // can deliver tokens.
        let (root_tx, _root_rx) = mpsc::channel(64);
        actor.handle_command(EngineCommand::Enqueue {
            req: request_with_tokens(vec![1, 2], 32),
            response_tx: root_tx,
        });
        actor.admit_waiting();
        assert_eq!(actor.active.len(), 1);

        // Ten strict descendants stay deferred while the root is active.
        let waiting = 10usize;
        let mut keep_rx = Vec::new();
        for _ in 0..waiting {
            let (tx, rx) = mpsc::channel(64);
            actor.handle_command(EngineCommand::Enqueue {
                req: request_with_tokens(vec![1, 2, 3], 1),
                response_tx: tx,
            });
            keep_rx.push(rx);
        }

        let probes = |f: &MockForward| {
            f.prefix_probe_calls
                .load(std::sync::atomic::Ordering::Relaxed)
        };
        let before_decode = probes(&forward);
        for _ in 0..10 {
            actor.run_decode_batch();
        }
        assert_eq!(
            probes(&forward) - before_decode,
            0,
            "decode steps must not evaluate the deferral predicate"
        );

        let before_admit = probes(&forward);
        actor.admit_waiting();
        assert_eq!(
            probes(&forward) - before_admit,
            waiting,
            "one admission scan evaluates each waiting row exactly once"
        );

        actor.refresh_snapshot();
        actor.refresh_deferral_gauge();
        assert_eq!(actor.snapshot.prefix_deferred_waiting, waiting);
        assert_eq!(actor.snapshot.queue_depth, waiting);
    }

    #[tokio::test]
    async fn prefix_aware_admission_defers_strict_descendants_but_admits_independent_rows() {
        let forward = Arc::new(MockForward {
            reusable_prefixes: true,
            ..MockForward::default()
        });
        let handle = BatchingEngineHandle::start_with_policy(
            forward.clone(),
            8,
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            true,
            8,
            false,
            ResponseDeliveryPolicy::default(),
        );

        let mut prefix_rx = handle
            .enqueue(request_with_tokens(vec![1, 2], 1))
            .await
            .unwrap();
        let mut descendant_rx = handle
            .enqueue(request_with_tokens(vec![1, 2, 3], 1))
            .await
            .unwrap();
        let mut independent_rx = handle
            .enqueue(request_with_tokens(vec![9, 9], 1))
            .await
            .unwrap();

        assert_token_event(prefix_rx.recv().await, 12);
        assert!(matches!(
            prefix_rx.recv().await,
            Some(EngineEvent::Done { .. })
        ));
        assert_token_event(independent_rx.recv().await, 19);
        assert!(matches!(
            independent_rx.recv().await,
            Some(EngineEvent::Done { .. })
        ));
        assert_token_event(descendant_rx.recv().await, 13);
        assert!(matches!(
            descendant_rx.recv().await,
            Some(EngineEvent::Done { .. })
        ));

        let calls = forward.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![vec![2, 9], vec![3]]);
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn token_budget_fairly_mixes_short_decode_1k_and_16k_prefill() {
        const SHORT_KEY: TokenId = 10;
        const MEDIUM_KEY: TokenId = 1_000;
        const LONG_KEY: TokenId = 16_000;
        const TOKEN_BUDGET: usize = 32;
        const PREFILL_TOKEN_BUDGET: usize = 7;

        let forward = Arc::new(SyntheticPrefillForward::default());
        let budget = BatchTokenBudget::new(TOKEN_BUDGET, ConfigValueSource::ConfigFile).unwrap();
        let handle = BatchingEngineHandle::start_with_policy(
            forward.clone(),
            3,
            budget,
            PrefillTokenBudget::new(PREFILL_TOKEN_BUDGET, ConfigValueSource::ConfigFile).unwrap(),
            PrefillLayerBudget::default(),
            false,
            3,
            true,
            ResponseDeliveryPolicy::default(),
        );

        let short = request_with_tokens(vec![SHORT_KEY], 8);
        let medium = request_with_tokens(vec![MEDIUM_KEY; 1_024], 1);
        let long = request_with_tokens(vec![LONG_KEY; 16_384], 1);
        let long_id = long.request_id;
        let (short_rx, medium_rx, long_rx) = tokio::join!(
            handle.enqueue(short),
            handle.enqueue(medium),
            handle.enqueue(long)
        );
        let mut short_rx = short_rx.unwrap();
        let mut medium_rx = medium_rx.unwrap();
        let mut long_rx = long_rx.unwrap();

        let short_tokens = tokio::time::timeout(Duration::from_secs(3), async {
            let mut tokens = Vec::new();
            loop {
                match short_rx.recv().await {
                    Some(EngineEvent::Token { token, .. }) => tokens.push(token),
                    Some(EngineEvent::Done { output }) => {
                        assert_eq!(output.finish_reason, FinishReason::MaxTokens);
                        break tokens;
                    }
                    Some(EngineEvent::Error(error)) => panic!("short decode failed: {error}"),
                    None => panic!("short decode response closed without a terminal event"),
                }
            }
        })
        .await
        .expect("short decode was starved by prefill");
        assert_eq!(short_tokens.len(), 8);

        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                match medium_rx.recv().await {
                    Some(EngineEvent::Token { .. }) => {}
                    Some(EngineEvent::Done { output }) => {
                        assert_eq!(output.finish_reason, FinishReason::MaxTokens);
                        break;
                    }
                    Some(EngineEvent::Error(error)) => panic!("1K prefill failed: {error}"),
                    None => panic!("1K prefill response closed without a terminal event"),
                }
            }
        })
        .await
        .expect("1K prefill was starved by 16K prefill");

        let in_flight = handle.snapshot().await.unwrap();
        assert_eq!(in_flight.active_prefill, 1);
        assert_eq!(in_flight.max_batch_tokens, TOKEN_BUDGET);
        assert_eq!(
            in_flight.max_batch_tokens_source,
            ConfigValueSource::ConfigFile
        );
        assert_eq!(in_flight.max_prefill_tokens_per_cycle, PREFILL_TOKEN_BUDGET);
        assert_eq!(
            in_flight.max_prefill_tokens_per_cycle_source,
            ConfigValueSource::ConfigFile
        );
        assert!(in_flight.last_prefill_tokens <= PREFILL_TOKEN_BUDGET);
        assert!(in_flight.total_prefill_forwards > 0);

        handle.cancel(long_id).await.unwrap();
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                match long_rx.recv().await {
                    Some(EngineEvent::Error(error)) => {
                        assert!(error.contains("cancelled"));
                        break;
                    }
                    Some(EngineEvent::Token { .. }) => {}
                    Some(EngineEvent::Done { .. }) => {
                        panic!("16K prefill completed before its cancellation barrier")
                    }
                    None => panic!("16K prefill response closed without cancellation"),
                }
            }
        })
        .await
        .expect("16K prefill cancellation did not settle");

        let settled = handle.snapshot().await.unwrap();
        assert_eq!(settled.active_prefill, 0);
        assert_eq!(settled.active_decode, 0);
        handle.stop().await.unwrap();

        let events = forward.events.lock().unwrap().clone();
        let prefills: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                SchedulingEvent::Prefill {
                    key,
                    tokens,
                    remaining,
                } => Some((*key, *tokens, *remaining)),
                _ => None,
            })
            .collect();
        assert!(prefills.len() > 2);
        assert_eq!(prefills[0].0, MEDIUM_KEY);
        assert_eq!(prefills[1].0, LONG_KEY);
        assert!(
            prefills
                .iter()
                .all(|(_, tokens, _)| *tokens <= PREFILL_TOKEN_BUDGET)
        );
        assert_eq!(
            prefills
                .iter()
                .filter(|(key, _, _)| *key == MEDIUM_KEY)
                .map(|(_, tokens, _)| *tokens)
                .sum::<usize>(),
            1_024
        );
        let long_progress: usize = prefills
            .iter()
            .filter(|(key, _, _)| *key == LONG_KEY)
            .map(|(_, tokens, _)| *tokens)
            .sum();
        assert!(long_progress > 0 && long_progress < 16_384);
        assert!(events.contains(&SchedulingEvent::Discard(LONG_KEY)));

        let short_decode_indices: Vec<_> = events
            .iter()
            .enumerate()
            .filter_map(|(idx, event)| match event {
                SchedulingEvent::Decode(keys) if keys.iter().any(|key| *key < MEDIUM_KEY) => {
                    Some(idx)
                }
                _ => None,
            })
            .collect();
        assert_eq!(short_decode_indices.len(), 8);
        let interleaved =
            &events[*short_decode_indices.first().unwrap()..=*short_decode_indices.last().unwrap()];
        assert!(interleaved.iter().any(|event| {
            matches!(
                event,
                SchedulingEvent::Prefill {
                    key: MEDIUM_KEY,
                    ..
                }
            )
        }));
        assert!(
            interleaved
                .iter()
                .any(|event| { matches!(event, SchedulingEvent::Prefill { key: LONG_KEY, .. }) })
        );

        for window in events.windows(2) {
            if let [
                SchedulingEvent::Decode(keys),
                SchedulingEvent::Prefill { tokens, .. },
            ] = window
                && keys.iter().any(|key| *key < MEDIUM_KEY)
            {
                assert!(
                    *tokens <= TOKEN_BUDGET - keys.len(),
                    "prefill quantum did not leave room for its decode cohort: {window:?}"
                );
            }
        }
    }

    #[tokio::test]
    async fn layer_budget_yields_retained_prefill_to_ready_decode() {
        const SHORT_KEY: TokenId = 10;
        const LONG_KEY: TokenId = 10_000;
        const LAYER_BUDGET: usize = 2;

        let forward = Arc::new(SyntheticPrefillForward {
            layers_per_chunk: 8,
            layer_delay: Duration::from_millis(2),
            ..SyntheticPrefillForward::default()
        });
        let handle = BatchingEngineHandle::start_with_policy(
            forward.clone(),
            2,
            BatchTokenBudget::new(8, ConfigValueSource::ConfigFile).unwrap(),
            PrefillTokenBudget::new(4, ConfigValueSource::ConfigFile).unwrap(),
            PrefillLayerBudget::new(LAYER_BUDGET, ConfigValueSource::ConfigFile).unwrap(),
            false,
            2,
            true,
            ResponseDeliveryPolicy::default(),
        );

        let mut long_rx = handle
            .enqueue(request_with_tokens(vec![LONG_KEY; 8], 1))
            .await
            .unwrap();
        let mut short_rx = handle
            .enqueue(request_with_tokens(vec![SHORT_KEY], 6))
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(3), async {
            let drain_long = async {
                loop {
                    match long_rx.recv().await {
                        Some(EngineEvent::Done { .. }) => break,
                        Some(EngineEvent::Error(error)) => {
                            panic!("long request failed: {error}")
                        }
                        Some(EngineEvent::Token { .. }) => {}
                        None => panic!("long response closed without a terminal event"),
                    }
                }
            };
            let drain_short = async {
                loop {
                    match short_rx.recv().await {
                        Some(EngineEvent::Done { .. }) => break,
                        Some(EngineEvent::Error(error)) => {
                            panic!("short request failed: {error}")
                        }
                        Some(EngineEvent::Token { .. }) => {}
                        None => panic!("short response closed without a terminal event"),
                    }
                }
            };
            tokio::join!(drain_long, drain_short);
        })
        .await
        .expect("layer-bounded prefill or peer decode stalled");

        let snapshot = handle.snapshot().await.unwrap();
        assert_eq!(snapshot.total_errors, 0);
        assert_eq!(snapshot.max_prefill_layers_per_cycle, LAYER_BUDGET);
        assert_eq!(
            snapshot.max_prefill_layers_per_cycle_source,
            ConfigValueSource::ConfigFile
        );
        assert!(snapshot.last_prefill_layers <= LAYER_BUDGET);
        assert!(snapshot.total_prefill_layers >= 8);
        assert!(snapshot.total_prefill_layer_yields >= 3);
        handle.stop().await.unwrap();

        let events = forward.events.lock().unwrap().clone();
        let yielded_then_decoded = events.iter().enumerate().any(|(idx, event)| {
            let SchedulingEvent::PrefillLayers { remaining, .. } = event else {
                return false;
            };
            if *remaining == 0 {
                return false;
            }
            let next_layer = events[idx + 1..]
                .iter()
                .position(|later| matches!(later, SchedulingEvent::PrefillLayers { .. }))
                .map_or(events.len(), |offset| idx + 1 + offset);
            events[idx + 1..next_layer].iter().any(|later| {
                matches!(
                    later,
                    SchedulingEvent::Decode(keys)
                        if keys.iter().any(|key| *key < LONG_KEY)
                )
            })
        });
        assert!(
            yielded_then_decoded,
            "ready decode did not run between retained prefill layer groups: {events:?}"
        );
    }

    #[test]
    fn retained_prefill_waits_for_its_original_token_width() {
        const KEY: TokenId = 10_000;

        let forward = Arc::new(SyntheticPrefillForward {
            layers_per_chunk: 8,
            ..SyntheticPrefillForward::default()
        });
        let (_command_tx, command_rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let mut actor = test_actor(
            command_rx,
            forward.clone(),
            1,
            false,
            1,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let req = request_with_tokens(vec![KEY; 128], 1);
        let RequestPreparation::Prefilling { slot, .. } = forward
            .prepare_request_chunked(&req, 64)
            .expect("initialize synthetic retained prefill")
        else {
            panic!("long synthetic prompt unexpectedly became ready")
        };
        let (response_tx, _response_rx) = mpsc::channel(8);
        push_test_active(&mut actor, req, response_tx, slot);

        assert!(actor.run_prefill_budget(64));
        assert_eq!(
            forward.inflight_prefill_token_width(&actor.active[0].slot),
            Some(64)
        );
        let events_before_deferral = forward.events.lock().unwrap().len();

        assert!(
            actor.run_prefill_budget(32),
            "a token-width deferral must keep the actor schedulable for its next cycle"
        );
        assert_eq!(forward.events.lock().unwrap().len(), events_before_deferral);
        assert_eq!(actor.snapshot.total_prefill_token_budget_deferrals, 1);
        assert_eq!(actor.snapshot.total_errors, 0);
        assert_eq!(
            forward.inflight_prefill_token_width(&actor.active[0].slot),
            Some(64)
        );

        assert!(actor.run_prefill_budget(64));
        assert_eq!(actor.snapshot.total_prefill_tokens, 64);
        assert_eq!(actor.snapshot.total_prefill_layers, 8);
        assert_eq!(actor.snapshot.total_prefill_forwards, 2);
        assert_eq!(actor.snapshot.total_errors, 0);
        assert!(matches!(
            forward.events.lock().unwrap().last(),
            Some(SchedulingEvent::Prefill {
                key: KEY,
                tokens: 64,
                remaining: 64,
            })
        ));

        actor.fail_all("test complete");
    }

    #[tokio::test]
    async fn eos_finish_counts_terminal_token_for_usage() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward.clone(), 8);

        let mut rx = handle.enqueue(request(0, 1)).await.unwrap();

        assert!(matches!(
            rx.recv().await,
            Some(EngineEvent::Done {
                output: BatchedGenerationOutput {
                    completion_tokens: 1,
                    token_ids,
                    finish_reason: FinishReason::Eos,
                    ..
                }
            }) if token_ids.is_empty()
        ));

        let calls = forward.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![vec![0]]);
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn routes_multiple_decode_steps_to_original_receivers() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward.clone(), 8);

        let mut rx = handle.enqueue(request(7, 2)).await.unwrap();

        assert_token_event(rx.recv().await, 17);
        assert_token_event(rx.recv().await, 27);
        assert!(matches!(
            rx.recv().await,
            Some(EngineEvent::Done {
                output: BatchedGenerationOutput {
                    completion_tokens: 2,
                    token_ids,
                    finish_reason: FinishReason::MaxTokens,
                    ..
                }
            }) if token_ids == vec![17, 27]
        ));

        let calls = forward.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![vec![7], vec![17]]);
        handle.stop().await.unwrap();
    }
}

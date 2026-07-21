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
use kiln_core::sampling::{SamplingParams, ThinkingBudgetTokenSource};
use kiln_core::token::TokenId;
use kiln_model::{
    BackendHealthHandle, CancelHandle, DecodeExecutionPolicy, FinishReason, GenerationOutput,
    ModelRunner, PagedBatchedDecodeState, PagedBatchedPrefillStart, PagedBatchedPrefillState,
    PagedKvCacheKt, PagedPrefixRegistration, PagedPrefixReuse,
};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::config::{
    ACTOR_CYCLE_IDLE_COMMAND_POLL_MS, ActorCycleIdleDiagnostics, BatchTokenBudget,
    BatchingActorAdmissionConfig, BatchingBackendPolicy, BatchingConfig, ConfigValueSource,
    DEFAULT_ROWWISE_DECODE, DecodeBatchEffectiveSource, DecodeRuntimeConfig,
    DeterministicInference, MaxDecodeBatch, MaxDecodeBatchDiagnostics, PrefillLayerBudget,
    PrefillTokenBudget, StreamStallGrace,
};
use crate::latency_observability::{BackendPhaseDurations, EngineTokenTiming, TokenPhaseDurations};
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
/// Latency-oriented actors may prepare a small number of interactive prompts
/// without widening the backend's decode cohort. Ordinary FIFO slots and this
/// staging lane are accounted independently so staged arrivals cannot consume
/// or indefinitely postpone ordinary admission capacity.
const MAX_PREFILL_STAGING_SLOTS: usize = 4;
/// A full staging lane may consume this many priority turns before the actor
/// forces one global round-robin prefill dispatch. Decode still runs before
/// every prefill dispatch in the actor loop.
const PREFILL_STAGING_ROUND_ROBIN_INTERVAL: usize = 5;
/// Force one round-robin dispatch after two opportunities to accelerate a
/// short prompt tail. This bounds long-prompt slowdown under continuous short
/// arrivals without making interactive prompts wait a full active-set rotation
/// for every layer group.
const SHORT_PREFILL_ROUND_ROBIN_INTERVAL: usize = 3;
const SHORT_PREFILL_PRIORITY_MAX_CHUNKS: usize = 4;

/// Actor work above this wall time is material to the qualification stall
/// gate and gets one bounded structured event after the phase completes.
const SLOW_ACTOR_PHASE_THRESHOLD: Duration = Duration::from_millis(100);

/// Fair worker retry cadence while a response lane is inside its grace window.
const RESPONSE_DELIVERY_POLL_CADENCE: Duration = Duration::from_millis(10);

fn observe_profiled_decode_phases(
    phases: &mut BackendPhaseDurations,
    sampling: Option<Duration>,
    readback: Option<Duration>,
    graph_capture: Option<Duration>,
    graph_replay: Option<Duration>,
) {
    if let Some(duration) = sampling {
        phases.observe_sampling(duration);
    }
    if let Some(duration) = readback {
        phases.observe_readback(duration);
    }
    if let Some(duration) = graph_capture {
        phases.observe_graph_capture(duration);
    }
    if let Some(duration) = graph_replay {
        phases.observe_graph_replay(duration);
    }
}

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

/// Replace the physical KV pool and publish its logical block space as one
/// transaction. `physical_resize` must leave the physical pool unchanged on
/// error; this function provides the matching guarantee for `block_manager`.
fn resize_block_manager_transaction<F>(
    block_manager: &mut BlockManager,
    current_physical_blocks: usize,
    target_blocks: usize,
    physical_resize: F,
) -> Result<usize>
where
    F: FnOnce(usize) -> Result<()>,
{
    anyhow::ensure!(
        block_manager.num_blocks() == current_physical_blocks,
        "KV capacity mismatch before resize: block manager has {} blocks, physical pool has {current_physical_blocks}",
        block_manager.num_blocks()
    );
    if target_blocks == current_physical_blocks {
        return Ok(current_physical_blocks);
    }

    let mut staged = block_manager.clone();
    let achieved = if target_blocks < current_physical_blocks {
        staged.set_target_usable(target_blocks);
        let achieved = target_blocks.max(staged.physical_floor());
        staged
            .physical_truncate(achieved)
            .map_err(|error| anyhow::anyhow!("kv shrink truncate to {achieved}: {error}"))?;
        achieved
    } else {
        staged.physical_grow(target_blocks);
        staged.set_target_usable(target_blocks);
        target_blocks
    };

    physical_resize(achieved)?;
    *block_manager = staged;
    Ok(achieved)
}

/// Resolve the actor's `max_decode_batch` from the reproducibility envelope,
/// typed startup configuration, or active backend policy. Deterministic inference stays
/// single-row even when an operator also configured a wider batch: changing
/// the request cohort can otherwise select a different BF16 GEMM shape and
/// change a greedy token at a close logit boundary. This remains the effective
/// concurrent-decode width reported through health and metrics even when a
/// latency-oriented actor exposes additional bounded prefill staging slots.
pub fn resolve_decode_runtime_config(
    deterministic: DeterministicInference,
    configured: MaxDecodeBatch,
    policy: Option<DecodeExecutionPolicy>,
    max_batch_tokens: BatchTokenBudget,
) -> DecodeRuntimeConfig {
    let backend_policy = policy.map_or(DEFAULT_MAX_DECODE_BATCH, |policy| policy.max_decode_batch);
    let selected = configured.limit().unwrap_or(backend_policy);
    let selected_source = match (configured.limit(), configured.source()) {
        (Some(_), ConfigValueSource::ConfigFile) => DecodeBatchEffectiveSource::ConfigFile,
        (Some(_), ConfigValueSource::Environment) => DecodeBatchEffectiveSource::Environment,
        (Some(_), ConfigValueSource::CommandLine) => DecodeBatchEffectiveSource::CommandLine,
        _ => DecodeBatchEffectiveSource::BackendPolicy,
    };
    let (selected, selected_source) = if deterministic.enabled() {
        (1, DecodeBatchEffectiveSource::Deterministic)
    } else {
        (selected, selected_source)
    };
    let (effective, effective_source) = if max_batch_tokens.tokens() < selected {
        (
            max_batch_tokens.tokens(),
            DecodeBatchEffectiveSource::MaxBatchTokens,
        )
    } else {
        (selected, selected_source)
    };

    DecodeRuntimeConfig {
        deterministic: deterministic.diagnostics(),
        max_decode_batch: MaxDecodeBatchDiagnostics {
            configured: configured.limit(),
            configured_source: configured.source(),
            backend_policy,
            effective,
            effective_source,
        },
    }
}

#[derive(Debug, Clone)]
pub struct EngineRequest {
    pub request_id: Uuid,
    pub prompt_tokens: Vec<TokenId>,
    pub sampling: SamplingParams,
    pub adapter: Option<LoadedAdapterIdentity>,
    /// Opt in to exact per-action behavior log-probabilities for rollout
    /// provenance. Ordinary serving requests leave this disabled.
    pub capture_behavior_logprobs: bool,
    pub cancel: CancelHandle,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EngineSampledToken {
    pub token_id: TokenId,
    pub behavior_logprob: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodeStepOutput {
    pub tokens: Vec<EngineSampledToken>,
    pub backend_phases: BackendPhaseDurations,
}

impl EngineSampledToken {
    fn untraced(token_id: TokenId) -> Self {
        Self {
            token_id,
            behavior_logprob: None,
        }
    }

    fn traced(token_id: TokenId, behavior_logprob: f32) -> Self {
        Self {
            token_id,
            behavior_logprob: Some(behavior_logprob),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineActionTokenSource {
    Sampled,
    Forced,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineActionToken {
    /// Index within the generated suffix, including a terminal EOS decision,
    /// before the API adds the prompt boundary to produce a full-sequence
    /// provenance index.
    pub generated_index: usize,
    pub token_id: TokenId,
    pub source: EngineActionTokenSource,
    pub behavior_logprob: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    Token {
        token: TokenId,
        timing: EngineTokenTiming,
    },
    Done {
        output: BatchedGenerationOutput,
    },
    Error(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchedGenerationOutput {
    pub text: String,
    pub token_ids: Vec<TokenId>,
    pub finish_reason: FinishReason,
    pub completion_tokens: usize,
    pub action_tokens: Option<Vec<EngineActionToken>>,
    pub prefill_duration: Duration,
    pub decode_duration: Duration,
    /// Time from API enqueue through the actor's waiting queue to slot admission.
    pub actor_queue_duration: Duration,
    /// Time spent preparing the request for its active slot.
    pub actor_admission_duration: Duration,
    /// Wall time from slot admission until the first sampled token became ready.
    pub actor_prefill_wall_duration: Option<Duration>,
    /// Whether this request completed prompt work through the native multi-row
    /// resident-prefill route.
    pub resident_prefill_used: bool,
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
    /// Configured safe-boundary delay after actor cycles that advanced model work.
    pub actor_cycle_idle_ms: u64,
    /// Startup source that selected `actor_cycle_idle_ms`.
    pub actor_cycle_idle_source: ConfigValueSource,
    /// True while the actor is cooperatively polling inside the configured delay.
    pub actor_cycle_idle_active: bool,
    /// Cooperative waits entered since actor startup.
    pub actor_cycle_idle_count: u64,
    /// Cumulative observed cooperative wait wall time.
    pub total_actor_cycle_idle_ms: f64,
    /// Largest observed cooperative wait wall time.
    pub max_actor_cycle_idle_ms: f64,
    pub accepting: bool,
    pub queue_depth: usize,
    pub active_decode: usize,
    pub active_prefill: usize,
    /// Whether cross-request prompt/KV/GDN prefix reuse is admitted for the
    /// active backend.
    pub prefix_cache_enabled: bool,
    /// Whether the production forward has admitted native resident token
    /// prefill as a correctness-qualified route.
    pub resident_prefill_enabled: bool,
    /// Prefill rows whose newest KV positions are owned only by the resident
    /// Vulkan route and therefore cannot fall back to generic prefill.
    pub active_resident_prefill: usize,
    pub max_batch_tokens: usize,
    pub max_batch_tokens_source: ConfigValueSource,
    pub max_prefill_tokens_per_cycle: usize,
    pub max_prefill_tokens_per_cycle_source: ConfigValueSource,
    pub max_prefill_layers_per_cycle: usize,
    pub max_prefill_layers_per_cycle_source: ConfigValueSource,
    pub max_prefill_admission_quantum: usize,
    /// Bounded short-prefill slots beyond the ordinary decode-width slots.
    pub max_prefill_staging_slots: usize,
    /// Total ordinary plus short-prefill staging capacity.
    pub max_active_requests: usize,
    /// Maximum staged-priority turns before a mandatory global prefill turn.
    pub max_prefill_staging_priority_burst: usize,
    /// Effective concurrent decode-row ceiling after the combined token budget
    /// has constrained the configured/backend-selected width.
    pub max_decode_batch: usize,
    pub active_staged_requests: usize,
    pub max_observed_active_requests: usize,
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
    pub total_resident_prefill_attempts: u64,
    pub total_resident_prefill_forwards: u64,
    pub total_resident_prefill_initial_declines: u64,
    pub total_resident_prefill_route_failures: u64,
    pub total_resident_prefill_rows: u64,
    pub total_resident_prefill_completed_rows: u64,
    pub last_resident_prefill_batch_size: usize,
    pub max_resident_prefill_batch_size: usize,
    pub total_prefill_layers: u64,
    pub total_prefill_layer_yields: u64,
    pub total_short_prefill_priority_forwards: u64,
    pub total_prefill_staging_priority_forwards: u64,
    pub total_prefill_staging_admissions: u64,
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
        tokens_scheduled: usize,
        tokens_processed: usize,
        layers_processed: usize,
    },
    Ready {
        slot: DecodeSlot,
        tokens_scheduled: usize,
        tokens_processed: usize,
        layers_processed: usize,
    },
}

pub struct PrefillBatchProgress {
    pub tokens_scheduled: usize,
    pub tokens_processed: usize,
    pub layers_processed: usize,
    pub ready: bool,
}

fn collect_ready_decode_indices(
    slots: &mut [&mut DecodeSlot],
    sampling: &[SamplingParams],
    output: &mut [EngineSampledToken],
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
                output[idx] = match state.next_token_logprob {
                    Some(logprob) => EngineSampledToken::traced(state.next_token, logprob),
                    None => EngineSampledToken::untraced(state.next_token),
                };
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum KvResizeReason {
    AutomaticMemoryPolicy,
    ForcedConfiguration,
    TrainingMemoryPreparation,
    Maintenance,
}

impl KvResizeReason {
    const fn as_str(self) -> &'static str {
        match self {
            Self::AutomaticMemoryPolicy => "automatic_memory_policy",
            Self::ForcedConfiguration => "forced_configuration",
            Self::TrainingMemoryPreparation => "training_memory_preparation",
            Self::Maintenance => "maintenance",
        }
    }
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
            tokens_scheduled: req.prompt_tokens.len(),
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
    /// Remaining prompt tokens, including an in-flight retained chunk. Used
    /// only for the bounded short-tail service opportunity; `None` keeps the
    /// row on the ordinary round-robin path.
    fn remaining_prefill_tokens(&self, _slot: &DecodeSlot) -> Option<usize> {
        None
    }
    fn prefix_cache_enabled(&self) -> bool {
        false
    }
    /// Whether the resident token-prefill optimization is admitted for this
    /// forward. The actor does not probe candidates or mutate counters while
    /// this capability is withdrawn.
    fn resident_prefill_enabled(&self) -> bool {
        false
    }
    /// Whether this row can enter the resident one-token Vulkan prefill batch.
    /// The classification must be mutation-free; the batch method revalidates
    /// every condition while holding model execution ownership.
    fn resident_prefill_batch_candidate(
        &self,
        _slot: &DecodeSlot,
        _sampling: &SamplingParams,
    ) -> bool {
        false
    }
    /// A row that has already written newer positions only to the resident KV
    /// cache must remain on that route even when it is the final row left.
    fn resident_prefill_batch_required(&self, _slot: &DecodeSlot) -> bool {
        false
    }
    fn advance_resident_prefill_batch(
        &self,
        _slots: &mut [&mut DecodeSlot],
        _sampling: &[SamplingParams],
        _cancels: &[CancelHandle],
    ) -> Result<Option<Vec<PrefillBatchProgress>>> {
        Ok(None)
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
    /// Decode with optional behavior-policy metadata. Test doubles and custom
    /// forwards inherit the token-only adapter; production overrides this to
    /// honor trace-mode rows.
    fn forward_decode_with_metadata(
        &self,
        slots: &mut [&mut DecodeSlot],
        sampling: &[SamplingParams],
    ) -> Result<Vec<EngineSampledToken>> {
        self.forward_decode(slots, sampling).map(|tokens| {
            tokens
                .into_iter()
                .map(EngineSampledToken::untraced)
                .collect()
        })
    }
    fn forward_decode_with_phases(
        &self,
        slots: &mut [&mut DecodeSlot],
        sampling: &[SamplingParams],
    ) -> Result<DecodeStepOutput> {
        self.forward_decode_with_metadata(slots, sampling)
            .map(|tokens| DecodeStepOutput {
                tokens,
                backend_phases: BackendPhaseDurations::default(),
            })
    }
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

    #[doc(hidden)]
    fn resize_kv_with_context(
        &self,
        target_blocks: usize,
        _reason: KvResizeReason,
        _barrier_wait: Duration,
    ) -> Result<usize> {
        self.resize_kv(target_blocks)
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
    // Resolved once from the immutable serving profile and selected backend.
    // The actor publishes this same value through health/debug/metrics.
    resident_prefill_enabled: bool,
    // When set, multi-row decode steps are dispatched as a loop of single-row
    // forwards instead of one batched forward. Startup configuration resolves
    // this once; the constructor default keeps library callers on true batched
    // decode unless they use the explicit builder.
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
            resident_prefill_enabled: false,
            rowwise_decode: DEFAULT_ROWWISE_DECODE,
        }
    }

    pub fn with_resident_prefill_enabled(mut self, enabled: bool) -> Self {
        self.resident_prefill_enabled = enabled;
        self
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

    fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache
            .lock()
            .map(|cache| cache.is_enabled())
            .unwrap_or(false)
    }

    fn resident_prefill_enabled(&self) -> bool {
        self.resident_prefill_enabled
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

    fn remaining_prefill_tokens(&self, slot: &DecodeSlot) -> Option<usize> {
        let DecodeSlot::RealPrefill {
            state: Some(state), ..
        } = slot
        else {
            return None;
        };
        Some(state.remaining_tokens())
    }

    fn resident_prefill_batch_candidate(
        &self,
        slot: &DecodeSlot,
        sampling: &SamplingParams,
    ) -> bool {
        self.resident_prefill_enabled()
            && matches!(
                slot,
                DecodeSlot::RealPrefill {
                    state: Some(state),
                    ..
                } if state.resident_token_prefill_candidate(sampling)
            )
    }

    fn resident_prefill_batch_required(&self, slot: &DecodeSlot) -> bool {
        matches!(
            slot,
            DecodeSlot::RealPrefill {
                state: Some(state),
                ..
            } if state.resident_token_prefill_started()
        )
    }

    fn advance_resident_prefill_batch(
        &self,
        slots: &mut [&mut DecodeSlot],
        sampling: &[SamplingParams],
        cancels: &[CancelHandle],
    ) -> Result<Option<Vec<PrefillBatchProgress>>> {
        anyhow::ensure!(
            slots.len() == sampling.len() && slots.len() == cancels.len(),
            "resident prefill batch metadata length mismatch"
        );
        let mut state_refs = Vec::with_capacity(slots.len());
        for slot in slots.iter_mut() {
            let DecodeSlot::RealPrefill { state, .. } = &mut **slot else {
                anyhow::bail!("non-prefill slot sent to resident prefill batch")
            };
            state_refs.push(state);
        }

        let gpu_guard = gpu_coordination_read_guard(&self.gpu_lock);
        let runner_guard = self.runner_guard()?;
        let cancel_refs: Vec<&CancelHandle> = cancels.iter().collect();
        let result = runner_guard.advance_paged_batched_prefill_resident_token_batch(
            &mut state_refs,
            sampling,
            self.paged_cache.as_ref(),
            &cancel_refs,
        );
        drop(state_refs);
        let synchronized =
            runner_guard.synchronize_external_yield("resident batched token-prefill quantum");
        drop(runner_guard);
        drop(gpu_guard);
        let mut progress = match (result, synchronized) {
            (Ok(progress), Ok(())) => progress,
            (Err(error), Ok(())) => return Err(error),
            (Ok(_), Err(error)) => return Err(error),
            (Err(error), Err(sync_error)) => {
                return Err(anyhow::anyhow!(
                    "{error:#}; resident token-prefill synchronization also failed: {sync_error:#}"
                ));
            }
        };
        let Some(progress) = progress.as_mut() else {
            return Ok(None);
        };
        anyhow::ensure!(
            progress.len() == slots.len(),
            "resident token-prefill returned {} rows for {} slots",
            progress.len(),
            slots.len()
        );

        let mut actor_progress = Vec::with_capacity(progress.len());
        for (slot, row) in slots.iter_mut().zip(progress.drain(..)) {
            let ready = row.decode_state.is_some();
            if let Some(decode_state) = row.decode_state {
                let DecodeSlot::RealPrefill {
                    state,
                    prefix_request,
                } = &mut **slot
                else {
                    unreachable!("resident token-prefill slot changed during forward")
                };
                anyhow::ensure!(
                    state.is_none(),
                    "completed resident token-prefill retained duplicate state"
                );
                let prefix_request = prefix_request.take();
                **slot = DecodeSlot::Real {
                    state: decode_state,
                    prefix_request,
                    first_token_pending: true,
                };
            }
            actor_progress.push(PrefillBatchProgress {
                tokens_scheduled: row.tokens_scheduled,
                tokens_processed: row.tokens_processed,
                layers_processed: row.layers_processed,
                ready,
            });
        }
        Ok(Some(actor_progress))
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

        let prepared = runner_guard
            .begin_paged_batched_decode_with_prefix_cache_and_behavior_logprobs(
                &req.prompt_tokens,
                &req.sampling,
                self.block_manager.as_ref(),
                self.paged_cache.as_ref(),
                cached_prefix,
                prefix_cache_enabled,
                req.capture_behavior_logprobs,
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
                tokens_scheduled: 0,
                tokens_processed: 0,
                layers_processed: 0,
            }),
            Ok(PagedBatchedPrefillStart::Prefilling(state)) => Ok(RequestPreparation::Prefilling {
                slot: DecodeSlot::RealPrefill {
                    state: Some(state),
                    prefix_request,
                },
                tokens_scheduled: 0,
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
        if synchronized.is_ok()
            && progress.is_err()
            && let Some(prefill) = state.as_ref()
        {
            runner_guard.release_paged_batched_prefill_state(prefill);
        }
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
                    tokens_scheduled: progress.tokens_scheduled,
                    tokens_processed: progress.tokens_processed,
                    layers_processed: progress.layers_processed,
                }),
                None => Ok(RequestPreparation::Prefilling {
                    slot: DecodeSlot::RealPrefill {
                        state,
                        prefix_request,
                    },
                    tokens_scheduled: progress.tokens_scheduled,
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
        self.forward_decode_with_phases(slots, sampling)
            .map(|step| {
                step.tokens
                    .into_iter()
                    .map(|sampled| sampled.token_id)
                    .collect()
            })
    }

    fn forward_decode_with_metadata(
        &self,
        slots: &mut [&mut DecodeSlot],
        sampling: &[SamplingParams],
    ) -> Result<Vec<EngineSampledToken>> {
        self.forward_decode_with_phases(slots, sampling)
            .map(|step| step.tokens)
    }

    fn forward_decode_with_phases(
        &self,
        slots: &mut [&mut DecodeSlot],
        sampling: &[SamplingParams],
    ) -> Result<DecodeStepOutput> {
        self.grow_ready_decode_slots(slots)?;
        let mut output = vec![EngineSampledToken::untraced(0); slots.len()];
        let mut backend_phases = BackendPhaseDurations::default();
        let (decode_indices, decode_params) =
            collect_ready_decode_indices(slots, sampling, &mut output)?;

        if !decode_indices.is_empty() {
            let gpu_lock_started = Instant::now();
            let gpu_guard = gpu_coordination_read_guard(&self.gpu_lock);
            backend_phases.observe_gpu_lock_wait(gpu_lock_started.elapsed());
            let mut ordinary_rows = Vec::with_capacity(decode_indices.len());
            let mut ordinary_params = Vec::with_capacity(decode_indices.len());
            let mut ordinary_output_indices = Vec::with_capacity(decode_indices.len());
            let mut traced_rows = Vec::new();
            let mut traced_params = Vec::new();
            let mut traced_output_indices = Vec::new();
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
                        let param_idx = ordinary_rows.len() + traced_rows.len();
                        let params = decode_params[param_idx].clone();
                        if state.capture_behavior_logprobs {
                            traced_rows.push(state);
                            traced_params.push(params);
                            traced_output_indices.push(idx);
                        } else {
                            ordinary_rows.push(state);
                            ordinary_params.push(params);
                            ordinary_output_indices.push(idx);
                        }
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
                ordinary_rows.len() + traced_rows.len() == decode_params.len(),
                "decode row length {} != params length {} after row selection",
                ordinary_rows.len() + traced_rows.len(),
                decode_params.len()
            );
            let runner_guard = self.runner_guard()?;
            let decode_result = (|| -> Result<Vec<(usize, EngineSampledToken)>> {
                let mut decoded = Vec::with_capacity(decode_indices.len());
                let ordinary_tokens = if ordinary_rows.is_empty() {
                    Vec::new()
                } else if self.rowwise_decode && ordinary_rows.len() > 1 {
                    let mut tokens = Vec::with_capacity(ordinary_rows.len());
                    for (row, params) in ordinary_rows.iter_mut().zip(ordinary_params.iter()) {
                        let mut single_row: [&mut PagedBatchedDecodeState; 1] = [&mut **row];
                        let single_params = std::slice::from_ref(params);
                        let mut next = runner_guard.paged_batched_decode_step_profiled(
                            &mut single_row,
                            single_params,
                            self.paged_cache.as_ref(),
                        )?;
                        observe_profiled_decode_phases(
                            &mut backend_phases,
                            next.sampling_duration,
                            next.readback_duration,
                            next.graph_capture_duration,
                            next.graph_replay_duration,
                        );
                        anyhow::ensure!(
                            next.tokens.len() == 1,
                            "rowwise decode returned {} tokens for a 1-row step",
                            next.tokens.len()
                        );
                        tokens.push(next.tokens.remove(0));
                    }
                    tokens
                } else {
                    let step = runner_guard.paged_batched_decode_step_profiled(
                        &mut ordinary_rows,
                        &ordinary_params,
                        self.paged_cache.as_ref(),
                    )?;
                    observe_profiled_decode_phases(
                        &mut backend_phases,
                        step.sampling_duration,
                        step.readback_duration,
                        step.graph_capture_duration,
                        step.graph_replay_duration,
                    );
                    step.tokens
                };
                for ((output_idx, row), token) in ordinary_output_indices
                    .iter()
                    .copied()
                    .zip(ordinary_rows.iter_mut())
                    .zip(ordinary_tokens)
                {
                    row.next_token_logprob = None;
                    decoded.push((output_idx, EngineSampledToken::untraced(token)));
                }

                let traced_tokens = if traced_rows.is_empty() {
                    Vec::new()
                } else if self.rowwise_decode && traced_rows.len() > 1 {
                    let mut tokens = Vec::with_capacity(traced_rows.len());
                    for (row, params) in traced_rows.iter_mut().zip(traced_params.iter()) {
                        let mut single_row: [&mut PagedBatchedDecodeState; 1] = [&mut **row];
                        let single_params = std::slice::from_ref(params);
                        let mut next = runner_guard
                            .paged_batched_decode_step_with_behavior_logprobs_profiled(
                                &mut single_row,
                                single_params,
                                self.paged_cache.as_ref(),
                            )?;
                        observe_profiled_decode_phases(
                            &mut backend_phases,
                            next.sampling_duration,
                            next.readback_duration,
                            next.graph_capture_duration,
                            next.graph_replay_duration,
                        );
                        anyhow::ensure!(
                            next.tokens.len() == 1,
                            "rowwise behavior-logprob decode returned {} tokens for a 1-row step",
                            next.tokens.len()
                        );
                        tokens.push(next.tokens.remove(0));
                    }
                    tokens
                } else {
                    let step = runner_guard
                        .paged_batched_decode_step_with_behavior_logprobs_profiled(
                            &mut traced_rows,
                            &traced_params,
                            self.paged_cache.as_ref(),
                        )?;
                    observe_profiled_decode_phases(
                        &mut backend_phases,
                        step.sampling_duration,
                        step.readback_duration,
                        step.graph_capture_duration,
                        step.graph_replay_duration,
                    );
                    step.tokens
                };
                for ((output_idx, row), sampled) in traced_output_indices
                    .iter()
                    .copied()
                    .zip(traced_rows.iter_mut())
                    .zip(traced_tokens)
                {
                    row.next_token_logprob = Some(sampled.logprob);
                    decoded.push((
                        output_idx,
                        EngineSampledToken::traced(sampled.token_id, sampled.logprob),
                    ));
                }
                Ok(decoded)
            })();
            let synchronization_started = Instant::now();
            let synchronized = runner_guard.synchronize_external_yield("batched decode step");
            backend_phases.observe_synchronization(synchronization_started.elapsed());
            drop(runner_guard);
            if let Err(err) = synchronized {
                std::mem::forget(gpu_guard);
                return Err(err);
            }
            for (idx, token) in decode_result? {
                output[idx] = token;
            }
        }

        Ok(DecodeStepOutput {
            tokens: output,
            backend_phases,
        })
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
                if let Some(prefill) = state.as_ref() {
                    let (runner, _) = self.runner_guard_for_finish();
                    runner.release_paged_batched_prefill_state(prefill);
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
        self.resize_kv_with_context(target_blocks, KvResizeReason::Maintenance, Duration::ZERO)
    }

    fn resize_kv_with_context(
        &self,
        target_blocks: usize,
        reason: KvResizeReason,
        barrier_wait: Duration,
    ) -> Result<usize> {
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
        let bytes_per_block = self.paged_cache.bytes_per_block() as u64;
        let previous_bytes = (cur as u64).saturating_mul(bytes_per_block);
        let requested_bytes = (target_blocks as u64).saturating_mul(bytes_per_block);
        let started = Instant::now();
        let mut gpu_coordination_wait = Duration::ZERO;
        let mut model_lock_wait = Duration::ZERO;
        let result = (|| {
            // EXCLUSIVE GPU access for the pool swap: the write guard blocks
            // decode actors and training until the transaction commits.
            let gpu_wait_started = Instant::now();
            let gpu_guard =
                gpu_coordination_write_guard_while_healthy(&self.gpu_lock, &self.backend_health);
            gpu_coordination_wait = gpu_wait_started.elapsed();
            let _gpu = gpu_guard?;

            let model_wait_started = Instant::now();
            let runner = self.runner.write().map_err(|error| {
                anyhow::anyhow!("model runner lock poisoned during KV resize: {error}")
            });
            model_lock_wait = model_wait_started.elapsed();
            let runner = runner?;
            runner.ensure_backend_healthy()?;
            runner.invalidate_decode_graphs_for_kv_pool_change()?;
            let mut block_manager = self.block_manager_guard()?;
            resize_block_manager_transaction(&mut block_manager, cur, target_blocks, |achieved| {
                self.paged_cache.physical_resize_to(achieved, device)
            })
        })();
        let mutation_duration_ms = started.elapsed().as_secs_f64() * 1000.0;
        let barrier_wait_ms = barrier_wait.as_secs_f64() * 1000.0;
        let gpu_coordination_wait_ms = gpu_coordination_wait.as_secs_f64() * 1000.0;
        let model_lock_wait_ms = model_lock_wait.as_secs_f64() * 1000.0;
        let wait_ms = barrier_wait_ms + gpu_coordination_wait_ms + model_lock_wait_ms;
        let duration_ms = barrier_wait_ms + mutation_duration_ms;
        let direction = if target_blocks < cur {
            "shrink"
        } else {
            "grow"
        };
        match result {
            Ok(achieved) => {
                let actual_bytes = (achieved as u64).saturating_mul(bytes_per_block);
                tracing::info!(
                    event = "gpu_memory_operation",
                    operation = "resize",
                    reason = reason.as_str(),
                    outcome = "completed",
                    direction,
                    from_blocks = cur,
                    requested_blocks = target_blocks,
                    actual_blocks = achieved,
                    previous_bytes,
                    requested_bytes,
                    actual_bytes,
                    released_bytes = previous_bytes.saturating_sub(actual_bytes),
                    added_bytes = actual_bytes.saturating_sub(previous_bytes),
                    barrier_wait_ms,
                    gpu_coordination_wait_ms,
                    model_lock_wait_ms,
                    wait_ms,
                    mutation_duration_ms,
                    duration_ms,
                    "KV cache physical resize completed"
                );
                Ok(achieved)
            }
            Err(error) => {
                tracing::warn!(
                    event = "gpu_memory_operation",
                    operation = "resize",
                    reason = reason.as_str(),
                    outcome = "failed",
                    direction,
                    error = %format!("{error:#}"),
                    from_blocks = cur,
                    requested_blocks = target_blocks,
                    actual_blocks = cur,
                    previous_bytes,
                    requested_bytes,
                    actual_bytes = previous_bytes,
                    released_bytes = 0,
                    added_bytes = 0,
                    barrier_wait_ms,
                    gpu_coordination_wait_ms,
                    model_lock_wait_ms,
                    wait_ms,
                    mutation_duration_ms,
                    duration_ms,
                    "KV cache physical resize failed"
                );
                Err(error)
            }
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
        Self::start_with_options(forward, DEFAULT_MAX_DECODE_BATCH)
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
        policy: Option<DecodeExecutionPolicy>,
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
        policy: Option<DecodeExecutionPolicy>,
        max_batch_tokens: BatchTokenBudget,
        max_prefill_tokens_per_cycle: PrefillTokenBudget,
        max_prefill_layers_per_cycle: PrefillLayerBudget,
        response_delivery_policy: ResponseDeliveryPolicy,
    ) -> Self {
        let max_decode_batch = max_decode_batch.max(1);
        let batching = BatchingConfig::default().resolve(
            BatchingBackendPolicy {
                use_decode_width_prefill_admission: policy
                    .is_some_and(|policy| policy.use_decode_width_prefill_admission),
                burst_prefill_admission: policy
                    .is_some_and(|policy| policy.burst_prefill_admission),
                actor_prefill_tile_alignment_required: policy
                    .is_some_and(|policy| policy.actor_prefill_tile_alignment_required),
            },
            max_decode_batch,
        );
        Self::start_with_admission_config(
            forward,
            max_decode_batch,
            batching.actor_admission_config(),
            max_batch_tokens,
            max_prefill_tokens_per_cycle,
            max_prefill_layers_per_cycle,
            response_delivery_policy,
        )
    }

    /// Start an actor from the immutable admission policy resolved during
    /// application startup. No actor thread reads process environment.
    pub fn start_with_admission_config(
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        admission: BatchingActorAdmissionConfig,
        max_batch_tokens: BatchTokenBudget,
        max_prefill_tokens_per_cycle: PrefillTokenBudget,
        max_prefill_layers_per_cycle: PrefillLayerBudget,
        response_delivery_policy: ResponseDeliveryPolicy,
    ) -> Self {
        Self::start_with_actor_runtime_config(
            forward,
            max_decode_batch,
            admission,
            ActorCycleIdleDiagnostics::default(),
            max_batch_tokens,
            max_prefill_tokens_per_cycle,
            max_prefill_layers_per_cycle,
            response_delivery_policy,
        )
    }

    /// Start an actor from the complete immutable actor policy resolved during
    /// application startup. The compatibility constructor above keeps tests and
    /// embedders on the unpaced default.
    pub fn start_with_actor_runtime_config(
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        admission: BatchingActorAdmissionConfig,
        actor_cycle_idle: ActorCycleIdleDiagnostics,
        max_batch_tokens: BatchTokenBudget,
        max_prefill_tokens_per_cycle: PrefillTokenBudget,
        max_prefill_layers_per_cycle: PrefillLayerBudget,
        response_delivery_policy: ResponseDeliveryPolicy,
    ) -> Self {
        let max_decode_batch = max_decode_batch.max(1);
        let BatchingActorAdmissionConfig {
            prefix_aware_admission,
            prefill_admission_quantum,
            burst_prefill_admission,
        } = admission;
        Self::start_with_policy_and_cycle_idle(
            forward,
            max_decode_batch,
            max_batch_tokens,
            max_prefill_tokens_per_cycle,
            max_prefill_layers_per_cycle,
            prefix_aware_admission,
            prefill_admission_quantum,
            burst_prefill_admission,
            actor_cycle_idle,
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
        Self::start_with_policy_and_cycle_idle(
            forward,
            max_decode_batch,
            max_batch_tokens,
            max_prefill_tokens_per_cycle,
            max_prefill_layers_per_cycle,
            prefix_aware_admission,
            prefill_admission_quantum,
            burst_refill,
            ActorCycleIdleDiagnostics::default(),
            response_delivery_policy,
        )
    }

    fn start_with_policy_and_cycle_idle(
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        max_batch_tokens: BatchTokenBudget,
        max_prefill_tokens_per_cycle: PrefillTokenBudget,
        max_prefill_layers_per_cycle: PrefillLayerBudget,
        prefix_aware_admission: bool,
        prefill_admission_quantum: usize,
        burst_refill: bool,
        actor_cycle_idle: ActorCycleIdleDiagnostics,
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
            actor_cycle_idle,
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
        let enqueued_at = Instant::now();
        self.tx
            .send(EngineCommand::Enqueue {
                req,
                response_tx,
                enqueued_at,
            })
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
        let enqueued_at = Instant::now();
        self.tx
            .send(EngineCommand::Enqueue {
                req,
                response_tx,
                enqueued_at,
            })
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
        self.resize_kv_with_reason(target_blocks, KvResizeReason::Maintenance)
            .await
    }

    async fn resize_kv_with_reason(
        &self,
        target_blocks: usize,
        reason: KvResizeReason,
    ) -> Result<usize> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::ResizeKv {
                target_blocks,
                reason,
                enqueued_at: Instant::now(),
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
    pub(crate) fn resize_kv_blocking(
        &self,
        target_blocks: usize,
        reason: KvResizeReason,
    ) -> Result<usize> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .blocking_send(EngineCommand::ResizeKv {
                target_blocks,
                reason,
                enqueued_at: Instant::now(),
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
        enqueued_at: Instant,
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
    /// Queued at the between-requests barrier so no active request retains a
    /// graph, recurrent state, or block-table reference to the old pool.
    ResizeKv {
        target_blocks: usize,
        reason: KvResizeReason,
        enqueued_at: Instant,
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

/// FIFO-exclusive GPU/model mutation waiting for the active batch to drain.
/// Keeping resize and adapter work in one queue preserves command ordering.
enum PendingExclusiveMutation {
    ResizeKv {
        target_blocks: usize,
        reason: KvResizeReason,
        enqueued_at: Instant,
        reply: oneshot::Sender<std::result::Result<usize, String>>,
    },
    SwapAdapter {
        swap: AdapterSwapClosure,
        reply: oneshot::Sender<std::result::Result<(), String>>,
    },
}

struct QueuedRequest {
    req: EngineRequest,
    delivery_key: DeliveryKey,
    enqueued_at: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveDeliveryState {
    Ready,
    InFlight { sequence: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveAdmissionLane {
    Ordinary,
    PrefillStaging,
}

struct ActiveRequest {
    req: EngineRequest,
    delivery_key: DeliveryKey,
    delivery_state: ActiveDeliveryState,
    next_delivery_sequence: u64,
    admission_lane: ActiveAdmissionLane,
    /// Actual prompt work remaining when this request entered the active set.
    /// This is immutable so the short-prefill lane compares request classes,
    /// not mutable progress that would split an equal-work cohort.
    initial_prefill_work_tokens: Option<usize>,
    actor_queue_duration: Duration,
    actor_admission_duration: Duration,
    admitted_at: Instant,
    first_token_ready_after_admission: Option<Duration>,
    phase_window_started_at: Instant,
    token_phase_durations: TokenPhaseDurations,
    inflight_token_ready_at: Option<Instant>,
    action_tokens: Option<Vec<EngineActionToken>>,
    resident_prefill_used: bool,
    slot: DecodeSlot,
}

#[derive(Default)]
struct AdmissionOutcome {
    submitted_first_tokens: bool,
    tokens_scheduled: usize,
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
    max_prefill_staging_slots: usize,
    max_active_requests: usize,
    next_prefill_index: usize,
    short_prefill_priority_cursor: usize,
    prefill_staging_priority_cursor: usize,
    next_staged_prefill_generation: u64,
    next_decode_generation: u64,
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
    actor_cycle_idle: ActorCycleIdleDiagnostics,
    response_delivery_policy: ResponseDeliveryPolicy,
    delivery_worker: Option<DeliveryWorker>,
    delivery_results: std_mpsc::Receiver<Vec<DeliveryResult>>,
    next_delivery_generation: u64,
    delivery_backpressured: HashSet<(DeliveryKey, u64)>,
    delivery_pending_terminal: HashSet<DeliveryKey>,
    delivery_outbox: Vec<(DeliveryKey, DeliveryBatch)>,
    defer_delivery_flush: bool,
    stop_replies: Vec<oneshot::Sender<()>>,
    /// KV resizes and adapter swaps waiting for the active batch to drain.
    /// Admission pauses until these ordered mutations finish.
    pending_exclusive_mutations: VecDeque<PendingExclusiveMutation>,
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
        actor_cycle_idle: ActorCycleIdleDiagnostics,
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
        let max_prefill_staging_slots = if max_decode_batch > 1 && !burst_refill {
            max_prefill_admissions_per_cycle.min(MAX_PREFILL_STAGING_SLOTS)
        } else {
            0
        };
        let max_active_requests = max_decode_batch.saturating_add(max_prefill_staging_slots);
        let max_prefill_staging_priority_burst = if max_prefill_staging_slots > 0 {
            PREFILL_STAGING_ROUND_ROBIN_INTERVAL.saturating_sub(1)
        } else {
            0
        };
        tracing::info!(
            max_decode_batch,
            max_prefill_staging_slots,
            max_active_requests,
            max_prefill_staging_priority_burst,
            burst_refill,
            "batching active-set policy resolved"
        );
        tracing::info!(
            actor_cycle_idle_ms = actor_cycle_idle.milliseconds,
            actor_cycle_idle_source = %actor_cycle_idle.source,
            actor_cycle_idle_enabled = actor_cycle_idle.enabled,
            actor_cycle_idle_command_poll_ms = actor_cycle_idle.command_poll_milliseconds,
            "batching actor cooperative cycle-idle policy resolved"
        );
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
            prefix_cache_enabled: forward.prefix_cache_enabled(),
            resident_prefill_enabled: forward.resident_prefill_enabled(),
            max_batch_tokens: max_batch_tokens.tokens(),
            max_batch_tokens_source: max_batch_tokens.source(),
            max_prefill_tokens_per_cycle,
            max_prefill_tokens_per_cycle_source,
            max_prefill_layers_per_cycle,
            max_prefill_layers_per_cycle_source,
            max_prefill_admission_quantum: max_prefill_admissions_per_cycle,
            max_prefill_staging_slots,
            max_active_requests,
            max_prefill_staging_priority_burst,
            max_decode_batch,
            stream_stall_grace_ms: duration_millis_saturating(
                response_delivery_policy.stream_stall_grace,
            ),
            stream_stall_grace_source: response_delivery_policy.stream_stall_grace_source,
            actor_cycle_idle_ms: actor_cycle_idle.milliseconds,
            actor_cycle_idle_source: actor_cycle_idle.source,
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
            max_prefill_staging_slots,
            max_active_requests,
            next_prefill_index: 0,
            short_prefill_priority_cursor: 0,
            prefill_staging_priority_cursor: 0,
            next_staged_prefill_generation: 0,
            next_decode_generation: 0,
            prefix_aware_admission,
            max_prefill_admissions_per_cycle,
            burst_refill,
            actor_cycle_idle,
            response_delivery_policy,
            delivery_worker: Some(delivery_worker),
            delivery_results,
            next_delivery_generation: 0,
            delivery_backpressured: HashSet::new(),
            delivery_pending_terminal: HashSet::new(),
            delivery_outbox: Vec::new(),
            defer_delivery_flush: false,
            stop_replies: Vec::new(),
            pending_exclusive_mutations: VecDeque::new(),
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
            // Between-requests barrier: with no decode step in flight and the
            // active batch drained, queued pool/weight mutations execute before
            // blocking on the channel.
            self.run_pending_exclusive_mutations_at_barrier();

            if self.active.is_empty()
                && self.waiting.is_empty()
                && self.pending_exclusive_mutations.is_empty()
            {
                match self.rx.blocking_recv() {
                    Some(cmd) => self.handle_command(cmd),
                    None => break,
                }
                // A mutation that arrived while idle executes immediately.
                self.run_pending_exclusive_mutations_at_barrier();
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
                .saturating_sub(admission.tokens_scheduled);
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
                .saturating_sub(admission.tokens_scheduled)
                .saturating_sub(decoded_tokens)
                .min(
                    self.max_prefill_tokens_per_cycle
                        .saturating_sub(admission.tokens_scheduled),
                );
            let advanced_prefill = self.run_prefill_budget(prefill_budget);
            if decoded_tokens > 0 || advanced_prefill {
                self.cooperative_actor_cycle_idle();
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

    fn cooperative_actor_cycle_idle(&mut self) {
        if !self.actor_cycle_idle.enabled || self.actor_cycle_idle.milliseconds == 0 {
            return;
        }

        let target = Duration::from_millis(self.actor_cycle_idle.milliseconds);
        let poll = Duration::from_millis(
            self.actor_cycle_idle
                .command_poll_milliseconds
                .clamp(1, ACTOR_CYCLE_IDLE_COMMAND_POLL_MS),
        );
        let started = Instant::now();
        self.snapshot.actor_cycle_idle_active = true;
        self.snapshot.actor_cycle_idle_count =
            self.snapshot.actor_cycle_idle_count.saturating_add(1);
        self.refresh_snapshot();

        while !self.stopped {
            let elapsed = started.elapsed();
            if elapsed >= target {
                break;
            }
            thread::sleep((target - elapsed).min(poll));
            self.drain_commands();
            self.drain_delivery_results();
        }

        let elapsed = started.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
        for active in &mut self.active {
            active.token_phase_durations.add_actor_cycle_idle(elapsed);
        }
        self.snapshot.actor_cycle_idle_active = false;
        self.snapshot.total_actor_cycle_idle_ms += elapsed_ms;
        self.snapshot.max_actor_cycle_idle_ms =
            self.snapshot.max_actor_cycle_idle_ms.max(elapsed_ms);
        self.refresh_snapshot();
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
                    if let Some(ready_at) = active.inflight_token_ready_at.take() {
                        active.token_phase_durations.add_response_delivery(
                            Instant::now().saturating_duration_since(ready_at),
                        );
                    }
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
            EngineCommand::Enqueue {
                req,
                response_tx,
                enqueued_at,
            } => {
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
                    self.waiting.push_back(QueuedRequest {
                        req,
                        delivery_key,
                        enqueued_at,
                    });
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
                reason,
                enqueued_at,
                reply,
            } => {
                self.pending_exclusive_mutations
                    .push_back(PendingExclusiveMutation::ResizeKv {
                        target_blocks,
                        reason,
                        enqueued_at,
                        reply,
                    });
            }
            EngineCommand::SwapAdapter { swap, reply } => {
                self.pending_exclusive_mutations
                    .push_back(PendingExclusiveMutation::SwapAdapter { swap, reply });
            }
        }
    }

    /// Execute ordered pool/weight mutations only after the active batch has
    /// drained. Admission remains paused until the queue is empty.
    fn run_pending_exclusive_mutations_at_barrier(&mut self) {
        if self.pending_exclusive_mutations.is_empty() || !self.active.is_empty() {
            return;
        }
        while let Some(pending) = self.pending_exclusive_mutations.pop_front() {
            match pending {
                PendingExclusiveMutation::ResizeKv {
                    target_blocks,
                    reason,
                    enqueued_at,
                    reply,
                } => {
                    let barrier_wait = enqueued_at.elapsed();
                    let result = self
                        .forward
                        .resize_kv_with_context(target_blocks, reason, barrier_wait)
                        .map_err(|error| format!("{error:#}"));
                    let _ = reply.send(result);
                }
                PendingExclusiveMutation::SwapAdapter { swap, reply } => {
                    let _ = reply.send(swap());
                }
            }
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

    fn active_admission_lane_counts(&self) -> (usize, usize) {
        let staging = self
            .active
            .iter()
            .filter(|active| active.admission_lane == ActiveAdmissionLane::PrefillStaging)
            .count();
        (self.active.len().saturating_sub(staging), staging)
    }

    fn prefill_staging_entry_token_limit(&self) -> usize {
        self.max_prefill_tokens_per_cycle
            .saturating_mul(SHORT_PREFILL_PRIORITY_MAX_CHUNKS.saturating_add(1))
    }

    fn prefill_staging_candidate(&self, queued: &QueuedRequest, token_budget: usize) -> bool {
        queued.req.prompt_tokens.len() <= self.prefill_staging_entry_token_limit()
            && (self.forward.supports_resumable_prefill()
                || queued.req.prompt_tokens.len() <= token_budget)
    }

    /// Admit queued requests and report whether this cycle submitted one or
    /// more prefill-produced first tokens to the delivery worker.
    fn admit_waiting_with_budget(&mut self, mut token_budget: usize) -> AdmissionOutcome {
        // Pending pool/weight mutations need the active batch to drain. Keep
        // new requests queued until the ordered mutation queue completes.
        if !self.pending_exclusive_mutations.is_empty() {
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
        // Burst refill only fills ordinary decode-width slots. Latency-oriented
        // actors may additionally use their separately bounded short-prefill
        // staging lane without changing the backend decode width.
        let admission_limit = if self.has_ready_decode_row() && !self.burst_refill {
            1
        } else {
            self.max_prefill_admissions_per_cycle
        };
        while admitted < admission_limit && !self.waiting.is_empty() && token_budget > 0 {
            if self.active.len() >= self.max_active_requests {
                break;
            }
            let (ordinary_active, staging_active) = self.active_admission_lane_counts();
            let admission_lane = if ordinary_active < self.max_decode_batch {
                ActiveAdmissionLane::Ordinary
            } else if staging_active < self.max_prefill_staging_slots {
                ActiveAdmissionLane::PrefillStaging
            } else {
                break;
            };
            // Count deferrals during the admission scan itself — when every
            // waiting row is deferred, position() has already evaluated the
            // predicate for all of them, so a second full filter pass would
            // double the (prefix-cache-locking) work for the same answer.
            let mut deferred_seen: u64 = 0;
            let Some(waiting_idx) = self.waiting.iter().position(|queued| {
                let defer = self.should_defer_for_active_prefix(queued);
                deferred_seen += u64::from(defer);
                !defer
                    && (admission_lane == ActiveAdmissionLane::Ordinary
                        || self.prefill_staging_candidate(queued, token_budget))
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
            let admission_started_at = Instant::now();
            let preparation = self
                .forward
                .prepare_request_chunked(&queued.req, token_budget);
            let actor_admission_duration = admission_started_at.elapsed();
            for active in &mut self.active {
                active
                    .token_phase_durations
                    .add_actor_admission(actor_admission_duration);
            }
            self.record_admission_duration(
                actor_admission_duration,
                queued.req.request_id,
                queued.req.prompt_tokens.len(),
                token_budget,
            );
            match preparation {
                Ok(preparation) => {
                    let (slot, tokens_scheduled, tokens_processed, ready) = match preparation {
                        RequestPreparation::Prefilling {
                            slot,
                            tokens_scheduled,
                            tokens_processed,
                            ..
                        } => (slot, tokens_scheduled, tokens_processed, false),
                        RequestPreparation::Ready {
                            slot,
                            tokens_scheduled,
                            tokens_processed,
                            ..
                        } => (slot, tokens_scheduled, tokens_processed, true),
                    };
                    if tokens_scheduled > token_budget || tokens_processed > tokens_scheduled {
                        self.forward.discard_request(slot);
                        self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                        self.terminate_delivery(
                            queued.delivery_key,
                            format!(
                                "prefill admission scheduled {tokens_scheduled} and completed {tokens_processed} tokens for a {token_budget}-token remainder"
                            ),
                        );
                        continue;
                    }
                    self.snapshot.last_prefill_tokens = tokens_processed;
                    self.snapshot.total_prefill_tokens = self
                        .snapshot
                        .total_prefill_tokens
                        .saturating_add(tokens_processed as u64);
                    token_budget -= tokens_scheduled;
                    admitted += 1;
                    if admission_lane == ActiveAdmissionLane::PrefillStaging {
                        self.snapshot.total_prefill_staging_admissions = self
                            .snapshot
                            .total_prefill_staging_admissions
                            .saturating_add(1);
                    }
                    let active_idx = self.active.len();
                    let admitted_at = Instant::now();
                    let actor_queue_duration =
                        admission_started_at.saturating_duration_since(queued.enqueued_at);
                    let phase_window_started_at = queued.enqueued_at;
                    let mut token_phase_durations = TokenPhaseDurations::default();
                    token_phase_durations.add_actor_queue(actor_queue_duration);
                    token_phase_durations.add_actor_admission(actor_admission_duration);
                    let initial_prefill_work_tokens = if ready {
                        None
                    } else {
                        self.forward.remaining_prefill_tokens(&slot)
                    };
                    let action_tokens = queued.req.capture_behavior_logprobs.then(Vec::new);
                    self.active.push(ActiveRequest {
                        req: queued.req,
                        delivery_key: queued.delivery_key,
                        delivery_state: ActiveDeliveryState::Ready,
                        next_delivery_sequence: 0,
                        admission_lane,
                        initial_prefill_work_tokens,
                        actor_queue_duration,
                        actor_admission_duration,
                        admitted_at,
                        first_token_ready_after_admission: None,
                        phase_window_started_at,
                        token_phase_durations,
                        inflight_token_ready_at: None,
                        action_tokens,
                        resident_prefill_used: false,
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
            tokens_scheduled: initial_token_budget.saturating_sub(token_budget),
        }
    }

    #[cfg(test)]
    fn admit_waiting(&mut self) -> bool {
        self.admit_waiting_with_budget(self.max_batch_tokens.min(self.max_prefill_tokens_per_cycle))
            .submitted_first_tokens
    }

    fn select_prefill_index(&mut self, budget: usize) -> Option<(usize, bool)> {
        let active_len = self.active.len();
        let round_robin_start = self.next_prefill_index % active_len;
        let eligible = |idx: usize| {
            self.active[idx].delivery_state == ActiveDeliveryState::Ready
                && self.forward.is_prefilling(&self.active[idx].slot)
                && (budget > 0
                    || self
                        .forward
                        .inflight_prefill_token_width(&self.active[idx].slot)
                        .is_some())
        };
        let round_robin = (0..active_len)
            .map(|offset| (round_robin_start + offset) % active_len)
            .find(|&idx| eligible(idx))?;

        let short_tail_limit = self
            .max_prefill_tokens_per_cycle
            .saturating_mul(SHORT_PREFILL_PRIORITY_MAX_CHUNKS);
        let staging_entry_limit = self.prefill_staging_entry_token_limit();
        // Staged rows already passed the bounded prompt-size gate at admission.
        // Rotate them on priority turns; the staged cadence still forces a
        // global turn that advances ordinary rows under a continuously full lane.
        let mut staged: Vec<(usize, u64)> = (0..active_len)
            .filter(|&idx| {
                eligible(idx)
                    && self.active[idx].admission_lane == ActiveAdmissionLane::PrefillStaging
                    && self
                        .forward
                        .remaining_prefill_tokens(&self.active[idx].slot)
                        .is_some_and(|remaining| remaining <= staging_entry_limit)
            })
            .map(|idx| (idx, self.active[idx].delivery_key.generation))
            .collect();
        staged.sort_unstable_by_key(|&(_, generation)| generation);
        if !staged.is_empty() {
            self.prefill_staging_priority_cursor =
                (self.prefill_staging_priority_cursor + 1) % PREFILL_STAGING_ROUND_ROBIN_INTERVAL;
            if self.prefill_staging_priority_cursor == 0 {
                return Some((round_robin, false));
            }
            let start = staged.partition_point(|&(_, generation)| {
                generation < self.next_staged_prefill_generation
            });
            let (priority, generation) = staged[start % staged.len()];
            self.next_staged_prefill_generation = generation.checked_add(1).unwrap_or(0);
            self.snapshot.total_short_prefill_priority_forwards = self
                .snapshot
                .total_short_prefill_priority_forwards
                .saturating_add(1);
            self.snapshot.total_prefill_staging_priority_forwards = self
                .snapshot
                .total_prefill_staging_priority_forwards
                .saturating_add(1);
            return Some((priority, true));
        }
        self.short_prefill_priority_cursor =
            (self.short_prefill_priority_cursor + 1) % SHORT_PREFILL_ROUND_ROBIN_INTERVAL;
        if self.short_prefill_priority_cursor == 0 {
            return Some((round_robin, false));
        }
        let Some(largest_initial_work) = (0..active_len)
            .filter(|&idx| eligible(idx))
            .filter_map(|idx| self.active[idx].initial_prefill_work_tokens)
            .max()
        else {
            return Some((round_robin, false));
        };
        let priority = (0..active_len)
            .filter(|&idx| eligible(idx))
            .filter_map(|idx| {
                let remaining = self
                    .forward
                    .remaining_prefill_tokens(&self.active[idx].slot)?;
                let initial_work = self.active[idx].initial_prefill_work_tokens?;
                // Compare immutable admission-time work classes. Mutable
                // remainders made equal prompts look shorter after ordinary
                // progress, while a half-remainder threshold withheld the
                // lane from legitimately shorter late arrivals.
                (remaining <= short_tail_limit && initial_work < largest_initial_work)
                    .then_some((idx, remaining))
            })
            .min_by_key(|&(idx, remaining)| {
                let round_robin_distance = (idx + active_len - round_robin_start) % active_len;
                (remaining, round_robin_distance)
            })
            .map(|(idx, _)| idx);
        if let Some(priority) = priority {
            self.snapshot.total_short_prefill_priority_forwards = self
                .snapshot
                .total_short_prefill_priority_forwards
                .saturating_add(1);
            Some((priority, true))
        } else {
            Some((round_robin, false))
        }
    }

    fn update_prefill_cursor(
        &mut self,
        selected_idx: usize,
        selected_by_priority: bool,
        reinserted: bool,
    ) {
        if self.active.is_empty() {
            self.next_prefill_index = 0;
        } else if selected_by_priority {
            self.next_prefill_index %= self.active.len();
        } else if reinserted {
            self.next_prefill_index = (selected_idx + 1) % self.active.len();
        } else {
            self.next_prefill_index = selected_idx % self.active.len();
        }
    }

    /// Use the resident Vulkan decode stack as a one-token prompt batch once
    /// every active prefill row has reached a safe committed-token boundary.
    /// Returning `None` leaves the ordinary layer-resumable scheduler in
    /// charge. Rows that already entered the resident route remain eligible as
    /// a single-row tail because their newer KV positions are resident-only.
    fn run_resident_prefill_batch(&mut self, budget: &mut usize) -> Option<bool> {
        if !self.snapshot.resident_prefill_enabled || *budget == 0 || self.active.is_empty() {
            return None;
        }
        let active_len = self.active.len();
        let round_robin_start = self.next_prefill_index % active_len;
        let ready_prefill_indices: Vec<usize> = (0..active_len)
            .filter(|&idx| {
                self.active[idx].delivery_state == ActiveDeliveryState::Ready
                    && self.forward.is_prefilling(&self.active[idx].slot)
            })
            .collect();
        if ready_prefill_indices.is_empty() {
            return None;
        }
        let required = ready_prefill_indices.iter().any(|&idx| {
            self.forward
                .resident_prefill_batch_required(&self.active[idx].slot)
        });
        let invalid_required_indices: Vec<usize> = ready_prefill_indices
            .iter()
            .copied()
            .filter(|&idx| {
                self.forward
                    .resident_prefill_batch_required(&self.active[idx].slot)
                    && !self.forward.resident_prefill_batch_candidate(
                        &self.active[idx].slot,
                        &self.active[idx].req.sampling,
                    )
            })
            .collect();
        if !invalid_required_indices.is_empty() {
            self.snapshot.total_resident_prefill_route_failures = self
                .snapshot
                .total_resident_prefill_route_failures
                .saturating_add(1);
            let error = "resident token-prefill row lost native-route eligibility".to_string();
            for idx in invalid_required_indices.iter().copied().rev() {
                let active = self.active.remove(idx);
                self.forward.discard_request(active.slot);
                self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                self.terminate_delivery(active.delivery_key, error.clone());
            }
            self.refresh_snapshot();
            return Some(true);
        }
        let every_prefill_is_candidate = ready_prefill_indices.iter().all(|&idx| {
            self.forward.resident_prefill_batch_candidate(
                &self.active[idx].slot,
                &self.active[idx].req.sampling,
            )
        });
        if !required && !every_prefill_is_candidate {
            return None;
        }

        let max_rows = self.max_decode_batch.min(*budget);
        let mut indices = Vec::with_capacity(max_rows);
        for offset in 0..active_len {
            let idx = (round_robin_start + offset) % active_len;
            if self.active[idx].delivery_state == ActiveDeliveryState::Ready
                && self.forward.resident_prefill_batch_candidate(
                    &self.active[idx].slot,
                    &self.active[idx].req.sampling,
                )
            {
                indices.push(idx);
                if indices.len() == max_rows {
                    break;
                }
            }
        }
        let next_prefill_index = indices
            .last()
            .map(|idx| (idx + 1) % active_len)
            .unwrap_or(round_robin_start);
        if required
            && !indices.iter().any(|&idx| {
                self.forward
                    .resident_prefill_batch_required(&self.active[idx].slot)
            })
        {
            let required_idx = (0..active_len)
                .map(|offset| (round_robin_start + offset) % active_len)
                .find(|&idx| {
                    self.forward
                        .resident_prefill_batch_required(&self.active[idx].slot)
                })
                .expect("at least one required resident prefill row was observed");
            if indices.len() == max_rows {
                if let Some(last) = indices.last_mut() {
                    *last = required_idx;
                }
            } else {
                indices.push(required_idx);
            }
        }
        if indices.is_empty()
            || (indices.len() == 1
                && !self
                    .forward
                    .resident_prefill_batch_required(&self.active[indices[0]].slot))
        {
            return None;
        }
        indices.sort_unstable();

        let sampling: Vec<SamplingParams> = indices
            .iter()
            .map(|&idx| self.active[idx].req.sampling.clone())
            .collect();
        let cancels: Vec<CancelHandle> = indices
            .iter()
            .map(|&idx| self.active[idx].req.cancel.clone())
            .collect();
        let started = Instant::now();
        self.snapshot.total_resident_prefill_attempts = self
            .snapshot
            .total_resident_prefill_attempts
            .saturating_add(1);
        let result = {
            let mut selected = self
                .active
                .iter_mut()
                .enumerate()
                .filter_map(|(idx, active)| {
                    indices.binary_search(&idx).ok().map(|_| &mut active.slot)
                })
                .collect::<Vec<_>>();
            self.forward
                .advance_resident_prefill_batch(&mut selected, &sampling, &cancels)
        };
        let elapsed = started.elapsed();
        let progress = match result {
            Ok(Some(progress)) => progress,
            Ok(None) if !required => {
                self.snapshot.total_resident_prefill_initial_declines = self
                    .snapshot
                    .total_resident_prefill_initial_declines
                    .saturating_add(1);
                return None;
            }
            Ok(None) => {
                self.snapshot.total_resident_prefill_route_failures = self
                    .snapshot
                    .total_resident_prefill_route_failures
                    .saturating_add(1);
                let error = "resident token-prefill native route declined an owned row".to_string();
                for idx in indices.iter().copied().rev() {
                    let active = self.active.remove(idx);
                    self.forward.discard_request(active.slot);
                    self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                    self.terminate_delivery(active.delivery_key, error.clone());
                }
                self.refresh_snapshot();
                return Some(true);
            }
            Err(error) => {
                self.snapshot.total_resident_prefill_route_failures = self
                    .snapshot
                    .total_resident_prefill_route_failures
                    .saturating_add(1);
                for idx in indices.iter().copied().rev() {
                    let active = self.active.remove(idx);
                    self.forward.discard_request(active.slot);
                    self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                    self.terminate_delivery(active.delivery_key, format!("{error:#}"));
                }
                self.refresh_snapshot();
                return Some(true);
            }
        };
        if progress.len() != indices.len() {
            self.snapshot.total_resident_prefill_route_failures = self
                .snapshot
                .total_resident_prefill_route_failures
                .saturating_add(1);
            let error = format!(
                "resident prefill batch returned {} rows for {} requests",
                progress.len(),
                indices.len()
            );
            for idx in indices.iter().copied().rev() {
                let active = self.active.remove(idx);
                self.forward.discard_request(active.slot);
                self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                self.terminate_delivery(active.delivery_key, error.clone());
            }
            self.refresh_snapshot();
            return Some(true);
        }

        let layers_processed = progress[0].layers_processed;
        let progress_valid = layers_processed > 0
            && progress.iter().all(|row| {
                row.tokens_scheduled == 1
                    && row.tokens_processed == 1
                    && row.layers_processed == layers_processed
            });
        if !progress_valid {
            self.snapshot.total_resident_prefill_route_failures = self
                .snapshot
                .total_resident_prefill_route_failures
                .saturating_add(1);
            let error =
                "resident prefill batch violated its one-token/full-stack progress contract"
                    .to_string();
            for idx in indices.iter().copied().rev() {
                let active = self.active.remove(idx);
                self.forward.discard_request(active.slot);
                self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                self.terminate_delivery(active.delivery_key, error.clone());
            }
            self.refresh_snapshot();
            return Some(true);
        }

        for active in &mut self.active {
            active.token_phase_durations.add_actor_prefill(elapsed);
        }
        self.record_prefill_forward_duration(
            elapsed,
            self.active[indices[0]].req.request_id,
            indices.len(),
            layers_processed,
        );
        self.snapshot.total_prefill_forwards =
            self.snapshot.total_prefill_forwards.saturating_add(1);
        self.snapshot.total_resident_prefill_forwards = self
            .snapshot
            .total_resident_prefill_forwards
            .saturating_add(1);
        self.snapshot.total_resident_prefill_rows = self
            .snapshot
            .total_resident_prefill_rows
            .saturating_add(indices.len() as u64);
        self.snapshot.total_resident_prefill_completed_rows = self
            .snapshot
            .total_resident_prefill_completed_rows
            .saturating_add(progress.iter().filter(|row| row.ready).count() as u64);
        self.snapshot.last_resident_prefill_batch_size = indices.len();
        self.snapshot.max_resident_prefill_batch_size = self
            .snapshot
            .max_resident_prefill_batch_size
            .max(indices.len());
        self.snapshot.last_prefill_layers = layers_processed;
        self.snapshot.total_prefill_layers = self
            .snapshot
            .total_prefill_layers
            .saturating_add(layers_processed as u64);
        self.snapshot.last_prefill_tokens = indices.len();
        self.snapshot.total_prefill_tokens = self
            .snapshot
            .total_prefill_tokens
            .saturating_add(indices.len() as u64);
        *budget -= indices.len();
        self.next_prefill_index = next_prefill_index;

        for &idx in &indices {
            self.active[idx].resident_prefill_used = true;
        }
        for (&idx, row) in indices.iter().zip(&progress).rev() {
            if row.ready {
                self.emit_pending_first_token_at(idx);
            }
        }
        self.refresh_snapshot();
        Some(true)
    }

    /// Spend the combined-cycle remainder on newly selected prompt chunks.
    /// Retained layer groups were charged when their chunk began and resume
    /// without a second token charge; the independent layer ceiling still
    /// bounds each forward. Partial rows are selected round-robin so a 16K
    /// prompt cannot hide a 1K prompt behind repeated quanta.
    fn run_prefill_budget(&mut self, mut budget: usize) -> bool {
        if let Some(advanced) = self.run_resident_prefill_batch(&mut budget) {
            return advanced;
        }
        let mut advanced = false;
        while !self.active.is_empty() {
            let Some((idx, selected_by_priority)) = self.select_prefill_index(budget) else {
                break;
            };
            let reserved_width = self
                .forward
                .inflight_prefill_token_width(&self.active[idx].slot);

            let ActiveRequest {
                req,
                delivery_key,
                delivery_state,
                next_delivery_sequence,
                admission_lane,
                initial_prefill_work_tokens,
                actor_queue_duration,
                actor_admission_duration,
                admitted_at,
                first_token_ready_after_admission,
                phase_window_started_at,
                mut token_phase_durations,
                inflight_token_ready_at,
                action_tokens,
                resident_prefill_used,
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
            token_phase_durations.add_actor_prefill(elapsed);
            for active in &mut self.active {
                active.token_phase_durations.add_actor_prefill(elapsed);
            }
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
                    self.update_prefill_cursor(idx, selected_by_priority, false);
                    self.refresh_snapshot();
                    advanced = true;
                    continue;
                }
            };
            let (slot, tokens_scheduled, tokens_processed, layers_processed, ready) =
                match preparation {
                    RequestPreparation::Prefilling {
                        slot,
                        tokens_scheduled,
                        tokens_processed,
                        layers_processed,
                    } => (
                        slot,
                        tokens_scheduled,
                        tokens_processed,
                        layers_processed,
                        false,
                    ),
                    RequestPreparation::Ready {
                        slot,
                        tokens_scheduled,
                        tokens_processed,
                        layers_processed,
                    } => (
                        slot,
                        tokens_scheduled,
                        tokens_processed,
                        layers_processed,
                        true,
                    ),
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
                self.update_prefill_cursor(idx, selected_by_priority, false);
                self.refresh_snapshot();
                advanced = true;
                continue;
            }
            let reservation_valid = match reserved_width {
                Some(width) => {
                    tokens_scheduled == 0 && (tokens_processed == 0 || tokens_processed == width)
                }
                None => {
                    tokens_scheduled > 0
                        && tokens_scheduled <= budget
                        && tokens_processed <= tokens_scheduled
                }
            };
            if !reservation_valid {
                self.forward.discard_request(slot);
                self.snapshot.total_errors = self.snapshot.total_errors.saturating_add(1);
                self.terminate_delivery(
                    delivery_key,
                    format!(
                        "prefill forward violated token reservation: prior={reserved_width:?}, scheduled={tokens_scheduled}, completed={tokens_processed}, new-token budget={budget}"
                    ),
                );
                self.update_prefill_cursor(idx, selected_by_priority, false);
                self.refresh_snapshot();
                advanced = true;
                continue;
            }
            budget -= tokens_scheduled;
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
                    self.update_prefill_cursor(idx, selected_by_priority, false);
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
                        admission_lane,
                        initial_prefill_work_tokens,
                        actor_queue_duration,
                        actor_admission_duration,
                        admitted_at,
                        first_token_ready_after_admission,
                        phase_window_started_at,
                        token_phase_durations,
                        inflight_token_ready_at,
                        action_tokens,
                        resident_prefill_used,
                        slot,
                    },
                );
                self.update_prefill_cursor(idx, selected_by_priority, true);
                self.snapshot.total_prefill_layer_yields =
                    self.snapshot.total_prefill_layer_yields.saturating_add(1);
                self.refresh_snapshot();
                return true;
            }
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
                    admission_lane,
                    initial_prefill_work_tokens,
                    actor_queue_duration,
                    actor_admission_duration,
                    admitted_at,
                    first_token_ready_after_admission,
                    phase_window_started_at,
                    token_phase_durations,
                    inflight_token_ready_at,
                    action_tokens,
                    resident_prefill_used,
                    slot,
                },
            );
            self.update_prefill_cursor(idx, selected_by_priority, true);
            if ready {
                self.emit_pending_first_token_at(idx);
            }
            self.refresh_snapshot();
            advanced = true;
            // A resumed chunk already consumed its new-token budget when it
            // was selected. Its final layer group still owns this cycle's
            // layer quantum, so return to decode before starting more work.
            if reserved_width.is_some() {
                return true;
            }
        }
        advanced
    }

    fn pending_first_token_at(&mut self, idx: usize) -> Option<EngineSampledToken> {
        match self.active.get_mut(idx).map(|active| &mut active.slot) {
            Some(DecodeSlot::Real {
                state,
                first_token_pending,
                ..
            }) if *first_token_pending => {
                *first_token_pending = false;
                Some(match state.next_token_logprob {
                    Some(logprob) => EngineSampledToken::traced(state.next_token, logprob),
                    None => EngineSampledToken::untraced(state.next_token),
                })
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

    fn emit_output_token_at(&mut self, idx: usize, sampled: EngineSampledToken) {
        let ready_at = Instant::now();
        if self.active[idx].req.capture_behavior_logprobs
            && !sampled
                .behavior_logprob
                .is_some_and(|logprob| logprob.is_finite() && logprob <= 1e-6)
        {
            self.finish_one_with_error(
                idx,
                format!(
                    "behavior-logprob capture produced no valid probability for sampled token {}",
                    sampled.token_id
                ),
                None,
            );
            return;
        }
        let generated_index = self.generated_tokens_for(idx).len();
        if generated_index == 0 && self.active[idx].first_token_ready_after_admission.is_none() {
            self.active[idx].first_token_ready_after_admission =
                Some(ready_at.saturating_duration_since(self.active[idx].admitted_at));
        }
        let decision = {
            let generated_tokens = self.generated_tokens_for(idx);
            self.active[idx]
                .req
                .sampling
                .apply_thinking_budget_with_source(generated_tokens, sampled.token_id)
        };
        let token = decision.token;
        let action = self.active[idx]
            .action_tokens
            .as_ref()
            .map(|_| EngineActionToken {
                generated_index,
                token_id: token,
                source: match decision.source {
                    ThinkingBudgetTokenSource::Sampled => EngineActionTokenSource::Sampled,
                    ThinkingBudgetTokenSource::Forced => EngineActionTokenSource::Forced,
                },
                behavior_logprob: match decision.source {
                    ThinkingBudgetTokenSource::Sampled => sampled.behavior_logprob,
                    ThinkingBudgetTokenSource::Forced => None,
                },
            });
        if !self.active[idx].req.sampling.ignore_eos {
            match self.forward.is_eos_token(token) {
                Ok(true) => {
                    if let Some(action) = action {
                        self.active[idx]
                            .action_tokens
                            .as_mut()
                            .expect("action trace disappeared while recording terminal EOS")
                            .push(action);
                    }
                    self.snapshot.total_decode_tokens += 1;
                    self.finish_active(idx, FinishReason::Eos, None);
                    return;
                }
                Ok(false) => {}
                Err(err) => {
                    self.finish_one_with_error(idx, format!("{err:#}"), None);
                    return;
                }
            }
        }

        let generated_count = match self.forward.accept_token(&mut self.active[idx].slot, token) {
            Ok(count) => count,
            Err(err) => {
                self.finish_one_with_error(idx, format!("{err:#}"), None);
                return;
            }
        };
        if let Some(action) = action {
            self.active[idx]
                .action_tokens
                .as_mut()
                .expect("action trace disappeared while recording accepted token")
                .push(action);
        }
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
        let timing = {
            let active = &mut self.active[idx];
            let mut phases = std::mem::take(&mut active.token_phase_durations);
            phases.account_unexplained_wall_time(
                ready_at.saturating_duration_since(active.phase_window_started_at),
            );
            active.phase_window_started_at = ready_at;
            EngineTokenTiming::ready(ready_at, phases)
        };
        match stop {
            Ok(Some(reason)) => {
                self.finish_active(idx, reason, Some((token, timing)));
            }
            Ok(None) if generated_count >= self.active[idx].req.sampling.max_tokens => {
                self.finish_active(idx, FinishReason::MaxTokens, Some((token, timing)));
            }
            Ok(None) => self.submit_token_delivery(idx, token, timing),
            Err(err) => {
                self.finish_one_with_error(idx, format!("{err:#}"), Some((token, timing)));
            }
        }
    }

    fn submit_token_delivery(&mut self, idx: usize, token: TokenId, timing: EngineTokenTiming) {
        let (key, sequence) = {
            let active = &mut self.active[idx];
            debug_assert_eq!(active.delivery_state, ActiveDeliveryState::Ready);
            let sequence = active.next_delivery_sequence;
            let Some(next_sequence) = sequence.checked_add(1) else {
                self.finish_one_with_error(
                    idx,
                    "response delivery sequence exhausted".to_string(),
                    Some((token, timing)),
                );
                return;
            };
            active.next_delivery_sequence = next_sequence;
            active.delivery_state = ActiveDeliveryState::InFlight { sequence };
            active.inflight_token_ready_at = Some(timing.ready_at);
            (active.delivery_key, sequence)
        };
        self.queue_delivery(
            key,
            DeliveryBatch::Token {
                token,
                timing,
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
        let (mut ready_indices, mut next_decode_generation) =
            self.ready_decode_selection_with_limit(max_rows);
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
                    if !starved_indices.is_empty() {
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
                        (ready_indices, next_decode_generation) =
                            self.ready_decode_selection_with_limit(max_rows);
                        if ready_indices.is_empty() {
                            self.refresh_snapshot();
                            return 0;
                        }
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
        let result = self
            .forward
            .forward_decode_with_phases(&mut slots, &sampling);
        let elapsed = started.elapsed();
        drop(slots);
        for &idx in &ready_indices {
            self.active[idx]
                .token_phase_durations
                .add_actor_decode(elapsed);
        }
        self.record_decode_forward_duration(elapsed, batch_len);
        self.snapshot.last_batch_size = batch_len;
        self.snapshot.current_batch_size = 0;
        self.refresh_snapshot();

        let output_tokens = match result {
            Ok(step) if step.tokens.len() == batch_len => {
                for &idx in &ready_indices {
                    self.active[idx]
                        .token_phase_durations
                        .add_backend(step.backend_phases);
                }
                step.tokens
            }
            Ok(step) => {
                self.finish_indices_with_error(
                    &ready_indices,
                    format!(
                        "batched decode returned {} rows for batch size {batch_len}",
                        step.tokens.len()
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
        if let Some(next_decode_generation) = next_decode_generation {
            self.next_decode_generation = next_decode_generation;
        }

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
        self.ready_decode_selection_with_limit(max_rows).0
    }

    fn ready_decode_selection_with_limit(&self, max_rows: usize) -> (Vec<usize>, Option<u64>) {
        let mut ready: Vec<(usize, u64)> = self
            .active
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
                    DecodeSlot::Real { .. } | DecodeSlot::Mock { .. } => {
                        Some((idx, active.delivery_key.generation))
                    }
                }
            })
            .collect();
        ready.sort_unstable_by_key(|&(_, generation)| generation);
        let limit = self.max_decode_batch.min(max_rows).min(ready.len());
        if limit == 0 {
            return (Vec::new(), None);
        }
        let start =
            ready.partition_point(|&(_, generation)| generation < self.next_decode_generation);
        let selected: Vec<(usize, u64)> = ready[start..]
            .iter()
            .chain(ready[..start].iter())
            .take(limit)
            .copied()
            .collect();
        let next_generation = selected
            .last()
            .map(|&(_, generation)| generation.checked_add(1).unwrap_or(0));
        let mut indices: Vec<usize> = selected.into_iter().map(|(idx, _)| idx).collect();
        // Slot and sampling vectors are materialized in active-index order.
        // Keep that order independent of the circular cohort's cursor.
        indices.sort_unstable();
        (indices, next_generation)
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
        preceding_token: Option<(TokenId, EngineTokenTiming)>,
    ) {
        let active = self.active.remove(idx);
        let key = active.delivery_key;
        let sequence = active.next_delivery_sequence;
        // Keep the last published snapshot conservative until model-owned
        // graph, recurrent-state, prefix, and KV resources are released by
        // finish_request. Publishing the now-empty scheduling vector here can
        // expose a false drained window to health readers.
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
                    action_tokens: active.action_tokens,
                    prefill_duration: output.prefill_duration,
                    decode_duration: output.decode_duration,
                    actor_queue_duration: active.actor_queue_duration,
                    actor_admission_duration: active.actor_admission_duration,
                    actor_prefill_wall_duration: active.first_token_ready_after_admission,
                    resident_prefill_used: active.resident_prefill_used,
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
        preceding_token: Option<(TokenId, EngineTokenTiming)>,
    ) {
        self.snapshot.total_errors += 1;
        let active = self.active.remove(idx);
        let key = active.delivery_key;
        let sequence = active.next_delivery_sequence;
        // discard_request is the terminal resource-ownership boundary. Leave
        // the removed row visible in the last published snapshot until that
        // cleanup completes.
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
        preceding_token: Option<(TokenId, EngineTokenTiming)>,
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
        while let Some(pending) = self.pending_exclusive_mutations.pop_front() {
            match pending {
                PendingExclusiveMutation::ResizeKv { reply, .. } => {
                    let _ = reply.send(Err(error.to_string()));
                }
                PendingExclusiveMutation::SwapAdapter { reply, .. } => {
                    let _ = reply.send(Err(error.to_string()));
                }
            }
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
        self.snapshot.active_resident_prefill = self
            .active
            .iter()
            .filter(|active| self.forward.resident_prefill_batch_required(&active.slot))
            .count();
        self.snapshot.active_decode = self
            .active
            .len()
            .saturating_sub(self.snapshot.active_prefill);
        self.snapshot.active_staged_requests = self
            .active
            .iter()
            .filter(|active| active.admission_lane == ActiveAdmissionLane::PrefillStaging)
            .count();
        self.snapshot.max_observed_active_requests = self
            .snapshot
            .max_observed_active_requests
            .max(self.active.len());
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

    #[test]
    fn profiled_decode_phases_preserve_sampling_and_readback_separately() {
        let mut phases = BackendPhaseDurations::default();
        observe_profiled_decode_phases(
            &mut phases,
            Some(Duration::from_millis(18)),
            Some(Duration::from_millis(7)),
            Some(Duration::from_millis(13)),
            Some(Duration::ZERO),
        );
        observe_profiled_decode_phases(
            &mut phases,
            Some(Duration::from_millis(4)),
            None,
            Some(Duration::from_millis(2)),
            Some(Duration::from_millis(5)),
        );
        assert_eq!(phases.sampling, Some(Duration::from_millis(22)));
        assert_eq!(phases.readback, Some(Duration::from_millis(7)));
        assert_eq!(phases.graph_capture, Some(Duration::from_millis(15)));
        assert_eq!(phases.graph_replay, Some(Duration::from_millis(5)));
    }

    #[derive(Default)]
    struct MockForward {
        calls: StdMutex<Vec<Vec<TokenId>>>,
        reusable_prefixes: bool,
        prefix_probe_calls: std::sync::atomic::AtomicUsize,
        reported_backend_phases: BackendPhaseDurations,
    }

    #[derive(Default)]
    struct PendingFirstTokenForward {
        calls: StdMutex<Vec<Vec<TokenId>>>,
        prepare_delay: Duration,
        eos_token: Option<TokenId>,
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
        ResidentPrefill(Vec<TokenId>),
        Discard(TokenId),
    }

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    enum ResidentBatchOutcome {
        #[default]
        Progress,
        Decline,
        Error,
        InvalidProgress,
    }

    #[derive(Default)]
    struct SyntheticPrefillForward {
        remaining: StdMutex<HashMap<TokenId, usize>>,
        pending_layers: StdMutex<HashMap<TokenId, usize>>,
        pending_token_widths: StdMutex<HashMap<TokenId, usize>>,
        events: StdMutex<Vec<SchedulingEvent>>,
        resident_rows: StdMutex<HashSet<TokenId>>,
        resident_batch_outcomes: StdMutex<VecDeque<ResidentBatchOutcome>>,
        resident_prefill_enabled: bool,
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

        fn resident_prefill_enabled(&self) -> bool {
            self.resident_prefill_enabled
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
                    tokens_scheduled: 0,
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
                tokens_scheduled: 0,
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
            let (reserved_tokens, tokens_scheduled) = if self.layers_per_chunk == 0 {
                (None, 0)
            } else {
                let remaining_tokens =
                    *self.remaining.lock().unwrap().get(&key).ok_or_else(|| {
                        anyhow::anyhow!("missing synthetic prefill state for {key}")
                    })?;
                let mut pending_token_widths = self.pending_token_widths.lock().unwrap();
                match pending_token_widths.entry(key) {
                    std::collections::hash_map::Entry::Occupied(entry) => (Some(*entry.get()), 0),
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        anyhow::ensure!(
                            max_tokens > 0,
                            "synthetic prefill received an empty token budget"
                        );
                        let tokens = remaining_tokens.min(max_tokens);
                        entry.insert(tokens);
                        (Some(tokens), tokens)
                    }
                }
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
                        tokens_scheduled,
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
            let tokens_scheduled = if reserved_tokens.is_some() {
                tokens_scheduled
            } else {
                tokens
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
                    tokens_scheduled,
                    tokens_processed: tokens,
                    layers_processed,
                })
            } else {
                Ok(RequestPreparation::Prefilling {
                    slot,
                    tokens_scheduled,
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

        fn remaining_prefill_tokens(&self, slot: &DecodeSlot) -> Option<usize> {
            self.remaining
                .lock()
                .unwrap()
                .get(&Self::slot_key(slot))
                .copied()
        }

        fn resident_prefill_batch_candidate(
            &self,
            slot: &DecodeSlot,
            _sampling: &SamplingParams,
        ) -> bool {
            if !self.resident_prefill_enabled {
                return false;
            }
            let key = Self::slot_key(slot);
            self.remaining.lock().unwrap().contains_key(&key)
                && !self.pending_layers.lock().unwrap().contains_key(&key)
                && !self.pending_token_widths.lock().unwrap().contains_key(&key)
        }

        fn resident_prefill_batch_required(&self, slot: &DecodeSlot) -> bool {
            self.resident_rows
                .lock()
                .unwrap()
                .contains(&Self::slot_key(slot))
        }

        fn advance_resident_prefill_batch(
            &self,
            slots: &mut [&mut DecodeSlot],
            _sampling: &[SamplingParams],
            cancels: &[CancelHandle],
        ) -> Result<Option<Vec<PrefillBatchProgress>>> {
            if !self.resident_prefill_enabled {
                return Ok(None);
            }
            anyhow::ensure!(
                slots.len() == cancels.len(),
                "synthetic resident metadata mismatch"
            );
            let keys: Vec<_> = slots.iter().map(|slot| Self::slot_key(slot)).collect();
            anyhow::ensure!(
                keys.iter()
                    .all(|key| self.remaining.lock().unwrap().contains_key(key)),
                "synthetic resident batch contains a completed row"
            );
            let outcome = self
                .resident_batch_outcomes
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_default();
            match outcome {
                ResidentBatchOutcome::Decline => return Ok(None),
                ResidentBatchOutcome::Error => {
                    anyhow::bail!("synthetic resident prefill failure")
                }
                ResidentBatchOutcome::Progress | ResidentBatchOutcome::InvalidProgress => {}
            }
            for cancel in cancels {
                anyhow::ensure!(
                    !cancel.is_cancelled(),
                    "synthetic resident prefill cancelled"
                );
            }

            let mut remaining = self.remaining.lock().unwrap();
            let mut resident_rows = self.resident_rows.lock().unwrap();
            let mut progress = Vec::with_capacity(keys.len());
            for key in &keys {
                let row_remaining = remaining
                    .get_mut(key)
                    .expect("synthetic resident row validated above");
                *row_remaining -= 1;
                let ready = *row_remaining == 0;
                if ready {
                    remaining.remove(key);
                }
                resident_rows.insert(*key);
                progress.push(PrefillBatchProgress {
                    tokens_scheduled: 1,
                    tokens_processed: usize::from(outcome != ResidentBatchOutcome::InvalidProgress),
                    layers_processed: 32,
                    ready,
                });
            }
            drop(resident_rows);
            drop(remaining);
            self.events
                .lock()
                .unwrap()
                .push(SchedulingEvent::ResidentPrefill(keys));
            Ok(Some(progress))
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
            let removed_resident = self.resident_rows.lock().unwrap().remove(&key);
            if removed_tokens || removed_layers || removed_width || removed_resident {
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

        fn forward_decode_with_phases(
            &self,
            slots: &mut [&mut DecodeSlot],
            sampling: &[SamplingParams],
        ) -> Result<DecodeStepOutput> {
            self.forward_decode(slots, sampling)
                .map(|tokens| DecodeStepOutput {
                    tokens: tokens
                        .into_iter()
                        .map(EngineSampledToken::untraced)
                        .collect(),
                    backend_phases: self.reported_backend_phases,
                })
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
            let mut slot = real_slot(next_token, true);
            let DecodeSlot::Real { state, .. } = &mut slot else {
                unreachable!()
            };
            state.capture_behavior_logprobs = req.capture_behavior_logprobs;
            state.next_token_logprob = req.capture_behavior_logprobs.then_some(-0.25);
            Ok(slot)
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

        fn forward_decode_with_metadata(
            &self,
            slots: &mut [&mut DecodeSlot],
            sampling: &[SamplingParams],
        ) -> Result<Vec<EngineSampledToken>> {
            let tokens = self.forward_decode(slots, sampling)?;
            Ok(tokens
                .into_iter()
                .zip(slots.iter())
                .map(|(token, slot)| match slot {
                    DecodeSlot::Real { state, .. } if state.capture_behavior_logprobs => {
                        EngineSampledToken::traced(token, -0.5)
                    }
                    DecodeSlot::Real { .. } => EngineSampledToken::untraced(token),
                    DecodeSlot::Mock { .. } | DecodeSlot::RealPrefill { .. } => unreachable!(),
                })
                .collect())
        }

        fn is_eos_token(&self, token: TokenId) -> Result<bool> {
            Ok(self.eos_token == Some(token))
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
            capture_behavior_logprobs: false,
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
                next_token_logprob: None,
                generated_tokens: Vec::new(),
                step_seed: None,
                capture_behavior_logprobs: false,
                registration: None,
                allocated_blocks: Vec::new(),
                prefill_duration: Duration::ZERO,
                decode_duration: Duration::ZERO,
                prompt_tokens: Vec::new(),
                block_size: 16,
                prefill_split_snapshot: None,
                rolling_snapshot: None,
                prefix_cache_registration_allowed: true,
                id: 0,
            },
            prefix_request: None,
            first_token_pending,
        }
    }

    fn assert_token_event(event: Option<EngineEvent>, expected: TokenId) {
        match event {
            Some(EngineEvent::Token { token, timing }) => {
                assert_eq!(token, expected);
                assert!(timing.ready_at <= Instant::now());
                assert!(timing.producer_delivered_at.is_some());
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
            ActorCycleIdleDiagnostics::default(),
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
        actor.waiting.push_back(QueuedRequest {
            req,
            delivery_key,
            enqueued_at: Instant::now(),
        });
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
        let initial_prefill_work_tokens = actor.forward.remaining_prefill_tokens(&slot);
        let action_tokens = req.capture_behavior_logprobs.then(Vec::new);
        let now = Instant::now();
        actor.active.push(ActiveRequest {
            req,
            delivery_key,
            delivery_state: ActiveDeliveryState::Ready,
            next_delivery_sequence: 0,
            admission_lane: ActiveAdmissionLane::Ordinary,
            initial_prefill_work_tokens,
            actor_queue_duration: Duration::ZERO,
            actor_admission_duration: Duration::ZERO,
            admitted_at: now,
            first_token_ready_after_admission: None,
            phase_window_started_at: now,
            token_phase_durations: TokenPhaseDurations::default(),
            inflight_token_ready_at: None,
            action_tokens,
            resident_prefill_used: false,
            slot,
        });
    }

    fn push_synthetic_prefill_rows(
        actor: &mut BatchingEngineActor,
        forward: &SyntheticPrefillForward,
        rows: &[(TokenId, usize)],
    ) -> Vec<mpsc::Receiver<EngineEvent>> {
        let mut receivers = Vec::with_capacity(rows.len());
        for &(key, remaining) in rows {
            forward.remaining.lock().unwrap().insert(key, remaining);
            let (response_tx, response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
            push_test_active(
                actor,
                request_with_tokens(vec![key; remaining.saturating_add(1)], 2),
                response_tx,
                SyntheticPrefillForward::mock_slot(key),
            );
            receivers.push(response_rx);
        }
        receivers
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

    #[test]
    fn disabled_resident_prefill_capability_never_attempts_the_route() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward::default());
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let _receivers = push_synthetic_prefill_rows(&mut actor, &forward, &[(11, 3), (22, 3)]);

        let mut budget = 2;
        assert!(!actor.snapshot.resident_prefill_enabled);
        assert_eq!(actor.run_resident_prefill_batch(&mut budget), None);
        assert_eq!(budget, 2);
        assert_eq!(actor.snapshot.total_resident_prefill_attempts, 0);
        assert_eq!(actor.snapshot.total_resident_prefill_forwards, 0);
        assert!(forward.events.lock().unwrap().is_empty());
        assert!(forward.resident_rows.lock().unwrap().is_empty());
    }

    #[test]
    fn resident_prefill_waits_for_every_row_to_reach_a_committed_boundary() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward {
            resident_prefill_enabled: true,
            ..SyntheticPrefillForward::default()
        });
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let _receivers = push_synthetic_prefill_rows(&mut actor, &forward, &[(11, 3), (22, 3)]);
        assert!(actor.snapshot.resident_prefill_enabled);
        forward.pending_layers.lock().unwrap().insert(22, 1);

        let mut budget = 2;
        assert_eq!(actor.run_resident_prefill_batch(&mut budget), None);
        assert_eq!(budget, 2);
        assert!(forward.events.lock().unwrap().is_empty());
        assert!(forward.resident_rows.lock().unwrap().is_empty());
        assert!(
            actor
                .active
                .iter()
                .all(|active| !active.resident_prefill_used)
        );

        forward.pending_layers.lock().unwrap().remove(&22);
        assert_eq!(actor.run_resident_prefill_batch(&mut budget), Some(true));
        assert_eq!(budget, 0);
        assert_eq!(
            forward.events.lock().unwrap().as_slice(),
            &[SchedulingEvent::ResidentPrefill(vec![11, 22])]
        );
        assert_eq!(actor.snapshot.active_resident_prefill, 2);
        assert_eq!(actor.snapshot.total_resident_prefill_attempts, 1);
        assert_eq!(actor.snapshot.total_resident_prefill_forwards, 1);
        assert_eq!(actor.snapshot.total_resident_prefill_rows, 2);
        assert!(
            actor
                .active
                .iter()
                .all(|active| active.resident_prefill_used)
        );
    }

    #[test]
    fn resident_prefill_rotates_bounded_cohorts_in_round_robin_order() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward {
            resident_prefill_enabled: true,
            ..SyntheticPrefillForward::default()
        });
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            5,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let _receivers = push_synthetic_prefill_rows(
            &mut actor,
            &forward,
            &[(1, 4), (2, 4), (3, 4), (4, 4), (5, 4)],
        );

        for _ in 0..3 {
            let mut budget = 2;
            assert_eq!(actor.run_resident_prefill_batch(&mut budget), Some(true));
            assert_eq!(budget, 0);
        }

        let events = forward.events.lock().unwrap().clone();
        assert_eq!(
            events,
            vec![
                SchedulingEvent::ResidentPrefill(vec![1, 2]),
                SchedulingEvent::ResidentPrefill(vec![1, 3]),
                SchedulingEvent::ResidentPrefill(vec![1, 5]),
            ]
        );
        assert_eq!(actor.next_prefill_index, 1);
        assert_eq!(actor.snapshot.total_prefill_forwards, 3);
        assert_eq!(actor.snapshot.total_prefill_tokens, 6);
        assert_eq!(actor.snapshot.total_resident_prefill_attempts, 3);
        assert_eq!(actor.snapshot.total_resident_prefill_forwards, 3);
        assert_eq!(actor.snapshot.total_resident_prefill_rows, 6);
        assert_eq!(actor.snapshot.last_resident_prefill_batch_size, 2);
        assert_eq!(actor.snapshot.max_resident_prefill_batch_size, 2);
    }

    #[test]
    fn resident_prefill_finishes_the_last_owned_row_as_a_single_row_tail() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward {
            resident_prefill_enabled: true,
            ..SyntheticPrefillForward::default()
        });
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let _receivers = push_synthetic_prefill_rows(&mut actor, &forward, &[(7, 1), (8, 2)]);

        let mut first_budget = 2;
        assert_eq!(
            actor.run_resident_prefill_batch(&mut first_budget),
            Some(true)
        );
        assert!(!forward.is_prefilling(&actor.active[0].slot));
        assert!(forward.is_prefilling(&actor.active[1].slot));
        assert!(forward.resident_prefill_batch_required(&actor.active[1].slot));

        let mut tail_budget = 1;
        assert_eq!(
            actor.run_resident_prefill_batch(&mut tail_budget),
            Some(true)
        );
        assert_eq!(tail_budget, 0);
        assert!(!forward.is_prefilling(&actor.active[1].slot));
        assert_eq!(
            forward.events.lock().unwrap().as_slice(),
            &[
                SchedulingEvent::ResidentPrefill(vec![7, 8]),
                SchedulingEvent::ResidentPrefill(vec![8]),
            ]
        );
    }

    #[test]
    fn resident_prefill_decline_is_mutation_free_and_falls_back_cleanly() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward {
            resident_prefill_enabled: true,
            ..SyntheticPrefillForward::default()
        });
        forward
            .resident_batch_outcomes
            .lock()
            .unwrap()
            .push_back(ResidentBatchOutcome::Decline);
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let _receivers = push_synthetic_prefill_rows(&mut actor, &forward, &[(31, 2), (32, 2)]);

        let mut budget = 2;
        assert_eq!(actor.run_resident_prefill_batch(&mut budget), None);
        assert_eq!(budget, 2);
        assert_eq!(forward.remaining.lock().unwrap().get(&31), Some(&2));
        assert_eq!(forward.remaining.lock().unwrap().get(&32), Some(&2));
        assert!(forward.resident_rows.lock().unwrap().is_empty());
        assert!(forward.events.lock().unwrap().is_empty());
        assert_eq!(actor.snapshot.total_resident_prefill_attempts, 1);
        assert_eq!(actor.snapshot.total_resident_prefill_initial_declines, 1);
        assert_eq!(actor.snapshot.total_resident_prefill_route_failures, 0);
    }

    #[test]
    fn resident_prefill_decline_after_entry_fails_closed() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward {
            resident_prefill_enabled: true,
            ..SyntheticPrefillForward::default()
        });
        forward.resident_batch_outcomes.lock().unwrap().extend([
            ResidentBatchOutcome::Progress,
            ResidentBatchOutcome::Decline,
        ]);
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let _receivers = push_synthetic_prefill_rows(&mut actor, &forward, &[(35, 3), (36, 3)]);

        let mut first_budget = 2;
        assert_eq!(
            actor.run_resident_prefill_batch(&mut first_budget),
            Some(true)
        );
        let mut second_budget = 2;
        assert_eq!(
            actor.run_resident_prefill_batch(&mut second_budget),
            Some(true)
        );

        assert!(actor.active.is_empty());
        assert_eq!(actor.snapshot.total_errors, 2);
        assert!(forward.remaining.lock().unwrap().is_empty());
        assert!(forward.resident_rows.lock().unwrap().is_empty());
        assert_eq!(actor.snapshot.total_resident_prefill_attempts, 2);
        assert_eq!(actor.snapshot.total_resident_prefill_forwards, 1);
        assert_eq!(actor.snapshot.total_resident_prefill_route_failures, 1);
        assert_eq!(
            forward.events.lock().unwrap().as_slice(),
            &[
                SchedulingEvent::ResidentPrefill(vec![35, 36]),
                SchedulingEvent::Discard(36),
                SchedulingEvent::Discard(35),
            ]
        );
    }

    #[test]
    fn resident_prefill_owned_row_losing_eligibility_is_discarded() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward {
            resident_prefill_enabled: true,
            ..SyntheticPrefillForward::default()
        });
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let _receivers = push_synthetic_prefill_rows(&mut actor, &forward, &[(37, 3), (38, 3)]);
        let mut first_budget = 2;
        assert_eq!(
            actor.run_resident_prefill_batch(&mut first_budget),
            Some(true)
        );
        forward.pending_layers.lock().unwrap().insert(37, 1);

        let mut second_budget = 2;
        assert_eq!(
            actor.run_resident_prefill_batch(&mut second_budget),
            Some(true)
        );

        assert_eq!(actor.active.len(), 1);
        assert_eq!(SyntheticPrefillForward::slot_key(&actor.active[0].slot), 38);
        assert_eq!(actor.snapshot.total_errors, 1);
        assert_eq!(actor.snapshot.total_resident_prefill_attempts, 1);
        assert_eq!(actor.snapshot.total_resident_prefill_route_failures, 1);
        assert!(!forward.resident_rows.lock().unwrap().contains(&37));
        assert!(forward.resident_rows.lock().unwrap().contains(&38));
        assert!(
            forward
                .events
                .lock()
                .unwrap()
                .contains(&SchedulingEvent::Discard(37))
        );
    }

    #[test]
    fn resident_prefill_error_discards_every_selected_row() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward {
            resident_prefill_enabled: true,
            ..SyntheticPrefillForward::default()
        });
        forward
            .resident_batch_outcomes
            .lock()
            .unwrap()
            .push_back(ResidentBatchOutcome::Error);
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let _receivers = push_synthetic_prefill_rows(&mut actor, &forward, &[(41, 2), (42, 2)]);

        let mut budget = 2;
        assert_eq!(actor.run_resident_prefill_batch(&mut budget), Some(true));
        assert_eq!(actor.active.len(), 0);
        assert_eq!(actor.snapshot.total_errors, 2);
        assert_eq!(actor.snapshot.total_resident_prefill_attempts, 1);
        assert_eq!(actor.snapshot.total_resident_prefill_route_failures, 1);
        assert_eq!(
            forward.events.lock().unwrap().as_slice(),
            &[SchedulingEvent::Discard(42), SchedulingEvent::Discard(41)]
        );
    }

    #[test]
    fn resident_prefill_invalid_progress_fails_closed_and_releases_rows() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward {
            resident_prefill_enabled: true,
            ..SyntheticPrefillForward::default()
        });
        forward
            .resident_batch_outcomes
            .lock()
            .unwrap()
            .push_back(ResidentBatchOutcome::InvalidProgress);
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let _receivers = push_synthetic_prefill_rows(&mut actor, &forward, &[(51, 2), (52, 2)]);

        let mut budget = 2;
        assert_eq!(actor.run_resident_prefill_batch(&mut budget), Some(true));
        assert_eq!(actor.active.len(), 0);
        assert_eq!(actor.snapshot.total_errors, 2);
        assert_eq!(actor.snapshot.total_resident_prefill_attempts, 1);
        assert_eq!(actor.snapshot.total_resident_prefill_route_failures, 1);
        assert!(forward.remaining.lock().unwrap().is_empty());
        assert!(forward.resident_rows.lock().unwrap().is_empty());
        assert_eq!(
            forward.events.lock().unwrap().as_slice(),
            &[
                SchedulingEvent::ResidentPrefill(vec![51, 52]),
                SchedulingEvent::Discard(52),
                SchedulingEvent::Discard(51),
            ]
        );
    }

    #[test]
    fn actor_constructor_consumes_only_resolved_admission_config() {
        let handle = BatchingEngineHandle::start_with_admission_config(
            Arc::new(MockForward::default()),
            8,
            BatchingActorAdmissionConfig {
                prefix_aware_admission: false,
                prefill_admission_quantum: 3,
                burst_prefill_admission: true,
            },
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            ResponseDeliveryPolicy::default(),
        );

        let snapshot = handle.cached_snapshot();
        assert_eq!(snapshot.max_prefill_admission_quantum, 3);
        assert_eq!(snapshot.max_prefill_staging_slots, 0);
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

        fn resize_kv_with_context(
            &self,
            target_blocks: usize,
            reason: KvResizeReason,
            barrier_wait: Duration,
        ) -> Result<usize> {
            let mut events = self.events.lock().unwrap();
            events.push(format!("resize:{target_blocks}"));
            events.push(format!("resize_reason:{}", reason.as_str()));
            events.push(format!(
                "resize_barrier_wait:{}",
                if barrier_wait.is_zero() {
                    "zero"
                } else {
                    "positive"
                }
            ));
            Ok(target_blocks)
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum TerminalCleanupMode {
        Finish,
        Discard,
    }

    struct GatedTerminalForward {
        inner: MockForward,
        mode: TerminalCleanupMode,
        started: std::sync::mpsc::Sender<()>,
        release: StdMutex<std::sync::mpsc::Receiver<()>>,
    }

    impl GatedTerminalForward {
        fn wait_if(&self, mode: TerminalCleanupMode) {
            if self.mode == mode {
                let _ = self.started.send(());
                self.release.lock().unwrap().recv().ok();
            }
        }
    }

    impl DecodeForward for GatedTerminalForward {
        fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
            self.inner.prepare_request(req)
        }

        fn forward_decode(
            &self,
            slots: &mut [&mut DecodeSlot],
            sampling: &[SamplingParams],
        ) -> Result<Vec<TokenId>> {
            if self.mode == TerminalCleanupMode::Discard {
                anyhow::bail!("synthetic decode failure")
            }
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
            self.wait_if(TerminalCleanupMode::Finish);
            self.inner.finish_request(slot, finish_reason)
        }

        fn discard_request(&self, slot: DecodeSlot) {
            self.wait_if(TerminalCleanupMode::Discard);
            self.inner.discard_request(slot);
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

    async fn assert_terminal_cleanup_remains_active(mode: TerminalCleanupMode) {
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let forward = Arc::new(GatedTerminalForward {
            inner: MockForward::default(),
            mode,
            started: started_tx,
            release: StdMutex::new(release_rx),
        });
        let handle = BatchingEngineHandle::start_with_options(forward, 1);
        let mut response = handle.enqueue(request(100, 1)).await.unwrap();

        started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("request did not enter terminal model cleanup");
        let during_cleanup = handle.cached_snapshot();
        assert_eq!(during_cleanup.active_decode, 1, "{during_cleanup:?}");
        assert_eq!(during_cleanup.active_prefill, 0, "{during_cleanup:?}");
        assert_eq!(during_cleanup.queue_depth, 0, "{during_cleanup:?}");

        release_tx.send(()).unwrap();
        let terminal = loop {
            match response.recv().await {
                Some(EngineEvent::Done { .. }) => break Ok(()),
                Some(EngineEvent::Error(error)) => break Err(error),
                Some(EngineEvent::Token { .. }) => {}
                None => panic!("engine closed before terminal delivery"),
            }
        };
        match mode {
            TerminalCleanupMode::Finish => assert_eq!(terminal, Ok(())),
            TerminalCleanupMode::Discard => assert!(
                terminal
                    .expect_err("discard cleanup must terminate with an error")
                    .contains("synthetic decode failure")
            ),
        }
        let drained = handle.snapshot().await.unwrap();
        assert_eq!(drained.active_decode, 0, "{drained:?}");
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn terminal_finish_remains_active_until_resource_cleanup_completes() {
        assert_terminal_cleanup_remains_active(TerminalCleanupMode::Finish).await;
    }

    #[tokio::test]
    async fn terminal_discard_remains_active_until_resource_cleanup_completes() {
        assert_terminal_cleanup_remains_active(TerminalCleanupMode::Discard).await;
    }

    #[tokio::test]
    async fn actor_queue_timing_includes_command_wait_behind_inflight_forward() {
        let events: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let (forward, release) = GatedForward::new(events);
        let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 1);
        let mut first = handle.enqueue(request(100, 1)).await.unwrap();

        let blocked_deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = handle.cached_snapshot();
            if snapshot.current_batch_size == 1 {
                break;
            }
            assert!(
                Instant::now() < blocked_deadline,
                "first request did not enter the gated forward: {snapshot:?}"
            );
            tokio::task::yield_now().await;
        }

        let mut second = handle.enqueue(request(200, 1)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        release.send(()).unwrap();
        release.send(()).unwrap();

        let first_output = loop {
            match first.recv().await {
                Some(EngineEvent::Done { output }) => break output,
                Some(EngineEvent::Token { .. }) => {}
                Some(EngineEvent::Error(error)) => panic!("first generation failed: {error}"),
                None => panic!("first generation closed before completion"),
            }
        };
        let second_output = loop {
            match second.recv().await {
                Some(EngineEvent::Done { output }) => break output,
                Some(EngineEvent::Token { .. }) => {}
                Some(EngineEvent::Error(error)) => panic!("second generation failed: {error}"),
                None => panic!("second generation closed before completion"),
            }
        };

        assert!(
            second_output.actor_queue_duration >= Duration::from_millis(20),
            "enqueue-to-admission timing must include actor command wait: first={:?} second={:?}",
            first_output.actor_queue_duration,
            second_output.actor_queue_duration
        );
        assert!(
            second_output.actor_prefill_wall_duration.is_some(),
            "the first sampled token must close admitted-prefill wall timing"
        );
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
    async fn kv_resize_waits_for_active_request_and_pauses_admission() {
        let events: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let (forward, release) = GatedForward::new(events.clone());
        let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 8);

        let mut active = handle.enqueue(request(100, 2)).await.unwrap();
        release.send(()).unwrap();
        assert!(matches!(
            active.recv().await,
            Some(EngineEvent::Token { .. })
        ));

        let resize_task = {
            let handle = handle.clone();
            tokio::spawn(async move { handle.resize_kv(17).await })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;
        let mut waiting = handle.enqueue(request(200, 1)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        assert!(!resize_task.is_finished(), "resize ran before active drain");
        let before_drain = events.lock().unwrap().clone();
        assert!(!before_drain.iter().any(|event| event == "resize:17"));
        assert!(!before_drain.iter().any(|event| event == "prepare:200"));

        release.send(()).unwrap();
        release.send(()).unwrap();
        assert_eq!(resize_task.await.unwrap().unwrap(), 17);

        for response in [&mut active, &mut waiting] {
            loop {
                match response.recv().await {
                    Some(EngineEvent::Done { .. }) => break,
                    Some(EngineEvent::Token { .. }) => {}
                    Some(EngineEvent::Error(error)) => panic!("request failed: {error}"),
                    None => panic!("request channel closed before completion"),
                }
            }
        }

        let log = events.lock().unwrap().clone();
        let resize_idx = log
            .iter()
            .position(|event| event == "resize:17")
            .expect("resize ran");
        assert!(
            log.iter().any(|event| event == "resize_reason:maintenance"),
            "public resize must carry a bounded maintenance reason: {log:?}"
        );
        assert!(
            log.iter()
                .any(|event| event == "resize_barrier_wait:positive"),
            "resize must carry its actor-barrier wait: {log:?}"
        );
        let waiting_prepare_idx = log
            .iter()
            .position(|event| event == "prepare:200")
            .expect("waiting request admitted");
        assert!(
            resize_idx < waiting_prepare_idx,
            "resize must precede admission: {log:?}"
        );
        assert_eq!(
            log[..resize_idx]
                .iter()
                .filter(|event| event.as_str() == "decode")
                .count(),
            2,
            "the active request must finish before resize: {log:?}"
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
        let batching = BatchingConfig::default().resolve(
            BatchingBackendPolicy {
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                actor_prefill_tile_alignment_required: false,
            },
            8,
        );
        let handle = BatchingEngineHandle::start_with_policy(
            forward,
            8,
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            batching.prefix_aware_admission.enabled,
            batching.prefill_admission_quantum.effective,
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
        let terminal = tokio::time::timeout(Duration::from_secs(1), response.recv())
            .await
            .expect("Done must follow after the final-token slot is released");
        let Some(EngineEvent::Done { output }) = terminal else {
            panic!("expected terminal generation output, got {terminal:?}");
        };
        assert_eq!(output.token_ids, vec![111]);
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(output.finish_reason, FinishReason::MaxTokens);
        assert!(
            output.actor_prefill_wall_duration.is_some(),
            "the first sampled token must close actor prefill wall timing"
        );

        let settled = handle.snapshot().await.unwrap();
        assert_eq!(settled.response_delivery_pending_terminal, 0, "{settled:?}");
        assert_eq!(settled.response_delivery_backpressured, 0, "{settled:?}");
        assert_eq!(settled.response_stall_evictions, 0, "{settled:?}");
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn cooperative_actor_cycle_idle_is_accounted_and_stop_responsive() {
        let batching = BatchingConfig::default().resolve(
            BatchingBackendPolicy {
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                actor_prefill_tile_alignment_required: false,
            },
            1,
        );
        let actor_cycle_idle = ActorCycleIdleDiagnostics {
            milliseconds: 500,
            source: ConfigValueSource::ConfigFile,
            enabled: true,
            command_poll_milliseconds: ACTOR_CYCLE_IDLE_COMMAND_POLL_MS,
        };
        let handle = BatchingEngineHandle::start_with_actor_runtime_config(
            Arc::new(MockForward::default()),
            1,
            batching.actor_admission_config(),
            actor_cycle_idle,
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            ResponseDeliveryPolicy::default(),
        );
        let mut response = handle.enqueue(request(1, 8)).await.unwrap();
        assert!(matches!(
            response.recv().await,
            Some(EngineEvent::Token { .. })
        ));

        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let snapshot = handle.cached_snapshot();
            if snapshot.actor_cycle_idle_active {
                assert_eq!(snapshot.actor_cycle_idle_ms, 500);
                assert_eq!(
                    snapshot.actor_cycle_idle_source,
                    ConfigValueSource::ConfigFile
                );
                assert_eq!(snapshot.actor_cycle_idle_count, 1);
                break;
            }
            assert!(Instant::now() < deadline, "actor never entered idle");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        let stop_started = Instant::now();
        tokio::time::timeout(Duration::from_millis(100), handle.stop())
            .await
            .expect("stop must interrupt cooperative idle")
            .unwrap();
        assert!(stop_started.elapsed() < Duration::from_millis(100));

        let snapshot = handle.cached_snapshot();
        assert!(!snapshot.actor_cycle_idle_active);
        assert_eq!(snapshot.actor_cycle_idle_count, 1);
        assert!(snapshot.total_actor_cycle_idle_ms > 0.0);
        assert!(snapshot.total_actor_cycle_idle_ms < 100.0, "{snapshot:?}");
        assert_eq!(
            snapshot.max_actor_cycle_idle_ms,
            snapshot.total_actor_cycle_idle_ms
        );
    }

    #[tokio::test]
    async fn cooperative_actor_cycle_idle_reaches_the_next_token_timing() {
        let batching = BatchingConfig::default().resolve(
            BatchingBackendPolicy {
                use_decode_width_prefill_admission: false,
                burst_prefill_admission: false,
                actor_prefill_tile_alignment_required: false,
            },
            1,
        );
        let actor_cycle_idle = ActorCycleIdleDiagnostics {
            milliseconds: 20,
            source: ConfigValueSource::ConfigFile,
            enabled: true,
            command_poll_milliseconds: ACTOR_CYCLE_IDLE_COMMAND_POLL_MS,
        };
        let handle = BatchingEngineHandle::start_with_actor_runtime_config(
            Arc::new(MockForward::default()),
            1,
            batching.actor_admission_config(),
            actor_cycle_idle,
            BatchTokenBudget::default(),
            PrefillTokenBudget::default(),
            PrefillLayerBudget::default(),
            ResponseDeliveryPolicy::default(),
        );
        let mut response = handle.enqueue(request(1, 2)).await.unwrap();
        assert!(matches!(
            response.recv().await,
            Some(EngineEvent::Token { .. })
        ));
        let second = tokio::time::timeout(Duration::from_secs(1), response.recv())
            .await
            .expect("second token must arrive")
            .expect("second token event must exist");
        let EngineEvent::Token { timing, .. } = second else {
            panic!("expected second token, got {second:?}");
        };
        assert!(
            timing.phases_since_previous_token.actor_cycle_idle >= Duration::from_millis(20),
            "{timing:?}"
        );
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

        actor.submit_token_delivery(
            0,
            111,
            EngineTokenTiming::ready(Instant::now(), TokenPhaseDurations::default()),
        );
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

        actor.submit_token_delivery(
            0,
            111,
            EngineTokenTiming::ready(Instant::now(), TokenPhaseDurations::default()),
        );
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
                timing: EngineTokenTiming::ready(Instant::now(), TokenPhaseDurations::default(),),
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

        actor.submit_token_delivery(
            0,
            111,
            EngineTokenTiming::ready(Instant::now(), TokenPhaseDurations::default()),
        );
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
        actor.submit_token_delivery(
            0,
            111,
            EngineTokenTiming::ready(ready_at, TokenPhaseDurations::default()),
        );
        let after = Instant::now();
        match response_rx.blocking_recv() {
            Some(EngineEvent::Token { token, timing }) => {
                assert_eq!(token, 111);
                assert!(timing.ready_at >= before);
                assert!(timing.ready_at <= after);
                assert!(timing.producer_delivered_at.is_some());
            }
            other => panic!("expected timed token, got {other:?}"),
        }
    }

    #[test]
    fn decode_backend_phases_follow_owned_ready_tokens_only() {
        let mut backend_phases = BackendPhaseDurations::default();
        backend_phases.observe_gpu_lock_wait(Duration::from_millis(7));
        backend_phases.observe_synchronization(Duration::from_millis(11));
        backend_phases.observe_graph_capture(Duration::from_millis(13));
        backend_phases.observe_graph_replay(Duration::from_millis(17));
        let forward = Arc::new(MockForward {
            reported_backend_phases: backend_phases,
            ..MockForward::default()
        });
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let mut actor = test_actor(
            rx,
            forward,
            8,
            false,
            1,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let (response_a_tx, mut response_a_rx) = mpsc::channel(2);
        push_test_active(
            &mut actor,
            request(101, 2),
            response_a_tx,
            DecodeSlot::Mock {
                next_token: 101,
                generated_tokens: Vec::new(),
            },
        );
        let (response_b_tx, mut response_b_rx) = mpsc::channel(2);
        push_test_active(
            &mut actor,
            request(201, 2),
            response_b_tx,
            DecodeSlot::Mock {
                next_token: 201,
                generated_tokens: Vec::new(),
            },
        );
        let (response_c_tx, _response_c_rx) = mpsc::channel(2);
        push_test_active(
            &mut actor,
            request(301, 2),
            response_c_tx,
            DecodeSlot::Mock {
                next_token: 301,
                generated_tokens: Vec::new(),
            },
        );
        actor.active[2].delivery_state = ActiveDeliveryState::InFlight { sequence: 0 };
        let non_ready_phases = actor.active[2].token_phase_durations;

        assert_eq!(actor.run_decode_batch(), 2);
        for (expected_token, response_rx) in [(111, &mut response_a_rx), (211, &mut response_b_rx)]
        {
            match response_rx.blocking_recv() {
                Some(EngineEvent::Token { token, timing }) => {
                    assert_eq!(token, expected_token);
                    assert_eq!(timing.phases_since_previous_token.backend, backend_phases);
                }
                other => panic!("expected timed token, got {other:?}"),
            }
        }
        assert_eq!(actor.active[2].token_phase_durations, non_ready_phases);
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
    fn kv_shrink_failure_leaves_block_manager_unchanged() {
        let mut block_manager = BlockManager::new(8, 16);
        let before = block_manager.clone();
        let mut attempted = None;

        let error = resize_block_manager_transaction(&mut block_manager, 8, 4, |achieved| {
            attempted = Some(achieved);
            anyhow::bail!("injected physical shrink failure")
        })
        .expect_err("injected shrink must fail");

        assert_eq!(attempted, Some(4));
        assert!(
            error
                .to_string()
                .contains("injected physical shrink failure")
        );
        assert_eq!(block_manager, before);
    }

    #[test]
    fn kv_grow_failure_leaves_block_manager_unchanged() {
        let mut block_manager = BlockManager::new(4, 16);
        let before = block_manager.clone();
        let mut attempted = None;

        let error = resize_block_manager_transaction(&mut block_manager, 4, 8, |achieved| {
            attempted = Some(achieved);
            anyhow::bail!("injected physical grow failure")
        })
        .expect_err("injected grow must fail");

        assert_eq!(attempted, Some(8));
        assert!(error.to_string().contains("injected physical grow failure"));
        assert_eq!(block_manager, before);
    }

    #[test]
    fn successful_kv_resize_publishes_matching_logical_capacity() {
        let mut block_manager = BlockManager::new(8, 16);
        let mut physical_blocks = 8;

        let shrunk = resize_block_manager_transaction(&mut block_manager, 8, 4, |achieved| {
            physical_blocks = achieved;
            Ok(())
        })
        .expect("shrink transaction");
        assert_eq!(shrunk, 4);
        assert_eq!(physical_blocks, 4);
        assert_eq!(block_manager.num_blocks(), physical_blocks);
        assert_eq!(block_manager.target_usable(), physical_blocks);

        let grown =
            resize_block_manager_transaction(&mut block_manager, physical_blocks, 10, |achieved| {
                physical_blocks = achieved;
                Ok(())
            })
            .expect("grow transaction");
        assert_eq!(grown, 10);
        assert_eq!(physical_blocks, 10);
        assert_eq!(block_manager.num_blocks(), physical_blocks);
        assert_eq!(block_manager.target_usable(), physical_blocks);
    }

    #[test]
    fn kv_resize_rejects_preexisting_capacity_mismatch_before_physical_work() {
        let mut block_manager = BlockManager::new(8, 16);
        let before = block_manager.clone();
        let mut physical_called = false;

        let error = resize_block_manager_transaction(&mut block_manager, 7, 4, |_| {
            physical_called = true;
            Ok(())
        })
        .expect_err("capacity mismatch must fail");

        assert!(!physical_called);
        assert!(
            error
                .to_string()
                .contains("KV capacity mismatch before resize")
        );
        assert_eq!(block_manager, before);
    }

    #[test]
    fn max_decode_batch_default_is_backend_aware() {
        let resolve = |deterministic: bool,
                       configured: Option<usize>,
                       source: ConfigValueSource,
                       policy: Option<DecodeExecutionPolicy>| {
            resolve_decode_runtime_config(
                DeterministicInference::new(deterministic, source),
                MaxDecodeBatch::new(configured, source).unwrap(),
                policy,
                BatchTokenBudget::default(),
            )
        };
        let vulkan_policy =
            DecodeExecutionPolicy::for_backend("vulkan", kiln_tensor::Device::Vulkan(0));
        let metal_policy =
            DecodeExecutionPolicy::for_backend("metal", kiln_tensor::Device::Metal(0));
        assert_eq!(
            resolve(false, None, ConfigValueSource::Default, None)
                .max_decode_batch
                .effective,
            8
        );
        let cuda_policy = DecodeExecutionPolicy::for_backend("cuda", kiln_tensor::Device::Cuda(0));
        assert_eq!(cuda_policy.max_decode_batch, 8);
        assert_eq!(
            resolve(false, None, ConfigValueSource::Default, Some(cuda_policy))
                .max_decode_batch
                .effective,
            8
        );
        assert_eq!(
            resolve(false, None, ConfigValueSource::Default, Some(vulkan_policy))
                .max_decode_batch
                .effective,
            64
        );
        assert_eq!(
            resolve(false, None, ConfigValueSource::Default, Some(metal_policy))
                .max_decode_batch
                .effective,
            8
        );
        let configured = resolve(false, Some(24), ConfigValueSource::Environment, None);
        assert_eq!(configured.max_decode_batch.effective, 24);
        assert_eq!(
            configured.max_decode_batch.effective_source,
            DecodeBatchEffectiveSource::Environment
        );
        assert_eq!(
            resolve(
                false,
                Some(24),
                ConfigValueSource::ConfigFile,
                Some(vulkan_policy)
            )
            .max_decode_batch
            .effective,
            24
        );
        let deterministic = resolve(true, Some(24), ConfigValueSource::Environment, None);
        assert_eq!(deterministic.max_decode_batch.effective, 1);
        assert_eq!(
            deterministic.max_decode_batch.effective_source,
            DecodeBatchEffectiveSource::Deterministic
        );
        assert_eq!(
            resolve(
                true,
                None,
                ConfigValueSource::Environment,
                Some(vulkan_policy)
            )
            .max_decode_batch
            .effective,
            1
        );
    }

    #[test]
    fn combined_token_budget_constrains_configured_decode_width() {
        let resolved = resolve_decode_runtime_config(
            DeterministicInference::default(),
            MaxDecodeBatch::new(Some(256), ConfigValueSource::ConfigFile).unwrap(),
            None,
            BatchTokenBudget::new(128, ConfigValueSource::Environment).unwrap(),
        );
        assert_eq!(resolved.max_decode_batch.configured, Some(256));
        assert_eq!(resolved.max_decode_batch.effective, 128);
        assert_eq!(
            resolved.max_decode_batch.effective_source,
            DecodeBatchEffectiveSource::MaxBatchTokens
        );
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
        let mut output = vec![EngineSampledToken::untraced(0); slots.len()];

        let (decode_indices, decode_params) =
            collect_ready_decode_indices(&mut slots, &sampling, &mut output).unwrap();

        assert_eq!(
            output,
            vec![
                EngineSampledToken::untraced(101),
                EngineSampledToken::untraced(0),
                EngineSampledToken::untraced(303),
                EngineSampledToken::untraced(0),
            ]
        );
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
    fn short_prefill_staging_is_bounded_and_preserves_ordinary_fifo_capacity() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(SyntheticPrefillForward::default());
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        assert_eq!(actor.max_prefill_staging_slots, 2);
        assert_eq!(actor.max_active_requests, 4);
        assert_eq!(actor.snapshot.max_prefill_staging_priority_burst, 4);
        let staging_entry_token_limit = actor.prefill_staging_entry_token_limit();
        let ordinary_short_tail = staging_entry_token_limit - actor.max_prefill_tokens_per_cycle;

        let mut receivers = Vec::new();
        for key in [10, 20] {
            let (response_tx, response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
            let mut prompt = vec![1; staging_entry_token_limit + 80];
            *prompt.last_mut().unwrap() = key;
            queue_test_request(&mut actor, request_with_tokens(prompt, 4), response_tx);
            receivers.push(response_rx);
        }
        actor.admit_waiting();
        assert_eq!(actor.active_admission_lane_counts(), (2, 0));

        for (key, prompt_len) in [
            (30, staging_entry_token_limit + 80),
            (40, ordinary_short_tail),
            (50, 100),
        ] {
            let (response_tx, response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
            let mut prompt = vec![1; prompt_len];
            *prompt.last_mut().unwrap() = key;
            queue_test_request(&mut actor, request_with_tokens(prompt, 4), response_tx);
            receivers.push(response_rx);
        }
        actor.admit_waiting();

        assert_eq!(actor.active_admission_lane_counts(), (2, 2));
        assert_eq!(actor.active.len(), 4);
        assert_eq!(actor.waiting.len(), 1);
        assert_eq!(actor.waiting[0].req.prompt_tokens.last(), Some(&30));
        assert_eq!(actor.snapshot.total_prefill_staging_admissions, 2);
        assert_eq!(actor.snapshot.max_observed_active_requests, 4);

        let ordinary = actor.active.remove(0);
        forward.discard_request(ordinary.slot);
        actor.admit_waiting();
        assert_eq!(actor.active_admission_lane_counts(), (2, 2));
        assert!(actor.waiting.is_empty());
        let admitted_long = actor
            .active
            .iter()
            .find(|active| active.req.prompt_tokens.last() == Some(&30))
            .expect("a freed ordinary slot must admit the FIFO long prompt");
        assert_eq!(admitted_long.admission_lane, ActiveAdmissionLane::Ordinary);

        actor.fail_all("test complete");
        drop(receivers);
    }

    #[test]
    fn prefill_staging_is_disabled_for_burst_refill_and_width_one() {
        for (max_decode_batch, burst_refill) in [(8, true), (1, false)] {
            let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
            let actor = test_actor(
                rx,
                Arc::new(MockForward::default()),
                max_decode_batch,
                false,
                4,
                burst_refill,
                ResponseDeliveryPolicy::default(),
            );
            assert_eq!(actor.max_prefill_staging_slots, 0);
            assert_eq!(actor.max_active_requests, max_decode_batch);
            assert_eq!(actor.snapshot.max_prefill_staging_priority_burst, 0);
        }
    }

    #[test]
    fn decode_cohorts_rotate_across_staged_ready_rows() {
        let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let forward = Arc::new(MockForward::default());
        let mut actor = test_actor(
            rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let mut receivers = Vec::new();
        for key in 1..=4 {
            let (response_tx, response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
            push_test_active(
                &mut actor,
                request(key, 3),
                response_tx,
                DecodeSlot::Mock {
                    next_token: key,
                    generated_tokens: Vec::new(),
                },
            );
            receivers.push(response_rx);
        }
        for active in &mut actor.active[2..] {
            active.admission_lane = ActiveAdmissionLane::PrefillStaging;
        }
        actor.refresh_snapshot();

        assert_eq!(actor.run_decode_batch(), 2);
        settle_active_deliveries(&mut actor);
        assert_eq!(actor.run_decode_batch(), 2);
        assert_eq!(
            forward.calls.lock().unwrap().as_slice(),
            &[vec![1, 2], vec![3, 4]]
        );
        assert_eq!(actor.snapshot.max_observed_batch_size, 2);

        actor.fail_all("test complete");
        drop(receivers);
    }

    #[test]
    fn staged_prefills_rotate_on_priority_turns_without_hiding_ordinary_rows() {
        const LONG_A: TokenId = 60_001;
        const LONG_B: TokenId = 60_002;
        const LONG_C: TokenId = 60_003;
        const STAGED_A: TokenId = 61_001;
        const STAGED_B: TokenId = 61_002;
        const STAGED_C: TokenId = 61_003;

        let forward = Arc::new(SyntheticPrefillForward {
            layers_per_chunk: 8,
            ..SyntheticPrefillForward::default()
        });
        let (_command_tx, command_rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let mut actor = test_actor(
            command_rx,
            forward.clone(),
            3,
            false,
            3,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let mut receivers = Vec::new();
        for (key, tokens, lane) in [
            (LONG_A, 1_024, ActiveAdmissionLane::Ordinary),
            (LONG_B, 1_024, ActiveAdmissionLane::Ordinary),
            (LONG_C, 1_024, ActiveAdmissionLane::Ordinary),
            (STAGED_A, 128, ActiveAdmissionLane::PrefillStaging),
            (STAGED_B, 128, ActiveAdmissionLane::PrefillStaging),
            (STAGED_C, 128, ActiveAdmissionLane::PrefillStaging),
        ] {
            let req = request_with_tokens(vec![key; tokens], 1);
            let RequestPreparation::Prefilling { slot, .. } = forward
                .prepare_request_chunked(&req, 64)
                .expect("initialize synthetic staged prefill")
            else {
                panic!("synthetic prompt unexpectedly became ready")
            };
            let (response_tx, response_rx) = mpsc::channel(8);
            push_test_active(&mut actor, req, response_tx, slot);
            actor.active.last_mut().unwrap().admission_lane = lane;
            receivers.push(response_rx);
        }

        for _ in 0..15 {
            assert!(actor.run_prefill_budget(64));
        }

        let layer_order: Vec<TokenId> = forward
            .events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                SchedulingEvent::PrefillLayers { key, .. } => Some(*key),
                _ => None,
            })
            .collect();
        assert_eq!(
            layer_order,
            vec![
                STAGED_A, STAGED_B, STAGED_C, STAGED_A, LONG_A, STAGED_B, STAGED_C, STAGED_A,
                STAGED_B, LONG_B, STAGED_C, STAGED_A, STAGED_B, STAGED_C, LONG_C,
            ]
        );
        assert_eq!(actor.snapshot.total_short_prefill_priority_forwards, 12);
        assert_eq!(actor.snapshot.total_prefill_staging_priority_forwards, 12);
        assert_eq!(actor.snapshot.total_errors, 0);

        actor.fail_all("test complete");
        drop(receivers);
    }

    #[test]
    fn staged_priority_owns_the_entry_quantum_beyond_the_ordinary_short_tail() {
        const LONG_A: TokenId = 62_001;
        const LONG_B: TokenId = 62_002;
        const BOUNDARY: TokenId = 62_003;

        let forward = Arc::new(SyntheticPrefillForward::default());
        let (_command_tx, command_rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let mut actor = test_actor(
            command_rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        let ordinary_short_tail = actor
            .max_prefill_tokens_per_cycle
            .saturating_mul(SHORT_PREFILL_PRIORITY_MAX_CHUNKS);
        let staging_entry_token_limit = actor.prefill_staging_entry_token_limit();
        let boundary_tokens = ordinary_short_tail + 20;
        assert!(boundary_tokens <= staging_entry_token_limit);
        let long_tokens = staging_entry_token_limit.saturating_mul(2);
        let mut receivers = Vec::new();
        for (key, tokens, lane) in [
            (LONG_A, long_tokens, ActiveAdmissionLane::Ordinary),
            (LONG_B, long_tokens, ActiveAdmissionLane::Ordinary),
            (
                BOUNDARY,
                boundary_tokens,
                ActiveAdmissionLane::PrefillStaging,
            ),
        ] {
            let req = request_with_tokens(vec![key; tokens], 1);
            let RequestPreparation::Prefilling { slot, .. } = forward
                .prepare_request_chunked(&req, 64)
                .expect("initialize synthetic staging-boundary prefill")
            else {
                panic!("synthetic prompt unexpectedly became ready")
            };
            let (response_tx, response_rx) = mpsc::channel(8);
            push_test_active(&mut actor, req, response_tx, slot);
            actor.active.last_mut().unwrap().admission_lane = lane;
            receivers.push(response_rx);
        }

        assert_eq!(
            staging_entry_token_limit,
            ordinary_short_tail + actor.max_prefill_tokens_per_cycle
        );
        assert_eq!(actor.select_prefill_index(64), Some((2, true)));
        assert_eq!(actor.snapshot.total_prefill_staging_priority_forwards, 1);

        actor.active[2].admission_lane = ActiveAdmissionLane::Ordinary;
        actor.next_prefill_index = 0;
        actor.short_prefill_priority_cursor = 0;
        assert_eq!(actor.select_prefill_index(64), Some((0, false)));
        assert_eq!(actor.snapshot.total_prefill_staging_priority_forwards, 1);

        actor.fail_all("test complete");
        drop(receivers);
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
        assert_eq!(snapshot.max_decode_batch, 8);
        assert_eq!(snapshot.last_batch_size, 2);
        assert_eq!(snapshot.max_observed_batch_size, 2);
        assert_eq!(snapshot.total_decode_forwards, 1);
        assert_eq!(snapshot.total_batched_decode_forwards, 1);
        assert_eq!(snapshot.total_decode_rows, 2);
        assert_eq!(snapshot.total_decode_tokens, 2);
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn behavior_trace_records_sampled_actions_from_prefill_and_decode() {
        let handle = BatchingEngineHandle::start_with_options(
            Arc::new(PendingFirstTokenForward::default()),
            8,
        );
        let mut req = request(100, 2);
        req.capture_behavior_logprobs = true;
        let mut rx = handle.enqueue(req).await.unwrap();

        assert_token_event(rx.recv().await, 110);
        assert_token_event(rx.recv().await, 120);
        let Some(EngineEvent::Done { output }) = rx.recv().await else {
            panic!("traced request did not finish")
        };
        assert_eq!(output.token_ids, vec![110, 120]);
        assert_eq!(
            output.action_tokens,
            Some(vec![
                EngineActionToken {
                    generated_index: 0,
                    token_id: 110,
                    source: EngineActionTokenSource::Sampled,
                    behavior_logprob: Some(-0.25),
                },
                EngineActionToken {
                    generated_index: 1,
                    token_id: 120,
                    source: EngineActionTokenSource::Sampled,
                    behavior_logprob: Some(-0.5),
                },
            ])
        );
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn behavior_trace_records_terminal_eos_as_a_sampled_action() {
        let handle = BatchingEngineHandle::start_with_options(
            Arc::new(PendingFirstTokenForward {
                eos_token: Some(110),
                ..Default::default()
            }),
            8,
        );
        let mut req = request(100, 2);
        req.capture_behavior_logprobs = true;
        let mut rx = handle.enqueue(req).await.unwrap();

        let Some(EngineEvent::Done { output }) = rx.recv().await else {
            panic!("EOS-traced request did not finish")
        };
        assert_eq!(output.finish_reason, FinishReason::Eos);
        assert!(output.token_ids.is_empty());
        assert_eq!(output.completion_tokens, 1);
        assert_eq!(
            output.action_tokens,
            Some(vec![EngineActionToken {
                generated_index: 0,
                token_id: 110,
                source: EngineActionTokenSource::Sampled,
                behavior_logprob: Some(-0.25),
            }])
        );
        let snapshot = handle.snapshot().await.unwrap();
        assert_eq!(snapshot.total_decode_tokens, 1);
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn behavior_trace_marks_every_controller_close_token_as_forced() {
        let handle = BatchingEngineHandle::start_with_options(
            Arc::new(PendingFirstTokenForward::default()),
            8,
        );
        let mut req = request(100, 2);
        req.capture_behavior_logprobs = true;
        req.sampling.thinking_budget =
            Some(ThinkingBudget::new(Some(0), None, 2, vec![90, 91]).unwrap());
        let mut rx = handle.enqueue(req).await.unwrap();

        assert_token_event(rx.recv().await, 90);
        assert_token_event(rx.recv().await, 91);
        let Some(EngineEvent::Done { output }) = rx.recv().await else {
            panic!("forced traced request did not finish")
        };
        assert_eq!(output.token_ids, vec![90, 91]);
        assert_eq!(
            output.action_tokens,
            Some(vec![
                EngineActionToken {
                    generated_index: 0,
                    token_id: 90,
                    source: EngineActionTokenSource::Forced,
                    behavior_logprob: None,
                },
                EngineActionToken {
                    generated_index: 1,
                    token_id: 91,
                    source: EngineActionTokenSource::Forced,
                    behavior_logprob: None,
                },
            ])
        );
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn behavior_trace_fails_closed_when_forward_omits_probability() {
        let handle = BatchingEngineHandle::start_with_options(Arc::new(MockForward::default()), 8);
        let mut req = request(100, 1);
        req.capture_behavior_logprobs = true;
        let mut rx = handle.enqueue(req).await.unwrap();

        let Some(EngineEvent::Error(error)) = rx.recv().await else {
            panic!("trace without a probability did not fail")
        };
        assert!(error.contains("produced no valid probability"), "{error}");
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
            eos_token: None,
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
            ActorCycleIdleDiagnostics::default(),
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
            enqueued_at: Instant::now(),
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
                enqueued_at: Instant::now(),
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
    fn retained_prefill_charges_its_token_width_only_when_selected() {
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
        assert_eq!(actor.snapshot.total_prefill_tokens, 0);
        assert_eq!(actor.snapshot.total_prefill_layers, 4);
        assert_eq!(actor.snapshot.total_prefill_forwards, 1);

        assert!(
            actor.run_prefill_budget(32),
            "a retained chunk must resume without competing for new-token budget"
        );
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

    #[test]
    fn short_prefill_priority_is_bounded_by_round_robin_service() {
        const LONG_A: TokenId = 10_001;
        const LONG_B: TokenId = 10_002;
        const LONG_C: TokenId = 10_003;
        const LONG_D: TokenId = 10_004;
        const SHORT: TokenId = 20_000;

        let forward = Arc::new(SyntheticPrefillForward {
            layers_per_chunk: 8,
            ..SyntheticPrefillForward::default()
        });
        let (_command_tx, command_rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let mut actor = test_actor(
            command_rx,
            forward.clone(),
            5,
            false,
            5,
            false,
            ResponseDeliveryPolicy::default(),
        );
        for (key, tokens) in [
            (LONG_A, 1_024),
            (LONG_B, 1_024),
            (LONG_C, 1_024),
            (LONG_D, 1_024),
            (SHORT, 128),
        ] {
            let req = request_with_tokens(vec![key; tokens], 1);
            let RequestPreparation::Prefilling { slot, .. } = forward
                .prepare_request_chunked(&req, 64)
                .expect("initialize synthetic mixed prefill")
            else {
                panic!("synthetic prompt unexpectedly became ready")
            };
            let (response_tx, _response_rx) = mpsc::channel(8);
            push_test_active(&mut actor, req, response_tx, slot);
        }

        for _ in 0..5 {
            assert!(actor.run_prefill_budget(64));
        }

        let layer_order = || -> Vec<TokenId> {
            forward
                .events
                .lock()
                .unwrap()
                .iter()
                .filter_map(|event| match event {
                    SchedulingEvent::PrefillLayers { key, .. } => Some(*key),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(layer_order(), vec![SHORT, SHORT, LONG_A, SHORT, SHORT]);
        assert_eq!(actor.snapshot.total_short_prefill_priority_forwards, 4);
        assert_eq!(actor.snapshot.total_errors, 0);
        assert!(
            !forward.is_prefilling(&actor.active[4].slot),
            "the short row must become decode-ready within five bounded dispatches"
        );

        for _ in 0..3 {
            assert!(actor.run_prefill_budget(64));
        }
        assert_eq!(
            layer_order(),
            vec![SHORT, SHORT, LONG_A, SHORT, SHORT, LONG_B, LONG_C, LONG_D]
        );
        assert!(!forward.is_prefilling(&actor.active[4].slot));
        for active in &actor.active[..4] {
            assert!(forward.is_prefilling(&active.slot));
        }

        actor.fail_all("test complete");
    }

    #[test]
    fn shorter_round_robin_row_uses_priority_without_a_half_remainder_gap() {
        const LONG: TokenId = 40_001;
        const SHORT: TokenId = 40_002;

        let forward = Arc::new(SyntheticPrefillForward {
            layers_per_chunk: 8,
            ..SyntheticPrefillForward::default()
        });
        let (_command_tx, command_rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let mut actor = test_actor(
            command_rx,
            forward.clone(),
            2,
            false,
            2,
            false,
            ResponseDeliveryPolicy::default(),
        );
        for (key, tokens) in [(SHORT, 238), (LONG, 432)] {
            let req = request_with_tokens(vec![key; tokens], 1);
            let RequestPreparation::Prefilling { slot, .. } = forward
                .prepare_request_chunked(&req, 64)
                .expect("initialize synthetic prefill")
            else {
                panic!("synthetic prompt unexpectedly became ready")
            };
            let (response_tx, _response_rx) = mpsc::channel(8);
            push_test_active(&mut actor, req, response_tx, slot);
        }

        actor.next_prefill_index = 0;
        actor.short_prefill_priority_cursor = 0;
        assert_eq!(actor.select_prefill_index(64), Some((0, true)));
        assert_eq!(actor.snapshot.total_short_prefill_priority_forwards, 1);

        actor.fail_all("test complete");
    }

    #[test]
    fn equal_prefill_work_never_consumes_the_short_tail_lane() {
        const KEYS: [TokenId; 5] = [30_001, 30_002, 30_003, 30_004, 30_005];

        let forward = Arc::new(SyntheticPrefillForward {
            layers_per_chunk: 8,
            ..SyntheticPrefillForward::default()
        });
        let (_command_tx, command_rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let mut actor = test_actor(
            command_rx,
            forward.clone(),
            KEYS.len(),
            false,
            KEYS.len(),
            false,
            ResponseDeliveryPolicy::default(),
        );
        for key in KEYS {
            let req = request_with_tokens(vec![key; 128], 1);
            let RequestPreparation::Prefilling { slot, .. } = forward
                .prepare_request_chunked(&req, 64)
                .expect("initialize equal synthetic prefill")
            else {
                panic!("synthetic prompt unexpectedly became ready")
            };
            let (response_tx, _response_rx) = mpsc::channel(8);
            push_test_active(&mut actor, req, response_tx, slot);
        }

        for _ in 0..20 {
            assert!(actor.run_prefill_budget(64));
        }

        let layer_order: Vec<TokenId> = forward
            .events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                SchedulingEvent::PrefillLayers { key, .. } => Some(*key),
                _ => None,
            })
            .collect();
        assert_eq!(layer_order, KEYS.repeat(4));
        assert_eq!(actor.snapshot.total_short_prefill_priority_forwards, 0);
        assert_eq!(actor.snapshot.total_errors, 0);
        assert!(
            actor
                .active
                .iter()
                .all(|active| { !forward.is_prefilling(&active.slot) })
        );

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
    async fn ignore_eos_treats_eos_as_an_ordinary_token_until_max_tokens() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward.clone(), 8);
        let mut req = request(0, 2);
        req.sampling.ignore_eos = true;

        let mut rx = handle.enqueue(req).await.unwrap();

        assert_token_event(rx.recv().await, 10);
        assert_token_event(rx.recv().await, 20);
        assert!(matches!(
            rx.recv().await,
            Some(EngineEvent::Done {
                output: BatchedGenerationOutput {
                    completion_tokens: 2,
                    token_ids,
                    finish_reason: FinishReason::MaxTokens,
                    ..
                }
            }) if token_ids == vec![10, 20]
        ));

        let calls = forward.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![vec![0], vec![10]]);
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

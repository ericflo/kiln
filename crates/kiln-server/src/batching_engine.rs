//! Real-model batching engine actor scaffolding.
//!
//! Phase 1 keeps the actor in `kiln-server` so HTTP request routing, prefix
//! cache ownership, GPU coordination, and metrics can be wired incrementally.
//! The production seam still uses the row-loop `model_forward_paged_batched_decode`
//! API in `kiln-model`; follow-up phases can replace that seam with a true
//! layer-wise batched CUDA path without changing HTTP routing.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use kiln_core::block::BlockManager;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_model::{
    CancelHandle, FinishReason, GenerationOutput, ModelRunner, PagedBatchedDecodeState,
    PagedKvCache, PagedPrefixReuse,
};
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

use crate::state::{GpuCoordinationLock, RealPrefixCache};

const DEFAULT_ENGINE_CHANNEL: usize = 1024;
const DEFAULT_RESPONSE_CHANNEL: usize = 64;
const DEFAULT_MAX_DECODE_BATCH: usize = 8;
const DEFAULT_PREFIX_AWARE_ADMISSION: bool = true;

fn blocks_needed_for_tokens(num_tokens: usize, block_size: usize) -> usize {
    num_tokens.div_ceil(block_size)
}

/// Resolve the actor's `max_decode_batch` from the environment, falling back
/// to [`DEFAULT_MAX_DECODE_BATCH`] when `KILN_MAX_DECODE_BATCH` is unset or
/// cannot be parsed as a positive integer. The actor caps `active.len()` at
/// this value, so it is the effective concurrent-decode width — bumping it
/// without a corresponding scheduler-side improvement is a no-op past the
/// natural single-stream-row-loop ceiling.
fn env_max_decode_batch() -> usize {
    std::env::var("KILN_MAX_DECODE_BATCH")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_DECODE_BATCH)
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

/// Decide whether multi-row decode steps should be issued one-row-at-a-time
/// instead of as a single batched forward. The current Vulkan batched-decode
/// path materializes per-layer state cats/narrows that cost more than running
/// N single-row forwards through the resident-decode fast path, so rowwise is
/// the throughput win on Vulkan today. CUDA / Metal keep batched decode by
/// default. `KILN_BATCH_DECODE_ROWWISE` (0/1) overrides the auto choice.
fn default_rowwise_decode(backend_name: Option<&'static str>) -> bool {
    if let Ok(raw) = std::env::var("KILN_BATCH_DECODE_ROWWISE") {
        return !matches!(
            raw.trim(),
            "0" | "false" | "FALSE" | "off" | "OFF" | "no" | "NO"
        );
    }
    matches!(backend_name, Some("vulkan"))
}

#[derive(Debug, Clone)]
pub struct EngineRequest {
    pub request_id: Uuid,
    pub prompt_tokens: Vec<TokenId>,
    pub sampling: SamplingParams,
    pub adapter: Option<String>,
    pub cancel: CancelHandle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    Token(TokenId),
    Done { output: BatchedGenerationOutput },
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchedGenerationOutput {
    pub text: String,
    pub token_ids: Vec<TokenId>,
    pub finish_reason: FinishReason,
    pub completion_tokens: usize,
}

fn completion_usage_tokens(visible_token_count: usize, finish_reason: &FinishReason) -> usize {
    visible_token_count + usize::from(matches!(finish_reason, FinishReason::Eos))
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchingEngineSnapshot {
    pub accepting: bool,
    pub queue_depth: usize,
    pub active_decode: usize,
    pub current_batch_size: usize,
    pub last_batch_size: usize,
    pub last_forward_ms: f64,
    pub last_prefill_ms: f64,
    pub total_decode_tokens: u64,
    pub total_prefill_tokens: u64,
    pub total_errors: u64,
    pub adapter_groups_waiting: usize,
    pub prefix_deferred_waiting: usize,
    pub prefix_admission_deferrals: u64,
}

pub enum DecodeSlot {
    Mock {
        next_token: TokenId,
        generated_tokens: Vec<TokenId>,
    },
    Real {
        state: PagedBatchedDecodeState,
        hit_entry_id: Option<u64>,
        adapter: Option<String>,
        first_token_pending: bool,
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
            DecodeSlot::Mock { .. } => anyhow::bail!("mock slot sent to real decode forward"),
        }
    }

    Ok((decode_indices, decode_params))
}

pub trait DecodeForward: Send + Sync + 'static {
    fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot>;
    fn can_reuse_as_strict_prefix(&self, _prompt_token_len: usize) -> bool {
        false
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
    ) -> Result<GenerationOutput>;
    fn discard_request(&self, _slot: DecodeSlot) {}
}

pub struct RealDecodeForward {
    runner: Arc<RwLock<ModelRunner>>,
    block_manager: Arc<Mutex<BlockManager>>,
    paged_cache: Arc<PagedKvCache>,
    prefix_cache: Arc<Mutex<RealPrefixCache>>,
    gpu_lock: GpuCoordinationLock,
    // When set, multi-row decode steps are dispatched as a loop of single-row
    // forwards instead of one batched forward. Default-on for Vulkan because
    // the resident-decode fast path (single-row, GPU-resident state/scratch)
    // is dramatically faster than the legacy batched-row path that materializes
    // per-layer cats/narrows for every step. Two concurrent playground
    // requests measured ~0.16 tok/s aggregate batched vs ~14 tok/s rowwise.
    rowwise_decode: bool,
}

impl RealDecodeForward {
    pub fn new(
        runner: Arc<RwLock<ModelRunner>>,
        block_manager: Arc<Mutex<BlockManager>>,
        paged_cache: Arc<PagedKvCache>,
        prefix_cache: Arc<Mutex<RealPrefixCache>>,
        gpu_lock: GpuCoordinationLock,
    ) -> Self {
        let rowwise_decode = default_rowwise_decode(runner.read().ok().map(|g| g.backend_name()));
        Self {
            runner,
            block_manager,
            paged_cache,
            prefix_cache,
            gpu_lock,
            rowwise_decode,
        }
    }

    pub fn with_rowwise_decode(mut self, rowwise: bool) -> Self {
        self.rowwise_decode = rowwise;
        self
    }

    fn release_hit(&self, hit_entry_id: Option<u64>) {
        if let Some(entry_id) = hit_entry_id {
            if let Ok(mut cache) = self.prefix_cache.lock() {
                cache.release_hit(entry_id);
            }
        }
    }

    fn free_uncached_blocks(
        &self,
        output: &mut kiln_model::PrefixCachedGenerationOutput,
        adapter: Option<String>,
    ) {
        let registration = output.registration.take();
        let extra_registrations = std::mem::take(&mut output.extra_registrations);
        let allocated_blocks = std::mem::take(&mut output.allocated_blocks);
        let mut retained_blocks: Vec<u32> = Vec::new();
        let mut evicted_blocks: Vec<u32> = Vec::new();
        {
            let mut cache = self.prefix_cache.lock().unwrap();
            if let Some(registration) = registration {
                let outcome = cache.register(adapter.clone(), registration);
                retained_blocks.extend(outcome.retained_blocks);
                evicted_blocks.extend(outcome.evicted_blocks);
            }
            // Register the extra (block-aligned, strict-prefix-reusable)
            // entries after the prompt entry so that, when the cache is at
            // capacity, LRU eviction prefers the older prompt-only entry.
            // Multi-turn agentic workloads care about these landing — they
            // are the entries the next turn's prefix lookup will hit on a
            // length beyond what the bare prompt registration could match.
            for reg in extra_registrations {
                let outcome = cache.register(adapter.clone(), reg);
                retained_blocks.extend(outcome.retained_blocks);
                evicted_blocks.extend(outcome.evicted_blocks);
            }
        }

        // The two registrations may share blocks (the extended entry is a
        // strict superset of the original on disjoint prompts, but the
        // outcome filters out blocks already refcounted by an earlier
        // entry, so dedup matters here for correctness of the freed set).
        let retained_set: std::collections::HashSet<u32> = retained_blocks.iter().copied().collect();

        let mut blocks_to_free: Vec<u32> = allocated_blocks
            .into_iter()
            .filter(|block_id| !retained_set.contains(block_id))
            .collect();
        // Evictions from the second registration can list blocks that the
        // first registration's retained set contained — strip those before
        // freeing so we don't hand a still-held block back to the manager.
        evicted_blocks.retain(|block_id| !retained_set.contains(block_id));
        blocks_to_free.extend(evicted_blocks);
        debug_assert!(
            blocks_to_free
                .iter()
                .all(|id| !retained_set.contains(id)),
            "blocks_to_free overlaps retained_blocks: free={blocks_to_free:?} retained={retained_set:?}",
        );
        debug_assert!({
            let mut seen = std::collections::HashSet::with_capacity(blocks_to_free.len());
            blocks_to_free.iter().all(|id| seen.insert(*id))
        });
        if !blocks_to_free.is_empty() {
            let mut bm_guard = self.block_manager.lock().unwrap();
            bm_guard.free_all(&blocks_to_free);
        }
    }

    fn grow_ready_decode_slots(&self, slots: &mut [&mut DecodeSlot]) -> Result<()> {
        let block_size = self
            .block_manager
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock block manager: {e}"))?
            .block_size();

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
            let mut bm_guard = self
                .block_manager
                .lock()
                .map_err(|e| anyhow::anyhow!("failed to lock block manager: {e}"))?;
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
}

impl DecodeForward for RealDecodeForward {
    fn can_reuse_as_strict_prefix(&self, prompt_token_len: usize) -> bool {
        self.prefix_cache
            .lock()
            .map(|cache| cache.can_register_strict_prefix_len(prompt_token_len))
            .unwrap_or(false)
    }

    fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
        let _gpu_guard = self.gpu_lock.read().unwrap();
        let hit = {
            let mut cache = self.prefix_cache.lock().unwrap();
            if cache.is_enabled() {
                cache.lookup(&req.adapter, &req.prompt_tokens)?
            } else {
                None
            }
        };
        let hit_entry_id = hit.as_ref().map(|hit| hit.entry_id);
        let cached_prefix = hit.map(|hit| PagedPrefixReuse {
            cached_tokens: hit.cached_tokens,
            block_ids: hit.block_ids,
            linear_state: hit.linear_state,
            next_token: hit.next_token,
        });

        let prepared = self
            .runner
            .read()
            .unwrap()
            .prepare_paged_batched_decode_with_prefix_cache(
                &req.prompt_tokens,
                &req.sampling,
                self.block_manager.as_ref(),
                self.paged_cache.as_ref(),
                cached_prefix,
                Some(&req.cancel),
            );

        match prepared {
            Ok(state) => Ok(DecodeSlot::Real {
                state,
                hit_entry_id,
                adapter: req.adapter.clone(),
                first_token_pending: true,
            }),
            Err(err) => {
                self.release_hit(hit_entry_id);
                Err(err)
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
            let _gpu_guard = self.gpu_lock.read().unwrap();
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
            let runner_guard = self.runner.read().unwrap();
            let next_tokens: Vec<TokenId> = if self.rowwise_decode && row_refs.len() > 1 {
                // Dispatch one single-row forward per active slot. On Vulkan
                // each row then hits the resident-decode fast path
                // (`gated_deltanet_forward_decode_resident_b1` + cached weight
                // buffers + GPU-resident scratch), which the multi-row path
                // does not. Even though the rows now serialize on the GPU
                // queue, total throughput is dramatically higher because the
                // single-row kernel is hand-tuned for decode.
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
                tokens
            } else {
                runner_guard.paged_batched_decode_step(
                    &mut row_refs,
                    &decode_params,
                    self.paged_cache.as_ref(),
                )?
            };
            for (idx, token) in decode_indices.into_iter().zip(next_tokens) {
                output[idx] = token;
            }
        }

        Ok(output)
    }

    fn is_eos_token(&self, token: TokenId) -> Result<bool> {
        Ok(self.runner.read().unwrap().is_eos_token(token))
    }

    fn stop_reason_after_emit(
        &self,
        generated_tokens: &[TokenId],
        sampling: &SamplingParams,
    ) -> Result<Option<FinishReason>> {
        if let Some(stop) = self
            .runner
            .read()
            .unwrap()
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
    ) -> Result<GenerationOutput> {
        let DecodeSlot::Real {
            state,
            hit_entry_id,
            adapter,
            ..
        } = slot
        else {
            anyhow::bail!("mock slot sent to real finish_request");
        };
        self.release_hit(hit_entry_id);
        let mut output = self
            .runner
            .read()
            .unwrap()
            .finish_paged_batched_decode(state, finish_reason)?;
        self.free_uncached_blocks(&mut output, adapter);
        Ok(output.output)
    }

    fn discard_request(&self, slot: DecodeSlot) {
        if let DecodeSlot::Real {
            state,
            hit_entry_id,
            adapter,
            ..
        } = slot
        {
            self.release_hit(hit_entry_id);
            let mut output = kiln_model::PrefixCachedGenerationOutput {
                output: GenerationOutput {
                    text: String::new(),
                    token_ids: state.generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                },
                registration: None,
                extra_registrations: Vec::new(),
                allocated_blocks: state.allocated_blocks,
                prefill_duration: state.prefill_duration,
                decode_duration: state.decode_duration,
            };
            self.free_uncached_blocks(&mut output, adapter);
        }
    }
}

#[derive(Clone)]
pub struct BatchingEngineHandle {
    tx: mpsc::Sender<EngineCommand>,
}

impl BatchingEngineHandle {
    pub fn start(forward: Arc<dyn DecodeForward>) -> Self {
        Self::start_with_options(forward, env_max_decode_batch())
    }

    pub fn start_with_options(forward: Arc<dyn DecodeForward>, max_decode_batch: usize) -> Self {
        Self::start_with_policy(
            forward,
            max_decode_batch.max(1),
            env_prefix_aware_admission(),
        )
    }

    fn start_with_policy(
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        prefix_aware_admission: bool,
    ) -> Self {
        let (tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let actor =
            BatchingEngineActor::new(rx, forward, max_decode_batch.max(1), prefix_aware_admission);
        thread::Builder::new()
            .name("kiln-batching-engine".to_string())
            .spawn(move || actor.run())
            .expect("spawn batching engine actor");
        Self { tx }
    }

    pub async fn enqueue(&self, req: EngineRequest) -> Result<mpsc::Receiver<EngineEvent>> {
        let (response_tx, response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
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
            .map_err(|_| anyhow::anyhow!("batching engine stopped before snapshot"))
    }
}

enum EngineCommand {
    Enqueue {
        req: EngineRequest,
        response_tx: mpsc::Sender<EngineEvent>,
    },
    Cancel {
        request_id: Uuid,
    },
    Drain {
        reply: oneshot::Sender<()>,
    },
    Stop {
        reply: oneshot::Sender<()>,
    },
    Snapshot {
        reply: oneshot::Sender<BatchingEngineSnapshot>,
    },
}

struct QueuedRequest {
    req: EngineRequest,
    response_tx: mpsc::Sender<EngineEvent>,
}

struct ActiveRequest {
    req: EngineRequest,
    response_tx: mpsc::Sender<EngineEvent>,
    slot: DecodeSlot,
}

struct BatchingEngineActor {
    rx: mpsc::Receiver<EngineCommand>,
    forward: Arc<dyn DecodeForward>,
    waiting: VecDeque<QueuedRequest>,
    active: Vec<ActiveRequest>,
    accepting: bool,
    stopped: bool,
    max_decode_batch: usize,
    prefix_aware_admission: bool,
    snapshot: BatchingEngineSnapshot,
}

impl BatchingEngineActor {
    fn new(
        rx: mpsc::Receiver<EngineCommand>,
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
        prefix_aware_admission: bool,
    ) -> Self {
        Self {
            rx,
            forward,
            waiting: VecDeque::new(),
            active: Vec::new(),
            accepting: true,
            stopped: false,
            max_decode_batch,
            prefix_aware_admission,
            snapshot: BatchingEngineSnapshot {
                accepting: true,
                ..BatchingEngineSnapshot::default()
            },
        }
    }

    fn run(mut self) {
        while !self.stopped {
            if self.active.is_empty() && self.waiting.is_empty() {
                match self.rx.blocking_recv() {
                    Some(cmd) => self.handle_command(cmd),
                    None => break,
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
            self.admit_waiting();
            if !self.active.is_empty() {
                self.run_decode_batch();
                continue;
            }

            thread::sleep(Duration::from_millis(1));
        }

        self.fail_all("batching engine stopped");
    }

    fn drain_commands(&mut self) {
        while let Ok(cmd) = self.rx.try_recv() {
            self.handle_command(cmd);
        }
    }

    fn handle_command(&mut self, cmd: EngineCommand) {
        match cmd {
            EngineCommand::Enqueue { req, response_tx } => {
                if self.accepting {
                    self.waiting.push_back(QueuedRequest { req, response_tx });
                } else {
                    let _ = response_tx.blocking_send(EngineEvent::Error(
                        "batching engine is draining".to_string(),
                    ));
                }
            }
            EngineCommand::Cancel { request_id } => self.cancel(request_id),
            EngineCommand::Drain { reply } => {
                self.accepting = false;
                self.refresh_snapshot();
                let _ = reply.send(());
            }
            EngineCommand::Stop { reply } => {
                self.accepting = false;
                self.stopped = true;
                self.refresh_snapshot();
                let _ = reply.send(());
            }
            EngineCommand::Snapshot { reply } => {
                self.refresh_snapshot();
                let _ = reply.send(self.snapshot.clone());
            }
        }
    }

    fn cancel(&mut self, request_id: Uuid) {
        self.waiting.retain(|queued| {
            let keep = queued.req.request_id != request_id;
            if !keep {
                queued.req.cancel.cancel();
                let _ = queued
                    .response_tx
                    .blocking_send(EngineEvent::Error("request cancelled".to_string()));
            }
            keep
        });

        let mut idx = 0;
        while idx < self.active.len() {
            if self.active[idx].req.request_id == request_id {
                let active = self.active.remove(idx);
                active.req.cancel.cancel();
                self.forward.discard_request(active.slot);
                let _ = active
                    .response_tx
                    .blocking_send(EngineEvent::Error("request cancelled".to_string()));
            } else {
                idx += 1;
            }
        }
    }

    fn admit_waiting(&mut self) {
        while self.active.len() < self.max_decode_batch && !self.waiting.is_empty() {
            let Some(waiting_idx) = self
                .waiting
                .iter()
                .position(|queued| !self.should_defer_for_active_prefix(queued))
            else {
                self.snapshot.prefix_admission_deferrals =
                    self.snapshot.prefix_admission_deferrals.saturating_add(
                        self.waiting
                            .iter()
                            .filter(|queued| self.should_defer_for_active_prefix(queued))
                            .count() as u64,
                    );
                break;
            };
            let queued = self
                .waiting
                .remove(waiting_idx)
                .expect("waiting index selected from VecDeque");
            let started = Instant::now();
            match self.forward.prepare_request(&queued.req) {
                Ok(slot) => {
                    self.snapshot.last_prefill_ms = started.elapsed().as_secs_f64() * 1000.0;
                    self.snapshot.total_prefill_tokens += queued.req.prompt_tokens.len() as u64;
                    self.active.push(ActiveRequest {
                        req: queued.req,
                        response_tx: queued.response_tx,
                        slot,
                    });
                }
                Err(err) => {
                    self.snapshot.total_errors += 1;
                    let _ = queued
                        .response_tx
                        .blocking_send(EngineEvent::Error(err.to_string()));
                }
            }
        }
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

    fn run_decode_batch(&mut self) {
        let batch_len = self.active.len().min(self.max_decode_batch);
        let sampling: Vec<SamplingParams> = self
            .active
            .iter()
            .take(batch_len)
            .map(|active| active.req.sampling.clone())
            .collect();
        let mut slots: Vec<&mut DecodeSlot> = self
            .active
            .iter_mut()
            .take(batch_len)
            .map(|active| &mut active.slot)
            .collect();

        self.snapshot.current_batch_size = batch_len;
        let started = Instant::now();
        let result = self.forward.forward_decode(&mut slots, &sampling);
        self.snapshot.last_forward_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.snapshot.last_batch_size = batch_len;
        self.snapshot.current_batch_size = 0;

        let output_tokens = match result {
            Ok(tokens) if tokens.len() == batch_len => tokens,
            Ok(tokens) => {
                self.snapshot.total_errors += batch_len as u64;
                self.finish_batch_with_error(
                    batch_len,
                    format!(
                        "batched decode returned {} rows for batch size {batch_len}",
                        tokens.len()
                    ),
                );
                self.refresh_snapshot();
                return;
            }
            Err(err) => {
                self.snapshot.total_errors += batch_len as u64;
                self.finish_batch_with_error(batch_len, format!("{err:#}"));
                self.refresh_snapshot();
                return;
            }
        };

        for idx in (0..batch_len).rev() {
            if idx >= self.active.len() {
                continue;
            }
            let token = output_tokens[idx];
            match self.forward.is_eos_token(token) {
                Ok(true) => {
                    self.finish_active(idx, FinishReason::Eos);
                    continue;
                }
                Ok(false) => {}
                Err(err) => {
                    self.finish_one_with_error(idx, err.to_string());
                    continue;
                }
            }

            let generated_count = match self.forward.accept_token(&mut self.active[idx].slot, token)
            {
                Ok(count) => count,
                Err(err) => {
                    self.finish_one_with_error(idx, err.to_string());
                    continue;
                }
            };
            self.snapshot.total_decode_tokens += 1;

            if self.active[idx]
                .response_tx
                .blocking_send(EngineEvent::Token(token))
                .is_err()
            {
                self.forward.discard_request(self.active.remove(idx).slot);
                continue;
            }

            let generated_tokens = self.generated_tokens_for(idx).to_vec();
            let sampling = self.active[idx].req.sampling.clone();
            match self
                .forward
                .stop_reason_after_emit(&generated_tokens, &sampling)
            {
                Ok(Some(reason)) => {
                    self.finish_active(idx, reason);
                    continue;
                }
                Ok(None) if generated_count >= sampling.max_tokens => {
                    self.finish_active(idx, FinishReason::MaxTokens);
                    continue;
                }
                Ok(None) => {}
                Err(err) => {
                    self.finish_one_with_error(idx, err.to_string());
                    continue;
                }
            }
        }
        self.refresh_snapshot();
    }

    fn generated_tokens_for(&self, idx: usize) -> &[TokenId] {
        match &self.active[idx].slot {
            DecodeSlot::Mock {
                generated_tokens, ..
            } => generated_tokens,
            DecodeSlot::Real { state, .. } => &state.generated_tokens,
        }
    }

    fn finish_active(&mut self, idx: usize, finish_reason: FinishReason) {
        let active = self.active.remove(idx);
        match self.forward.finish_request(active.slot, finish_reason) {
            Ok(output) => {
                let completion_tokens =
                    completion_usage_tokens(output.token_ids.len(), &output.finish_reason);
                let _ = active.response_tx.blocking_send(EngineEvent::Done {
                    output: BatchedGenerationOutput {
                        text: output.text,
                        token_ids: output.token_ids,
                        finish_reason: output.finish_reason,
                        completion_tokens,
                    },
                });
            }
            Err(err) => {
                self.snapshot.total_errors += 1;
                let _ = active
                    .response_tx
                    .blocking_send(EngineEvent::Error(err.to_string()));
            }
        }
    }

    fn finish_one_with_error(&mut self, idx: usize, error: String) {
        self.snapshot.total_errors += 1;
        let active = self.active.remove(idx);
        self.forward.discard_request(active.slot);
        let _ = active.response_tx.blocking_send(EngineEvent::Error(error));
    }

    fn finish_batch_with_error(&mut self, batch_len: usize, error: String) {
        for _ in 0..batch_len.min(self.active.len()) {
            let active = self.active.remove(0);
            self.forward.discard_request(active.slot);
            let _ = active
                .response_tx
                .blocking_send(EngineEvent::Error(error.clone()));
        }
    }

    fn fail_all(&mut self, error: &str) {
        while let Some(queued) = self.waiting.pop_front() {
            queued.req.cancel.cancel();
            let _ = queued
                .response_tx
                .blocking_send(EngineEvent::Error(error.to_string()));
        }
        for active in self.active.drain(..) {
            active.req.cancel.cancel();
            self.forward.discard_request(active.slot);
            let _ = active
                .response_tx
                .blocking_send(EngineEvent::Error(error.to_string()));
        }
    }

    fn refresh_snapshot(&mut self) {
        self.snapshot.accepting = self.accepting;
        self.snapshot.queue_depth = self.waiting.len();
        self.snapshot.active_decode = self.active.len();
        self.snapshot.adapter_groups_waiting = usize::from(!self.waiting.is_empty());
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
    use kiln_model::LinearAttentionState;
    use std::sync::Mutex as StdMutex;

    #[derive(Default)]
    struct MockForward {
        calls: StdMutex<Vec<Vec<TokenId>>>,
        reusable_prefixes: bool,
    }

    impl DecodeForward for MockForward {
        fn can_reuse_as_strict_prefix(&self, prompt_token_len: usize) -> bool {
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
                    DecodeSlot::Real { .. } => unreachable!(),
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
        ) -> Result<GenerationOutput> {
            let DecodeSlot::Mock {
                generated_tokens, ..
            } = slot
            else {
                unreachable!();
            };
            Ok(GenerationOutput {
                text: String::new(),
                token_ids: generated_tokens,
                finish_reason,
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
            hit_entry_id: None,
            adapter: None,
            first_token_pending,
        }
    }

    #[test]
    fn decode_capacity_block_count_rounds_up() {
        assert_eq!(blocks_needed_for_tokens(0, 16), 0);
        assert_eq!(blocks_needed_for_tokens(1, 16), 1);
        assert_eq!(blocks_needed_for_tokens(16, 16), 1);
        assert_eq!(blocks_needed_for_tokens(17, 16), 2);
    }

    #[test]
    fn default_rowwise_decode_picks_vulkan_unless_overridden() {
        // Snapshot + restore the env var so other tests that share this
        // process see the original value.
        let prior = std::env::var("KILN_BATCH_DECODE_ROWWISE").ok();
        // SAFETY: tests in this crate that touch this env var must not run in
        // parallel; cargo defaults to serial within a single test binary.
        unsafe {
            std::env::remove_var("KILN_BATCH_DECODE_ROWWISE");
        }
        assert!(default_rowwise_decode(Some("vulkan")));
        assert!(!default_rowwise_decode(Some("cuda")));
        assert!(!default_rowwise_decode(Some("metal")));
        assert!(!default_rowwise_decode(None));
        unsafe {
            std::env::set_var("KILN_BATCH_DECODE_ROWWISE", "0");
        }
        assert!(!default_rowwise_decode(Some("vulkan")));
        assert!(!default_rowwise_decode(Some("cuda")));
        unsafe {
            std::env::set_var("KILN_BATCH_DECODE_ROWWISE", "1");
        }
        assert!(default_rowwise_decode(Some("cuda")));
        assert!(default_rowwise_decode(None));
        match prior {
            Some(v) => unsafe { std::env::set_var("KILN_BATCH_DECODE_ROWWISE", v) },
            None => unsafe { std::env::remove_var("KILN_BATCH_DECODE_ROWWISE") },
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

    #[tokio::test]
    async fn enqueue_batches_forward_shape_and_routes_responses() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward.clone(), 8);

        let mut rx1 = handle.enqueue(request(101, 1)).await.unwrap();
        let mut rx2 = handle.enqueue(request(202, 1)).await.unwrap();

        assert_eq!(rx1.recv().await, Some(EngineEvent::Token(111)));
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
        assert_eq!(rx2.recv().await, Some(EngineEvent::Token(212)));
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
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn prefix_aware_admission_defers_strict_descendants_but_admits_independent_rows() {
        let forward = Arc::new(MockForward {
            reusable_prefixes: true,
            ..MockForward::default()
        });
        let handle = BatchingEngineHandle::start_with_policy(forward.clone(), 8, true);

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

        assert_eq!(prefix_rx.recv().await, Some(EngineEvent::Token(12)));
        assert!(matches!(
            prefix_rx.recv().await,
            Some(EngineEvent::Done { .. })
        ));
        assert_eq!(independent_rx.recv().await, Some(EngineEvent::Token(19)));
        assert!(matches!(
            independent_rx.recv().await,
            Some(EngineEvent::Done { .. })
        ));
        assert_eq!(descendant_rx.recv().await, Some(EngineEvent::Token(13)));
        assert!(matches!(
            descendant_rx.recv().await,
            Some(EngineEvent::Done { .. })
        ));

        let calls = forward.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![vec![2, 9], vec![3]]);
        handle.stop().await.unwrap();
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

        assert_eq!(rx.recv().await, Some(EngineEvent::Token(17)));
        assert_eq!(rx.recv().await, Some(EngineEvent::Token(27)));
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

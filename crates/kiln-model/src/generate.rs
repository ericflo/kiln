//! End-to-end autoregressive text generation pipeline.
//!
//! Wires together tokenizer, model weights, forward pass, and sampling into
//! a `ModelRunner` that accepts text prompts and produces text output.

use anyhow::{Context, Result};

use std::collections::VecDeque;
use std::path::Path;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicUsize, Ordering},
    mpsc,
};

use kiln_core::config::ModelConfig;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use kiln_core::tokenizer::KilnTokenizer;

use crate::backend::{
    self, BackendIdentity, BackendRuntime, LinearBackend, ReplayBackend, ResidencyBackend,
    SamplingBackend, StartupBackend, TrainingLossBackend, TrainingPrecisionPolicy,
    capability::{
        BackendCapabilities, BackendCapabilityQueries, DecodeBatcherPolicy, ReplayNativePrimitive,
        ReplayRequest, Support, decode_hot_path_fallback_policy_for_backend,
        decode_hot_path_generic_fallback_enabled_for_backend,
    },
};
use crate::cancel::CancelHandle;
use crate::cuda_graph::CudaGraphRunner;
use crate::decode_buffers::{DecodeBufferConfig, DecodeBuffers, DecodeElementType};
use crate::forward::lm_head_sample_backend_decode_if;
use crate::forward::{
    GpuWeights, LinearAttentionState, model_forward_kt, model_forward_paged,
    model_forward_paged_batched_decode_hidden,
    model_forward_paged_decode_contiguous_batch_greedy_with_ids,
    model_forward_paged_decode_contiguous_batch_hidden_with_ids,
    model_forward_paged_decode_contiguous_batch_sample_with_ids, model_forward_paged_last_token,
    model_forward_paged_last_token_greedy, model_forward_paged_last_token_with_last_hidden,
    model_forward_paged_next_token_greedy, model_forward_paged_streaming,
    model_forward_paged_streaming_last_token_with_last_hidden,
    model_forward_paged_streaming_with_progress, streaming_prefill_enabled_for,
};
use crate::metal_graph::MetalGraphRunner;
use crate::rocm_graph::RocmGraphRunner;
// (#1082) Native single-submit Vulkan-resident decode entry — only referenced
// from the `#[cfg(feature = "vulkan")]` single-row fast path below.
#[cfg(feature = "vulkan")]
use crate::forward::model_forward_paged_last_token_resident;
use crate::kv_cache::KvCache;
use crate::lora_loader::LoraWeights;
use crate::packed_weight_registry::GpuPackedWeightRegistry;
// (#1082) the candle `crate::paged_kv_cache` module is gone; the kt twin
// `PagedKvCacheKt` is the production cache. Alias it to `PagedKvCache` so the
// existing call sites + the `model_forward_paged*` params (which the PAGED
// agent resolves to the same kt cache) converge on one type.
use crate::paged_kv_cache_kt::PagedKvCacheKt as PagedKvCache;
use crate::sampling::{greedy_sample, sample_step, sample_with_full_params};
use crate::speculative::{
    SpeculativeConfig, speculative_decode_step, speculative_decode_step_paged_greedy,
    speculative_mtp_decode_step,
};

use kiln_core::block::{BlockManager, BlockTable};

/// Returns `Err` with a stable error message if `cancel` has been signalled.
///
/// Decode loops poll this between tokens so that `kiln-server` can drain a
/// `tokio::task::spawn_blocking` whose outer `tokio::time::timeout` already
/// fired, instead of leaving it running with locks held (see #664).
#[inline]
fn check_cancelled(cancel: Option<&CancelHandle>) -> Result<()> {
    if let Some(c) = cancel {
        if c.is_cancelled() {
            anyhow::bail!("generation cancelled by client (request timeout)");
        }
    }
    Ok(())
}

/// (#1082) Map the model config dtype to the kt `DType` the kt paged cache
/// (`PagedKvCacheKt::new`) expects.
fn paged_cache_kt_dtype(dtype: kiln_core::config::DType) -> kiln_tensor::DType {
    match dtype {
        kiln_core::config::DType::BF16 => kiln_tensor::DType::BF16,
        kiln_core::config::DType::FP16 => kiln_tensor::DType::F16,
        kiln_core::config::DType::FP32 => kiln_tensor::DType::F32,
    }
}

/// (#1082) Validate + return the device the kt paged cache allocates its
/// pools on. The cache now allocates per-arm on the model's *runtime* device
/// (`PagedKvCacheKt::new_with_fp8` matches on the `Device`), so we hand it the
/// `Device` directly instead of a bare index. Native MTP generation support is
/// backend-owned capability data; unsupported backends fail before allocating
/// speculative caches.
fn paged_cache_device(
    backend: &dyn BackendRuntime,
    device: &kiln_tensor::Device,
) -> Result<kiln_tensor::Device> {
    let support = BackendCapabilityQueries::backend_capabilities(backend)
        .decode
        .mtp_speculative_generation;
    if matches!(support, Support::Native | Support::NativeWithConstraints) {
        Ok(*device)
    } else {
        anyhow::bail!("native MTP speculative generation requires backend support; got {support:?}")
    }
}

fn fast_batched_linear_state_scatter_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_FAST_BATCHED_LINEAR_STATE_SCATTER").is_err())
}

fn skip_final_gdn_state_readback_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("KILN_DISABLE_VULKAN_SKIP_FINAL_GDN_STATE_READBACK").is_err())
}

struct GdnRecurrentResidentStateScope<'a> {
    backend: &'a dyn BackendRuntime,
    active: bool,
}

impl<'a> GdnRecurrentResidentStateScope<'a> {
    fn new(backend: &'a dyn BackendRuntime) -> Self {
        let active = ResidencyBackend::runtime_enter_gdn_recurrent_resident_state_scope(backend);
        Self { backend, active }
    }
}

impl Drop for GdnRecurrentResidentStateScope<'_> {
    fn drop(&mut self) {
        if self.active {
            ResidencyBackend::runtime_exit_gdn_recurrent_resident_state_scope(self.backend);
        }
    }
}

fn env_truthy_for_profile(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "" | "0" | "false" | "off" | "no")
        })
        .unwrap_or(false)
}

fn profile_decode_batcher_stages_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_truthy_for_profile("KILN_PROFILE_DECODE_BATCHER_STAGES"))
}

/// #1082 CRASHER FIX: detect whether any row's KV pages are NOT physically
/// contiguous within a kBlockN-token tile — the contract the vendored FA2
/// split-KV paged-decode kernel silently assumes (it reads each tile as one
/// contiguous gather from `block_table[base_idx]`, never consulting the
/// intervening entries). When a fragmented free list hands the kernel
/// non-adjacent pages it reads a foreign page / off the pool →
/// CUDA_ERROR_ILLEGAL_ADDRESS. Mirrors the bs=1 check in
/// `forward.rs::try_flash_attn_paged_decode`.
///
/// Chunk = `FA2_KBLOCK_N` tokens. Qwen3.5-4B's GQA full-attn is head_dim=256
/// only, so kBlockN = 64 (`flash_fwd_launch_template.h:170`: hd>128 → 64). A
/// 64-token tile spans `64/block_size` pages, which is the run that must be
/// physically adjacent. With the #1082 default `block_size = 64` this is ONE
/// page per tile → every block_table is trivially "contiguous" → no row ever
/// routes to the slow per-row loop for FA2 reasons (the n=64 fix). At the old
/// `block_size = 16` it is 4 pages/tile. (If a head_dim=128 model is ever
/// served — kiln is Qwen3.5-4B-only today — kBlockN would be 128; bump
/// `FA2_KBLOCK_N` or thread head_dim before then.) Returns true → caller must
/// route to the contiguity-safe per-row decode.
/// FA2 split-KV decode tile width (tokens) for Qwen3.5-4B's head_dim=256 GQA
/// full-attn (`flash_fwd_launch_template.h:170`: hd>128 → 64). The K/V pages
/// backing one tile must be physically adjacent; with `block_size >= 64` a tile
/// is one page so the requirement is vacuous. kiln is Qwen3.5-4B-only (hd=256);
/// a head_dim=128 model would need 128 here (or head_dim threaded through).
pub(crate) const FA2_KBLOCK_N: usize = 64;

pub(crate) fn batch_has_noncontiguous_kv_tiles(
    block_tables: &[&BlockTable],
    seq_lens: &[usize],
    block_size: usize,
) -> bool {
    block_tables.iter().enumerate().any(|(row, bt)| {
        row_has_noncontiguous_kv_tiles(
            bt.blocks.as_slice(),
            seq_lens.get(row).copied().unwrap_or(0),
            block_size,
        )
    })
}

/// Per-row sibling of [`batch_has_noncontiguous_kv_tiles`]: true when THIS row's
/// live KV pages violate the intra-tile physical-contiguity contract the FA2
/// split-KV kernel assumes. Lets the batched-decode partition row-loop only the
/// genuinely-fragmented rows instead of the #1445 all-or-nothing whole-batch
/// serialization that caused the concurrent n=64 cliff (366s p50 -> 43s).
pub(crate) fn row_has_noncontiguous_kv_tiles(
    blocks: &[u32],
    seqlen: usize,
    block_size: usize,
) -> bool {
    if block_size == 0 {
        return false;
    }
    let pages_per_chunk = (FA2_KBLOCK_N / block_size).max(1);
    // Only the pages actually covering live tokens are read by the kernel.
    let n_pages = seqlen.div_ceil(block_size).min(blocks.len());
    let mut c = 0usize;
    while c < n_pages {
        let base = blocks[c];
        let end = (c + pages_per_chunk).min(n_pages);
        for (k, &phys) in blocks[c..end].iter().enumerate() {
            if phys != base.wrapping_add(k as u32) {
                return true;
            }
        }
        c += pages_per_chunk;
    }
    false
}

fn gdn_batched_decode_row_loop_debug_enabled() -> bool {
    // Flipped to false-by-default after the matmul broadcast-copy fix made
    // the true-batched contiguous-batch path strictly faster than the
    // row-loop at every bs > 1. nsys profile (May 2026) showed candle's
    // `broadcast_matmul` materializing a 168 MB BF16 weight copy across the
    // batch dim before every GDN in-proj matmul, which made the batched
    // path slower than just running N row-loop iterations sequentially.
    // With that copy removed, bs=16 jumped from a flat ~100 tok/s ceiling
    // to 790 tok/s (7.8×) on L40S + Qwen3.5-4B. Opt back into the row-loop
    // with `KILN_ENABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP=1` (the old
    // `KILN_DISABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP` env var is still
    // honored for symmetry with prior docs / rollback runbooks — when set
    // to anything other than "0"/"false"/"off" it keeps the row-loop off).
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        // Legacy disable knob: when set to a truthy value, row-loop stays off
        // (i.e. continues to use the new true-batched path).
        if std::env::var("KILN_DISABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP").is_ok() {
            return false;
        }
        // New opt-in knob to re-enable the row-loop fallback for debug /
        // rollback. Recognizes the common truthy spellings.
        match std::env::var("KILN_ENABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP")
            .ok()
            .as_deref()
        {
            Some("1" | "true" | "TRUE" | "yes" | "on" | "ON") => true,
            _ => false,
        }
    })
}

fn finish_decode_batcher_stage_profile(
    stage: &str,
    batch: usize,
    start: Option<std::time::Instant>,
) {
    let Some(start) = start else {
        return;
    };
    eprintln!(
        "kiln_profile_decode_batcher_stage stage={stage} batch={batch} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}

/// Holds loaded model weights and tokenizer, provides text generation.
pub struct ModelRunner {
    pub weights: GpuWeights,
    pub tokenizer: KilnTokenizer,
    pub config: ModelConfig,
    /// EOS token IDs cached from the tokenizer.
    eos_token_ids: Vec<TokenId>,
    /// Currently active LoRA adapter weights (None = base model only).
    active_lora: Option<LoraWeights>,
    /// CUDA graph runner for accelerated decode steps.
    /// Uses Mutex for interior mutability (graph state changes during &self generation).
    cuda_graph: Mutex<CudaGraphRunner>,
    /// ROCm HIP-graph runner for accelerated decode steps (R.9). Independent of
    /// `cuda_graph`; active by default on ROCm devices unless
    /// `KILN_ROCM_GRAPHS=0` is set. Same per-step interior-mutability pattern.
    rocm_graph: Mutex<RocmGraphRunner>,
    /// Metal ICB graph runner for accelerated decode steps. Active only on a
    /// Metal device with `KILN_METAL_GRAPHS` set; otherwise eager Metal decode
    /// is preserved.
    metal_graph: Mutex<MetalGraphRunner>,
    /// Phase A explicit decode weight registry. Decode kernels address weights
    /// by enum keys instead of safetensors/Candle names.
    /// Phase A.5: lazily built on first hot-path access via `packed_weight_registry()`.
    /// Building eagerly in `ModelRunner::new` measured a 22% c=1 paged decode regression
    /// (Validation #4: 42.6 vs 54.76 baseline), so construction stays cheap and the
    /// registry is materialized only when decode actually needs it.
    packed_weight_registry: OnceLock<GpuPackedWeightRegistry>,
    /// Phase A raw decode buffer pool. The first decode/warmup materializes
    /// stable typed tensors for the active graph bucket, then reuses them.
    /// Phase A.6: `OnceLock` instead of `Mutex<Option<_>>` — the buffer is
    /// allocated once at the largest configured graph bucket
    /// (`decode_buffer_max_batch()`), so subsequent decode steps just need a
    /// `get()` (load-acquire) instead of a `Mutex::lock()` per step.
    decode_buffers: OnceLock<DecodeBuffers>,
    /// Phase A.5: lazily built on first hot-path access via `ensure_decode_buffers()`.
    /// Mirrors the lazy registry pattern above so `ModelRunner::new` doesn't validate
    /// shapes that decode hasn't asked for yet.
    decode_buffer_config: OnceLock<DecodeBufferConfig>,
    /// Cached batched `LinearAttentionState` carried across consecutive
    /// `decode_next_tokens_paged_contiguous_batch_greedy` invocations. When
    /// the next call's per-row state-id set is identical to what produced
    /// this cache, we skip the `from_batch_rows` cat (24 GDN layers × 2
    /// state-kinds = 48 cats per step, ~1.6 ms total at bs=16) and reuse
    /// the cached batched state directly. The cache is invalidated on
    /// adapter swap (same lifecycle as `cuda_graph`) and on id-set
    /// mismatch.
    batched_state_cache: Mutex<Option<CachedBatchedState>>,
    backend: Arc<dyn BackendRuntime>,
}

/// Persistent batched-state cache entry. The fingerprint is the set of
/// per-row `PagedBatchedDecodeState::id` values *in order*. We use the
/// stable atomic-counter id rather than a pointer fingerprint because
/// the batching-engine actor's `Vec<ActiveRequest>` shifts surviving
/// requests down in memory whenever a finished request is removed mid-
/// batch via `Vec::remove`, which invalidates pointer-based keys even
/// though the requests themselves are the same. The id survives the
/// shift.
pub(crate) struct CachedBatchedState {
    pub(crate) state: crate::forward::LinearAttentionState,
    pub(crate) row_ids: Vec<u64>,
}

/// Output from a generation call.
#[derive(Debug)]
pub struct GenerationOutput {
    /// The generated text (not including the prompt).
    pub text: String,
    /// The generated token IDs (not including prompt tokens).
    pub token_ids: Vec<TokenId>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

/// A block-aligned paged prefix that can be reused by a later prompt.
pub struct PagedPrefixReuse {
    pub cached_tokens: usize,
    pub block_ids: Vec<u32>,
    pub linear_state: LinearAttentionState,
    pub next_token: Option<PagedPrefixNextToken>,
}

/// A completed block-aligned prompt prefix produced by generation.
pub struct PagedPrefixRegistration {
    pub prompt_tokens: Vec<TokenId>,
    pub block_ids: Vec<u32>,
    pub linear_state: LinearAttentionState,
    pub next_token: Option<PagedPrefixNextToken>,
}

/// Saved first-token source for an exact prompt-cache hit.
#[derive(Clone)]
pub enum PagedPrefixNextToken {
    /// Full last-position logits. Supports both greedy and stochastic sampling.
    // (#1082) kt-native logits — forward + sampler are both kt; no candle bridge.
    Logits(kiln_tensor::Tensor),
    /// Greedy token only. Usable only when the later request is also greedy.
    GreedyToken(TokenId),
}

/// Result of paged generation plus an optional prefix-cache registration.
pub struct PrefixCachedGenerationOutput {
    pub output: GenerationOutput,
    pub registration: Option<PagedPrefixRegistration>,
    /// Additional block-aligned registrations covering positions strictly
    /// less than the full prompt, captured opportunistically during prefill
    /// or decode. These exist so multi-turn agentic loops (e.g. pi) can hit
    /// the cache on subsequent turns when the chat template's generation
    /// prompt differs from how the same assistant message is rendered in
    /// history on later turns. For Qwen3.5 with enable_thinking=false the
    /// generation prompt appends `<|im_start|>assistant\n<think>\n\n</think>\n\n`,
    /// while later-turn history renders the same assistant turn as just
    /// `<|im_start|>assistant\n{content}<|im_end|>\n` — the only way to
    /// share KV across turns there is to register an entry whose token
    /// sequence stops before that divergent tail.
    pub extra_registrations: Vec<PagedPrefixRegistration>,
    pub allocated_blocks: Vec<u32>,
    pub prefill_duration: std::time::Duration,
    pub decode_duration: std::time::Duration,
}

/// Snapshot of the recurrent linear-attention state taken when decode crosses
/// a block-aligned position. Used at request finish time to register an
/// extended prefix-cache entry covering the prompt + the assistant tokens
/// emitted so far. Without this, only the prompt is cached and every
/// follow-up turn re-prefills the entire growing conversation from scratch.
pub struct RollingPrefixSnapshot {
    /// Total position covered by the snapshot (number of leading tokens with
    /// committed KV state). Always a multiple of the block size.
    pub position: usize,
    pub linear_state: LinearAttentionState,
}

/// Per-request state owned by the server batching actor between prefill and
/// decode iterations.
pub struct PagedBatchedDecodeState {
    pub block_table: BlockTable,
    pub linear_state: LinearAttentionState,
    pub seq_len: usize,
    pub next_token: TokenId,
    pub generated_tokens: Vec<TokenId>,
    pub step_seed: Option<u64>,
    pub registration: Option<PagedPrefixRegistration>,
    pub allocated_blocks: Vec<u32>,
    pub prefill_duration: std::time::Duration,
    pub decode_duration: std::time::Duration,
    /// Original prompt tokens, retained so finish-time prefix registration
    /// can synthesize an "extended" entry covering prompt + decoded tokens.
    pub prompt_tokens: Vec<TokenId>,
    /// Block size of the paged KV cache. Stored so the per-step decode
    /// loop can detect block-aligned positions without re-locking the
    /// block manager.
    pub block_size: usize,
    /// Snapshot of the linear-attention state taken during prefill at the
    /// largest block-aligned offset strictly less than the prompt length.
    /// Used to register a cross-turn-safe prefix-cache entry whose token
    /// sequence stops before any chat-template generation-prompt tail.
    pub prefill_split_snapshot: Option<RollingPrefixSnapshot>,
    /// Latest block-aligned snapshot of the linear attention state, taken
    /// during decode. None until decode first crosses a block boundary;
    /// replaced (drop+alloc) at each subsequent boundary.
    pub rolling_snapshot: Option<RollingPrefixSnapshot>,
    /// Stable per-generation identity used for decode graph and state caching
    /// keys. Assigned from the same process-global namespace as direct
    /// generation owners so no two live decode rows can alias. The value is
    /// independent of where the `PagedBatchedDecodeState` happens to live
    /// in memory — important because the batching-engine actor's
    /// `Vec<ActiveRequest>` shifts elements down via `Vec::remove` when a
    /// request finishes mid-batch, which moves the surrounding
    /// `PagedBatchedDecodeState`s to new memory addresses. A pointer-based
    /// cache key would lose its hits on every such shift; this stable id
    /// survives them.
    pub id: u64,
}

static DECODE_ROW_NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn allocate_decode_row_id(counter: &std::sync::atomic::AtomicU64) -> u64 {
    counter
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |current| match current {
                0 => None,
                u64::MAX => Some(0),
                _ => Some(current + 1),
            },
        )
        .unwrap_or_else(|_| panic!("decode row id namespace exhausted"))
}

pub(crate) fn next_decode_row_id() -> u64 {
    allocate_decode_row_id(&DECODE_ROW_NEXT_ID)
}

/// Owns one direct generation's ROCm graph row until its decode loop exits.
///
/// Direct generation has many early exits (EOS, stop sequences, cancellation,
/// receiver disconnects, and forward errors). Tying cleanup to the stack scope
/// ensures all of them release captured graphs and continuity state before the
/// caller can recycle this generation's KV blocks.
struct RocmDecodeOwnerLease<'a> {
    graph: &'a Mutex<RocmGraphRunner>,
    row_id: u64,
}

impl<'a> RocmDecodeOwnerLease<'a> {
    fn new(graph: &'a Mutex<RocmGraphRunner>) -> Self {
        Self {
            graph,
            row_id: next_decode_row_id(),
        }
    }

    fn row_id(&self) -> u64 {
        self.row_id
    }
}

impl Drop for RocmDecodeOwnerLease<'_> {
    fn drop(&mut self) {
        match self.graph.lock() {
            Ok(mut graph) => graph.release_decode_row(self.row_id),
            Err(poisoned) => {
                tracing::warn!(
                    row_id = self.row_id,
                    "recovering poisoned ROCm graph lock to release direct decode owner"
                );
                poisoned.into_inner().release_decode_row(self.row_id);
            }
        }
    }
}

/// Build a strict-prefix prefix-cache registration covering the prompt plus
/// as many decoded assistant tokens as we have a block-aligned linear-state
/// snapshot for. Returns `None` when there's nothing useful to register —
/// e.g. decode never crossed a block boundary, the snapshot's position
/// doesn't extend past the prompt, or the block table doesn't have enough
/// blocks committed (which would indicate a bookkeeping bug upstream).
fn build_extended_registration(
    prompt_tokens: &[TokenId],
    generated_tokens: &[TokenId],
    block_table: &BlockTable,
    block_size: usize,
    rolling_snapshot: Option<RollingPrefixSnapshot>,
) -> Option<PagedPrefixRegistration> {
    let snapshot = rolling_snapshot?;
    if block_size == 0 || snapshot.position == 0 || snapshot.position % block_size != 0 {
        return None;
    }
    let total_available = prompt_tokens.len() + generated_tokens.len();
    if snapshot.position > total_available {
        // Bookkeeping mismatch: snapshot says we have KV for positions
        // beyond what we actually emitted. Skip rather than register a
        // corrupt entry.
        return None;
    }
    let num_blocks = snapshot.position / block_size;
    if num_blocks == 0 || block_table.blocks.len() < num_blocks {
        return None;
    }
    // Build the prompt-token sequence corresponding to this snapshot. When
    // the snapshot is inside the prompt, the entry covers a strict prefix
    // of the prompt (cross-turn-safe — the chat template's generation tail
    // is usually past this point). When the snapshot is past the prompt,
    // the entry covers prompt + decoded tokens (only safe when subsequent
    // turns re-render the assistant message verbatim, i.e. no template
    // divergence — Qwen3.5 with enable_thinking=false does have such a
    // divergence, so prefer the prefill-split-side snapshot there).
    let mut combined = Vec::with_capacity(snapshot.position);
    let prompt_take = prompt_tokens.len().min(snapshot.position);
    combined.extend_from_slice(&prompt_tokens[..prompt_take]);
    let extra_generated = snapshot.position.saturating_sub(prompt_tokens.len());
    if extra_generated > 0 {
        combined.extend_from_slice(&generated_tokens[..extra_generated]);
    }
    debug_assert_eq!(combined.len(), snapshot.position);
    Some(PagedPrefixRegistration {
        prompt_tokens: combined,
        block_ids: block_table.blocks[..num_blocks].to_vec(),
        linear_state: snapshot.linear_state,
        next_token: None,
    })
}

fn strict_prompt_prefix_split_pos(
    prompt_len: usize,
    cached_tokens: usize,
    block_size: usize,
) -> Option<usize> {
    if block_size == 0 || prompt_len <= 1 {
        return None;
    }
    let split_pos = ((prompt_len - 1) / block_size) * block_size;
    (split_pos > cached_tokens && split_pos < prompt_len).then_some(split_pos)
}

fn env_positive_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|&value| value > 0)
}

fn decode_buffer_max_batch(backend: &dyn BackendRuntime) -> usize {
    let explicit = env_positive_usize("KILN_DECODE_BUFFER_MAX_BATCH");
    if let Some(value) = explicit {
        return value;
    }
    // Scale the per-step decode buffer to the widest configured scheduler so
    // the first large batch does not immediately error with `decode batch N
    // exceeds buffer max_batch M`. Vulkan gets a wider unconfigured default
    // because its resident path keeps scaling past b16 on this target.
    let actor_max = env_positive_usize("KILN_MAX_DECODE_BATCH").unwrap_or(0);
    let live_batcher_max = env_positive_usize("KILN_DECODE_BATCH_MAX").unwrap_or(0);
    let backend_default = BackendCapabilityQueries::backend_capabilities(backend)
        .decode_batcher
        .max_batch;
    actor_max.max(live_batcher_max).max(backend_default)
}

enum PrefillSampleSource {
    // (#1082) kt-native logits — forward + sampler are both kt; no candle bridge.
    Logits(kiln_tensor::Tensor),
    GreedyToken(TokenId),
}

impl PrefillSampleSource {
    fn cached_next_token(&self) -> PagedPrefixNextToken {
        match self {
            Self::Logits(logits) => PagedPrefixNextToken::Logits(logits.clone()),
            Self::GreedyToken(token) => PagedPrefixNextToken::GreedyToken(*token),
        }
    }
}

/// Result of streaming paged generation plus prefix-cache ownership metadata.
pub struct PrefixCachedStreamingOutput {
    pub receiver: mpsc::Receiver<StreamEvent>,
    pub registration: Option<PagedPrefixRegistration>,
    pub extra_registrations: Vec<PagedPrefixRegistration>,
    pub allocated_blocks: Vec<u32>,
    /// Channel the API layer uses to hand the *final* "blocks to free" list
    /// to the spawned decode thread, AFTER prefix-cache registration has
    /// computed which of `allocated_blocks` were retained vs evicted. The
    /// decode thread waits on this channel after the decode loop finishes
    /// before freeing, which closes a race where the API layer would call
    /// `bm.free_all(...)` immediately on return — *while* the decode worker
    /// was still reading those same blocks for KV. The visible symptom of
    /// that race was second-and-later same-prompt streaming requests
    /// regressing to a degenerate token loop ("毎回毎回..."). Send `vec![]`
    /// when nothing should be freed (e.g. if the cache retained all blocks).
    /// Drop without sending only on caller failure — the worker then frees
    /// `allocated_blocks` itself as a safe fallback.
    pub block_free_signal: Option<mpsc::Sender<Vec<u32>>>,
}

/// Output from a native MTP speculative generation call.
///
/// Carries everything [`GenerationOutput`] does plus the per-call MTP draft
/// accept/reject counters used by bench reporting to compute α (acceptance
/// rate = `draft_accepted_count / total_draft_attempts`).
#[derive(Debug)]
pub struct MtpGenerationOutput {
    /// The generated text (not including the prompt).
    pub text: String,
    /// The generated token IDs (not including prompt tokens).
    pub token_ids: Vec<TokenId>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// How many MTP draft tokens were accepted across the decode loop.
    pub draft_accepted_count: usize,
    /// How many MTP draft attempts were made (one per [`speculative_mtp_decode_step`] call).
    pub total_draft_attempts: usize,
}

/// A single token emitted during streaming generation.
#[derive(Debug, Clone)]
pub struct StreamToken {
    /// The generated token ID.
    pub token_id: TokenId,
    /// The decoded text for this token.
    pub text: String,
}

/// Final event sent when streaming generation completes.
#[derive(Debug, Clone)]
pub struct StreamDone {
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// Total number of generated tokens.
    pub completion_tokens: usize,
    /// Text held back by the emit gates (UTF-8 char-boundary + stop-window
    /// holdback) that became safe to emit only at end-of-stream. Empty
    /// after a stop match (the held text WAS the stop) and on error paths.
    pub trailing_text: String,
}

/// Events emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A new token was generated.
    Token(StreamToken),
    /// Generation is complete.
    Done(StreamDone),
}

enum StreamTokenDisposition {
    Continue,
    Finished(FinishReason),
    ReceiverDropped,
}

/// Configuration for the live greedy decode rendezvous worker.
///
/// The worker is enabled by default. Metal uses a small default admission delay
/// to collect compatible peers; CUDA drains immediately and defaults to one row
/// per worker pass because the current coalesced CUDA GDN decode path is slower
/// than rowwise scheduling. Set `KILN_DECODE_BATCHER=0` to force the legacy
/// direct rowwise path, `KILN_DECODE_BATCH_WAIT_US` to override the admission
/// delay, `KILN_DECODE_BATCH_MAX` to force this worker's batch size for A/B
/// testing, or `KILN_MAX_DECODE_BATCH` to set the shared actor/worker batch
/// width. Vulkan defaults to a longer wait because same-position peers tend
/// to arrive just outside a short polling window after independent prefills.
#[derive(Debug, Clone, Copy)]
pub struct DecodeBatcherConfig {
    /// Maximum compatible rows to execute in one decode forward pass.
    pub max_batch: usize,
    /// Optional admission delay for collecting peers.
    pub wait: std::time::Duration,
    /// Whether one batch may contain rows at different decode positions.
    pub allow_mixed_seq_lens: bool,
}

impl Default for DecodeBatcherConfig {
    fn default() -> Self {
        Self {
            max_batch: 8,
            wait: std::time::Duration::ZERO,
            allow_mixed_seq_lens: false,
        }
    }
}

impl DecodeBatcherConfig {
    pub fn from_env() -> Self {
        let mut config = Self::default();
        if let Some(parsed) = env_positive_usize("KILN_DECODE_BATCH_MAX") {
            config.max_batch = parsed;
        }
        if let Ok(value) = std::env::var("KILN_DECODE_BATCH_WAIT_US")
            && let Ok(parsed) = value.parse::<u64>()
        {
            config.wait = std::time::Duration::from_micros(parsed);
        }
        if let Some(enabled) = env_flag_value("KILN_DECODE_BATCH_MIXED_SEQ") {
            config.allow_mixed_seq_lens = enabled;
        }
        config
    }

    // (#1082) candle-typed `from_env_for_backend`/`from_env_for_device`/
    // `enabled_for_device` deleted — the kt-typed variants below are the sole
    // entry points now that callers (kiln-server state.rs) and tests use kt
    // `Device`.

    /// Builds the decode-batcher config from env, applying backend-aware
    /// defaults derived from the backend policy.
    pub fn from_env_for_policy(policy: DecodeBatcherPolicy) -> Self {
        let mut config = Self::from_env();
        if env_positive_usize("KILN_DECODE_BATCH_MAX").is_none() {
            config.max_batch =
                env_positive_usize("KILN_MAX_DECODE_BATCH").unwrap_or(policy.max_batch);
        }
        if std::env::var_os("KILN_DECODE_BATCH_WAIT_US").is_none() {
            config.wait = std::time::Duration::from_micros(policy.wait_micros);
        }
        if env_flag_value("KILN_DECODE_BATCH_MIXED_SEQ").is_none() {
            config.allow_mixed_seq_lens = policy.allow_mixed_seq_lens;
        }
        config
    }

    /// Builds the decode-batcher config from env, applying backend-aware
    /// defaults derived from the kt `Device`.
    pub fn from_env_for_backend_kt(device: &kiln_tensor::Device, backend_name: &str) -> Self {
        Self::from_env_for_policy(DecodeBatcherPolicy::for_backend(backend_name, *device))
    }

    /// kt-typed parallel of [`Self::from_env_for_device`].
    pub fn from_env_for_device_kt(device: &kiln_tensor::Device) -> Self {
        Self::from_env_for_backend_kt(device, "")
    }

    /// kt-typed parallel of [`Self::enabled_for_device`].
    pub fn enabled_for_device_kt(device: &kiln_tensor::Device) -> bool {
        let _ = device;
        env_flag_enabled("KILN_DECODE_BATCHER", true)
    }
}

// (#1082) candle-typed `default_decode_batcher_*` helpers deleted. Backend
// defaults now come from `DecodeBatcherPolicy`; env overrides stay here.

fn env_flag_value(name: &str) -> Option<bool> {
    let value = std::env::var(name).ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "false" | "off" | "no" => Some(false),
        "1" | "true" | "on" | "yes" => Some(true),
        _ => None,
    }
}

fn env_flag_enabled(name: &str, default: bool) -> bool {
    env_flag_value(name).unwrap_or(default)
}

fn decode_batcher_rowwise_retry_enabled(backend: &dyn BackendRuntime) -> bool {
    let policy = BackendCapabilityQueries::backend_capabilities(backend).decode_batcher;
    if let Some(env_var) = policy.rowwise_retry_env
        && env_flag_enabled(env_var, false)
    {
        return true;
    }
    decode_hot_path_fallback_policy_for_backend(backend).allows_fallback()
}

fn greedy_token_decode_enabled(backend: &dyn BackendRuntime) -> bool {
    BackendCapabilityQueries::backend_capabilities(backend)
        .decode_batcher
        .use_greedy_token_decode
}

fn prefix_cache_split_snapshot_allowed(backend: &dyn BackendRuntime) -> bool {
    BackendCapabilityQueries::backend_capabilities(backend)
        .decode_batcher
        .allow_prefix_cache_split_snapshot
}

fn native_support_enabled(support: Support) -> bool {
    matches!(support, Support::Native | Support::NativeWithConstraints)
}

fn paged_decode_graph_replay_request(config: &ModelConfig, max_batch: usize) -> ReplayRequest {
    ReplayRequest::paged_decode_graph_outputs(
        config.hidden_size,
        config.intermediate_size,
        max_batch.max(1),
    )
    .with_dtype(paged_cache_kt_dtype(config.dtype))
}

fn paged_decode_replay_primitive_enabled(
    backend: &dyn BackendRuntime,
    config: &ModelConfig,
    max_batch: usize,
    primitive: ReplayNativePrimitive,
) -> bool {
    let req = paged_decode_graph_replay_request(config, max_batch);
    let support = ReplayBackend::runtime_supports_replay_request(backend, &req);
    let authority = ReplayBackend::runtime_replay_authority(backend);
    native_support_enabled(support) && authority.native_primitive == primitive
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GreedyBatchRoute {
    HipGraph,
    Contiguous,
    Later,
}

fn greedy_batch_route(
    all_greedy: bool,
    cache_is_fp8: bool,
    row_count: usize,
    hip_graph_ready: bool,
) -> GreedyBatchRoute {
    if all_greedy && row_count == 1 && hip_graph_ready {
        GreedyBatchRoute::HipGraph
    } else if all_greedy && !cache_is_fp8 {
        GreedyBatchRoute::Contiguous
    } else {
        GreedyBatchRoute::Later
    }
}

fn decode_hot_path_fallback_disabled_context(
    backend: &dyn BackendRuntime,
    operation: &'static str,
) -> String {
    format!(
        "{operation}; fallback policy {:?} for {} decode hot path \
         (set KILN_DECODE_HOT_PATH_DEBUG_FALLBACK=1 to opt in)",
        decode_hot_path_fallback_policy_for_backend(backend),
        BackendIdentity::runtime_name(backend)
    )
}

/// Shared live decode rendezvous for greedy streaming requests.
///
/// Requests keep ownership of stop handling, output routing, block lifetime,
/// and one-row GDN state. At each eligible decode step they temporarily hand a
/// single-token job to this worker; the worker groups same-position jobs and
/// calls `ModelRunner::decode_next_tokens_paged_contiguous_batch_greedy`.
pub struct DecodeBatcher {
    sender: mpsc::Sender<DecodeBatchJob>,
    counters: Arc<DecodeBatcherCounters>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeBatcherStats {
    pub submitted_jobs: usize,
    pub executed_batches: usize,
    pub executed_rows: usize,
    pub runner_calls: usize,
    pub max_runner_calls_per_token: usize,
    pub max_observed_batch: usize,
    pub runner_busy_jobs: usize,
    pub failed_jobs: usize,
}

impl DecodeBatcherStats {
    /// Phase 8 sentinel budget: a live greedy decode row should normally cost
    /// one runner call, with one extra call allowed for the explicit rowwise
    /// retry path after a failed batched attempt.
    pub const MAX_RUNNER_CALLS_PER_TOKEN_BUDGET: usize = 2;

    pub fn runner_calls_per_token(&self) -> Option<f64> {
        if self.executed_rows == 0 {
            None
        } else {
            Some(self.runner_calls as f64 / self.executed_rows as f64)
        }
    }

    pub const fn runner_call_budget_per_token(&self) -> usize {
        Self::MAX_RUNNER_CALLS_PER_TOKEN_BUDGET
    }

    pub const fn runner_call_budget_exceeded(&self) -> bool {
        self.max_runner_calls_per_token > Self::MAX_RUNNER_CALLS_PER_TOKEN_BUDGET
    }
}

struct DecodeBatcherCounters {
    submitted_jobs: AtomicUsize,
    executed_batches: AtomicUsize,
    executed_rows: AtomicUsize,
    runner_calls: AtomicUsize,
    max_runner_calls_per_token: AtomicUsize,
    max_observed_batch: AtomicUsize,
    runner_busy_jobs: AtomicUsize,
    failed_jobs: AtomicUsize,
}

struct DecodeBatchJob {
    input_token: TokenId,
    seq_len: usize,
    block_table: BlockTable,
    linear_state: LinearAttentionState,
    skip_gdn_state_readback: bool,
    response: mpsc::Sender<DecodeBatchReply>,
}

enum DecodeBatchReply {
    Decoded {
        token: TokenId,
        linear_state: LinearAttentionState,
    },
    RunnerBusy {
        linear_state: LinearAttentionState,
    },
    Failed {
        error: String,
        linear_state: LinearAttentionState,
    },
}

enum DecodeBatcherDecode {
    Decoded(TokenId),
    RunnerBusy,
}

impl DecodeBatcher {
    pub fn spawn(
        runner_lock: Arc<std::sync::RwLock<ModelRunner>>,
        paged_cache: Arc<PagedKvCache>,
        config: DecodeBatcherConfig,
    ) -> Result<Arc<Self>> {
        let (sender, receiver) = mpsc::channel();
        let backend = runner_lock
            .read()
            .map_err(|err| anyhow::anyhow!("failed to acquire model runner for batcher: {err}"))?
            .backend
            .clone();
        let counters = Arc::new(DecodeBatcherCounters {
            submitted_jobs: AtomicUsize::new(0),
            executed_batches: AtomicUsize::new(0),
            executed_rows: AtomicUsize::new(0),
            runner_calls: AtomicUsize::new(0),
            max_runner_calls_per_token: AtomicUsize::new(0),
            max_observed_batch: AtomicUsize::new(0),
            runner_busy_jobs: AtomicUsize::new(0),
            failed_jobs: AtomicUsize::new(0),
        });
        let counters_for_worker = counters.clone();
        std::thread::Builder::new()
            .name("kiln-decode-batcher".to_string())
            .spawn(move || {
                run_decode_batcher_worker(
                    runner_lock,
                    paged_cache,
                    backend,
                    receiver,
                    config,
                    counters_for_worker,
                );
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn decode batcher worker: {e}"))?;

        Ok(Arc::new(Self { sender, counters }))
    }

    pub fn max_observed_batch(&self) -> usize {
        self.counters.max_observed_batch.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> DecodeBatcherStats {
        DecodeBatcherStats {
            submitted_jobs: self.counters.submitted_jobs.load(Ordering::Relaxed),
            executed_batches: self.counters.executed_batches.load(Ordering::Relaxed),
            executed_rows: self.counters.executed_rows.load(Ordering::Relaxed),
            runner_calls: self.counters.runner_calls.load(Ordering::Relaxed),
            max_runner_calls_per_token: self
                .counters
                .max_runner_calls_per_token
                .load(Ordering::Relaxed),
            max_observed_batch: self.counters.max_observed_batch.load(Ordering::Relaxed),
            runner_busy_jobs: self.counters.runner_busy_jobs.load(Ordering::Relaxed),
            failed_jobs: self.counters.failed_jobs.load(Ordering::Relaxed),
        }
    }

    fn decode_next_token_greedy(
        &self,
        input_token: TokenId,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        skip_gdn_state_readback: bool,
    ) -> Result<DecodeBatcherDecode> {
        let (response_tx, response_rx) = mpsc::channel();
        let owned_state = take_linear_attention_state(linear_state);
        let job = DecodeBatchJob {
            input_token,
            seq_len,
            block_table: block_table.clone(),
            linear_state: owned_state,
            skip_gdn_state_readback,
            response: response_tx,
        };
        if let Err(err) = self.sender.send(job) {
            *linear_state = err.0.linear_state;
            anyhow::bail!("decode batcher worker is not running");
        }
        self.counters.submitted_jobs.fetch_add(1, Ordering::Relaxed);

        match response_rx.recv() {
            Ok(DecodeBatchReply::Decoded {
                token,
                linear_state: returned_state,
            }) => {
                *linear_state = returned_state;
                Ok(DecodeBatcherDecode::Decoded(token))
            }
            Ok(DecodeBatchReply::RunnerBusy {
                linear_state: returned_state,
            }) => {
                *linear_state = returned_state;
                Ok(DecodeBatcherDecode::RunnerBusy)
            }
            Ok(DecodeBatchReply::Failed {
                error,
                linear_state: returned_state,
            }) => {
                *linear_state = returned_state;
                anyhow::bail!("{error}");
            }
            Err(err) => anyhow::bail!("decode batcher worker disconnected before reply: {err}"),
        }
    }
}

fn take_linear_attention_state(state: &mut LinearAttentionState) -> LinearAttentionState {
    std::mem::replace(
        state,
        LinearAttentionState {
            recurrent_states: Vec::new(),
            conv_states: Vec::new(),
        },
    )
}

fn materialize_decode_job_resident_states(
    backend: &dyn BackendRuntime,
    jobs: &mut [DecodeBatchJob],
) -> Result<()> {
    for job in jobs {
        job.linear_state
            .materialize_gdn_recurrent_resident_states(backend)?;
    }
    Ok(())
}

fn run_decode_batcher_worker(
    runner_lock: Arc<std::sync::RwLock<ModelRunner>>,
    paged_cache: Arc<PagedKvCache>,
    backend: Arc<dyn BackendRuntime>,
    receiver: mpsc::Receiver<DecodeBatchJob>,
    config: DecodeBatcherConfig,
    counters: Arc<DecodeBatcherCounters>,
) {
    let max_batch = config.max_batch.max(1);
    let allow_mixed_seq_lens = config.allow_mixed_seq_lens;
    let mut deferred = VecDeque::new();
    let mut disconnected = false;

    while !disconnected || !deferred.is_empty() {
        let Some(first) = deferred.pop_front().or_else(|| receiver.recv().ok()) else {
            break;
        };
        let seq_len = first.seq_len;
        let mut jobs = vec![first];

        while jobs.len() < max_batch {
            match receiver.try_recv() {
                Ok(job) if allow_mixed_seq_lens || job.seq_len == seq_len => jobs.push(job),
                Ok(job) => deferred.push_back(job),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if config.wait > std::time::Duration::ZERO && jobs.len() < max_batch && !disconnected {
            let deadline = std::time::Instant::now() + config.wait;
            while jobs.len() < max_batch {
                let now = std::time::Instant::now();
                if now >= deadline {
                    break;
                }
                match receiver.recv_timeout(deadline.saturating_duration_since(now)) {
                    Ok(job) if allow_mixed_seq_lens || job.seq_len == seq_len => jobs.push(job),
                    Ok(job) => deferred.push_back(job),
                    Err(mpsc::RecvTimeoutError::Timeout) => break,
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }

        counters
            .max_observed_batch
            .fetch_max(jobs.len(), Ordering::Relaxed);
        counters.executed_batches.fetch_add(1, Ordering::Relaxed);
        counters
            .executed_rows
            .fetch_add(jobs.len(), Ordering::Relaxed);
        process_decode_batch_jobs(
            &runner_lock,
            paged_cache.as_ref(),
            &*backend,
            jobs,
            &counters,
        );
    }
}

fn process_decode_batch_jobs(
    runner_lock: &std::sync::RwLock<ModelRunner>,
    paged_cache: &PagedKvCache,
    fallback_backend: &dyn BackendRuntime,
    mut jobs: Vec<DecodeBatchJob>,
    counters: &DecodeBatcherCounters,
) {
    let runner_guard = match runner_lock.try_read() {
        Ok(guard) => guard,
        Err(std::sync::TryLockError::WouldBlock) => {
            counters
                .runner_busy_jobs
                .fetch_add(jobs.len(), Ordering::Relaxed);
            if let Err(err) = materialize_decode_job_resident_states(fallback_backend, &mut jobs) {
                let message = format!(
                    "failed to materialize resident GDN state before runner-busy fallback: {err:#}"
                );
                counters
                    .failed_jobs
                    .fetch_add(jobs.len(), Ordering::Relaxed);
                for job in jobs {
                    let _ = job.response.send(DecodeBatchReply::Failed {
                        error: message.clone(),
                        linear_state: job.linear_state,
                    });
                }
                return;
            }
            for job in jobs {
                let _ = job.response.send(DecodeBatchReply::RunnerBusy {
                    linear_state: job.linear_state,
                });
            }
            return;
        }
        Err(std::sync::TryLockError::Poisoned(err)) => {
            let mut message =
                format!("failed to acquire runner read lock in decode batcher: {err}");
            if let Err(materialize_err) =
                materialize_decode_job_resident_states(fallback_backend, &mut jobs)
            {
                tracing::warn!(
                    error = %materialize_err,
                    "failed to materialize resident GDN state after poisoned runner lock"
                );
                message = format!(
                    "{message}; also failed to materialize resident GDN state: {materialize_err:#}"
                );
            }
            counters
                .failed_jobs
                .fetch_add(jobs.len(), Ordering::Relaxed);
            for job in jobs {
                let _ = job.response.send(DecodeBatchReply::Failed {
                    error: message.clone(),
                    linear_state: job.linear_state,
                });
            }
            return;
        }
    };

    let backend = &*runner_guard.backend;
    let job_count = jobs.len();
    let mut runner_calls_for_jobs = 1usize;
    let rowwise_retry_enabled = decode_batcher_rowwise_retry_enabled(backend);
    let tokens =
        match decode_batch_jobs_with_runner(&runner_guard, paged_cache, &mut jobs, counters) {
            Ok(tokens) => Ok(tokens),
            Err(err) if jobs.len() > 1 && rowwise_retry_enabled => {
                tracing::debug!(
                    batch = jobs.len(),
                    error = %err,
                    "batched greedy decode failed; falling back to rowwise decode jobs"
                );
                let mut tokens = Vec::with_capacity(jobs.len());
                let mut fallback_error = None;
                for idx in 0..jobs.len() {
                    runner_calls_for_jobs += 1;
                    match decode_batch_jobs_with_runner(
                        &runner_guard,
                        paged_cache,
                        &mut jobs[idx..idx + 1],
                        counters,
                    ) {
                        Ok(mut row_tokens) => tokens.push(row_tokens.remove(0)),
                        Err(row_err) => {
                            fallback_error = Some(row_err);
                            break;
                        }
                    }
                }
                match fallback_error {
                    Some(err) => Err(err),
                    None => Ok(tokens),
                }
            }
            Err(err) if jobs.len() > 1 => {
                tracing::debug!(
                    batch = jobs.len(),
                    error = %err,
                    "batched greedy decode failed; rowwise retry disabled"
                );
                Err(err)
            }
            Err(err) => Err(err),
        };
    counters.max_runner_calls_per_token.fetch_max(
        if job_count > 0 && runner_calls_for_jobs > 1 {
            2
        } else {
            usize::from(job_count > 0)
        },
        Ordering::Relaxed,
    );

    match tokens {
        Ok(tokens) => {
            for (job, token) in jobs.into_iter().zip(tokens.into_iter()) {
                if job.skip_gdn_state_readback {
                    job.linear_state
                        .evict_gdn_recurrent_resident_states(backend);
                }
                let _ = job.response.send(DecodeBatchReply::Decoded {
                    token,
                    linear_state: job.linear_state,
                });
            }
        }
        Err(err) => {
            let message = format!("{err:#}");
            if let Err(materialize_err) = materialize_decode_job_resident_states(backend, &mut jobs)
            {
                tracing::warn!(
                    error = %materialize_err,
                    "failed to materialize resident GDN state after decode batch error"
                );
            }
            counters
                .failed_jobs
                .fetch_add(jobs.len(), Ordering::Relaxed);
            for job in jobs {
                let _ = job.response.send(DecodeBatchReply::Failed {
                    error: message.clone(),
                    linear_state: job.linear_state,
                });
            }
        }
    }
}

fn decode_batch_jobs_with_runner(
    runner: &ModelRunner,
    paged_cache: &PagedKvCache,
    jobs: &mut [DecodeBatchJob],
    counters: &DecodeBatcherCounters,
) -> Result<Vec<TokenId>> {
    counters.runner_calls.fetch_add(1, Ordering::Relaxed);
    let profile_stages = profile_decode_batcher_stages_enabled();
    let total_start = profile_stages.then(std::time::Instant::now);
    let stage_start = profile_stages.then(std::time::Instant::now);
    let input_tokens: Vec<TokenId> = jobs.iter().map(|job| job.input_token).collect();
    let seq_lens: Vec<usize> = jobs.iter().map(|job| job.seq_len).collect();
    let block_tables: Vec<BlockTable> = jobs.iter().map(|job| job.block_table.clone()).collect();
    let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
    let skip_gdn_state_readback = skip_final_gdn_state_readback_enabled()
        && jobs.iter().all(|job| job.skip_gdn_state_readback);
    finish_decode_batcher_stage_profile("job_metadata", jobs.len(), stage_start);

    let stage_start = profile_stages.then(std::time::Instant::now);
    let _skip_scope = crate::forward::VulkanSkipGdnStateReadbackScope::new(skip_gdn_state_readback);
    let tokens = if runner.has_linear_attention_layers() {
        let mut linear_states: Vec<&mut LinearAttentionState> =
            jobs.iter_mut().map(|job| &mut job.linear_state).collect();
        runner.decode_next_tokens_paged_contiguous_batch_greedy(
            &input_tokens,
            paged_cache,
            &block_table_refs,
            &seq_lens,
            &mut linear_states,
        )
    } else {
        let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
        runner.decode_next_tokens_paged_contiguous_batch_greedy(
            &input_tokens,
            paged_cache,
            &block_table_refs,
            &seq_lens,
            &mut no_linear_states,
        )
    };
    finish_decode_batcher_stage_profile("runner_call", jobs.len(), stage_start);
    finish_decode_batcher_stage_profile("worker_total", jobs.len(), total_start);
    tokens
}

struct SharedBlockReservation<'a> {
    block_manager: &'a Mutex<BlockManager>,
    block_ids: Vec<u32>,
}

impl Drop for SharedBlockReservation<'_> {
    fn drop(&mut self) {
        if self.block_ids.is_empty() {
            return;
        }
        match self.block_manager.lock() {
            Ok(mut guard) => guard.free_all(&self.block_ids),
            Err(e) => tracing::error!("failed to lock block manager to free blocks: {e}"),
        }
    }
}

fn lock_block_manager(
    block_manager: &Mutex<BlockManager>,
) -> Result<std::sync::MutexGuard<'_, BlockManager>> {
    block_manager
        .lock()
        .map_err(|e| anyhow::anyhow!("failed to lock block manager: {e}"))
}

// `PagedKvCache` no longer hides behind a `Mutex` — its write methods take
// `&self` and rely on the underlying tensor storage's interior mutability,
// so callers can simply pass the `&PagedKvCache` straight through. This
// helper is kept as a pass-through identity to minimize call-site churn
// during the lock-removal sweep; it can be inlined later.
fn lock_paged_cache(paged_cache: &PagedKvCache) -> Result<&PagedKvCache> {
    Ok(paged_cache)
}

pub fn append_prefix_block_table(cached_blocks: &[u32], allocated_blocks: &[u32]) -> BlockTable {
    let mut block_table = BlockTable::new();
    for &block_id in cached_blocks {
        block_table.push(block_id);
    }
    for &block_id in allocated_blocks {
        block_table.push(block_id);
    }
    block_table
}

/// Legacy "lm_head → host sampler" batched path. Used when the backend
/// doesn't expose the fused on-device sampler, when the sampling request is
/// outside the backend kernel's supported envelope, or as the final fallback
/// for mixed shapes that cannot take a greedy or fused sampled path.
fn run_legacy_lm_head_sample_batch(
    backend: &dyn crate::backend::BackendRuntime,
    hidden: &kiln_tensor::Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    params: &[SamplingParams],
    states: &[&mut PagedBatchedDecodeState],
) -> Result<Vec<TokenId>> {
    // (#1082) lm head + sampler are both kt-native — `hidden` arrives kt from
    // the batched decode forward and the sampler takes kt logits directly. No
    // candle bridge.
    let logits = crate::forward::model_forward_head_backend_decode_if(
        Some(backend),
        hidden,
        weights,
        config,
    )
    .context("batched decode lm head")?;
    let mut sampled = Vec::with_capacity(states.len());
    for (idx, params) in params.iter().enumerate() {
        let row = logits
            .narrow(0, idx, 1)
            .with_context(|| format!("batched decode lm head row {idx}"))?;
        let token = if params.temperature == 0.0 {
            greedy_sample(&row)?
        } else {
            let mut row_params = params.clone();
            row_params.seed = states[idx].step_seed;
            sample_with_full_params(&row, &row_params, &states[idx].generated_tokens)?
        };
        sampled.push(token);
    }
    Ok(sampled)
}

fn unique_history_counts_for_batch_sample(history: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut counts: std::collections::BTreeMap<u32, u32> = std::collections::BTreeMap::new();
    for &token in history {
        *counts.entry(token).or_default() += 1;
    }
    let mut indices = Vec::with_capacity(counts.len());
    let mut values = Vec::with_capacity(counts.len());
    for (token, count) in counts {
        indices.push(token);
        values.push(count);
    }
    (indices, values)
}

fn sample_seed_for_batch_row(step_seed: Option<u64>, history: &[u32]) -> u64 {
    step_seed.unwrap_or_else(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let history_hash = history.iter().fold(0xCBF29CE484222325u64, |acc, &token| {
            (acc ^ token as u64).wrapping_mul(0x100000001B3)
        });
        nanos.wrapping_add(history_hash)
    })
}

fn run_lm_head_sample_batch_with_contexts(
    backend: &dyn crate::backend::BackendRuntime,
    hidden: &kiln_tensor::Tensor,
    weights: &GpuWeights,
    config: &kiln_core::config::ModelConfig,
    params: &[SamplingParams],
    step_seeds: &[Option<u64>],
    generated_tokens: &[Vec<TokenId>],
) -> Result<Vec<TokenId>> {
    anyhow::ensure!(
        params.len() == step_seeds.len() && params.len() == generated_tokens.len(),
        "batched decode sampling context length mismatch"
    );
    let top_k_values: Vec<u32> = params.iter().map(|param| param.top_k).collect();
    let temperature_values: Vec<f32> = params.iter().map(|param| param.temperature).collect();
    if SamplingBackend::runtime_supports_linear_decode_sample_batch(
        backend,
        &top_k_values,
        &temperature_values,
    ) {
        let normed = crate::forward::model_forward_final_norm(hidden, weights, config)
            .context("batched decode final norm for fused sampling")?;
        let repetition_values: Vec<f32> = params
            .iter()
            .map(|param| param.repetition_penalty)
            .collect();
        let presence_values: Vec<f32> = params.iter().map(|param| param.presence_penalty).collect();
        let frequency_values: Vec<f32> =
            params.iter().map(|param| param.frequency_penalty).collect();
        let top_p_values: Vec<f32> = params.iter().map(|param| param.top_p).collect();
        let min_p_values: Vec<f32> = params.iter().map(|param| param.min_p).collect();
        let seed_values: Vec<u64> = step_seeds
            .iter()
            .zip(generated_tokens.iter())
            .map(|(&seed, history)| sample_seed_for_batch_row(seed, history))
            .collect();
        let mut history_rows = Vec::new();
        let mut history_indices = Vec::new();
        let mut history_counts = Vec::new();
        for (row_idx, (param, history)) in params.iter().zip(generated_tokens.iter()).enumerate() {
            if param.is_effectively_greedy()
                || param.token_penalties_are_no_op()
                || history.is_empty()
            {
                continue;
            }
            let (indices, counts) = unique_history_counts_for_batch_sample(history);
            for (idx, count) in indices.into_iter().zip(counts.into_iter()) {
                history_rows.push(row_idx as u32);
                history_indices.push(idx);
                history_counts.push(count);
            }
        }
        if let Some(tokens) = SamplingBackend::runtime_linear_decode_sample_batch(
            backend,
            &normed,
            &weights.embed_tokens_t,
            &history_rows,
            &history_indices,
            &history_counts,
            &repetition_values,
            &presence_values,
            &frequency_values,
            &temperature_values,
            &top_k_values,
            &top_p_values,
            &min_p_values,
            &seed_values,
        )
        .context("fused batched linear_decode_sample failed")?
        {
            return Ok(tokens);
        }
    }
    let logits = crate::forward::model_forward_head_backend_decode_if(
        Some(backend),
        hidden,
        weights,
        config,
    )
    .context("batched decode lm head")?;
    let mut sampled = Vec::with_capacity(params.len());
    for (idx, params) in params.iter().enumerate() {
        let row = logits
            .narrow(0, idx, 1)
            .with_context(|| format!("batched decode lm head row {idx}"))?;
        let token = if params.temperature == 0.0 {
            greedy_sample(&row)?
        } else {
            let mut row_params = params.clone();
            row_params.seed = step_seeds[idx];
            sample_with_full_params(&row, &row_params, &generated_tokens[idx])?
        };
        sampled.push(token);
    }
    Ok(sampled)
}

fn sample_first_decode_token(
    // (#1082) kt-native logits — sampler is kt now.
    logits: &kiln_tensor::Tensor,
    params: &SamplingParams,
) -> Result<TokenId> {
    if params.is_effectively_greedy() {
        Ok(greedy_sample(logits)?)
    } else {
        // First decode token has no generated history yet — penalties
        // become a no-op even when set, which is the correct OpenAI
        // semantics (penalties apply to *generated* tokens only).
        Ok(sample_with_full_params(logits, params, &[])?)
    }
}

/// Composite per-request emit gate: incremental detokenization + stop
/// holdback. One per streaming generation; finish() drains residue at
/// non-stop exits (a stop can complete inside held bytes).
struct StreamTextGate {
    detok: crate::stream_text::IncrementalDetokenizer,
    stop: crate::stream_text::StopTailGate,
}

impl StreamTextGate {
    fn new(stop_sequences: &[String]) -> Self {
        Self {
            detok: crate::stream_text::IncrementalDetokenizer::new(),
            stop: crate::stream_text::StopTailGate::new(stop_sequences),
        }
    }

    /// Non-stop loop exit: push the detokenizer residual through the stop
    /// gate, then drain the stop holdback. Returns
    /// `(trailing_text, late_stop)` — when `late_stop` is `Some`, a stop
    /// completed inside the held bytes and the caller must override its
    /// finish reason.
    fn finish(
        &mut self,
        tokenizer: &KilnTokenizer,
        tokens: &[TokenId],
    ) -> (String, Option<String>) {
        let residual = self.detok.flush(tokenizer, tokens);
        let scan = self.stop.push(&residual);
        if let Some(stop) = scan.matched_stop {
            return (scan.emit, Some(stop));
        }
        let mut trailing = scan.emit;
        trailing.push_str(&self.stop.flush());
        (trailing, None)
    }
}

fn emit_stream_token(
    tx: &mpsc::Sender<StreamEvent>,
    tokenizer: &KilnTokenizer,
    gate: &mut StreamTextGate,
    generated_tokens: &mut Vec<TokenId>,
    token: TokenId,
) -> StreamTokenDisposition {
    generated_tokens.push(token);

    // CHECK BEFORE EMIT: the delta passes the stop gate first, so the
    // matched stop never reaches the wire (the pre-gate code emitted the
    // token THEN ran the stop check on the full decoded prefix — pi's
    // stop-marker parsers saw phantom delimiters in every stream).
    // Exactly one StreamEvent::Token per accepted token (text may be ""),
    // keeping completion-token counting and usage exact.
    let delta = gate.detok.next_delta(tokenizer, generated_tokens);
    let scan = gate.stop.push(&delta);
    if tx
        .send(StreamEvent::Token(StreamToken {
            token_id: token,
            text: scan.emit,
        }))
        .is_err()
    {
        return StreamTokenDisposition::ReceiverDropped;
    }
    match scan.matched_stop {
        Some(stop) => StreamTokenDisposition::Finished(FinishReason::StopSequence(stop)),
        None => StreamTokenDisposition::Continue,
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// Hit an EOS token.
    Eos,
    /// Reached max_tokens limit.
    MaxTokens,
    /// Hit a stop sequence in the decoded text.
    StopSequence(String),
}

impl ModelRunner {
    pub fn is_eos_token(&self, token: TokenId) -> bool {
        self.eos_token_ids.contains(&token)
    }

    pub fn stop_sequence_match(
        &self,
        generated_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<Option<String>> {
        if params.stop.is_empty() {
            return Ok(None);
        }
        let Some(text) = self
            .tokenizer
            .decode(generated_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .ok()
        else {
            return Ok(None);
        };
        Ok(params
            .stop
            .iter()
            .find(|stop_seq| text.contains(stop_seq.as_str()))
            .cloned())
    }

    /// Create a new ModelRunner from pre-loaded weights, tokenizer, and config.
    ///
    /// Create a runner with the production default: CUDA graphs disabled.
    /// Pass `cuda_graphs: true` to [`Self::new_with_options`] to opt in.
    pub fn new(weights: GpuWeights, tokenizer: KilnTokenizer, config: ModelConfig) -> Self {
        Self::new_with_options(weights, tokenizer, config, false)
    }

    /// Create a new ModelRunner with explicit CUDA graph control.
    pub fn new_with_options(
        weights: GpuWeights,
        tokenizer: KilnTokenizer,
        config: ModelConfig,
        cuda_graphs: bool,
    ) -> Self {
        let eos_token_ids = tokenizer.eos_token_ids();
        // (#1082) `embed_tokens.device()` is a kt `Device`. The backend
        // dispatcher is kt-native (`for_device_kt`). (#1082) `CudaGraphRunner::new`
        // is kt-native now — no candle device bridge.
        let kt_device = weights.embed_tokens.device();
        let backend = backend::for_device_kt(&kt_device);
        let cuda_graph = CudaGraphRunner::new(&kt_device, cuda_graphs);
        // R.9: the ROCm graph runner gates itself on a Rocm device and the
        // `KILN_ROCM_GRAPHS=0` kill switch, so pass `true` here; it is inert on
        // every other backend.
        let rocm_graph = RocmGraphRunner::new(&kt_device, true);
        let metal_graph = MetalGraphRunner::new(&kt_device, true);
        let training_caps = TrainingLossBackend::runtime_training_capabilities(backend.as_ref());
        tracing::info!(
            backend = BackendIdentity::runtime_name(backend.as_ref()),
            projection_training = training_caps.projection_training,
            flce_loss = training_caps.flce_loss,
            rmsnorm_training = training_caps.rmsnorm_training,
            resident_activation = training_caps.resident_activation,
            lora_delta_training = training_caps.lora_delta_training,
            sgd_step = training_caps.sgd_step,
            adamw_step = training_caps.adamw_step,
            native_training = training_caps.native_training,
            "Backend training capability profile"
        );
        // Phase A.5: registry + decode-buffer config are deferred to first hot-path
        // access. Building them eagerly here regressed c=1 paged decode by 22%
        // (Validation #4: 42.6 tok/s vs 54.76 baseline). The lazy `OnceLock` keeps
        // construction cheap and matches the production-path warmup contract.
        Self {
            weights,
            tokenizer,
            config,
            eos_token_ids,
            active_lora: None,
            cuda_graph: Mutex::new(cuda_graph),
            rocm_graph: Mutex::new(rocm_graph),
            metal_graph: Mutex::new(metal_graph),
            packed_weight_registry: OnceLock::new(),
            decode_buffers: OnceLock::new(),
            decode_buffer_config: OnceLock::new(),
            batched_state_cache: Mutex::new(None),
            backend,
        }
    }

    pub fn backend_name(&self) -> &'static str {
        BackendIdentity::runtime_name(self.backend.as_ref())
    }

    pub fn backend_capabilities(&self) -> BackendCapabilities {
        BackendCapabilityQueries::backend_capabilities(self.backend.as_ref())
    }

    pub fn training_precision_policy(&self) -> TrainingPrecisionPolicy {
        TrainingLossBackend::runtime_training_precision_policy(self.backend.as_ref())
    }

    /// Eagerly allocate the backend-resident decode scratch ring when the
    /// backend supports it. This keeps the first live decode request from
    /// paying the pool feasibility/allocation cost on the request path.
    pub fn warm_resident_decode_pool(&self, max_batch: usize) -> bool {
        ReplayBackend::runtime_decode_resident_pool_ready(
            self.backend.as_ref(),
            self.config.hidden_size,
            self.config.intermediate_size,
            max_batch,
        )
    }

    pub fn precompile_backend_startup_kernels(&self) -> Result<()> {
        StartupBackend::runtime_precompile_startup_kernels(self.backend.as_ref())
    }

    /// Preload backend-specific decode weights into any persistent device cache.
    ///
    /// After upload, on backends that opt in (Vulkan today), drop the
    /// pre-transposed candle CPU storage of those weights so that the
    /// device-resident buffer is the only canonical copy. Saves
    /// ~6-7 GB peak RSS on Qwen3.5-4B at T=918 training shape — see
    /// `docs/audits/candle_cpu_residency_2026-05-11.md`.
    pub fn prewarm_backend_decode_weights(&mut self) -> Result<()> {
        LinearBackend::runtime_prewarm_decode_weights(self.backend.as_ref(), &self.weights)?;
        // (#1082) `drop_uploaded_bf16_weights` is kt-native — pass kt device.
        let kt_device = self.weights.embed_tokens.device();
        LinearBackend::runtime_drop_uploaded_bf16_weights(
            self.backend.as_ref(),
            &mut self.weights,
            &kt_device,
        )?;
        Ok(())
    }

    /// Load a LoRA adapter from a PEFT-compatible directory.
    ///
    /// The directory must contain `adapter_config.json` and `adapter_model.safetensors`.
    /// Replaces any previously loaded adapter.
    pub fn load_adapter(&mut self, path: &Path) -> Result<()> {
        // (#1082) `LoraWeights::load` is kt-native — pass kt device by value.
        let kt_device = self.weights.embed_tokens.device();
        let num_layers = self.config.num_layers;
        let lora = LoraWeights::load(path, num_layers, kt_device)
            .context("failed to load LoRA adapter")?;
        // Phase 4.1: register the adapter's LoRA tensors in the
        // backend's resident activation registry so the inference
        // path's `add_lora_delta_to_base` dispatches through
        // `lora_delta_resident` (on-device LoRA matmul) instead of
        // candle CPU `compute_lora_delta`. No-op on backends without
        // registry support.
        if let Err(e) = lora.register_with_backend(&*self.backend) {
            tracing::warn!(error = %e, "failed to register LoRA adapter with backend; \
                falling back to candle CPU LoRA delta path");
        }
        // If a previous adapter is loaded, evict it first so the
        // registry doesn't accumulate stale entries.
        if let Some(prev) = self.active_lora.take() {
            prev.evict_from_backend(&*self.backend);
        }
        self.active_lora = Some(lora);
        if let Ok(mut graph) = self.cuda_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut graph) = self.rocm_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut graph) = self.metal_graph.lock() {
            graph.invalidate();
        }
        // Adapter swap rewires the matmul weights; any cached batched
        // LinearAttentionState is per-request data (independent of weights)
        // but the cache lifecycle follows the same conservative
        // invalidation rule as `cuda_graph` so we don't try to skip the
        // assemble step across a weight-change boundary.
        if let Ok(mut cache) = self.batched_state_cache.lock() {
            *cache = None;
        }
        Ok(())
    }

    /// Unload the currently active LoRA adapter, reverting to base model.
    pub fn unload_adapter(&mut self) {
        if let Some(prev) = self.active_lora.take() {
            // Phase 4.1: evict the now-removed adapter's LoRA Vars
            // from the resident registry so they don't leak.
            prev.evict_from_backend(&*self.backend);
        }
        if let Ok(mut graph) = self.cuda_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut graph) = self.rocm_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut graph) = self.metal_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut cache) = self.batched_state_cache.lock() {
            *cache = None;
        }
    }

    /// Returns a reference to the active LoRA weights, if any.
    pub fn active_lora(&self) -> Option<&LoraWeights> {
        self.active_lora.as_ref()
    }

    pub fn packed_weight_registry(&self) -> &GpuPackedWeightRegistry {
        // Phase A.5: lazy build on first access. See `new_with_options` for the
        // 22% c=1 regression that this defers.
        self.packed_weight_registry.get_or_init(|| {
            GpuPackedWeightRegistry::from_gpu_weights(&self.weights)
                .expect("Qwen3.5 packed-weight registry must build from loaded GPU weights")
        })
    }

    pub fn ensure_decode_buffers(&self, batch: usize) -> Result<()> {
        // Phase A.6: lock-free fast path. The buffer is allocated once at the
        // largest configured graph bucket; subsequent decode steps only need a
        // load-acquire on the `OnceLock`, eliminating the ~11% c=1 regression
        // measured in Validation #5 from a per-step `Mutex::lock`.
        if let Some(buffers) = self.decode_buffers.get() {
            return buffers.ensure_batch_fits(batch);
        }
        // Phase A.5: lazy decode-buffer-config build (see `new_with_options`).
        let cfg = self
            .decode_buffer_config
            .get_or_init(|| {
                DecodeBufferConfig::graph_bucket(
                    decode_buffer_max_batch(self.backend.as_ref()),
                    self.config.max_position_embeddings,
                    1,
                    16,
                    DecodeElementType::Bf16,
                )
                .expect("Qwen3.5 decode buffer config must be valid")
            })
            .clone();
        // (#1082) kt-native — DecodeBuffers::allocate takes the kt device directly.
        let kt_device = self.weights.embed_tokens.device();
        let buffers = DecodeBuffers::allocate(cfg, &kt_device)?;
        // If another thread won the race, drop our newly allocated copy
        // harmlessly and fall through to the winner's buffer.
        let _ = self.decode_buffers.set(buffers);
        self.decode_buffers
            .get()
            .expect("decode buffers initialized")
            .ensure_batch_fits(batch)
    }

    /// Atomically swap the active LoRA adapter.
    ///
    /// Pass `Some(lora)` to activate pre-loaded weights, or `None` to revert to
    /// the base model. Designed for use with `RwLock`: load weights outside the
    /// lock, then take a brief write lock to call this method.
    ///
    /// Invalidates any captured CUDA graph since the adapter change alters
    /// weight tensor pointers embedded in the graph.
    pub fn swap_lora(&mut self, lora: Option<LoraWeights>) {
        self.active_lora = lora;
        if let Ok(mut graph) = self.cuda_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut graph) = self.rocm_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut graph) = self.metal_graph.lock() {
            graph.invalidate();
        }
        if let Ok(mut cache) = self.batched_state_cache.lock() {
            *cache = None;
        }
    }

    fn snapshot_draft_linear_state(
        &self,
        linear_state: &LinearAttentionState,
        spec_config: &SpeculativeConfig,
    ) -> Result<LinearAttentionState> {
        let draft_linear_layers = self
            .weights
            .linear_attention_layers_in_prefix(spec_config.draft_layers);
        linear_state
            .snapshot_for_decode_rollback_prefix(draft_linear_layers)
            .context("clone draft linear-attention prefix from skip-layer prefill")
    }

    /// Generate text from a prompt string.
    ///
    /// Tokenizes the prompt, runs the autoregressive generation loop,
    /// and decodes the output tokens back to text.
    pub fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens(&prompt_tokens, params)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Create a new KV cache sized for this model.
    fn new_kv_cache(&self, max_seq_len: usize) -> Result<KvCache> {
        // #1082: `embed_tokens.device()` is now a kt `Device` (by value);
        // route through the kt-typed `KvCache::new_kt` so no candle Device
        // import is needed at the call site.
        let dtype = match self.config.dtype {
            kiln_core::config::DType::BF16 => kiln_tensor::DType::BF16,
            kiln_core::config::DType::FP16 => kiln_tensor::DType::F16,
            kiln_core::config::DType::FP32 => kiln_tensor::DType::F32,
        };
        let device = self.weights.embed_tokens.device();
        KvCache::new_kt(
            self.config.num_full_attention_layers,
            self.config.num_kv_heads,
            self.config.head_dim,
            max_seq_len,
            dtype,
            &device,
        )
    }

    /// Create a new linear attention state for GDN layers.
    fn new_linear_state(&self) -> Result<LinearAttentionState> {
        // #1082: kt `Device` by value -> pass by reference.
        let device = self.weights.embed_tokens.device();
        LinearAttentionState::new_with_batch_for_inference_runtime(
            &self.config,
            1,
            &device,
            self.backend.as_ref(),
        )
    }

    fn has_linear_attention_layers(&self) -> bool {
        self.weights.layers.iter().any(|layer| {
            matches!(
                layer.attention,
                crate::forward::GpuAttentionWeights::Linear(_)
            )
        })
    }

    pub fn cuda_graph_enabled(&self) -> Result<bool> {
        Ok(self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?
            .is_enabled())
    }

    pub fn rocm_graph_enabled(&self) -> Result<bool> {
        Ok(self
            .rocm_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock ROCm graph runner: {e}"))?
            .is_enabled())
    }

    /// Snapshot ROCm graph configuration, circuit-breaker state, and execution
    /// counters. Counters are lifetime-monotonic for this model runner.
    pub fn rocm_graph_stats(&self) -> Result<crate::rocm_graph::RocmGraphStats> {
        Ok(self
            .rocm_graph
            .try_lock()
            .map_err(|e| anyhow::anyhow!("ROCm graph runner snapshot unavailable: {e}"))?
            .stats())
    }

    pub fn metal_graph_enabled(&self) -> Result<bool> {
        Ok(self
            .metal_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?
            .is_enabled())
    }

    /// Generate text token-by-token, sending each token to a channel as it is produced.
    ///
    /// Returns an `mpsc::Receiver<StreamEvent>` that yields `Token` events
    /// followed by a final `Done` event.  The generation runs synchronously
    /// on the calling thread (caller should use `spawn_blocking`).
    pub fn generate_streaming(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let (tx, rx) = mpsc::channel();

        let max_total = prompt_tokens.len() + params.max_tokens;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        // Prefill: run forward pass on all prompt tokens at once
        let logits = model_forward_kt(
            &*self.backend,
            &prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        // Sample first token from the last position's logits
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);

        let mut next_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, step_seed, &[])?
        };

        for _step in 0..params.max_tokens {
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            // Check for EOS
            if self.eos_token_ids.contains(&next_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                next_token,
            ) {
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                StreamTokenDisposition::Finished(reason) => {
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: reason,
                        completion_tokens: generated_tokens.len(),
                        trailing_text: String::new(),
                    }));
                    return Ok(rx);
                }
                StreamTokenDisposition::Continue => {}
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            // Decode step: forward pass on just the new token
            let logits = model_forward_kt(
                &*self.backend,
                &[next_token],
                &self.weights,
                &self.config,
                Some(&mut kv_cache),
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
            .context("decode forward pass failed")?;
            kv_cache.advance(1);

            next_token = if params.is_effectively_greedy() {
                greedy_sample(&logits)?
            } else {
                sample_step(&logits, params, step_seed, &generated_tokens)?
            };
        }

        let (trailing_text, late_stop) = gate.finish(&self.tokenizer, &generated_tokens);
        let (finish_reason, trailing_text) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, trailing_text),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text,
        }));

        Ok(rx)
    }

    /// Autoregressive generation loop operating on token IDs.
    ///
    /// 1. Prefill: run forward pass on the full prompt to get first next-token logits.
    /// 2. Decode: repeatedly sample a token, run forward on just the new token.
    /// 3. Stop on EOS, max_tokens, or stop sequence.
    pub fn generate_from_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let max_total = prompt_tokens.len() + params.max_tokens;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        // Prefill: run forward pass on all prompt tokens at once
        let logits = model_forward_kt(
            &*self.backend,
            prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        // Sample first token from the last position's logits
        let mut next_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, step_seed, &[])?
        };

        for _step in 0..params.max_tokens {
            // Advance seed for next step
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            // Check for EOS
            if self.eos_token_ids.contains(&next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            // Check stop sequences against decoded text so far
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            // Decode step: forward pass on just the new token (KV cache has all previous)
            let logits = model_forward_kt(
                &*self.backend,
                &[next_token],
                &self.weights,
                &self.config,
                Some(&mut kv_cache),
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
            .context("decode forward pass failed")?;
            kv_cache.advance(1);

            // Sample next token from the new logits
            next_token = if params.is_effectively_greedy() {
                greedy_sample(&logits)?
            } else {
                sample_step(&logits, params, step_seed, &generated_tokens)?
            };
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Compute the number of blocks needed for a given number of tokens.
    fn blocks_needed(num_tokens: usize, block_size: usize) -> usize {
        (num_tokens + block_size - 1) / block_size
    }

    /// Initial block capacity for the batching engine.
    ///
    /// Batched decode can grow its per-request block table as generation crosses
    /// block boundaries, so it should not reserve `prompt + max_tokens` up front.
    /// Large OpenAI-compatible clients commonly send very high `max_tokens`;
    /// making the decode block table that large turns every token into a
    /// long-context operation even when the model stops after a tool call.
    fn initial_batched_decode_blocks_needed(
        prompt_tokens: usize,
        max_tokens: usize,
        block_size: usize,
    ) -> usize {
        let initial_tokens = prompt_tokens.saturating_add(usize::from(max_tokens > 0));
        Self::blocks_needed(initial_tokens, block_size)
    }

    /// Generate text from a prompt using paged KV cache backed by a BlockManager.
    ///
    /// This is the memory-efficient path: blocks are allocated on demand from the
    /// shared BlockManager pool and freed when generation completes.
    pub fn generate_paged(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &PagedKvCache,
    ) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_paged(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
            None,
        )?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Autoregressive generation using paged KV cache.
    ///
    /// Allocates blocks from `block_manager` as needed and frees them when done.
    pub fn generate_from_tokens_paged(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &PagedKvCache,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = block_manager.block_size();
        let max_total = prompt_tokens.len() + params.max_tokens;

        // Pre-allocate blocks for the maximum possible sequence length
        let num_blocks = Self::blocks_needed(max_total, block_size);
        let allocated_blocks = block_manager
            .allocate(num_blocks)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut block_table = BlockTable::new();
        for &block_id in &allocated_blocks {
            block_table.push(block_id);
        }

        // Run generation with paged cache; free blocks on completion (or error)
        let result = self.generate_from_tokens_paged_inner(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            cancel,
        );

        // Always free allocated blocks
        block_manager.free_all(&allocated_blocks);

        result
    }

    /// Generate text from a prompt using shared paged-cache state protected by
    /// short-lived mutexes.
    ///
    /// On backends with CUDA graph replay enabled we preserve the existing
    /// whole-request lock scope because the graph state is runner-global.
    /// On non-CUDA desktop paths (Metal / CPU), blocks are reserved up front,
    /// the block manager is released immediately, and the paged cache is locked
    /// only around prefill / decode forward passes so concurrent requests can
    /// interleave between decode steps.
    pub fn generate_paged_shared(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
    ) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_paged_shared(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
            None,
        )?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Same as [`generate_paged_shared`], but optionally reuses a
    /// block-aligned cached prefix and returns a completed prompt snapshot that
    /// the caller may register after successful generation.
    pub fn generate_paged_shared_tokens_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
        cancel: Option<&CancelHandle>,
    ) -> Result<PrefixCachedGenerationOutput> {
        let output = self.generate_from_tokens_paged_interleaved_with_prefix_cache(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
            cached_prefix,
            cancel,
        )?;

        let text = self
            .tokenizer
            .decode(&output.output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(PrefixCachedGenerationOutput {
            output: GenerationOutput {
                text,
                token_ids: output.output.token_ids,
                finish_reason: output.output.finish_reason,
            },
            registration: output.registration,
            extra_registrations: output.extra_registrations,
            allocated_blocks: output.allocated_blocks,
            prefill_duration: output.prefill_duration,
            decode_duration: output.decode_duration,
        })
    }

    pub fn prepare_paged_batched_decode_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
        capture_prefix_split: bool,
        cancel: Option<&CancelHandle>,
    ) -> Result<PagedBatchedDecodeState> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = {
            let bm_guard = lock_block_manager(block_manager)?;
            bm_guard.block_size()
        };

        let cached_prefix = cached_prefix.filter(|prefix| {
            if prefix.cached_tokens == 0 || prefix.cached_tokens > prompt_tokens.len() {
                return false;
            }

            let exact_candidate = prefix.cached_tokens == prompt_tokens.len();
            let expected_blocks = if exact_candidate {
                Self::blocks_needed(prefix.cached_tokens, block_size)
            } else {
                prefix.cached_tokens / block_size
            };
            let block_shape_valid = prefix.block_ids.len() == expected_blocks;
            let partial_hit = prefix.cached_tokens < prompt_tokens.len()
                && prefix.cached_tokens % block_size == 0;
            let exact_hit = prefix.cached_tokens == prompt_tokens.len()
                && prefix.next_token.as_ref().is_some_and(|next| match next {
                    PagedPrefixNextToken::Logits(_) => true,
                    PagedPrefixNextToken::GreedyToken(_) => params.is_effectively_greedy(),
                });
            block_shape_valid && (partial_hit || exact_hit)
        });

        let cached_blocks = cached_prefix
            .as_ref()
            .map(|prefix| prefix.block_ids.as_slice())
            .unwrap_or(&[]);

        let total_blocks = Self::initial_batched_decode_blocks_needed(
            prompt_tokens.len(),
            params.max_tokens,
            block_size,
        );
        let additional_blocks_needed = total_blocks.saturating_sub(cached_blocks.len());
        let allocated_blocks = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            bm_guard
                .allocate(additional_blocks_needed)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let block_table = append_prefix_block_table(cached_blocks, &allocated_blocks);

        let prepared = self.prepare_paged_batched_decode_with_prefix_blocks(
            prompt_tokens,
            params,
            paged_cache,
            block_table,
            cached_prefix,
            block_size,
            allocated_blocks.clone(),
            capture_prefix_split,
            cancel,
        );

        if prepared.is_err() && !allocated_blocks.is_empty() {
            let mut bm_guard = lock_block_manager(block_manager)?;
            bm_guard.free_all(&allocated_blocks);
        }

        prepared
    }

    /// Same as [`generate_paged_shared`], but accepts an already-tokenized
    /// prompt so API callers do not render/tokenize the same prompt twice.
    ///
    /// The optional `cancel` handle is polled between decode tokens so that
    /// callers (notably `kiln-server`'s `tokio::time::timeout` path) can drain
    /// the still-running blocking work after a request timeout fires, instead
    /// of leaving the closure running with `runner` / `prefix_cache` locks
    /// held — see #664.
    pub fn generate_paged_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let output = self.generate_from_tokens_paged_shared(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
            cancel,
        )?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    fn generate_from_tokens_paged_shared(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let cuda_graph_enabled = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?
            .is_enabled();

        // Phase 12-B'': allocate blocks under a one-shot BlockManager lock and
        // wrap them in `SharedBlockReservation` so the BM lock is released
        // before any forward passes run. The CUDA-graph branch previously held
        // both the BM and the PagedKvCache locks for the entire generation
        // (~2.3 s for a 512-prompt, 128-decode run), which forced concurrent
        // requests onto a serial staircase even with c=8. Phase 12-C removed
        // the global `Mutex<PagedKvCache>` entirely: the cache now uses
        // interior mutability so forward passes can take `&PagedKvCache`
        // concurrently, with disjoint block tables per request providing
        // safety.
        let max_total = prompt_tokens.len() + params.max_tokens;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let result = if cuda_graph_enabled {
            self.generate_from_tokens_paged_cuda_graph_interleaved(
                prompt_tokens,
                params,
                paged_cache,
                &block_table,
                cancel,
            )
        } else {
            self.generate_from_tokens_paged_interleaved(
                prompt_tokens,
                params,
                paged_cache,
                &block_table,
                cancel,
            )
        };

        drop(reservation);
        result
    }

    fn generate_from_tokens_paged_interleaved_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
        cancel: Option<&CancelHandle>,
    ) -> Result<PrefixCachedGenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = {
            let bm_guard = lock_block_manager(block_manager)?;
            bm_guard.block_size()
        };

        let cached_prefix = cached_prefix.filter(|prefix| {
            if prefix.cached_tokens == 0 || prefix.cached_tokens > prompt_tokens.len() {
                return false;
            }

            let exact_candidate = prefix.cached_tokens == prompt_tokens.len();
            let expected_blocks = if exact_candidate {
                Self::blocks_needed(prefix.cached_tokens, block_size)
            } else {
                prefix.cached_tokens / block_size
            };
            let block_shape_valid = prefix.block_ids.len() == expected_blocks;
            let partial_hit = prefix.cached_tokens < prompt_tokens.len()
                && prefix.cached_tokens % block_size == 0;
            let exact_hit = prefix.cached_tokens == prompt_tokens.len()
                && prefix.next_token.as_ref().is_some_and(|next| match next {
                    PagedPrefixNextToken::Logits(_) => true,
                    PagedPrefixNextToken::GreedyToken(_) => params.is_effectively_greedy(),
                });
            block_shape_valid && (partial_hit || exact_hit)
        });

        let cached_blocks = cached_prefix
            .as_ref()
            .map(|prefix| prefix.block_ids.as_slice())
            .unwrap_or(&[]);

        let max_total = prompt_tokens.len() + params.max_tokens;
        let total_blocks = Self::blocks_needed(max_total, block_size);
        let additional_blocks_needed = total_blocks.saturating_sub(cached_blocks.len());
        let allocated_blocks = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            bm_guard
                .allocate(additional_blocks_needed)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let block_table = append_prefix_block_table(cached_blocks, &allocated_blocks);

        let result = self.generate_from_tokens_paged_interleaved_with_prefix_blocks(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            cached_prefix,
            block_size,
            cancel,
        );

        match result {
            Ok(mut output) => {
                output.allocated_blocks = allocated_blocks;
                Ok(output)
            }
            Err(err) => {
                if !allocated_blocks.is_empty() {
                    let mut bm_guard = lock_block_manager(block_manager)?;
                    bm_guard.free_all(&allocated_blocks);
                }
                Err(err)
            }
        }
    }

    fn generate_from_tokens_paged_interleaved_with_prefix_blocks(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cached_prefix: Option<PagedPrefixReuse>,
        block_size: usize,
        cancel: Option<&CancelHandle>,
    ) -> Result<PrefixCachedGenerationOutput> {
        let (cached_tokens, exact_next_token, mut linear_state) = match cached_prefix {
            Some(prefix) => {
                let exact_next_token = if prefix.cached_tokens == prompt_tokens.len() {
                    prefix.next_token
                } else {
                    None
                };
                (prefix.cached_tokens, exact_next_token, prefix.linear_state)
            }
            None => (0, None, self.new_linear_state()?),
        };

        if let Some(next_token) = exact_next_token {
            let decode_start = std::time::Instant::now();
            let output = match next_token {
                PagedPrefixNextToken::Logits(logits) => self.decode_from_prefill_logits(
                    logits,
                    prompt_tokens.len(),
                    params,
                    paged_cache,
                    block_table,
                    &mut linear_state,
                    cancel,
                )?,
                PagedPrefixNextToken::GreedyToken(token) => {
                    anyhow::ensure!(
                        params.is_effectively_greedy(),
                        "greedy cached first token cannot serve non-greedy sampling"
                    );
                    self.decode_from_prefill_token(
                        token,
                        prompt_tokens.len(),
                        params,
                        paged_cache,
                        block_table,
                        &mut linear_state,
                        params.seed,
                        cancel,
                    )?
                }
            };

            return Ok(PrefixCachedGenerationOutput {
                output,
                registration: None,
                extra_registrations: Vec::new(),
                allocated_blocks: Vec::new(),
                prefill_duration: std::time::Duration::ZERO,
                decode_duration: decode_start.elapsed(),
            });
        }

        let prefill_tokens = &prompt_tokens[cached_tokens..];
        anyhow::ensure!(
            !prefill_tokens.is_empty(),
            "non-exact prefix cache hit must leave at least one suffix token"
        );

        let use_greedy_prefill_token = params.is_effectively_greedy()
            && greedy_token_decode_enabled(self.backend.as_ref())
            && !streaming_prefill_enabled_for(
                &self.weights.embed_tokens.device(),
                prefill_tokens.len(),
            );
        // Same capability gate as the batching-engine path: the split
        // snapshot is what makes multi-turn strict-prefix lookups possible
        // (RealPrefixCache only serves longer prompts from block-aligned
        // entries), so a backend opting out here opts out of multi-turn
        // prefix caching entirely.
        let split_pos = prefix_cache_split_snapshot_allowed(self.backend.as_ref())
            .then(|| strict_prompt_prefix_split_pos(prompt_tokens.len(), cached_tokens, block_size))
            .flatten();
        let mut prefill_split_snapshot: Option<RollingPrefixSnapshot> = None;
        let prefill_start = std::time::Instant::now();
        let prefill_source = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(
                &self.weights.embed_tokens.device(),
                prefill_tokens.len(),
            ) {
                if let Some(split_pos) = split_pos {
                    let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                    let _ = model_forward_paged_streaming_with_progress(
                        &*self.backend,
                        head_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        cancel,
                    )
                    .context("prefill forward pass (paged prefix cache, streaming head)")?;
                    prefill_split_snapshot = Some(RollingPrefixSnapshot {
                        position: split_pos,
                        linear_state: linear_state
                            .snapshot()
                            .context("snapshot linear state at streaming prefill split")?,
                    });

                    let tail_tokens = &prompt_tokens[split_pos..];
                    let logits = model_forward_paged_streaming_with_progress(
                        &*self.backend,
                        tail_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        split_pos,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                    .context("prefill forward pass (paged prefix cache, streaming tail)")?;
                    if let Some(cancel) = cancel {
                        cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                    }
                    // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                    PrefillSampleSource::Logits(logits)
                } else {
                    let logits = model_forward_paged_streaming_with_progress(
                        &*self.backend,
                        prefill_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        cancel,
                    )
                    .context("prefill forward pass (paged prefix cache, streaming) failed")?;
                    PrefillSampleSource::Logits(logits)
                }
            } else if use_greedy_prefill_token {
                if let Some(split_pos) = split_pos {
                    // Split the prefill at the last block boundary so the
                    // linear-attention state can be snapshotted at the
                    // cross-turn-safe position (mirrors the batching-engine
                    // path). The head pass's logits are discarded.
                    let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                    let _ = model_forward_paged_last_token(
                        &*self.backend,
                        head_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                    .context("greedy prefill forward pass (paged prefix cache, head) failed")?;
                    prefill_split_snapshot = Some(RollingPrefixSnapshot {
                        position: split_pos,
                        linear_state: linear_state
                            .snapshot()
                            .context("snapshot linear state at greedy prefill split")?,
                    });

                    let tail_tokens = &prompt_tokens[split_pos..];
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    PrefillSampleSource::GreedyToken(
                        model_forward_paged_last_token_greedy(
                            &*self.backend,
                            tail_tokens,
                            &self.weights,
                            &self.config,
                            pc_guard,
                            block_table,
                            split_pos,
                            Some(&mut linear_state),
                            self.active_lora.as_ref(),
                            None,
                        )
                        .context("greedy prefill forward pass (paged prefix cache, tail) failed")?,
                    )
                } else {
                    PrefillSampleSource::GreedyToken(
                        model_forward_paged_last_token_greedy(
                            &*self.backend,
                            prefill_tokens,
                            &self.weights,
                            &self.config,
                            pc_guard,
                            block_table,
                            cached_tokens,
                            Some(&mut linear_state),
                            self.active_lora.as_ref(),
                            None,
                        )
                        .context("greedy prefill forward pass (paged prefix cache) failed")?,
                    )
                }
            } else if let Some(split_pos) = split_pos {
                // Split the prefill at the last block boundary so the
                // linear-attention state can be snapshotted at the
                // cross-turn-safe position (mirrors the batching-engine
                // path). The head pass's logits are discarded.
                let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                let _ = model_forward_paged_last_token(
                    &*self.backend,
                    head_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    cached_tokens,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged prefix cache, head) failed")?;
                prefill_split_snapshot = Some(RollingPrefixSnapshot {
                    position: split_pos,
                    linear_state: linear_state
                        .snapshot()
                        .context("snapshot linear state at prefill split")?,
                });
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(head_tokens.len() as u64);
                }

                let tail_tokens = &prompt_tokens[split_pos..];
                let pc_guard = lock_paged_cache(paged_cache)?;
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    tail_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    split_pos,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged prefix cache, tail) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                }
                // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                PrefillSampleSource::Logits(logits)
            } else {
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    prefill_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    cached_tokens,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged prefix cache) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                }
                // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                PrefillSampleSource::Logits(logits)
            }
        };

        let prefill_duration = prefill_start.elapsed();
        let registration = self.completed_prompt_registration(
            prompt_tokens,
            block_table,
            &linear_state,
            block_size,
            Some(prefill_source.cached_next_token()),
        )?;

        let decode_start = std::time::Instant::now();
        let output = match prefill_source {
            PrefillSampleSource::Logits(logits) => self.decode_from_prefill_logits(
                logits,
                prompt_tokens.len(),
                params,
                paged_cache,
                block_table,
                &mut linear_state,
                cancel,
            )?,
            PrefillSampleSource::GreedyToken(token) => self.decode_from_prefill_token(
                token,
                prompt_tokens.len(),
                params,
                paged_cache,
                block_table,
                &mut linear_state,
                params.seed,
                cancel,
            )?,
        };

        let decode_duration = decode_start.elapsed();
        let mut extra_registrations = Vec::new();
        if let Some(reg) = build_extended_registration(
            prompt_tokens,
            &output.token_ids,
            block_table,
            block_size,
            prefill_split_snapshot,
        ) {
            extra_registrations.push(reg);
        }

        Ok(PrefixCachedGenerationOutput {
            output,
            registration,
            extra_registrations,
            allocated_blocks: Vec::new(),
            prefill_duration,
            decode_duration,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_paged_batched_decode_with_prefix_blocks(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: BlockTable,
        cached_prefix: Option<PagedPrefixReuse>,
        block_size: usize,
        allocated_blocks: Vec<u32>,
        capture_prefix_split: bool,
        cancel: Option<&CancelHandle>,
    ) -> Result<PagedBatchedDecodeState> {
        let (cached_tokens, exact_next_token, mut linear_state) = match cached_prefix {
            Some(prefix) => {
                let exact_next_token = if prefix.cached_tokens == prompt_tokens.len() {
                    prefix.next_token
                } else {
                    None
                };
                (prefix.cached_tokens, exact_next_token, prefix.linear_state)
            }
            None => (0, None, self.new_linear_state()?),
        };

        if let Some(next_token) = exact_next_token {
            let next_token = match next_token {
                PagedPrefixNextToken::Logits(logits) => sample_first_decode_token(&logits, params)?,
                PagedPrefixNextToken::GreedyToken(token) => {
                    anyhow::ensure!(
                        params.is_effectively_greedy(),
                        "greedy cached first token cannot serve non-greedy sampling"
                    );
                    token
                }
            };
            return Ok(PagedBatchedDecodeState {
                block_table,
                linear_state,
                seq_len: prompt_tokens.len(),
                next_token,
                generated_tokens: Vec::new(),
                step_seed: params.seed,
                registration: None,
                allocated_blocks,
                prefill_duration: std::time::Duration::ZERO,
                decode_duration: std::time::Duration::ZERO,
                prompt_tokens: prompt_tokens.to_vec(),
                block_size,
                prefill_split_snapshot: None,
                rolling_snapshot: None,
                id: next_decode_row_id(),
            });
        }

        let prefill_tokens = &prompt_tokens[cached_tokens..];
        anyhow::ensure!(
            !prefill_tokens.is_empty(),
            "prefix cache hit must leave at least one suffix token"
        );

        // Capture an intermediate snapshot at the largest block-aligned
        // position strictly inside the prefill range. This lets us register
        // an extra prefix-cache entry whose token sequence stops *before*
        // the chat template's generation-prompt tail (e.g. Qwen3.5's
        // `<|im_start|>assistant\n<think>\n\n</think>\n\n` when
        // enable_thinking=false), which is the divergent portion that
        // every subsequent turn's prompt does NOT contain — without this
        // snapshot, multi-turn lookups miss because the cached entry's
        // last block contains generation-prompt-only tokens.
        let capture_prefix_split =
            capture_prefix_split && prefix_cache_split_snapshot_allowed(self.backend.as_ref());
        let split_pos = capture_prefix_split
            .then(|| strict_prompt_prefix_split_pos(prompt_tokens.len(), cached_tokens, block_size))
            .flatten();
        let mut prefill_split_snapshot: Option<LinearAttentionState> = None;

        let prefill_start = std::time::Instant::now();
        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(
                &self.weights.embed_tokens.device(),
                prefill_tokens.len(),
            ) {
                if let Some(split_pos) = split_pos {
                    let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                    let _ = model_forward_paged_streaming_with_progress(
                        &*self.backend,
                        head_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        cancel,
                    )
                    .context("batched-engine prefill forward pass (streaming head) failed")?;
                    prefill_split_snapshot = Some(
                        linear_state
                            .snapshot()
                            .context("snapshot linear state at streaming prefill split")?,
                    );

                    let tail_tokens = &prompt_tokens[split_pos..];
                    let logits = model_forward_paged_streaming_with_progress(
                        &*self.backend,
                        tail_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_table,
                        split_pos,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                    .context("batched-engine prefill forward pass (streaming tail) failed")?;
                    if let Some(cancel) = cancel {
                        cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                    }
                    // (#1082) forward returns kt logits; sampler is kt — no bridge.
                    logits
                } else {
                    let logits = model_forward_paged_streaming_with_progress(
                        &*self.backend,
                        prefill_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        cancel,
                    )
                    .context("batched-engine prefill forward pass (streaming) failed")?;
                    // (#1082) forward returns kt logits; sampler is kt — no bridge.
                    logits
                }
            } else if let Some(split_pos) = split_pos {
                // Split the prefill at the last block boundary so we can
                // snapshot the linear-attention state at the cross-turn-safe
                // position. The first call's logits are discarded.
                let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                let _ = model_forward_paged_last_token(
                    &*self.backend,
                    head_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_table,
                    cached_tokens,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("batched-engine prefill forward pass (head) failed")?;
                prefill_split_snapshot = Some(
                    linear_state
                        .snapshot()
                        .context("snapshot linear state at prefill split")?,
                );
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(head_tokens.len() as u64);
                }

                let tail_tokens = &prompt_tokens[split_pos..];
                let pc_guard = lock_paged_cache(paged_cache)?;
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    tail_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_table,
                    split_pos,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("batched-engine prefill forward pass (tail) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                }
                // (#1082) forward returns kt logits; sampler is kt — no bridge.
                logits
            } else {
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    prefill_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_table,
                    cached_tokens,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("batched-engine prefill forward pass failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prefill_tokens.len() as u64);
                }
                // (#1082) forward returns kt logits; sampler is kt — no bridge.
                logits
            }
        };
        let prefill_duration = prefill_start.elapsed();

        let next_token = sample_first_decode_token(&logits, params)?;

        let registration = self.completed_prompt_registration(
            prompt_tokens,
            &block_table,
            &linear_state,
            block_size,
            Some(PagedPrefixNextToken::Logits(logits.clone())),
        )?;

        // Stash the prefill-split snapshot for finish-time registration.
        // Multi-turn agentic workloads against chat templates (notably
        // Qwen3.5 with enable_thinking=false) need a cache entry whose
        // token sequence stops *before* the generation-prompt tail; the
        // split position is the largest block-aligned offset ≤ prompt_len,
        // which for typical prompts lands just before that tail.
        let prefill_split_snapshot = match (split_pos, prefill_split_snapshot) {
            (Some(position), Some(state)) => Some(RollingPrefixSnapshot {
                position,
                linear_state: state,
            }),
            _ => None,
        };

        Ok(PagedBatchedDecodeState {
            block_table,
            linear_state,
            seq_len: prompt_tokens.len(),
            next_token,
            generated_tokens: Vec::new(),
            step_seed: params.seed,
            registration,
            allocated_blocks,
            prefill_duration,
            decode_duration: std::time::Duration::ZERO,
            prompt_tokens: prompt_tokens.to_vec(),
            block_size,
            prefill_split_snapshot,
            rolling_snapshot: None,
            id: next_decode_row_id(),
        })
    }

    pub fn paged_batched_decode_step(
        &self,
        states: &mut [&mut PagedBatchedDecodeState],
        params: &[SamplingParams],
        paged_cache: &PagedKvCache,
    ) -> Result<Vec<TokenId>> {
        anyhow::ensure!(
            states.len() == params.len(),
            "decode state length {} != params length {}",
            states.len(),
            params.len()
        );
        anyhow::ensure!(!states.is_empty(), "batched decode step requires rows");

        let row_count = states.len();
        self.ensure_decode_buffers(row_count)?;
        let input_tokens: Vec<TokenId> = states.iter().map(|state| state.next_token).collect();
        let block_tables: Vec<BlockTable> = states
            .iter()
            .map(|state| state.block_table.clone())
            .collect();
        let sequence_lengths: Vec<usize> = states.iter().map(|state| state.seq_len).collect();
        // Collect stable batched-state-cache fingerprint *before* the
        // `linear_states` mutable borrow below — otherwise the borrow
        // checker rejects the immutable `states.iter()`.
        let row_ids: Vec<u64> = states.iter().map(|state| state.id).collect();
        let all_greedy = params.iter().all(|p| p.temperature == 0.0);
        // (#1082) Capture row-0 sampling context BEFORE the `linear_states`
        // mutable borrow so the Vulkan native single-row decode branch below can
        // sample (temperature > 0) without re-borrowing `states[0]` while
        // `linear_states` holds it mutably. Only row 0 matters (the native branch
        // is row_count == 1) and only on Vulkan; one Option copy + one small Vec
        // clone per step.
        #[cfg(feature = "vulkan")]
        let vk_row0_sampling: Option<(Option<u64>, Vec<TokenId>)> = if row_count == 1 {
            Some((states[0].step_seed, states[0].generated_tokens.clone()))
        } else {
            None
        };
        #[cfg(feature = "vulkan")]
        let vk_batch_sampling_contexts: Option<(Vec<Option<u64>>, Vec<Vec<TokenId>>)> =
            if row_count > 1 && !all_greedy {
                Some((
                    states.iter().map(|state| state.step_seed).collect(),
                    states
                        .iter()
                        .map(|state| state.generated_tokens.clone())
                        .collect(),
                ))
            } else {
                None
            };
        let batch_sampling_contexts: Option<(Vec<Option<u64>>, Vec<Vec<TokenId>>)> = if !all_greedy
        {
            Some((
                states.iter().map(|state| state.step_seed).collect(),
                states
                    .iter()
                    .map(|state| state.generated_tokens.clone())
                    .collect(),
            ))
        } else {
            None
        };
        let mut linear_states: Vec<&mut LinearAttentionState> = states
            .iter_mut()
            .map(|state| &mut state.linear_state)
            .collect();

        let started = std::time::Instant::now();

        // Fast path: when all rows are greedy and the cache is non-FP8, route
        // compatible rows through the contiguous-batched
        // primitive. Uniform-position full-attention batches use a single
        // forward pass with fused argmax. CUDA GDN batches may also enter with
        // mixed sequence lengths because their implementation row-loops through
        // the single-row paged greedy path while preserving scheduler-visible
        // batching.
        let common_seq_len = sequence_lengths[0];
        let positions_uniform = sequence_lengths.iter().all(|&n| n == common_seq_len);
        let cache_is_fp8 = lock_paged_cache(paged_cache)?.is_fp8();
        let has_linear_layers = self.has_linear_attention_layers();
        #[cfg(any(feature = "vulkan", feature = "metal"))]
        let decode_batcher_policy =
            BackendCapabilityQueries::backend_capabilities(self.backend.as_ref()).decode_batcher;
        #[cfg(feature = "vulkan")]
        let sampled_contiguous_resident_decode_ready = decode_batcher_policy
            .use_native_sampled_contiguous_decode
            && decode_batcher_policy.sampled_contiguous_decode_requires_resident_decode
            && ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref());
        #[cfg(feature = "metal")]
        let sampled_contiguous_nonresident_decode_ready = decode_batcher_policy
            .use_native_sampled_contiguous_decode
            && !decode_batcher_policy.sampled_contiguous_decode_requires_resident_decode;
        // `model_forward_paged_decode_contiguous_batch_hidden` already handles
        // per-row positions via dyn-seqlen flash attention for full-attn
        // layers, and the GDN layers operate on the batched
        // `LinearAttentionState` regardless. The `positions_uniform` gate was
        // a leftover from before the dyn-seqlen path landed — dropping it
        // routes every bs > 1 greedy decode through the true-batched path
        // (which also batches the LM-head argmax into a single kernel
        // launch instead of `run_legacy_lm_head_sample_batch`'s per-row
        // narrow + argmax loop).
        let _ = positions_uniform;
        let hip_graph_single_row_ready = row_count == 1
            && paged_decode_replay_primitive_enabled(
                self.backend.as_ref(),
                &self.config,
                1,
                ReplayNativePrimitive::HipGraph,
            )
            && self
                .rocm_graph
                .lock()
                .map(|graph| graph.is_enabled())
                .unwrap_or(false);
        let greedy_route = greedy_batch_route(
            all_greedy,
            cache_is_fp8,
            row_count,
            hip_graph_single_row_ready,
        );
        let try_contiguous_batched = greedy_route == GreedyBatchRoute::Contiguous;

        let mut sampled: Option<Vec<TokenId>> = None;
        // Multi-batch CUDA graph fast path.
        if row_count > 1 && try_contiguous_batched && has_linear_layers {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                linear_states.iter_mut().map(|s| &mut **s).collect();
            let graph_result = {
                let mut graph_runner = self
                    .cuda_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;
                if graph_runner.is_batched_enabled() {
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    graph_runner.decode_step_paged_batched(
                        &*self.backend,
                        &input_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_table_refs,
                        &sequence_lengths,
                        &mut linear_state_refs,
                        self.active_lora.as_ref(),
                    )
                } else {
                    Ok(None)
                }
            };
            match graph_result {
                Ok(Some(tokens)) => sampled = Some(tokens),
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(
                        batch = row_count,
                        error = %err,
                        "batched CUDA graph path errored; falling back to eager"
                    );
                }
            }
        }

        if sampled.is_none() && try_contiguous_batched {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let result = if has_linear_layers {
                let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                    linear_states.iter_mut().map(|s| &mut **s).collect();
                self.decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut linear_state_refs,
                    Some(&row_ids),
                )
            } else {
                let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                self.decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut no_linear_states,
                    Some(&row_ids),
                )
            };
            match result {
                Ok(tokens) => sampled = Some(tokens),
                Err(err)
                    if !decode_hot_path_generic_fallback_enabled_for_backend(&*self.backend) =>
                {
                    return Err(err).context(decode_hot_path_fallback_disabled_context(
                        &*self.backend,
                        "contiguous-batched decode declined",
                    ));
                }
                Err(err) => {
                    tracing::debug!(
                        batch = row_count,
                        error = %err,
                        "contiguous-batched decode declined; falling back to per-row path"
                    );
                }
            }
        }

        // (#1082) Vulkan single-row decode: route the production serving path
        // (batching engine -> paged_batched_decode_step, row_count==1) through
        // native single-submit resident forwards. Greedy uses the token-only
        // resident argmax entry. Stochastic rows try the resident decode +
        // sampler tail first so they only read back one token; unsupported
        // sampler settings fall back to the older resident-logits path.
        // The row-0 sampling context (seed + generated tokens) was snapshotted
        // into `vk_row0_sampling` before the `linear_states` mutable borrow, so
        // the sampler doesn't re-borrow `states[0]` here.
        // Skipped when the contiguous-batched path above already produced tokens
        // (row > 1).
        #[cfg(feature = "vulkan")]
        if sampled.is_none()
            && row_count == 1
            && ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref())
        {
            let token = if params[0].temperature == 0.0 {
                let linear_state = if has_linear_layers {
                    Some(&mut *linear_states[0])
                } else {
                    None
                };
                model_forward_paged_next_token_greedy(
                    &*self.backend,
                    input_tokens[0],
                    &self.weights,
                    &self.config,
                    paged_cache,
                    &block_tables[0],
                    sequence_lengths[0],
                    linear_state,
                    self.active_lora.as_ref(),
                    None,
                )
                .context("vulkan resident single-row greedy decode forward failed")?
            } else {
                let (step_seed, generated) = vk_row0_sampling
                    .as_ref()
                    .expect("vk_row0_sampling captured for row_count == 1");
                let sample_result = if !cache_is_fp8
                    && sampled_contiguous_resident_decode_ready
                    && self.active_lora.is_none()
                {
                    let block_table_refs = [&block_tables[0]];
                    if has_linear_layers {
                        let mut linear_state_refs: [&mut LinearAttentionState; 1] =
                            [&mut *linear_states[0]];
                        self.decode_sample_paged_contiguous_batch_with_ids(
                            &input_tokens[..1],
                            paged_cache,
                            &block_table_refs,
                            &sequence_lengths[..1],
                            &mut linear_state_refs,
                            Some(&row_ids[..1]),
                            &params[..1],
                            std::slice::from_ref(step_seed),
                            std::slice::from_ref(generated),
                        )
                    } else {
                        let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                        self.decode_sample_paged_contiguous_batch_with_ids(
                            &input_tokens[..1],
                            paged_cache,
                            &block_table_refs,
                            &sequence_lengths[..1],
                            &mut no_linear_states,
                            Some(&row_ids[..1]),
                            &params[..1],
                            std::slice::from_ref(step_seed),
                            std::slice::from_ref(generated),
                        )
                    }
                } else {
                    Ok(None)
                };
                match sample_result {
                    Ok(Some(tokens)) => *tokens
                        .first()
                        .context("resident single-row sample returned no token")?,
                    Ok(None) => {
                        let logits = model_forward_paged_last_token_resident(
                            &*self.backend,
                            &input_tokens,
                            &self.weights,
                            &self.config,
                            paged_cache,
                            &block_tables[0],
                            sequence_lengths[0],
                            Some(&mut *linear_states[0]),
                            self.active_lora.as_ref(),
                            None,
                        )
                        .context("vulkan resident single-row decode forward failed")?;
                        let mut row_params = params[0].clone();
                        row_params.seed = *step_seed;
                        sample_with_full_params(&logits, &row_params, generated)?
                    }
                    Err(err) => {
                        return Err(err).context(
                            "resident single-row sample decode failed after native path selection",
                        );
                    }
                }
            };
            sampled = Some(vec![token]);
        }

        #[cfg(feature = "vulkan")]
        if sampled.is_none()
            && row_count > 1
            && !all_greedy
            && !cache_is_fp8
            && sampled_contiguous_resident_decode_ready
            && self.active_lora.is_none()
        {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let (step_seeds, generated_tokens) = vk_batch_sampling_contexts
                .as_ref()
                .expect("vk_batch_sampling_contexts captured for non-greedy row_count > 1");
            let sample_result = if has_linear_layers {
                let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                    linear_states.iter_mut().map(|s| &mut **s).collect();
                self.decode_sample_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut linear_state_refs,
                    Some(&row_ids),
                    params,
                    step_seeds,
                    generated_tokens,
                )
            } else {
                let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                self.decode_sample_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut no_linear_states,
                    Some(&row_ids),
                    params,
                    step_seeds,
                    generated_tokens,
                )
            };
            match sample_result {
                Ok(Some(tokens)) => sampled = Some(tokens),
                Ok(None) => {}
                Err(err) => {
                    return Err(err).context(
                        "resident batched sample decode failed after native path selection",
                    );
                }
            }
            if sampled.is_none() {
                let hidden_result = if has_linear_layers {
                    let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                        linear_states.iter_mut().map(|s| &mut **s).collect();
                    self.decode_hidden_paged_contiguous_batch_with_ids(
                        &input_tokens,
                        paged_cache,
                        &block_table_refs,
                        &sequence_lengths,
                        &mut linear_state_refs,
                        Some(&row_ids),
                    )
                } else {
                    let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                    self.decode_hidden_paged_contiguous_batch_with_ids(
                        &input_tokens,
                        paged_cache,
                        &block_table_refs,
                        &sequence_lengths,
                        &mut no_linear_states,
                        Some(&row_ids),
                    )
                };
                match hidden_result {
                    Ok(hidden) => {
                        let tokens = run_lm_head_sample_batch_with_contexts(
                            &*self.backend,
                            &hidden,
                            &self.weights,
                            &self.config,
                            params,
                            step_seeds,
                            generated_tokens,
                        )
                        .context("sample Vulkan resident multi-row hidden batch")?;
                        sampled = Some(tokens);
                    }
                    Err(err)
                        if !decode_hot_path_generic_fallback_enabled_for_backend(
                            &*self.backend,
                        ) =>
                    {
                        return Err(err).context(decode_hot_path_fallback_disabled_context(
                            &*self.backend,
                            "resident batched hidden decode declined",
                        ));
                    }
                    Err(err) => {
                        tracing::debug!(
                            batch = row_count,
                            error = %err,
                            "resident batched hidden decode declined; falling back to generic hidden path"
                        );
                    }
                }
            }
        }

        #[cfg(feature = "metal")]
        if sampled.is_none()
            && !all_greedy
            && !cache_is_fp8
            && sampled_contiguous_nonresident_decode_ready
            && self.active_lora.is_none()
        {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let (step_seeds, generated_tokens) = batch_sampling_contexts
                .as_ref()
                .expect("batch_sampling_contexts captured for non-greedy Metal decode");
            let sample_result = if has_linear_layers {
                let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                    linear_states.iter_mut().map(|s| &mut **s).collect();
                self.decode_sample_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut linear_state_refs,
                    Some(&row_ids),
                    params,
                    step_seeds,
                    generated_tokens,
                )
            } else {
                let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                self.decode_sample_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut no_linear_states,
                    Some(&row_ids),
                    params,
                    step_seeds,
                    generated_tokens,
                )
            };
            match sample_result {
                Ok(Some(tokens)) => sampled = Some(tokens),
                Ok(None) => {}
                Err(err)
                    if !decode_hot_path_generic_fallback_enabled_for_backend(&*self.backend) =>
                {
                    return Err(err).context(decode_hot_path_fallback_disabled_context(
                        &*self.backend,
                        "Metal sampled decode declined",
                    ));
                }
                Err(err) => {
                    tracing::warn!(
                        batch = row_count,
                        error = %err,
                        "Metal sampled decode declined; falling back to eager hidden sample path"
                    );
                }
            }
            if sampled.is_none()
                && !decode_hot_path_generic_fallback_enabled_for_backend(&*self.backend)
            {
                anyhow::bail!(
                    "{}",
                    decode_hot_path_fallback_disabled_context(
                        &*self.backend,
                        "Metal sampled decode did not produce tokens"
                    )
                );
            }
        }

        // ROCm sampled serving batches need a native decode path even when the
        // HIP-graph bs=1 optimization is disabled or inapplicable. Decode the
        // hidden rows through the contiguous batched ROCm path, then sample from
        // those rows outside the transformer hot path. This keeps
        // NativeRequired from silently depending on the generic fallback when
        // concurrent sampled streams coalesce into row_count > 1.
        if sampled.is_none()
            && !all_greedy
            && matches!(
                BackendIdentity::runtime_device(self.backend.as_ref()),
                kiln_tensor::Device::Rocm(_)
            )
            && (row_count > 1
                || !paged_decode_replay_primitive_enabled(
                    self.backend.as_ref(),
                    &self.config,
                    1,
                    ReplayNativePrimitive::HipGraph,
                )
                || !self
                    .rocm_graph
                    .lock()
                    .map(|g| g.is_enabled())
                    .unwrap_or(false))
        {
            let block_table_refs: Vec<&BlockTable> = block_tables.iter().collect();
            let (step_seeds, generated_tokens) = batch_sampling_contexts
                .as_ref()
                .context("missing sampling contexts for ROCm sampled batched decode")?;
            let hidden_result = if has_linear_layers {
                let mut linear_state_refs: Vec<&mut LinearAttentionState> =
                    linear_states.iter_mut().map(|s| &mut **s).collect();
                self.decode_hidden_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut linear_state_refs,
                    Some(&row_ids),
                )
            } else {
                let mut no_linear_states: [&mut LinearAttentionState; 0] = [];
                self.decode_hidden_paged_contiguous_batch_with_ids(
                    &input_tokens,
                    paged_cache,
                    &block_table_refs,
                    &sequence_lengths,
                    &mut no_linear_states,
                    Some(&row_ids),
                )
            };
            let hidden = hidden_result.context("ROCm sampled batched hidden decode failed")?;
            sampled = Some(
                run_lm_head_sample_batch_with_contexts(
                    &*self.backend,
                    &hidden,
                    &self.weights,
                    &self.config,
                    params,
                    step_seeds,
                    generated_tokens,
                )
                .context("sample ROCm hidden batch")?,
            );
        }

        // R.9: ROCm HIP-graph single-row decode for the batched/batching-engine
        // path. Gated by the ROCm runner, so when disabled `sampled` stays as
        // set above and the cuda/eager block below runs unchanged. Sampled rows
        // use the hidden-only graph path and keep the stochastic lm-head sampler
        // outside the captured graph.
        if sampled.is_none() && hip_graph_single_row_ready {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if params[0].temperature == 0.0 {
                let token = self
                    .rocm_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock ROCm graph runner: {e}"))?
                    .decode_step_paged_greedy(
                        &*self.backend,
                        input_tokens[0],
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_tables[0],
                        sequence_lengths[0],
                        &mut *linear_states[0],
                        self.active_lora.as_ref(),
                        row_ids[0],
                    )
                    .context("batched decode ROCm graph greedy row failed")?;
                sampled = Some(vec![token]);
            } else {
                let (step_seeds, generated_tokens) = batch_sampling_contexts
                    .as_ref()
                    .context("missing row-0 sampling context for ROCm graph sampled decode")?;
                let hidden = self
                    .rocm_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock ROCm graph runner: {e}"))?
                    .decode_step_paged_hidden(
                        &*self.backend,
                        input_tokens[0],
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_tables[0],
                        sequence_lengths[0],
                        &mut *linear_states[0],
                        self.active_lora.as_ref(),
                        row_ids[0],
                    )
                    .context("batched decode ROCm graph hidden row failed")?;
                let token = if let Some(token) = lm_head_sample_backend_decode_if(
                    Some(&*self.backend),
                    &hidden,
                    &self.weights,
                    &self.config,
                    &params[0],
                    step_seeds[0],
                    &generated_tokens[0],
                )
                .context("fused ROCm graph linear_decode_sample failed")?
                {
                    token
                } else {
                    run_lm_head_sample_batch_with_contexts(
                        &*self.backend,
                        &hidden,
                        &self.weights,
                        &self.config,
                        params,
                        step_seeds,
                        generated_tokens,
                    )?[0]
                };
                sampled = Some(vec![token]);
            }
        }

        let sampled = if let Some(tokens) = sampled {
            tokens
        } else {
            if !decode_hot_path_generic_fallback_enabled_for_backend(&*self.backend) {
                anyhow::bail!(
                    "{}",
                    decode_hot_path_fallback_disabled_context(
                        &*self.backend,
                        "no native batched decode path produced tokens"
                    )
                );
            }
            let pc_guard = lock_paged_cache(paged_cache)?;
            let mut graph_runner = self
                .cuda_graph
                .lock()
                .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;
            if graph_runner.is_enabled() && row_count == 1 {
                let row = graph_runner
                    .decode_step_paged(
                        &*self.backend,
                        input_tokens[0],
                        &self.weights,
                        &self.config,
                        pc_guard,
                        &block_tables[0],
                        sequence_lengths[0],
                        &mut *linear_states[0],
                        self.active_lora.as_ref(),
                        Some(row_ids[0]),
                    )
                    .context("batched decode CUDA graph row failed")?;
                // #1082: `decode_step_paged` now returns a kt `Tensor` — feed it
                // straight to the kt-typed samplers, no candle->kt bridge.
                let token = if params[0].temperature == 0.0 {
                    greedy_sample(&row)?
                } else {
                    let mut row_params = params[0].clone();
                    row_params.seed = states[0].step_seed;
                    sample_with_full_params(&row, &row_params, &states[0].generated_tokens)?
                };
                vec![token]
            } else {
                let hidden = model_forward_paged_batched_decode_hidden(
                    &*self.backend,
                    &input_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_tables,
                    &sequence_lengths,
                    &mut linear_states,
                    self.active_lora.as_ref(),
                )
                .context("batched decode forward pass failed")?;

                // Single-row non-greedy: try the backend-fused fused
                // sample path first. It does lm_head + penalty + top-k
                // + softmax + min_p + top_p + categorical entirely
                // on-device and reads back only the 4-byte token.
                // Falls back to the legacy "lm_head + host sample"
                // flow when the backend declines.
                if row_count == 1 && params[0].temperature > 0.0 {
                    let row_hidden = hidden
                        .narrow(0, 0, 1)
                        .context("batched decode hidden row 0")?;
                    if let Some(token) = lm_head_sample_backend_decode_if(
                        Some(&*self.backend),
                        &row_hidden,
                        &self.weights,
                        &self.config,
                        &params[0],
                        states[0].step_seed,
                        &states[0].generated_tokens,
                    )
                    .context("fused linear_decode_sample failed")?
                    {
                        vec![token]
                    } else {
                        // Backend declined (top_k > kernel max, dtype
                        // mismatch, etc.) — fall through to the legacy
                        // lm_head + host sampler.
                        run_legacy_lm_head_sample_batch(
                            &*self.backend,
                            &hidden,
                            &self.weights,
                            &self.config,
                            params,
                            states,
                        )?
                    }
                } else {
                    let step_seeds: Vec<Option<u64>> =
                        states.iter().map(|state| state.step_seed).collect();
                    let generated_tokens: Vec<Vec<TokenId>> = states
                        .iter()
                        .map(|state| state.generated_tokens.clone())
                        .collect();
                    run_lm_head_sample_batch_with_contexts(
                        &*self.backend,
                        &hidden,
                        &self.weights,
                        &self.config,
                        params,
                        &step_seeds,
                        &generated_tokens,
                    )?
                }
            }
        };
        let decode_duration = started.elapsed();

        for state in states.iter_mut() {
            state.seq_len += 1;
            state.decode_duration += decode_duration;
            // When the new seq_len lands on a block boundary, snapshot the
            // recurrent linear-attention state. At finish time the most
            // recent snapshot becomes the basis for an extended prefix-cache
            // entry covering prompt + decoded tokens, which is the only way
            // the next agentic turn (whose prompt = previous prompt +
            // assistant reply + new user input) can hit the cache on
            // anything beyond the original prompt.
            if state.block_size > 0 && state.seq_len % state.block_size == 0 {
                match state.linear_state.snapshot() {
                    Ok(snap) => {
                        state.rolling_snapshot = Some(RollingPrefixSnapshot {
                            position: state.seq_len,
                            linear_state: snap,
                        });
                    }
                    Err(err) => {
                        tracing::warn!(
                            seq_len = state.seq_len,
                            block_size = state.block_size,
                            error = %err,
                            "failed to snapshot linear state at block boundary; \
                             extended prefix-cache entry will not be available for this request"
                        );
                    }
                }
            }
        }

        Ok(sampled)
    }

    pub fn finish_paged_batched_decode(
        &self,
        state: PagedBatchedDecodeState,
        finish_reason: FinishReason,
    ) -> Result<PrefixCachedGenerationOutput> {
        // This is the common completion boundary for normal, cancelled,
        // disconnected, and failed batching-engine requests. Release the unique
        // decode-row owner before token decoding or other fallible finish work so
        // stale graphs and timelines cannot accumulate in a long-running server.
        match self.rocm_graph.lock() {
            Ok(mut graph) => graph.release_decode_row(state.id),
            Err(poisoned) => {
                tracing::warn!(
                    row_id = state.id,
                    "recovering poisoned ROCm graph lock to release finished decode row"
                );
                poisoned.into_inner().release_decode_row(state.id);
            }
        }

        let PagedBatchedDecodeState {
            block_table,
            generated_tokens,
            registration,
            allocated_blocks,
            prefill_duration,
            decode_duration,
            prompt_tokens,
            block_size,
            prefill_split_snapshot,
            rolling_snapshot,
            ..
        } = state;

        let text = self
            .tokenizer
            .decode(&generated_tokens)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        let mut extra_registrations = Vec::new();
        if let Some(reg) = build_extended_registration(
            &prompt_tokens,
            &generated_tokens,
            &block_table,
            block_size,
            prefill_split_snapshot,
        ) {
            extra_registrations.push(reg);
        }
        if let Some(reg) = build_extended_registration(
            &prompt_tokens,
            &generated_tokens,
            &block_table,
            block_size,
            rolling_snapshot,
        ) {
            extra_registrations.push(reg);
        }

        Ok(PrefixCachedGenerationOutput {
            output: GenerationOutput {
                text,
                token_ids: generated_tokens,
                finish_reason,
            },
            registration,
            extra_registrations,
            allocated_blocks,
            prefill_duration,
            decode_duration,
        })
    }

    fn completed_prompt_registration(
        &self,
        prompt_tokens: &[TokenId],
        block_table: &BlockTable,
        linear_state: &LinearAttentionState,
        block_size: usize,
        next_token: Option<PagedPrefixNextToken>,
    ) -> Result<Option<PagedPrefixRegistration>> {
        if prompt_tokens.is_empty() {
            return Ok(None);
        }
        let num_prompt_blocks = if next_token.is_some() {
            Self::blocks_needed(prompt_tokens.len(), block_size)
        } else {
            if prompt_tokens.len() % block_size != 0 {
                return Ok(None);
            }
            prompt_tokens.len() / block_size
        };
        if num_prompt_blocks == 0 || block_table.blocks.len() < num_prompt_blocks {
            return Ok(None);
        }
        Ok(Some(PagedPrefixRegistration {
            prompt_tokens: prompt_tokens.to_vec(),
            block_ids: block_table.blocks[..num_prompt_blocks].to_vec(),
            linear_state: linear_state.snapshot()?,
            next_token,
        }))
    }

    fn decode_from_prefill_logits(
        &self,
        // (#1082) kt-native logits — sampler (greedy_sample/sample_step) is kt.
        logits: kiln_tensor::Tensor,
        seq_len: usize,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        linear_state: &mut LinearAttentionState,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let step_seed = params.seed;

        let next_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, step_seed, &[])?
        };
        self.decode_from_prefill_token(
            next_token,
            seq_len,
            params,
            paged_cache,
            block_table,
            linear_state,
            step_seed,
            cancel,
        )
    }

    fn decode_from_prefill_token(
        &self,
        mut next_token: TokenId,
        mut seq_len: usize,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        linear_state: &mut LinearAttentionState,
        mut step_seed: Option<u64>,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let rocm_owner = RocmDecodeOwnerLease::new(&self.rocm_graph);
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        for _step in 0..params.max_tokens {
            check_cancelled(cancel)?;
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            if self.eos_token_ids.contains(&next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let skip_gdn_state_readback = skip_final_gdn_state_readback_enabled()
                && generated_tokens.len() + 1 >= params.max_tokens;
            next_token = self.decode_next_token_paged_interleaved(
                params,
                next_token,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                step_seed,
                &generated_tokens,
                rocm_owner.row_id(),
                skip_gdn_state_readback,
            )?;
            seq_len += 1;
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Decode one greedy token for multiple compatible paged requests in one
    /// model-forward call.
    ///
    /// This is the scheduler admission primitive for true decode batching: the
    /// caller still owns request readiness, stop handling, and output routing,
    /// while this method owns the row assembly needed to call
    /// `model_forward_paged_decode_contiguous_batch_greedy`.
    ///
    /// Current constraints intentionally mirror the lower-level helper:
    /// non-empty rows, one token per row, one `BlockTable` per row, non-FP8
    /// cache, backend-compatible paged attention windows, and shared base
    /// model/LoRA state for every row. Qwen-style GDN models must pass one
    /// mutable one-row `LinearAttentionState` per row; the method assembles
    /// those into batch state before the forward pass and scatters the updated
    /// rows back afterward.
    pub fn decode_next_tokens_paged_contiguous_batch_greedy(
        &self,
        input_tokens: &[TokenId],
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_states: &mut [&mut LinearAttentionState],
    ) -> Result<Vec<TokenId>> {
        // Stable-id-less call site (e.g. tests). Skip the batched-state
        // cache.
        self.decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
            input_tokens,
            paged_cache,
            block_tables,
            seq_lens,
            linear_states,
            None,
        )
    }

    pub fn decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
        &self,
        input_tokens: &[TokenId],
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_states: &mut [&mut LinearAttentionState],
        row_ids: Option<&[u64]>,
    ) -> Result<Vec<TokenId>> {
        let _resident_scope = GdnRecurrentResidentStateScope::new(&*self.backend);
        let batch = input_tokens.len();
        let profile_stages = profile_decode_batcher_stages_enabled();
        let total_start = profile_stages.then(std::time::Instant::now);
        anyhow::ensure!(batch > 0, "batched decode requires at least one row");
        anyhow::ensure!(
            block_tables.len() == batch && seq_lens.len() == batch,
            "batched decode metadata length mismatch"
        );

        let has_linear_layers = self.has_linear_attention_layers();
        if has_linear_layers {
            anyhow::ensure!(
                linear_states.len() == batch,
                "batched decode requires one LinearAttentionState per row"
            );
        } else {
            anyhow::ensure!(
                linear_states.is_empty(),
                "full-attention-only batched decode does not accept linear states"
            );
        }

        if batch == 1 {
            let stage_start = profile_stages.then(std::time::Instant::now);
            let pc_guard = lock_paged_cache(paged_cache)?;
            #[cfg(feature = "metal")]
            let token = {
                let mut token = None;
                if paged_decode_replay_primitive_enabled(
                    self.backend.as_ref(),
                    &self.config,
                    1,
                    ReplayNativePrimitive::MetalIcb,
                ) && self.active_lora.is_none()
                {
                    let graph_tokens = {
                        let one_tokens = [input_tokens[0]];
                        let one_block_tables = [block_tables[0]];
                        let one_seq_lens = [seq_lens[0]];
                        let linear_state_for_graph = if has_linear_layers {
                            Some(&mut *linear_states[0])
                        } else {
                            None
                        };
                        let mut runner = self.metal_graph.lock().map_err(|e| {
                            anyhow::anyhow!("failed to lock Metal graph runner: {e}")
                        })?;
                        if runner.is_enabled() {
                            runner.decode_step_paged_greedy_batch(
                                &*self.backend,
                                &one_tokens,
                                &self.weights,
                                &self.config,
                                pc_guard,
                                &one_block_tables,
                                &one_seq_lens,
                                linear_state_for_graph,
                                self.active_lora.as_ref(),
                            )?
                        } else {
                            None
                        }
                    };
                    if let Some(graph_tokens) = graph_tokens {
                        anyhow::ensure!(
                            graph_tokens.len() == 1,
                            "Metal graph single-row greedy returned {} tokens",
                            graph_tokens.len()
                        );
                        token = graph_tokens.first().copied();
                    }
                }
                token
            };
            #[cfg(not(feature = "metal"))]
            let token = None;
            let token = (match token {
                Some(token) => Ok(token),
                None => {
                    let linear_state = if has_linear_layers {
                        Some(&mut *linear_states[0])
                    } else {
                        None
                    };
                    model_forward_paged_next_token_greedy(
                        &*self.backend,
                        input_tokens[0],
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_tables[0],
                        seq_lens[0],
                        linear_state,
                        self.active_lora.as_ref(),
                        None,
                    )
                }
            })
            .context("single-row greedy decode forward pass (paged) failed")?;
            finish_decode_batcher_stage_profile("single_forward", batch, stage_start);
            finish_decode_batcher_stage_profile("decode_total", batch, total_start);
            return Ok(vec![token]);
        }

        // #1082 PERF + CRASHER FIX (per-row contiguity partition).
        // The vendored FA2 split-KV paged-decode kernel reads each kBlockN-token
        // K/V tile as ONE physically-contiguous run of pages from a single
        // block_table entry (it never consults the intervening entries — see
        // flash_fwd_kernel.h block_table_idx stride). A fragmented BlockManager
        // free list (concurrent finish->free->re-admit) hands the kernel
        // NON-ADJACENT pages -> CUDA_ERROR_ILLEGAL_ADDRESS / wrong KV. #1445
        // guarded this by forcing the WHOLE batch onto the per-row loop when ANY
        // row was fragmented — but under concurrency the detector fires on the
        // whole batch nearly every step, serializing bs=N into N single-row
        // forwards (the n=64 cliff: 366s p50, 11 tok/s). PARTITION instead:
        // row-loop ONLY the genuinely-fragmented rows and batch the contiguous
        // majority through the fast path. Crash-safe (no non-adjacent pages ever
        // reach the kernel) and a strict superset of #1445's correctness.
        let decode_policy =
            BackendCapabilityQueries::backend_capabilities(self.backend.as_ref()).decode_batcher;
        if decode_policy.partition_noncontiguous_gdn_kv_tiles && has_linear_layers {
            let row_loop_all = gdn_batched_decode_row_loop_debug_enabled();
            let block_size = paged_cache.block_size();
            let noncontig: Vec<bool> = (0..batch)
                .map(|row| {
                    row_loop_all
                        || row_has_noncontiguous_kv_tiles(
                            block_tables[row].blocks.as_slice(),
                            seq_lens[row],
                            block_size,
                        )
                })
                .collect();
            let n_noncontig = noncontig.iter().filter(|&&x| x).count();

            if n_noncontig == batch {
                // Every row fragmented (or debug row-loop-all): the original
                // contiguity-safe per-row loop, unchanged.
                let stage_start = profile_stages.then(std::time::Instant::now);
                let mut tokens = Vec::with_capacity(batch);
                for row in 0..batch {
                    let linear_state =
                        Some(&mut **linear_states.get_mut(row).with_context(|| {
                            format!("missing linear state for CUDA row-loop decode row {row}")
                        })?);
                    let token = {
                        let pc_guard = lock_paged_cache(paged_cache)?;
                        model_forward_paged_next_token_greedy(
                            &*self.backend,
                            input_tokens[row],
                            &self.weights,
                            &self.config,
                            pc_guard,
                            block_tables[row],
                            seq_lens[row],
                            linear_state,
                            self.active_lora.as_ref(),
                            None,
                        )
                        .with_context(|| {
                            format!(
                                "CUDA row-loop greedy decode row {row} forward pass (paged) failed"
                            )
                        })?
                    };
                    tokens.push(token);
                }
                finish_decode_batcher_stage_profile(
                    "cuda_gdn_row_loop_forward",
                    batch,
                    stage_start,
                );
                finish_decode_batcher_stage_profile("decode_total", batch, total_start);
                return Ok(tokens);
            } else if n_noncontig > 0 {
                // MIXED: row-loop only the fragmented rows; batch the contiguous
                // majority through the fast path (recurse on the all-contiguous
                // subset, which falls straight through to it). Scatter back to
                // input order. This is what keeps the fast path alive at n=64 when
                // only a handful of rows hold freshly-recycled pages.
                let stage_start = profile_stages.then(std::time::Instant::now);
                let mut out = vec![0u32; batch];
                // Disjoint partition of the &mut linear states in one pass.
                let mut contig_idx: Vec<usize> = Vec::new();
                let mut contig_states: Vec<&mut LinearAttentionState> = Vec::new();
                let mut noncontig_rows: Vec<(usize, &mut LinearAttentionState)> = Vec::new();
                for (row, ls) in linear_states.iter_mut().enumerate() {
                    if noncontig[row] {
                        noncontig_rows.push((row, &mut **ls));
                    } else {
                        contig_idx.push(row);
                        contig_states.push(&mut **ls);
                    }
                }
                // Fragmented rows: contiguity-safe single-row path.
                for (row, ls) in noncontig_rows.iter_mut() {
                    let token = {
                        let pc_guard = lock_paged_cache(paged_cache)?;
                        model_forward_paged_next_token_greedy(
                            &*self.backend,
                            input_tokens[*row],
                            &self.weights,
                            &self.config,
                            pc_guard,
                            block_tables[*row],
                            seq_lens[*row],
                            Some(&mut **ls),
                            self.active_lora.as_ref(),
                            None,
                        )
                        .with_context(|| {
                            format!(
                                "CUDA mixed-batch fragmented-row greedy decode row {row} failed"
                            )
                        })?
                    };
                    out[*row] = token;
                }
                // Contiguous majority: one fast batched forward via recursion (the
                // all-contiguous subset hits n_noncontig==0 and falls through).
                let contig_tokens: Vec<TokenId> =
                    contig_idx.iter().map(|&i| input_tokens[i]).collect();
                let contig_bts: Vec<&BlockTable> =
                    contig_idx.iter().map(|&i| block_tables[i]).collect();
                let contig_seqlens: Vec<usize> = contig_idx.iter().map(|&i| seq_lens[i]).collect();
                let contig_row_ids: Option<Vec<u64>> =
                    row_ids.map(|r| contig_idx.iter().map(|&i| r[i]).collect());
                let contig_out = self
                    .decode_next_tokens_paged_contiguous_batch_greedy_with_ids(
                        &contig_tokens,
                        paged_cache,
                        &contig_bts,
                        &contig_seqlens,
                        &mut contig_states,
                        contig_row_ids.as_deref(),
                    )
                    .context("CUDA mixed-batch contiguous-subset batched decode failed")?;
                for (k, &row) in contig_idx.iter().enumerate() {
                    out[row] = contig_out[k];
                }
                finish_decode_batcher_stage_profile(
                    "cuda_gdn_partition_forward",
                    batch,
                    stage_start,
                );
                finish_decode_batcher_stage_profile("decode_total", batch, total_start);
                return Ok(out);
            }
            // n_noncontig == 0: all rows contiguous -> fall through to fast path.
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        if has_linear_layers {
            if ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref())
                && ReplayBackend::runtime_decode_resident_pool_ready(
                    self.backend.as_ref(),
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    64,
                )
            {
                for state in linear_states.iter() {
                    state.ensure_gdn_state_resident_kt(&*self.backend)?;
                }
            }
            let any_resident = linear_states
                .iter()
                .any(|state| state.has_any_gdn_state_resident_kt(&*self.backend));
            let all_resident = any_resident
                && linear_states
                    .iter()
                    .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
            if any_resident && !all_resident {
                anyhow::bail!(
                    "mixed kt-resident GDN state rows are not supported for batched decode"
                );
            }
        }

        let all_rows_resident = has_linear_layers
            && linear_states
                .iter()
                .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
        let mut batched_state_cache_hit = false;
        let mut batch_state = if has_linear_layers {
            // Cache lookup: when the same set of per-row state IDs came in
            // last decode step, the cached batched state is already what
            // `from_batch_rows` would re-produce (because we scatter to
            // per-row after every forward — the batched state post-scatter
            // and the per-row states post-scatter are byte-for-byte
            // equivalent). Taking the cached state directly skips the
            // per-step 24-GDN-layer × 2-state-kind tensor `cat` workload
            // (~1.6 ms / step at bs=16). The cache is only consulted when
            // the caller supplied a `row_ids` fingerprint that survives the
            // batching-engine actor's `Vec::remove` shifts.
            let mut cache_guard = self
                .batched_state_cache
                .lock()
                .map_err(|e| anyhow::anyhow!("failed to lock batched state cache: {e}"))?;
            let id_match = match (row_ids, cache_guard.as_ref()) {
                (
                    Some(ids),
                    Some(CachedBatchedState {
                        row_ids: cached, ..
                    }),
                ) => cached == ids,
                _ => false,
            };
            if id_match {
                let cached = cache_guard
                    .take()
                    .expect("id_match implies cache_guard.is_some()");
                batched_state_cache_hit = true;
                drop(cache_guard);
                Some(cached.state)
            } else {
                // Cache miss: discard any stale entry, assemble fresh.
                *cache_guard = None;
                drop(cache_guard);
                let state_refs: Vec<&LinearAttentionState> =
                    linear_states.iter().map(|state| &**state).collect();
                let state = LinearAttentionState::from_batch_rows(&state_refs)?;
                if all_rows_resident {
                    state.assemble_gdn_state_resident_batch_rows_kt(&*self.backend, &state_refs)?;
                }
                Some(state)
            }
        } else {
            None
        };
        if batched_state_cache_hit {
            finish_decode_batcher_stage_profile(
                "batch_state_assemble_cache_hit",
                batch,
                stage_start,
            );
        } else {
            finish_decode_batcher_stage_profile("batch_state_assemble", batch, stage_start);
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        let tokens = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            let graph_tokens = if paged_decode_replay_primitive_enabled(
                self.backend.as_ref(),
                &self.config,
                batch,
                ReplayNativePrimitive::MetalIcb,
            ) {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                if runner.is_enabled() {
                    runner.decode_step_paged_greedy_batch(
                        &*self.backend,
                        input_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_tables,
                        seq_lens,
                        batch_state.as_mut(),
                        self.active_lora.as_ref(),
                    )?
                } else {
                    None
                }
            } else {
                None
            };
            match graph_tokens {
                Some(tokens) => tokens,
                None => model_forward_paged_decode_contiguous_batch_greedy_with_ids(
                    &*self.backend,
                    input_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_tables,
                    seq_lens,
                    batch_state.as_mut(),
                    self.active_lora.as_ref(),
                    row_ids,
                )
                .context("batched greedy decode forward pass (paged) failed")?,
            }
        };
        finish_decode_batcher_stage_profile("batched_forward", batch, stage_start);

        if let Some(state) = batch_state.as_ref() {
            let stage_start = profile_stages.then(std::time::Instant::now);
            if fast_batched_linear_state_scatter_enabled() {
                if !state.scatter_gdn_state_resident_batch_rows_kt(&*self.backend, linear_states)? {
                    state.scatter_batch_rows_replace(linear_states)?;
                }
                finish_decode_batcher_stage_profile(
                    "batch_state_scatter_replace",
                    batch,
                    stage_start,
                );
            } else {
                state.scatter_batch_rows(linear_states)?;
                finish_decode_batcher_stage_profile("batch_state_scatter_copy", batch, stage_start);
            }
        }
        // Park the (now updated) batched state back in the cache so the
        // next decode step on the same id set can skip the
        // `from_batch_rows` cat. The per-row states are byte-for-byte
        // equivalent to this batched state right now (we just scattered),
        // so the next cache-hit reuses correct data; the next cache-miss
        // (different id set, or caller without ids) discards this entry
        // and re-assembles. We only cache when the caller supplied ids;
        // otherwise the next call has no way to match and we'd just hold
        // dead memory.
        if let (Some(state), Some(ids)) = (batch_state.take(), row_ids) {
            if let Ok(mut cache_guard) = self.batched_state_cache.lock() {
                *cache_guard = Some(CachedBatchedState {
                    state,
                    row_ids: ids.to_vec(),
                });
            }
        }
        finish_decode_batcher_stage_profile("decode_total", batch, total_start);

        Ok(tokens)
    }

    /// Decode multiple compatible paged requests through the transformer stack,
    /// returning final hidden states for caller-owned sampling.
    ///
    /// This mirrors the greedy continuous-batch state assembly/scatter path, but
    /// stops before the LM-head argmax so mixed sampling parameters can still be
    /// handled by the existing sampler.
    #[allow(clippy::too_many_arguments)]
    fn decode_sample_paged_contiguous_batch_with_ids(
        &self,
        input_tokens: &[TokenId],
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_states: &mut [&mut LinearAttentionState],
        row_ids: Option<&[u64]>,
        params: &[SamplingParams],
        step_seeds: &[Option<u64>],
        generated_tokens: &[Vec<TokenId>],
    ) -> Result<Option<Vec<TokenId>>> {
        let top_k_values: Vec<u32> = params.iter().map(|param| param.top_k).collect();
        let temperature_values: Vec<f32> = params.iter().map(|param| param.temperature).collect();
        if !SamplingBackend::runtime_supports_linear_decode_sample_batch(
            self.backend.as_ref(),
            &top_k_values,
            &temperature_values,
        ) {
            return Ok(None);
        }

        let _resident_scope = GdnRecurrentResidentStateScope::new(&*self.backend);
        let batch = input_tokens.len();
        let profile_stages = profile_decode_batcher_stages_enabled();
        let total_start = profile_stages.then(std::time::Instant::now);
        anyhow::ensure!(batch > 0, "batched sample decode requires at least one row");
        anyhow::ensure!(
            block_tables.len() == batch
                && seq_lens.len() == batch
                && params.len() == batch
                && step_seeds.len() == batch
                && generated_tokens.len() == batch,
            "batched sample decode metadata length mismatch"
        );

        let mut repetition_values = Vec::with_capacity(batch);
        let mut presence_values = Vec::with_capacity(batch);
        let mut frequency_values = Vec::with_capacity(batch);
        let mut top_p_values = Vec::with_capacity(batch);
        let mut min_p_values = Vec::with_capacity(batch);
        let mut seed_values = Vec::with_capacity(batch);
        let mut history_rows = Vec::new();
        let mut history_indices = Vec::new();
        let mut history_counts = Vec::new();
        for (row_idx, ((param, step_seed), history)) in params
            .iter()
            .zip(step_seeds.iter())
            .zip(generated_tokens.iter())
            .enumerate()
        {
            repetition_values.push(param.repetition_penalty);
            presence_values.push(param.presence_penalty);
            frequency_values.push(param.frequency_penalty);
            top_p_values.push(param.top_p);
            min_p_values.push(param.min_p);
            seed_values.push(sample_seed_for_batch_row(*step_seed, history));
            if param.is_effectively_greedy()
                || param.token_penalties_are_no_op()
                || history.is_empty()
            {
                continue;
            }
            let (indices, counts) = unique_history_counts_for_batch_sample(history);
            for (idx, count) in indices.into_iter().zip(counts.into_iter()) {
                history_rows.push(row_idx as u32);
                history_indices.push(idx);
                history_counts.push(count);
            }
        }

        let has_linear_layers = self.has_linear_attention_layers();
        if has_linear_layers {
            anyhow::ensure!(
                linear_states.len() == batch,
                "batched sample decode requires one LinearAttentionState per row"
            );
        } else {
            anyhow::ensure!(
                linear_states.is_empty(),
                "full-attention-only batched sample decode does not accept linear states"
            );
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        if has_linear_layers {
            if ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref())
                && ReplayBackend::runtime_decode_resident_pool_ready(
                    self.backend.as_ref(),
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    64,
                )
            {
                for state in linear_states.iter() {
                    state.ensure_gdn_state_resident_kt(&*self.backend)?;
                }
            }
            let any_resident = linear_states
                .iter()
                .any(|state| state.has_any_gdn_state_resident_kt(&*self.backend));
            let all_resident = any_resident
                && linear_states
                    .iter()
                    .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
            if any_resident && !all_resident {
                anyhow::bail!(
                    "mixed kt-resident GDN state rows are not supported for batched sample decode"
                );
            }
        }

        let single_row_direct_state = has_linear_layers && batch == 1;
        let all_rows_resident = has_linear_layers
            && linear_states
                .iter()
                .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
        let mut batched_state_cache_hit = false;
        let mut batch_state = if has_linear_layers && !single_row_direct_state {
            let mut cache_guard = self
                .batched_state_cache
                .lock()
                .map_err(|e| anyhow::anyhow!("failed to lock batched state cache: {e}"))?;
            let id_match = match (row_ids, cache_guard.as_ref()) {
                (
                    Some(ids),
                    Some(CachedBatchedState {
                        row_ids: cached, ..
                    }),
                ) => cached == ids,
                _ => false,
            };
            if id_match {
                let cached = cache_guard
                    .take()
                    .expect("id_match implies cache_guard.is_some()");
                batched_state_cache_hit = true;
                drop(cache_guard);
                Some(cached.state)
            } else {
                *cache_guard = None;
                drop(cache_guard);
                let state_refs: Vec<&LinearAttentionState> =
                    linear_states.iter().map(|state| &**state).collect();
                let state = LinearAttentionState::from_batch_rows(&state_refs)?;
                if all_rows_resident {
                    state.assemble_gdn_state_resident_batch_rows_kt(&*self.backend, &state_refs)?;
                }
                Some(state)
            }
        } else {
            None
        };
        if single_row_direct_state {
            finish_decode_batcher_stage_profile(
                "sample_batch_state_direct_row",
                batch,
                stage_start,
            );
        } else if batched_state_cache_hit {
            finish_decode_batcher_stage_profile(
                "sample_batch_state_assemble_cache_hit",
                batch,
                stage_start,
            );
        } else {
            finish_decode_batcher_stage_profile("sample_batch_state_assemble", batch, stage_start);
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        let mut tokens = None;
        #[cfg(feature = "metal")]
        if paged_decode_replay_primitive_enabled(
            self.backend.as_ref(),
            &self.config,
            batch,
            ReplayNativePrimitive::MetalIcb,
        ) && self.active_lora.is_none()
        {
            let pc_guard = lock_paged_cache(paged_cache)?;
            let linear_state_for_graph = if has_linear_layers {
                if single_row_direct_state {
                    Some(&mut *linear_states[0])
                } else {
                    batch_state.as_mut()
                }
            } else {
                None
            };
            let graph_result = {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                runner.decode_step_paged_sample_batch(
                    &*self.backend,
                    input_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_tables,
                    seq_lens,
                    linear_state_for_graph,
                    self.active_lora.as_ref(),
                    &history_rows,
                    &history_indices,
                    &history_counts,
                    &repetition_values,
                    &presence_values,
                    &frequency_values,
                    &temperature_values,
                    &top_k_values,
                    &top_p_values,
                    &min_p_values,
                    &seed_values,
                )
            };
            match graph_result {
                Ok(Some(graph_tokens)) => tokens = Some(graph_tokens),
                Ok(None) => {}
                Err(err) => {
                    tracing::warn!(
                        batch,
                        error = %err,
                        "Metal graph sampled decode declined; falling back to eager sample decode"
                    );
                }
            }
        }
        if tokens.is_none() {
            let pc_guard = lock_paged_cache(paged_cache)?;
            let linear_state_for_forward = if single_row_direct_state {
                Some(&*linear_states[0])
            } else {
                batch_state.as_ref()
            };
            tokens = model_forward_paged_decode_contiguous_batch_sample_with_ids(
                &*self.backend,
                input_tokens,
                &self.weights,
                &self.config,
                pc_guard,
                block_tables,
                seq_lens,
                linear_state_for_forward,
                self.active_lora.as_ref(),
                row_ids,
                &history_rows,
                &history_indices,
                &history_counts,
                &repetition_values,
                &presence_values,
                &frequency_values,
                &temperature_values,
                &top_k_values,
                &top_p_values,
                &min_p_values,
                &seed_values,
            )
            .context("batched sample decode forward pass (paged) failed")?;
        }
        let Some(tokens) = tokens else {
            return Ok(None);
        };
        finish_decode_batcher_stage_profile("sample_batched_forward", batch, stage_start);

        if let Some(state) = batch_state.as_ref() {
            let stage_start = profile_stages.then(std::time::Instant::now);
            if fast_batched_linear_state_scatter_enabled() {
                if !state.scatter_gdn_state_resident_batch_rows_kt(&*self.backend, linear_states)? {
                    state.scatter_batch_rows_replace(linear_states)?;
                }
                finish_decode_batcher_stage_profile(
                    "sample_batch_state_scatter_replace",
                    batch,
                    stage_start,
                );
            } else {
                state.scatter_batch_rows(linear_states)?;
                finish_decode_batcher_stage_profile(
                    "sample_batch_state_scatter_copy",
                    batch,
                    stage_start,
                );
            }
        }

        if let (Some(state), Some(ids)) = (batch_state.take(), row_ids) {
            if let Ok(mut cache_guard) = self.batched_state_cache.lock() {
                *cache_guard = Some(CachedBatchedState {
                    state,
                    row_ids: ids.to_vec(),
                });
            }
        }
        finish_decode_batcher_stage_profile("sample_decode_total", batch, total_start);

        Ok(Some(tokens))
    }

    /// Decode multiple compatible paged requests through the transformer stack,
    /// returning final hidden states for caller-owned sampling.
    ///
    /// This mirrors the greedy continuous-batch state assembly/scatter path, but
    /// stops before the LM-head argmax so mixed sampling parameters can still be
    /// handled by the existing sampler.
    fn decode_hidden_paged_contiguous_batch_with_ids(
        &self,
        input_tokens: &[TokenId],
        paged_cache: &PagedKvCache,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_states: &mut [&mut LinearAttentionState],
        row_ids: Option<&[u64]>,
    ) -> Result<kiln_tensor::Tensor> {
        let _resident_scope = GdnRecurrentResidentStateScope::new(&*self.backend);
        let batch = input_tokens.len();
        let profile_stages = profile_decode_batcher_stages_enabled();
        let total_start = profile_stages.then(std::time::Instant::now);
        anyhow::ensure!(batch > 0, "batched hidden decode requires at least one row");
        anyhow::ensure!(
            block_tables.len() == batch && seq_lens.len() == batch,
            "batched hidden decode metadata length mismatch"
        );

        let has_linear_layers = self.has_linear_attention_layers();
        if has_linear_layers {
            anyhow::ensure!(
                linear_states.len() == batch,
                "batched hidden decode requires one LinearAttentionState per row"
            );
        } else {
            anyhow::ensure!(
                linear_states.is_empty(),
                "full-attention-only batched hidden decode does not accept linear states"
            );
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        if has_linear_layers {
            if ReplayBackend::runtime_supports_resident_decode(self.backend.as_ref())
                && ReplayBackend::runtime_decode_resident_pool_ready(
                    self.backend.as_ref(),
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    64,
                )
            {
                for state in linear_states.iter() {
                    state.ensure_gdn_state_resident_kt(&*self.backend)?;
                }
            }
            let any_resident = linear_states
                .iter()
                .any(|state| state.has_any_gdn_state_resident_kt(&*self.backend));
            let all_resident = any_resident
                && linear_states
                    .iter()
                    .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
            if any_resident && !all_resident {
                anyhow::bail!(
                    "mixed kt-resident GDN state rows are not supported for batched hidden decode"
                );
            }
        }

        let all_rows_resident = has_linear_layers
            && linear_states
                .iter()
                .all(|state| state.has_all_gdn_state_resident_kt(&*self.backend));
        let mut batched_state_cache_hit = false;
        let single_row_direct_state = has_linear_layers && batch == 1;
        let mut batch_state = if has_linear_layers && !single_row_direct_state {
            let mut cache_guard = self
                .batched_state_cache
                .lock()
                .map_err(|e| anyhow::anyhow!("failed to lock batched state cache: {e}"))?;
            let id_match = match (row_ids, cache_guard.as_ref()) {
                (
                    Some(ids),
                    Some(CachedBatchedState {
                        row_ids: cached, ..
                    }),
                ) => cached == ids,
                _ => false,
            };
            if id_match {
                let cached = cache_guard
                    .take()
                    .expect("id_match implies cache_guard.is_some()");
                batched_state_cache_hit = true;
                drop(cache_guard);
                Some(cached.state)
            } else {
                *cache_guard = None;
                drop(cache_guard);
                let state_refs: Vec<&LinearAttentionState> =
                    linear_states.iter().map(|state| &**state).collect();
                let state = LinearAttentionState::from_batch_rows(&state_refs)?;
                if all_rows_resident {
                    state.assemble_gdn_state_resident_batch_rows_kt(&*self.backend, &state_refs)?;
                }
                Some(state)
            }
        } else {
            None
        };
        if single_row_direct_state {
            finish_decode_batcher_stage_profile(
                "hidden_batch_state_direct_row",
                batch,
                stage_start,
            );
        } else if batched_state_cache_hit {
            finish_decode_batcher_stage_profile(
                "hidden_batch_state_assemble_cache_hit",
                batch,
                stage_start,
            );
        } else {
            finish_decode_batcher_stage_profile("hidden_batch_state_assemble", batch, stage_start);
        }

        let stage_start = profile_stages.then(std::time::Instant::now);
        let hidden = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            let linear_state_for_forward = if single_row_direct_state {
                Some(&mut *linear_states[0])
            } else {
                batch_state.as_mut()
            };
            model_forward_paged_decode_contiguous_batch_hidden_with_ids(
                &*self.backend,
                input_tokens,
                &self.weights,
                &self.config,
                pc_guard,
                block_tables,
                seq_lens,
                linear_state_for_forward,
                self.active_lora.as_ref(),
                row_ids,
            )
            .context("batched hidden decode forward pass (paged) failed")?
        };
        finish_decode_batcher_stage_profile("hidden_batched_forward", batch, stage_start);

        if let Some(state) = batch_state.as_ref() {
            let stage_start = profile_stages.then(std::time::Instant::now);
            if fast_batched_linear_state_scatter_enabled() {
                if !state.scatter_gdn_state_resident_batch_rows_kt(&*self.backend, linear_states)? {
                    state.scatter_batch_rows_replace(linear_states)?;
                }
                finish_decode_batcher_stage_profile(
                    "hidden_batch_state_scatter_replace",
                    batch,
                    stage_start,
                );
            } else {
                state.scatter_batch_rows(linear_states)?;
                finish_decode_batcher_stage_profile(
                    "hidden_batch_state_scatter_copy",
                    batch,
                    stage_start,
                );
            }
        }

        if let (Some(state), Some(ids)) = (batch_state.take(), row_ids) {
            if let Ok(mut cache_guard) = self.batched_state_cache.lock() {
                *cache_guard = Some(CachedBatchedState {
                    state,
                    row_ids: ids.to_vec(),
                });
            }
        }
        finish_decode_batcher_stage_profile("hidden_decode_total", batch, total_start);

        Ok(hidden)
    }

    fn decode_next_token_paged_greedy_metal_graph(
        &self,
        input_token: TokenId,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: Option<&mut LinearAttentionState>,
    ) -> Result<Option<TokenId>> {
        #[cfg(feature = "metal")]
        {
            if !paged_decode_replay_primitive_enabled(
                self.backend.as_ref(),
                &self.config,
                1,
                ReplayNativePrimitive::MetalIcb,
            ) || self.active_lora.is_some()
            {
                return Ok(None);
            }

            let pc_guard = lock_paged_cache(paged_cache)?;
            let token_ids = [input_token];
            let block_tables = [block_table];
            let seq_lens = [seq_len];
            let graph_tokens = {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                if !runner.is_enabled() {
                    return Ok(None);
                }
                runner.decode_step_paged_greedy_batch(
                    &*self.backend,
                    &token_ids,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_tables,
                    &seq_lens,
                    linear_state,
                    self.active_lora.as_ref(),
                )?
            };

            if let Some(tokens) = graph_tokens {
                anyhow::ensure!(
                    tokens.len() == 1,
                    "Metal graph single-row greedy returned {} tokens",
                    tokens.len()
                );
                return Ok(tokens.first().copied());
            }
            Ok(None)
        }

        #[cfg(not(feature = "metal"))]
        {
            let _ = (input_token, paged_cache, block_table, seq_len, linear_state);
            Ok(None)
        }
    }

    fn decode_next_token_paged_sample_metal_graph(
        &self,
        params: &SamplingParams,
        input_token: TokenId,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: Option<&mut LinearAttentionState>,
        step_seed: Option<u64>,
        history: &[TokenId],
    ) -> Result<Option<TokenId>> {
        #[cfg(feature = "metal")]
        {
            if params.is_effectively_greedy()
                || !paged_decode_replay_primitive_enabled(
                    self.backend.as_ref(),
                    &self.config,
                    1,
                    ReplayNativePrimitive::MetalIcb,
                )
                || self.active_lora.is_some()
            {
                return Ok(None);
            }
            let top_k = [params.top_k];
            let temperatures = [params.temperature];
            if !SamplingBackend::runtime_supports_linear_decode_sample_batch(
                self.backend.as_ref(),
                &top_k,
                &temperatures,
            ) {
                return Ok(None);
            }

            let mut history_rows = Vec::new();
            let mut history_indices = Vec::new();
            let mut history_counts = Vec::new();
            if !params.token_penalties_are_no_op() && !history.is_empty() {
                let (indices, counts) = unique_history_counts_for_batch_sample(history);
                for (idx, count) in indices.into_iter().zip(counts.into_iter()) {
                    history_rows.push(0);
                    history_indices.push(idx);
                    history_counts.push(count);
                }
            }

            let pc_guard = lock_paged_cache(paged_cache)?;
            let token_ids = [input_token];
            let block_tables = [block_table];
            let seq_lens = [seq_len];
            let repetition_penalties = [params.repetition_penalty];
            let presence_penalties = [params.presence_penalty];
            let frequency_penalties = [params.frequency_penalty];
            let top_p = [params.top_p];
            let min_p = [params.min_p];
            let seeds = [sample_seed_for_batch_row(step_seed, history)];
            let graph_tokens = {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                if !runner.is_enabled() {
                    return Ok(None);
                }
                runner.decode_step_paged_sample_batch(
                    &*self.backend,
                    &token_ids,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    &block_tables,
                    &seq_lens,
                    linear_state,
                    self.active_lora.as_ref(),
                    &history_rows,
                    &history_indices,
                    &history_counts,
                    &repetition_penalties,
                    &presence_penalties,
                    &frequency_penalties,
                    &temperatures,
                    &top_k,
                    &top_p,
                    &min_p,
                    &seeds,
                )?
            };

            if let Some(tokens) = graph_tokens {
                anyhow::ensure!(
                    tokens.len() == 1,
                    "Metal graph single-row sampled decode returned {} tokens",
                    tokens.len()
                );
                return Ok(tokens.first().copied());
            }
            Ok(None)
        }

        #[cfg(not(feature = "metal"))]
        {
            let _ = (
                params,
                input_token,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                step_seed,
                history,
            );
            Ok(None)
        }
    }

    fn decode_next_token_paged_interleaved(
        &self,
        params: &SamplingParams,
        input_token: TokenId,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        step_seed: Option<u64>,
        history: &[TokenId],
        graph_row_id: u64,
        skip_gdn_state_readback: bool,
    ) -> Result<TokenId> {
        let _resident_scope = GdnRecurrentResidentStateScope::new(&*self.backend);
        let _skip_scope =
            crate::forward::VulkanSkipGdnStateReadbackScope::new(skip_gdn_state_readback);
        if params.is_effectively_greedy() && greedy_token_decode_enabled(self.backend.as_ref()) {
            let linear_state_for_graph = if self.has_linear_attention_layers() {
                Some(&mut *linear_state)
            } else {
                None
            };
            if let Some(token) = self
                .decode_next_token_paged_greedy_metal_graph(
                    input_token,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state_for_graph,
                )
                .context("greedy Metal graph decode forward pass (paged) failed")?
            {
                if skip_gdn_state_readback {
                    linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
                }
                return Ok(token);
            }
            let pc_guard = lock_paged_cache(paged_cache)?;
            let token = {
                let mut runner = self
                    .metal_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock Metal graph runner: {e}"))?;
                if runner.is_enabled() {
                    runner.decode_step_paged_greedy(
                        &*self.backend,
                        input_token,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        seq_len,
                        linear_state,
                        self.active_lora.as_ref(),
                    )
                } else {
                    model_forward_paged_next_token_greedy(
                        &*self.backend,
                        input_token,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        seq_len,
                        Some(linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                }
            }
            .context("greedy Metal decode forward pass (paged) failed")?;
            if skip_gdn_state_readback {
                linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
            }
            return Ok(token);
        }

        if !params.is_effectively_greedy()
            && paged_decode_replay_primitive_enabled(
                self.backend.as_ref(),
                &self.config,
                1,
                ReplayNativePrimitive::MetalIcb,
            )
        {
            let linear_state_for_graph = if self.has_linear_attention_layers() {
                Some(&mut *linear_state)
            } else {
                None
            };
            if let Some(token) = self
                .decode_next_token_paged_sample_metal_graph(
                    params,
                    input_token,
                    paged_cache,
                    block_table,
                    seq_len,
                    linear_state_for_graph,
                    step_seed,
                    history,
                )
                .context("sampled Metal graph decode forward pass (paged) failed")?
            {
                if skip_gdn_state_readback {
                    linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
                }
                return Ok(token);
            }
        }

        // R.9: ROCm HIP-graph decode. On a Rocm device, route the step through
        // the graph runner (capture/replay, with eager fallback). When the
        // runner is disabled via `KILN_ROCM_GRAPHS=0`, this is skipped entirely
        // and the eager path below runs unchanged.
        if paged_decode_replay_primitive_enabled(
            self.backend.as_ref(),
            &self.config,
            1,
            ReplayNativePrimitive::HipGraph,
        ) {
            let maybe_logits = {
                let mut runner = self
                    .rocm_graph
                    .lock()
                    .map_err(|e| anyhow::anyhow!("failed to lock ROCm graph runner: {e}"))?;
                if runner.is_enabled() {
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    Some(
                        runner
                            .decode_step_paged(
                                &*self.backend,
                                input_token,
                                &self.weights,
                                &self.config,
                                pc_guard,
                                block_table,
                                seq_len,
                                linear_state,
                                self.active_lora.as_ref(),
                                graph_row_id,
                            )
                            .context("ROCm graph decode step failed")?,
                    )
                } else {
                    None
                }
            };
            if let Some(logits) = maybe_logits {
                let token = if params.is_effectively_greedy() {
                    greedy_sample(&logits)
                } else {
                    sample_step(&logits, params, step_seed, history)
                }?;
                if skip_gdn_state_readback {
                    linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
                }
                return Ok(token);
            }
        }

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            model_forward_paged(
                &*self.backend,
                &[input_token],
                &self.weights,
                &self.config,
                pc_guard,
                block_table,
                seq_len,
                Some(linear_state),
                self.active_lora.as_ref(),
                None,
            )
            .context("decode forward pass (paged) failed")?
        };
        // (#1082) forward returns kt logits; sampler is kt — no bridge.

        let token = if params.is_effectively_greedy() {
            greedy_sample(&logits)
        } else {
            sample_step(&logits, params, step_seed, history)
        }?;
        if skip_gdn_state_readback {
            linear_state.evict_gdn_recurrent_resident_states(&*self.backend);
        }
        Ok(token)
    }

    fn decode_next_token_paged_interleaved_or_batched(
        &self,
        params: &SamplingParams,
        input_token: TokenId,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        step_seed: Option<u64>,
        decode_batcher: Option<&DecodeBatcher>,
        history: &[TokenId],
        graph_row_id: u64,
        skip_gdn_state_readback: bool,
    ) -> Result<TokenId> {
        if params.is_effectively_greedy()
            && let Some(batcher) = decode_batcher
        {
            match batcher.decode_next_token_greedy(
                input_token,
                block_table,
                seq_len,
                linear_state,
                skip_gdn_state_readback,
            )? {
                DecodeBatcherDecode::Decoded(token) => return Ok(token),
                DecodeBatcherDecode::RunnerBusy => {}
            }
        }

        self.decode_next_token_paged_interleaved(
            params,
            input_token,
            paged_cache,
            block_table,
            seq_len,
            linear_state,
            step_seed,
            history,
            graph_row_id,
            skip_gdn_state_readback,
        )
    }

    pub fn generate_paged_speculative_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        spec_config: &SpeculativeConfig,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        if params.thinking_budget.is_some() {
            return self.generate_paged_shared_tokens(
                prompt_tokens,
                params,
                block_manager,
                paged_cache,
                cancel,
            );
        }
        anyhow::ensure!(
            params.temperature == 0.0,
            "paged skip-layer speculative decode is greedy-only"
        );
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let output = self.generate_from_tokens_paged_speculative_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            spec_config,
            cancel,
        );

        drop(reservation);

        let output = output?;
        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    fn generate_from_tokens_paged_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(
                &self.weights.embed_tokens.device(),
                prompt_tokens.len(),
            ) {
                model_forward_paged_streaming_with_progress(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    cancel,
                )
                .context("prefill forward pass (paged, streaming) failed")?
            } else {
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prompt_tokens.len() as u64);
                }
                logits
            }
        };
        // (#1082) forward returns kt logits; sampler is kt — no bridge.

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        let mut next_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, step_seed, &[])?
        };
        let rocm_owner = RocmDecodeOwnerLease::new(&self.rocm_graph);

        for _step in 0..params.max_tokens {
            check_cancelled(cancel)?;
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            if self.eos_token_ids.contains(&next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let skip_gdn_state_readback = skip_final_gdn_state_readback_enabled()
                && generated_tokens.len() + 1 >= params.max_tokens;
            next_token = self.decode_next_token_paged_interleaved(
                params,
                next_token,
                paged_cache,
                block_table,
                seq_len,
                &mut linear_state,
                step_seed,
                &generated_tokens,
                rocm_owner.row_id(),
                skip_gdn_state_readback,
            )?;
            seq_len += 1;
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// CUDA-graph variant of the interleaved decode path (Phase 12-B'').
    ///
    /// Mirrors `generate_from_tokens_paged_inner` (the path the old CUDA-graph
    /// branch used) but takes `paged_cache: &PagedKvCache` (Phase 12-C
    /// removed the surrounding `Mutex`; the cache uses interior mutability
    /// for concurrent `&self` writes). The CUDA graph runner mutex is still
    /// acquired per decode step, so that concurrent c=8 requests can
    /// interleave on a per-step granularity rather
    /// than serialising on a generation-lifetime lock. Blocks are still
    /// allocated once up-front by the caller (`generate_from_tokens_paged_shared`)
    /// and freed via `SharedBlockReservation` when the caller drops the
    /// reservation guard, mirroring the non-graph interleaved path.
    fn generate_from_tokens_paged_cuda_graph_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let mut linear_state = self.new_linear_state()?;

        // Prefill: lock the paged cache for one forward pass and drop it
        // before the decode loop starts. The decode loop then re-acquires the
        // cache per step.
        let streaming_prefill =
            streaming_prefill_enabled_for(&self.weights.embed_tokens.device(), prompt_tokens.len());
        let prefill_source = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill {
                let logits = model_forward_paged_streaming(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("prefill forward pass (paged, streaming) failed")?;
                // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                PrefillSampleSource::Logits(logits)
            } else if params.is_effectively_greedy()
                && greedy_token_decode_enabled(self.backend.as_ref())
            {
                PrefillSampleSource::GreedyToken(
                    model_forward_paged_last_token_greedy(
                        &*self.backend,
                        prompt_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        0,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )
                    .context("greedy prefill forward pass (paged) failed")?,
                )
            } else {
                let logits = model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged) failed")?;
                if let Some(cancel) = cancel {
                    cancel.report_prefill_tokens_completed(prompt_tokens.len() as u64);
                }
                // (#1082) kt-native logits — sampler is kt now; no candle bridge.
                PrefillSampleSource::Logits(logits)
            }
        };

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        let mut next_token = match prefill_source {
            PrefillSampleSource::GreedyToken(token) => token,
            PrefillSampleSource::Logits(logits) => {
                if params.is_effectively_greedy() {
                    greedy_sample(&logits)?
                } else {
                    sample_step(&logits, params, step_seed, &[])?
                }
            }
        };

        for _step in 0..params.max_tokens {
            check_cancelled(cancel)?;
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            if self.eos_token_ids.contains(&next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            next_token = if params.is_effectively_greedy()
                && greedy_token_decode_enabled(self.backend.as_ref())
            {
                let linear_state_for_graph = if self.has_linear_attention_layers() {
                    Some(&mut linear_state)
                } else {
                    None
                };
                if let Some(token) = self
                    .decode_next_token_paged_greedy_metal_graph(
                        next_token,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state_for_graph,
                    )
                    .context("greedy Metal graph decode forward pass (paged) failed")?
                {
                    seq_len += 1;
                    token
                } else {
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    let token = model_forward_paged_next_token_greedy(
                        &*self.backend,
                        next_token,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        seq_len,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )?;
                    seq_len += 1;
                    token
                }
            } else {
                // CUDA graph decode step: acquire the graph runner and the
                // paged cache for one step, then drop both before sampling so
                // concurrent requests can interleave on the next step.
                let logits = {
                    let mut graph_runner = self
                        .cuda_graph
                        .lock()
                        .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;
                    let pc_guard = lock_paged_cache(paged_cache)?;
                    graph_runner.decode_step_paged(
                        &*self.backend,
                        next_token,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        seq_len,
                        &mut linear_state,
                        self.active_lora.as_ref(),
                        None,
                    )?
                };
                seq_len += 1;
                // #1082: `decode_step_paged` now returns kt — feed `sample_step`
                // directly, no candle->kt bridge.
                sample_step(&logits, params, step_seed, &generated_tokens)?
            };
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    fn generate_from_tokens_paged_speculative_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        spec_config: &SpeculativeConfig,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(
                &self.weights.embed_tokens.device(),
                prompt_tokens.len(),
            ) {
                model_forward_paged_streaming(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("prefill forward pass (paged skip-layer, streaming) failed")?
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged skip-layer) failed")?
            }
        };
        // (#1082) forward returns kt logits; sampler is kt — no bridge.

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut base_pos = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut last_token = greedy_sample(&logits)?;

        loop {
            check_cancelled(cancel)?;
            if generated_tokens.len() >= params.max_tokens {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                });
            }

            if self.eos_token_ids.contains(&last_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(last_token);
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                });
            }

            let remaining = params.max_tokens - generated_tokens.len();
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: spec_config.num_speculative_tokens.min(remaining),
                draft_layers: spec_config.draft_layers,
            };

            let result = {
                let pc_guard = lock_paged_cache(paged_cache)?;
                speculative_decode_step_paged_greedy(
                    &*self.backend,
                    last_token,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    base_pos,
                    &mut linear_state,
                    &mut draft_linear_state,
                    &effective_config,
                    params,
                    &self.eos_token_ids,
                    self.active_lora.as_ref(),
                )
                .context("paged skip-layer speculative decode step failed")?
            };
            base_pos += result.base_advance;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                    });
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                generated_tokens.push(token);
                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(GenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::MaxTokens,
                    });
                }
            }

            last_token = *result.accepted_tokens.last().unwrap();
            if result.hit_eos {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Inner generation loop using paged KV cache (blocks already allocated).
    fn generate_from_tokens_paged_inner(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cancel: Option<&CancelHandle>,
    ) -> Result<GenerationOutput> {
        let mut linear_state = self.new_linear_state()?;

        // Prefill: forward pass on all prompt tokens (never uses CUDA graphs).
        // Long Metal prompts use tiled streaming prefill by default; env
        // overrides can force either path.
        let streaming_prefill =
            streaming_prefill_enabled_for(&self.weights.embed_tokens.device(), prompt_tokens.len());
        let prefill_source = if streaming_prefill {
            let logits = model_forward_paged_streaming(
                &*self.backend,
                prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
            .context("prefill forward pass (paged, streaming) failed")?;
            // (#1082) kt-native logits — sampler is kt now; no candle bridge.
            PrefillSampleSource::Logits(logits)
        } else if params.is_effectively_greedy()
            && greedy_token_decode_enabled(self.backend.as_ref())
        {
            PrefillSampleSource::GreedyToken(
                model_forward_paged_last_token_greedy(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    paged_cache,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("greedy prefill forward pass (paged) failed")?,
            )
        } else {
            let logits = model_forward_paged_last_token(
                &*self.backend,
                prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                None,
            )
            .context("prefill forward pass (paged) failed")?;
            // (#1082) kt-native logits — sampler is kt now; no candle bridge.
            PrefillSampleSource::Logits(logits)
        };

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;

        // Acquire the CUDA graph runner for decode steps
        let mut graph_runner = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;

        let mut next_token = match prefill_source {
            PrefillSampleSource::GreedyToken(token) => token,
            PrefillSampleSource::Logits(logits) => {
                if params.is_effectively_greedy() {
                    greedy_sample(&logits)?
                } else {
                    sample_step(&logits, params, step_seed, &[])?
                }
            }
        };

        for _step in 0..params.max_tokens {
            check_cancelled(cancel)?;
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            // Check for EOS
            if self.eos_token_ids.contains(&next_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(next_token);

            // Check stop sequences
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            next_token = if params.is_effectively_greedy()
                && greedy_token_decode_enabled(self.backend.as_ref())
            {
                let linear_state_for_graph = if self.has_linear_attention_layers() {
                    Some(&mut linear_state)
                } else {
                    None
                };
                if let Some(token) = self
                    .decode_next_token_paged_greedy_metal_graph(
                        next_token,
                        paged_cache,
                        block_table,
                        seq_len,
                        linear_state_for_graph,
                    )
                    .context("greedy Metal graph decode forward pass (paged) failed")?
                {
                    seq_len += 1;
                    token
                } else {
                    let token = model_forward_paged_next_token_greedy(
                        &*self.backend,
                        next_token,
                        &self.weights,
                        &self.config,
                        paged_cache,
                        block_table,
                        seq_len,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                        None,
                    )?;
                    seq_len += 1;
                    token
                }
            } else {
                // Decode step: use CUDA graph runner (captures/replays when enabled)
                let logits = graph_runner.decode_step_paged(
                    &*self.backend,
                    next_token,
                    &self.weights,
                    &self.config,
                    paged_cache,
                    block_table,
                    seq_len,
                    &mut linear_state,
                    self.active_lora.as_ref(),
                    None,
                )?;
                seq_len += 1;
                // #1082: `decode_step_paged` now returns kt — feed `sample_step`
                // directly, no candle->kt bridge.
                sample_step(&logits, params, step_seed, &generated_tokens)?
            };
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Generate text using self-speculative decoding (skip-layer draft).
    ///
    /// The first `spec_config.draft_layers` layers of the model propose candidate
    /// tokens, and the full model verifies them in a single forward pass. Accepted
    /// tokens are emitted in batches, giving 1.5-2.5x decode speedup.
    ///
    /// Falls back to standard generation if speculative config is invalid.
    pub fn generate_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
        spec_config: &SpeculativeConfig,
    ) -> Result<GenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_speculative(&prompt_tokens, params, spec_config)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(GenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
        })
    }

    /// Speculative generation loop operating on token IDs.
    ///
    /// 1. Prefill: standard full-model forward pass on the prompt.
    /// 2. Decode: draft K tokens with first N layers, verify with full model,
    ///    accept/reject via rejection sampling.
    pub fn generate_from_tokens_speculative(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        spec_config: &SpeculativeConfig,
    ) -> Result<GenerationOutput> {
        use rand::SeedableRng;

        if params.thinking_budget.is_some() {
            return self.generate_from_tokens(prompt_tokens, params);
        }
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        // Verification writes the full speculative window (`last_token + k`)
        // before the loop commits accepted tokens, so flat KV needs temporary
        // headroom beyond the user-visible max token budget.
        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        // Prefill: full model forward pass on all prompt tokens
        let logits = model_forward_kt(
            &*self.backend,
            prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::make_rng::<rand::rngs::StdRng>(),
        };

        // Sample first token from prefill logits
        let mut last_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, params.seed, &[])?
        };

        loop {
            // Check if we've hit max_tokens
            if generated_tokens.len() >= params.max_tokens {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                });
            }

            // Check for EOS
            if self.eos_token_ids.contains(&last_token) {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }

            generated_tokens.push(last_token);

            // Check stop sequences
            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(GenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                            });
                        }
                    }
                }
            }

            // Run one speculative decode step
            let remaining = params.max_tokens - generated_tokens.len();
            let effective_k = spec_config.num_speculative_tokens.min(remaining);
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: effective_k,
                draft_layers: spec_config.draft_layers,
            };

            let result = speculative_decode_step(
                &*self.backend,
                last_token,
                &self.weights,
                &self.config,
                &mut kv_cache,
                &mut linear_state,
                &mut draft_linear_state,
                &effective_config,
                params,
                &self.eos_token_ids,
                &mut rng,
                self.active_lora.as_ref(),
            )
            .context("speculative decode step failed")?;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                    });
                }
                // No tokens accepted and no EOS — shouldn't happen normally,
                // but fall back to sampling from the verification logits.
                // Break to avoid infinite loop.
                break;
            }

            // Add accepted tokens (except the last one which becomes last_token)
            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                generated_tokens.push(token);

                // Check stop sequences after each token
                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(GenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    return Ok(GenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::MaxTokens,
                    });
                }
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                return Ok(GenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                });
            }
        }

        Ok(GenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
        })
    }

    /// Generate text using native MTP (Multi-Token Prediction) speculative decoding.
    ///
    /// Uses the model's pretrained MTP head to draft a single candidate token per
    /// step (Qwen3.5-4B ships `num_nextn_predict_layers=1`), which the base model
    /// then verifies in a fused forward pass that emits both the draft-position
    /// target and a bonus token for the accept case.
    ///
    /// Requires the checkpoint to carry `mtp.*` tensors; returns an error
    /// otherwise. Greedy-only (temperature == 0); the stochastic
    /// rejection-sampling variant is a follow-up.
    ///
    /// Reports α (acceptance rate) via the returned [`MtpGenerationOutput`] so
    /// bench callers can publish it alongside throughput numbers.
    pub fn generate_mtp_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<MtpGenerationOutput> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        let output = self.generate_from_tokens_mtp_speculative(&prompt_tokens, params)?;

        let text = self
            .tokenizer
            .decode(&output.token_ids)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to decode output tokens")?;

        Ok(MtpGenerationOutput {
            text,
            token_ids: output.token_ids,
            finish_reason: output.finish_reason,
            draft_accepted_count: output.draft_accepted_count,
            total_draft_attempts: output.total_draft_attempts,
        })
    }

    /// Native MTP speculative generation operating on token IDs.
    ///
    /// 1. Prefill: paged forward pass on the prompt that returns both logits and
    ///    the last-row pre-final-norm hidden state (`h_prev`).
    /// 2. Decode: per iteration, call [`speculative_mtp_decode_step`] which
    ///    drafts via the MTP head, verifies via the base model, and reports the
    ///    accepted tokens plus advanced positions for the next call.
    ///
    /// Two paged caches are used: the base cache (sized for the model's
    /// full-attention layers) and a 1-layer MTP cache. They have independent
    /// position counters because the MTP layer only commits a slot on accept.
    pub fn generate_from_tokens_mtp_speculative(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
    ) -> Result<MtpGenerationOutput> {
        use rand::SeedableRng;

        if params.thinking_budget.is_some() {
            let output = self.generate_from_tokens(prompt_tokens, params)?;
            return Ok(MtpGenerationOutput {
                text: output.text,
                token_ids: output.token_ids,
                finish_reason: output.finish_reason,
                draft_accepted_count: 0,
                total_draft_attempts: 0,
            });
        }
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        anyhow::ensure!(
            self.weights.mtp.is_some(),
            "generate_mtp_speculative requires the checkpoint to carry mtp.* tensors \
             (Qwen3.5-4B native MTP head)"
        );
        anyhow::ensure!(
            params.temperature == 0.0,
            "generate_mtp_speculative currently only supports greedy decoding (temperature == 0)"
        );

        // Block size matches the kiln-core default + the bench convention
        // (#1082: 16 -> 64 so each FA2 kBlockN=64 tile is one physical page).
        const BLOCK_SIZE: usize = 64;

        let max_total = prompt_tokens.len() + params.max_tokens;
        // (#1082) kt-native paged cache — `PagedKvCacheKt::new` allocates pools
        // on the model's runtime `Device` (kiln is single-GPU).
        let cache_device =
            paged_cache_device(self.backend.as_ref(), &self.weights.embed_tokens.device())?;
        let dtype = paged_cache_kt_dtype(self.config.dtype);

        // Two independent paged caches:
        //   * `base_cache` covers the model's full-attention layers.
        //   * `mtp_cache` is a single-layer cache for the MTP block.
        // Each gets its own block table mapping logical block i -> physical i.
        let num_blocks = Self::blocks_needed(max_total, BLOCK_SIZE);
        let base_cache = PagedKvCache::new(
            self.config.num_full_attention_layers,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            cache_device,
        )?;
        let mtp_cache = PagedKvCache::new(
            1,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            cache_device,
        )?;
        let mut base_block_table = BlockTable::new();
        let mut mtp_block_table = BlockTable::new();
        for i in 0..num_blocks as u32 {
            base_block_table.push(i);
            mtp_block_table.push(i);
        }

        let mut linear_state = self.new_linear_state()?;

        // Prefill: feed the prompt through the base model and capture the
        // post-final-norm last hidden row as the seed `h_prev`.
        let (prefill_logits_kt, h_prev_kt) = if streaming_prefill_enabled_for(
            &self.weights.embed_tokens.device(),
            prompt_tokens.len(),
        ) {
            model_forward_paged_streaming_last_token_with_last_hidden(
                &*self.backend,
                prompt_tokens,
                &self.weights,
                &self.config,
                &base_cache,
                &base_block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
            .context("mtp streaming prefill forward pass failed")?
        } else {
            model_forward_paged_last_token_with_last_hidden(
                &*self.backend,
                prompt_tokens,
                &self.weights,
                &self.config,
                &base_cache,
                &base_block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                None,
            )
            .context("mtp prefill forward pass failed")?
        };
        // (#1082) MTP speculative step + speculative.rs are fully kt now —
        // `h_prev`/`prefill_logits` stay kt; no candle bridge.
        let prefill_logits = prefill_logits_kt;
        let mut h_prev = h_prev_kt;

        // The last-row logits drive the first emitted token (same as the
        // skip-layer path).
        let prefill_last = prefill_logits.squeeze(1)?;
        let mut last_token = greedy_sample(&prefill_last)?;

        let mut base_pos = prompt_tokens.len();
        let mut mtp_pos = 0usize;
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut draft_accepted_count: usize = 0;
        let mut total_draft_attempts: usize = 0;

        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::make_rng::<rand::rngs::StdRng>(),
        };

        loop {
            if generated_tokens.len() >= params.max_tokens {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }

            if self.eos_token_ids.contains(&last_token) {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }

            generated_tokens.push(last_token);

            if !params.stop.is_empty() {
                let decoded_so_far = self
                    .tokenizer
                    .decode(&generated_tokens)
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .ok();
                if let Some(text) = &decoded_so_far {
                    for stop_seq in &params.stop {
                        if text.contains(stop_seq.as_str()) {
                            return Ok(MtpGenerationOutput {
                                text: String::new(),
                                token_ids: generated_tokens,
                                finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                draft_accepted_count,
                                total_draft_attempts,
                            });
                        }
                    }
                }
            }

            if generated_tokens.len() >= params.max_tokens {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::MaxTokens,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }

            total_draft_attempts += 1;
            let mut replay_prefix =
                Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
            replay_prefix.extend_from_slice(prompt_tokens);
            replay_prefix.extend_from_slice(&generated_tokens);
            crate::mtp_debug::set_h_main_replay_prefix_tokens(&replay_prefix);
            let result = speculative_mtp_decode_step(
                &*self.backend,
                last_token,
                &h_prev,
                &self.weights,
                &self.config,
                &base_cache,
                &base_block_table,
                base_pos,
                &mut linear_state,
                &mtp_cache,
                &mtp_block_table,
                mtp_pos,
                params,
                &self.eos_token_ids,
                &mut rng,
                self.active_lora.as_ref(),
            );
            crate::mtp_debug::clear_h_main_replay_prefix_tokens();
            let result = result.context("mtp speculative decode step failed")?;

            if result.draft_accepted {
                draft_accepted_count += 1;
            }
            base_pos += result.base_advance;
            mtp_pos += result.mtp_advance;
            h_prev = result.new_h_prev;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    return Ok(MtpGenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::Eos,
                        draft_accepted_count,
                        total_draft_attempts,
                    });
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                generated_tokens.push(token);

                if !params.stop.is_empty() {
                    let decoded_so_far = self
                        .tokenizer
                        .decode(&generated_tokens)
                        .map_err(|e| anyhow::anyhow!("{e}"))
                        .ok();
                    if let Some(text) = &decoded_so_far {
                        for stop_seq in &params.stop {
                            if text.contains(stop_seq.as_str()) {
                                return Ok(MtpGenerationOutput {
                                    text: String::new(),
                                    token_ids: generated_tokens,
                                    finish_reason: FinishReason::StopSequence(stop_seq.clone()),
                                    draft_accepted_count,
                                    total_draft_attempts,
                                });
                            }
                        }
                    }
                }

                if generated_tokens.len() >= params.max_tokens {
                    return Ok(MtpGenerationOutput {
                        text: String::new(),
                        token_ids: generated_tokens,
                        finish_reason: FinishReason::MaxTokens,
                        draft_accepted_count,
                        total_draft_attempts,
                    });
                }
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                return Ok(MtpGenerationOutput {
                    text: String::new(),
                    token_ids: generated_tokens,
                    finish_reason: FinishReason::Eos,
                    draft_accepted_count,
                    total_draft_attempts,
                });
            }
        }

        Ok(MtpGenerationOutput {
            text: String::new(),
            token_ids: generated_tokens,
            finish_reason: FinishReason::MaxTokens,
            draft_accepted_count,
            total_draft_attempts,
        })
    }

    /// Streaming self-speculative decoding (skip-layer draft).
    ///
    /// Mirrors [`generate_from_tokens_speculative`] but emits committed tokens
    /// incrementally through the returned channel so the SSE desktop path can
    /// benefit from the existing speculative setting.
    pub fn generate_streaming_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
        spec_config: &SpeculativeConfig,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        use rand::SeedableRng;

        if params.thinking_budget.is_some() {
            return self.generate_streaming(prompt, params);
        }
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        let (tx, rx) = mpsc::channel();
        // Verification writes the full speculative window (`last_token + k`)
        // before the loop commits accepted tokens, so flat KV needs temporary
        // headroom beyond the user-visible max token budget.
        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let mut kv_cache = self.new_kv_cache(max_total)?;
        let mut linear_state = self.new_linear_state()?;

        let logits = model_forward_kt(
            &*self.backend,
            &prompt_tokens,
            &self.weights,
            &self.config,
            Some(&mut kv_cache),
            Some(&mut linear_state),
            self.active_lora.as_ref(),
        )
        .context("prefill forward pass failed")?;
        kv_cache.advance(prompt_tokens.len());

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);
        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::make_rng::<rand::rngs::StdRng>(),
        };

        let mut last_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, params.seed, &[])?
        };

        loop {
            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            if self.eos_token_ids.contains(&last_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                last_token,
            ) {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    let completion_tokens = generated_tokens.len();
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: reason,
                        completion_tokens,
                        trailing_text: String::new(),
                    }));
                    return Ok(rx);
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let remaining = params.max_tokens - generated_tokens.len();
            let effective_k = spec_config.num_speculative_tokens.min(remaining);
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: effective_k,
                draft_layers: spec_config.draft_layers,
            };

            let result = speculative_decode_step(
                &*self.backend,
                last_token,
                &self.weights,
                &self.config,
                &mut kv_cache,
                &mut linear_state,
                &mut draft_linear_state,
                &effective_config,
                params,
                &self.eos_token_ids,
                &mut rng,
                self.active_lora.as_ref(),
            )
            .context("speculative decode step failed")?;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    finish_reason = FinishReason::Eos;
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                match emit_stream_token(
                    &tx,
                    &self.tokenizer,
                    &mut gate,
                    &mut generated_tokens,
                    token,
                ) {
                    StreamTokenDisposition::Continue => {}
                    StreamTokenDisposition::Finished(reason) => {
                        let completion_tokens = generated_tokens.len();
                        let _ = tx.send(StreamEvent::Done(StreamDone {
                            finish_reason: reason,
                            completion_tokens,
                            trailing_text: String::new(),
                        }));
                        return Ok(rx);
                    }
                    StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }
            }

            if !matches!(finish_reason, FinishReason::MaxTokens) {
                break;
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                finish_reason = FinishReason::Eos;
                break;
            }
        }

        let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens);
        let (finish_reason, gate_trailing) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, gate_trailing),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text: gate_trailing,
        }));

        Ok(rx)
    }

    /// Streaming native-MTP speculative decoding.
    ///
    /// Mirrors [`generate_from_tokens_mtp_speculative`] but emits committed
    /// tokens as they are accepted so the desktop streaming path can use MTP
    /// when the checkpoint and request settings allow it.
    pub fn generate_streaming_mtp_speculative(
        &self,
        prompt: &str,
        params: &SamplingParams,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        use rand::SeedableRng;

        if params.thinking_budget.is_some() {
            return self.generate_streaming(prompt, params);
        }
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");
        anyhow::ensure!(
            self.weights.mtp.is_some(),
            "generate_streaming_mtp_speculative requires the checkpoint to carry mtp.* tensors \
             (Qwen3.5-4B native MTP head)"
        );
        anyhow::ensure!(
            params.temperature == 0.0,
            "generate_streaming_mtp_speculative currently only supports greedy decoding \
             (temperature == 0)"
        );

        // #1082: 16 -> 64 so each FA2 kBlockN=64 tile is one physical page.
        const BLOCK_SIZE: usize = 64;

        let max_total = prompt_tokens.len() + params.max_tokens;
        // (#1082) kt-native paged cache — kt `DType` + runtime `Device`.
        let cache_device =
            paged_cache_device(self.backend.as_ref(), &self.weights.embed_tokens.device())?;
        let dtype = paged_cache_kt_dtype(self.config.dtype);

        let num_blocks = Self::blocks_needed(max_total, BLOCK_SIZE);
        let base_cache = PagedKvCache::new(
            self.config.num_full_attention_layers,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            cache_device,
        )?;
        let mtp_cache = PagedKvCache::new(
            1,
            num_blocks,
            BLOCK_SIZE,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype,
            cache_device,
        )?;
        let mut base_block_table = BlockTable::new();
        let mut mtp_block_table = BlockTable::new();
        for i in 0..num_blocks as u32 {
            base_block_table.push(i);
            mtp_block_table.push(i);
        }

        let (tx, rx) = mpsc::channel();
        let mut linear_state = self.new_linear_state()?;

        let (prefill_logits_kt, h_prev_kt) = if streaming_prefill_enabled_for(
            &self.weights.embed_tokens.device(),
            prompt_tokens.len(),
        ) {
            model_forward_paged_streaming_last_token_with_last_hidden(
                &*self.backend,
                &prompt_tokens,
                &self.weights,
                &self.config,
                &base_cache,
                &base_block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
            .context("mtp streaming prefill forward pass failed")?
        } else {
            model_forward_paged_last_token_with_last_hidden(
                &*self.backend,
                &prompt_tokens,
                &self.weights,
                &self.config,
                &base_cache,
                &base_block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                None,
            )
            .context("mtp prefill forward pass failed")?
        };
        // (#1082) MTP speculative step + speculative.rs are fully kt now —
        // `h_prev`/`prefill_logits` stay kt; no candle bridge.
        let prefill_logits = prefill_logits_kt;
        let mut h_prev = h_prev_kt;

        let prefill_last = prefill_logits.squeeze(1)?;
        let mut last_token = greedy_sample(&prefill_last)?;

        let mut base_pos = prompt_tokens.len();
        let mut mtp_pos = 0usize;
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);
        let mut rng = match params.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::make_rng::<rand::rngs::StdRng>(),
        };

        loop {
            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            if self.eos_token_ids.contains(&last_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                last_token,
            ) {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    let completion_tokens = generated_tokens.len();
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: reason,
                        completion_tokens,
                        trailing_text: String::new(),
                    }));
                    return Ok(rx);
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let mut replay_prefix =
                Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
            replay_prefix.extend_from_slice(&prompt_tokens);
            replay_prefix.extend_from_slice(&generated_tokens);
            crate::mtp_debug::set_h_main_replay_prefix_tokens(&replay_prefix);
            let result = speculative_mtp_decode_step(
                &*self.backend,
                last_token,
                &h_prev,
                &self.weights,
                &self.config,
                &base_cache,
                &base_block_table,
                base_pos,
                &mut linear_state,
                &mtp_cache,
                &mtp_block_table,
                mtp_pos,
                params,
                &self.eos_token_ids,
                &mut rng,
                self.active_lora.as_ref(),
            );
            crate::mtp_debug::clear_h_main_replay_prefix_tokens();
            let result = result.context("mtp speculative decode step failed")?;

            base_pos += result.base_advance;
            mtp_pos += result.mtp_advance;
            h_prev = result.new_h_prev;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    finish_reason = FinishReason::Eos;
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                match emit_stream_token(
                    &tx,
                    &self.tokenizer,
                    &mut gate,
                    &mut generated_tokens,
                    token,
                ) {
                    StreamTokenDisposition::Continue => {}
                    StreamTokenDisposition::Finished(reason) => {
                        let completion_tokens = generated_tokens.len();
                        let _ = tx.send(StreamEvent::Done(StreamDone {
                            finish_reason: reason,
                            completion_tokens,
                            trailing_text: String::new(),
                        }));
                        return Ok(rx);
                    }
                    StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }
            }

            if !matches!(finish_reason, FinishReason::MaxTokens) {
                break;
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            last_token = *result.accepted_tokens.last().unwrap();

            if result.hit_eos {
                finish_reason = FinishReason::Eos;
                break;
            }
        }

        let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens);
        let (finish_reason, gate_trailing) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, gate_trailing),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text: gate_trailing,
        }));

        Ok(rx)
    }

    /// Streaming generation using shared paged-cache state protected by
    /// short-lived mutexes.
    ///
    /// Mirrors [`generate_paged_shared`]: CUDA graph-enabled runtimes keep the
    /// existing whole-request lock scope, while non-CUDA desktop paths reserve
    /// blocks up front and lock the paged cache only around prefill / decode
    /// forward passes.
    pub fn generate_streaming_paged_shared(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        self.generate_from_tokens_streaming_paged_shared(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )
    }

    /// Streaming variant of [`generate_paged_shared_tokens`].
    pub fn generate_streaming_paged_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        self.generate_from_tokens_streaming_paged_shared(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )
    }

    /// Same as [`generate_streaming_paged_shared_tokens`], but optionally reuses
    /// a block-aligned cached prefix and returns completed prompt metadata that
    /// the caller may register after successful generation.
    pub fn generate_streaming_paged_shared_tokens_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
    ) -> Result<PrefixCachedStreamingOutput> {
        self.generate_from_tokens_streaming_paged_interleaved_with_prefix_cache(
            prompt_tokens,
            params,
            block_manager,
            paged_cache,
            cached_prefix,
        )
    }

    /// Threaded variant of [`generate_streaming_paged_shared_tokens`] that
    /// performs prefill on the calling thread and runs the decode loop on a
    /// spawned `std::thread`. The returned receiver yields tokens as they are
    /// produced, instead of after the entire `max_tokens` loop has completed
    /// (which is the behavior of the legacy `&self` variant — fine for unit
    /// tests but it makes `stream: true` look hung at the HTTP layer because
    /// the receiver only becomes observable when generation finishes).
    ///
    /// Holds an `Arc<RwLock<Self>>` so the spawned worker can re-acquire a
    /// read lock for decode steps without keeping the lock guard alive across
    /// thread boundaries (which `RwLockReadGuard` cannot do).
    pub fn spawn_streaming_paged_shared_tokens(
        runner_lock: Arc<std::sync::RwLock<Self>>,
        prompt_tokens: Vec<TokenId>,
        params: SamplingParams,
        block_manager: Arc<Mutex<BlockManager>>,
        paged_cache: Arc<PagedKvCache>,
        decode_batcher: Option<Arc<DecodeBatcher>>,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        // Allocate the full block reservation up front so the prompt + decode
        // window has its KV cache pages laid out before we hand the receiver
        // back to the caller. The legacy synchronous path uses
        // `SharedBlockReservation` for RAII free-on-drop; here we own the
        // block ids through to the end of the spawned thread instead.
        let max_total = prompt_tokens.len() + params.max_tokens;
        let block_table = {
            let mut bm_guard = lock_block_manager(block_manager.as_ref())?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            block_table
        };

        // Run prefill on the calling thread so a malformed prompt fails the
        // request synchronously rather than via an SSE error chunk. The decode
        // loop is what actually benefits from being threaded.
        let (logits, mut linear_state) = {
            let runner_guard = runner_lock
                .read()
                .map_err(|e| anyhow::anyhow!("failed to acquire runner read lock: {e}"))?;
            let mut linear_state = runner_guard.new_linear_state()?;
            let logits = {
                let pc_guard = lock_paged_cache(paged_cache.as_ref())?;
                if streaming_prefill_enabled_for(
                    &runner_guard.weights.embed_tokens.device(),
                    prompt_tokens.len(),
                ) {
                    model_forward_paged_streaming(
                        &*runner_guard.backend,
                        &prompt_tokens,
                        &runner_guard.weights,
                        &runner_guard.config,
                        pc_guard,
                        &block_table,
                        0,
                        Some(&mut linear_state),
                        runner_guard.active_lora.as_ref(),
                    )
                    .context("prefill forward pass (paged, streaming) failed")?
                } else {
                    model_forward_paged_last_token(
                        &*runner_guard.backend,
                        &prompt_tokens,
                        &runner_guard.weights,
                        &runner_guard.config,
                        pc_guard,
                        &block_table,
                        0,
                        Some(&mut linear_state),
                        runner_guard.active_lora.as_ref(),
                        None,
                    )
                    .context("prefill forward pass (paged) failed")?
                }
            };
            // (#1082) forward returns kt logits; sampler is kt — no bridge.
            (logits, linear_state)
        };

        let next_token = sample_first_decode_token(&logits, &params)?;
        drop(logits);

        let (tx, rx) = mpsc::channel();
        let seq_len = prompt_tokens.len();
        let runner_for_thread = runner_lock;
        let bm_for_thread = block_manager;
        let pc_for_thread = paged_cache;
        let decode_batcher_for_thread = decode_batcher;
        let block_ids_to_free: Vec<u32> = block_table.blocks.clone();

        std::thread::Builder::new()
            .name("kiln-stream-decode".to_string())
            .spawn(move || {
                let result = (|| -> Result<()> {
                    let runner_guard = runner_for_thread
                        .read()
                        .map_err(|e| anyhow::anyhow!("failed to acquire runner read lock in decode thread: {e}"))?;
                    runner_guard.run_stream_decode_loop_with_first(
                        &tx,
                        next_token,
                        seq_len,
                        &params,
                        pc_for_thread.as_ref(),
                        &block_table,
                        &mut linear_state,
                        decode_batcher_for_thread.as_deref(),
                    )
                })();
                if let Err(err) = result {
                    tracing::error!(error = %err, "spawn_streaming_paged_shared_tokens decode thread failed");
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: FinishReason::MaxTokens,
                        completion_tokens: 0,
                        trailing_text: String::new(),
                    }));
                }
                drop(tx);
                if !block_ids_to_free.is_empty() {
                    match bm_for_thread.lock() {
                        Ok(mut guard) => guard.free_all(&block_ids_to_free),
                        Err(e) => tracing::error!(
                            error = %e,
                            "failed to lock block manager to free blocks after streaming decode"
                        ),
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn streaming decode thread: {e}"))?;

        Ok(rx)
    }

    /// Threaded variant of
    /// [`generate_streaming_paged_shared_tokens_with_prefix_cache`]. Same
    /// motivation as [`spawn_streaming_paged_shared_tokens`]: hand the
    /// receiver back before decode starts so the SSE layer can stream tokens
    /// in real time.
    pub fn spawn_streaming_paged_shared_tokens_with_prefix_cache(
        runner_lock: Arc<std::sync::RwLock<Self>>,
        prompt_tokens: Vec<TokenId>,
        params: SamplingParams,
        block_manager: Arc<Mutex<BlockManager>>,
        paged_cache: Arc<PagedKvCache>,
        cached_prefix: Option<PagedPrefixReuse>,
        decode_batcher: Option<Arc<DecodeBatcher>>,
    ) -> Result<PrefixCachedStreamingOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = {
            let bm_guard = lock_block_manager(block_manager.as_ref())?;
            bm_guard.block_size()
        };

        let cached_prefix = cached_prefix.filter(|prefix| {
            if prefix.cached_tokens == 0 || prefix.cached_tokens > prompt_tokens.len() {
                return false;
            }

            let exact_candidate = prefix.cached_tokens == prompt_tokens.len();
            let expected_blocks = if exact_candidate {
                Self::blocks_needed(prefix.cached_tokens, block_size)
            } else {
                prefix.cached_tokens / block_size
            };
            let block_shape_valid = prefix.block_ids.len() == expected_blocks;
            let partial_hit = prefix.cached_tokens < prompt_tokens.len()
                && prefix.cached_tokens % block_size == 0;
            let exact_hit = prefix.cached_tokens == prompt_tokens.len()
                && prefix.next_token.as_ref().is_some_and(|next| match next {
                    PagedPrefixNextToken::Logits(_) => true,
                    PagedPrefixNextToken::GreedyToken(_) => params.is_effectively_greedy(),
                });
            block_shape_valid && (partial_hit || exact_hit)
        });

        let cached_blocks = cached_prefix
            .as_ref()
            .map(|prefix| prefix.block_ids.clone())
            .unwrap_or_default();
        let cached_tokens = cached_prefix
            .as_ref()
            .map(|prefix| prefix.cached_tokens)
            .unwrap_or(0);

        let max_total = prompt_tokens.len() + params.max_tokens;
        let total_blocks = Self::blocks_needed(max_total, block_size);
        let additional_blocks_needed = total_blocks.saturating_sub(cached_blocks.len());
        let allocated_blocks = {
            let mut bm_guard = lock_block_manager(block_manager.as_ref())?;
            bm_guard
                .allocate(additional_blocks_needed)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let block_table = append_prefix_block_table(&cached_blocks, &allocated_blocks);

        // Free helper for failure paths so a prefill error does not leak the
        // freshly-allocated suffix blocks (the cached-prefix blocks remain
        // owned by the prefix cache and must not be freed here).
        let free_allocated = |allocated: &[u32]| {
            if allocated.is_empty() {
                return;
            }
            match block_manager.lock() {
                Ok(mut guard) => guard.free_all(allocated),
                Err(e) => tracing::error!(
                    error = %e,
                    "failed to lock block manager to free blocks after prefix-cache prefill error"
                ),
            }
        };

        let (exact_next_token, mut linear_state) = match cached_prefix {
            Some(prefix) => {
                let exact_next_token = if prefix.cached_tokens == prompt_tokens.len() {
                    prefix.next_token
                } else {
                    None
                };
                (exact_next_token, prefix.linear_state)
            }
            None => {
                let runner_guard = runner_lock
                    .read()
                    .map_err(|e| anyhow::anyhow!("failed to acquire runner read lock: {e}"))?;
                (None, runner_guard.new_linear_state()?)
            }
        };

        let (next_token, registration, extra_registrations) = if let Some(next_token) =
            exact_next_token
        {
            let next_token = match next_token {
                PagedPrefixNextToken::Logits(logits) => {
                    match sample_first_decode_token(&logits, &params) {
                        Ok(token) => token,
                        Err(err) => {
                            free_allocated(&allocated_blocks);
                            return Err(err);
                        }
                    }
                }
                PagedPrefixNextToken::GreedyToken(token) => {
                    if params.temperature != 0.0 {
                        free_allocated(&allocated_blocks);
                        anyhow::bail!("greedy cached first token cannot serve non-greedy sampling");
                    }
                    token
                }
            };
            (next_token, None, Vec::new())
        } else {
            let prefill_result =
                (|| -> Result<(
                    // (#1082) kt-native logits — store + sampler are kt now.
                    kiln_tensor::Tensor,
                    Option<PagedPrefixRegistration>,
                    Vec<PagedPrefixRegistration>,
                )> {
                    let prefill_tokens = &prompt_tokens[cached_tokens..];
                    anyhow::ensure!(
                        !prefill_tokens.is_empty(),
                        "non-exact streaming prefix cache hit must leave at least one suffix token"
                    );

                    let runner_guard = runner_lock
                        .read()
                        .map_err(|e| anyhow::anyhow!("failed to acquire runner read lock: {e}"))?;
                    let split_pos =
                        strict_prompt_prefix_split_pos(prompt_tokens.len(), cached_tokens, block_size);
                    let mut prefill_split_snapshot: Option<RollingPrefixSnapshot> = None;
                    let logits = {
                        let pc_guard = lock_paged_cache(paged_cache.as_ref())?;
                        if streaming_prefill_enabled_for(
                            &runner_guard.weights.embed_tokens.device(),
                            prompt_tokens.len(),
                        ) {
                            if let Some(split_pos) = split_pos {
                                let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                                let _ = model_forward_paged_streaming(
                                    &*runner_guard.backend,
                                    head_tokens,
                                    &runner_guard.weights,
                                    &runner_guard.config,
                                    pc_guard,
                                    &block_table,
                                    cached_tokens,
                                    Some(&mut linear_state),
                                    runner_guard.active_lora.as_ref(),
                                )
                                .context(
                                    "prefill forward pass (streaming paged prefix cache head)",
                                )?;
                                prefill_split_snapshot = Some(RollingPrefixSnapshot {
                                    position: split_pos,
                                    linear_state: linear_state.snapshot().context(
                                        "snapshot linear state at streaming prefix-cache split",
                                    )?,
                                });

                                let tail_tokens = &prompt_tokens[split_pos..];
                                model_forward_paged_streaming(
                                    &*runner_guard.backend,
                                    tail_tokens,
                                    &runner_guard.weights,
                                    &runner_guard.config,
                                    pc_guard,
                                    &block_table,
                                    split_pos,
                                    Some(&mut linear_state),
                                    runner_guard.active_lora.as_ref(),
                                )
                                .context(
                                    "prefill forward pass (streaming paged prefix cache tail)",
                                )?
                            } else {
                                model_forward_paged_streaming(
                                    &*runner_guard.backend,
                                    prefill_tokens,
                                    &runner_guard.weights,
                                    &runner_guard.config,
                                    pc_guard,
                                    &block_table,
                                    cached_tokens,
                                    Some(&mut linear_state),
                                    runner_guard.active_lora.as_ref(),
                                )
                                .context(
                                    "prefill forward pass (streaming paged prefix cache) failed",
                                )?
                            }
                        } else {
                            model_forward_paged_last_token(
                                &*runner_guard.backend,
                                prefill_tokens,
                                &runner_guard.weights,
                                &runner_guard.config,
                                pc_guard,
                                &block_table,
                                cached_tokens,
                                Some(&mut linear_state),
                                runner_guard.active_lora.as_ref(),
                                None,
                            )
                            .context("prefill forward pass (paged prefix cache) failed")?
                        }
                    };
                    // (#1082) kt-native logits — next-token store is kt; no bridge.
                    let registration = runner_guard.completed_prompt_registration(
                        &prompt_tokens,
                        &block_table,
                        &linear_state,
                        block_size,
                        Some(PagedPrefixNextToken::Logits(logits.clone())),
                    )?;
                    let mut extra_registrations = Vec::new();
                    if let Some(reg) = build_extended_registration(
                        &prompt_tokens,
                        &[],
                        &block_table,
                        block_size,
                        prefill_split_snapshot,
                    ) {
                        extra_registrations.push(reg);
                    }
                    Ok((logits, registration, extra_registrations))
                })();

            let (logits, registration, extra_registrations) = match prefill_result {
                Ok(t) => t,
                Err(err) => {
                    free_allocated(&allocated_blocks);
                    return Err(err);
                }
            };
            let next_token = match sample_first_decode_token(&logits, &params) {
                Ok(t) => t,
                Err(err) => {
                    free_allocated(&allocated_blocks);
                    return Err(err);
                }
            };
            drop(logits);
            (next_token, registration, extra_registrations)
        };

        let (tx, rx) = mpsc::channel();
        // Rendezvous channel for the final "blocks to free" list. The API
        // layer sends `(allocated - retained) ∪ evicted` here as soon as
        // prefix-cache registration is done; the decode thread `recv()`s
        // this AFTER the decode loop completes, then frees. If the API
        // layer drops the sender without sending (panic / error path), the
        // thread falls back to freeing `allocated_blocks` so we don't leak.
        let (free_tx, free_rx) = mpsc::channel::<Vec<u32>>();
        let seq_len = prompt_tokens.len();
        let runner_for_thread = runner_lock;
        let bm_for_thread = block_manager;
        let pc_for_thread = paged_cache;
        let decode_batcher_for_thread = decode_batcher;
        let block_table_for_thread = block_table.clone();
        let allocated_for_fallback: Vec<u32> = allocated_blocks.clone();

        std::thread::Builder::new()
            .name("kiln-stream-decode-prefix".to_string())
            .spawn(move || {
                let result = (|| -> Result<()> {
                    let runner_guard = runner_for_thread
                        .read()
                        .map_err(|e| anyhow::anyhow!("failed to acquire runner read lock in decode thread: {e}"))?;
                    runner_guard.run_stream_decode_loop_with_first(
                        &tx,
                        next_token,
                        seq_len,
                        &params,
                        pc_for_thread.as_ref(),
                        &block_table_for_thread,
                        &mut linear_state,
                        decode_batcher_for_thread.as_deref(),
                    )
                })();
                if let Err(err) = result {
                    tracing::error!(error = %err, "spawn_streaming_paged_shared_tokens_with_prefix_cache decode thread failed");
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: FinishReason::MaxTokens,
                        completion_tokens: 0,
                        trailing_text: String::new(),
                    }));
                }
                drop(tx);

                // Decode is fully drained by here — the SSE side has either
                // received `Done` or the receiver was dropped. Now and only
                // now is it safe to release physical blocks back to the
                // BlockManager. Wait for the API layer to tell us the
                // exact set; fall back to the full allocation on error.
                let blocks_to_free = match free_rx.recv() {
                    Ok(list) => list,
                    Err(_) => {
                        tracing::warn!(
                            "decode thread did not receive a free list from the API layer; \
                             falling back to freeing all allocated blocks"
                        );
                        allocated_for_fallback
                    }
                };
                if !blocks_to_free.is_empty() {
                    match bm_for_thread.lock() {
                        Ok(mut guard) => guard.free_all(&blocks_to_free),
                        Err(e) => tracing::error!(
                            error = %e,
                            "failed to lock block manager to free blocks after streaming decode (prefix cache)"
                        ),
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn streaming decode thread: {e}"))?;

        Ok(PrefixCachedStreamingOutput {
            receiver: rx,
            registration,
            extra_registrations,
            allocated_blocks,
            block_free_signal: Some(free_tx),
        })
    }

    pub fn generate_streaming_paged_speculative_shared_tokens(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        spec_config: &SpeculativeConfig,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        if params.thinking_budget.is_some() {
            return self.generate_streaming_paged_shared_tokens(
                prompt_tokens,
                params,
                block_manager,
                paged_cache,
            );
        }
        anyhow::ensure!(
            params.temperature == 0.0,
            "paged skip-layer speculative streaming is greedy-only"
        );
        spec_config
            .validate(&self.config)
            .context("invalid speculative config")?;

        let max_spec_window = spec_config
            .num_speculative_tokens
            .min(params.max_tokens.max(1));
        let max_total = prompt_tokens.len() + params.max_tokens + max_spec_window + 1;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let result = self.generate_from_tokens_streaming_paged_speculative_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            spec_config,
        );

        drop(reservation);
        result
    }

    fn generate_from_tokens_streaming_paged_shared(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let cuda_graph_enabled = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?
            .is_enabled();
        if cuda_graph_enabled {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let pc_guard = lock_paged_cache(paged_cache)?;
            return self.generate_from_tokens_streaming_paged_locked(
                prompt_tokens,
                params,
                &mut bm_guard,
                pc_guard,
            );
        }

        let max_total = prompt_tokens.len() + params.max_tokens;
        let (reservation, block_table) = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            let block_size = bm_guard.block_size();
            let num_blocks = Self::blocks_needed(max_total, block_size);
            let block_ids = bm_guard
                .allocate(num_blocks)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut block_table = BlockTable::new();
            for &block_id in &block_ids {
                block_table.push(block_id);
            }
            (
                SharedBlockReservation {
                    block_manager,
                    block_ids,
                },
                block_table,
            )
        };

        let result = self.generate_from_tokens_streaming_paged_interleaved(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
        );

        drop(reservation);
        result
    }

    fn generate_from_tokens_streaming_paged_interleaved_with_prefix_cache(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &Mutex<BlockManager>,
        paged_cache: &PagedKvCache,
        cached_prefix: Option<PagedPrefixReuse>,
    ) -> Result<PrefixCachedStreamingOutput> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = {
            let bm_guard = lock_block_manager(block_manager)?;
            bm_guard.block_size()
        };

        let cached_prefix = cached_prefix.filter(|prefix| {
            prefix.cached_tokens > 0
                && prefix.cached_tokens < prompt_tokens.len()
                && prefix.cached_tokens % block_size == 0
                && prefix.block_ids.len() == prefix.cached_tokens / block_size
        });

        let cached_blocks = cached_prefix
            .as_ref()
            .map(|prefix| prefix.block_ids.as_slice())
            .unwrap_or(&[]);

        let max_total = prompt_tokens.len() + params.max_tokens;
        let total_blocks = Self::blocks_needed(max_total, block_size);
        let additional_blocks_needed = total_blocks.saturating_sub(cached_blocks.len());
        let allocated_blocks = {
            let mut bm_guard = lock_block_manager(block_manager)?;
            bm_guard
                .allocate(additional_blocks_needed)
                .map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let block_table = append_prefix_block_table(cached_blocks, &allocated_blocks);

        let result = self.generate_from_tokens_streaming_paged_interleaved_with_prefix_blocks(
            prompt_tokens,
            params,
            paged_cache,
            &block_table,
            cached_prefix,
            block_size,
        );

        match result {
            Ok(mut output) => {
                output.allocated_blocks = allocated_blocks;
                Ok(output)
            }
            Err(err) => {
                if !allocated_blocks.is_empty() {
                    let mut bm_guard = lock_block_manager(block_manager)?;
                    bm_guard.free_all(&allocated_blocks);
                }
                Err(err)
            }
        }
    }

    fn generate_from_tokens_streaming_paged_interleaved_with_prefix_blocks(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        cached_prefix: Option<PagedPrefixReuse>,
        block_size: usize,
    ) -> Result<PrefixCachedStreamingOutput> {
        let cached_tokens = cached_prefix
            .as_ref()
            .map(|prefix| prefix.cached_tokens)
            .unwrap_or(0);
        let mut linear_state = match cached_prefix {
            Some(prefix) => prefix.linear_state,
            None => self.new_linear_state()?,
        };

        let prefill_tokens = &prompt_tokens[cached_tokens..];
        anyhow::ensure!(
            !prefill_tokens.is_empty(),
            "streaming prefix cache hit must leave at least one suffix token"
        );

        let split_pos =
            strict_prompt_prefix_split_pos(prompt_tokens.len(), cached_tokens, block_size);
        let mut prefill_split_snapshot: Option<RollingPrefixSnapshot> = None;
        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(
                &self.weights.embed_tokens.device(),
                prompt_tokens.len(),
            ) {
                if let Some(split_pos) = split_pos {
                    let head_tokens = &prompt_tokens[cached_tokens..split_pos];
                    let _ = model_forward_paged_streaming(
                        &*self.backend,
                        head_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                    )
                    .context("prefill forward pass (streaming paged prefix cache head)")?;
                    prefill_split_snapshot = Some(RollingPrefixSnapshot {
                        position: split_pos,
                        linear_state: linear_state
                            .snapshot()
                            .context("snapshot linear state at streaming prefix-cache split")?,
                    });

                    let tail_tokens = &prompt_tokens[split_pos..];
                    model_forward_paged_streaming(
                        &*self.backend,
                        tail_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        split_pos,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                    )
                    .context("prefill forward pass (streaming paged prefix cache tail)")?
                } else {
                    model_forward_paged_streaming(
                        &*self.backend,
                        prefill_tokens,
                        &self.weights,
                        &self.config,
                        pc_guard,
                        block_table,
                        cached_tokens,
                        Some(&mut linear_state),
                        self.active_lora.as_ref(),
                    )
                    .context("prefill forward pass (streaming paged prefix cache) failed")?
                }
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prefill_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    cached_tokens,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged prefix cache) failed")?
            }
        };
        // (#1082) kt-native logits — next-token store + sampler are both kt;
        // no candle bridge.

        let registration = self.completed_prompt_registration(
            prompt_tokens,
            block_table,
            &linear_state,
            block_size,
            Some(PagedPrefixNextToken::Logits(logits.clone())),
        )?;
        let mut extra_registrations = Vec::new();
        if let Some(reg) = build_extended_registration(
            prompt_tokens,
            &[],
            block_table,
            block_size,
            prefill_split_snapshot,
        ) {
            extra_registrations.push(reg);
        }

        let receiver = self.stream_decode_from_prefill_logits(
            logits,
            prompt_tokens.len(),
            params,
            paged_cache,
            block_table,
            &mut linear_state,
        )?;

        // Legacy synchronous path: receiver is fully populated before return,
        // no decode thread is alive, so the API layer is free to call
        // bm.free_all on the same call frame. No rendezvous channel needed.
        Ok(PrefixCachedStreamingOutput {
            receiver,
            registration,
            extra_registrations,
            allocated_blocks: Vec::new(),
            block_free_signal: None,
        })
    }

    fn generate_from_tokens_streaming_paged_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(
                &self.weights.embed_tokens.device(),
                prompt_tokens.len(),
            ) {
                model_forward_paged_streaming(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("prefill forward pass (paged, streaming) failed")?
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (paged) failed")?
            }
        };
        // (#1082) forward returns kt logits; sampler entry is kt now — no bridge.

        self.stream_decode_from_prefill_logits(
            logits,
            prompt_tokens.len(),
            params,
            paged_cache,
            block_table,
            &mut linear_state,
        )
    }

    fn stream_decode_from_prefill_logits(
        &self,
        // (#1082) kt-native logits — sample_first_decode_token is kt.
        logits: kiln_tensor::Tensor,
        seq_len: usize,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        linear_state: &mut LinearAttentionState,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel();
        // Sample the first decode token from prefill logits and run the loop on
        // the calling thread. Used by tests and the synchronous (non-spawned)
        // entry points. The receiver is fully populated by the time we return.
        // Threaded callers should use [`run_stream_decode_loop_with_first`]
        // directly so they can sample the first token before spawning.
        let next_token = sample_first_decode_token(&logits, params)?;
        self.run_stream_decode_loop_with_first(
            &tx,
            next_token,
            seq_len,
            params,
            paged_cache,
            block_table,
            linear_state,
            None,
        )?;
        Ok(rx)
    }

    /// Streaming decode loop body, sending each generated token to `tx` as it
    /// is produced. The `next_token` argument is the first token to emit (the
    /// argmax/sample of the prefill logits). The caller owns `tx` so that
    /// threaded callers can spawn the loop and return the receiver to the
    /// async layer immediately, instead of waiting for `max_tokens` decode
    /// steps before the receiver becomes observable.
    pub(crate) fn run_stream_decode_loop_with_first(
        &self,
        tx: &mpsc::Sender<StreamEvent>,
        mut next_token: TokenId,
        mut seq_len: usize,
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        linear_state: &mut LinearAttentionState,
        decode_batcher: Option<&DecodeBatcher>,
    ) -> Result<()> {
        let rocm_owner = RocmDecodeOwnerLease::new(&self.rocm_graph);
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);

        for _step in 0..params.max_tokens {
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            if self.eos_token_ids.contains(&next_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                next_token,
            ) {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    finish_reason = reason;
                    break;
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(()),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let skip_gdn_state_readback = skip_final_gdn_state_readback_enabled()
                && generated_tokens.len() + 1 >= params.max_tokens;
            next_token = self.decode_next_token_paged_interleaved_or_batched(
                params,
                next_token,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                step_seed,
                decode_batcher,
                &generated_tokens,
                rocm_owner.row_id(),
                skip_gdn_state_readback,
            )?;
            seq_len += 1;
        }

        let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens);
        let (finish_reason, gate_trailing) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, gate_trailing),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text: gate_trailing,
        }));

        Ok(())
    }

    fn generate_from_tokens_streaming_paged_speculative_interleaved(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        paged_cache: &PagedKvCache,
        block_table: &BlockTable,
        spec_config: &SpeculativeConfig,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel();
        let mut linear_state = self.new_linear_state()?;

        let logits = {
            let pc_guard = lock_paged_cache(paged_cache)?;
            if streaming_prefill_enabled_for(
                &self.weights.embed_tokens.device(),
                prompt_tokens.len(),
            ) {
                model_forward_paged_streaming(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                )
                .context("prefill forward pass (streaming paged skip-layer, streaming) failed")?
            } else {
                model_forward_paged_last_token(
                    &*self.backend,
                    prompt_tokens,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    0,
                    Some(&mut linear_state),
                    self.active_lora.as_ref(),
                    None,
                )
                .context("prefill forward pass (streaming paged skip-layer) failed")?
            }
        };
        // (#1082) forward returns kt logits; sampler is kt — no bridge.

        let mut draft_linear_state =
            self.snapshot_draft_linear_state(&linear_state, spec_config)?;

        let mut base_pos = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);
        let mut last_token = greedy_sample(&logits)?;

        loop {
            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            if self.eos_token_ids.contains(&last_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                last_token,
            ) {
                StreamTokenDisposition::Continue => {}
                StreamTokenDisposition::Finished(reason) => {
                    finish_reason = reason;
                    break;
                }
                StreamTokenDisposition::ReceiverDropped => return Ok(rx),
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            let remaining = params.max_tokens - generated_tokens.len();
            let effective_config = SpeculativeConfig {
                num_speculative_tokens: spec_config.num_speculative_tokens.min(remaining),
                draft_layers: spec_config.draft_layers,
            };

            let result = {
                let pc_guard = lock_paged_cache(paged_cache)?;
                speculative_decode_step_paged_greedy(
                    &*self.backend,
                    last_token,
                    &self.weights,
                    &self.config,
                    pc_guard,
                    block_table,
                    base_pos,
                    &mut linear_state,
                    &mut draft_linear_state,
                    &effective_config,
                    params,
                    &self.eos_token_ids,
                    self.active_lora.as_ref(),
                )
                .context("streaming paged skip-layer speculative decode step failed")?
            };
            base_pos += result.base_advance;

            if result.accepted_tokens.is_empty() {
                if result.hit_eos {
                    finish_reason = FinishReason::Eos;
                }
                break;
            }

            for &token in &result.accepted_tokens[..result.accepted_tokens.len() - 1] {
                match emit_stream_token(
                    &tx,
                    &self.tokenizer,
                    &mut gate,
                    &mut generated_tokens,
                    token,
                ) {
                    StreamTokenDisposition::Continue => {}
                    StreamTokenDisposition::Finished(reason) => {
                        finish_reason = reason;
                        break;
                    }
                    StreamTokenDisposition::ReceiverDropped => return Ok(rx),
                }

                if generated_tokens.len() >= params.max_tokens {
                    break;
                }
            }

            if !matches!(finish_reason, FinishReason::MaxTokens) {
                break;
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            last_token = *result.accepted_tokens.last().unwrap();
            if result.hit_eos {
                finish_reason = FinishReason::Eos;
                break;
            }
        }

        let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens);
        let (finish_reason, gate_trailing) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, gate_trailing),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text: gate_trailing,
        }));

        Ok(rx)
    }

    /// Streaming generation using paged KV cache.
    ///
    /// Same as [`generate_streaming`] but uses paged KV cache for memory-efficient
    /// serving with the BlockManager.
    pub fn generate_streaming_paged(
        &self,
        prompt: &str,
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let prompt_tokens = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .context("failed to tokenize prompt")?;

        self.generate_from_tokens_streaming_paged_locked(
            &prompt_tokens,
            params,
            block_manager,
            paged_cache,
        )
    }

    fn generate_from_tokens_streaming_paged_locked(
        &self,
        prompt_tokens: &[TokenId],
        params: &SamplingParams,
        block_manager: &mut BlockManager,
        paged_cache: &PagedKvCache,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        anyhow::ensure!(!prompt_tokens.is_empty(), "prompt must not be empty");

        let block_size = block_manager.block_size();
        let max_total = prompt_tokens.len() + params.max_tokens;

        let num_blocks = Self::blocks_needed(max_total, block_size);
        let allocated_blocks = block_manager
            .allocate(num_blocks)
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        let mut block_table = BlockTable::new();
        for &block_id in &allocated_blocks {
            block_table.push(block_id);
        }

        let (tx, rx) = mpsc::channel();
        let mut linear_state = self.new_linear_state()?;

        // Prefill. Long Metal prompts use tiled streaming prefill by default;
        // env overrides can force either path.
        let prefill_result = if streaming_prefill_enabled_for(
            &self.weights.embed_tokens.device(),
            prompt_tokens.len(),
        ) {
            model_forward_paged_streaming(
                &*self.backend,
                &prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                &block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
            )
        } else {
            model_forward_paged_last_token(
                &*self.backend,
                &prompt_tokens,
                &self.weights,
                &self.config,
                paged_cache,
                &block_table,
                0,
                Some(&mut linear_state),
                self.active_lora.as_ref(),
                None,
            )
        };
        let logits = match prefill_result {
            Ok(l) => l,
            Err(e) => {
                block_manager.free_all(&allocated_blocks);
                return Err(e.context("prefill forward pass (paged) failed"));
            }
        };
        // (#1082) forward returns kt logits; sampler is kt — no bridge.

        let mut seq_len = prompt_tokens.len();
        let mut generated_tokens: Vec<TokenId> = Vec::new();
        let mut step_seed = params.seed;
        let mut finish_reason = FinishReason::MaxTokens;
        let mut gate = StreamTextGate::new(&params.stop);

        // Acquire CUDA graph runner for decode steps
        let mut graph_runner = self
            .cuda_graph
            .lock()
            .map_err(|e| anyhow::anyhow!("failed to lock CUDA graph runner: {e}"))?;

        let mut next_token = if params.is_effectively_greedy() {
            greedy_sample(&logits)?
        } else {
            sample_step(&logits, params, step_seed, &[])?
        };

        for _step in 0..params.max_tokens {
            if let Some(s) = step_seed.as_mut() {
                *s = s.wrapping_add(1);
            }

            next_token = params.apply_thinking_budget(&generated_tokens, next_token);
            if self.eos_token_ids.contains(&next_token) {
                finish_reason = FinishReason::Eos;
                break;
            }

            match emit_stream_token(
                &tx,
                &self.tokenizer,
                &mut gate,
                &mut generated_tokens,
                next_token,
            ) {
                StreamTokenDisposition::ReceiverDropped => {
                    block_manager.free_all(&allocated_blocks);
                    return Ok(rx);
                }
                StreamTokenDisposition::Finished(reason) => {
                    let _ = tx.send(StreamEvent::Done(StreamDone {
                        finish_reason: reason,
                        completion_tokens: generated_tokens.len(),
                        trailing_text: String::new(),
                    }));
                    block_manager.free_all(&allocated_blocks);
                    return Ok(rx);
                }
                StreamTokenDisposition::Continue => {}
            }

            if generated_tokens.len() >= params.max_tokens {
                break;
            }

            next_token = if params.is_effectively_greedy()
                && greedy_token_decode_enabled(self.backend.as_ref())
            {
                let linear_state_for_graph = if self.has_linear_attention_layers() {
                    Some(&mut linear_state)
                } else {
                    None
                };
                match self.decode_next_token_paged_greedy_metal_graph(
                    next_token,
                    paged_cache,
                    &block_table,
                    seq_len,
                    linear_state_for_graph,
                ) {
                    Ok(Some(token)) => {
                        seq_len += 1;
                        token
                    }
                    Ok(None) => {
                        let token = match model_forward_paged_next_token_greedy(
                            &*self.backend,
                            next_token,
                            &self.weights,
                            &self.config,
                            paged_cache,
                            &block_table,
                            seq_len,
                            Some(&mut linear_state),
                            self.active_lora.as_ref(),
                            None,
                        ) {
                            Ok(token) => token,
                            Err(e) => {
                                block_manager.free_all(&allocated_blocks);
                                return Err(e.context("decode forward pass (paged greedy) failed"));
                            }
                        };
                        seq_len += 1;
                        token
                    }
                    Err(e) => {
                        block_manager.free_all(&allocated_blocks);
                        return Err(
                            e.context("greedy Metal graph decode forward pass (paged) failed")
                        );
                    }
                }
            } else {
                // Decode step: use CUDA graph runner
                let logits = match graph_runner.decode_step_paged(
                    &*self.backend,
                    next_token,
                    &self.weights,
                    &self.config,
                    paged_cache,
                    &block_table,
                    seq_len,
                    &mut linear_state,
                    self.active_lora.as_ref(),
                    None,
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        block_manager.free_all(&allocated_blocks);
                        return Err(e.context("decode forward pass (paged) failed"));
                    }
                };
                seq_len += 1;
                // #1082: `decode_step_paged` now returns kt — feed `sample_step`
                // directly, no candle->kt bridge.
                sample_step(&logits, params, step_seed, &generated_tokens)?
            };
        }

        let (gate_trailing, late_stop) = gate.finish(&self.tokenizer, &generated_tokens);
        let (finish_reason, gate_trailing) = match late_stop {
            Some(stop) => (FinishReason::StopSequence(stop), String::new()),
            None => (finish_reason, gate_trailing),
        };
        let _ = tx.send(StreamEvent::Done(StreamDone {
            finish_reason,
            completion_tokens: generated_tokens.len(),
            trailing_text: gate_trailing,
        }));

        block_manager.free_all(&allocated_blocks);

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::FallbackPolicy;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    const DECODE_FALLBACK_ENV: &[&str] = &[
        "KILN_DECODE_HOT_PATH_DEBUG_FALLBACK",
        "KILN_METAL_DECODE_BATCH_GENERIC_FALLBACK",
        "KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK",
        "KILN_ROCM_DECODE_BATCH_GENERIC_FALLBACK",
    ];

    #[test]
    fn decode_row_ids_are_process_unique_across_threads() {
        const THREADS: usize = 8;
        const IDS_PER_THREAD: usize = 128;
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..IDS_PER_THREAD)
                        .map(|_| next_decode_row_id())
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let ids: Vec<u64> = workers
            .into_iter()
            .flat_map(|worker| {
                worker
                    .join()
                    .expect("decode owner allocator worker panicked")
            })
            .collect();
        let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();

        assert_eq!(ids.len(), THREADS * IDS_PER_THREAD);
        assert_eq!(unique.len(), ids.len());
        assert!(ids.into_iter().all(|id| id != 0));
    }

    #[test]
    fn decode_row_id_exhaustion_never_wraps_or_reuses_zero() {
        let counter = std::sync::atomic::AtomicU64::new(u64::MAX);
        assert_eq!(allocate_decode_row_id(&counter), u64::MAX);
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);

        let exhausted = std::panic::catch_unwind(|| allocate_decode_row_id(&counter));
        assert!(
            exhausted.is_err(),
            "exhausted owner namespace must fail closed"
        );
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
    const DECODE_BATCHER_ROWWISE_ENV: &[&str] = &["KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY"];

    struct EnvRestore(Vec<(&'static str, Option<String>)>);

    impl EnvRestore {
        fn clear(keys: &[&'static str]) -> Self {
            let prior = keys
                .iter()
                .map(|&key| (key, std::env::var(key).ok()))
                .collect::<Vec<_>>();
            unsafe {
                for &key in keys {
                    std::env::remove_var(key);
                }
            }
            Self(prior)
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            unsafe {
                for (key, value) in &self.0 {
                    if let Some(value) = value {
                        std::env::set_var(key, value);
                    } else {
                        std::env::remove_var(key);
                    }
                }
            }
        }
    }

    #[test]
    fn single_row_hip_graph_preempts_generic_greedy_batch_route() {
        assert_eq!(
            greedy_batch_route(true, false, 1, true),
            GreedyBatchRoute::HipGraph,
            "BF16 single-row ROCm decode must reach the enabled HIP graph runner"
        );
        assert_eq!(
            greedy_batch_route(true, true, 1, true),
            GreedyBatchRoute::HipGraph,
            "FP8 single-row ROCm decode is also graph-capturable"
        );
        assert_eq!(
            greedy_batch_route(true, false, 1, false),
            GreedyBatchRoute::Contiguous,
            "graphs-off and non-ROCm single-row decode must retain the eager path"
        );
        assert_eq!(
            greedy_batch_route(true, false, 4, true),
            GreedyBatchRoute::Contiguous,
            "multi-row greedy decode must retain true batching"
        );
        assert_eq!(
            greedy_batch_route(false, false, 1, true),
            GreedyBatchRoute::Later,
            "sampled single-row decode is handled by the later HIP graph branch"
        );
    }

    #[derive(Debug)]
    struct NamedTestBackend {
        name: &'static str,
        device: kiln_tensor::Device,
    }

    impl BackendIdentity for NamedTestBackend {
        fn runtime_name(&self) -> &'static str {
            self.name
        }

        fn runtime_device(&self) -> kiln_tensor::Device {
            self.device
        }

        fn runtime_as_any(&self) -> &dyn std::any::Any {
            &()
        }
    }

    impl StartupBackend for NamedTestBackend {}

    impl crate::backend::AttentionBackend for NamedTestBackend {}

    impl crate::backend::GdnBackend for NamedTestBackend {}

    impl crate::backend::ConvBackend for NamedTestBackend {}

    impl crate::backend::LinearBackend for NamedTestBackend {}

    impl crate::backend::residency::ResidentRegistry for NamedTestBackend {}

    impl crate::backend::ResidencyBackend for NamedTestBackend {}

    impl crate::backend::SamplingBackend for NamedTestBackend {}

    impl crate::backend::OptimizerBackend for NamedTestBackend {}

    impl crate::backend::PagedKvBackend for NamedTestBackend {}

    impl crate::backend::ReplayBackend for NamedTestBackend {}

    impl crate::backend::TrainingLossBackend for NamedTestBackend {}

    impl BackendRuntime for NamedTestBackend {}

    #[test]
    fn decode_batcher_stats_report_runner_calls_per_token() {
        let stats = DecodeBatcherStats {
            executed_rows: 4,
            runner_calls: 5,
            max_runner_calls_per_token: 2,
            ..DecodeBatcherStats::default()
        };

        assert_eq!(stats.runner_calls_per_token(), Some(1.25));
        assert_eq!(stats.max_runner_calls_per_token, 2);
        assert_eq!(stats.runner_call_budget_per_token(), 2);
        assert!(!stats.runner_call_budget_exceeded());
        assert_eq!(DecodeBatcherStats::default().runner_calls_per_token(), None);

        let exceeded = DecodeBatcherStats {
            max_runner_calls_per_token: 3,
            ..DecodeBatcherStats::default()
        };
        assert!(exceeded.runner_call_budget_exceeded());
    }

    #[test]
    fn test_decode_batcher_default_backend_policy() {
        for (
            backend_name,
            device,
            max_batch,
            wait_micros,
            allow_mixed_seq_lens,
            rowwise_retry_env,
            use_native_sampled_contiguous_decode,
            sampled_contiguous_decode_requires_resident_decode,
            partition_noncontiguous_gdn_kv_tiles,
        ) in [
            (
                "cpu",
                kiln_tensor::Device::Cpu,
                8,
                0,
                false,
                None,
                false,
                false,
                false,
            ),
            (
                "cuda",
                kiln_tensor::Device::Cpu,
                1,
                0,
                false,
                None,
                false,
                false,
                true,
            ),
            (
                "cuda",
                kiln_tensor::Device::Cuda(0),
                1,
                0,
                false,
                None,
                false,
                false,
                true,
            ),
            (
                "metal",
                kiln_tensor::Device::Metal(0),
                8,
                100,
                true,
                None,
                true,
                false,
                false,
            ),
            (
                "vulkan",
                kiln_tensor::Device::Cpu,
                64,
                5_000,
                true,
                Some("KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY"),
                true,
                true,
                false,
            ),
            (
                "vulkan",
                kiln_tensor::Device::Vulkan(0),
                64,
                5_000,
                true,
                Some("KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY"),
                true,
                true,
                false,
            ),
            (
                "rocm",
                kiln_tensor::Device::Rocm(0),
                8,
                0,
                false,
                None,
                false,
                false,
                false,
            ),
        ] {
            let policy = DecodeBatcherPolicy::for_backend(backend_name, device);
            assert_eq!(
                policy.max_batch, max_batch,
                "{backend_name} max batch policy drifted"
            );
            assert_eq!(
                policy.wait_micros, wait_micros,
                "{backend_name} wait policy drifted"
            );
            assert_eq!(
                policy.allow_mixed_seq_lens, allow_mixed_seq_lens,
                "{backend_name} mixed-seq policy drifted"
            );
            assert_eq!(
                policy.rowwise_retry_env, rowwise_retry_env,
                "{backend_name} rowwise retry policy drifted"
            );
            assert_eq!(
                policy.use_native_sampled_contiguous_decode, use_native_sampled_contiguous_decode,
                "{backend_name} sampled contiguous decode policy drifted"
            );
            assert_eq!(
                policy.sampled_contiguous_decode_requires_resident_decode,
                sampled_contiguous_decode_requires_resident_decode,
                "{backend_name} sampled contiguous resident requirement policy drifted"
            );
            assert_eq!(
                policy.partition_noncontiguous_gdn_kv_tiles, partition_noncontiguous_gdn_kv_tiles,
                "{backend_name} GDN KV contiguity partition policy drifted"
            );
        }
    }

    #[test]
    fn test_decode_batcher_max_batch_env_policy() {
        let prior_specific = std::env::var("KILN_DECODE_BATCH_MAX").ok();
        let prior_shared = std::env::var("KILN_MAX_DECODE_BATCH").ok();
        // SAFETY: tests in this module that touch these env vars restore them
        // before returning.
        unsafe {
            std::env::remove_var("KILN_DECODE_BATCH_MAX");
            std::env::remove_var("KILN_MAX_DECODE_BATCH");
        }

        let device = kiln_tensor::Device::Cpu;
        assert_eq!(
            DecodeBatcherConfig::from_env_for_backend_kt(&device, "vulkan").max_batch,
            64
        );
        unsafe {
            std::env::set_var("KILN_MAX_DECODE_BATCH", "24");
        }
        assert_eq!(
            DecodeBatcherConfig::from_env_for_backend_kt(&device, "vulkan").max_batch,
            24
        );
        unsafe {
            std::env::set_var("KILN_DECODE_BATCH_MAX", "12");
        }
        assert_eq!(
            DecodeBatcherConfig::from_env_for_backend_kt(&device, "vulkan").max_batch,
            12
        );

        match prior_specific {
            Some(v) => unsafe { std::env::set_var("KILN_DECODE_BATCH_MAX", v) },
            None => unsafe { std::env::remove_var("KILN_DECODE_BATCH_MAX") },
        }
        match prior_shared {
            Some(v) => unsafe { std::env::set_var("KILN_MAX_DECODE_BATCH", v) },
            None => unsafe { std::env::remove_var("KILN_MAX_DECODE_BATCH") },
        }
    }

    #[test]
    fn test_decode_batcher_rowwise_retry_uses_backend_policy() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _fallback_env = EnvRestore::clear(DECODE_FALLBACK_ENV);
        let _rowwise_env = EnvRestore::clear(DECODE_BATCHER_ROWWISE_ENV);

        let vulkan_cpu_sentinel = NamedTestBackend {
            name: "vulkan",
            device: kiln_tensor::Device::Cpu,
        };
        let metal = NamedTestBackend {
            name: "metal",
            device: kiln_tensor::Device::Metal(0),
        };

        assert!(!decode_batcher_rowwise_retry_enabled(&vulkan_cpu_sentinel));
        assert!(!decode_batcher_rowwise_retry_enabled(&metal));

        unsafe {
            std::env::set_var("KILN_VULKAN_DECODE_BATCH_ROWWISE_RETRY", "1");
        }
        assert!(decode_batcher_rowwise_retry_enabled(&vulkan_cpu_sentinel));
        assert!(
            !decode_batcher_rowwise_retry_enabled(&metal),
            "Vulkan rowwise retry env should not apply to Metal policy"
        );
    }

    #[test]
    fn test_decode_hot_path_fallback_policy_defaults() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvRestore::clear(DECODE_FALLBACK_ENV);
        for (backend_name, device, expected, debug_env) in [
            (
                "cpu",
                kiln_tensor::Device::Cpu,
                FallbackPolicy::CorrectnessAllowed,
                None,
            ),
            (
                "cuda",
                kiln_tensor::Device::Cuda(0),
                FallbackPolicy::CorrectnessAllowed,
                None,
            ),
            (
                "metal",
                kiln_tensor::Device::Metal(0),
                FallbackPolicy::NativeRequired,
                Some("KILN_METAL_DECODE_BATCH_GENERIC_FALLBACK"),
            ),
            (
                "vulkan",
                kiln_tensor::Device::Vulkan(0),
                FallbackPolicy::NativeRequired,
                Some("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK"),
            ),
            (
                "rocm",
                kiln_tensor::Device::Rocm(0),
                FallbackPolicy::NativeRequired,
                Some("KILN_ROCM_DECODE_BATCH_GENERIC_FALLBACK"),
            ),
        ] {
            let fallback =
                backend::capability::BackendFallbackCapabilities::for_backend(backend_name, device);
            assert_eq!(fallback.decode_hot_path, expected);
            assert_eq!(
                fallback.decode_hot_path_debug_env, debug_env,
                "{backend_name} decode debug fallback env drifted"
            );
            let backend = NamedTestBackend {
                name: backend_name,
                device,
            };
            assert_eq!(
                decode_hot_path_fallback_policy_for_backend(&backend),
                expected
            );
        }

        let vulkan_cpu_sentinel = NamedTestBackend {
            name: "vulkan",
            device: kiln_tensor::Device::Cpu,
        };
        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&vulkan_cpu_sentinel),
            FallbackPolicy::NativeRequired
        );
        assert_eq!(
            backend::capability::BackendFallbackCapabilities::for_backend(
                "vulkan",
                kiln_tensor::Device::Cpu,
            )
            .decode_hot_path_debug_env,
            Some("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK")
        );
    }

    #[test]
    fn test_decode_hot_path_debug_fallback_opt_in_warns_and_counts() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvRestore::clear(DECODE_FALLBACK_ENV);
        unsafe {
            std::env::set_var("KILN_DECODE_HOT_PATH_DEBUG_FALLBACK", "1");
        }
        for (backend_name, device) in [
            ("metal", kiln_tensor::Device::Metal(0)),
            ("vulkan", kiln_tensor::Device::Vulkan(0)),
            ("rocm", kiln_tensor::Device::Rocm(0)),
        ] {
            let backend = NamedTestBackend {
                name: backend_name,
                device,
            };
            let policy = decode_hot_path_fallback_policy_for_backend(&backend);
            assert_eq!(policy, FallbackPolicy::WarnAndCount);
            assert!(policy.allows_fallback());
        }
    }

    #[test]
    fn test_decode_hot_path_backend_debug_fallback_uses_policy_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _env = EnvRestore::clear(DECODE_FALLBACK_ENV);

        let metal = NamedTestBackend {
            name: "metal",
            device: kiln_tensor::Device::Metal(0),
        };
        let vulkan = NamedTestBackend {
            name: "vulkan",
            device: kiln_tensor::Device::Cpu,
        };
        let rocm = NamedTestBackend {
            name: "rocm",
            device: kiln_tensor::Device::Rocm(0),
        };

        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&metal),
            FallbackPolicy::NativeRequired
        );
        unsafe {
            std::env::set_var("KILN_VULKAN_DECODE_BATCH_GENERIC_FALLBACK", "1");
        }
        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&vulkan),
            FallbackPolicy::WarnAndCount
        );
        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&metal),
            FallbackPolicy::NativeRequired,
            "Vulkan decode fallback env should not apply to Metal policy"
        );
        assert_eq!(
            decode_hot_path_fallback_policy_for_backend(&rocm),
            FallbackPolicy::NativeRequired,
            "Vulkan decode fallback env should not apply to ROCm policy"
        );
    }

    fn block_table_with(blocks: &[u32]) -> BlockTable {
        let mut bt = BlockTable::new();
        bt.blocks = blocks.to_vec();
        bt
    }

    fn empty_linear_state() -> LinearAttentionState {
        LinearAttentionState {
            recurrent_states: Vec::new(),
            conv_states: Vec::new(),
        }
    }

    #[test]
    fn noncontiguous_kv_tiles_detection() {
        // #1082: FA2_KBLOCK_N=64 (hdim256). At block_size=16 → pages_per_chunk
        // = 64/16 = 4. CONTIGUOUS within each 4-page chunk → safe (false).
        let bt_contig = block_table_with(&[100, 101, 102, 103, 104, 105, 106, 107, 200, 201]);
        assert!(
            !batch_has_noncontiguous_kv_tiles(&[&bt_contig], &[160], 16),
            "physically-contiguous pages within a tile must NOT force the row-loop"
        );
        // A gap (999) starting the 2nd 4-page chunk: base=999 then 105 != 1000
        // → non-contiguous (true).
        let bt_frag = block_table_with(&[100, 101, 102, 103, 999, 105, 106, 107]);
        assert!(
            batch_has_noncontiguous_kv_tiles(&[&bt_frag], &[128], 16),
            "a fragmented page inside a tile must force the contiguity-safe row-loop"
        );
        // Chunk BOUNDARY discontinuity (idx 8 starts a new 4-page chunk) is
        // allowed — the kernel re-reads block_table at chunk starts.
        assert!(
            !batch_has_noncontiguous_kv_tiles(&[&bt_contig], &[144], 16),
            "discontinuity at a chunk boundary (every 4 pages) is allowed"
        );
        // bs=1 short row (1 page) is trivially contiguous.
        let bt_one = block_table_with(&[42]);
        assert!(!batch_has_noncontiguous_kv_tiles(&[&bt_one], &[5], 16));
        // Mixed batch: one bad row anywhere → true.
        assert!(batch_has_noncontiguous_kv_tiles(
            &[&bt_contig, &bt_frag],
            &[160, 128],
            16
        ));
        // Only check pages covering live tokens: a fragmented page BEYOND
        // seqused_k is not read by the kernel → not flagged.
        let bt_tail_frag = block_table_with(&[100, 101, 999]);
        assert!(
            !batch_has_noncontiguous_kv_tiles(&[&bt_tail_frag], &[20], 16),
            "fragmentation beyond the live window (seqlen=20 → 2 pages) is not read"
        );
        // #1082 KEY: at the new default block_size=64, pages_per_chunk =
        // FA2_KBLOCK_N/64 = 1, so each FA2 tile is exactly one page and the
        // kernel looks it up independently. Arbitrarily strided (non-adjacent)
        // blocks that WOULD trip at block_size=16 are safe at 64 → the row-loop
        // never fires for FA2 reasons (this is what restores bs=64 concurrent).
        let bt_strided = block_table_with(&[5, 7, 9, 11]);
        assert!(
            !batch_has_noncontiguous_kv_tiles(&[&bt_strided], &[256], 64),
            "block_size>=kBlockN makes every FA2 tile one page → no fragmentation trips"
        );
    }

    #[test]
    fn per_row_contiguity_mask_partitions_mixed_batch() {
        // #1082 partition fix: the batched-decode partition routes ONLY the
        // genuinely-fragmented rows to the per-row loop and batches the
        // contiguous majority through the fast path (vs #1445's all-or-nothing
        // whole-batch serialization). Validate the per-row mask it is built on.
        let bt_contig = block_table_with(&[100, 101, 102, 103, 104, 105, 106, 107]);
        let bt_frag = block_table_with(&[100, 101, 102, 103, 999, 105, 106, 107]);
        let bt_short = block_table_with(&[42]);
        // Per-row helper agrees with the batch wrapper, row by row.
        assert!(!row_has_noncontiguous_kv_tiles(
            bt_contig.blocks.as_slice(),
            128,
            16
        ));
        assert!(row_has_noncontiguous_kv_tiles(
            bt_frag.blocks.as_slice(),
            128,
            16
        ));
        assert!(!row_has_noncontiguous_kv_tiles(
            bt_short.blocks.as_slice(),
            5,
            16
        ));
        // A mixed batch yields a mask that picks out exactly the fragmented row;
        // the partition batches rows 0,2 (fast path) and row-loops only row 1.
        let bts = [&bt_contig, &bt_frag, &bt_short];
        let seqlens = [128usize, 128, 5];
        let mask: Vec<bool> = (0..3)
            .map(|r| row_has_noncontiguous_kv_tiles(bts[r].blocks.as_slice(), seqlens[r], 16))
            .collect();
        assert_eq!(mask, vec![false, true, false]);
        // The batch wrapper is exactly the OR of the per-row mask — so #1445's
        // detector fired on the WHOLE batch for a single bad row; the partition
        // resolves that per-row instead of serializing all of it.
        assert_eq!(
            batch_has_noncontiguous_kv_tiles(&bts, &seqlens, 16),
            mask.iter().any(|&x| x)
        );
    }

    #[test]
    fn extended_registration_requires_snapshot() {
        let bt = block_table_with(&[10, 11, 12, 13]);
        assert!(
            build_extended_registration(&[1, 2, 3, 4, 5], &[6, 7, 8], &bt, 4, None).is_none(),
            "no snapshot → no extended registration"
        );
    }

    #[test]
    fn strict_prompt_prefix_split_is_inside_prompt() {
        assert_eq!(strict_prompt_prefix_split_pos(9, 0, 4), Some(8));
        assert_eq!(
            strict_prompt_prefix_split_pos(8, 0, 4),
            Some(4),
            "block-aligned prompts still need an inside split before the final prompt block"
        );
        assert_eq!(strict_prompt_prefix_split_pos(8, 4, 4), None);
        assert_eq!(strict_prompt_prefix_split_pos(3, 0, 4), None);
    }

    #[test]
    fn extended_registration_skipped_when_not_block_aligned() {
        let bt = block_table_with(&[10, 11, 12, 13]);
        let snap = Some(RollingPrefixSnapshot {
            position: 7,
            linear_state: empty_linear_state(),
        });
        assert!(
            build_extended_registration(&[1, 2, 3, 4], &[5, 6, 7], &bt, 4, snap).is_none(),
            "position 7 is not block-aligned at block_size 4"
        );
    }

    #[test]
    fn extended_registration_inside_prompt_covers_prefix() {
        // Prefill-time snapshot at position 4, prompt is 8 tokens, no
        // generation yet. The registration should be a strict-prefix entry
        // covering the first 4 prompt tokens — this is what makes
        // multi-turn lookups hit when the chat template appends a divergent
        // generation-prompt tail to the prompt (the tail is past position
        // 4, so the entry remains valid for subsequent turns).
        let bt = block_table_with(&[10, 11, 12, 13]);
        let snap = Some(RollingPrefixSnapshot {
            position: 4,
            linear_state: empty_linear_state(),
        });
        let reg = build_extended_registration(&[1, 2, 3, 4, 5, 6, 7, 8], &[], &bt, 4, snap)
            .expect("expected strict-prefix registration");
        assert_eq!(reg.prompt_tokens, vec![1, 2, 3, 4]);
        assert_eq!(reg.block_ids, vec![10]);
        assert!(reg.next_token.is_none());
    }

    #[test]
    fn extended_registration_covers_prompt_plus_decoded() {
        let bt = block_table_with(&[10, 11, 12, 13, 14]);
        // 5-token prompt + 7 generated. Snapshot lands at position 12
        // (block-aligned at block_size 4) — covers prompt + 7 generated.
        let snap = Some(RollingPrefixSnapshot {
            position: 12,
            linear_state: empty_linear_state(),
        });
        let reg = build_extended_registration(
            &[1, 2, 3, 4, 5],
            &[10, 11, 12, 13, 14, 15, 16],
            &bt,
            4,
            snap,
        )
        .expect("expected extended registration");
        assert_eq!(
            reg.prompt_tokens,
            vec![1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16]
        );
        assert_eq!(reg.block_ids, vec![10, 11, 12]);
        assert!(reg.next_token.is_none());
    }

    #[test]
    fn extended_registration_truncates_to_last_boundary() {
        let bt = block_table_with(&[10, 11, 12, 13, 14]);
        // 5-token prompt + 11 generated → position 16 if all written, but
        // snapshot was only taken at position 12 (last boundary crossed).
        let snap = Some(RollingPrefixSnapshot {
            position: 12,
            linear_state: empty_linear_state(),
        });
        let reg = build_extended_registration(
            &[1, 2, 3, 4, 5],
            &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            &bt,
            4,
            snap,
        )
        .expect("expected extended registration");
        // Only the 7 generated tokens up to the snapshotted boundary are
        // included; the rest of the generation tail is discarded for the
        // cache entry (no linear-state snapshot for it).
        assert_eq!(reg.prompt_tokens.len(), 12);
        assert_eq!(reg.block_ids.len(), 3);
    }

    #[test]
    fn extended_registration_bails_when_block_table_short() {
        // Snapshot says position 12 (3 blocks) but block table only has 2
        // blocks — bookkeeping bug upstream; refuse to register a corrupt
        // entry instead of indexing out of bounds.
        let bt = block_table_with(&[10, 11]);
        let snap = Some(RollingPrefixSnapshot {
            position: 12,
            linear_state: empty_linear_state(),
        });
        assert!(
            build_extended_registration(&[1, 2, 3, 4, 5], &[6, 7, 8, 9, 10, 11, 12], &bt, 4, snap)
                .is_none(),
            "must not produce a registration referencing missing blocks"
        );
    }
}
